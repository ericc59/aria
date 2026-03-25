"""Run ARIA on ARC tasks with a proposer model for bootstrap/training.

Usage:
    python scripts/bootstrap_solve.py --task 0d3d703e
    python scripts/bootstrap_solve.py --dataset v1-train --provider claude
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from aria.graph.extract import extract_with_delta
from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.proposer.harness import propose_and_verify, ProposalAttempt
from aria.proposer.models import AnthropicProposer, ClaudeCodeProposer, MockProposer, OpenAIProposer
from aria.reporting import build_solve_report, extract_library_ops_used, extract_op_names
from aria.retrieval import retrieve_program
from aria.runtime.ops import has_op
from aria.runtime.program import program_to_text
from aria.solver import load_task
from aria.types import Task, grid_eq
from aria.runtime.executor import execute
from aria.verify.mode import detect_mode
from aria.types import TaskContext, VerifyMode


DATA_ROOTS = {
    "v1-train": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v1/training"
    ),
    "v1-eval": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v1/evaluation"
    ),
    "v2-train": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v2/training"
    ),
    "v2-eval": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v2/evaluation"
    ),
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOG_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_PROGRAM_STORE = RESULTS_DIR / "program_store.json"
DEFAULT_LIBRARY_STORE = RESULTS_DIR / "library.json"

log = logging.getLogger("aria.bootstrap_solve")


def make_proposer(args):
    if args.provider == "claude":
        return ClaudeCodeProposer(model=args.model)
    elif args.provider == "anthropic":
        return AnthropicProposer(
            model=args.model or "claude-sonnet-4-6",
            temperature=args.temperature,
        )
    elif args.provider == "openai":
        return OpenAIProposer(
            model=args.model or "gpt-4o-mini",
            base_url=args.base_url,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")


def _load_program_store(store_path: Path, extra_sources: list[str]) -> tuple[ProgramStore, list[str]]:
    store = ProgramStore.load_json(store_path)
    imported_notes: list[str] = []

    sources = [Path(source) for source in extra_sources]
    if not store and not sources:
        sources = [
            RESULTS_DIR / "v1-train.json",
            Path(__file__).parent / "corpus_report.json",
        ]

    for source in sources:
        if source == store_path or not source.exists():
            continue
        imported = store.import_path(source)
        if imported:
            imported_notes.append(f"{imported} from {source}")

    if imported_notes:
        store.save_json(store_path)

    return store, imported_notes

def _find_missing_ops(attempts: list[list[ProposalAttempt]]) -> Counter:
    """Find ops the model tried to use that don't exist in our registry."""
    missing: Counter = Counter()
    skip = {"bind", "yield", "assert", "true", "false"}
    for round_atts in attempts:
        for att in round_atts:
            if att.program_text and not att.program_text.startswith("--"):
                # Collect bound variable names so we don't flag them as missing ops
                bound = set(re.findall(r"bind\s+(\w+)\s*=", att.program_text))
                bound.add("input")
                bound.add("ctx")
                for op in extract_op_names(att.program_text):
                    if op and not has_op(op) and op not in skip and op not in bound:
                        missing[op] += 1
    return missing


def solve_single(
    task: Task,
    task_id: str,
    proposer,
    library: Library,
    program_store: ProgramStore,
    args,
) -> dict:
    """Solve a single task and return result dict."""
    task_signatures = tuple(sorted(compute_task_signatures(task.train)))
    in_shape = task.train[0].input.shape
    out_shape = task.train[0].output.shape
    log.info("=" * 60)
    log.info(f"TASK {task_id}: {len(task.train)} demos, {in_shape} -> {out_shape}")

    state_graphs = []
    deltas = []
    for demo in task.train:
        sg_in, sg_out, delta = extract_with_delta(demo.input, demo.output)
        state_graphs.append(sg_in)
        deltas.append(delta)

    sg0 = state_graphs[0]
    log.info(f"  State graph: {sg0.context.obj_count} objects, bg={sg0.context.bg_color}, "
             f"palette={sorted(sg0.context.palette)}")
    if deltas[0].dims_changed:
        log.info(f"  Dims change: {deltas[0].dims_changed}")

    t0 = time.time()
    retrieval_hit = None
    if len(program_store) > 0:
        retrieval_hit = retrieve_program(
            task.train,
            program_store,
            max_candidates=args.retrieval_limit,
        )

    if retrieval_hit:
        elapsed = time.time() - t0
        prog = retrieval_hit.program
        prog_text = program_to_text(prog)
        outcome = {
            "task_id": task_id,
            "solved": True,
            "rounds": 0,
            "total_candidates": retrieval_hit.candidates_tried,
            "time_sec": round(elapsed, 2),
            "program": prog_text,
            "retrieved": True,
            "retrieval_candidates_tried": retrieval_hit.candidates_tried,
            "retrieval_sources": list(retrieval_hit.record.sources),
            "solve_source": "retrieval",
            "task_signatures": list(task_signatures),
        }
        log.info(
            f"  RETRIEVED in {elapsed:.1f}s after replaying "
            f"{retrieval_hit.candidates_tried} stored programs"
        )
        log.info(f"  Program:\n    " + prog_text.replace("\n", "\n    "))
    else:
        result = propose_and_verify(
            model=proposer,
            demos=task.train,
            state_graphs=state_graphs,
            deltas=deltas,
            library=library,
            max_rounds=args.max_rounds,
            k=args.k,
            task_id=task_id,
        )
        elapsed = time.time() - t0

        # Log every attempt
        for rnd_idx, round_atts in enumerate(result.all_attempts):
            log.info(f"  Round {rnd_idx + 1}: {len(round_atts)} candidates")
            for att in round_atts:
                if att.parse_error:
                    log.info(f"    PARSE ERROR: {att.parse_error[:120]}")
                    log.debug(f"    Raw text: {att.program_text[:300]}")
                elif att.execution_error:
                    log.info(f"    EXEC ERROR: {att.execution_error[:150]}")
                    log.info(f"    Program: {att.program_text[:200]}")
                elif att.verify_result and att.verify_result.passed:
                    log.info(f"    VERIFIED: {att.program_text[:200]}")
                elif att.verify_result:
                    vr = att.verify_result
                    log.info(f"    WRONG OUTPUT demo={vr.failed_demo}: {att.program_text[:150]}")
                    if vr.diff:
                        log.info(f"      Diff: {vr.diff}")
                else:
                    log.debug(f"    Skipped: {att.program_text[:100]}")

        outcome = {
            "task_id": task_id,
            "solved": result.solved,
            "rounds": result.rounds,
            "total_candidates": result.total_candidates,
            "time_sec": round(elapsed, 2),
            "retrieved": False,
            "solve_source": "unsolved",
            "task_signatures": list(task_signatures),
        }

        # Check for missing ops
        missing = _find_missing_ops(result.all_attempts)
        if missing:
            outcome["missing_ops"] = dict(missing)
            log.warning(f"  Missing ops: {dict(missing)}")

        if not (result.solved and result.winning_program):
            log.info(f"  FAILED after {elapsed:.1f}s, {result.rounds} rounds, "
                     f"{result.total_candidates} candidates")
            errors = []
            for round_atts in result.all_attempts:
                for att in round_atts:
                    if att.parse_error and not att.parse_error.startswith("--"):
                        errors.append({"type": "parse", "msg": att.parse_error[:200]})
                    elif att.execution_error:
                        errors.append({
                            "type": "execution",
                            "msg": att.execution_error[:200],
                            "program": att.program_text[:200] if att.program_text else None,
                        })
                    elif att.verify_result and not att.verify_result.passed:
                        errors.append({
                            "type": att.verify_result.error_type,
                            "demo": att.verify_result.failed_demo,
                            "program": att.program_text[:200] if att.program_text else None,
                        })
            outcome["errors"] = errors[:6]
            return outcome

        prog = result.winning_program
        prog_text = program_to_text(prog)
        outcome["program"] = prog_text
        outcome["solve_source"] = "proposal"
        log.info(f"  SOLVED in {elapsed:.1f}s, {result.rounds} rounds")
        log.info(f"  Program:\n    " + prog_text.replace("\n", "\n    "))

    program_store.add_program(
        prog,
        task_id=task_id,
        source="solve",
        signatures=frozenset(task_signatures),
    )
    outcome["library_ops_used"] = extract_library_ops_used(outcome["program"], library.names())

    if outcome["solved"]:
        mode = detect_mode(prog)
        test_results = []
        for i, test_pair in enumerate(task.test):
            ctx = None if mode == VerifyMode.STATELESS else TaskContext(demos=task.train)
            try:
                output = execute(prog, test_pair.input, ctx)
                correct = grid_eq(output, test_pair.output)
                test_results.append({"test_idx": i, "correct": correct})
                log.info(f"  Test {i}: {'CORRECT' if correct else 'WRONG'}")
            except Exception as e:
                test_results.append({"test_idx": i, "correct": False, "error": str(e)})
                log.info(f"  Test {i}: ERROR {e}")
        outcome["test_results"] = test_results

    return outcome


def _load_progress(results_file: Path) -> dict[str, dict]:
    """Load existing results to skip already-solved tasks."""
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        return {t["task_id"]: t for t in data.get("tasks", [])}
    return {}


def _save_progress(results_file: Path, tasks: list[dict], config: dict):
    """Save results incrementally."""
    report = build_solve_report(tasks, config)

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap ARC solves with an LLM proposer")
    parser.add_argument("--task", help="Single task ID (e.g., 0d3d703e)")
    parser.add_argument("--dataset", default="v1-train", choices=DATA_ROOTS.keys())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="Run continuously, save as you go")
    parser.add_argument("--skip-solved", action="store_true", help="Skip tasks already solved in output file")
    parser.add_argument("--provider", default="claude", choices=["claude", "anthropic", "openai"])
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--base-url", help="API base URL (for OpenAI-compatible)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-k", type=int, default=4, help="Candidates per round")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max retry rounds")
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=0,
        help="Max stored programs to replay before proposer (0=all)",
    )
    parser.add_argument(
        "--program-store",
        default=str(DEFAULT_PROGRAM_STORE),
        help="Canonical verified-program store path",
    )
    parser.add_argument(
        "--program-source",
        action="append",
        default=[],
        help="Additional solve/corpus/program-store JSON to import",
    )
    parser.add_argument(
        "--library-store",
        default=str(DEFAULT_LIBRARY_STORE),
        help="Persisted library JSON path",
    )
    parser.add_argument(
        "--snapshot-dir",
        help="Load frozen program_store.json and library.json from a snapshot directory",
    )
    parser.add_argument(
        "--freeze-stores",
        action="store_true",
        help="Do not mutate shared program/library stores across tasks",
    )
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    # Set up logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"bootstrap_solve_{ts}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)

    # Also log INFO to stderr so it's visible alongside stdout
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(stderr_handler)

    log.info(f"ARIA solve started at {datetime.now().isoformat()}")
    log.info(f"Args: {vars(args)}")
    log.info(f"Log file: {log_file}")

    proposer = make_proposer(args)
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        program_store_path = snapshot_dir / "program_store.json"
        library_path = snapshot_dir / "library.json"
        freeze_stores = True
        if not program_store_path.exists() or not library_path.exists():
            raise SystemExit(
                f"Incomplete snapshot: expected {program_store_path} and {library_path}"
            )
    else:
        snapshot_dir = None
        program_store_path = Path(args.program_store)
        library_path = Path(args.library_store)
        freeze_stores = args.freeze_stores

    program_store, imported_programs = _load_program_store(
        program_store_path,
        args.program_source,
    )
    library = Library.load_json(library_path)

    results_file = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"bootstrap_{args.dataset}.json"
    )
    print(f"Log: {log_file}")
    if snapshot_dir:
        print(f"Snapshot: {snapshot_dir}")
    print(f"Program store: {program_store_path} ({len(program_store)} programs)")
    print(f"Library store: {library_path} ({len(library.all_entries())} entries)")
    if freeze_stores:
        print("Store mode: frozen")
    for note in imported_programs:
        print(f"Bootstrapped program store with {note}")

    if args.task:
        # ---- Single task mode ----
        data_dir = DATA_ROOTS[args.dataset]
        path = os.path.join(data_dir, f"{args.task}.json")
        if not os.path.exists(path):
            print(f"Task not found: {path}")
            sys.exit(1)

        with open(path) as f:
            task = load_task(json.load(f))

        print(f"Solving {args.task} ({len(task.train)} demos, "
              f"{task.train[0].input.shape} -> {task.train[0].output.shape})")
        print(f"Provider: {args.provider}, k={args.k}, max_rounds={args.max_rounds}")
        print("-" * 60)

        task_program_store = program_store.clone() if freeze_stores else program_store
        task_library = library.clone() if freeze_stores else library
        result = solve_single(task, args.task, proposer, task_library, task_program_store, args)
        if not freeze_stores:
            program_store.save_json(program_store_path)
            if library.all_entries():
                library.save_json(library_path)

        if result["solved"]:
            print(f"\nSOLVED in {result['time_sec']}s ({result['rounds']} rounds, "
                  f"{result['total_candidates']} candidates)")
            print(f"Source: {result.get('solve_source', 'proposal')}")
            print(f"\nProgram:\n{result['program']}")
            if result.get("library_ops_used"):
                print(f"Library ops used: {', '.join(result['library_ops_used'])}")
            if result.get("test_results"):
                for tr in result["test_results"]:
                    status = "CORRECT" if tr["correct"] else "WRONG"
                    print(f"  Test {tr['test_idx']}: {status}")
        else:
            print(f"\nFAILED after {result['time_sec']}s ({result['rounds']} rounds, "
                  f"{result['total_candidates']} candidates)")
            for err in result.get("errors", [])[:3]:
                print(f"  {err.get('type')}: {err.get('msg', err.get('program', ''))[:100]}")

        if result.get("missing_ops"):
            print(f"\n  Missing ops the model tried to use: {result['missing_ops']}")

    else:
        # ---- Batch / loop mode ----
        data_dir = DATA_ROOTS[args.dataset]
        fnames = sorted(os.listdir(data_dir))
        if args.limit > 0:
            fnames = fnames[:args.limit]

        # Load existing progress
        existing = _load_progress(results_file) if args.skip_solved else {}
        config = {
            "provider": args.provider,
            "model": args.model,
            "k": args.k,
            "max_rounds": args.max_rounds,
            "temperature": args.temperature,
            "retrieval_limit": args.retrieval_limit,
            "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
            "freeze_stores": freeze_stores,
        }

        all_tasks = list(existing.values())
        solved = sum(1 for t in all_tasks if t["solved"])
        skipped = 0

        print(f"Solving {len(fnames)} tasks from {args.dataset}")
        print(f"Provider: {args.provider}, k={args.k}, max_rounds={args.max_rounds}")
        if existing:
            print(f"Resuming: {len(existing)} already done, {solved} solved")
        print("=" * 60)

        try:
            for i, fname in enumerate(fnames):
                task_id = fname.replace(".json", "")

                # Skip if already done
                if task_id in existing:
                    if existing[task_id].get("solved"):
                        skipped += 1
                        continue
                    elif args.skip_solved:
                        skipped += 1
                        continue

                with open(os.path.join(data_dir, fname)) as f:
                    task = load_task(json.load(f))

                task_program_store = program_store.clone() if freeze_stores else program_store
                task_library = library.clone() if freeze_stores else library
                result = solve_single(task, task_id, proposer, task_library, task_program_store, args)

                # Update tracking
                existing[task_id] = result
                all_tasks = list(existing.values())
                solved = sum(1 for t in all_tasks if t["solved"])

                status = "SOLVED" if result["solved"] else "      "
                source_str = ""
                if result["solved"]:
                    source_str = f" [{result.get('solve_source', 'proposal')}]"
                missing_str = ""
                if result.get("missing_ops"):
                    missing_str = f" [missing: {', '.join(result['missing_ops'].keys())}]"

                print(f"  [{i+1}/{len(fnames)}] {task_id}: {status} "
                      f"({result['time_sec']}s, {result['total_candidates']} cand)"
                      f"{source_str}"
                      f"{missing_str}"
                      f"  [{solved}/{len(all_tasks)} total]")

                # Save after every task
                _save_progress(results_file, all_tasks, config)
                if not freeze_stores:
                    program_store.save_json(program_store_path)
                    if library.all_entries():
                        library.save_json(library_path)

                if not args.loop and args.limit and i + 1 >= args.limit:
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted — saving progress...")
            _save_progress(results_file, all_tasks, config)
            if not freeze_stores:
                program_store.save_json(program_store_path)
                if library.all_entries():
                    library.save_json(library_path)

        # Final summary
        all_tasks = list(existing.values())
        report = build_solve_report(all_tasks, config)
        solved = report["solved"]

        print("=" * 60)
        print(f"Solved: {solved}/{len(all_tasks)} ({report['solve_rate']})")
        if skipped:
            print(f"Skipped: {skipped} (already done)")
        print(
            "Sources: "
            f"retrieval={report['source_counts'].get('retrieval', 0)}, "
            f"proposal={report['source_counts'].get('proposal', 0)}, "
            f"unsolved={report['source_counts'].get('unsolved', 0)}"
        )
        print(
            "Retrieval: "
            f"{report['retrieval_hit_count']}/{len(all_tasks)} "
            f"({report['retrieval_hit_rate']}); "
            f"share of solves {report['retrieval_share_of_solves']}"
        )
        if report.get("retrieval_sources_global"):
            print(f"Top retrieval sources: {report['retrieval_sources_global']}")
        if report.get("library_ops_global"):
            print(f"Library ops in winning programs: {report['library_ops_global']}")
        if report.get("missing_ops_global"):
            print("\nMissing ops (model tried but we don't have):")
            for op, cnt in report["missing_ops_global"].items():
                print(f"  {op}: {cnt}x")

        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
