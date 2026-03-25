"""Run ARIA on ARC tasks using only the offline system.

Usage:
    python scripts/solve.py --task 0d3d703e
    python scripts/solve.py --dataset v1-eval --snapshot-dir results/snapshots/v1_bootstrap
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.reporting import build_solve_report, extract_library_ops_used
from aria.runtime.program import program_to_text
from aria.solver import load_task, solve_task
from aria.types import Task, grid_eq


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

log = logging.getLogger("aria.solve")


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


def _load_progress(results_file: Path) -> dict[str, dict]:
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        return {t["task_id"]: t for t in data.get("tasks", [])}
    return {}


def _save_progress(results_file: Path, tasks: list[dict], config: dict) -> None:
    report = build_solve_report(tasks, config)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def solve_single(
    task: Task,
    task_id: str,
    library: Library,
    program_store: ProgramStore,
    args,
) -> dict:
    t0 = time.time()
    result = solve_task(
        task,
        library=library,
        program_store=program_store,
        task_id=task_id,
        retrieval_limit=args.retrieval_limit,
        max_search_steps=args.max_search_steps,
        max_search_candidates=args.max_search_candidates,
        include_core_ops=not args.library_only,
    )
    elapsed = time.time() - t0

    outcome = {
        "task_id": task_id,
        "solved": result.solved,
        "time_sec": round(elapsed, 2),
        "retrieved": result.retrieved,
        "searched": result.searched,
        "retrieval_candidates_tried": result.retrieval_candidates_tried,
        "search_candidates_tried": result.search_candidates_tried,
        "total_candidates": (
            result.retrieval_candidates_tried
            if result.retrieved
            else result.search_candidates_tried
        ),
        "solve_source": (
            "retrieval" if result.retrieved else "search" if result.solved else "unsolved"
        ),
        "abstractions_mined": result.abstractions_mined,
        "task_signatures": list(result.task_signatures),
    }

    if not result.solved or result.winning_program is None:
        log.info(
            "FAILED %s retrieval=%s search=%s",
            task_id,
            result.retrieval_candidates_tried,
            result.search_candidates_tried,
        )
        return outcome

    program_text = program_to_text(result.winning_program)
    outcome["program"] = program_text
    outcome["library_ops_used"] = extract_library_ops_used(program_text, library.names())

    test_results = []
    for idx, (test_pair, output) in enumerate(zip(task.test, result.test_outputs)):
        correct = grid_eq(output, test_pair.output)
        test_results.append({"test_idx": idx, "correct": correct})
    outcome["test_results"] = test_results

    log.info(
        "SOLVED %s via %s in %.2fs",
        task_id,
        outcome["solve_source"],
        elapsed,
    )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve ARC tasks with the offline ARIA runtime")
    parser.add_argument("--task", help="Single task ID (e.g., 0d3d703e)")
    parser.add_argument("--dataset", default="v1-train", choices=DATA_ROOTS.keys())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="Ignored; kept for compatibility")
    parser.add_argument("--skip-solved", action="store_true", help="Skip tasks already solved in output file")
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=0,
        help="Max stored programs to replay before search (0=all)",
    )
    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
        help="Max bind steps for offline search",
    )
    parser.add_argument(
        "--max-search-candidates",
        type=int,
        default=5000,
        help="Max GRID-valued candidates to verify during offline search",
    )
    parser.add_argument(
        "--library-only",
        action="store_true",
        help="Search only library ops, not the full core DSL",
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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"solve_{ts}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(file_handler)
    log.setLevel(logging.INFO)

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

    results_file = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.json"
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
        data_dir = DATA_ROOTS[args.dataset]
        path = os.path.join(data_dir, f"{args.task}.json")
        if not os.path.exists(path):
            print(f"Task not found: {path}")
            sys.exit(1)

        with open(path) as f:
            task = load_task(json.load(f))

        print(
            f"Solving {args.task} ({len(task.train)} demos, "
            f"{task.train[0].input.shape} -> {task.train[0].output.shape})"
        )
        print(
            "Engine: offline "
            f"(max_search_steps={args.max_search_steps}, "
            f"max_search_candidates={args.max_search_candidates}, "
            f"core_ops={'off' if args.library_only else 'on'})"
        )
        print("-" * 60)

        task_program_store = program_store.clone() if freeze_stores else program_store
        task_library = library.clone() if freeze_stores else library
        result = solve_single(task, args.task, task_library, task_program_store, args)
        if not freeze_stores:
            program_store.save_json(program_store_path)
            if library.all_entries():
                library.save_json(library_path)

        if result["solved"]:
            print(
                f"\nSOLVED in {result['time_sec']}s "
                f"({result['solve_source']}, {result['total_candidates']} cand)"
            )
            print(f"\nProgram:\n{result['program']}")
            if result.get("library_ops_used"):
                print(f"Library ops used: {', '.join(result['library_ops_used'])}")
            if result.get("test_results"):
                for tr in result["test_results"]:
                    status = "CORRECT" if tr["correct"] else "WRONG"
                    print(f"  Test {tr['test_idx']}: {status}")
        else:
            print(
                f"\nFAILED after {result['time_sec']}s "
                f"(retrieval={result['retrieval_candidates_tried']}, "
                f"search={result['search_candidates_tried']})"
            )
        return

    data_dir = DATA_ROOTS[args.dataset]
    fnames = sorted(os.listdir(data_dir))
    if args.limit > 0:
        fnames = fnames[:args.limit]

    existing = _load_progress(results_file) if args.skip_solved else {}
    config = {
        "engine": "offline",
        "retrieval_limit": args.retrieval_limit,
        "max_search_steps": args.max_search_steps,
        "max_search_candidates": args.max_search_candidates,
        "library_only": args.library_only,
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "freeze_stores": freeze_stores,
    }

    all_tasks = list(existing.values())
    solved = sum(1 for t in all_tasks if t["solved"])
    skipped = 0

    print(f"Solving {len(fnames)} tasks from {args.dataset}")
    print(
        "Engine: offline "
        f"(max_search_steps={args.max_search_steps}, "
        f"max_search_candidates={args.max_search_candidates}, "
        f"core_ops={'off' if args.library_only else 'on'})"
    )
    if existing:
        print(f"Resuming: {len(existing)} already done, {solved} solved")
    print("=" * 60)

    try:
        for idx, fname in enumerate(fnames, start=1):
            task_id = fname.replace(".json", "")
            if task_id in existing and args.skip_solved:
                skipped += 1
                continue

            with open(os.path.join(data_dir, fname)) as f:
                task = load_task(json.load(f))

            task_program_store = program_store.clone() if freeze_stores else program_store
            task_library = library.clone() if freeze_stores else library
            result = solve_single(task, task_id, task_library, task_program_store, args)

            existing[task_id] = result
            all_tasks = list(existing.values())
            solved = sum(1 for item in all_tasks if item["solved"])

            status = "SOLVED" if result["solved"] else "      "
            source_str = f" [{result['solve_source']}]" if result["solved"] else ""
            print(
                f"  [{idx}/{len(fnames)}] {task_id}: {status} "
                f"({result['time_sec']}s, {result['total_candidates']} cand)"
                f"{source_str}  [{solved}/{len(all_tasks)} total]"
            )

            _save_progress(results_file, all_tasks, config)
            if not freeze_stores:
                program_store.save_json(program_store_path)
                if library.all_entries():
                    library.save_json(library_path)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving progress...")
        _save_progress(results_file, all_tasks, config)
        if not freeze_stores:
            program_store.save_json(program_store_path)
            if library.all_entries():
                library.save_json(library_path)

    report = build_solve_report(all_tasks, config)
    print("=" * 60)
    print(f"Solved: {report['solved']}/{len(all_tasks)} ({report['solve_rate']})")
    if skipped:
        print(f"Skipped: {skipped} (already done)")
    print(
        "Sources: "
        f"retrieval={report['source_counts'].get('retrieval', 0)}, "
        f"search={report['source_counts'].get('search', 0)}, "
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
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
