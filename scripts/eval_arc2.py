#!/usr/bin/env python3
"""Evaluate the canonical `aria/search` solver on ARC datasets.

Usage:
    # Full ARC-2 eval set (120 tasks):
    python scripts/eval_arc2.py

    # Quick smoke test (5 tasks):
    python scripts/eval_arc2.py --limit 5

    # Single task:
    python scripts/eval_arc2.py --task 0934a4d8

    # Use ARC-2 training split instead:
    python scripts/eval_arc2.py --dataset v2-train --limit 20

    # With beam refinement:
    python scripts/eval_arc2.py --beam-width 8

    # With a frozen snapshot from an ARC-1 corpus run:
    python scripts/eval_arc2.py --snapshot-dir results/snapshots/v1_bootstrap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from aria.datasets import DatasetInfo, dataset_names, get_dataset, load_arc_task
from aria.eval import EvalConfig, evaluate_task, run_evaluation
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.trace_store import RefinementTraceStore

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOG_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_PROGRAM_STORE = RESULTS_DIR / "program_store.json"
DEFAULT_LIBRARY_STORE = RESULTS_DIR / "library.json"

log = logging.getLogger("aria.eval_arc2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARIA on ARC-AGI-2 (or any ARC dataset)"
    )
    parser.add_argument(
        "--dataset", default="v2-eval", choices=dataset_names(),
        help="Dataset to evaluate on (default: v2-eval)",
    )
    parser.add_argument("--task", help="Single task ID")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks to evaluate (0=all)")
    parser.add_argument(
        "--time-budget", type=float, default=30.0,
        help="Per-task time budget in seconds for the search solver",
    )

    parser.add_argument("--retrieval-limit", type=int, default=0)
    parser.add_argument("--max-search-steps", type=int, default=3)
    parser.add_argument("--max-search-candidates", type=int, default=5000)
    parser.add_argument("--max-refinement-rounds", type=int, default=2)
    parser.add_argument("--library-only", action="store_true")
    parser.add_argument("--beam-width", type=int, default=0)
    parser.add_argument("--beam-rounds", type=int, default=3)
    parser.add_argument("--beam-mutations-per-candidate", type=int, default=30)

    parser.add_argument("--program-store", default=str(DEFAULT_PROGRAM_STORE))
    parser.add_argument("--program-source", action="append", default=[])
    parser.add_argument("--library-store", default=str(DEFAULT_LIBRARY_STORE))
    parser.add_argument("--snapshot-dir", help="Frozen program_store + library directory")
    parser.add_argument("--trace-store", help="Refinement trace store path (optional)")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--submission", metavar="PATH",
        help="Also write an ARC-AGI-2 submission.json to PATH",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"eval_{args.dataset}_{ts}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(file_handler)
    log.setLevel(logging.INFO)

    ds = get_dataset(args.dataset)

    if args.snapshot_dir:
        snap = Path(args.snapshot_dir)
        ps_path = snap / "program_store.json"
        lib_path = snap / "library.json"
        if not ps_path.exists() or not lib_path.exists():
            print(f"Incomplete snapshot: expected {ps_path} and {lib_path}", file=sys.stderr)
            sys.exit(1)
    else:
        ps_path = Path(args.program_store)
        lib_path = Path(args.library_store)

    program_store = ProgramStore.load_json(ps_path)
    for src in args.program_source:
        src_path = Path(src)
        if src_path.exists() and src_path != ps_path:
            program_store.import_path(src_path)

    library = Library.load_json(lib_path)

    trace_store = None
    if args.trace_store:
        trace_store = RefinementTraceStore.load_json(Path(args.trace_store))

    config = EvalConfig(
        retrieval_limit=args.retrieval_limit,
        max_search_steps=args.max_search_steps,
        max_search_candidates=args.max_search_candidates,
        max_refinement_rounds=args.max_refinement_rounds,
        include_core_ops=not args.library_only,
        beam_width=args.beam_width,
        beam_rounds=args.beam_rounds,
        beam_mutations_per_candidate=args.beam_mutations_per_candidate,
        time_budget_sec=args.time_budget,
    )

    results_file = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"eval_{args.dataset}_{ts}.json"
    )

    print(f"Dataset: {ds.short_version} {ds.split} ({ds.name})")
    print(f"Data dir: {ds.root}")
    print(f"Program store: {ps_path} ({len(program_store)} programs)")
    print(f"Library: {lib_path} ({len(library.all_entries())} entries)")
    print(f"Log: {log_file}")
    beam_note = (
        f", beam_width={config.beam_width}"
        if config.beam_width > 0 else ""
    )
    print(
        f"Config: steps={config.max_search_steps}, "
        f"cand={config.max_search_candidates}, "
        f"rounds={config.max_refinement_rounds}, "
        f"time_budget={config.time_budget_sec:.1f}s"
        f"{beam_note}"
    )

    # --- single-task mode ---
    if args.task:
        task = load_arc_task(ds, args.task)
        print(
            f"Evaluating {args.task} "
            f"({len(task.train)} demos, "
            f"{task.train[0].input.shape} -> {task.train[0].output.shape})"
        )
        print("-" * 60)

        outcome = evaluate_task(
            args.task, task,
            library=library,
            program_store=program_store,
            config=config,
            trace_store=trace_store,
            freeze_stores=True,
        )

        if trace_store is not None:
            trace_store.save_json(Path(args.trace_store))

        if outcome["solved"]:
            cand_note = (
                f", {outcome['total_candidates']} cand"
                if outcome.get("total_candidates", 0) > 0 else ""
            )
            print(
                f"\nSOLVED in {outcome['time_sec']}s "
                f"({outcome['solve_source']}{cand_note})"
            )
            print(f"\nProgram:\n{outcome.get('program', '')}")
            for tr in outcome.get("test_results", []):
                status = "CORRECT" if tr["correct"] else "WRONG"
                print(f"  Test {tr['test_idx']}: {status}")
        else:
            cand_note = (
                f", search={outcome['search_candidates_tried']}"
                if outcome.get("search_candidates_tried", 0) > 0 else ""
            )
            print(
                f"\nFAILED after {outcome['time_sec']}s "
                f"(source={outcome.get('solve_phase', 'unsolved')}"
                f"{cand_note}, rounds={outcome['refinement_rounds']})"
            )
        return

    # --- batch mode ---
    task_ids = [args.task] if args.task else None
    solved_count = 0
    total_count = 0

    print("=" * 60)

    def on_done(task_id: str, outcome: dict) -> None:
        nonlocal solved_count, total_count
        total_count += 1
        if outcome["solved"]:
            solved_count += 1
        status = "SOLVED" if outcome["solved"] else "      "
        src = f" [{outcome['solve_source']}]" if outcome["solved"] else ""
        cand_note = (
            f", {outcome['total_candidates']} cand"
            if outcome.get("total_candidates", 0) > 0 else ""
        )
        print(
            f"  [{total_count}] {task_id}: {status} "
            f"({outcome['time_sec']}s{cand_note})"
            f"{src}  [{solved_count}/{total_count}]"
        )
        log.info(
            "%s %s cand=%d %.2fs",
            "SOLVED" if outcome["solved"] else "FAILED",
            task_id,
            outcome["total_candidates"],
            outcome["time_sec"],
        )

    t0 = time.time()
    try:
        report = run_evaluation(
            ds,
            library=library,
            program_store=program_store,
            config=config,
            trace_store=trace_store,
            freeze_stores=True,
            limit=args.limit,
            task_ids=task_ids,
            on_task_done=on_done,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        report = {"interrupted": True, "solved": solved_count, "total": total_count}

    elapsed = time.time() - t0
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if trace_store is not None:
        trace_store.save_json(Path(args.trace_store))

    print("=" * 60)
    print(
        f"Solved: {report.get('solved', '?')}/{report.get('total', '?')} "
        f"({report.get('solve_rate', '?')}) in {elapsed:.1f}s"
    )
    tm = report.get("transfer_metrics", {})
    if tm:
        print(
            f"Transfer: "
            f"retrieval_total={tm.get('retrieval_solves_total', 0)}, "
            f"transfer_backed={tm.get('retrieval_solves_transfer_backed', 0)}, "
            f"leaf_only={tm.get('retrieval_solves_leaf_only', 0)}"
        )
        lq = tm.get("library_quality", {})
        if lq:
            print(
                f"Library: "
                f"entries={lq.get('total_entries', 0)}, "
                f"strong={lq.get('strong_abstractions', 0)}, "
                f"weak={lq.get('weak_abstractions', 0)}, "
                f"avg_mdl_gain={lq.get('avg_mdl_gain', 0)}"
            )
    print(f"Results: {results_file}")

    # --- failure clusters ---
    fc = report.get("failure_clusters", {})
    clusters = fc.get("clusters", {})
    if clusters:
        print(f"Failure clusters ({fc.get('total_unsolved', 0)} unsolved):")
        for name, info in clusters.items():
            print(f"  {name}: {info['count']}")

    # --- gap diagnosis ---
    if "tasks" in report and report.get("total", 0) > 0:
        from aria.diagnosis import diagnose
        diag = diagnose(report["tasks"])
        nm = diag["near_misses"]
        gc = diag["gap_categories"]
        hc = diag["hypothesis_coverage"]
        if hc["skeletons_tested"] > 0 or nm["count"] > 0:
            print(
                f"Hypotheses: "
                f"skeleton_tested={hc['skeletons_tested']}, "
                f"skeleton_solved={hc['solved_by_skeleton']}, "
                f"near_misses={nm['count']}"
            )
        if gc["no_hints"] > 0 or gc["near_miss_not_solved"] > 0:
            print(
                f"Gaps: "
                f"no_hints={gc['no_hints']}, "
                f"hints_no_near_miss={gc['hints_no_near_miss']}, "
                f"near_miss_not_solved={gc['near_miss_not_solved']}"
            )

    # --- optional submission.json ---
    if args.submission and "tasks" in report:
        from aria.submission import (
            build_submission,
            hypotheses_from_eval_outcomes,
            save_submission,
        )

        hyps = hypotheses_from_eval_outcomes(report["tasks"])
        sub = build_submission(hyps)
        sub_path = Path(args.submission)
        save_submission(sub, sub_path)
        print(f"Submission: {sub_path} ({len(sub)} tasks)")


if __name__ == "__main__":
    main()
