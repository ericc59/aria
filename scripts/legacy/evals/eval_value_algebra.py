"""Benchmark: measure value-algebra impact on v2-eval.

Runs the solver on v2-eval with and without value-algebra templates,
reporting solve counts, near-misses, and per-task deltas.

Usage:
  python scripts/eval_value_algebra.py [--limit N] [--dataset v2-eval] [--budget N]
  python scripts/eval_value_algebra.py --mode baseline  # templates OFF
  python scripts/eval_value_algebra.py --mode enabled   # templates ON (default)
  python scripts/eval_value_algebra.py --mode ab        # both, side by side
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from aria.datasets import get_dataset, list_task_ids, load_arc_task
from aria.eval import evaluate_task, EvalConfig
from aria.library.store import Library
from aria.synthesize import set_value_algebra_templates


def run_eval(task_ids, ds, config, label):
    """Run eval on tasks, return (summary_dict, per_task_list)."""
    lib = Library()
    results = []
    solved_count = 0
    near_miss_count = 0
    total_candidates = 0
    solve_phases = {}
    t_start = time.time()

    for i, tid in enumerate(task_ids):
        task = load_arc_task(ds, tid)
        t0 = time.time()
        outcome = evaluate_task(tid, task, library=lib, config=config)
        elapsed = time.time() - t0

        is_solved = outcome.get("solved", False)
        solve_phase = outcome.get("solve_phase", "unsolved")
        candidates = outcome.get("total_candidates", 0)

        has_near_miss = False
        if outcome.get("skeleton_near_miss"):
            has_near_miss = True
        if outcome.get("failure_bucket") == "wrong_output":
            has_near_miss = True

        if is_solved:
            solved_count += 1
        if has_near_miss:
            near_miss_count += 1
        total_candidates += candidates
        solve_phases[solve_phase] = solve_phases.get(solve_phase, 0) + 1

        entry = {
            "task_id": tid,
            "solved": is_solved,
            "solve_phase": solve_phase,
            "time_sec": round(elapsed, 1),
            "candidates": candidates,
            "near_miss": has_near_miss,
        }
        results.append(entry)

        status = "SOLVED" if is_solved else ("NEAR" if has_near_miss else "FAIL")
        print(
            f"  [{label}] [{i+1}/{len(task_ids)}] {tid}: {status} "
            f"({solve_phase}, {candidates} cands, {elapsed:.1f}s)",
            flush=True,
        )

    total_time = time.time() - t_start
    summary = {
        "label": label,
        "tasks_run": len(task_ids),
        "solved": solved_count,
        "near_misses": near_miss_count,
        "total_candidates": total_candidates,
        "total_time_sec": round(total_time, 1),
        "solve_phases": solve_phases,
        "results": results,
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default="v2-eval")
    parser.add_argument("--output", default=None)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--mode", choices=["baseline", "enabled", "ab"], default="ab")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    config = EvalConfig(
        max_search_candidates=args.budget,
        max_refinement_rounds=2,
        max_search_steps=3,
    )

    summaries = {}

    if args.mode in ("baseline", "ab"):
        print(f"\n{'='*60}")
        print("BASELINE: value-algebra templates DISABLED")
        print(f"{'='*60}")
        set_value_algebra_templates(False)
        summaries["baseline"], baseline_results = run_eval(task_ids, ds, config, "BASE")

    if args.mode in ("enabled", "ab"):
        print(f"\n{'='*60}")
        print("ENABLED: value-algebra templates ENABLED")
        print(f"{'='*60}")
        set_value_algebra_templates(True)
        summaries["enabled"], enabled_results = run_eval(task_ids, ds, config, "VA")

    # Report
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for label, s in summaries.items():
        print(f"\n{label.upper()}:")
        print(f"  Solved: {s['solved']}/{s['tasks_run']} ({100*s['solved']/max(1,s['tasks_run']):.1f}%)")
        print(f"  Near-misses: {s['near_misses']}")
        print(f"  Total candidates: {s['total_candidates']}")
        print(f"  Time: {s['total_time_sec']}s")
        print(f"  Solve phases: {s['solve_phases']}")

    if args.mode == "ab":
        # Delta analysis
        base_solved = {r["task_id"] for r in baseline_results if r["solved"]}
        va_solved = {r["task_id"] for r in enabled_results if r["solved"]}
        new_solves = va_solved - base_solved
        lost_solves = base_solved - va_solved

        base_near = {r["task_id"] for r in baseline_results if r["near_miss"]}
        va_near = {r["task_id"] for r in enabled_results if r["near_miss"]}
        new_near = va_near - base_near
        lost_near = base_near - va_near

        print(f"\nDELTA:")
        print(f"  New solves: {len(new_solves)} {sorted(new_solves)}")
        print(f"  Lost solves: {len(lost_solves)} {sorted(lost_solves)}")
        print(f"  New near-misses: {len(new_near)} {sorted(new_near)}")
        print(f"  Lost near-misses: {len(lost_near)} {sorted(lost_near)}")

        # Per-task VA solve phase attribution
        va_phase_tasks = {}
        for r in enabled_results:
            if r["solved"] and r["task_id"] not in base_solved:
                phase = r["solve_phase"]
                va_phase_tasks.setdefault(phase, []).append(r["task_id"])
        if va_phase_tasks:
            print(f"\n  New solves by phase:")
            for phase, tids in va_phase_tasks.items():
                print(f"    {phase}: {tids}")

        summaries["delta"] = {
            "new_solves": sorted(new_solves),
            "lost_solves": sorted(lost_solves),
            "new_near_misses": sorted(new_near),
            "lost_near_misses": sorted(lost_near),
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
