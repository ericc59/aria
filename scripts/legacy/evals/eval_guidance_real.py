#!/usr/bin/env python3
"""Run real-data evaluation of learned guidance models on ARC tasks.

Usage:
    python scripts/eval_guidance_real.py --dataset v1-train --limit 100
    python scripts/eval_guidance_real.py --dataset v1-train --output /tmp/guidance_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data guidance evaluation")
    parser.add_argument("--dataset", default="v1-train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--subproblems", nargs="*", default=None)
    args = parser.parse_args()

    from aria.datasets import get_dataset, list_task_ids, load_arc_task
    from aria.core.guidance_real_eval import run_real_evaluation

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    print(f"Loading {len(task_ids)} tasks from {args.dataset}...")
    t0 = time.time()

    pairs = []
    skipped = 0
    for tid in task_ids:
        try:
            task = load_arc_task(ds, tid)
            pairs.append((tid, task.train))
        except Exception:
            skipped += 1

    print(f"Loaded {len(pairs)} tasks ({skipped} skipped) in {time.time()-t0:.1f}s")

    print("\nGenerating datasets and evaluating...")
    t1 = time.time()
    reports = run_real_evaluation(pairs, subproblems=args.subproblems)
    print(f"Done in {time.time()-t1:.1f}s\n")

    # Print summary
    print("=" * 70)
    print("REAL-DATA GUIDANCE EVALUATION")
    print("=" * 70)

    for name, report in reports.items():
        print(f"\n{'─' * 50}")
        print(f"Subproblem: {name}")
        print(f"  Tasks: {report.n_tasks}")
        print(f"  Examples: {report.n_examples}")
        print(f"  Classes: {report.n_classes}")
        print(f"  Distribution: {report.label_distribution}")
        print(f"  Sufficient: {report.sufficient_data}")

        if not report.sufficient_data:
            print(f"  Reason: {report.reason}")
            continue

        print(f"\n  {'Model':<25} {'Top-1':>8} {'±σ':>8} {'Top-3':>8} {'AvgRank':>8}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for cv in report.cv_results:
            print(f"  {cv.model_name:<25} {cv.mean_top1:>8.1%} {cv.std_top1:>7.1%} {cv.mean_top3:>8.1%} {cv.mean_avg_rank:>8.2f}")

        print(f"\n  Integration ready: {report.integration_ready}")
        print(f"  Reason: {report.integration_reason}")

    # Save full report
    if args.output:
        full = {name: report.to_dict() for name, report in reports.items()}
        with open(args.output, "w") as f:
            json.dump(full, f, indent=2)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
