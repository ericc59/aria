"""Run the program enumerator on ARC tasks to build a training corpus.

Usage:
    python scripts/enumerate_corpus.py --dataset v1-train --limit 400 --timeout 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from aria.solver import load_task
from aria.enumerate import enumerate_programs
from aria.runtime.program import program_to_text


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


def main():
    parser = argparse.ArgumentParser(description="Enumerate programs for ARC tasks")
    parser.add_argument("--dataset", default="v1-train", choices=DATA_ROOTS.keys())
    parser.add_argument("--limit", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds per task")
    parser.add_argument("--max-steps", type=int, default=5, help="Max program steps")
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--output", default="scripts/corpus_report.json")
    args = parser.parse_args()

    data_dir = DATA_ROOTS[args.dataset]
    fnames = sorted(os.listdir(data_dir))
    if args.limit > 0:
        fnames = fnames[:args.limit]

    results = {}
    solved = 0
    total = 0
    total_programs = 0
    t_start = time.time()

    print(f"Processing {len(fnames)} tasks from '{args.dataset}'")
    print(f"  max_steps={args.max_steps}, timeout={args.timeout}s, max_candidates={args.max_candidates}")
    print("-" * 60)

    for i, fname in enumerate(fnames):
        task_id = fname.replace(".json", "")
        path = os.path.join(data_dir, fname)

        try:
            with open(path) as f:
                task = load_task(json.load(f))
        except Exception as e:
            results[task_id] = {"error": str(e), "programs": []}
            total += 1
            continue

        total += 1
        t0 = time.time()
        programs = enumerate_programs(
            task,
            max_steps=args.max_steps,
            timeout_sec=args.timeout,
            max_candidates=args.max_candidates,
        )
        elapsed = time.time() - t0

        if programs:
            solved += 1
            total_programs += len(programs)

        results[task_id] = {
            "solved": bool(programs),
            "num_programs": len(programs),
            "time_sec": round(elapsed, 3),
            "programs": [program_to_text(p) for p in programs[:5]],  # cap at 5
        }

        status = "SOLVED" if programs else "      "
        print(f"  [{i+1}/{len(fnames)}] {task_id}: {status} ({len(programs)} programs, {elapsed:.1f}s)")

    wall_time = time.time() - t_start

    print("=" * 60)
    print(f"Tasks: {total} total, {solved} solved ({100*solved/max(total,1):.1f}%)")
    print(f"Programs found: {total_programs}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time/max(total,1):.1f}s/task avg)")
    print("=" * 60)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "total_tasks": total,
        "solved": solved,
        "total_programs": total_programs,
        "wall_time_sec": round(wall_time, 1),
        "config": {
            "max_steps": args.max_steps,
            "timeout_sec": args.timeout,
            "max_candidates": args.max_candidates,
        },
        "tasks": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
