#!/usr/bin/env python3
"""Run mechanism evidence audit on a v2-train slice.

Usage:
    python scripts/audit_evidence.py --n 100
    python scripts/audit_evidence.py --n 400 --output logs/audit_evidence.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, list_task_ids, load_arc_task
from aria.core.mechanism_audit import run_audit, format_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of tasks")
    parser.add_argument("--dataset", default="v2-train")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)[:args.n]

    def demos_fn(tid):
        return load_arc_task(ds, tid).train

    print(f"Auditing {len(task_ids)} tasks from {args.dataset}...")
    report = run_audit(task_ids, demos_fn)

    print(format_report(report))

    # Per-task detail for tasks where evidence was nontrivial
    print("\n--- Per-Task Detail (tasks with compilable lanes) ---")
    for r in report.records:
        if not any(lf.compile_attempted for lf in r.lanes):
            continue
        top = r.lanes[0] if r.lanes else None
        verified_lanes = [lf.name for lf in r.lanes if lf.verified]
        non_verified = [(lf.name, lf.residual_diff) for lf in r.lanes
                        if lf.compile_attempted and not lf.verified and lf.residual_diff >= 0]
        print(f"  {r.task_id}: top_class={top.name if top else '?'}({top.class_score:.2f}) "
              f"verified={verified_lanes or 'none'} "
              f"residuals={non_verified or 'none'} "
              f"match={r.class_exec_match}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_tasks": report.n_tasks,
            "n_solved": report.n_solved,
            "top_lane_verifies": report.top_lane_verifies,
            "top2_contains_verifier": report.top2_contains_verifier,
            "per_lane": {
                "replication": {"top": report.replication_top_count, "verifies": report.replication_top_verifies},
                "relocation": {"top": report.relocation_top_count, "verifies": report.relocation_top_verifies},
                "periodic": {"top": report.periodic_top_count, "verifies": report.periodic_top_verifies},
                "transform": {"top": report.transform_top_count, "verifies": report.transform_top_verifies},
            },
            "false_positives": [{"task_id": t, "lane": l, "score": s} for t, l, s in report.false_positives],
        }
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
