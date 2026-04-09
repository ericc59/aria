#!/usr/bin/env python3
"""Export guidance data from ARC tasks.

Usage:
    python scripts/export_guidance_data.py --dataset v1-train --limit 20 --output /tmp/guidance.jsonl
    python scripts/export_guidance_data.py --dataset v1-train --labels --output /tmp/labels.jsonl
    python scripts/export_guidance_data.py --evaluate /tmp/guidance.jsonl --labels-path /tmp/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export guidance data from ARC tasks")
    parser.add_argument("--dataset", default="v1-train", help="Dataset name (v1-train, v1-eval, etc.)")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks to process (0 = all)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--labels", action="store_true", help="Export labels instead of full records")
    parser.add_argument("--coverage", action="store_true", help="Report label coverage")
    parser.add_argument("--evaluate", help="Evaluate proposals from a records JSONL file")
    parser.add_argument("--labels-path", help="Labels JSONL for evaluation")

    args = parser.parse_args()

    if args.evaluate:
        _evaluate(args.evaluate, args.labels_path, args.output)
        return

    from aria.datasets import get_dataset, list_task_ids, load_arc_task

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    print(f"Processing {len(task_ids)} tasks from {args.dataset}...")

    if args.labels:
        _export_labels(ds, task_ids, args.output, args.coverage)
    else:
        _export_records(ds, task_ids, args.output)


def _export_records(ds, task_ids: list[str], output: str) -> None:
    from aria.core.guidance_export import export_batch
    from aria.datasets import load_arc_task

    def demos_fn(tid: str):
        task = load_arc_task(ds, tid)
        return task.train

    counts = export_batch(task_ids, demos_fn, output)
    print(f"Exported: {counts['exported']}, Skipped: {counts['skipped']}")
    print(f"Output: {output}")


def _export_labels(ds, task_ids: list[str], output: str, report_coverage: bool) -> None:
    from aria.core.guidance_labels import extract_labels, save_labels, label_coverage
    from aria.datasets import load_arc_task

    labels = []
    skipped = 0
    for i, tid in enumerate(task_ids):
        try:
            task = load_arc_task(ds, tid)
            lbl = extract_labels(tid, task.train)
            labels.append(lbl)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(task_ids)} tasks labeled")
        except Exception as e:
            skipped += 1
            print(f"  Skipped {tid}: {e}", file=sys.stderr)

    save_labels(labels, output)
    print(f"Labels: {len(labels)} exported, {skipped} skipped")
    print(f"Output: {output}")

    if report_coverage:
        cov = label_coverage(labels)
        print("\nLabel coverage:")
        for k, v in cov.items():
            print(f"  {k}: {v}")


def _evaluate(records_path: str, labels_path: str | None, output: str) -> None:
    from aria.core.guidance_export import load_records
    from aria.core.guidance_eval import evaluate_proposals

    records = load_records(records_path)
    print(f"Loaded {len(records)} records from {records_path}")

    if labels_path:
        from aria.core.guidance_labels import load_labels
        labels = load_labels(labels_path)
    else:
        # Extract labels on the fly from records
        from aria.core.guidance_labels import extract_labels, TaskGuidanceLabels
        from aria.types import DemoPair, grid_from_list

        labels = []
        for rec in records:
            demos = tuple(
                DemoPair(
                    input=grid_from_list(d["input"]),
                    output=grid_from_list(d["output"]),
                )
                for d in rec.get("train_demos", [])
            )
            labels.append(extract_labels(rec["task_id"], demos))

    report = evaluate_proposals(records, labels)
    result = report.to_dict()

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nEvaluation results written to {output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
