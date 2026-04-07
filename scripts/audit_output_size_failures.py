#!/usr/bin/env python3
"""Audit stage-1 output-size failures.

Focuses only on tasks where `infer_output_size_spec` fails and groups them into
coarse structural buckets so size-rule work can be prioritized.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.core.output_size import infer_output_size_spec
from aria.datasets import get_dataset, list_task_ids, load_arc_task
from aria.decomposition import detect_bg, detect_framed_regions, extract_objects
from aria.graph.partition import detect_partition


LABEL_SELECTED_REGION_CROP = "selected_region_crop"
LABEL_SUMMARY_STRIP = "summary_strip_or_table"
LABEL_TEMPLATE_ASSEMBLY = "template_assembly_size"
LABEL_OBJECT_BBOX = "object_bbox_derived"
LABEL_SHELL_EXPANSION = "shell_or_border_expansion"
LABEL_CANVAS_EXPANSION = "content_derived_canvas_expand"
LABEL_LAYOUT_DERIVED = "layout_or_count_derived"
LABEL_MIXED = "mixed_or_unknown"


def classify_failure(task) -> tuple[str, str]:
    demos = task.train
    if not demos:
        return LABEL_MIXED, "no demos"

    out_smaller_all = all(
        d.output.shape[0] <= d.input.shape[0] and d.output.shape[1] <= d.input.shape[1]
        and d.output.shape != d.input.shape
        for d in demos
    )
    out_larger_all = all(
        d.output.shape[0] >= d.input.shape[0] and d.output.shape[1] >= d.input.shape[1]
        and d.output.shape != d.input.shape
        for d in demos
    )

    # Does every demo have an object whose bbox matches output dims?
    if _every_demo_has_matching_object_bbox(demos):
        return LABEL_OBJECT_BBOX, "every demo has object bbox matching output dims"

    # Marker/template assembly: few non-bg objects, output dims vary with marker/template layout.
    if _looks_like_template_assembly(demos):
        return LABEL_TEMPLATE_ASSEMBLY, "few objects + output likely assembled from template/markers"

    # Smaller outputs
    if out_smaller_all:
        if _looks_like_summary_strip(demos):
            return LABEL_SUMMARY_STRIP, "outputs are thin strips/small summaries"
        if _has_partition_or_frames(demos):
            return LABEL_SELECTED_REGION_CROP, "smaller outputs with partition/frame structure"
        return LABEL_SELECTED_REGION_CROP, "smaller outputs likely crop/select a region"

    # Larger outputs
    if out_larger_all:
        if _looks_like_shell_expansion(demos):
            return LABEL_SHELL_EXPANSION, "larger outputs with shell/border expansion cues"
        return LABEL_CANVAS_EXPANSION, "larger outputs likely derive canvas from content structure"

    # Same-dims or mixed dims but size still unresolved -> layout/count dependence.
    if _looks_like_layout_or_count_task(demos):
        return LABEL_LAYOUT_DERIVED, "size appears derived from object count/layout rather than whole-grid rule"

    return LABEL_MIXED, "no dominant size pattern detected"


def _every_demo_has_matching_object_bbox(demos) -> bool:
    for demo in demos:
        bg = detect_bg(demo.input)
        objs = extract_objects(demo.input, bg)
        out_dims = demo.output.shape
        if not any((o.bbox_h, o.bbox_w) == out_dims for o in objs):
            return False
    return True


def _looks_like_template_assembly(demos) -> bool:
    for demo in demos:
        bg = detect_bg(demo.input)
        objs = extract_objects(demo.input, bg)
        non_single = [o for o in objs if o.size > 1]
        singles = [o for o in objs if o.size == 1]
        if len(non_single) > 3 or len(singles) > 10:
            return False
        if len(non_single) < 1 or len(singles) < 1:
            return False
    return True


def _looks_like_summary_strip(demos) -> bool:
    return all(
        demo.output.shape[0] <= 3 or demo.output.shape[1] <= 3
        for demo in demos
    )


def _has_partition_or_frames(demos) -> bool:
    for demo in demos:
        bg = detect_bg(demo.input)
        if detect_partition(demo.input, background=bg) is not None:
            return True
        if detect_framed_regions(demo.input, bg=bg):
            return True
    return False


def _looks_like_shell_expansion(demos) -> bool:
    for demo in demos:
        ir, ic = demo.input.shape
        or_, oc = demo.output.shape
        if or_ != oc:
            return False
        if or_ <= max(ir, ic):
            return False
        bg = detect_bg(demo.input)
        objs = extract_objects(demo.input, bg)
        if len(objs) > 8:
            return False
    return True


def _looks_like_layout_or_count_task(demos) -> bool:
    for demo in demos:
        bg = detect_bg(demo.input)
        objs = extract_objects(demo.input, bg)
        if len(objs) < 2:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit output-size failures")
    parser.add_argument("--dataset", default="v2-train")
    parser.add_argument("--n", type=int, default=0, help="Limit to first N tasks")
    parser.add_argument("--output", default="", help="Optional JSON output")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)
    if args.n > 0:
        task_ids = task_ids[:args.n]

    counts = Counter()
    records = []
    for tid in task_ids:
        task = load_arc_task(ds, tid)
        if infer_output_size_spec(task.train) is not None:
            continue
        label, rationale = classify_failure(task)
        counts[label] += 1
        records.append({
            "task_id": tid,
            "label": label,
            "rationale": rationale,
            "shapes": [
                {
                    "input": tuple(int(v) for v in demo.input.shape),
                    "output": tuple(int(v) for v in demo.output.shape),
                }
                for demo in task.train
            ],
        })

    print(f"Dataset: {args.dataset}")
    print(f"Failed stage-1 tasks: {len(records)}")
    print("Buckets:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")
    print("\nSample tasks:")
    for record in records[:20]:
        print(f"  {record['task_id']}: {record['label']} :: {record['rationale']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "dataset": args.dataset,
            "failed": len(records),
            "counts": dict(counts),
            "records": records,
        }, indent=2))
        print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
