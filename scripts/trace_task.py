#!/usr/bin/env python3
"""Trace one task through the canonical pipeline and generate an HTML viewer.

Usage:
    python scripts/trace_task.py --task 00576224 --dataset v2-train
    python scripts/trace_task.py --task 00576224 --mode static
    python scripts/trace_task.py --task 00576224 --mode learned

Output:
    logs/trace_{task_id}.json   — raw trace data
    logs/trace_{task_id}.html   — standalone HTML viewer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, list_task_ids, load_arc_task
from aria.core.trace_solve import solve_with_trace
from aria.core.trace_viewer import generate_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one task through the canonical pipeline")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument("--dataset", default="v2-train", help="Dataset name (default: v2-train)")
    parser.add_argument("--mode", default="all", choices=["static", "deterministic", "learned", "all"],
                        help="Which phases to run")
    parser.add_argument("--output-dir", default="logs", help="Output directory")
    args = parser.parse_args()

    # Load task
    ds = get_dataset(args.dataset)
    task = load_arc_task(ds, args.task)
    demos = task.train
    print(f"Task: {args.task} ({args.dataset})")
    print(f"  Demos: {len(demos)}")
    for i, d in enumerate(demos):
        print(f"  Demo {i}: {d.input.shape} -> {d.output.shape}")

    # Run traced solve
    use_det = args.mode in ("deterministic", "all")
    use_learn = args.mode in ("learned", "all")

    print(f"\nRunning canonical pipeline (det={use_det}, learned={use_learn})...")
    trace = solve_with_trace(
        demos,
        task_id=args.task,
        use_deterministic=use_det,
        use_learned=use_learn,
    )

    print(f"  Solved: {trace.solved} (by: {trace.solver or 'none'})")
    print(f"  Seeds: {len(trace.seeds)}")
    print(f"  Events: {len(trace.events)}")

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"trace_{args.task}.json"
    json_path.write_text(trace.to_json())
    print(f"  Trace JSON: {json_path}")

    html_path = out_dir / f"trace_{args.task}.html"
    html_path.write_text(generate_html(trace))
    print(f"  Trace HTML: {html_path}")


if __name__ == "__main__":
    main()
