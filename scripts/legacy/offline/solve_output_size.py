#!/usr/bin/env python3
"""Solve stage-1 output size only.

Usage:
    python scripts/solve_output_size.py --task 009d5c81
    python scripts/solve_output_size.py --task 0520fde7 --all-specs
    python scripts/solve_output_size.py --n 10
    python scripts/solve_output_size.py --n 100 --output logs/output_size_100.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.core.output_size import (
    OutputSizeSpec,
    infer_output_size_spec,
    infer_verified_output_size_specs,
    predict_output_size,
)
from aria.datasets import get_dataset, list_task_ids, load_arc_task


def _spec_to_dict(spec: OutputSizeSpec) -> dict[str, object]:
    params: dict[str, object] = {}
    for key, value in spec.params.items():
        if isinstance(value, Fraction):
            params[key] = f"{value.numerator}/{value.denominator}"
        else:
            params[key] = value
    return {
        "mode": spec.mode,
        "params": params,
        "rationale": spec.rationale,
    }


def _task_report(dataset: str, task_id: str, *, all_specs: bool) -> tuple[dict[str, object], bool]:
    ds = get_dataset(dataset)
    task = load_arc_task(ds, task_id)
    specs = infer_verified_output_size_specs(task.train)
    primary = specs[0] if specs else None

    demos: list[dict[str, object]] = []
    for idx, demo in enumerate(task.train):
        predictions = []
        for spec in specs:
            predictions.append({
                "mode": spec.mode,
                "predicted": predict_output_size(spec, demo.input),
            })
        demos.append({
            "demo_idx": idx,
            "input_dims": tuple(int(v) for v in demo.input.shape),
            "output_dims": tuple(int(v) for v in demo.output.shape),
            "predictions": predictions,
        })

    report: dict[str, object] = {
        "dataset": dataset,
        "task_id": task_id,
        "n_demos": len(task.train),
        "solved_stage1": primary is not None,
        "primary_spec": _spec_to_dict(primary) if primary else None,
        "demos": demos,
    }
    if all_specs:
        report["all_specs"] = [_spec_to_dict(spec) for spec in specs]
    return report, primary is not None


def _slice_report(dataset: str, n: int) -> dict[str, object]:
    ds = get_dataset(dataset)
    task_ids = list_task_ids(ds)[:n]
    records: list[dict[str, object]] = []
    solved = 0
    for task_id in task_ids:
        task = load_arc_task(ds, task_id)
        spec = infer_output_size_spec(task.train)
        ok = spec is not None
        solved += int(ok)
        records.append({
            "task_id": task_id,
            "solved_stage1": ok,
            "mode": spec.mode if spec else None,
            "rationale": spec.rationale if spec else "NO_VERIFIED_SIZE_RULE",
        })
    return {
        "dataset": dataset,
        "n_tasks": len(task_ids),
        "solved_stage1": solved,
        "coverage": (solved / len(task_ids)) if task_ids else 0.0,
        "tasks": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve stage-1 output size only")
    parser.add_argument("--dataset", default="v2-train", help="Dataset name (default: v2-train)")
    parser.add_argument("--task", default="", help="Single task ID to inspect")
    parser.add_argument("--n", type=int, default=0, help="Run first N tasks instead of one task")
    parser.add_argument("--all-specs", action="store_true", help="Include all verified size specs for a single task")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    if not args.task and args.n <= 0:
        parser.error("provide --task TASK_ID or --n N")
    if args.task and args.n > 0:
        parser.error("use either --task or --n, not both")

    if args.task:
        report, ok = _task_report(args.dataset, args.task, all_specs=args.all_specs)
        print(f"Task: {report['task_id']} ({report['dataset']})")
        print(f"  Stage-1 solved: {report['solved_stage1']}")
        if report["primary_spec"] is not None:
            spec = report["primary_spec"]
            print(f"  Mode: {spec['mode']}")
            print(f"  Params: {spec['params']}")
            print(f"  Rationale: {spec['rationale']}")
        else:
            print("  Mode: NONE")
            print("  Rationale: no verified size rule")
        print("  Demos:")
        for demo in report["demos"]:
            print(
                f"    demo {demo['demo_idx']}: "
                f"{demo['input_dims']} -> {demo['output_dims']}"
            )
            for pred in demo["predictions"]:
                print(f"      {pred['mode']}: predicts {pred['predicted']}")
        if args.all_specs:
            print("  All verified specs:")
            for spec in report.get("all_specs", []):
                print(f"    - {spec['mode']} {spec['params']} :: {spec['rationale']}")
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
            print(f"  JSON: {out_path}")
        if not ok:
            raise SystemExit(1)
        return

    report = _slice_report(args.dataset, args.n)
    print(f"Dataset: {report['dataset']}")
    print(f"Tasks: {report['n_tasks']}")
    print(f"Stage-1 solved: {report['solved_stage1']}")
    print(f"Coverage: {report['coverage']:.1%}")
    print("Per task:")
    for row in report["tasks"]:
        print(
            f"  {row['task_id']}: "
            f"{row['mode'] or 'NONE'} :: {row['rationale']}"
        )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()
