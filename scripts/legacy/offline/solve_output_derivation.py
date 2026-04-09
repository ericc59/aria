#!/usr/bin/env python3
"""Solve stage-1 output derivation only.

Usage:
    python scripts/solve_output_derivation.py --task 1a6449f1
    python scripts/solve_output_derivation.py --n 20
    python scripts/solve_output_derivation.py --task 1a6449f1 --output logs/output_derivation_1a6449f1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.core.output_derivation import (  # noqa: E402
    OutputDerivationSpec,
    infer_output_derivation_spec,
    infer_verified_output_derivation_specs,
    predict_output_derivation,
)
from aria.datasets import get_dataset, list_task_ids, load_arc_task  # noqa: E402


def _spec_to_dict(spec: OutputDerivationSpec) -> dict[str, object]:
    return {
        "candidate_kind": spec.candidate_kind,
        "relation": spec.relation,
        "selector": spec.selector,
        "params": dict(spec.params),
        "rationale": spec.rationale,
    }


def _task_report(dataset: str, task_id: str) -> tuple[dict[str, object], bool]:
    ds = get_dataset(dataset)
    task = load_arc_task(ds, task_id)
    specs = infer_verified_output_derivation_specs(task.train)
    primary = specs[0] if specs else None

    demos: list[dict[str, object]] = []
    for idx, demo in enumerate(task.train):
        predictions = []
        for spec in specs:
            predicted = predict_output_derivation(spec, demo.input)
            predictions.append({
                "candidate_kind": spec.candidate_kind,
                "relation": spec.relation,
                "selector": spec.selector,
                "predicted_shape": None if predicted is None else tuple(int(v) for v in predicted.shape),
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
        "solved_stage1_derivation": primary is not None,
        "primary_spec": _spec_to_dict(primary) if primary else None,
        "all_specs": [_spec_to_dict(spec) for spec in specs],
        "demos": demos,
    }
    return report, primary is not None


def _slice_report(dataset: str, n: int) -> dict[str, object]:
    ds = get_dataset(dataset)
    task_ids = list_task_ids(ds)[:n]
    records: list[dict[str, object]] = []
    solved = 0
    for task_id in task_ids:
        task = load_arc_task(ds, task_id)
        spec = infer_output_derivation_spec(task.train)
        ok = spec is not None
        solved += int(ok)
        records.append({
            "task_id": task_id,
            "solved_stage1_derivation": ok,
            "candidate_kind": None if spec is None else spec.candidate_kind,
            "relation": None if spec is None else spec.relation,
            "selector": None if spec is None else spec.selector,
            "rationale": "NO_VERIFIED_DERIVATION_RULE" if spec is None else spec.rationale,
        })
    return {
        "dataset": dataset,
        "n_tasks": len(task_ids),
        "solved_stage1_derivation": solved,
        "coverage": (solved / len(task_ids)) if task_ids else 0.0,
        "tasks": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve stage-1 output derivation only")
    parser.add_argument("--dataset", default="v2-train", help="Dataset name (default: v2-train)")
    parser.add_argument("--task", default="", help="Single task ID to inspect")
    parser.add_argument("--n", type=int, default=0, help="Run first N tasks instead of one task")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    if not args.task and args.n <= 0:
        parser.error("provide --task TASK_ID or --n N")
    if args.task and args.n > 0:
        parser.error("use either --task or --n, not both")

    if args.task:
        report, ok = _task_report(args.dataset, args.task)
        print(f"Task: {report['task_id']} ({report['dataset']})")
        print(f"  Stage-1 derivation solved: {report['solved_stage1_derivation']}")
        if report["primary_spec"] is None:
            print("  Rule: NONE")
        else:
            spec = report["primary_spec"]
            print(f"  Candidate kind: {spec['candidate_kind']}")
            print(f"  Relation: {spec['relation']}")
            print(f"  Selector: {spec['selector']}")
            print(f"  Params: {spec['params']}")
            print(f"  Rationale: {spec['rationale']}")
        print("  Demos:")
        for demo in report["demos"]:
            print(f"    demo {demo['demo_idx']}: {demo['input_dims']} -> {demo['output_dims']}")
            for pred in demo["predictions"]:
                print(
                    f"      {pred['candidate_kind']} / {pred['relation']} / "
                    f"{pred['selector']}: predicts {pred['predicted_shape']}"
                )
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
    print(f"Stage-1 derivation solved: {report['solved_stage1_derivation']}")
    print(f"Coverage: {report['coverage']:.1%}")
    print("Per task:")
    for row in report["tasks"]:
        label = row["candidate_kind"] or "NONE"
        relation = row["relation"] or "-"
        selector = row["selector"] or "-"
        print(f"  {row['task_id']}: {label} / {relation} / {selector} :: {row['rationale']}")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()
