#!/usr/bin/env python3
"""Solve combined stage 1: output size, then direct derivation relation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.core.output_stage1 import infer_output_stage1_spec  # noqa: E402
from aria.datasets import get_dataset, list_task_ids, load_arc_task  # noqa: E402


def _task_report(dataset: str, task_id: str) -> tuple[dict[str, object], bool]:
    ds = get_dataset(dataset)
    task = load_arc_task(ds, task_id)
    spec = infer_output_stage1_spec(task.train)
    report = {
        "dataset": dataset,
        "task_id": task_id,
        "n_demos": len(task.train),
        "solved_stage1": spec is not None,
        "size_spec": None if spec is None else {
            "mode": spec.size_spec.mode,
            "params": dict(spec.size_spec.params),
            "rationale": spec.size_spec.rationale,
        },
        "derivation_spec": None if spec is None or spec.derivation_spec is None else {
            "candidate_kind": spec.derivation_spec.candidate_kind,
            "relation": spec.derivation_spec.relation,
            "selector": spec.derivation_spec.selector,
            "params": dict(spec.derivation_spec.params),
            "rationale": spec.derivation_spec.rationale,
        },
        "render_spec": None if spec is None or spec.render_spec is None else dict(spec.render_spec),
    }
    return report, spec is not None


def _slice_report(dataset: str, n: int) -> dict[str, object]:
    ds = get_dataset(dataset)
    task_ids = list_task_ids(ds)[:n]
    solved = 0
    with_derivation = 0
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        task = load_arc_task(ds, task_id)
        spec = infer_output_stage1_spec(task.train)
        if spec is not None:
            solved += 1
            if spec.derivation_spec is not None:
                with_derivation += 1
        rows.append({
            "task_id": task_id,
            "solved_stage1": spec is not None,
            "size_mode": None if spec is None else spec.size_spec.mode,
            "derivation": None if spec is None or spec.derivation_spec is None else (
                f"{spec.derivation_spec.candidate_kind}/{spec.derivation_spec.relation}/{spec.derivation_spec.selector}"
            ),
            "render": None if spec is None or spec.render_spec is None else spec.render_spec.get("kind"),
        })
    return {
        "dataset": dataset,
        "n_tasks": len(task_ids),
        "solved_stage1": solved,
        "coverage": 0.0 if not task_ids else solved / len(task_ids),
        "with_derivation": with_derivation,
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve combined stage 1")
    parser.add_argument("--dataset", default="v2-train")
    parser.add_argument("--task", default="")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.task and args.n <= 0:
        parser.error("provide --task TASK_ID or --n N")
    if args.task and args.n > 0:
        parser.error("use either --task or --n, not both")

    if args.task:
        report, ok = _task_report(args.dataset, args.task)
        print(f"Task: {report['task_id']} ({report['dataset']})")
        print(f"  Stage-1 solved: {report['solved_stage1']}")
        if report["size_spec"] is None:
            print("  Size: NONE")
            print("  Derivation: NONE")
            print("  Render: NONE")
        else:
            size = report["size_spec"]
            print(f"  Size mode: {size['mode']}")
            print(f"  Size params: {size['params']}")
            print(f"  Size rationale: {size['rationale']}")
            deriv = report["derivation_spec"]
            if deriv is None:
                print("  Derivation: NONE")
            else:
                print(f"  Derivation kind: {deriv['candidate_kind']}")
                print(f"  Derivation relation: {deriv['relation']}")
                print(f"  Derivation selector: {deriv['selector']}")
                print(f"  Derivation params: {deriv['params']}")
                print(f"  Derivation rationale: {deriv['rationale']}")
            render = report["render_spec"]
            if render is None:
                print("  Render: NONE")
            else:
                params = {k: v for k, v in render.items() if k not in {"kind", "rationale"}}
                print(f"  Render kind: {render['kind']}")
                print(f"  Render params: {params}")
                print(f"  Render rationale: {render['rationale']}")
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
    print(f"With derivation: {report['with_derivation']}")
    print("Per task:")
    for row in report["tasks"]:
        print(
            f"  {row['task_id']}: "
            f"{row['size_mode'] or 'NONE'} :: "
            f"{row['derivation'] or '-'} :: "
            f"{row['render'] or '-'}"
        )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()
