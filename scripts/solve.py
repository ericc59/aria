#!/usr/bin/env python3
"""Solve ARC tasks with the canonical search-only ARIA solver.

Usage:
    python scripts/solve.py 38007db0
    python scripts/solve.py 38007db0 --dataset v2-eval --budget 5
    python scripts/solve.py 38007db0 3e6067c3
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from aria.datasets import dataset_names, get_dataset, load_arc_task
from aria.solve import solve_task


def _format_grid(grid) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in grid)


def _solve_one(task_id: str, dataset: str, budget: float) -> tuple[dict, int]:
    ds = get_dataset(dataset)
    task = load_arc_task(ds, task_id)
    demos = [(pair.input, pair.output) for pair in task.train]

    result = solve_task(demos, time_budget=budget)

    payload: dict = {
        "task_id": task_id,
        "dataset": dataset,
        "solved": result["program"] is not None,
        "source": result["source"],
        "time_sec": round(float(result["time"]), 3),
        "description": result["description"],
    }

    if result["program"] is not None:
        program = result["program"]
        test_outputs = [program.execute(pair.input) for pair in task.test]
        payload["test_results"] = [
            {
                "test_idx": idx,
                "correct": (bool((pred == pair.output).all()) if pair.output is not None else None),
                "output": pred.tolist(),
            }
            for idx, (pair, pred) in enumerate(zip(task.test, test_outputs))
        ]
    else:
        payload["test_results"] = []

    return payload, (0 if payload["solved"] else 1)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve ARC tasks with the canonical aria/search solver",
    )
    parser.add_argument("task_ids", nargs="+", help="One or more ARC task IDs")
    parser.add_argument(
        "--dataset",
        default="v2-eval",
        choices=dataset_names(),
        help="Dataset split to load from (default: v2-eval)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=30.0,
        help="Time budget per task in seconds (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of human-readable output",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = []
    exit_code = 0
    for task_id in args.task_ids:
        payload, code = _solve_one(task_id, args.dataset, args.budget)
        results.append(payload)
        exit_code = max(exit_code, code)

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
        return exit_code

    for idx, payload in enumerate(results):
        if idx:
            print("\n" + "=" * 60)
        print(f"Task: {payload['task_id']} ({payload['dataset']})")
        print(f"Solved: {'yes' if payload['solved'] else 'no'}")
        print(f"Source: {payload['source'] or 'unsolved'}")
        print(f"Time: {payload['time_sec']}s")
        if payload["description"]:
            print(f"Program: {payload['description']}")

        for item in payload["test_results"]:
            if item["correct"] is None:
                status = "PREDICTED"
            else:
                status = "CORRECT" if item["correct"] else "WRONG"
            print(f"\nTest {item['test_idx']}: {status}")
            print(_format_grid(item["output"]))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
