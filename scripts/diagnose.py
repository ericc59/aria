#!/usr/bin/env python3
"""Diagnose eval results: what's working, what's close, what's missing.

Usage:
    python scripts/diagnose.py results/eval_v2-eval_20260325_120000.json
    python scripts/diagnose.py results/v1-train.json
    python scripts/diagnose.py results/eval_v2-eval_*.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aria.diagnosis import diagnose, format_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose eval/solve results to identify gaps and near-misses"
    )
    parser.add_argument(
        "report", nargs="+",
        help="One or more eval/solve report JSON files",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON diagnosis instead of formatted text",
    )
    parser.add_argument(
        "--near-misses", type=int, default=5,
        help="Max near-miss examples to show (default: 5)",
    )
    args = parser.parse_args()

    all_tasks: list[dict] = []
    for report_path in args.report:
        path = Path(report_path)
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        tasks = data.get("tasks", [])
        if not tasks:
            print(f"No tasks found in {path}", file=sys.stderr)
            continue
        all_tasks.extend(tasks)

    if not all_tasks:
        print("No task outcomes to diagnose.", file=sys.stderr)
        sys.exit(1)

    diag = diagnose(all_tasks)

    if args.json:
        print(json.dumps(diag, indent=2, default=str))
    else:
        # Respect --near-misses limit in formatted output
        if diag["near_misses"]["examples"]:
            diag["near_misses"]["examples"] = diag["near_misses"]["examples"][:args.near_misses]
        print(format_diagnosis(diag))


if __name__ == "__main__":
    main()
