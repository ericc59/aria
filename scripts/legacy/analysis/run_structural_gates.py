#!/usr/bin/env python3
"""Run the structural gates evaluation on the gold annotation set.

Usage:
    python scripts/run_structural_gates.py [--no-solve] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.eval.structural_gates_runner import run_structural_gates
from aria.eval.structural_gates_report import format_text_report, format_json_report


GOLD_PATH = Path(__file__).resolve().parent.parent / "aria" / "eval" / "structural_gates_tasks.yaml"


def main():
    parser = argparse.ArgumentParser(description="Structural gates evaluation")
    parser.add_argument("--no-solve", action="store_true", help="Skip exact solve (faster)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--dataset", default="v2-eval", help="Dataset name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for recall metrics")
    parser.add_argument("--task", action="append", dest="tasks", help="Run only specific task IDs")
    args = parser.parse_args()

    print(f"Running structural gates evaluation...", file=sys.stderr)
    print(f"  Gold: {GOLD_PATH}", file=sys.stderr)
    print(f"  Dataset: {args.dataset}", file=sys.stderr)
    print(f"  Top-K: {args.top_k}", file=sys.stderr)
    print(f"  Exact solve: {not args.no_solve}", file=sys.stderr)
    print(file=sys.stderr)

    report = run_structural_gates(
        gold_path=GOLD_PATH,
        dataset_name=args.dataset,
        top_k=args.top_k,
        run_exact_solve=not args.no_solve,
        task_ids=args.tasks,
    )

    if args.json:
        print(json.dumps(format_json_report(report), indent=2))
    else:
        print(format_text_report(report))


if __name__ == "__main__":
    main()
