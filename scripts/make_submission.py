#!/usr/bin/env python3
"""Generate an ARC-AGI-2 submission.json from eval results.

Usage:
    # From a single eval report:
    python scripts/make_submission.py results/eval_v2-eval_*.json

    # From multiple reports (top-k merging):
    python scripts/make_submission.py results/run1.json results/run2.json

    # Custom output path:
    python scripts/make_submission.py results/*.json -o submission.json

    # Score against ground truth:
    python scripts/make_submission.py results/*.json --score --dataset v2-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aria.submission import (
    build_submission,
    hypotheses_from_eval_outcomes,
    load_submission,
    save_submission,
    score_submission,
)


def _load_outcomes(paths: list[Path]) -> list[dict]:
    """Load eval outcomes from one or more report JSON files."""
    outcomes: list[dict] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        # A report wraps outcomes in a "tasks" key.
        if "tasks" in data:
            tasks = data["tasks"]
        elif isinstance(data, list):
            tasks = data
        else:
            print(f"Warning: unrecognized format in {path}, skipping", file=sys.stderr)
            continue

        # Assign rank based on file order so that earlier files are preferred.
        rank_base = len(outcomes)
        for i, t in enumerate(tasks):
            t.setdefault("rank", rank_base + i)
        outcomes.extend(tasks)

    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an ARC-AGI-2 submission.json from eval results"
    )
    parser.add_argument(
        "reports", nargs="+", type=Path,
        help="One or more eval report JSON files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("submission.json"),
        help="Output path (default: submission.json)",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Score the submission against ground truth",
    )
    parser.add_argument(
        "--dataset", default="v2-eval",
        help="Dataset for scoring (default: v2-eval)",
    )
    args = parser.parse_args()

    outcomes = _load_outcomes(args.reports)
    if not outcomes:
        print("No outcomes found in the provided files.", file=sys.stderr)
        sys.exit(1)

    hypotheses = hypotheses_from_eval_outcomes(outcomes)
    submission = build_submission(hypotheses)
    save_submission(submission, args.output)

    n_tasks = len(submission)
    n_outputs = sum(len(v) for v in submission.values())
    print(f"Submission: {n_tasks} tasks, {n_outputs} outputs -> {args.output}")

    if args.score:
        from aria.datasets import get_dataset, iter_tasks
        import numpy as np

        ds = get_dataset(args.dataset)
        ground_truth: dict[str, list] = {}
        for task_id, task in iter_tasks(ds):
            ground_truth[task_id] = [pair.output for pair in task.test]

        scores = score_submission(submission, ground_truth)
        print(
            f"Single-attempt: {scores['single_attempt_correct']}/{scores['total']} "
            f"({scores['single_attempt_rate']:.2%})"
        )
        print(
            f"Two-attempt:    {scores['two_attempt_correct']}/{scores['total']} "
            f"({scores['two_attempt_rate']:.2%})"
        )


if __name__ == "__main__":
    main()
