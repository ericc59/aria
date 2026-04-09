#!/usr/bin/env python3
"""Generate a compact, comparable offline scorecard.

Combines training data statistics with policy evaluation metrics into
a single JSON object that can be diffed across runs.

Usage:
    # Stats only (no eval data):
    python scripts/offline_scorecard.py --training-data training_data/

    # Full scorecard with policy eval:
    python scripts/offline_scorecard.py \
        --training-data training_data/ \
        --eval-data data/eval.jsonl \
        --policy heuristic \
        --label "baseline-v1"

    # Compare two saved scorecards:
    python scripts/offline_scorecard.py \
        --compare scorecards/run1.json scorecards/run2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.local_policy import HeuristicBaselinePolicy, LocalCausalLMPolicy
from aria.policy_eval import (
    Scorecard,
    compare_scorecards,
    evaluate,
    load_eval_examples,
)
from aria.training_data import (
    TrainingExample,
    compute_dataset_stats,
    deduplicate,
    load_jsonl,
    validate_examples,
)


def _load_training_examples(data_dir: Path) -> list[TrainingExample]:
    """Load all JSONL files from a training data directory as TrainingExample objects."""
    examples: list[TrainingExample] = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        for row in load_jsonl(jsonl_file):
            examples.append(TrainingExample(
                schema_version=row.get("schema_version", 2),
                task_type=row.get("task_type", ""),
                task_id=row.get("task_id"),
                task_signatures=tuple(row.get("task_signatures", [])),
                current_program=row.get("current_program"),
                round_index=row.get("round_index"),
                feedback=row.get("feedback"),
                target=row.get("target", {}),
                winning_program=row.get("winning_program"),
            ))
    return examples


def build_scorecard(
    *,
    label: str = "",
    policy_name: str = "",
    training_data_dir: Path | None = None,
    eval_path: Path | None = None,
    policy_type: str = "heuristic",
    model_path: str = "",
) -> Scorecard:
    """Build a scorecard from training data and/or eval data."""
    card = Scorecard(run_label=label, policy_name=policy_name or policy_type)

    if training_data_dir is not None:
        examples = _load_training_examples(training_data_dir)
        deduped, n_removed = deduplicate(examples)
        validation = validate_examples(deduped)

        stats = compute_dataset_stats(deduped)
        stats.duplicates_removed = n_removed
        stats.validation_errors = validation.invalid
        card.dataset_stats = stats.to_dict()

        if validation.errors:
            card.dataset_stats["sample_errors"] = validation.errors[:10]

    if eval_path is not None:
        eval_examples = load_eval_examples(eval_path)
        if policy_type == "heuristic":
            policy = HeuristicBaselinePolicy()
        else:
            policy = LocalCausalLMPolicy(
                model_name_or_path=model_path,
                dry_run=not model_path,
            )
        report = evaluate(policy, eval_examples)
        card.eval_report = report.to_dict()

    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline scorecard.")
    parser.add_argument("--training-data", type=Path, default=None,
                        help="Directory with training JSONL files.")
    parser.add_argument("--eval-data", type=Path, default=None,
                        help="JSONL file for policy evaluation.")
    parser.add_argument("--policy", choices=["heuristic", "local_lm"],
                        default="heuristic", help="Policy to evaluate.")
    parser.add_argument("--model", type=str, default="",
                        help="Model path for local_lm policy.")
    parser.add_argument("--label", type=str, default="",
                        help="Label for this run.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write scorecard JSON to file.")
    parser.add_argument("--compare", nargs="+", type=Path, default=None,
                        help="Compare saved scorecard JSON files.")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of summary.")
    args = parser.parse_args()

    if args.compare:
        cards = []
        for p in args.compare:
            data = json.loads(p.read_text())
            cards.append(Scorecard(
                run_label=data.get("run_label", p.stem),
                policy_name=data.get("policy_name", ""),
                eval_report=data.get("eval", {}),
                dataset_stats=data.get("dataset", {}),
            ))
        print(compare_scorecards(cards))
        return

    if args.training_data is None and args.eval_data is None:
        parser.error("Provide --training-data, --eval-data, or --compare.")

    card = build_scorecard(
        label=args.label,
        training_data_dir=args.training_data,
        eval_path=args.eval_data,
        policy_type=args.policy,
        model_path=args.model,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card.to_json())
        print(f"Scorecard written to {args.output}")

    if args.json:
        print(card.to_json())
    else:
        print(card.summary())


if __name__ == "__main__":
    main()
