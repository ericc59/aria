#!/usr/bin/env python3
"""Offline evaluation script for local refinement policies.

Usage:
    python scripts/eval_local_policy.py data/eval.jsonl
    python scripts/eval_local_policy.py data/eval.jsonl --policy heuristic
    python scripts/eval_local_policy.py data/eval.jsonl --policy local_lm --model path/to/model
    python scripts/eval_local_policy.py data/eval.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.local_policy import HeuristicBaselinePolicy, LocalCausalLMPolicy
from aria.policy_eval import evaluate, load_eval_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local refinement policy offline.")
    parser.add_argument("eval_path", type=Path, help="Path to JSONL eval data.")
    parser.add_argument(
        "--policy",
        choices=["heuristic", "local_lm"],
        default="heuristic",
        help="Policy implementation to evaluate (default: heuristic).",
    )
    parser.add_argument("--model", type=str, default="", help="Model path for local_lm policy.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for local_lm policy.")
    parser.add_argument("--json", action="store_true", help="Output report as JSON.")
    args = parser.parse_args()

    if not args.eval_path.exists():
        print(f"error: {args.eval_path} not found", file=sys.stderr)
        sys.exit(1)

    examples = load_eval_examples(args.eval_path)
    if not examples:
        print("error: no examples loaded", file=sys.stderr)
        sys.exit(1)

    if args.policy == "heuristic":
        policy = HeuristicBaselinePolicy()
    else:
        policy = LocalCausalLMPolicy(
            model_name_or_path=args.model,
            device=args.device,
            dry_run=not args.model,
        )

    report = evaluate(policy, examples)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
