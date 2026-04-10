#!/usr/bin/env python3
"""Build a JSONL corpus from solved `aria/search` eval reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from aria.search.proposal_corpus import examples_from_eval_reports, write_jsonl


def _default_output() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "search_proposal_corpus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a solved-search proposal corpus from eval reports",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Eval report path(s). Defaults to results/eval_*.json",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output()),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if args.report:
        paths = [Path(p) for p in args.report]
    else:
        root = Path(__file__).resolve().parents[1] / "results"
        paths = sorted(root.glob("eval_*.json"))

    examples = examples_from_eval_reports(paths)
    out = Path(args.output)
    write_jsonl(out, examples)
    print(f"Wrote {out}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    main()
