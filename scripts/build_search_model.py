#!/usr/bin/env python3
"""Train a small family model from the solved-search proposal corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria.search.proposal_corpus import SearchProposalExample
from aria.search.proposal_model import SearchFamilyModel, default_model_path


def _load_examples(path: Path) -> list[SearchProposalExample]:
    examples: list[SearchProposalExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(SearchProposalExample(
                schema_version=int(row["schema_version"]),
                task_id=str(row["task_id"]),
                task_signatures=tuple(sorted(str(sig) for sig in row["task_signatures"])),
                family=str(row["family"]),
                description=str(row["description"]),
                source_report=str(row["source_report"]),
            ))
    return examples


def _default_corpus() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "search_proposal_corpus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small search-family model from the solved-search corpus",
    )
    parser.add_argument(
        "--corpus",
        default=str(_default_corpus()),
        help="Input corpus JSONL path",
    )
    parser.add_argument(
        "--output",
        default=str(default_model_path()),
        help="Output model JSON path",
    )
    args = parser.parse_args()

    examples = _load_examples(Path(args.corpus))
    model = SearchFamilyModel.train(examples)
    out = Path(args.output)
    model.save_json(out)
    print(f"Wrote {out}")
    print(
        f"Families: {len(model.family_counts)} | "
        f"Vocabulary: {len(model.vocabulary)} | "
        f"Examples: {model.total_examples}"
    )


if __name__ == "__main__":
    main()
