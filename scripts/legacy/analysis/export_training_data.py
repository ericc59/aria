#!/usr/bin/env python3
"""CLI: export training JSONL from program_store.json and refinement traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.training_data import (
    export_examples,
    examples_from_program_store,
    examples_from_refinement_traces,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export canonical training JSONL from program store and refinement traces.",
    )
    parser.add_argument(
        "--program-store",
        type=Path,
        default=None,
        help="Path to program_store.json",
    )
    parser.add_argument(
        "--trace-store",
        type=Path,
        default=None,
        help="Path to refinement_traces.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data"),
        help="Directory to write JSONL files (default: training_data/)",
    )
    args = parser.parse_args()

    if args.program_store is None and args.trace_store is None:
        parser.error("At least one of --program-store or --trace-store is required.")

    examples = []

    if args.program_store is not None:
        with open(args.program_store) as f:
            ps_data = json.load(f)
        records = ps_data.get("programs", [])
        examples.extend(examples_from_program_store(records))
        print(f"Program store: {len(records)} programs read")

    if args.trace_store is not None:
        with open(args.trace_store) as f:
            ts_data = json.load(f)
        records = ts_data.get("records", [])
        examples.extend(examples_from_refinement_traces(records))
        print(f"Trace store: {len(records)} trace records read")

    counts = export_examples(examples, args.output_dir)
    total = sum(counts.values())
    print(f"\nExported {total} examples to {args.output_dir}/")
    for task_type, count in sorted(counts.items()):
        print(f"  {task_type}: {count}")


if __name__ == "__main__":
    main()
