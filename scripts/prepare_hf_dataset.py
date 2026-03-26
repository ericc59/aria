#!/usr/bin/env python3
"""Convert ARIA training-data JSONL into a HuggingFace Dataset on disk.

The exporter writes JSONL with schema_version 1 (input-wrapped) or 2 (flat).
This script:
  1. Loads the JSONL.
  2. Filters by task_type(s) and quality tier.
  3. Formats each record into a prompt/response text pair using stable templates.
  4. Optionally splits into train/val by task_id.
  5. Saves as a HF Dataset (Arrow) ready for the trainer.

Usage:
    python scripts/prepare_hf_dataset.py \\
        --input traces/training_export.jsonl \\
        --output datasets/next_focus \\
        --task-types NEXT_FOCUS

    # With val split and quality filter:
    python scripts/prepare_hf_dataset.py \\
        --input traces/training_export.jsonl \\
        --output datasets/next_edit \\
        --task-types NEXT_EDIT \\
        --val-split 0.1 \\
        --next-edit-min-quality medium
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Single source of truth for formatting — import from train_lora.
from scripts.train_lora import (
    EDIT_QUALITY_TIERS,
    SUPPORTED_TASK_TYPES,
    dataset_stats,
    format_records,
    load_dataset_from_jsonl,
    print_dataset_stats,
    split_by_task_id,
)


def load_and_format(
    input_path: str | Path,
    task_types: set[str] | None,
    *,
    min_edit_quality: str = "medium",
) -> list[dict[str, str]]:
    """Compatibility helper used by tests and lightweight callers."""
    filter_set = task_types or set(SUPPORTED_TASK_TYPES)
    raw = load_dataset_from_jsonl(
        input_path,
        filter_set,
        min_edit_quality=min_edit_quality,
    )
    return format_records(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF dataset from ARIA JSONL.")
    parser.add_argument("--input", required=True, help="Path to JSONL file.")
    parser.add_argument("--output", required=True, help="Output directory for HF Dataset.")
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Filter to these task types (e.g. NEXT_FOCUS NEXT_EDIT SKETCH CANDIDATE_RANK).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction to hold out as validation (0-1). Saved as <output>_val/.",
    )
    parser.add_argument(
        "--next-edit-min-quality",
        choices=EDIT_QUALITY_TIERS,
        default="medium",
        help="Minimum NEXT_EDIT quality tier (default: medium).",
    )
    args = parser.parse_args()

    task_types = set(args.task_types) if args.task_types else None

    # Lazy import so the rest works without datasets installed.
    from datasets import Dataset

    input_path = Path(args.input)
    filter_set = task_types or set(SUPPORTED_TASK_TYPES)

    raw = load_dataset_from_jsonl(
        input_path, filter_set,
        min_edit_quality=args.next_edit_min_quality,
    )
    if not raw:
        print("ERROR: no records matched — check --input and --task-types.", file=sys.stderr)
        sys.exit(1)

    stats = dataset_stats(raw)
    print_dataset_stats(stats)

    if args.val_split > 0:
        train_raw, val_raw = split_by_task_id(raw, args.val_split)
        train_rows = format_records(train_raw)
        val_rows = format_records(val_raw)

        ds_train = Dataset.from_list(train_rows)
        ds_train.save_to_disk(args.output)
        print(f"Saved {len(ds_train)} train examples to {args.output}")

        if val_rows:
            val_dir = str(Path(args.output).parent / (Path(args.output).name + "_val"))
            ds_val = Dataset.from_list(val_rows)
            ds_val.save_to_disk(val_dir)
            print(f"Saved {len(ds_val)} val examples to {val_dir}")
    else:
        rows = format_records(raw)
        if not rows:
            print("ERROR: no records formatted successfully.", file=sys.stderr)
            sys.exit(1)
        ds = Dataset.from_list(rows)
        ds.save_to_disk(args.output)
        print(f"Saved {len(ds)} examples to {args.output}")


if __name__ == "__main__":
    main()
