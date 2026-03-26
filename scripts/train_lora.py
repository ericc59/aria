#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning for ARIA's small Qwen-class models.

Trains a causal-LM adapter on ARIA training data (JSONL or pre-built HF
Dataset) for one of the four task types:
    NEXT_FOCUS      — predict refinement focus from task signatures + feedback
    NEXT_EDIT       — predict program edit from before/after programs + feedback
    SKETCH          — predict full program from task signatures alone
    CANDIDATE_RANK  — rank candidates by quality given scores + error info

Usage:
    # From raw JSONL (formats on-the-fly):
    python scripts/train_lora.py --config training/configs/next_focus.yaml

    # Dry run (validate config + dataset, no GPU):
    python scripts/train_lora.py --config training/configs/next_focus.yaml --dry-run

    # Train with val split:
    python scripts/train_lora.py --config training/configs/next_focus.yaml --val-split 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Prompt format version — bump when the template structure changes.
# Inference code uses this to select the matching prompt formatter.
PROMPT_FORMAT_VERSION = 3

SUPPORTED_SCHEMA_VERSIONS = (1, 2)

SUPPORTED_TASK_TYPES = ("NEXT_FOCUS", "NEXT_EDIT", "SKETCH", "CANDIDATE_RANK")

# NEXT_EDIT quality tiers, ordered from strongest to weakest.
EDIT_QUALITY_TIERS = ("strong", "medium", "weak")

# Fields from the richer diff/progress schema. Listed here explicitly so the
# prompt is stable and reproducible — we never glob unknown keys into prompts.
_RICH_FEEDBACK_FIELDS = (
    "best_candidate_score",
    "best_candidate_error_type",
    "best_candidate_dims_match",
    "best_candidate_pixel_diff_count",
    "best_candidate_wrong_row_count",
    "best_candidate_wrong_col_count",
    "best_candidate_palette_expected_coverage",
    "best_candidate_palette_precision",
    "best_candidate_preserved_input_ratio",
    "best_candidate_changed_cells_ratio",
)

# Legacy feedback fields (schema_version 1).
_LEGACY_FEEDBACK_FIELDS = (
    "dominant_error_type",
    "dimension_mismatch_count",
    "pixel_mismatch_count",
    "execution_error_count",
)


# ---------------------------------------------------------------------------
# Record normalization: v1 (input-wrapped) and v2 (flat) → common shape
# ---------------------------------------------------------------------------

def normalize_record(record: dict) -> dict:
    """Normalize a JSONL record to a common internal shape.

    v1 records wrap fields in ``record["input"]``.
    v2 records (from the exporter) use a flat layout.
    Returns a dict with keys: task_type, task_id, task_signatures,
    current_program, round_index, feedback, target, schema_version,
    winning_program, and any extra v2 fields.
    """
    sv = record.get("schema_version", 1)
    task_type = record.get("task_type", "")
    target = record.get("target", {})

    # v1: fields live under "input"
    if "input" in record:
        inp = record["input"]
        return {
            "schema_version": sv,
            "task_type": task_type,
            "task_id": record.get("task_id") or inp.get("task_id"),
            "task_signatures": inp.get("task_signatures", []),
            "current_program": inp.get("current_program"),
            "round_index": inp.get("round_index", 0),
            "feedback": inp,  # v1: the input dict IS the feedback source
            "target": target,
            "winning_program": record.get("winning_program"),
        }

    # v2: flat layout from exporter
    return {
        "schema_version": sv,
        "task_type": task_type,
        "task_id": record.get("task_id"),
        "task_signatures": record.get("task_signatures", []),
        "current_program": record.get("current_program"),
        "round_index": record.get("round_index", 0),
        "feedback": record.get("feedback") or {},
        "target": target,
        "winning_program": record.get("winning_program"),
    }


# ---------------------------------------------------------------------------
# Prompt building — deterministic, schema-tolerant
# ---------------------------------------------------------------------------

def _format_feedback_from_dict(d: dict) -> str:
    """Build a compact, deterministic feedback string from a dict.

    Checks rich diff fields first; falls back to legacy fields.
    Unknown fields are ignored.
    """
    parts: list[str] = []
    for key in _RICH_FEEDBACK_FIELDS:
        val = d.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    if parts:
        return " | ".join(parts)

    # Fallback: legacy fields at top level of the dict.
    for key in _LEGACY_FEEDBACK_FIELDS:
        val = d.get(key)
        if val is not None:
            parts.append(f"{key}={val}")

    # v1 compat: check nested verifier_feedback
    vf = d.get("verifier_feedback")
    if not parts and isinstance(vf, dict) and vf:
        for key in _LEGACY_FEEDBACK_FIELDS:
            val = vf.get(key)
            if val is not None:
                parts.append(f"{key}={val}")
        extra_keys = sorted(set(vf) - set(_LEGACY_FEEDBACK_FIELDS))
        for key in extra_keys:
            parts.append(f"{key}={vf[key]}")

    return " | ".join(parts) if parts else "none"


def _format_diff_progress(d: dict) -> str:
    """Format a diff-progress sub-dict (before_feedback/after_feedback)."""
    parts: list[str] = []
    for key in _RICH_FEEDBACK_FIELDS:
        val = d.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return " | ".join(parts) if parts else "none"


def _format_feedback(rec: dict) -> str:
    """Format feedback from a normalized record or a raw feedback dict.

    Callers inside this module pass normalized records with a ``feedback`` key.
    Some tests and utilities still pass the feedback payload directly. Support
    both shapes so formatting remains backward compatible.
    """
    fb = rec.get("feedback")
    if isinstance(fb, dict) and fb:
        return _format_feedback_from_dict(fb)
    if rec:
        return _format_feedback_from_dict(rec)
    return "none"


def build_prompt(task_type: str, rec: dict) -> str | None:
    """Build the prompt string for a normalized record. Returns None if unknown."""
    if task_type == "NEXT_FOCUS":
        return _build_next_focus_prompt(rec)
    if task_type == "NEXT_EDIT":
        return _build_next_edit_prompt(rec)
    if task_type == "SKETCH":
        return _build_sketch_prompt(rec)
    if task_type == "CANDIDATE_RANK":
        return _build_candidate_rank_prompt(rec)
    return None


def _build_next_focus_prompt(rec: dict) -> str:
    sigs = ", ".join(rec.get("task_signatures", []))
    feedback = _format_feedback(rec)
    return (
        f"### Task: Choose the next refinement focus.\n"
        f"Signatures: {sigs}\n"
        f"Round: {rec.get('round_index', 0)}\n"
        f"Feedback: {feedback}\n"
        f"### Focus:"
    )


def _build_next_edit_prompt(rec: dict) -> str:
    sigs = ", ".join(rec.get("task_signatures", []))
    target = rec.get("target", {})
    edit_quality = target.get("edit_quality")

    # v2 strong/medium: use before/after structure
    if edit_quality in ("strong", "medium") and target.get("before_program") is not None:
        before_prog = target["before_program"]
        before_score = target.get("before_score")
        after_score = target.get("after_score")
        score_delta = target.get("score_delta")

        lines = [
            f"### Task: Propose the next program edit.",
            f"Signatures: {sigs}",
            f"Quality: {edit_quality}",
            f"Before program:\n{before_prog}",
        ]
        if before_score is not None:
            lines.append(f"Before score: {before_score}")

        # Before/after feedback from diff progress
        bf = target.get("before_feedback")
        if isinstance(bf, dict) and bf:
            lines.append(f"Before feedback: {_format_diff_progress(bf)}")
        af = target.get("after_feedback")
        if isinstance(af, dict) and af:
            lines.append(f"After feedback: {_format_diff_progress(af)}")

        if score_delta is not None:
            lines.append(f"Score delta: {score_delta}")
        lines.append(f"Round: {rec.get('round_index', 0)}")
        lines.append(f"### Edit:")
        return "\n".join(lines)

    # v1 or weak: use current_program + feedback
    feedback = _format_feedback(rec)
    program = rec.get("current_program") or ""
    return (
        f"### Task: Propose the next program edit.\n"
        f"Signatures: {sigs}\n"
        f"Current program:\n{program}\n"
        f"Feedback: {feedback}\n"
        f"Round: {rec.get('round_index', 0)}\n"
        f"### Edit:"
    )


def _build_sketch_prompt(rec: dict) -> str:
    sigs = ", ".join(rec.get("task_signatures", []))
    return (
        f"### Task: Generate a full program sketch.\n"
        f"Signatures: {sigs}\n"
        f"### Program:"
    )


def _build_candidate_rank_prompt(rec: dict) -> str:
    sigs = ", ".join(rec.get("task_signatures", []))
    target = rec.get("target", {})
    feedback = _format_feedback(rec)

    preferred = target.get("preferred", "")
    rejected = target.get("rejected", "")

    lines = [
        f"### Task: Rank programs by quality.",
        f"Signatures: {sigs}",
        f"Feedback: {feedback}",
        f"Preferred program:\n{preferred}",
        f"Rejected program:\n{rejected}",
    ]
    err = target.get("rejected_error_type")
    if err is not None:
        lines.append(f"Rejected error: {err}")
    rscore = target.get("rejected_score")
    if rscore is not None:
        lines.append(f"Rejected score: {rscore}")
    sd = target.get("score_delta")
    if sd is not None:
        lines.append(f"Score delta: {sd}")
    reasons = target.get("rejected_score_reasons")
    if reasons:
        lines.append(f"Rejected reasons: {', '.join(str(r) for r in reasons)}")
    lines.append(f"Round: {rec.get('round_index', 0)}")
    lines.append(f"### Rank:")
    return "\n".join(lines)


def _build_response(task_type: str, target: dict) -> str:
    """Build the response string from the target dict."""
    # Prefer explicit text field.
    if "text" in target:
        return str(target["text"])

    if task_type == "NEXT_FOCUS":
        return target.get("focus", json.dumps(target, separators=(",", ":")))

    if task_type == "NEXT_EDIT":
        prog = target.get("after_program") or target.get("program")
        if prog:
            return prog
        return json.dumps(target, separators=(",", ":"))

    if task_type == "SKETCH":
        prog = target.get("program")
        if prog:
            return prog
        return json.dumps(target, separators=(",", ":"))

    if task_type == "CANDIDATE_RANK":
        return "preferred"

    return json.dumps(target, separators=(",", ":"))


def format_record(record: dict) -> dict[str, str] | None:
    """Format a single JSONL record into {text: prompt+response}."""
    rec = normalize_record(record)
    task_type = rec["task_type"]
    prompt = build_prompt(task_type, rec)
    if prompt is None:
        return None
    response = _build_response(task_type, rec["target"])
    return {"text": f"{prompt}\n{response}"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-3B"
    use_4bit: bool = False

    # Data
    data_path: str = ""           # JSONL file or HF Dataset dir
    data_format: str = "jsonl"    # "jsonl" or "hf_dataset"
    task_types: list[str] = field(default_factory=lambda: ["NEXT_FOCUS"])
    max_seq_len: int = 1024
    val_split: float = 0.0        # 0 = train-only, >0 = fraction held out

    # Quality filtering
    next_edit_min_quality: str = "medium"  # "strong", "medium", or "weak"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training
    output_dir: str = "checkpoints/next_focus"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _passes_quality_filter(record: dict, min_quality: str) -> bool:
    """Check if a NEXT_EDIT record meets the minimum quality tier."""
    if record.get("task_type") != "NEXT_EDIT":
        return True

    target = record.get("target", {})
    quality = target.get("edit_quality")

    # v1 records have no edit_quality — treat as meeting any threshold.
    if quality is None:
        return True

    allowed_idx = EDIT_QUALITY_TIERS.index(min_quality) if min_quality in EDIT_QUALITY_TIERS else len(EDIT_QUALITY_TIERS)
    try:
        quality_idx = EDIT_QUALITY_TIERS.index(quality)
    except ValueError:
        return False  # Unknown quality tier — reject.
    return quality_idx <= allowed_idx


def _validate_schema_version(record: dict) -> bool:
    """Return True if the record's schema_version is supported or absent."""
    sv = record.get("schema_version")
    if sv is None:
        return True  # Permissive for records without version.
    return sv in SUPPORTED_SCHEMA_VERSIONS


def load_dataset_from_jsonl(
    path: Path,
    task_types: set[str],
    *,
    min_edit_quality: str = "medium",
) -> list[dict]:
    """Load JSONL, filter by task type and quality, return raw records."""
    rows: list[dict] = []
    skipped_schema = 0
    skipped_quality = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("task_type") not in task_types:
                continue
            if not _validate_schema_version(record):
                skipped_schema += 1
                continue
            if not _passes_quality_filter(record, min_edit_quality):
                skipped_quality += 1
                continue
            rows.append(record)
    if skipped_schema:
        print(f"  skipped {skipped_schema} records with unsupported schema_version", file=sys.stderr)
    if skipped_quality:
        print(f"  skipped {skipped_quality} NEXT_EDIT records below quality={min_edit_quality}", file=sys.stderr)
    return rows


def format_records(records: list[dict]) -> list[dict[str, str]]:
    """Format raw records into {text} dicts, dropping any that fail."""
    out: list[dict[str, str]] = []
    for r in records:
        fmt = format_record(r)
        if fmt is not None:
            out.append(fmt)
    return out


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def dataset_stats(records: list[dict]) -> dict[str, Any]:
    """Compute lightweight statistics for sanity-checking before training."""
    by_type: Counter[str] = Counter()
    by_version: Counter[int] = Counter()
    by_edit_quality: Counter[str] = Counter()

    for r in records:
        by_type[r.get("task_type", "UNKNOWN")] += 1
        by_version[r.get("schema_version", 0)] += 1
        if r.get("task_type") == "NEXT_EDIT":
            q = r.get("target", {}).get("edit_quality", "unset")
            by_edit_quality[q] += 1

    return {
        "total": len(records),
        "by_task_type": dict(sorted(by_type.items())),
        "by_schema_version": {str(k): v for k, v in sorted(by_version.items())},
        "next_edit_by_quality": dict(sorted(by_edit_quality.items())) if by_edit_quality else None,
    }


def print_dataset_stats(stats: dict[str, Any]) -> None:
    """Print dataset statistics to stdout."""
    print(f"  total records: {stats['total']}")
    print(f"  by task type: {stats['by_task_type']}")
    print(f"  by schema version: {stats['by_schema_version']}")
    if stats.get("next_edit_by_quality"):
        print(f"  NEXT_EDIT by quality: {stats['next_edit_by_quality']}")


# ---------------------------------------------------------------------------
# Deterministic train/val split
# ---------------------------------------------------------------------------

def split_by_task_id(
    records: list[dict], val_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Deterministic split: hash task_id to assign train vs val.

    If task_id is missing, falls back to hashing the full record.
    Records with the same task_id always land in the same split.
    """
    train: list[dict] = []
    val: list[dict] = []
    for r in records:
        key = r.get("task_id") or r.get("input", {}).get("task_id")
        if key is None:
            key = json.dumps(r, sort_keys=True, separators=(",", ":"))
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10000
        if h < int(val_fraction * 10000):
            val.append(r)
        else:
            train.append(r)
    return train, val


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------

def write_training_meta(cfg: TrainConfig, num_train: int, num_val: int) -> None:
    """Write training_meta.json alongside the checkpoint."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "base_model": cfg.base_model,
        "task_types": cfg.task_types,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_target_modules": cfg.lora_target_modules,
        "max_seq_len": cfg.max_seq_len,
        "use_4bit": cfg.use_4bit,
        "num_train_examples": num_train,
        "num_val_examples": num_val,
        "epochs": cfg.epochs,
        "next_edit_min_quality": cfg.next_edit_min_quality,
    }
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def dry_run(cfg: TrainConfig) -> bool:
    """Validate config and dataset without loading model or GPU."""
    print("=== DRY RUN ===")
    print(f"  base_model:           {cfg.base_model}")
    print(f"  use_4bit:             {cfg.use_4bit}")
    print(f"  task_types:           {cfg.task_types}")
    print(f"  data_path:            {cfg.data_path}")
    print(f"  data_format:          {cfg.data_format}")
    print(f"  output_dir:           {cfg.output_dir}")
    print(f"  lora_r:               {cfg.lora_r}")
    print(f"  lora_alpha:           {cfg.lora_alpha}")
    print(f"  epochs:               {cfg.epochs}")
    print(f"  batch_size:           {cfg.batch_size}")
    print(f"  max_seq_len:          {cfg.max_seq_len}")
    print(f"  val_split:            {cfg.val_split}")
    print(f"  next_edit_min_quality: {cfg.next_edit_min_quality}")

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        print(f"  ERROR: data_path does not exist: {data_path}")
        return False

    task_types_set = set(cfg.task_types)
    if cfg.data_format == "jsonl":
        raw = load_dataset_from_jsonl(
            data_path, task_types_set,
            min_edit_quality=cfg.next_edit_min_quality,
        )

        stats = dataset_stats(raw)
        print_dataset_stats(stats)

        if cfg.val_split > 0:
            train_raw, val_raw = split_by_task_id(raw, cfg.val_split)
            train_rows = format_records(train_raw)
            val_rows = format_records(val_raw)
            print(f"  train records: {len(train_rows)}")
            print(f"  val records:   {len(val_rows)}")
            rows = train_rows
        else:
            rows = format_records(raw)
            print(f"  formatted records: {len(rows)}")
        if not rows:
            print("  ERROR: no records matched task_types filter.")
            return False
        print(f"  sample text ({len(rows[0]['text'])} chars):")
        print("    " + rows[0]["text"][:300].replace("\n", "\n    "))
    elif cfg.data_format == "hf_dataset":
        from datasets import load_from_disk
        ds = load_from_disk(str(data_path))
        print(f"  records loaded: {len(ds)}")
        if len(ds) == 0:
            print("  ERROR: dataset is empty.")
            return False
        print(f"  sample text ({len(ds[0]['text'])} chars):")
        print("    " + ds[0]["text"][:300].replace("\n", "\n    "))
    else:
        print(f"  ERROR: unknown data_format: {cfg.data_format}")
        return False

    for tt in cfg.task_types:
        if tt not in SUPPORTED_TASK_TYPES:
            print(f"  ERROR: unsupported task_type={tt}")
            return False

    print(f"  prompt_format_version: {PROMPT_FORMAT_VERSION}")
    print("=== DRY RUN PASSED ===")
    return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    import torch
    from datasets import Dataset, load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # 1. Load dataset ---------------------------------------------------
    task_types_set = set(cfg.task_types)
    eval_dataset = None

    if cfg.data_format == "jsonl":
        raw = load_dataset_from_jsonl(
            Path(cfg.data_path), task_types_set,
            min_edit_quality=cfg.next_edit_min_quality,
        )

        stats = dataset_stats(raw)
        print_dataset_stats(stats)

        if cfg.val_split > 0:
            train_raw, val_raw = split_by_task_id(raw, cfg.val_split)
            train_rows = format_records(train_raw)
            val_rows = format_records(val_raw)
            num_train, num_val = len(train_rows), len(val_rows)
        else:
            train_rows = format_records(raw)
            val_rows = []
            num_train, num_val = len(train_rows), 0
        dataset = Dataset.from_list(train_rows)
        if val_rows:
            eval_dataset = Dataset.from_list(val_rows)
    else:
        dataset = load_from_disk(cfg.data_path)
        num_train, num_val = len(dataset), 0

    print(f"Training on {num_train} examples, validating on {num_val}.")

    # 2. Load tokenizer -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize -------------------------------------------------------
    def tokenize(example: dict) -> dict:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # 4. Load model -----------------------------------------------------
    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True,
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 5. LoRA adapter ---------------------------------------------------
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Training args --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.save_steps if eval_dataset is not None else None,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # 7. Train ----------------------------------------------------------
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    write_training_meta(cfg, num_train, num_val)
    print(f"Adapter + meta saved to {cfg.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for ARIA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no training.")
    parser.add_argument("--model", default=None, help="Override base_model from config.")
    parser.add_argument("--data", default=None, help="Override data_path from config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--use-4bit", action="store_true", default=None, help="Force 4-bit quantization.")
    parser.add_argument("--val-split", type=float, default=None, help="Val fraction (0-1). Overrides config.")
    parser.add_argument(
        "--next-edit-min-quality",
        choices=EDIT_QUALITY_TIERS,
        default=None,
        help="Minimum NEXT_EDIT quality tier (default: from config or 'medium').",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.model:
        cfg.base_model = args.model
    if args.data:
        cfg.data_path = args.data
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.use_4bit:
        cfg.use_4bit = True
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.next_edit_min_quality is not None:
        cfg.next_edit_min_quality = args.next_edit_min_quality

    if args.dry_run:
        ok = dry_run(cfg)
        sys.exit(0 if ok else 1)

    if not cfg.data_path:
        print("ERROR: data_path is required (set in config or --data).", file=sys.stderr)
        sys.exit(1)

    train(cfg)


if __name__ == "__main__":
    main()
