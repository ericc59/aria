# ARIA Local Fine-Tuning Pipeline

Offline LoRA/QLoRA training for Qwen-class models (3B-4B) on ARIA refinement traces.

## Quick Start

```bash
pip install -r requirements-training.txt

# Dry run (validates config + data, no GPU needed):
python scripts/train_lora.py --config training/configs/next_focus.yaml \
    --data traces/training_export.jsonl --dry-run

# Train with 10% val holdout:
python scripts/train_lora.py --config training/configs/next_focus.yaml \
    --data traces/training_export.jsonl

# QLoRA on smaller GPU (8GB+):
python scripts/train_lora.py --config training/configs/next_focus.yaml \
    --data traces/training_export.jsonl --use-4bit
```

## Task Types

Train these in order:

### 1. NEXT_FOCUS (train first)

Predicts which refinement focus to use next (marker_geometry, size, color_map,
generic) given task signatures and verifier feedback. Replaces the hand-built
`HeuristicRefinementPolicy` in `aria/refinement.py`.

- **Input**: task signatures + feedback (rich diff-scored fields when available,
  legacy verifier_feedback dict as fallback)
- **Output**: focus label
- **Why first**: smallest search space, highest leverage.

### 2. NEXT_EDIT (train second)

Predicts the next program edit given a before program, scores, and diff feedback.

Schema v2 provides three quality tiers:

| Tier | Signal | Description |
|------|--------|-------------|
| **strong** | best failing candidate -> exact winner | Prior round's best failing candidate is the before_program; current round's winner is the after_program |
| **medium** | improved best candidate | Best candidate improved across rounds but didn't solve |
| **weak** | coarse round transition | No before_program anchor; falls back to legacy prompt |

Default config excludes weak examples (`next_edit_min_quality: "medium"`).

- **Input**: before program + scores + diff feedback
- **Output**: after program
- **Why second**: depends on having a good focus selector; longer context.

### 3. SKETCH (train third)

Generates a full program from task signatures alone.

- **Input**: task signatures only
- **Output**: complete DSL program
- **Why third**: largest output space, benefits from more training data.

### 4. CANDIDATE_RANK (optional)

Ranks candidate programs by quality. Uses score-aware hard negatives with
error-type diversity.

- **Input**: preferred/rejected programs + scores + error info
- **Output**: preference signal
- **When**: useful for reward modeling or preference tuning.

## JSONL Schema

The pipeline accepts both v1 (input-wrapped) and v2 (flat) schemas.
Unknown schema versions are rejected with a warning.

### v1 (input-wrapped)

```json
{
    "schema_version": 1,
    "task_type": "NEXT_FOCUS",
    "input": {
        "task_signatures": ["dims:same", "change:additive"],
        "round_index": 1,
        "verifier_feedback": {
            "dominant_error_type": "wrong_output",
            "pixel_mismatch_count": 5
        }
    },
    "target": {"text": "marker_geometry"}
}
```

### v2 (flat, from exporter)

```json
{
    "schema_version": 2,
    "task_type": "NEXT_EDIT",
    "task_id": "task_020",
    "task_signatures": ["color:new_in_output", "dims:same"],
    "current_program": "let v0: Grid = recolor(input, 5)\noutput = v0",
    "round_index": 1,
    "feedback": {
        "best_candidate_score": 320.0,
        "best_candidate_dims_match": true,
        "best_candidate_pixel_diff_count": 12
    },
    "target": {
        "before_program": "let v0: Grid = recolor(input, 5)\noutput = v0",
        "after_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
        "edit_quality": "strong",
        "before_score": 320.0,
        "after_score": 1000000.0,
        "score_delta": 999680.0,
        "before_feedback": {"best_candidate_score": 320.0},
        "after_feedback": {}
    },
    "winning_program": "let v0: Grid = recolor(input, 3)\noutput = v0"
}
```

## Quality Filtering

NEXT_EDIT examples can be filtered by quality tier:

```yaml
# In config:
next_edit_min_quality: "medium"   # strong, medium, or weak
```

```bash
# CLI override:
--next-edit-min-quality strong
```

v1 records (no `edit_quality` field) pass any quality filter.

## Dataset Statistics

Dry-run and training both report statistics before processing:

```
  total records: 500
  by task type: {'CANDIDATE_RANK': 120, 'NEXT_EDIT': 80, 'NEXT_FOCUS': 200, 'SKETCH': 100}
  by schema version: {'1': 50, '2': 450}
  NEXT_EDIT by quality: {'medium': 30, 'strong': 50}
```

## Train/Val Split

Deterministic and task-level (records with the same `task_id` always land in
the same split). If `task_id` is missing, a content hash is used.

```yaml
val_split: 0.1   # in YAML config
```

```bash
--val-split 0.1   # CLI override
```

## Data Preparation

Optional pre-processing to HF Dataset format (faster repeated loads):

```bash
python scripts/prepare_hf_dataset.py \
    --input traces/training_export.jsonl \
    --output datasets/next_edit \
    --task-types NEXT_EDIT \
    --val-split 0.1 \
    --next-edit-min-quality medium
```

Then set `data_format: hf_dataset` and `data_path: datasets/next_edit` in config.

## Output Artifacts

After training, `<output_dir>/` contains:

| File | Purpose |
|------|---------|
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | PEFT adapter config |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer files |
| `training_meta.json` | Training metadata (see below) |

`training_meta.json` records everything inference code needs to load the adapter:

```json
{
    "prompt_format_version": 3,
    "base_model": "Qwen/Qwen2.5-3B",
    "task_types": ["NEXT_EDIT"],
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_seq_len": 1024,
    "use_4bit": false,
    "num_train_examples": 450,
    "num_val_examples": 50,
    "epochs": 3,
    "next_edit_min_quality": "medium"
}
```

Inference code should check `prompt_format_version` to select the matching
prompt formatter.

## Hardware

| Task | VRAM (fp16) | VRAM (QLoRA 4-bit) |
|------|-------------|---------------------|
| NEXT_FOCUS | ~10 GB | ~6 GB |
| NEXT_EDIT | ~12 GB | ~7 GB |
| SKETCH | ~14 GB | ~8 GB |
| CANDIDATE_RANK | ~12 GB | ~7 GB |

## Config Reference

```yaml
base_model: "Qwen/Qwen2.5-3B"
use_4bit: false
data_path: "traces/export.jsonl"
data_format: "jsonl"              # "jsonl" or "hf_dataset"
task_types: ["NEXT_EDIT"]
max_seq_len: 1024
val_split: 0.1
next_edit_min_quality: "medium"   # "strong", "medium", or "weak"
lora_r: 16
lora_alpha: 32
lora_target_modules: ["q_proj", "v_proj"]
output_dir: "checkpoints/next_edit"
epochs: 3
batch_size: 4
learning_rate: 2e-4
bf16: true
```

CLI overrides: `--model`, `--data`, `--output-dir`, `--use-4bit`, `--val-split`,
`--next-edit-min-quality`.

## Tests

```bash
pytest tests/test_training_pipeline.py -v
```

All tests run without GPU. They validate:
- Record normalization (v1 input-wrapped and v2 flat)
- Prompt formatting for all four task types with both schemas
- NEXT_EDIT strong/medium/weak prompt structure
- CANDIDATE_RANK score-aware formatting
- Schema version validation (rejects unknown versions)
- Quality-tier filtering for NEXT_EDIT
- Dataset statistics reporting
- Backward compatibility (v1 records still format correctly)
- Deterministic train/val split behavior
- Config loading for all task types
- Output metadata writing
- Tokenization smoke tests (requires `transformers` + `gpt2` tokenizer)
