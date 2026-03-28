"""Tests for the offline training pipeline — dry-run only, no GPU required."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Sample records — v1 (input-wrapped) schema
# ---------------------------------------------------------------------------

SAMPLE_NEXT_FOCUS_V1 = {
    "schema_version": 1,
    "task_type": "NEXT_FOCUS",
    "input": {
        "task_signatures": ["dims:same", "change:additive", "role:has_marker"],
        "round_index": 1,
        "verifier_feedback": {
            "dominant_error_type": "wrong_output",
            "dimension_mismatch_count": 0,
            "pixel_mismatch_count": 5,
            "execution_error_count": 0,
        },
    },
    "target": {"text": "marker_geometry"},
}

SAMPLE_NEXT_EDIT_V1 = {
    "schema_version": 1,
    "task_type": "NEXT_EDIT",
    "input": {
        "task_signatures": ["dims:same", "color:new_in_output"],
        "current_program": "v0 = find_objects(input)\nv1 = recolor(v0, RED)\n-> v1",
        "verifier_feedback": {
            "dominant_error_type": "wrong_output",
            "pixel_mismatch_count": 3,
        },
        "round_index": 0,
    },
    "target": {"text": "v0 = find_objects(input)\nv1 = apply_color_map(v0, infer_map(input))\n-> v1"},
}

SAMPLE_SKETCH_V1 = {
    "schema_version": 1,
    "task_type": "SKETCH",
    "input": {
        "task_signatures": ["dims:different", "scale:uniform_2x"],
    },
    "target": {"text": "v0 = upscale_grid(input, 2)\n-> v0"},
}


# ---------------------------------------------------------------------------
# Sample records — v2 (flat) schema from exporter
# ---------------------------------------------------------------------------

SAMPLE_NEXT_FOCUS_V2 = {
    "schema_version": 2,
    "task_type": "NEXT_FOCUS",
    "task_id": "abc123",
    "task_signatures": ["dims:same", "change:additive", "role:has_marker"],
    "current_program": None,
    "round_index": 2,
    "feedback": {
        "best_candidate_score": 320.0,
        "best_candidate_error_type": "wrong_output",
        "best_candidate_dims_match": True,
        "best_candidate_pixel_diff_count": 12,
        "best_candidate_wrong_row_count": 2,
        "best_candidate_wrong_col_count": 0,
        "best_candidate_palette_expected_coverage": 1.0,
        "best_candidate_palette_precision": 0.8,
        "best_candidate_preserved_input_ratio": 0.95,
        "best_candidate_changed_cells_ratio": 0.05,
    },
    "target": {"focus": "marker_geometry"},
    "winning_program": "let v0: Grid = overlay(input, input)\noutput = v0",
}

SAMPLE_NEXT_FOCUS_V2_PARTIAL = {
    "schema_version": 2,
    "task_type": "NEXT_FOCUS",
    "task_id": "def456",
    "task_signatures": ["dims:different"],
    "current_program": None,
    "round_index": 1,
    "feedback": {
        "best_candidate_score": 0.3,
        "best_candidate_error_type": "dimension_mismatch",
        "best_candidate_dims_match": False,
    },
    "target": {"focus": "size"},
    "winning_program": None,
}

SAMPLE_NEXT_EDIT_V2_STRONG = {
    "schema_version": 2,
    "task_type": "NEXT_EDIT",
    "task_id": "task_020",
    "task_signatures": ["color:new_in_output", "dims:same"],
    "current_program": "let v0: Grid = recolor(input, 5)\noutput = v0",
    "round_index": 1,
    "feedback": {
        "dominant_error_type": None,
        "best_candidate_score": None,
    },
    "target": {
        "before_program": "let v0: Grid = recolor(input, 5)\noutput = v0",
        "after_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
        "edit_quality": "strong",
        "before_score": 320.0,
        "after_score": 1000000.0,
        "score_delta": 999680.0,
        "before_feedback": {
            "best_candidate_score": 320.0,
            "best_candidate_dims_match": True,
            "best_candidate_pixel_diff_count": 12,
        },
        "after_feedback": {},
    },
    "winning_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
}

SAMPLE_NEXT_EDIT_V2_MEDIUM = {
    "schema_version": 2,
    "task_type": "NEXT_EDIT",
    "task_id": "task_030",
    "task_signatures": ["dims:same"],
    "current_program": "let v0: Grid = identity(input)\noutput = v0",
    "round_index": 1,
    "feedback": {
        "best_candidate_score": 380.0,
    },
    "target": {
        "before_program": "let v0: Grid = identity(input)\noutput = v0",
        "after_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
        "edit_quality": "medium",
        "before_score": 200.0,
        "after_score": 380.0,
        "score_delta": 180.0,
    },
    "winning_program": None,
}

SAMPLE_NEXT_EDIT_V2_WEAK = {
    "schema_version": 2,
    "task_type": "NEXT_EDIT",
    "task_id": "task_040",
    "task_signatures": ["dims:same"],
    "current_program": None,
    "round_index": 1,
    "feedback": {},
    "target": {
        "before_program": None,
        "after_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
        "edit_quality": "weak",
    },
    "winning_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
}

SAMPLE_CANDIDATE_RANK_V2 = {
    "schema_version": 2,
    "task_type": "CANDIDATE_RANK",
    "task_id": "task_020",
    "task_signatures": ["color:new_in_output", "dims:same"],
    "current_program": "let v0: Grid = infer_map(input)\noutput = v0",
    "round_index": 1,
    "feedback": {
        "dominant_error_type": "wrong_output",
        "best_candidate_score": 350.0,
    },
    "target": {
        "preferred": "let v0: Grid = recolor(input, 3)\noutput = v0",
        "rejected": "let v0: Grid = infer_map(input)\noutput = v0",
        "rejected_error_type": "wrong_output",
        "rejected_score": 350.0,
        "preferred_score": 1000000.0,
        "score_delta": 999650.0,
        "rejected_score_reasons": ["dims_match", "pixel_diff=8"],
    },
    "winning_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
}

SAMPLE_SKETCH_V2 = {
    "schema_version": 2,
    "task_type": "SKETCH",
    "task_id": "task_001",
    "task_signatures": ["dims:same", "change:additive"],
    "current_program": None,
    "round_index": None,
    "feedback": None,
    "target": {"program": "let v0: Grid = overlay(input, input)\noutput = v0"},
    "winning_program": "let v0: Grid = overlay(input, input)\noutput = v0",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v1_jsonl_file(tmp_path: Path) -> Path:
    """Write a small JSONL file with v1 records."""
    p = tmp_path / "v1_train.jsonl"
    records = [SAMPLE_NEXT_FOCUS_V1, SAMPLE_NEXT_EDIT_V1, SAMPLE_SKETCH_V1]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


@pytest.fixture
def v2_jsonl_file(tmp_path: Path) -> Path:
    """Write a JSONL file with v2 records."""
    p = tmp_path / "v2_train.jsonl"
    records = [
        SAMPLE_NEXT_FOCUS_V2,
        SAMPLE_NEXT_FOCUS_V2_PARTIAL,
        SAMPLE_NEXT_EDIT_V2_STRONG,
        SAMPLE_NEXT_EDIT_V2_MEDIUM,
        SAMPLE_NEXT_EDIT_V2_WEAK,
        SAMPLE_CANDIDATE_RANK_V2,
        SAMPLE_SKETCH_V2,
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


@pytest.fixture
def mixed_jsonl_file(tmp_path: Path) -> Path:
    """JSONL with both v1 and v2 records."""
    p = tmp_path / "mixed_train.jsonl"
    records = [
        SAMPLE_NEXT_FOCUS_V1,
        SAMPLE_NEXT_FOCUS_V2,
        SAMPLE_NEXT_FOCUS_V2_PARTIAL,
        SAMPLE_NEXT_EDIT_V1,
        SAMPLE_NEXT_EDIT_V2_STRONG,
        SAMPLE_SKETCH_V1,
        SAMPLE_SKETCH_V2,
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


@pytest.fixture
def split_jsonl_file(tmp_path: Path) -> Path:
    """JSONL with enough records and task_ids to test splitting."""
    p = tmp_path / "split_train.jsonl"
    records = []
    for i in range(50):
        records.append({
            "schema_version": 2,
            "task_type": "NEXT_FOCUS",
            "task_id": f"task_{i:04d}",
            "task_signatures": ["dims:same"],
            "round_index": 0,
            "feedback": {"best_candidate_score": i * 0.01},
            "target": {"focus": "generic"},
        })
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


# ---------------------------------------------------------------------------
# Tests: record normalization
# ---------------------------------------------------------------------------


class TestNormalizeRecord:
    def test_v1_record_normalized(self):
        from scripts.train_lora import normalize_record

        rec = normalize_record(SAMPLE_NEXT_FOCUS_V1)
        assert rec["task_type"] == "NEXT_FOCUS"
        assert rec["task_signatures"] == ["dims:same", "change:additive", "role:has_marker"]
        assert rec["round_index"] == 1
        assert rec["schema_version"] == 1

    def test_v2_record_normalized(self):
        from scripts.train_lora import normalize_record

        rec = normalize_record(SAMPLE_NEXT_FOCUS_V2)
        assert rec["task_type"] == "NEXT_FOCUS"
        assert rec["task_signatures"] == ["dims:same", "change:additive", "role:has_marker"]
        assert rec["round_index"] == 2
        assert rec["task_id"] == "abc123"
        assert rec["schema_version"] == 2
        assert isinstance(rec["feedback"], dict)
        assert rec["feedback"]["best_candidate_score"] == 320.0

    def test_v2_flat_edit_normalized(self):
        from scripts.train_lora import normalize_record

        rec = normalize_record(SAMPLE_NEXT_EDIT_V2_STRONG)
        assert rec["task_type"] == "NEXT_EDIT"
        assert rec["target"]["edit_quality"] == "strong"
        assert rec["target"]["before_score"] == 320.0

    def test_v2_candidate_rank_normalized(self):
        from scripts.train_lora import normalize_record

        rec = normalize_record(SAMPLE_CANDIDATE_RANK_V2)
        assert rec["task_type"] == "CANDIDATE_RANK"
        assert rec["target"]["rejected_score"] == 350.0


# ---------------------------------------------------------------------------
# Tests: prompt formatting
# ---------------------------------------------------------------------------


class TestFormatRecord:
    def test_format_next_focus_v1(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V1)
        assert result is not None
        text = result["text"]
        assert "### Task: Choose the next refinement focus." in text
        assert "marker_geometry" in text
        assert "dims:same" in text
        assert "dominant_error_type=wrong_output" in text
        assert "pixel_mismatch_count=5" in text

    def test_format_next_focus_v2_rich(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V2)
        assert result is not None
        text = result["text"]
        assert "best_candidate_score=320.0" in text
        assert "best_candidate_pixel_diff_count=12" in text
        assert "best_candidate_palette_precision=0.8" in text
        assert "marker_geometry" in text

    def test_format_next_focus_v2_partial(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V2_PARTIAL)
        assert result is not None
        text = result["text"]
        assert "best_candidate_score=0.3" in text
        assert "best_candidate_dims_match=False" in text
        assert "pixel_diff_count" not in text

    def test_format_next_edit_v1(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_EDIT_V1)
        assert result is not None
        text = result["text"]
        assert "### Task: Propose the next program edit." in text
        assert "apply_color_map" in text
        assert "find_objects(input)" in text

    def test_format_next_edit_v2_strong(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_EDIT_V2_STRONG)
        assert result is not None
        text = result["text"]
        assert "### Task: Propose the next program edit." in text
        assert "Quality: strong" in text
        assert "Before program:" in text
        assert "recolor(input, 5)" in text
        assert "Before score: 320.0" in text
        assert "Score delta: 999680.0" in text
        # Response should be the after_program
        assert "recolor(input, 3)" in text

    def test_format_next_edit_v2_strong_has_before_feedback(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_EDIT_V2_STRONG)
        text = result["text"]
        assert "Before feedback:" in text
        assert "best_candidate_score=320.0" in text

    def test_format_next_edit_v2_medium(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_EDIT_V2_MEDIUM)
        assert result is not None
        text = result["text"]
        assert "Quality: medium" in text
        assert "Before score: 200.0" in text
        assert "Score delta: 180.0" in text
        assert "recolor(input, 1)" in text

    def test_format_next_edit_v2_weak_uses_fallback(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_EDIT_V2_WEAK)
        assert result is not None
        text = result["text"]
        # Weak edits use the v1-style fallback prompt (no Quality: line)
        assert "Quality:" not in text
        assert "Current program:" in text

    def test_format_sketch_v1(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_SKETCH_V1)
        assert result is not None
        text = result["text"]
        assert "### Task: Generate a full program sketch." in text
        assert "upscale_grid" in text

    def test_format_sketch_v2(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_SKETCH_V2)
        assert result is not None
        text = result["text"]
        assert "### Task: Generate a full program sketch." in text
        assert "overlay(input, input)" in text

    def test_format_candidate_rank_v2(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_CANDIDATE_RANK_V2)
        assert result is not None
        text = result["text"]
        assert "### Task: Rank programs by quality." in text
        assert "Preferred program:" in text
        assert "Rejected program:" in text
        assert "Rejected score: 350.0" in text
        assert "Score delta: 999650.0" in text
        assert "Rejected reasons: dims_match, pixel_diff=8" in text
        assert "Rejected error: wrong_output" in text

    def test_format_candidate_rank_response_is_preferred(self):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_CANDIDATE_RANK_V2)
        text = result["text"]
        assert text.endswith("preferred")

    def test_format_unknown_type_returns_none(self):
        from scripts.train_lora import format_record

        result = format_record({"task_type": "UNKNOWN", "input": {}, "target": {}})
        assert result is None

    def test_format_empty_feedback(self):
        from scripts.train_lora import format_record

        record = {
            "schema_version": 1,
            "task_type": "NEXT_FOCUS",
            "input": {"task_signatures": ["dims:same"], "round_index": 0},
            "target": {"text": "generic"},
        }
        result = format_record(record)
        assert result is not None
        assert "Feedback: none" in result["text"]

    def test_format_extra_unknown_fields_ignored(self):
        from scripts.train_lora import format_record

        record = {
            "schema_version": 2,
            "task_type": "NEXT_FOCUS",
            "task_signatures": ["dims:same"],
            "round_index": 0,
            "feedback": {
                "some_future_field": "should_not_appear",
                "best_candidate_score": 0.5,
            },
            "target": {"focus": "generic"},
        }
        result = format_record(record)
        assert result is not None
        assert "some_future_field" not in result["text"]
        assert "best_candidate_score=0.5" in result["text"]


# ---------------------------------------------------------------------------
# Tests: feedback formatting internals
# ---------------------------------------------------------------------------


class TestFeedbackFormatting:
    def test_rich_fields_take_priority(self):
        from scripts.train_lora import _format_feedback_from_dict

        d = {
            "best_candidate_score": 0.9,
            "verifier_feedback": {"dominant_error_type": "wrong_output"},
        }
        result = _format_feedback_from_dict(d)
        assert "best_candidate_score=0.9" in result
        assert "dominant_error_type" not in result

    def test_legacy_fallback(self):
        from scripts.train_lora import _format_feedback_from_dict

        d = {
            "verifier_feedback": {
                "dominant_error_type": "wrong_output",
                "pixel_mismatch_count": 3,
            },
        }
        result = _format_feedback_from_dict(d)
        assert "dominant_error_type=wrong_output" in result
        assert "pixel_mismatch_count=3" in result

    def test_empty_returns_none_string(self):
        from scripts.train_lora import _format_feedback_from_dict

        assert _format_feedback_from_dict({}) == "none"

    def test_deterministic_ordering(self):
        from scripts.train_lora import _format_feedback_from_dict

        d = {
            "best_candidate_changed_cells_ratio": 0.1,
            "best_candidate_score": 0.5,
        }
        result = _format_feedback_from_dict(d)
        score_pos = result.index("best_candidate_score")
        cells_pos = result.index("best_candidate_changed_cells_ratio")
        assert score_pos < cells_pos


# ---------------------------------------------------------------------------
# Tests: schema version validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_v1_accepted(self):
        from scripts.train_lora import _validate_schema_version

        assert _validate_schema_version({"schema_version": 1}) is True

    def test_v2_accepted(self):
        from scripts.train_lora import _validate_schema_version

        assert _validate_schema_version({"schema_version": 2}) is True

    def test_unknown_version_rejected(self):
        from scripts.train_lora import _validate_schema_version

        assert _validate_schema_version({"schema_version": 99}) is False

    def test_missing_version_accepted(self):
        from scripts.train_lora import _validate_schema_version

        assert _validate_schema_version({}) is True

    def test_unsupported_version_skipped_in_loader(self, tmp_path: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        p = tmp_path / "bad.jsonl"
        records = [
            {"schema_version": 99, "task_type": "NEXT_FOCUS", "target": {}},
            SAMPLE_NEXT_FOCUS_V2,
        ]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        rows = load_dataset_from_jsonl(p, {"NEXT_FOCUS"})
        assert len(rows) == 1
        assert rows[0]["schema_version"] == 2


# ---------------------------------------------------------------------------
# Tests: quality filtering
# ---------------------------------------------------------------------------


class TestQualityFiltering:
    def test_default_excludes_weak(self, v2_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        rows = load_dataset_from_jsonl(v2_jsonl_file, {"NEXT_EDIT"}, min_edit_quality="medium")
        qualities = [r["target"]["edit_quality"] for r in rows]
        assert "weak" not in qualities
        assert "strong" in qualities
        assert "medium" in qualities

    def test_strong_only(self, v2_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        rows = load_dataset_from_jsonl(v2_jsonl_file, {"NEXT_EDIT"}, min_edit_quality="strong")
        qualities = [r["target"]["edit_quality"] for r in rows]
        assert qualities == ["strong"]

    def test_weak_includes_all(self, v2_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        rows = load_dataset_from_jsonl(v2_jsonl_file, {"NEXT_EDIT"}, min_edit_quality="weak")
        assert len(rows) == 3

    def test_v1_records_pass_filter(self, v1_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        rows = load_dataset_from_jsonl(v1_jsonl_file, {"NEXT_EDIT"}, min_edit_quality="strong")
        # v1 records have no edit_quality — they pass any filter
        assert len(rows) == 1

    def test_non_edit_records_unaffected(self, v2_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl

        rows = load_dataset_from_jsonl(v2_jsonl_file, {"NEXT_FOCUS"}, min_edit_quality="strong")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Tests: dataset statistics
# ---------------------------------------------------------------------------


class TestDatasetStats:
    def test_stats_counts(self):
        from scripts.train_lora import dataset_stats

        records = [
            SAMPLE_NEXT_FOCUS_V1,
            SAMPLE_NEXT_FOCUS_V2,
            SAMPLE_NEXT_EDIT_V2_STRONG,
            SAMPLE_NEXT_EDIT_V2_MEDIUM,
            SAMPLE_NEXT_EDIT_V2_WEAK,
            SAMPLE_CANDIDATE_RANK_V2,
            SAMPLE_SKETCH_V2,
        ]
        stats = dataset_stats(records)
        assert stats["total"] == 7
        assert stats["by_task_type"]["NEXT_FOCUS"] == 2
        assert stats["by_task_type"]["NEXT_EDIT"] == 3
        assert stats["by_task_type"]["CANDIDATE_RANK"] == 1
        assert stats["by_task_type"]["SKETCH"] == 1

    def test_stats_schema_versions(self):
        from scripts.train_lora import dataset_stats

        records = [SAMPLE_NEXT_FOCUS_V1, SAMPLE_NEXT_FOCUS_V2]
        stats = dataset_stats(records)
        assert stats["by_schema_version"]["1"] == 1
        assert stats["by_schema_version"]["2"] == 1

    def test_stats_edit_quality(self):
        from scripts.train_lora import dataset_stats

        records = [
            SAMPLE_NEXT_EDIT_V2_STRONG,
            SAMPLE_NEXT_EDIT_V2_MEDIUM,
            SAMPLE_NEXT_EDIT_V2_WEAK,
        ]
        stats = dataset_stats(records)
        assert stats["next_edit_by_quality"]["strong"] == 1
        assert stats["next_edit_by_quality"]["medium"] == 1
        assert stats["next_edit_by_quality"]["weak"] == 1

    def test_stats_no_edit_records(self):
        from scripts.train_lora import dataset_stats

        stats = dataset_stats([SAMPLE_NEXT_FOCUS_V1])
        assert stats["next_edit_by_quality"] is None


# ---------------------------------------------------------------------------
# Tests: dataset loading
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    def test_load_and_format_filters(self, v1_jsonl_file: Path):
        from scripts.prepare_hf_dataset import load_and_format

        rows = load_and_format(v1_jsonl_file, {"NEXT_FOCUS"})
        assert len(rows) == 1
        assert "refinement focus" in rows[0]["text"]

    def test_load_and_format_all(self, v1_jsonl_file: Path):
        from scripts.prepare_hf_dataset import load_and_format

        rows = load_and_format(v1_jsonl_file, None)
        assert len(rows) == 3

    def test_load_mixed_schema(self, mixed_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl, format_records

        raw = load_dataset_from_jsonl(mixed_jsonl_file, {"NEXT_FOCUS"})
        assert len(raw) == 3  # 1 v1 + 2 v2
        rows = format_records(raw)
        assert len(rows) == 3
        texts = [r["text"] for r in rows]
        assert any("dominant_error_type=" in t for t in texts)
        assert any("best_candidate_score=320.0" in t for t in texts)

    def test_load_v2_all_types(self, v2_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl, format_records

        all_types = {"NEXT_FOCUS", "NEXT_EDIT", "SKETCH", "CANDIDATE_RANK"}
        raw = load_dataset_from_jsonl(v2_jsonl_file, all_types, min_edit_quality="weak")
        rows = format_records(raw)
        # Each record should format successfully
        assert len(rows) == 7
        # All four task type headers should be present
        headers = {r["text"].split("\n")[0] for r in rows}
        assert any("refinement focus" in h for h in headers)
        assert any("program edit" in h for h in headers)
        assert any("program sketch" in h for h in headers)
        assert any("Rank programs" in h for h in headers)


# ---------------------------------------------------------------------------
# Tests: deterministic split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_deterministic(self, split_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl, split_by_task_id

        raw = load_dataset_from_jsonl(split_jsonl_file, {"NEXT_FOCUS"})
        train1, val1 = split_by_task_id(raw, 0.2)
        train2, val2 = split_by_task_id(raw, 0.2)
        assert [r["task_id"] for r in train1] == [r["task_id"] for r in train2]
        assert [r["task_id"] for r in val1] == [r["task_id"] for r in val2]

    def test_split_respects_fraction(self, split_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl, split_by_task_id

        raw = load_dataset_from_jsonl(split_jsonl_file, {"NEXT_FOCUS"})
        train, val = split_by_task_id(raw, 0.2)
        total = len(train) + len(val)
        assert total == 50
        assert 3 <= len(val) <= 17

    def test_split_zero_means_all_train(self, split_jsonl_file: Path):
        from scripts.train_lora import load_dataset_from_jsonl, split_by_task_id

        raw = load_dataset_from_jsonl(split_jsonl_file, {"NEXT_FOCUS"})
        train, val = split_by_task_id(raw, 0.0)
        assert len(train) == 50
        assert len(val) == 0

    def test_split_no_task_id_uses_content_hash(self):
        from scripts.train_lora import split_by_task_id

        records = [
            {"task_type": "NEXT_FOCUS", "input": {"task_signatures": [f"sig_{i}"]}, "target": {"text": "x"}}
            for i in range(30)
        ]
        train1, val1 = split_by_task_id(records, 0.2)
        train2, val2 = split_by_task_id(records, 0.2)
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(train1) + len(val1) == 30


# ---------------------------------------------------------------------------
# Tests: config
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_from_yaml(self, tmp_path: Path):
        from scripts.train_lora import TrainConfig

        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(textwrap.dedent("""\
            base_model: "Qwen/Qwen2.5-3B"
            task_types:
              - NEXT_FOCUS
            data_path: "data/train.jsonl"
            lora_r: 8
            epochs: 2
            val_split: 0.1
            next_edit_min_quality: "strong"
        """))
        cfg = TrainConfig.from_yaml(cfg_path)
        assert cfg.base_model == "Qwen/Qwen2.5-3B"
        assert cfg.task_types == ["NEXT_FOCUS"]
        assert cfg.lora_r == 8
        assert cfg.epochs == 2
        assert cfg.val_split == 0.1
        assert cfg.next_edit_min_quality == "strong"
        assert cfg.lora_alpha == 32
        assert cfg.bf16 is True

    def test_unknown_keys_ignored(self, tmp_path: Path):
        from scripts.train_lora import TrainConfig

        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("bogus_key: 42\nepochs: 1\n")
        cfg = TrainConfig.from_yaml(cfg_path)
        assert cfg.epochs == 1

    def test_defaults(self):
        from scripts.train_lora import TrainConfig

        cfg = TrainConfig()
        assert cfg.base_model == "Qwen/Qwen2.5-3B"
        assert cfg.use_4bit is False
        assert cfg.lora_target_modules == ["q_proj", "v_proj"]
        assert cfg.val_split == 0.0
        assert cfg.next_edit_min_quality == "medium"


class TestConfigFiles:
    @pytest.mark.parametrize("name", ["next_focus", "next_edit", "sketch", "candidate_rank"])
    def test_config_parses(self, name: str):
        from scripts.train_lora import TrainConfig

        cfg_path = Path(__file__).parent.parent / "training" / "configs" / f"{name}.yaml"
        if not cfg_path.exists():
            pytest.skip(f"Config not found: {cfg_path}")
        cfg = TrainConfig.from_yaml(cfg_path)
        assert cfg.base_model.startswith("Qwen/")
        assert len(cfg.task_types) >= 1
        assert cfg.lora_r > 0

    def test_next_edit_config_has_quality_setting(self):
        from scripts.train_lora import TrainConfig

        cfg_path = Path(__file__).parent.parent / "training" / "configs" / "next_edit.yaml"
        if not cfg_path.exists():
            pytest.skip("Config not found")
        cfg = TrainConfig.from_yaml(cfg_path)
        assert cfg.next_edit_min_quality in ("strong", "medium", "weak")


# ---------------------------------------------------------------------------
# Tests: dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_passes_v1(self, v1_jsonl_file: Path, capsys):
        from scripts.train_lora import TrainConfig, dry_run

        cfg = TrainConfig(
            data_path=str(v1_jsonl_file),
            data_format="jsonl",
            task_types=["NEXT_FOCUS"],
        )
        assert dry_run(cfg) is True
        captured = capsys.readouterr()
        assert "DRY RUN PASSED" in captured.out

    def test_dry_run_passes_v2(self, v2_jsonl_file: Path, capsys):
        from scripts.train_lora import TrainConfig, dry_run

        cfg = TrainConfig(
            data_path=str(v2_jsonl_file),
            data_format="jsonl",
            task_types=["NEXT_FOCUS"],
        )
        assert dry_run(cfg) is True
        captured = capsys.readouterr()
        assert "DRY RUN PASSED" in captured.out
        assert "by task type" in captured.out
        assert "by schema version" in captured.out

    def test_dry_run_with_val_split(self, mixed_jsonl_file: Path, capsys):
        from scripts.train_lora import TrainConfig, dry_run

        cfg = TrainConfig(
            data_path=str(mixed_jsonl_file),
            data_format="jsonl",
            task_types=["NEXT_FOCUS"],
            val_split=0.3,
        )
        assert dry_run(cfg) is True
        captured = capsys.readouterr()
        assert "train records:" in captured.out
        assert "val records:" in captured.out

    def test_dry_run_fails_missing_file(self, tmp_path: Path):
        from scripts.train_lora import TrainConfig, dry_run

        cfg = TrainConfig(
            data_path=str(tmp_path / "nonexistent.jsonl"),
            data_format="jsonl",
            task_types=["NEXT_FOCUS"],
        )
        assert dry_run(cfg) is False

    def test_dry_run_fails_no_matching_records(self, v1_jsonl_file: Path):
        from scripts.train_lora import TrainConfig, dry_run

        cfg = TrainConfig(
            data_path=str(v1_jsonl_file),
            data_format="jsonl",
            task_types=["NONEXISTENT"],
        )
        assert dry_run(cfg) is False


# ---------------------------------------------------------------------------
# Tests: output metadata
# ---------------------------------------------------------------------------


class TestOutputMeta:
    def test_write_training_meta(self, tmp_path: Path):
        from scripts.train_lora import TrainConfig, write_training_meta, PROMPT_FORMAT_VERSION

        cfg = TrainConfig(output_dir=str(tmp_path / "ckpt"), task_types=["NEXT_FOCUS"])
        write_training_meta(cfg, num_train=100, num_val=10)

        meta_path = tmp_path / "ckpt" / "training_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["prompt_format_version"] == PROMPT_FORMAT_VERSION
        assert meta["base_model"] == "Qwen/Qwen2.5-3B"
        assert meta["task_types"] == ["NEXT_FOCUS"]
        assert meta["num_train_examples"] == 100
        assert meta["num_val_examples"] == 10
        assert meta["lora_r"] == 16
        assert meta["next_edit_min_quality"] == "medium"


# ---------------------------------------------------------------------------
# Tests: formatting smoke test (tokenization without real model)
# ---------------------------------------------------------------------------


class TestTokenizationSmoke:
    @pytest.fixture
    def tiny_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        try:
            tok = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("could not load gpt2 tokenizer (offline?)")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def test_v1_record_tokenizes(self, tiny_tokenizer):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V1)
        assert result is not None
        tokens = tiny_tokenizer(result["text"], truncation=True, max_length=512)
        assert len(tokens["input_ids"]) > 10
        assert len(tokens["input_ids"]) <= 512

    def test_v2_record_tokenizes(self, tiny_tokenizer):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V2)
        assert result is not None
        tokens = tiny_tokenizer(result["text"], truncation=True, max_length=512)
        assert len(tokens["input_ids"]) > 10

    def test_all_v2_types_tokenize(self, tiny_tokenizer):
        from scripts.train_lora import format_record

        for record in [
            SAMPLE_NEXT_FOCUS_V2,
            SAMPLE_NEXT_EDIT_V2_STRONG,
            SAMPLE_NEXT_EDIT_V2_MEDIUM,
            SAMPLE_CANDIDATE_RANK_V2,
            SAMPLE_SKETCH_V2,
        ]:
            result = format_record(record)
            assert result is not None, f"Failed for {record['task_type']}"
            tokens = tiny_tokenizer(result["text"], truncation=True, max_length=2048)
            assert len(tokens["input_ids"]) > 5

    def test_round_trip_decode(self, tiny_tokenizer):
        from scripts.train_lora import format_record

        result = format_record(SAMPLE_NEXT_FOCUS_V2)
        text = result["text"]
        ids = tiny_tokenizer(text)["input_ids"]
        decoded = tiny_tokenizer.decode(ids)
        assert "best_candidate_score" in decoded
        assert "### Focus:" in decoded


# ---------------------------------------------------------------------------
# Tests: prompt format version
# ---------------------------------------------------------------------------


class TestPromptFormatVersion:
    def test_version_is_int(self):
        from scripts.train_lora import PROMPT_FORMAT_VERSION

        assert isinstance(PROMPT_FORMAT_VERSION, int)
        assert PROMPT_FORMAT_VERSION >= 3

    def test_build_prompt_covers_all_types(self):
        from scripts.train_lora import build_prompt, SUPPORTED_TASK_TYPES, normalize_record

        for tt in SUPPORTED_TASK_TYPES:
            rec = normalize_record({
                "schema_version": 2,
                "task_type": tt,
                "task_signatures": ["test"],
                "round_index": 0,
                "feedback": {},
                "target": {"preferred": "a", "rejected": "b"} if tt == "CANDIDATE_RANK" else {},
            })
            result = build_prompt(tt, rec)
            assert result is not None, f"build_prompt returned None for {tt}"
            assert "###" in result


# ---------------------------------------------------------------------------
# Tests: deterministic prompt output
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    def test_same_record_same_prompt(self):
        from scripts.train_lora import format_record

        result1 = format_record(SAMPLE_NEXT_EDIT_V2_STRONG)
        result2 = format_record(SAMPLE_NEXT_EDIT_V2_STRONG)
        assert result1 == result2

    def test_same_rank_prompt(self):
        from scripts.train_lora import format_record

        result1 = format_record(SAMPLE_CANDIDATE_RANK_V2)
        result2 = format_record(SAMPLE_CANDIDATE_RANK_V2)
        assert result1 == result2
