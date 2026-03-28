"""Tests for training data validation, deduplication, and dataset statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aria.training_data import (
    SCHEMA_VERSION,
    TASK_TYPES,
    DatasetStats,
    TrainingExample,
    ValidationResult,
    compute_dataset_stats,
    deduplicate,
    examples_from_program_store,
    examples_from_refinement_traces,
    export_examples,
    load_jsonl,
    validate_example,
    validate_examples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    task_type: str = "NEXT_FOCUS",
    task_id: str = "task_001",
    task_signatures: tuple[str, ...] = ("dims:same",),
    target: dict | None = None,
    **kwargs,
) -> TrainingExample:
    defaults = {
        "schema_version": SCHEMA_VERSION,
        "task_type": task_type,
        "task_id": task_id,
        "task_signatures": task_signatures,
        "current_program": None,
        "round_index": 0,
        "feedback": None,
        "target": target or {},
        "winning_program": None,
    }
    defaults.update(kwargs)
    return TrainingExample(**defaults)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateExample:
    def test_valid_focus(self):
        ex = _make_example(
            task_type="NEXT_FOCUS",
            target={"focus": "marker_geometry"},
        )
        assert validate_example(ex) == []

    def test_valid_sketch(self):
        ex = _make_example(
            task_type="SKETCH",
            target={"program": "let v0: Grid = identity(input)\noutput = v0"},
        )
        assert validate_example(ex) == []

    def test_valid_next_edit_strong(self):
        ex = _make_example(
            task_type="NEXT_EDIT",
            target={
                "before_program": "prog_a",
                "after_program": "prog_b",
                "edit_quality": "strong",
            },
        )
        assert validate_example(ex) == []

    def test_valid_candidate_rank(self):
        ex = _make_example(
            task_type="CANDIDATE_RANK",
            target={"preferred": "prog_a", "rejected": "prog_b"},
        )
        assert validate_example(ex) == []

    def test_missing_target_fields(self):
        ex = _make_example(task_type="NEXT_FOCUS", target={})
        errors = validate_example(ex)
        assert any("target missing" in e for e in errors)

    def test_empty_signatures(self):
        ex = _make_example(task_signatures=(), target={"focus": "generic"})
        errors = validate_example(ex)
        assert any("empty task_signatures" in e for e in errors)

    def test_unknown_task_type(self):
        ex = _make_example(task_type="BOGUS", target={})
        errors = validate_example(ex)
        assert any("unknown task_type" in e for e in errors)

    def test_invalid_edit_quality(self):
        ex = _make_example(
            task_type="NEXT_EDIT",
            target={"after_program": "x", "edit_quality": "bogus"},
        )
        errors = validate_example(ex)
        assert any("invalid edit_quality" in e for e in errors)

    def test_strong_edit_missing_before(self):
        ex = _make_example(
            task_type="NEXT_EDIT",
            target={"after_program": "x", "edit_quality": "strong"},
        )
        errors = validate_example(ex)
        assert any("before_program" in e for e in errors)

    def test_sketch_empty_program(self):
        ex = _make_example(task_type="SKETCH", target={"program": ""})
        errors = validate_example(ex)
        assert any("empty program" in e for e in errors)


class TestValidateExamples:
    def test_mixed_valid_invalid(self):
        examples = [
            _make_example(target={"focus": "generic"}),
            _make_example(task_type="BOGUS", target={}),
            _make_example(target={"focus": "size"}),
        ]
        result = validate_examples(examples)
        assert result.valid == 2
        assert result.invalid == 1
        assert len(result.errors) >= 1

    def test_all_valid(self):
        examples = [
            _make_example(target={"focus": "generic"}),
            _make_example(target={"focus": "size"}),
        ]
        result = validate_examples(examples)
        assert result.valid == 2
        assert result.invalid == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        ex = _make_example(target={"focus": "generic"})
        deduped, n_removed = deduplicate([ex, ex, ex])
        assert len(deduped) == 1
        assert n_removed == 2

    def test_keeps_different_examples(self):
        ex1 = _make_example(task_id="t1", target={"focus": "generic"})
        ex2 = _make_example(task_id="t2", target={"focus": "size"})
        deduped, n_removed = deduplicate([ex1, ex2])
        assert len(deduped) == 2
        assert n_removed == 0

    def test_content_hash_ignores_winning_program(self):
        """Two examples differing only in winning_program are NOT deduped."""
        ex1 = _make_example(target={"focus": "generic"}, winning_program="prog_a")
        ex2 = _make_example(target={"focus": "generic"}, winning_program="prog_b")
        # winning_program is not part of content hash, so these ARE dupes
        deduped, n_removed = deduplicate([ex1, ex2])
        assert len(deduped) == 1

    def test_different_targets_kept(self):
        ex1 = _make_example(target={"focus": "generic"})
        ex2 = _make_example(target={"focus": "size"})
        deduped, _ = deduplicate([ex1, ex2])
        assert len(deduped) == 2

    def test_empty_input(self):
        deduped, n_removed = deduplicate([])
        assert deduped == []
        assert n_removed == 0


# ---------------------------------------------------------------------------
# Dataset statistics tests
# ---------------------------------------------------------------------------

class TestDatasetStats:
    def test_basic_stats(self):
        examples = [
            _make_example(task_type="NEXT_FOCUS", task_id="t1", target={"focus": "generic"}),
            _make_example(task_type="NEXT_FOCUS", task_id="t2", target={"focus": "size"}),
            _make_example(task_type="SKETCH", task_id="t3", target={"program": "prog"}),
        ]
        stats = compute_dataset_stats(examples)
        assert stats.total == 3
        assert stats.by_task_type == {"NEXT_FOCUS": 2, "SKETCH": 1}
        assert stats.unique_task_ids == 3

    def test_edit_quality_distribution(self):
        examples = [
            _make_example(
                task_type="NEXT_EDIT", task_id=f"t{i}",
                target={"after_program": "x", "edit_quality": q},
            )
            for i, q in enumerate(["strong", "strong", "medium", "weak"])
        ]
        stats = compute_dataset_stats(examples)
        assert stats.by_edit_quality["strong"] == 2
        assert stats.by_edit_quality["medium"] == 1
        assert stats.by_edit_quality["weak"] == 1

    def test_to_dict(self):
        stats = DatasetStats(total=5, by_task_type={"SKETCH": 5}, unique_task_ids=3)
        d = stats.to_dict()
        assert d["total"] == 5
        assert d["unique_task_ids"] == 3
        assert d["by_edit_quality"] is None

    def test_no_task_ids(self):
        examples = [_make_example(task_id=None, target={"focus": "generic"})]
        stats = compute_dataset_stats(examples)
        assert stats.unique_task_ids == 0


# ---------------------------------------------------------------------------
# Integration: validation + dedup + export
# ---------------------------------------------------------------------------

class TestValidationDeduplicationPipeline:
    def test_full_pipeline(self, tmp_path: Path):
        """Validate, deduplicate, export, and verify the round-trip."""
        examples = [
            _make_example(task_type="NEXT_FOCUS", task_id="t1",
                          target={"focus": "generic"}),
            _make_example(task_type="NEXT_FOCUS", task_id="t1",
                          target={"focus": "generic"}),  # duplicate
            _make_example(task_type="SKETCH", task_id="t2",
                          target={"program": "let v0: Grid = id(input)\noutput = v0"}),
            _make_example(task_type="BOGUS", task_id="t3", target={}),  # invalid
        ]

        # Validate
        validation = validate_examples(examples)
        assert validation.valid == 3
        assert validation.invalid == 1

        # Filter to valid only
        valid_examples = [
            ex for ex in examples
            if not validate_example(ex)
        ]
        assert len(valid_examples) == 3

        # Deduplicate
        deduped, n_removed = deduplicate(valid_examples)
        assert len(deduped) == 2
        assert n_removed == 1

        # Export
        counts = export_examples(deduped, tmp_path)
        assert counts["NEXT_FOCUS"] == 1
        assert counts["SKETCH"] == 1

        # Verify round-trip
        focus_rows = load_jsonl(tmp_path / "next_focus.jsonl")
        assert len(focus_rows) == 1
        assert focus_rows[0]["target"]["focus"] == "generic"
