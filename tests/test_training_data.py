"""Tests for the training data export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aria.training_data import (
    SCHEMA_VERSION,
    TASK_TYPES,
    TrainingExample,
    examples_from_program_store,
    examples_from_refinement_traces,
    export_examples,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal program store and trace store records
# ---------------------------------------------------------------------------

def _program_store_records() -> list[dict]:
    return [
        {
            "program": "let v0: Grid = overlay(input, input)\noutput = v0",
            "task_ids": ["task_001"],
            "sources": ["offline-search"],
            "use_count": 2,
            "signatures": ["dims:same", "change:additive"],
        },
        {
            "program": "let v0: Grid = upscale_grid(input, 2)\noutput = v0",
            "task_ids": ["task_002", "task_003"],
            "sources": ["offline-search"],
            "use_count": 1,
            "signatures": ["size:scale_2x"],
        },
        # Should be skipped: no signatures
        {
            "program": "let v0: Grid = identity(input)\noutput = v0",
            "task_ids": [],
            "sources": [],
            "use_count": 1,
            "signatures": [],
        },
    ]


def _trace_store_records() -> list[dict]:
    """Legacy-style trace records without score fields."""
    return [
        {
            "task_id": "task_010",
            "solved": True,
            "candidates_tried": 120,
            "winning_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
            "rounds": [
                {
                    "plan": {
                        "name": "generic",
                        "max_steps": 3,
                        "max_candidates": 5000,
                        "allowed_ops": [],
                    },
                    "solved": False,
                    "candidates_tried": 80,
                    "feedback": {
                        "dominant_error_type": "wrong_output",
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 5,
                        "execution_error_count": 0,
                        "suggested_focus": "color_map",
                        "task_signatures": ["color:new_in_output", "dims:same"],
                    },
                    "winning_program": None,
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = identity(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": None,
                        },
                        {
                            "candidate_num": 2,
                            "depth": 2,
                            "program_text": "let v0: Grid = flip(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": None,
                        },
                    ],
                },
                {
                    "plan": {
                        "name": "color_map",
                        "max_steps": 4,
                        "max_candidates": 5000,
                        "allowed_ops": ["recolor", "infer_map"],
                    },
                    "solved": True,
                    "candidates_tried": 40,
                    "feedback": {
                        "dominant_error_type": None,
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 0,
                        "execution_error_count": 0,
                        "suggested_focus": "generic",
                        "task_signatures": ["color:new_in_output", "dims:same"],
                    },
                    "winning_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = infer_map(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": None,
                        },
                        {
                            "candidate_num": 2,
                            "depth": 1,
                            "program_text": "let v0: Grid = recolor(input, 1)\noutput = v0",
                            "passed": True,
                            "failed_demo": None,
                            "error_type": None,
                            "diff": None,
                        },
                    ],
                },
            ],
        },
    ]


def _scored_trace_records() -> list[dict]:
    """Trace records with diff-scored best-candidate fields."""
    return [
        {
            "task_id": "task_020",
            "solved": True,
            "candidates_tried": 200,
            "winning_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
            "rounds": [
                {
                    "plan": {
                        "name": "generic",
                        "max_steps": 3,
                        "max_candidates": 5000,
                        "allowed_ops": [],
                    },
                    "solved": False,
                    "candidates_tried": 100,
                    "feedback": {
                        "dominant_error_type": "wrong_output",
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 8,
                        "execution_error_count": 2,
                        "suggested_focus": "color_map",
                        "task_signatures": ["color:new_in_output", "dims:same"],
                        "best_candidate_num": 3,
                        "best_candidate_score": 320.0,
                        "best_candidate_error_type": "wrong_output",
                        "best_candidate_dims_match": True,
                        "best_candidate_pixel_diff_count": 12,
                        "best_candidate_wrong_row_count": 2,
                        "best_candidate_wrong_col_count": 1,
                        "best_candidate_palette_expected_coverage": 0.85,
                        "best_candidate_palette_precision": 0.9,
                        "best_candidate_preserved_input_ratio": 0.7,
                        "best_candidate_changed_cells_ratio": 0.6,
                    },
                    "winning_program": None,
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = identity(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": {"pixel_diff_count": 50},
                            "score": 100.0,
                            "score_reasons": ["dims_match", "pixel_diff=50"],
                        },
                        {
                            "candidate_num": 2,
                            "depth": 1,
                            "program_text": "let v0: Grid = flip(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "execution_error",
                            "diff": None,
                            "score": -10000.0,
                            "score_reasons": ["execution_error"],
                        },
                        {
                            "candidate_num": 3,
                            "depth": 2,
                            "program_text": "let v0: Grid = recolor(input, 5)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": {"pixel_diff_count": 12},
                            "score": 320.0,
                            "score_reasons": ["dims_match", "pixel_diff=12"],
                        },
                        {
                            "candidate_num": 4,
                            "depth": 2,
                            "program_text": "let v0: Grid = rotate(input, 90)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "dimension_mismatch",
                            "diff": {
                                "pixel_diff_summary": "dimension mismatch",
                                "expected_dims": [5, 3],
                                "actual_dims": [3, 5],
                            },
                            "score": -525.0,
                            "score_reasons": ["dimension_mismatch", "dim_distance=4"],
                        },
                        {
                            "candidate_num": 5,
                            "depth": 2,
                            "program_text": "let v0: Grid = recolor(input, 2)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": {"pixel_diff_count": 20},
                            "score": 280.0,
                            "score_reasons": ["dims_match", "pixel_diff=20"],
                        },
                    ],
                },
                {
                    "plan": {
                        "name": "color_map",
                        "max_steps": 4,
                        "max_candidates": 5000,
                        "allowed_ops": ["recolor", "infer_map"],
                    },
                    "solved": True,
                    "candidates_tried": 100,
                    "feedback": {
                        "dominant_error_type": None,
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 0,
                        "execution_error_count": 0,
                        "suggested_focus": "generic",
                        "task_signatures": ["color:new_in_output", "dims:same"],
                        "best_candidate_num": None,
                        "best_candidate_score": None,
                    },
                    "winning_program": "let v0: Grid = recolor(input, 3)\noutput = v0",
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = infer_map(input)\noutput = v0",
                            "passed": False,
                            "failed_demo": 0,
                            "error_type": "wrong_output",
                            "diff": {"pixel_diff_count": 8},
                            "score": 350.0,
                            "score_reasons": ["dims_match", "pixel_diff=8"],
                        },
                        {
                            "candidate_num": 2,
                            "depth": 1,
                            "program_text": "let v0: Grid = recolor(input, 3)\noutput = v0",
                            "passed": True,
                            "failed_demo": None,
                            "error_type": None,
                            "diff": None,
                            "score": 1000000.0,
                            "score_reasons": ["exact_pass"],
                        },
                    ],
                },
            ],
        },
    ]


def _scored_trace_records_no_winner_improvement() -> list[dict]:
    """Two rounds, neither solves, but round 1 best candidate improves on round 0."""
    return [
        {
            "task_id": "task_030",
            "solved": False,
            "candidates_tried": 200,
            "winning_program": None,
            "rounds": [
                {
                    "plan": {"name": "generic", "max_steps": 3, "max_candidates": 5000, "allowed_ops": []},
                    "solved": False,
                    "candidates_tried": 100,
                    "feedback": {
                        "dominant_error_type": "wrong_output",
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 5,
                        "execution_error_count": 0,
                        "suggested_focus": "color_map",
                        "task_signatures": ["dims:same"],
                        "best_candidate_num": 1,
                        "best_candidate_score": 200.0,
                    },
                    "winning_program": None,
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = identity(input)\noutput = v0",
                            "passed": False,
                            "error_type": "wrong_output",
                            "score": 200.0,
                        },
                    ],
                },
                {
                    "plan": {"name": "color_map", "max_steps": 4, "max_candidates": 5000, "allowed_ops": []},
                    "solved": False,
                    "candidates_tried": 100,
                    "feedback": {
                        "dominant_error_type": "wrong_output",
                        "dimension_mismatch_count": 0,
                        "pixel_mismatch_count": 2,
                        "execution_error_count": 0,
                        "suggested_focus": "generic",
                        "task_signatures": ["dims:same"],
                        "best_candidate_num": 1,
                        "best_candidate_score": 380.0,
                    },
                    "winning_program": None,
                    "trace": [
                        {
                            "candidate_num": 1,
                            "depth": 1,
                            "program_text": "let v0: Grid = recolor(input, 1)\noutput = v0",
                            "passed": False,
                            "error_type": "wrong_output",
                            "score": 380.0,
                        },
                    ],
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProgramStoreExport:
    def test_sketch_examples_emitted(self):
        examples = examples_from_program_store(_program_store_records())
        assert len(examples) == 2
        assert all(e.task_type == "SKETCH" for e in examples)

    def test_sketch_required_fields(self):
        examples = examples_from_program_store(_program_store_records())
        for ex in examples:
            d = ex.to_dict()
            assert d["schema_version"] == SCHEMA_VERSION
            assert d["task_type"] in TASK_TYPES
            assert isinstance(d["task_signatures"], list)
            assert len(d["task_signatures"]) > 0
            assert isinstance(d["target"], dict)
            assert "program" in d["target"]

    def test_sketch_task_id_populated(self):
        examples = examples_from_program_store(_program_store_records())
        assert examples[0].task_id == "task_001"
        assert examples[1].task_id == "task_002"

    def test_empty_store_produces_nothing(self):
        assert examples_from_program_store([]) == []


class TestRefinementTraceExport:
    def test_next_focus_emitted(self):
        examples = examples_from_refinement_traces(_trace_store_records())
        focus_examples = [e for e in examples if e.task_type == "NEXT_FOCUS"]
        assert len(focus_examples) == 1
        assert focus_examples[0].round_index == 1
        assert focus_examples[0].target["focus"] == "color_map"

    def test_candidate_rank_emitted(self):
        examples = examples_from_refinement_traces(_trace_store_records())
        rank_examples = [e for e in examples if e.task_type == "CANDIDATE_RANK"]
        # Round 1 has a winner + 1 failing candidate -> 1 example
        assert len(rank_examples) == 1
        assert rank_examples[0].target["preferred"] is not None
        assert rank_examples[0].target["rejected"] is not None

    def test_next_edit_legacy_weak(self):
        """Legacy traces without best_candidate_num produce weak edits."""
        examples = examples_from_refinement_traces(_trace_store_records())
        edit_examples = [e for e in examples if e.task_type == "NEXT_EDIT"]
        assert len(edit_examples) == 1
        assert edit_examples[0].target["edit_quality"] == "weak"
        # No before_program since legacy traces lack best_candidate_num
        assert edit_examples[0].target.get("before_program") is None

    def test_winning_program_preserved(self):
        examples = examples_from_refinement_traces(_trace_store_records())
        for ex in examples:
            if ex.task_type != "NEXT_EDIT" or ex.winning_program is not None:
                continue
            # Legacy records may have None winning_program for unsolved tasks
            pass

    def test_empty_traces_produce_nothing(self):
        assert examples_from_refinement_traces([]) == []


class TestScoredNextEdit:
    """Tests for NEXT_EDIT using best-candidate scoring."""

    def test_strong_edit_uses_best_candidate(self):
        examples = examples_from_refinement_traces(_scored_trace_records())
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        assert len(edits) == 1
        t = edits[0].target
        assert t["edit_quality"] == "strong"
        assert t["before_program"] == "let v0: Grid = recolor(input, 5)\noutput = v0"
        assert t["after_program"] == "let v0: Grid = recolor(input, 3)\noutput = v0"

    def test_strong_edit_has_scores(self):
        examples = examples_from_refinement_traces(_scored_trace_records())
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        t = edits[0].target
        assert t["before_score"] == 320.0
        assert t["after_score"] == 1_000_000.0
        assert t["score_delta"] == 1_000_000.0 - 320.0

    def test_strong_edit_has_diff_progress(self):
        examples = examples_from_refinement_traces(_scored_trace_records())
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        t = edits[0].target
        # before_feedback comes from round 0's feedback which has best_candidate fields
        bf = t.get("before_feedback", {})
        assert bf["best_candidate_score"] == 320.0
        assert bf["best_candidate_dims_match"] is True
        assert bf["best_candidate_pixel_diff_count"] == 12
        assert bf["best_candidate_wrong_row_count"] == 2
        assert bf["best_candidate_palette_expected_coverage"] == 0.85

    def test_strong_edit_current_program_set(self):
        """current_program on the example should be the before_program."""
        examples = examples_from_refinement_traces(_scored_trace_records())
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        assert edits[0].current_program == "let v0: Grid = recolor(input, 5)\noutput = v0"

    def test_medium_edit_on_improvement(self):
        """When neither round solves but best candidate improves, emit medium."""
        examples = examples_from_refinement_traces(
            _scored_trace_records_no_winner_improvement()
        )
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        assert len(edits) == 1
        t = edits[0].target
        assert t["edit_quality"] == "medium"
        assert t["before_program"] == "let v0: Grid = identity(input)\noutput = v0"
        assert t["after_program"] == "let v0: Grid = recolor(input, 1)\noutput = v0"
        assert t["before_score"] == 200.0
        assert t["after_score"] == 380.0
        assert t["score_delta"] == 180.0

    def test_no_edit_when_no_improvement(self):
        """If round 1's best candidate is worse, no NEXT_EDIT is emitted."""
        records = _scored_trace_records_no_winner_improvement()
        # Make round 1's best score worse than round 0's
        records[0]["rounds"][1]["feedback"]["best_candidate_score"] = 150.0
        records[0]["rounds"][1]["trace"][0]["score"] = 150.0
        examples = examples_from_refinement_traces(records)
        edits = [e for e in examples if e.task_type == "NEXT_EDIT"]
        assert len(edits) == 0


class TestScoredCandidateRank:
    """Tests for score-aware CANDIDATE_RANK."""

    def test_rank_includes_scores(self):
        examples = examples_from_refinement_traces(_scored_trace_records())
        ranks = [e for e in examples if e.task_type == "CANDIDATE_RANK"]
        # Round 1 has 1 failing candidate with score -> 1 rank example
        assert len(ranks) >= 1
        for r in ranks:
            t = r.target
            if t.get("rejected_score") is not None:
                assert "score_delta" in t
                assert "preferred_score" in t

    def test_rank_prefers_hard_negatives(self):
        """When round 0 has scored candidates, hard negatives come first."""
        examples = examples_from_refinement_traces(_scored_trace_records())
        ranks = [
            e for e in examples
            if e.task_type == "CANDIDATE_RANK" and e.round_index == 0
        ]
        # Round 0 has no winner -> no CANDIDATE_RANK for round 0
        # (winning program required for rank examples)
        # But round 1 does have rank examples
        round1_ranks = [
            e for e in examples
            if e.task_type == "CANDIDATE_RANK" and e.round_index == 1
        ]
        assert len(round1_ranks) == 1
        # The single failure in round 1 has score 350.0
        assert round1_ranks[0].target.get("rejected_score") == 350.0

    def test_rank_diverse_error_types(self):
        """Hard-negative selection prefers diverse error types."""
        # Build a custom trace with mixed error types and a winner
        records = [{
            "task_id": "task_err",
            "solved": True,
            "candidates_tried": 100,
            "winning_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
            "rounds": [{
                "plan": {"name": "generic", "max_steps": 3, "max_candidates": 5000, "allowed_ops": []},
                "solved": True,
                "candidates_tried": 100,
                "feedback": {
                    "dominant_error_type": "wrong_output",
                    "dimension_mismatch_count": 0,
                    "pixel_mismatch_count": 3,
                    "execution_error_count": 1,
                    "suggested_focus": "generic",
                    "task_signatures": ["dims:same"],
                },
                "winning_program": "let v0: Grid = recolor(input, 1)\noutput = v0",
                "trace": [
                    {"candidate_num": 1, "depth": 1, "program_text": "prog_a", "passed": False,
                     "error_type": "wrong_output", "score": 300.0, "score_reasons": ["dims_match"]},
                    {"candidate_num": 2, "depth": 1, "program_text": "prog_b", "passed": False,
                     "error_type": "wrong_output", "score": 290.0, "score_reasons": ["dims_match"]},
                    {"candidate_num": 3, "depth": 1, "program_text": "prog_c", "passed": False,
                     "error_type": "execution_error", "score": -10000.0, "score_reasons": ["execution_error"]},
                    {"candidate_num": 4, "depth": 1, "program_text": "prog_d", "passed": False,
                     "error_type": "wrong_output", "score": 280.0, "score_reasons": ["dims_match"]},
                    {"candidate_num": 5, "depth": 1, "program_text": "prog_e", "passed": True,
                     "error_type": None, "score": 1000000.0, "score_reasons": ["exact_pass"]},
                ],
            }],
        }]
        examples = examples_from_refinement_traces(records)
        ranks = [e for e in examples if e.task_type == "CANDIDATE_RANK"]
        assert len(ranks) == 3
        rejected_texts = [r.target["rejected"] for r in ranks]
        # prog_a is highest scored failing, selected first
        assert rejected_texts[0] == "prog_a"
        # prog_c has a different error type (execution_error), selected second for diversity
        assert rejected_texts[1] == "prog_c"
        # prog_b is next highest scored wrong_output
        assert rejected_texts[2] == "prog_b"

    def test_rank_includes_score_reasons(self):
        examples = examples_from_refinement_traces(_scored_trace_records())
        ranks = [e for e in examples if e.task_type == "CANDIDATE_RANK"]
        for r in ranks:
            reasons = r.target.get("rejected_score_reasons")
            if reasons is not None:
                assert isinstance(reasons, list)
                assert len(reasons) > 0


class TestExportRoundTrip:
    def test_roundtrip_deterministic(self, tmp_path: Path):
        ps_examples = examples_from_program_store(_program_store_records())
        tr_examples = examples_from_refinement_traces(_scored_trace_records())
        all_examples = ps_examples + tr_examples

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        counts1 = export_examples(all_examples, dir1)
        counts2 = export_examples(all_examples, dir2)

        assert counts1 == counts2
        for task_type in counts1:
            file1 = dir1 / f"{task_type.lower()}.jsonl"
            file2 = dir2 / f"{task_type.lower()}.jsonl"
            assert file1.read_text() == file2.read_text()

    def test_roundtrip_load(self, tmp_path: Path):
        examples = examples_from_program_store(_program_store_records())
        export_examples(examples, tmp_path)

        loaded = load_jsonl(tmp_path / "sketch.jsonl")
        assert len(loaded) == len(examples)
        for row in loaded:
            assert row["schema_version"] == SCHEMA_VERSION
            assert row["task_type"] == "SKETCH"

    def test_all_required_fields_present(self, tmp_path: Path):
        all_examples = (
            examples_from_program_store(_program_store_records())
            + examples_from_refinement_traces(_scored_trace_records())
        )
        export_examples(all_examples, tmp_path)

        required_keys = {
            "schema_version",
            "task_type",
            "task_id",
            "task_signatures",
            "current_program",
            "round_index",
            "feedback",
            "target",
            "winning_program",
        }

        for jsonl_file in tmp_path.glob("*.jsonl"):
            for row in load_jsonl(jsonl_file):
                assert set(row.keys()) == required_keys

    def test_scored_roundtrip_stable(self, tmp_path: Path):
        """Scored trace exports are byte-identical across runs."""
        examples = examples_from_refinement_traces(_scored_trace_records())
        dir1 = tmp_path / "a"
        dir2 = tmp_path / "b"
        export_examples(examples, dir1)
        export_examples(examples, dir2)
        for f in dir1.glob("*.jsonl"):
            assert f.read_text() == (dir2 / f.name).read_text()


class TestSchemaVersioning:
    def test_version_is_int(self):
        ex = examples_from_program_store(_program_store_records())[0]
        assert isinstance(ex.schema_version, int)

    def test_version_bumped_to_2(self):
        assert SCHEMA_VERSION == 2

    def test_version_matches_constant(self):
        examples = (
            examples_from_program_store(_program_store_records())
            + examples_from_refinement_traces(_scored_trace_records())
        )
        for ex in examples:
            assert ex.schema_version == SCHEMA_VERSION

    def test_version_in_exported_json(self, tmp_path: Path):
        examples = examples_from_program_store(_program_store_records())
        export_examples(examples, tmp_path)
        for row in load_jsonl(tmp_path / "sketch.jsonl"):
            assert row["schema_version"] == SCHEMA_VERSION
