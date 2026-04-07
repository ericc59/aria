"""Tests for the guidance/data track.

Covers: export format, label schemas, eval harness, baselines, synthetic hooks.
Uses synthetic mini-tasks — no ARC data files required.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Synthetic task helpers
# ---------------------------------------------------------------------------


def _identity_demos() -> tuple[DemoPair, ...]:
    """3x3 identity task: output == input."""
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


def _simple_scale_demos() -> tuple[DemoPair, ...]:
    """Task where output is 2x the input."""
    g1 = grid_from_list([[1, 2], [3, 4]])
    o1 = grid_from_list([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    g2 = grid_from_list([[5, 6], [7, 8]])
    o2 = grid_from_list([[5, 5, 6, 6], [5, 5, 6, 6], [7, 7, 8, 8], [7, 7, 8, 8]])
    return (DemoPair(input=g1, output=o1), DemoPair(input=g2, output=o2))


# ---------------------------------------------------------------------------
# Phase 1: Export format tests
# ---------------------------------------------------------------------------


class TestGuidanceExport:
    def test_export_returns_all_required_keys(self):
        from aria.core.guidance_export import export_task, REQUIRED_KEYS

        demos = _identity_demos()
        record = export_task("test_identity", demos)

        missing = REQUIRED_KEYS - set(record.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_export_is_json_serializable(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_identity", demos)

        # Should not raise
        serialized = json.dumps(record, sort_keys=True)
        roundtrip = json.loads(serialized)
        assert roundtrip["task_id"] == "test_identity"
        assert roundtrip["schema_version"] == 1

    def test_export_perception_summaries_populated(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_identity", demos)

        assert len(record["perception_summaries"]) == len(demos)
        for ps in record["perception_summaries"]:
            assert "dims" in ps
            assert "bg_color" in ps
            assert "palette" in ps

    def test_export_train_demos_correct(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_identity", demos)

        assert len(record["train_demos"]) == len(demos)
        for d in record["train_demos"]:
            assert "input" in d
            assert "output" in d

    def test_validate_record(self):
        from aria.core.guidance_export import export_task, validate_record

        demos = _identity_demos()
        record = export_task("test_identity", demos)

        errors = validate_record(record)
        assert not errors, f"Validation errors: {errors}"

    def test_validate_record_catches_missing_keys(self):
        from aria.core.guidance_export import validate_record

        errors = validate_record({"task_id": "x"})
        assert any("missing keys" in e for e in errors)

    def test_batch_export(self):
        from aria.core.guidance_export import export_batch, load_records

        demos_map = {
            "t1": _identity_demos(),
            "t2": _identity_demos(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.jsonl"
            counts = export_batch(
                list(demos_map.keys()),
                lambda tid: demos_map[tid],
                path,
            )
            assert counts["exported"] == 2
            assert counts["skipped"] == 0

            records = load_records(path)
            assert len(records) == 2
            assert {r["task_id"] for r in records} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Phase 2: Label schema tests
# ---------------------------------------------------------------------------


class TestGuidanceLabels:
    def test_extract_labels_returns_all_fields(self):
        from aria.core.guidance_labels import extract_labels

        demos = _identity_demos()
        labels = extract_labels("test_identity", demos)

        assert labels.task_id == "test_identity"
        assert labels.roles is not None
        assert labels.legend is not None
        assert labels.slot_grid is not None
        assert labels.correspondences is not None
        assert labels.program_family is not None

    def test_labels_roundtrip(self):
        from aria.core.guidance_labels import extract_labels, TaskGuidanceLabels

        demos = _identity_demos()
        labels = extract_labels("test_identity", demos)

        d = labels.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        roundtrip_dict = json.loads(serialized)
        roundtrip = TaskGuidanceLabels.from_dict(roundtrip_dict)

        assert roundtrip.task_id == labels.task_id
        assert roundtrip.roles.bg_color == labels.roles.bg_color
        assert roundtrip.legend.present == labels.legend.present
        assert roundtrip.slot_grid.present == labels.slot_grid.present
        assert roundtrip.program_family.top_family == labels.program_family.top_family

    def test_save_load_labels(self):
        from aria.core.guidance_labels import extract_labels, save_labels, load_labels

        demos = _identity_demos()
        labels = [extract_labels("t1", demos), extract_labels("t2", demos)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            save_labels(labels, path)
            loaded = load_labels(path)

            assert len(loaded) == 2
            assert loaded[0].task_id == "t1"
            assert loaded[1].task_id == "t2"

    def test_label_coverage(self):
        from aria.core.guidance_labels import extract_labels, label_coverage

        demos = _identity_demos()
        labels = [extract_labels("t1", demos)]

        cov = label_coverage(labels)
        assert cov["total"] == 1
        assert isinstance(cov["has_program_family"], int)


# ---------------------------------------------------------------------------
# Phase 3: Eval harness tests
# ---------------------------------------------------------------------------


class TestGuidanceEval:
    def test_recall_at_k_basic(self):
        from aria.core.guidance_eval import recall_at_k

        assert recall_at_k(["a", "b", "c"], "a", 1) is True
        assert recall_at_k(["a", "b", "c"], "b", 1) is False
        assert recall_at_k(["a", "b", "c"], "b", 2) is True
        assert recall_at_k(["a", "b", "c"], "d", 3) is False

    def test_prf_all_correct(self):
        from aria.core.guidance_eval import PRFResult

        prf = PRFResult(tp=10, fp=0, fn=0, tn=5)
        assert prf.precision == 1.0
        assert prf.recall == 1.0
        assert prf.f1 == 1.0

    def test_prf_all_wrong(self):
        from aria.core.guidance_eval import PRFResult

        prf = PRFResult(tp=0, fp=5, fn=5, tn=0)
        assert prf.precision == 0.0
        assert prf.recall == 0.0
        assert prf.f1 == 0.0

    def test_legend_detection_prf(self):
        from aria.core.guidance_eval import legend_detection_prf
        from aria.core.guidance_labels import LegendLabel

        preds = [LegendLabel(present=True, edge="left"), LegendLabel(present=False)]
        golds = [LegendLabel(present=True, edge="left"), LegendLabel(present=True, edge="right")]

        prf = legend_detection_prf(preds, golds)
        assert prf.tp == 1
        assert prf.fn == 1
        assert prf.fp == 0

    def test_slot_grid_detection_prf(self):
        from aria.core.guidance_eval import slot_grid_detection_prf
        from aria.core.guidance_labels import SlotGridLabel

        preds = [SlotGridLabel(present=True, n_rows=3, n_cols=3)]
        golds = [SlotGridLabel(present=True, n_rows=3, n_cols=3)]

        prf = slot_grid_detection_prf(preds, golds)
        assert prf.tp == 1

    def test_evaluate_proposals_runs(self):
        from aria.core.guidance_eval import evaluate_proposals
        from aria.core.guidance_export import export_task
        from aria.core.guidance_labels import extract_labels

        demos = _identity_demos()
        record = export_task("t1", demos)
        labels = extract_labels("t1", demos)

        report = evaluate_proposals([record], [labels])
        assert report.n_tasks == 1
        d = report.to_dict()
        assert "output_size" in d
        assert "legend" in d


# ---------------------------------------------------------------------------
# Phase 4: Baselines tests
# ---------------------------------------------------------------------------


class TestGuidanceBaselines:
    def test_perception_signature_from_record(self):
        from aria.core.guidance_baselines import PerceptionSignature
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos)

        sig = PerceptionSignature.from_record(record)
        assert sig.same_dims is True
        assert isinstance(sig.n_colors, int)

    def test_signature_overlap(self):
        from aria.core.guidance_baselines import PerceptionSignature

        sig1 = PerceptionSignature(
            same_dims=True, has_partition=False, has_frame=False,
            has_legend=False, has_slot_grid=False,
            n_objects_bucket="1-3", n_colors=3,
        )
        sig2 = PerceptionSignature(
            same_dims=True, has_partition=False, has_frame=True,
            has_legend=False, has_slot_grid=False,
            n_objects_bucket="1-3", n_colors=3,
        )
        # All match except has_frame
        assert sig1.overlap(sig2) == 6

    def test_retrieval_index(self):
        from aria.core.guidance_baselines import RetrievalIndex
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        r1 = export_task("t1", demos)
        r1["verify_result"] = "solved"  # pretend solved for indexing

        index = RetrievalIndex.from_records([r1], solved_only=True)
        assert len(index.entries) == 1

        results = index.query(r1, k=1)
        assert len(results) == 1

    def test_size_mode_ranker(self):
        from aria.core.guidance_baselines import SizeModeRanker

        records = [
            {"verify_result": "solved", "size_spec": {"mode": "same_as_input"},
             "perception_summaries": [{"n_objects_4": 2, "palette": [0, 1, 2],
                                       "partition": None, "n_framed_regions": 0,
                                       "has_legend": False}],
             "slot_grid": None,
             "train_demos": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}]},
        ]

        ranker = SizeModeRanker()
        ranker.fit(records)

        assert "same_as_input" in ranker.rank(records[0], top_k=3)

    def test_decision_stumps(self):
        from aria.core.guidance_baselines import fit_decision_stumps

        records = [
            {"verify_result": "solved", "size_spec": {"mode": "same_as_input"},
             "perception_summaries": [{"n_objects_4": 2, "palette": [0, 1],
                                       "partition": None, "n_framed_regions": 0,
                                       "has_legend": False}],
             "slot_grid": None,
             "train_demos": [{"input": [[1]], "output": [[1]]}]},
        ]

        rules = fit_decision_stumps(records)
        assert len(rules) > 0
        assert all(r.accuracy >= 0 for r in rules)


# ---------------------------------------------------------------------------
# Phase 5: Synthetic hooks tests
# ---------------------------------------------------------------------------


class TestGuidanceSynthetic:
    def test_color_rotation_preserves_shape(self):
        from aria.core.guidance_synthetic import rotate_colors

        grid = grid_from_list([[1, 2], [3, 4]])
        rotated = rotate_colors(grid, {1: 5, 2: 6, 3: 7, 4: 8})

        assert rotated.shape == grid.shape
        assert int(rotated[0, 0]) == 5
        assert int(rotated[1, 1]) == 8

    def test_random_color_rotation_is_permutation(self):
        from aria.core.guidance_synthetic import random_color_rotation

        grid = grid_from_list([[0, 1, 2], [3, 4, 5]])
        rng = np.random.default_rng(42)
        rotated, mapping = random_color_rotation(grid, rng)

        # Same shape, same number of unique colors
        assert rotated.shape == grid.shape
        assert len(np.unique(rotated)) == len(np.unique(grid))

    def test_flip_grid(self):
        from aria.core.guidance_synthetic import flip_grid

        grid = grid_from_list([[1, 2], [3, 4]])
        lr = flip_grid(grid, "lr")
        assert int(lr[0, 0]) == 2
        assert int(lr[0, 1]) == 1

    def test_inject_noise_changes_pixels(self):
        from aria.core.guidance_synthetic import inject_noise

        grid = np.zeros((10, 10), dtype=np.uint8)
        noisy = inject_noise(grid, fraction=0.5, rng=np.random.default_rng(0))
        assert not np.array_equal(grid, noisy)

    def test_augment_color_rotation_record(self):
        from aria.core.guidance_synthetic import augment_color_rotation

        record = {
            "train_demos": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "perception_summaries": [{"bg_color": 0, "palette": [0, 1, 2, 3, 4], "non_bg_colors": [1, 2, 3, 4]}],
            "roles": [{"role": "FRAME", "color": 1}],
            "legend": None,
        }

        rng = np.random.default_rng(42)
        augmented = augment_color_rotation(record, rng=rng)

        # Structure preserved
        assert len(augmented["train_demos"]) == 1
        assert len(augmented["train_demos"][0]["input"]) == 2

    def test_augment_spatial_flip_record(self):
        from aria.core.guidance_synthetic import augment_spatial_flip

        record = {
            "train_demos": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
        }

        flipped = augment_spatial_flip(record, axis="lr")
        assert flipped["train_demos"][0]["input"][0] == [2, 1]

    def test_generate_family_variants(self):
        from aria.core.guidance_synthetic import generate_family_variants

        record = {
            "train_demos": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "perception_summaries": [{"bg_color": 0, "palette": [0, 1, 2, 3, 4], "non_bg_colors": [1, 2, 3, 4]}],
            "roles": [],
            "legend": None,
            "correspondences": None,
        }

        variants = generate_family_variants(record, n_variants=3, rng=np.random.default_rng(0))
        assert len(variants) == 3


# ---------------------------------------------------------------------------
# Integrity: no task-id logic
# ---------------------------------------------------------------------------


class TestNoTaskIdLogic:
    """Verify no new file uses task_id for dispatch or branching."""

    def _guidance_source_files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return sorted(core.glob("guidance_*.py"))

    def test_no_task_id_dispatch(self):
        """task_id should only be used as a metadata key, never for control flow."""
        dispatch_patterns = [
            r'if\s+.*task_id\s*==',
            r'if\s+.*task_id\s+in\b',
            r'task_id\s*\[',
            r'\.startswith\(',
        ]

        for path in self._guidance_source_files():
            content = path.read_text()
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in dispatch_patterns:
                    if "task_id" in stripped and re.search(pattern, stripped):
                        # Allow dict lookups and metadata assignments
                        if "label_by_id" in stripped or "record.get" in stripped:
                            continue
                        pytest.fail(
                            f"{path.name}:{i} looks like task-id dispatch: {stripped}"
                        )
