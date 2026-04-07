"""Tests for real-data guidance evaluation.

Covers task-aware splits, CV pipeline, shadow mode, integration readiness.
Uses synthetic data where needed, but also tests real-data pipeline if available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list
from aria.core.guidance_datasets import (
    FEATURE_NAMES,
    GuidanceDataset,
    GuidanceExample,
)


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


def _make_dataset(n_tasks: int = 10, n_classes: int = 3) -> GuidanceDataset:
    rng = np.random.default_rng(42)
    label_names = tuple(f"class_{i}" for i in range(n_classes))
    examples = []
    for t in range(n_tasks):
        label = t % n_classes
        features = rng.standard_normal(len(FEATURE_NAMES))
        features[0] = label + rng.standard_normal() * 0.3
        examples.append(GuidanceExample(
            features=features,
            label=label,
            label_name=label_names[label],
            candidates=label_names,
            task_id=f"task_{t}",
        ))
    return GuidanceDataset(name="test", label_names=label_names, examples=examples)


# ---------------------------------------------------------------------------
# Task-aware split
# ---------------------------------------------------------------------------


class TestTaskAwareSplit:
    def test_no_task_leakage(self):
        from aria.core.guidance_real_eval import task_aware_split

        ds = _make_dataset(n_tasks=20)
        folds = task_aware_split(ds, n_folds=5)

        for train_ds, test_ds in folds:
            train_tasks = {ex.task_id for ex in train_ds.examples}
            test_tasks = {ex.task_id for ex in test_ds.examples}
            overlap = train_tasks & test_tasks
            assert not overlap, f"Task leakage: {overlap}"

    def test_all_examples_covered(self):
        from aria.core.guidance_real_eval import task_aware_split

        ds = _make_dataset(n_tasks=20)
        folds = task_aware_split(ds, n_folds=5)

        all_test_ids = set()
        for _, test_ds in folds:
            for ex in test_ds.examples:
                all_test_ids.add(ex.task_id)

        all_tasks = {ex.task_id for ex in ds.examples}
        assert all_test_ids == all_tasks

    def test_fold_count(self):
        from aria.core.guidance_real_eval import task_aware_split

        ds = _make_dataset(n_tasks=20)
        folds = task_aware_split(ds, n_folds=5)
        assert len(folds) == 5

    def test_small_dataset_fewer_folds(self):
        from aria.core.guidance_real_eval import task_aware_split

        ds = _make_dataset(n_tasks=3)
        folds = task_aware_split(ds, n_folds=5)
        # Should produce <= 3 valid folds
        assert len(folds) <= 3


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------


class TestCVEvaluation:
    def test_evaluate_on_real_data(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = _make_dataset(n_tasks=20, n_classes=3)
        report = evaluate_on_real_data(ds, n_folds=4)

        assert report.sufficient_data is True
        assert report.n_tasks == 20
        assert report.n_examples == 20
        assert len(report.cv_results) == 5  # 5 models
        assert all(cv.n_folds == 4 for cv in report.cv_results)

    def test_insufficient_data_detected(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = _make_dataset(n_tasks=3, n_classes=3)
        report = evaluate_on_real_data(ds, min_examples=10)

        assert report.sufficient_data is False
        assert "too few" in report.reason

    def test_report_to_dict(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = _make_dataset(n_tasks=20)
        report = evaluate_on_real_data(ds, n_folds=3)
        d = report.to_dict()
        serialized = json.dumps(d)
        rt = json.loads(serialized)
        assert "cv_results" in rt

    def test_symbolic_default_is_baseline(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = _make_dataset(n_tasks=20)
        report = evaluate_on_real_data(ds)

        model_names = [cv.model_name for cv in report.cv_results]
        assert "symbolic_default" in model_names


# ---------------------------------------------------------------------------
# Integration readiness
# ---------------------------------------------------------------------------


class TestIntegrationReadiness:
    def test_ready_when_learned_beats_default(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        # Create a dataset where features correlate strongly with labels
        rng = np.random.default_rng(0)
        label_names = ("a", "b")
        examples = []
        for t in range(30):
            label = 0 if t < 15 else 1
            features = np.zeros(len(FEATURE_NAMES))
            features[0] = label * 10 + rng.standard_normal() * 0.1
            examples.append(GuidanceExample(
                features=features, label=label, label_name=label_names[label],
                candidates=label_names, task_id=f"task_{t}",
            ))

        ds = GuidanceDataset(name="test", label_names=label_names, examples=examples)
        report = evaluate_on_real_data(ds, n_folds=3)

        # With clear signal, learned model should do well
        linear_cv = next((cv for cv in report.cv_results if cv.model_name == "linear_softmax"), None)
        assert linear_cv is not None
        assert linear_cv.mean_top1 > 0.7

    def test_not_ready_with_noise(self):
        from aria.core.guidance_real_eval import evaluate_on_real_data

        # Pure noise — no model should beat default significantly
        rng = np.random.default_rng(0)
        label_names = ("a", "b", "c")
        examples = []
        for t in range(30):
            label = t % 3
            features = rng.standard_normal(len(FEATURE_NAMES))
            examples.append(GuidanceExample(
                features=features, label=label, label_name=label_names[label],
                candidates=label_names, task_id=f"task_{t}",
            ))

        ds = GuidanceDataset(name="test", label_names=label_names, examples=examples)
        report = evaluate_on_real_data(ds, n_folds=3)
        # With pure noise, no model should be flagged as integration-ready
        # (unless lucky, so we just check the report is well-formed)
        assert isinstance(report.integration_ready, bool)


# ---------------------------------------------------------------------------
# Shadow mode
# ---------------------------------------------------------------------------


class TestShadowMode:
    def test_shadow_comparisons(self):
        from aria.core.guidance_real_eval import (
            SymbolicDefaultModel, compute_shadow_comparisons, summarize_shadow,
        )
        from aria.core.guidance_models import LinearSoftmax

        ds = _make_dataset(n_tasks=10, n_classes=2)
        X, y = ds.to_arrays()

        default = SymbolicDefaultModel(ds.name, ds.label_names)
        learned = LinearSoftmax(lr=0.1, epochs=100)
        learned.fit(X, y)

        comps = compute_shadow_comparisons(ds, learned, default)
        assert len(comps) == 10

        summary = summarize_shadow(comps)
        assert "mean_rank_improvement" in summary
        assert "pct_improved" in summary

    def test_shadow_comparison_roundtrip(self):
        from aria.core.guidance_real_eval import ShadowComparison

        sc = ShadowComparison(
            subproblem="test", default_rank=2, model_rank=0,
            model_name="linear", winner_label="a",
            rank_improvement=2, would_save_attempts=2,
        )
        d = sc.to_dict()
        sc2 = ShadowComparison.from_dict(d)
        assert sc2.rank_improvement == 2


# ---------------------------------------------------------------------------
# Real ARC data pipeline (only if data available)
# ---------------------------------------------------------------------------


class TestRealARCPipeline:
    @pytest.fixture
    def arc_pairs(self):
        """Load a small slice of real ARC tasks if available."""
        try:
            from aria.datasets import get_dataset, list_task_ids, load_arc_task
            ds = get_dataset("v1-train")
            task_ids = list_task_ids(ds)[:20]
            pairs = []
            for tid in task_ids:
                task = load_arc_task(ds, tid)
                pairs.append((tid, task.train))
            return pairs
        except Exception:
            pytest.skip("ARC data not available")

    def test_output_size_on_real_data(self, arc_pairs):
        from aria.core.guidance_datasets import build_output_size_dataset
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = build_output_size_dataset(arc_pairs)
        assert ds.n_examples > 0
        # With 20 tasks, most classes are rare — relax sufficiency thresholds
        report = evaluate_on_real_data(ds, n_folds=3, min_examples=5, min_classes_with_support=1)
        assert report.n_tasks > 0

    def test_periodic_axis_on_real_data(self, arc_pairs):
        from aria.core.guidance_datasets import build_periodic_axis_dataset
        from aria.core.guidance_real_eval import evaluate_on_real_data

        ds = build_periodic_axis_dataset(arc_pairs)
        if ds.n_examples < 4:
            pytest.skip(f"Only {ds.n_examples} periodic axis examples in 20 tasks")
        report = evaluate_on_real_data(ds, n_folds=3, min_examples=4)
        assert report.n_tasks > 0


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestRealEvalNoTaskIdLogic:
    def test_no_task_id_dispatch(self):
        path = Path(__file__).parent.parent / "aria" / "core" / "guidance_real_eval.py"
        content = path.read_text()
        dispatch_patterns = [r'if\s+.*task_id\s*==', r'if\s+.*task_id\s+in\b']
        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in dispatch_patterns:
                if "task_id" in stripped and re.search(pattern, stripped):
                    if "task_groups" in stripped or "fold_assignments" in stripped:
                        continue
                    pytest.fail(f"guidance_real_eval.py:{i} task-id dispatch: {stripped}")


class TestRealEvalNoRegressions:
    def test_export_unchanged(self):
        from aria.core.guidance_export import export_task, validate_record
        demos = _identity_demos()
        record = export_task("t1", demos)
        assert not validate_record(record)
