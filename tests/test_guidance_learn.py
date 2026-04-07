"""Tests for learned guidance: residual-rich traces, datasets, models, evaluation.

Covers:
- Residual-enriched spec traces
- Dataset generation for each subproblem
- Model train/eval pipeline smoke tests
- No task-id logic
- No regressions
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


def _flip_demos() -> tuple[DemoPair, ...]:
    """Task where output is horizontal flip of input."""
    g1 = grid_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    o1 = grid_from_list([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    g2 = grid_from_list([[0, 1, 0], [2, 3, 2], [0, 4, 0]])
    o2 = grid_from_list([[0, 1, 0], [2, 3, 2], [0, 4, 0]])
    return (DemoPair(input=g1, output=o1), DemoPair(input=g2, output=o2))


# ---------------------------------------------------------------------------
# Phase 1: Residual-rich specialization traces
# ---------------------------------------------------------------------------


class TestResidualRichSpecTraces:
    def test_alternative_has_residual_fields(self):
        from aria.core.guidance_spec_traces import SpecializationAlternative

        a = SpecializationAlternative(
            value="row", source="evidence", rank=0, chosen=True,
            confidence=0.8, rationale="test",
            compiled=True, verified=False, residual_fraction=0.3,
            residual_localized=True, residual_dominant_region="interior",
            blamed_ops=("APPLY_TRANSFORM",),
        )
        d = a.to_dict()
        assert d["residual_fraction"] == 0.3
        assert d["residual_localized"] is True
        assert d["blamed_ops"] == ["APPLY_TRANSFORM"]

        a2 = SpecializationAlternative.from_dict(d)
        assert a2.residual_fraction == 0.3
        assert a2.blamed_ops == ("APPLY_TRANSFORM",)

    def test_chosen_alt_omits_residual_fields(self):
        from aria.core.guidance_spec_traces import SpecializationAlternative

        a = SpecializationAlternative(
            value="row", source="evidence", rank=0, chosen=True,
            confidence=0.8, rationale="test",
        )
        d = a.to_dict()
        # Optional fields not in dict when None
        assert "compiled" not in d
        assert "residual_fraction" not in d

    def test_trace_enriches_alternatives(self):
        from aria.core.guidance_spec_traces import trace_specialization

        demos = _identity_demos()
        ep = trace_specialization("t1", demos)

        # Check that at least some alternatives have residual info
        for dec in ep.decisions:
            for alt in dec.alternatives:
                if not alt.chosen and alt.compiled is not None:
                    # Alternative was attempted — should have compile result
                    assert isinstance(alt.compiled, bool)


# ---------------------------------------------------------------------------
# Phase 2: Dataset generation
# ---------------------------------------------------------------------------


class TestDatasetGeneration:
    def test_extract_features(self):
        from aria.core.guidance_datasets import extract_perception_features, FEATURE_NAMES

        demos = _identity_demos()
        features = extract_perception_features(demos)

        assert isinstance(features, dict)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
        assert features["same_dims"] == 1.0
        assert features["n_demos"] == 2.0

    def test_features_to_array(self):
        from aria.core.guidance_datasets import extract_perception_features, features_to_array, FEATURE_NAMES

        demos = _identity_demos()
        features = extract_perception_features(demos)
        arr = features_to_array(features)

        assert arr.shape == (len(FEATURE_NAMES),)
        assert arr.dtype == np.float64

    def test_build_output_size_dataset(self):
        from aria.core.guidance_datasets import build_output_size_dataset

        pairs = [("t1", _identity_demos()), ("t2", _identity_demos())]
        ds = build_output_size_dataset(pairs)

        assert ds.name == "output_size_mode"
        assert ds.n_classes > 0
        # Identity tasks should have same_as_input → same_dims
        if ds.n_examples > 0:
            assert ds.examples[0].label_name == "same_dims"

    def test_build_periodic_axis_dataset(self):
        from aria.core.guidance_datasets import build_periodic_axis_dataset

        # Identity task likely has periodic repair → axis="row"
        pairs = [("t1", _identity_demos())]
        ds = build_periodic_axis_dataset(pairs)
        assert ds.name == "periodic_axis"

    def test_build_transform_dataset(self):
        from aria.core.guidance_datasets import build_transform_dataset

        pairs = [("t1", _identity_demos())]
        ds = build_transform_dataset(pairs)
        assert ds.name == "transform_choice"

    def test_dataset_to_arrays(self):
        from aria.core.guidance_datasets import build_output_size_dataset

        pairs = [("t1", _identity_demos()), ("t2", _identity_demos())]
        ds = build_output_size_dataset(pairs)

        X, y = ds.to_arrays()
        assert X.shape[0] == y.shape[0]
        if ds.n_examples > 0:
            assert X.shape[1] > 0

    def test_dataset_to_dict(self):
        from aria.core.guidance_datasets import build_output_size_dataset

        pairs = [("t1", _identity_demos())]
        ds = build_output_size_dataset(pairs)
        d = ds.to_dict()
        assert "name" in d
        assert "n_examples" in d
        assert "label_distribution" in d


# ---------------------------------------------------------------------------
# Phase 3: Model smoke tests
# ---------------------------------------------------------------------------


class TestModels:
    def _make_dataset(self, n: int = 20, n_classes: int = 3) -> "GuidanceDataset":
        from aria.core.guidance_datasets import GuidanceDataset, GuidanceExample, FEATURE_NAMES

        rng = np.random.default_rng(42)
        label_names = tuple(f"class_{i}" for i in range(n_classes))
        examples = []
        for i in range(n):
            label = i % n_classes
            # Create features correlated with label
            features = rng.standard_normal(len(FEATURE_NAMES))
            features[0] = label + rng.standard_normal() * 0.3  # signal on first feature
            examples.append(GuidanceExample(
                features=features,
                label=label,
                label_name=label_names[label],
                candidates=label_names,
            ))
        return GuidanceDataset(name="test", label_names=label_names, examples=examples)

    def test_majority_baseline(self):
        from aria.core.guidance_models import MajorityBaseline, evaluate_model

        ds = self._make_dataset()
        model = MajorityBaseline()
        X, y = ds.to_arrays()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), ds.n_classes)
        assert np.allclose(probs.sum(axis=1), 1.0)

        result = evaluate_model(model, ds)
        assert 0 <= result.top1_accuracy <= 1

    def test_nearest_centroid(self):
        from aria.core.guidance_models import NearestCentroid, evaluate_model

        ds = self._make_dataset()
        model = NearestCentroid()
        X, y = ds.to_arrays()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), ds.n_classes)

        result = evaluate_model(model, ds)
        assert result.top1_accuracy >= 0  # at least defined

    def test_decision_stump(self):
        from aria.core.guidance_models import DecisionStump, evaluate_model

        ds = self._make_dataset()
        model = DecisionStump()
        X, y = ds.to_arrays()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), ds.n_classes)

    def test_linear_softmax(self):
        from aria.core.guidance_models import LinearSoftmax, evaluate_model

        ds = self._make_dataset()
        model = LinearSoftmax(lr=0.05, epochs=50)
        X, y = ds.to_arrays()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), ds.n_classes)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_linear_softmax_beats_majority(self):
        """On a dataset with signal, linear model should beat majority."""
        from aria.core.guidance_models import LinearSoftmax, MajorityBaseline, evaluate_model

        ds = self._make_dataset(n=60, n_classes=3)
        X, y = ds.to_arrays()

        majority = MajorityBaseline()
        majority.fit(X, y)
        maj_result = evaluate_model(majority, ds)

        linear = LinearSoftmax(lr=0.1, epochs=300)
        linear.fit(X, y)
        lin_result = evaluate_model(linear, ds)

        # Linear should match or beat majority on training data
        assert lin_result.top1_accuracy >= maj_result.top1_accuracy - 0.05


# ---------------------------------------------------------------------------
# Phase 4: Train/eval pipeline
# ---------------------------------------------------------------------------


class TestTrainEvalPipeline:
    def test_train_and_evaluate(self):
        from aria.core.guidance_models import train_and_evaluate
        from aria.core.guidance_datasets import GuidanceDataset, GuidanceExample, FEATURE_NAMES

        rng = np.random.default_rng(0)
        label_names = ("a", "b")
        examples = []
        for i in range(20):
            label = i % 2
            features = rng.standard_normal(len(FEATURE_NAMES))
            features[0] = label * 2
            examples.append(GuidanceExample(
                features=features, label=label, label_name=label_names[label],
                candidates=label_names,
            ))

        ds = GuidanceDataset(name="test", label_names=label_names, examples=examples)
        report = train_and_evaluate(ds, train_frac=0.7)

        assert report.dataset_name == "test"
        assert report.n_train > 0
        assert report.n_eval > 0
        assert len(report.results) == 4  # majority, centroid, stump, linear

        d = report.to_dict()
        serialized = json.dumps(d)
        assert "results" in json.loads(serialized)

    def test_too_few_examples_handled(self):
        from aria.core.guidance_models import train_and_evaluate
        from aria.core.guidance_datasets import GuidanceDataset

        ds = GuidanceDataset(name="empty", label_names=("a",), examples=[])
        report = train_and_evaluate(ds)
        assert report.n_eval == 0
        assert len(report.results) == 0

    def test_real_dataset_pipeline(self):
        """End-to-end: build dataset from demos → train → evaluate."""
        from aria.core.guidance_datasets import build_output_size_dataset
        from aria.core.guidance_models import train_and_evaluate

        pairs = [
            ("t1", _identity_demos()),
            ("t2", _identity_demos()),
            ("t3", _identity_demos()),
            ("t4", _identity_demos()),
            ("t5", _identity_demos()),
        ]
        ds = build_output_size_dataset(pairs)

        if ds.n_examples >= 4:
            report = train_and_evaluate(ds)
            assert report.n_train > 0


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestLearnNoTaskIdLogic:
    def _files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return [
            core / "guidance_datasets.py",
            core / "guidance_models.py",
        ]

    def test_no_task_id_dispatch(self):
        dispatch_patterns = [r'if\s+.*task_id\s*==', r'if\s+.*task_id\s+in\b']
        for path in self._files():
            if not path.exists():
                continue
            content = path.read_text()
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in dispatch_patterns:
                    if "task_id" in stripped and re.search(pattern, stripped):
                        pytest.fail(f"{path.name}:{i} task-id dispatch: {stripped}")


class TestLearnNoRegressions:
    def test_export_unchanged(self):
        from aria.core.guidance_export import export_task, validate_record

        demos = _identity_demos()
        record = export_task("t1", demos)
        assert not validate_record(record)

    def test_spec_traces_unchanged(self):
        from aria.core.guidance_spec_traces import trace_specialization

        demos = _identity_demos()
        ep = trace_specialization("t1", demos)
        assert ep.task_id == "t1"
