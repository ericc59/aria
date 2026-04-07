"""Tests for output-size model artifact, shadow reranking, and evaluation."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list
from aria.core.guidance_datasets import FEATURE_NAMES, SIZE_FAMILY_NAMES


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


class TestModelArtifact:
    def test_train_produces_artifact(self):
        from aria.core.guidance_size_model import train_size_model

        pairs = [("t1", _identity_demos()), ("t2", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        assert model.version == 1
        assert model.name == "output_size_linear_softmax_v1"
        assert model.feature_names == tuple(FEATURE_NAMES)
        assert model.label_names == SIZE_FAMILY_NAMES
        assert model.weights.shape == (len(FEATURE_NAMES), len(SIZE_FAMILY_NAMES))

    def test_predict_proba_valid(self):
        from aria.core.guidance_size_model import train_size_model
        from aria.core.guidance_datasets import extract_perception_features, features_to_array

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        features = extract_perception_features(_identity_demos())
        probs = model.predict_proba(features_to_array(features))

        assert probs.shape == (len(SIZE_FAMILY_NAMES),)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_rank_families(self):
        from aria.core.guidance_size_model import train_size_model
        from aria.core.guidance_datasets import extract_perception_features, features_to_array

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        features = extract_perception_features(_identity_demos())
        ranking = model.rank_families(features_to_array(features))

        assert len(ranking) == len(SIZE_FAMILY_NAMES)
        assert set(ranking) == set(SIZE_FAMILY_NAMES)

    def test_save_load_roundtrip(self):
        from aria.core.guidance_size_model import train_size_model, SizeModelArtifact
        from aria.core.guidance_datasets import extract_perception_features, features_to_array

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        features = extract_perception_features(_identity_demos())
        feat_arr = features_to_array(features)
        probs_before = model.predict_proba(feat_arr)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            model.save(path)

            # Check files exist
            assert (path.with_suffix(".npz")).exists()
            assert (path.with_suffix(".json")).exists()

            # Load and compare
            loaded = SizeModelArtifact.load(path)
            assert loaded.version == model.version
            assert loaded.name == model.name
            assert loaded.feature_names == model.feature_names
            assert loaded.label_names == model.label_names

            probs_after = loaded.predict_proba(feat_arr)
            np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)

    def test_feature_label_order_stability(self):
        """Feature and label order must be stable across versions."""
        from aria.core.guidance_size_model import train_size_model

        pairs = [("t1", _identity_demos())]
        m1 = train_size_model(pairs, epochs=5)
        m2 = train_size_model(pairs, epochs=5)

        assert m1.feature_names == m2.feature_names
        assert m1.label_names == m2.label_names

    def test_empty_training_data(self):
        from aria.core.guidance_size_model import train_size_model

        model = train_size_model([])
        assert model.weights.shape[0] == len(FEATURE_NAMES)
        assert model.weights.shape[1] == len(SIZE_FAMILY_NAMES)


class TestShadowReranking:
    def test_shadow_rerank_task(self):
        from aria.core.guidance_size_model import train_size_model, shadow_rerank_task

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        result = shadow_rerank_task("t1", _identity_demos(), model)
        assert result is not None
        assert result.task_id == "t1"
        assert result.n_candidates > 0
        assert result.default_winner_rank == 0  # first verified is rank 0
        assert isinstance(result.model_winner_rank, int)
        assert isinstance(result.rank_improvement, int)

    def test_shadow_result_roundtrip(self):
        from aria.core.guidance_size_model import ShadowRankResult

        r = ShadowRankResult(
            task_id="t1", n_candidates=3,
            default_winner_mode="same_as_input",
            default_winner_family="same_dims",
            default_winner_rank=0, model_winner_rank=0,
            model_top_family="same_dims",
            rank_improvement=0, would_save_attempts=0,
            model_ranking=("same_dims", "scale"),
        )
        d = r.to_dict()
        serialized = json.dumps(d)
        r2 = ShadowRankResult.from_dict(json.loads(serialized))
        assert r2.task_id == "t1"
        assert r2.model_ranking == ("same_dims", "scale")

    def test_shadow_batch_evaluation(self):
        from aria.core.guidance_size_model import train_size_model, shadow_evaluate_batch

        pairs = [
            ("t1", _identity_demos()),
            ("t2", _identity_demos()),
            ("t3", _identity_demos()),
        ]
        model = train_size_model(pairs, epochs=10)

        report, results = shadow_evaluate_batch(pairs, model)
        assert report.n_evaluated > 0
        assert len(results) > 0
        assert 0.0 <= report.pct_improved <= 1.0
        assert 0.0 <= report.pct_same <= 1.0
        assert 0.0 <= report.pct_worse <= 1.0

    def test_shadow_report_to_dict(self):
        from aria.core.guidance_size_model import train_size_model, shadow_evaluate_batch

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=5)
        report, _ = shadow_evaluate_batch(pairs, model)

        d = report.to_dict()
        serialized = json.dumps(d)
        rt = json.loads(serialized)
        assert "avg_rank_improvement" in rt
        assert "pct_improved" in rt


class TestRealARCShadow:
    @pytest.fixture
    def arc_pairs(self):
        try:
            from aria.datasets import get_dataset, list_task_ids, load_arc_task
            ds = get_dataset("v1-train")
            task_ids = list_task_ids(ds)[:30]
            pairs = []
            for tid in task_ids:
                task = load_arc_task(ds, tid)
                pairs.append((tid, task.train))
            return pairs
        except Exception:
            pytest.skip("ARC data not available")

    def test_train_and_shadow_on_real_data(self, arc_pairs):
        from aria.core.guidance_size_model import train_size_model, shadow_evaluate_batch

        # Split: 20 train, 10 eval
        train = arc_pairs[:20]
        held_out = arc_pairs[20:]

        model = train_size_model(train, epochs=50)
        report, results = shadow_evaluate_batch(held_out, model)

        assert report.n_evaluated > 0
        # Model should not make things dramatically worse
        assert report.pct_worse < 0.5


class TestNoTaskIdLogic:
    def test_no_task_id_dispatch(self):
        path = Path(__file__).parent.parent / "aria" / "core" / "guidance_size_model.py"
        content = path.read_text()
        patterns = [r'if\s+.*task_id\s*==', r'if\s+.*task_id\s+in\b']
        for i, line in enumerate(content.split("\n"), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for p in patterns:
                if "task_id" in s and re.search(p, s):
                    pytest.fail(f"guidance_size_model.py:{i}: {s}")


class TestNoRegressions:
    def test_export_unchanged(self):
        from aria.core.guidance_export import export_task, validate_record
        record = export_task("t1", _identity_demos())
        assert not validate_record(record)

    def test_infer_output_size_unchanged(self):
        from aria.core.output_size import infer_output_size_spec
        spec = infer_output_size_spec(_identity_demos())
        assert spec is not None
        assert spec.mode == "same_as_input"
