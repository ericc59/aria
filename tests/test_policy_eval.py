"""Tests for aria.policy_eval – evaluator metrics and report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aria.local_policy import HeuristicBaselinePolicy, PolicyInput
from aria.policy_eval import (
    EvalExample,
    EvalReport,
    evaluate,
    focus_accuracy,
    load_eval_examples,
    mean_reciprocal_rank,
    top_k_accuracy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _example(
    *,
    sigs: tuple[str, ...] = (),
    round_index: int = 0,
    gold_focus: str | None = None,
    gold_ranking: tuple[str, ...] | None = None,
    candidate_ops: tuple[str, ...] = (),
) -> EvalExample:
    return EvalExample(
        policy_input=PolicyInput(
            task_signatures=sigs,
            round_index=round_index,
            prior_focuses=(),
            prior_error_types=(),
            candidate_ops=candidate_ops,
        ),
        gold_focus=gold_focus,
        gold_ranking=gold_ranking,
        candidate_ops=candidate_ops,
    )


@pytest.fixture
def marker_example() -> EvalExample:
    return _example(
        sigs=("change:additive", "dims:same", "role:has_marker"),
        gold_focus="marker_geometry",
    )


@pytest.fixture
def color_example() -> EvalExample:
    return _example(
        sigs=("color:new_in_output", "dims:same"),
        gold_focus="color_map",
    )


@pytest.fixture
def ranking_example() -> EvalExample:
    return _example(
        sigs=("dims:same",),
        gold_focus="generic",
        gold_ranking=("overlay", "fill_region", "recolor"),
        candidate_ops=("recolor", "fill_region", "overlay"),
    )


# ---------------------------------------------------------------------------
# focus_accuracy
# ---------------------------------------------------------------------------

class TestFocusAccuracy:

    def test_perfect_score(self, marker_example, color_example):
        policy = HeuristicBaselinePolicy()
        acc = focus_accuracy(policy, [marker_example, color_example])
        assert acc == 1.0

    def test_no_focus_examples(self):
        policy = HeuristicBaselinePolicy()
        ex = _example()  # no gold_focus
        assert focus_accuracy(policy, [ex]) == 0.0

    def test_partial(self, marker_example):
        policy = HeuristicBaselinePolicy()
        wrong = _example(sigs=("dims:same",), gold_focus="size")
        acc = focus_accuracy(policy, [marker_example, wrong])
        assert acc == 0.5


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

class TestRankingMetrics:

    def test_mrr_empty(self):
        policy = HeuristicBaselinePolicy()
        assert mean_reciprocal_rank(policy, []) == 0.0

    def test_mrr_passthrough(self, ranking_example):
        """Heuristic preserves input order, so gold[0]='overlay' position depends on candidate order."""
        policy = HeuristicBaselinePolicy()
        mrr = mean_reciprocal_rank(policy, [ranking_example])
        # candidates = ("recolor", "fill_region", "overlay"), gold[0] = "overlay" → rank 3
        assert mrr == pytest.approx(1 / 3)

    def test_top_k(self, ranking_example):
        policy = HeuristicBaselinePolicy()
        assert top_k_accuracy(policy, [ranking_example], k=3) == 1.0
        assert top_k_accuracy(policy, [ranking_example], k=2) == 0.0


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

class TestLoadExamples:

    def test_roundtrip(self, marker_example, ranking_example):
        records = [
            {
                "task_signatures": list(marker_example.policy_input.task_signatures),
                "round_index": 0,
                "prior_focuses": [],
                "prior_error_types": [],
                "gold_focus": "marker_geometry",
            },
            {
                "task_signatures": list(ranking_example.policy_input.task_signatures),
                "round_index": 0,
                "prior_focuses": [],
                "prior_error_types": [],
                "gold_ranking": list(ranking_example.gold_ranking),
                "candidate_ops": list(ranking_example.candidate_ops),
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name

        loaded = load_eval_examples(path)
        assert len(loaded) == 2
        assert loaded[0].gold_focus == "marker_geometry"
        assert loaded[1].gold_ranking == ("overlay", "fill_region", "recolor")
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_report_fields(self, marker_example, color_example, ranking_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [marker_example, color_example, ranking_example])
        assert isinstance(report, EvalReport)
        assert report.n_examples == 3
        assert report.n_focus_examples == 3  # ranking_example also has gold_focus
        assert report.n_ranking_examples == 1
        assert report.focus_accuracy == 1.0

    def test_report_summary_is_string(self, marker_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [marker_example])
        assert isinstance(report.summary(), str)
        assert "focus_accuracy" in report.summary()

    def test_report_to_dict(self, marker_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [marker_example])
        d = report.to_dict()
        assert "focus_accuracy" in d
        assert "mrr" in d
