"""Tests for aria.policy_eval – evaluator metrics, report, and scorecard."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aria.local_policy import HeuristicBaselinePolicy, PolicyInput
from aria.policy_eval import (
    EvalExample,
    EvalReport,
    Scorecard,
    _normalized_edit_distance,
    compare_scorecards,
    edit_recovery_exact_match,
    edit_recovery_similarity,
    evaluate,
    focus_accuracy,
    load_eval_examples,
    mean_reciprocal_rank,
    ndcg,
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
# NDCG
# ---------------------------------------------------------------------------

class TestNDCG:

    def test_ndcg_empty(self):
        policy = HeuristicBaselinePolicy()
        assert ndcg(policy, []) == 0.0

    def test_ndcg_perfect_preserves_order(self):
        """If gold ranking matches candidate order, NDCG should be 1.0."""
        policy = HeuristicBaselinePolicy()
        # Heuristic preserves input order, so if candidates == gold ranking, NDCG = 1
        ex = _example(
            sigs=("dims:same",),
            gold_ranking=("a", "b", "c"),
            candidate_ops=("a", "b", "c"),
        )
        result = ndcg(policy, [ex])
        assert result == pytest.approx(1.0)

    def test_ndcg_reversed_is_less_than_one(self):
        policy = HeuristicBaselinePolicy()
        ex = _example(
            sigs=("dims:same",),
            gold_ranking=("c", "b", "a"),
            candidate_ops=("a", "b", "c"),
        )
        result = ndcg(policy, [ex])
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# Edit recovery metrics
# ---------------------------------------------------------------------------

class TestEditRecovery:

    def test_exact_match_identical(self):
        assert edit_recovery_exact_match(["prog"], ["prog"]) == 1.0

    def test_exact_match_different(self):
        assert edit_recovery_exact_match(["prog_a"], ["prog_b"]) == 0.0

    def test_exact_match_empty(self):
        assert edit_recovery_exact_match([], []) == 0.0

    def test_exact_match_strips_whitespace(self):
        assert edit_recovery_exact_match(["prog  \n"], ["prog"]) == 1.0

    def test_similarity_identical(self):
        assert edit_recovery_similarity(["a\nb\nc"], ["a\nb\nc"]) == 1.0

    def test_similarity_completely_different(self):
        result = edit_recovery_similarity(["x"], ["y"])
        assert result == 0.0

    def test_similarity_partial(self):
        result = edit_recovery_similarity(["a\nb\nc"], ["a\nb\nd"])
        assert 0.0 < result < 1.0


class TestNormalizedEditDistance:

    def test_identical(self):
        assert _normalized_edit_distance("a\nb", "a\nb") == 0.0

    def test_completely_different(self):
        assert _normalized_edit_distance("a", "b") == 1.0

    def test_empty(self):
        assert _normalized_edit_distance("", "") == 0.0

    def test_partial_overlap(self):
        dist = _normalized_edit_distance("a\nb\nc", "a\nx\nc")
        assert dist == pytest.approx(1 / 3)


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

    def test_load_with_edit_fields(self):
        records = [
            {
                "task_signatures": ["dims:same"],
                "round_index": 1,
                "gold_edit_before": "prog_a",
                "gold_edit_after": "prog_b",
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name

        loaded = load_eval_examples(path)
        assert loaded[0].gold_edit_before == "prog_a"
        assert loaded[0].gold_edit_after == "prog_b"
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

    def test_report_has_ndcg(self, ranking_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [ranking_example])
        assert "ndcg" in report.to_dict()

    def test_report_has_top1(self, ranking_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [ranking_example])
        assert "top1_accuracy" in report.to_dict()

    def test_report_summary_is_string(self, marker_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [marker_example])
        assert isinstance(report.summary(), str)
        assert "focus_accuracy" in report.summary()
        assert "ndcg" in report.summary()

    def test_report_to_dict(self, marker_example):
        policy = HeuristicBaselinePolicy()
        report = evaluate(policy, [marker_example])
        d = report.to_dict()
        assert "focus_accuracy" in d
        assert "mrr" in d
        assert "ndcg" in d
        assert "top1_accuracy" in d


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class TestScorecard:

    def test_scorecard_to_dict(self):
        card = Scorecard(
            run_label="test-run",
            policy_name="heuristic",
            eval_report={"focus_accuracy": 0.8, "mrr": 0.5},
            dataset_stats={"total": 100, "by_task_type": {"NEXT_FOCUS": 50}},
        )
        d = card.to_dict()
        assert d["run_label"] == "test-run"
        assert d["eval"]["focus_accuracy"] == 0.8
        assert d["dataset"]["total"] == 100

    def test_scorecard_to_json(self):
        card = Scorecard(run_label="run1", eval_report={"mrr": 0.3})
        j = card.to_json()
        parsed = json.loads(j)
        assert parsed["eval"]["mrr"] == 0.3

    def test_scorecard_summary(self):
        card = Scorecard(
            run_label="baseline",
            policy_name="heuristic",
            eval_report={"focus_accuracy": 1.0, "mrr": 0.5, "ndcg": 0.8,
                         "top1_accuracy": 0.3, "top3_accuracy": 0.9},
            dataset_stats={"total": 200, "by_task_type": {"NEXT_FOCUS": 100},
                           "unique_task_ids": 50},
        )
        s = card.summary()
        assert "baseline" in s
        assert "focus_accuracy" in s
        assert "ndcg" in s


class TestCompareScorecard:

    def test_compare_two(self):
        c1 = Scorecard(run_label="v1", eval_report={"focus_accuracy": 0.7, "mrr": 0.3,
                        "top1_accuracy": 0.2, "top3_accuracy": 0.5, "ndcg": 0.6},
                        dataset_stats={"total": 100})
        c2 = Scorecard(run_label="v2", eval_report={"focus_accuracy": 0.9, "mrr": 0.5,
                        "top1_accuracy": 0.4, "top3_accuracy": 0.8, "ndcg": 0.85},
                        dataset_stats={"total": 200})
        table = compare_scorecards([c1, c2])
        assert "v1" in table
        assert "v2" in table
        assert "focus_accuracy" in table
        assert "ndcg" in table
        assert "dataset_total" in table

    def test_compare_empty(self):
        assert compare_scorecards([]) == "(no scorecards)"

    def test_compare_single(self):
        c = Scorecard(run_label="solo", eval_report={"focus_accuracy": 1.0, "mrr": 1.0,
                       "top1_accuracy": 1.0, "top3_accuracy": 1.0, "ndcg": 1.0},
                       dataset_stats={"total": 50})
        table = compare_scorecards([c])
        assert "solo" in table
