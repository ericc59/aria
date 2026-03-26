"""Tests for aria.local_policy – interface contract and baseline behaviour."""

from __future__ import annotations

import pytest

from aria.local_policy import (
    ActionRanking,
    FocusPrediction,
    HeuristicBaselinePolicy,
    LocalCausalLMPolicy,
    LocalPolicy,
    PolicyInput,
    Sketch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> PolicyInput:
    defaults = dict(
        task_signatures=(),
        round_index=0,
        prior_focuses=(),
        prior_error_types=(),
    )
    defaults.update(overrides)
    return PolicyInput(**defaults)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestLocalPolicyContract:
    """Any LocalPolicy subclass must return the right types."""

    @pytest.fixture(params=["heuristic", "local_lm_dry"])
    def policy(self, request) -> LocalPolicy:
        if request.param == "heuristic":
            return HeuristicBaselinePolicy()
        return LocalCausalLMPolicy(dry_run=True)

    def test_predict_next_focus_returns_focus_prediction(self, policy: LocalPolicy):
        result = policy.predict_next_focus(_make_input())
        assert isinstance(result, FocusPrediction)
        assert isinstance(result.focus, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_rank_actions_returns_action_ranking(self, policy: LocalPolicy):
        candidates = ["fill_region", "overlay", "recolor"]
        result = policy.rank_actions(_make_input(), candidates)
        assert isinstance(result, ActionRanking)
        assert set(result.ranked_actions) == set(candidates)

    def test_emit_sketch_returns_none_or_sketch(self, policy: LocalPolicy):
        result = policy.emit_sketch(_make_input())
        assert result is None or isinstance(result, Sketch)


# ---------------------------------------------------------------------------
# Heuristic baseline specifics
# ---------------------------------------------------------------------------

class TestHeuristicBaseline:

    def test_marker_geometry_focus(self):
        policy = HeuristicBaselinePolicy()
        inp = _make_input(
            task_signatures=("change:additive", "dims:same", "role:has_marker"),
        )
        pred = policy.predict_next_focus(inp)
        assert pred.focus == "marker_geometry"

    def test_color_map_focus(self):
        policy = HeuristicBaselinePolicy()
        inp = _make_input(
            task_signatures=("color:new_in_output", "dims:same"),
        )
        pred = policy.predict_next_focus(inp)
        assert pred.focus == "color_map"

    def test_size_focus_when_dims_differ(self):
        policy = HeuristicBaselinePolicy()
        inp = _make_input(task_signatures=("change:recolor",))
        pred = policy.predict_next_focus(inp)
        assert pred.focus == "size"

    def test_generic_fallback(self):
        policy = HeuristicBaselinePolicy()
        inp = _make_input(task_signatures=("dims:same",))
        pred = policy.predict_next_focus(inp)
        assert pred.focus == "generic"

    def test_escalates_to_generic_after_non_generic(self):
        policy = HeuristicBaselinePolicy()
        inp = _make_input(
            task_signatures=("change:additive", "dims:same", "role:has_marker"),
            prior_focuses=("marker_geometry",),
        )
        pred = policy.predict_next_focus(inp)
        assert pred.focus == "generic"

    def test_rank_preserves_order(self):
        policy = HeuristicBaselinePolicy()
        candidates = ["c", "a", "b"]
        result = policy.rank_actions(_make_input(), candidates)
        assert list(result.ranked_actions) == candidates

    def test_emit_sketch_is_none(self):
        policy = HeuristicBaselinePolicy()
        assert policy.emit_sketch(_make_input()) is None


# ---------------------------------------------------------------------------
# LocalCausalLMPolicy dry-run
# ---------------------------------------------------------------------------

class TestLocalCausalLMDryRun:

    def test_not_loaded_in_dry_run(self):
        p = LocalCausalLMPolicy(dry_run=True)
        assert not p.is_loaded

    def test_focus_returns_generic(self):
        p = LocalCausalLMPolicy(dry_run=True)
        pred = p.predict_next_focus(_make_input())
        assert pred.focus == "generic"
        assert pred.confidence == 0.0

    def test_rank_preserves_input_order(self):
        p = LocalCausalLMPolicy(dry_run=True)
        cands = ["x", "y", "z"]
        result = p.rank_actions(_make_input(), cands)
        assert list(result.ranked_actions) == cands

    def test_emit_sketch_none(self):
        p = LocalCausalLMPolicy(dry_run=True)
        assert p.emit_sketch(_make_input()) is None

    def test_real_load_requires_transformers(self):
        with pytest.raises((ImportError, OSError)):
            LocalCausalLMPolicy(model_name_or_path="nonexistent", dry_run=False)
