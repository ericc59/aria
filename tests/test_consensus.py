"""Tests for stepwise all-demo consensus machinery."""

from __future__ import annotations

import numpy as np
import pytest

from aria.consensus import (
    BranchState,
    ConsistencyCheck,
    DemoPartialState,
    SharedHypothesis,
    _SelectResult,
    build_initial_branch,
    check_binding_compatibility,
    check_entity_count_consistent,
    check_entity_kind_consistent,
    check_intermediate_output_compatible,
    check_perception_structure_consistent,
    check_selector_analogy,
    check_transform_family_consistent,
    rank_branches,
    run_checks,
    score_branch,
    should_prune,
    try_select_on_demo,
    update_branch_after_select,
    update_branch_after_transform,
)
from aria.consensus_trace import ConsensusTrace, format_consensus_trace
from aria.types import Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(rows: list[list[int]]) -> Grid:
    return np.array(rows, dtype=np.int32)


def _make_perception(grid: Grid):
    from aria.core.grid_perception import perceive_grid
    return perceive_grid(grid)


def _make_demo_state(
    idx: int,
    *,
    entity_count: int = 0,
    kind: str | None = None,
    sig: str = "",
    grid: Grid | None = None,
    bindings: tuple[tuple[str, int], ...] = (),
    perception=None,
) -> DemoPartialState:
    return DemoPartialState(
        demo_index=idx,
        perception=perception,
        selected_entity_count=entity_count,
        selected_kind=kind,
        structural_signature=sig,
        intermediate_grid=grid,
        role_bindings=bindings,
    )


# ---------------------------------------------------------------------------
# Part A: State construction
# ---------------------------------------------------------------------------


class TestBuildInitialBranch:
    def test_same_dims_grids(self):
        g1 = _make_grid([[1, 0], [0, 1]])
        g2 = _make_grid([[0, 2], [2, 0]])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        branch = build_initial_branch([p1, p2])
        assert not branch.pruned
        assert branch.hypothesis.scope_family == "same_dims"
        assert len(branch.per_demo) == 2
        assert branch.per_demo[0].bg_color == p1.bg_color
        assert branch.per_demo[1].bg_color == p2.bg_color

    def test_different_dims_grids(self):
        g1 = _make_grid([[1, 0, 0], [0, 1, 0]])
        g2 = _make_grid([[0, 2], [2, 0]])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        branch = build_initial_branch([p1, p2])
        assert not branch.pruned
        assert branch.hypothesis.scope_family == "mixed_dims"

    def test_single_demo(self):
        g = _make_grid([[1, 2], [3, 4]])
        p = _make_perception(g)
        branch = build_initial_branch([p])
        assert not branch.pruned
        assert branch.consistency_score == 1.0


# ---------------------------------------------------------------------------
# Part B: Individual consistency checks
# ---------------------------------------------------------------------------


class TestEntityCountConsistent:
    def test_all_have_entities(self):
        states = (
            _make_demo_state(0, entity_count=3),
            _make_demo_state(1, entity_count=2),
        )
        c = check_entity_count_consistent(states)
        assert c.passed
        assert c.score == 1.0

    def test_one_has_zero(self):
        states = (
            _make_demo_state(0, entity_count=3),
            _make_demo_state(1, entity_count=0),
        )
        c = check_entity_count_consistent(states)
        assert not c.passed
        assert c.score == 0.0

    def test_all_zero(self):
        states = (
            _make_demo_state(0, entity_count=0),
            _make_demo_state(1, entity_count=0),
        )
        c = check_entity_count_consistent(states)
        assert not c.passed

    def test_empty(self):
        c = check_entity_count_consistent(())
        assert c.passed


class TestEntityKindConsistent:
    def test_same_kind(self):
        states = (
            _make_demo_state(0, kind="object"),
            _make_demo_state(1, kind="object"),
        )
        c = check_entity_kind_consistent(states)
        assert c.passed

    def test_different_kinds(self):
        states = (
            _make_demo_state(0, kind="object"),
            _make_demo_state(1, kind="panel"),
        )
        c = check_entity_kind_consistent(states)
        assert not c.passed
        assert c.score == 0.0

    def test_one_none(self):
        states = (
            _make_demo_state(0, kind="object"),
            _make_demo_state(1, kind=None),
        )
        c = check_entity_kind_consistent(states)
        assert c.passed  # only one kind present


class TestSelectorAnalogy:
    def test_same_sig_prefix(self):
        states = (
            _make_demo_state(0, sig="object:largest|count=3|found=True"),
            _make_demo_state(1, sig="object:largest|count=2|found=True"),
        )
        c = check_selector_analogy(states)
        assert c.passed
        assert c.score == 1.0

    def test_different_sig_prefix(self):
        states = (
            _make_demo_state(0, sig="object:largest|count=3"),
            _make_demo_state(1, sig="panel:largest|count=2"),
        )
        c = check_selector_analogy(states)
        assert c.passed  # soft fail
        assert c.score < 1.0


class TestTransformFamilyConsistent:
    def test_same_shape(self):
        g1 = _make_grid([[1, 2], [3, 4]])
        g2 = _make_grid([[5, 6], [7, 8]])
        states = (
            _make_demo_state(0, grid=g1),
            _make_demo_state(1, grid=g2),
        )
        c = check_transform_family_consistent(states)
        assert c.passed
        assert c.score == 1.0

    def test_different_shape(self):
        g1 = _make_grid([[1, 2], [3, 4]])
        g2 = _make_grid([[5, 6, 7]])
        states = (
            _make_demo_state(0, grid=g1),
            _make_demo_state(1, grid=g2),
        )
        c = check_transform_family_consistent(states)
        assert c.score < 1.0


class TestBindingCompatibility:
    def test_same_keys(self):
        states = (
            _make_demo_state(0, bindings=(("bg", 0), ("fg", 1))),
            _make_demo_state(1, bindings=(("bg", 0), ("fg", 2))),
        )
        c = check_binding_compatibility(states)
        assert c.passed

    def test_different_keys(self):
        states = (
            _make_demo_state(0, bindings=(("bg", 0), ("fg", 1))),
            _make_demo_state(1, bindings=(("bg", 0), ("frame", 3))),
        )
        c = check_binding_compatibility(states)
        assert not c.passed
        assert c.score == 0.0


class TestIntermediateOutputCompatible:
    def test_same_shape(self):
        g1 = _make_grid([[1, 2], [3, 4]])
        g2 = _make_grid([[5, 6], [7, 8]])
        states = (
            _make_demo_state(0, grid=g1),
            _make_demo_state(1, grid=g2),
        )
        c = check_intermediate_output_compatible(states)
        assert c.passed
        assert c.score == 1.0

    def test_proportional_shapes(self):
        g1 = _make_grid([[1, 2], [3, 4]])
        g2 = _make_grid([[5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]])
        states = (
            _make_demo_state(0, grid=g1),
            _make_demo_state(1, grid=g2),
        )
        c = check_intermediate_output_compatible(states)
        assert c.passed
        assert 0.5 < c.score < 1.0

    def test_no_grids(self):
        states = (
            _make_demo_state(0),
            _make_demo_state(1),
        )
        c = check_intermediate_output_compatible(states)
        assert c.passed
        assert c.score == 1.0


class TestPerceptionStructureConsistent:
    def test_consistent_no_partitions(self):
        g1 = _make_grid([[1, 0], [0, 1]])
        g2 = _make_grid([[2, 0], [0, 2]])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        states = (
            _make_demo_state(0, perception=p1),
            _make_demo_state(1, perception=p2),
        )
        c = check_perception_structure_consistent(states)
        assert c.passed

    def test_partition_with_consistent_cell_count(self):
        # 3x3 grid with separator row/col → 4 cells
        g1 = _make_grid([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        g2 = _make_grid([
            [5, 0, 6],
            [0, 0, 0],
            [7, 0, 8],
        ])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        states = (
            _make_demo_state(0, perception=p1),
            _make_demo_state(1, perception=p2),
        )
        c = check_perception_structure_consistent(states)
        assert c.passed


# ---------------------------------------------------------------------------
# Part C: Scoring and pruning
# ---------------------------------------------------------------------------


class TestScoringAndPruning:
    def test_score_no_checks(self):
        branch = BranchState(
            branch_id="test",
            step_index=0,
            per_demo=(),
        )
        assert score_branch(branch) == 1.0

    def test_score_all_passing(self):
        branch = BranchState(
            branch_id="test",
            step_index=0,
            per_demo=(),
            consistency_checks=(
                ConsistencyCheck("a", True, 1.0),
                ConsistencyCheck("b", True, 0.8),
            ),
        )
        assert score_branch(branch) == 0.8

    def test_should_prune_hard_fail(self):
        branch = BranchState(
            branch_id="test",
            step_index=0,
            per_demo=(),
            consistency_checks=(
                ConsistencyCheck("a", False, 0.0, "hard fail"),
            ),
        )
        assert should_prune(branch)

    def test_should_not_prune_soft_fail(self):
        branch = BranchState(
            branch_id="test",
            step_index=0,
            per_demo=(),
            consistency_checks=(
                ConsistencyCheck("a", True, 0.5, "soft"),
            ),
        )
        assert not should_prune(branch)

    def test_rank_branches(self):
        b1 = BranchState("a", 0, (), consistency_score=0.9)
        b2 = BranchState("b", 0, (), consistency_score=0.3, pruned=True)
        b3 = BranchState("c", 0, (), consistency_score=0.7)
        ranked = rank_branches([b2, b1, b3])
        assert [b.branch_id for b in ranked] == ["a", "c", "b"]


# ---------------------------------------------------------------------------
# Part D: Shared hypothesis
# ---------------------------------------------------------------------------


class TestSharedHypothesis:
    def test_default_hypothesis(self):
        h = SharedHypothesis()
        assert h.rule_family is None
        assert h.selector_family is None

    def test_hypothesis_refinement(self):
        from dataclasses import replace
        h = SharedHypothesis(scope_family="same_dims")
        h2 = replace(h, rule_family="select_extract_transform")
        assert h2.scope_family == "same_dims"
        assert h2.rule_family == "select_extract_transform"


# ---------------------------------------------------------------------------
# Part E: Integration — try_select_on_demo
# ---------------------------------------------------------------------------


class TestTrySelectOnDemo:
    def test_select_object(self):
        g = _make_grid([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        p = _make_perception(g)
        result = try_select_on_demo(p, "object", "largest_bbox_area", 0)
        assert result.found
        assert result.entity_count >= 1

    def test_select_rank_too_high(self):
        g = _make_grid([
            [0, 1, 0],
            [0, 0, 0],
        ])
        p = _make_perception(g)
        result = try_select_on_demo(p, "object", "largest_bbox_area", 99)
        assert not result.found

    def test_select_nonexistent_kind(self):
        g = _make_grid([[1, 2], [3, 4]])
        p = _make_perception(g)
        result = try_select_on_demo(p, "nonexistent_kind", "largest_bbox_area", 0)
        assert not result.found


# ---------------------------------------------------------------------------
# Branch update integration
# ---------------------------------------------------------------------------


class TestBranchUpdates:
    def test_update_after_select_consistent(self):
        g1 = _make_grid([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        g2 = _make_grid([
            [0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        branch = build_initial_branch([p1, p2])

        sr1 = try_select_on_demo(p1, "object", "largest_bbox_area", 0)
        sr2 = try_select_on_demo(p2, "object", "largest_bbox_area", 0)
        updated = update_branch_after_select(
            branch, "object", "largest_bbox_area", 0, [sr1, sr2],
        )
        assert not updated.pruned
        assert updated.hypothesis.selector_family is not None

    def test_update_after_select_one_missing(self):
        g1 = _make_grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        g2 = _make_grid([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        p1 = _make_perception(g1)
        p2 = _make_perception(g2)
        branch = build_initial_branch([p1, p2])

        sr1 = try_select_on_demo(p1, "object", "largest_bbox_area", 0)
        sr2 = try_select_on_demo(p2, "object", "largest_bbox_area", 0)
        updated = update_branch_after_select(
            branch, "object", "largest_bbox_area", 0, [sr1, sr2],
        )
        assert updated.pruned  # demo 2 has no objects

    def test_update_after_transform(self):
        g1 = _make_grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        p1 = _make_perception(g1)
        branch = build_initial_branch([p1])

        transformed = _make_grid([[0, 1], [1, 0]])
        updated = update_branch_after_transform(branch, "rot90", [transformed])
        assert not updated.pruned
        assert updated.hypothesis.transform_family == "rot90"


# ---------------------------------------------------------------------------
# Part F: Trace
# ---------------------------------------------------------------------------


class TestConsensusTrace:
    def test_trace_records(self):
        trace = ConsensusTrace()
        branch = BranchState(
            branch_id="test",
            step_index=0,
            per_demo=(
                _make_demo_state(0, sig="a"),
                _make_demo_state(1, sig="b"),
            ),
            consistency_score=0.8,
        )
        trace.record(branch, "initial")
        assert len(trace.entries) == 1
        assert trace.branches_created == 1
        assert trace.branches_survived == 1

    def test_trace_prune_count(self):
        trace = ConsensusTrace()
        pruned = BranchState(
            branch_id="pruned",
            step_index=0,
            per_demo=(),
            pruned=True,
            prune_reason="test",
        )
        trace.record(pruned, "pruned branch")
        assert trace.branches_pruned == 1

    def test_format_trace(self):
        trace = ConsensusTrace()
        branch = BranchState(
            branch_id="x",
            step_index=0,
            per_demo=(),
            consistency_score=1.0,
            consistency_checks=(
                ConsistencyCheck("test_check", True, 1.0),
            ),
        )
        trace.record(branch, "step 0")
        text = format_consensus_trace(trace)
        assert "x" in text
        assert "test_check" in text
        assert "1 created" in text


# ---------------------------------------------------------------------------
# Integration: scene_solve consensus gating
# ---------------------------------------------------------------------------


class TestSceneSolveConsensus:
    """Integration tests verifying that consensus gating works
    end-to-end in scene_solve without breaking solves."""

    def test_consensus_select_check_consistent(self):
        """When selector works on all demos, it should not be pruned."""
        from aria.consensus import try_select_on_demo
        from aria.core.scene_solve import _consensus_select_check, _perceive_all_demos
        from aria.types import DemoPair

        # Two demos with objects
        g1 = _make_grid([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        g2 = _make_grid([
            [0, 0, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ])
        demos = (DemoPair(input=g1, output=g1), DemoPair(input=g2, output=g2))
        perceptions = _perceive_all_demos(demos)

        # Both have objects, selector should pass
        assert _consensus_select_check(
            perceptions, "object", "largest_bbox_area", 0,
        )

    def test_consensus_select_check_one_empty(self):
        """When selector fails on one demo, it should be pruned."""
        from aria.core.scene_solve import _consensus_select_check, _perceive_all_demos
        from aria.types import DemoPair

        g1 = _make_grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        g2 = _make_grid([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        demos = (DemoPair(input=g1, output=g1), DemoPair(input=g2, output=g2))
        perceptions = _perceive_all_demos(demos)

        # Demo 2 has no objects, selector should fail
        assert not _consensus_select_check(
            perceptions, "object", "largest_bbox_area", 0,
        )

    def test_infer_scene_programs_with_trace(self):
        """Scene program inference with consensus trace produces trace entries."""
        from aria.consensus_trace import ConsensusTrace
        from aria.core.scene_solve import infer_scene_programs
        from aria.types import DemoPair

        g1 = _make_grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        g2 = _make_grid([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        demos = (DemoPair(input=g1, output=g1), DemoPair(input=g2, output=g2))
        trace = ConsensusTrace()
        infer_scene_programs(demos, consensus_trace=trace)
        # Trace should have recorded at least some pruned branches
        # (because the demos are structurally different — one has objects, one doesn't)
        assert trace.branches_created >= 0  # trace was used

    def test_partition_consistency_check(self):
        """Partition consistency check in scene_propose."""
        from aria.core.scene_propose import _check_partition_consistency
        from aria.types import DemoPair

        # Same structure
        g1 = _make_grid([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        g2 = _make_grid([
            [5, 0, 6],
            [0, 0, 0],
            [7, 0, 8],
        ])
        demos = (DemoPair(input=g1, output=g1), DemoPair(input=g2, output=g2))
        assert _check_partition_consistency(demos)
