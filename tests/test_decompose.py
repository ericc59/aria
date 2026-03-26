"""Tests for bidirectional task decomposition."""

from __future__ import annotations

from aria.decompose import DecompPlan, SubGoal, decompose_task
from aria.types import DemoPair, Type, grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _same_size_color_change_task():
    """Same dims, different colors → should detect color_transform."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[5, 6], [7, 8]]),
        ),
    )


def _dim_change_task():
    """Different dims in every demo → should detect change_dims."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3], [4, 5, 6]]),
            output=grid_from_list([[1, 4], [2, 5], [3, 6]]),
        ),
        DemoPair(
            input=grid_from_list([[7, 8, 9], [0, 1, 2]]),
            output=grid_from_list([[7, 0], [8, 1], [9, 2]]),
        ),
    )


def _same_size_additive_task():
    """Same dims, objects added → should detect construct_new + compose."""
    return (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        ),
    )


def _identity_task():
    """Input == output → content_changed should be False."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------


def test_decompose_detects_dim_change():
    plan = decompose_task(_dim_change_task())
    assert isinstance(plan, DecompPlan)
    assert "dims_change" in plan.evidence
    names = {sg.name for sg in plan.sub_goals}
    assert "change_dims" in names


def test_decompose_detects_color_change():
    plan = decompose_task(_same_size_color_change_task())
    assert "palette_changed" in plan.evidence
    names = {sg.name for sg in plan.sub_goals}
    assert "color_transform" in names


def test_decompose_same_size_content_changed():
    plan = decompose_task(_same_size_additive_task())
    assert "same_dims_content_changed" in plan.evidence
    names = {sg.name for sg in plan.sub_goals}
    assert "compose_grid" in names


def test_decompose_identity_task():
    plan = decompose_task(_identity_task())
    # Identity: nothing changed, should produce a generic fallback
    assert plan.final_type == Type.GRID
    # No strong evidence for any sub-goal
    assert "generic_fallback" in plan.evidence or len(plan.sub_goals) == 0


def test_decompose_produces_candidate_ops():
    plan = decompose_task(_dim_change_task())
    ops = plan.candidate_op_names()
    assert isinstance(ops, frozenset)
    assert len(ops) > 0


def test_decompose_to_dict():
    plan = decompose_task(_same_size_color_change_task())
    d = plan.to_dict()
    assert "sub_goals" in d
    assert "evidence" in d
    assert isinstance(d["sub_goals"], list)
    for sg in d["sub_goals"]:
        assert "name" in sg
        assert "output_type" in sg
        assert "candidate_ops" in sg


def test_decompose_consistent_across_demos():
    """Decomposition from multiple demos should merge observations."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[5, 6], [7, 8]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 1], [2, 3]]),
            output=grid_from_list([[4, 5], [6, 7]]),
        ),
    )
    plan = decompose_task(demos)
    # Both demos have same dims, palette changed → should be consistent
    assert "palette_changed" in plan.evidence
    assert "same_dims_content_changed" in plan.evidence


# ---------------------------------------------------------------------------
# Integration with refinement
# ---------------------------------------------------------------------------


def test_refinement_carries_decomposition():
    from aria.library.store import Library
    from aria.refinement import run_refinement_loop

    demos = _dim_change_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    assert result.decomposition is not None
    assert isinstance(result.decomposition, DecompPlan)
    assert len(result.decomposition.evidence) > 0


def test_decomposition_ops_boost_search():
    """Decomposition candidate ops should appear in preferred_ops."""
    from aria.library.store import Library
    from aria.refinement import run_refinement_loop

    demos = _same_size_color_change_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    # The decomposition should have detected color_transform
    assert result.decomposition is not None
    decomp_ops = result.decomposition.candidate_op_names()
    # These ops should include color-related ops
    color_ops = {"apply_color_map", "recolor", "conditional_fill"}
    assert len(decomp_ops & color_ops) > 0


def test_inspection_includes_decomposition():
    from aria.inspection import inspect_task
    from aria.library.store import Library
    from aria.program_store import ProgramStore

    demos = _dim_change_task()
    inspection = inspect_task(
        demos,
        library=Library(),
        program_store=ProgramStore(),
        retrieval_limit=0,
        max_search_steps=1,
        max_search_candidates=10,
        max_refinement_rounds=1,
        search_trace_limit=5,
    )
    assert "decomposition" in inspection
    decomp = inspection["decomposition"]
    assert "sub_goals" in decomp
    assert "evidence" in decomp
    assert "dims_change" in decomp["evidence"]
