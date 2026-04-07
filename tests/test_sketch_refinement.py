"""Tests for sketch-oriented refinement integration."""

from __future__ import annotations

import numpy as np

from aria.library.store import Library
from aria.refinement import (
    RefinementResult,
    SketchRefinementResult,
    _run_sketch_refinement,
    run_refinement_loop,
)
from aria.sketch_compile import CompileFailure, CompilePerDemoPrograms, CompileTaskProgram
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _identity_task():
    """Identity task — solved by synthesis, sketch path not needed."""
    return (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )


def _composite_alignment_task():
    """Task with composites aligning to anchor. Colors rotate."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 8, 4, 8, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 8, 4, 8, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 1, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 3, 1, 3, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
        ),
    )


def _framed_periodic_task():
    """Task with periodic content inside frames."""
    return (
        DemoPair(
            input=grid_from_list([
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 2, 1, 3, 1, 3, 1, 3, 3, 3, 1, 2, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]),
            output=grid_from_list([
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 4, 2, 5, 2, 5, 2, 5, 5, 5, 2, 4, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]),
            output=grid_from_list([
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 4, 2, 5, 2, 5, 2, 5, 2, 5, 2, 4, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# _run_sketch_refinement unit tests
# ---------------------------------------------------------------------------


def test_sketch_refinement_proposes_composite():
    """Composite alignment task should get a sketch proposed."""
    demos = _composite_alignment_task()
    result = _run_sketch_refinement(demos)
    assert isinstance(result, SketchRefinementResult)
    assert result.sketches_proposed >= 1
    assert "composite_role_alignment" in result.sketch_families


def test_sketch_refinement_proposes_periodic():
    """Periodic task should get a sketch proposed."""
    demos = _framed_periodic_task()
    result = _run_sketch_refinement(demos)
    assert result.sketches_proposed >= 1
    assert "framed_periodic_repair" in result.sketch_families


def test_sketch_refinement_composite_not_promoted():
    """Composite alignment produces per-demo programs, NOT task-level solve."""
    demos = _composite_alignment_task()
    result = _run_sketch_refinement(demos)
    # Per-demo specialization should not be marked as solved
    assert result.solved is False
    assert result.winning_program is None
    # But should have compiled programs and verified some
    assert result.sketch_compiled >= 1
    assert result.sketch_verified >= 1


def test_sketch_refinement_periodic_solves():
    """Periodic repair should compile to a task-level solve."""
    demos = _framed_periodic_task()
    result = _run_sketch_refinement(demos)
    assert result.solved is True
    assert result.winning_program is not None
    assert result.winning_family == "framed_periodic_repair"


def test_sketch_refinement_reports_budget():
    """Budget tracking should be non-zero."""
    demos = _composite_alignment_task()
    result = _run_sketch_refinement(demos)
    assert result.sketch_budget_used > 0
    assert result.sketch_candidates_executed > 0


def test_sketch_refinement_empty_for_identity():
    """Identity task should have 0 sketches proposed."""
    demos = _identity_task()
    result = _run_sketch_refinement(demos)
    assert result.sketches_proposed == 0


# ---------------------------------------------------------------------------
# Integration: sketch in full refinement loop
# ---------------------------------------------------------------------------


def test_refinement_has_sketch_result_field():
    """RefinementResult should have sketch_result field."""
    demos = _identity_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    assert hasattr(result, "sketch_result")
    # May be None if task solved before sketch phase (e.g., by synthesis)


def test_refinement_sketch_result_populated_when_unsolved():
    """sketch_result should be populated for tasks that reach the sketch phase."""
    demos = _composite_alignment_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    assert result.sketch_result is not None
    assert result.sketch_result.sketches_proposed >= 1


def test_refinement_sketch_does_not_regress_synthesis():
    """Adding sketch phase should not prevent synthesis from solving identity."""
    demos = _identity_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    # Identity task is solved by synthesis (before sketches)
    assert result.solved


def test_refinement_sketch_composite_falls_through():
    """Composite task: sketch runs but doesn't solve (per-demo only).
    Legacy phases should still run."""
    demos = _composite_alignment_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    assert result.sketch_result is not None
    assert result.sketch_result.sketches_proposed >= 1
    # Sketch didn't solve (per-demo), so legacy search should have run
    assert result.candidates_tried > result.sketch_result.sketch_budget_used


def test_refinement_sketch_periodic_solves_via_sketch():
    """Periodic task: sketch compiles and solves — no legacy search needed."""
    demos = _framed_periodic_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    assert result.solved is True
    assert result.sketch_result is not None
    assert result.sketch_result.solved is True
    assert result.sketch_result.winning_family == "framed_periodic_repair"
    # No search rounds should have run (solved before search)
    assert len(result.rounds) == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_sketch_result_fields():
    """SketchRefinementResult should have all reporting fields."""
    demos = _composite_alignment_task()
    result = _run_sketch_refinement(demos)
    assert isinstance(result.sketches_proposed, int)
    assert isinstance(result.sketch_families, tuple)
    assert isinstance(result.sketch_compiled, int)
    assert isinstance(result.sketch_compile_failures, int)
    assert isinstance(result.sketch_verified, int)
    assert isinstance(result.sketch_budget_used, int)
    assert isinstance(result.sketch_candidates_executed, int)
    assert isinstance(result.solved, bool)
    assert isinstance(result.compile_results, tuple)


# ---------------------------------------------------------------------------
# Graph-native compilation in solver loop
# ---------------------------------------------------------------------------


def test_periodic_uses_graph_native_path():
    """Periodic task should compile via graph-native path, not fallback."""
    demos = _framed_periodic_task()
    result = _run_sketch_refinement(demos)
    assert result.solved is True
    assert result.graph_native_attempted >= 1
    assert result.graph_native_compiled >= 1
    assert result.graph_native_verified >= 1
    assert result.solve_path == "graph_native"
    assert result.fallback_used == 0


def test_composite_uses_graph_native_path():
    """Composite alignment should compile via graph-native path."""
    demos = _composite_alignment_task()
    result = _run_sketch_refinement(demos)
    assert result.graph_native_attempted >= 1
    assert result.graph_native_compiled >= 1
    # Per-demo programs should be verified via graph-native
    assert result.graph_native_verified >= 1
    assert result.fallback_used == 0


def test_identity_reports_no_graph_native():
    """Identity task: no sketches proposed, no graph-native attempts."""
    demos = _identity_task()
    result = _run_sketch_refinement(demos)
    assert result.graph_native_attempted == 0
    assert result.graph_native_compiled == 0
    assert result.fallback_used == 0
    assert result.solve_path == "none"


def test_solve_path_is_graph_native_for_periodic():
    """When periodic solves, solve_path should be 'graph_native'."""
    demos = _framed_periodic_task()
    result = _run_sketch_refinement(demos)
    assert result.solve_path == "graph_native"


def test_graph_native_result_in_full_refinement():
    """Graph-native reporting propagates through full refinement loop."""
    demos = _framed_periodic_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    assert result.solved is True
    assert result.sketch_result is not None
    assert result.sketch_result.solve_path == "graph_native"
    assert result.sketch_result.graph_native_compiled >= 1


def _tile_task():
    """Canvas task: tile 2x2 grid 2x2."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 6, 5, 6], [7, 8, 7, 8],
                [5, 6, 5, 6], [7, 8, 7, 8],
            ]),
        ),
    )


def test_canvas_uses_graph_native_in_solver():
    """Canvas construction should compile via graph-native path in the solver loop."""
    demos = _tile_task()
    result = _run_sketch_refinement(demos)
    assert result.solved is True
    assert result.graph_native_attempted >= 1
    assert result.graph_native_compiled >= 1
    assert result.graph_native_verified >= 1
    assert result.solve_path == "graph_native"
    assert result.fallback_used == 0


def test_canvas_solves_in_full_refinement():
    """Canvas task solves through the full refinement loop.

    Note: the legacy dims-change path may solve this before the sketch
    phase runs, in which case sketch_result will be None. Either way,
    the task is solved.
    """
    demos = _tile_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    assert result.solved is True
    # If sketch phase ran, it should use graph-native
    if result.sketch_result is not None and result.sketch_result.solved:
        assert result.sketch_result.solve_path == "graph_native"
