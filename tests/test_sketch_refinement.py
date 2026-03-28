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


def test_sketch_refinement_periodic_compile_failure():
    """Periodic repair should produce a compile failure, not a solve."""
    demos = _framed_periodic_task()
    result = _run_sketch_refinement(demos)
    assert result.solved is False
    assert result.sketch_compile_failures >= 1
    # Should have the CompileFailure in compile_results
    failures = [r for r in result.compile_results if isinstance(r, CompileFailure)]
    assert len(failures) >= 1
    assert "repair_periodic" in failures[0].missing_ops


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


def test_refinement_sketch_periodic_falls_through():
    """Periodic task: sketch compile fails, legacy phases still run.
    Note: the task may be solved by search, but sketch_result should still
    be populated because the sketch phase runs before search."""
    demos = _framed_periodic_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=50, max_rounds=1,
    )
    # Sketch phase ran before search
    assert result.sketch_result is not None
    assert result.sketch_result.sketch_compile_failures >= 1
    # Total candidates should include both sketch budget and search budget
    assert result.candidates_tried >= result.sketch_result.sketch_budget_used


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
