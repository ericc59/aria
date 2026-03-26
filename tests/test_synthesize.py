"""Tests for observation-driven program synthesis."""

from __future__ import annotations

from aria.synthesize import Observation, SynthesisResult, synthesize_from_observations
from aria.types import DemoPair, grid_from_list


def test_synthesis_solves_transpose():
    """Transpose is a direct single-op transform — should be found immediately."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 0], [3, 4, 0]]),
            output=grid_from_list([[1, 3], [2, 4], [0, 0]]),
        ),
    )
    result = synthesize_from_observations(demos)
    assert result.solved
    assert result.winning_program is not None
    assert result.candidates_tested < 20  # should find quickly


def test_synthesis_solves_reflect_h():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6, 7], [8, 9, 0]]),
            output=grid_from_list([[8, 9, 0], [5, 6, 7]]),
        ),
    )
    result = synthesize_from_observations(demos)
    assert result.solved
    assert result.candidates_tested < 50


def test_synthesis_solves_color_map():
    """Pure color mapping should be found by color map inference."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 1]]),
            output=grid_from_list([[4, 5], [6, 4]]),
        ),
        DemoPair(
            input=grid_from_list([[2, 1], [1, 3]]),
            output=grid_from_list([[5, 4], [4, 6]]),
        ),
    )
    result = synthesize_from_observations(demos)
    assert result.solved
    # Should have an observation of kind "color_map"
    color_obs = [o for o in result.observations if o.kind == "color_map"]
    assert len(color_obs) >= 1


def test_synthesis_solves_rotate_90():
    """90-degree rotation is a (literal, GRID) transform."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 1], [4, 2]]),
        ),
    )
    result = synthesize_from_observations(demos)
    assert result.solved


def test_synthesis_solves_composed_transform():
    """A task needing two steps: reflect then transpose."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 1], [4, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 5], [8, 6]]),
        ),
    )
    result = synthesize_from_observations(demos)
    # This might be found as rotate_90 or as reflect+transpose
    assert result.solved


def test_synthesis_unsolvable_returns_false():
    """Tasks beyond direct observation should fail gracefully."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]]),
        ),
    )
    result = synthesize_from_observations(demos)
    assert not result.solved
    assert result.candidates_tested > 0


def test_synthesis_observations_are_recorded():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),  # identity
        ),
    )
    result = synthesize_from_observations(demos)
    # Identity: should be found (it's just the input)
    # But our DSL doesn't have an identity op, so it may not solve
    # What matters: observations are recorded
    assert isinstance(result.observations, tuple)
    for obs in result.observations:
        assert isinstance(obs, Observation)
        assert isinstance(obs.program_text, str)
        assert isinstance(obs.source, str)


def test_synthesis_before_search_in_refinement():
    """Synthesis runs before search in the refinement loop."""
    from aria.library.store import Library
    from aria.refinement import run_refinement_loop

    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 0], [3, 4, 0]]),
            output=grid_from_list([[1, 3], [2, 4], [0, 0]]),
        ),
    )
    result = run_refinement_loop(
        demos, Library(),
        max_steps=3, max_candidates=5000, max_rounds=1,
    )
    assert result.solved
    assert result.synthesis_result is not None
    assert result.synthesis_result.solved
    # No search rounds needed — solved by observation
    assert len(result.rounds) == 0


def test_synthesis_falls_through_to_search():
    """When synthesis can't solve, search still runs."""
    from aria.library.store import Library
    from aria.refinement import run_refinement_loop

    # A task that needs object-level reasoning (not a direct transform)
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        ),
    )
    result = run_refinement_loop(
        demos, Library(),
        max_steps=3, max_candidates=200, max_rounds=1,
    )
    assert result.synthesis_result is not None
    # Synthesis tried but didn't solve (this needs fill_region etc)
    # Search may or may not solve in 200 candidates
    assert result.synthesis_result.candidates_tested > 0
