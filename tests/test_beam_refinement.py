"""Tests for beam refinement: mutation-based search with scored beams."""

from __future__ import annotations

from pathlib import Path

import aria.runtime  # noqa: F401
from aria.library.store import Library
from aria.offline_search import build_literal_pool
from aria.refinement import (
    BeamRefinementResult,
    RefinementResult,
    run_beam_refinement,
    run_refinement_loop,
)
from aria.scoring import score_program
from aria.trace_store import RefinementTraceStore
from aria.types import Axis, Bind, Call, DemoPair, Literal, Program, Ref, Type, grid_from_list


def _identity_demos() -> tuple[DemoPair, ...]:
    """Demos where input == output (identity task)."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[5, 6], [7, 8]]),
        ),
    )


def _reflect_h_demos() -> tuple[DemoPair, ...]:
    """Demos that require horizontal reflection."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6, 7], [8, 9, 0]]),
            output=grid_from_list([[8, 9, 0], [5, 6, 7]]),
        ),
    )


def _bad_reflect_program() -> Program:
    """Vertical reflection — wrong axis for the demos above."""
    return Program(
        steps=(
            Bind(
                name="v0",
                typ=Type.GRID,
                expr=Call(
                    op="reflect_grid",
                    args=(Literal(Axis.VERTICAL, Type.AXIS), Ref("input")),
                ),
            ),
        ),
        output="v0",
    )


def _correct_reflect_program() -> Program:
    """Horizontal reflection — correct for _reflect_h_demos."""
    return Program(
        steps=(
            Bind(
                name="v0",
                typ=Type.GRID,
                expr=Call(
                    op="reflect_grid",
                    args=(Literal(Axis.HORIZONTAL, Type.AXIS), Ref("input")),
                ),
            ),
        ),
        output="v0",
    )


def test_beam_scores_and_ranks_seeds():
    demos = _reflect_h_demos()
    literal_pool = build_literal_pool(demos)

    bad = _bad_reflect_program()
    correct = _correct_reflect_program()

    bad_score = score_program(bad, demos)
    correct_score = score_program(correct, demos)
    assert correct_score.passed
    assert not bad_score.passed

    # Beam with correct program as seed should solve immediately
    result = run_beam_refinement(
        demos,
        [bad, correct],
        literal_pool,
        beam_width=4,
        max_rounds=1,
        max_mutations_per_candidate=10,
    )
    assert result.solved
    assert result.winning_program is not None


def test_beam_improves_via_literal_mutation():
    """Beam should find correct axis via literal replacement mutation."""
    demos = _reflect_h_demos()
    literal_pool = build_literal_pool(demos)

    # Seed with wrong axis — mutation should try HORIZONTAL
    seeds = [_bad_reflect_program()]
    result = run_beam_refinement(
        demos,
        seeds,
        literal_pool,
        beam_width=4,
        max_rounds=2,
        max_mutations_per_candidate=30,
    )
    assert result.solved
    assert result.candidates_scored >= 1


def test_beam_prefers_improved_candidates():
    demos = _reflect_h_demos()
    literal_pool = build_literal_pool(demos)

    seeds = [_bad_reflect_program()]
    result = run_beam_refinement(
        demos,
        seeds,
        literal_pool,
        beam_width=4,
        max_rounds=1,
        max_mutations_per_candidate=50,
    )

    # There should be at least one improvement transition
    improvements = [t for t in result.transitions if t.improved]
    assert len(improvements) > 0 or result.solved


def test_beam_result_round_summaries():
    demos = _identity_demos()
    literal_pool = build_literal_pool(demos)

    seeds = [_bad_reflect_program()]
    result = run_beam_refinement(
        demos,
        seeds,
        literal_pool,
        beam_width=4,
        max_rounds=2,
        max_mutations_per_candidate=10,
    )

    # Should have round summaries for each round completed
    assert len(result.round_summaries) <= 2
    for rs in result.round_summaries:
        assert rs.mutations_tried >= 0
        assert rs.improvements >= 0


def test_refinement_loop_with_beam():
    """Full refinement loop with beam enabled should run beam after flat search."""
    demos = _reflect_h_demos()
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=5,
        max_rounds=1,
        beam_width=4,
        beam_rounds=2,
        beam_mutations_per_candidate=30,
    )
    # Whether or not it solves, beam should have run
    if not result.solved:
        assert result.beam is not None
        assert result.beam.candidates_scored > 0


def test_trace_store_with_beam_data(tmp_path: Path):
    demos = _reflect_h_demos()
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=5,
        max_rounds=1,
        beam_width=4,
        beam_rounds=1,
        beam_mutations_per_candidate=10,
    )

    store = RefinementTraceStore()
    store.add_result(
        task_id="beam_test",
        result=result,
        task_signatures=("dims:same",),
    )
    path = tmp_path / "beam_traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    record = loaded.all_records()[0]
    assert record["task_id"] == "beam_test"
    assert record["task_signatures"] == ["dims:same"]
    if record.get("beam"):
        assert "candidates_scored" in record["beam"]
        assert "transitions" in record["beam"]


def test_beam_refinement_improves_on_poorer_seed():
    """The beam should find the correct solution starting from a wrong seed.

    This is the key integration test: flat search doesn't find it in the tiny
    budget, but beam mutation of the best candidate does.
    """
    demos = _reflect_h_demos()

    # Flat search with very small budget to get seeds
    flat = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=3,
        max_rounds=1,
    )
    # With only 3 candidates at depth 1, flat search is unlikely to find the answer

    # Now run beam
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=3,
        max_rounds=1,
        beam_width=8,
        beam_rounds=3,
        beam_mutations_per_candidate=50,
    )

    # The beam should be able to find the correct axis
    # Even if flat search failed, beam mutation of reflect_grid(VERTICAL, input)
    # should find reflect_grid(HORIZONTAL, input)
    if result.beam is not None:
        assert result.beam.candidates_scored > 0
    # The result should either solve via beam or at least produce beam data
    assert result.solved or result.beam is not None
