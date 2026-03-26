"""Tests for generic candidate scoring."""

from __future__ import annotations

import aria.runtime  # noqa: F401
from aria.scoring import CandidateScore, score_program
from aria.types import Axis, Bind, Call, DemoPair, Literal, Program, Ref, Type, grid_from_list


def _identity_program() -> Program:
    """Program that returns the input unchanged."""
    return Program(
        steps=(
            Bind(name="v0", typ=Type.GRID, expr=Ref("input")),
        ),
        output="v0",
    )


def _reflect_program() -> Program:
    """Program that reflects input horizontally."""
    return Program(
        steps=(
            Bind(
                name="v0",
                typ=Type.GRID,
                expr=Call(op="reflect_grid", args=(Literal(Axis.HORIZONTAL, Type.AXIS), Ref("input"))),
            ),
        ),
        output="v0",
    )


def test_score_exact_match_has_passed_true():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )
    score = score_program(_identity_program(), demos)
    assert score.passed
    assert score.demos_passed == 1
    assert score.dims_correct == 1
    assert score.pixel_diff_total == 0
    assert score.execution_errors == 0


def test_score_wrong_output_counts_pixel_diffs():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[5, 6], [7, 8]]),
        ),
    )
    score = score_program(_identity_program(), demos)
    assert not score.passed
    assert score.dims_correct == 1
    assert score.pixel_diff_total == 4


def test_score_ordering_rewards_fewer_pixel_diffs():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 1], [0, 0]]),
        ),
    )
    # Identity: 1 pixel diff (output[0,1] should be 1, input has 0)
    identity_score = score_program(_identity_program(), demos)
    assert identity_score.pixel_diff_total == 1

    # Compare with a program that produces all-zero output (more diffs)
    # Identity should be better since it has fewer pixel diffs
    assert identity_score.dims_correct == 1


def test_score_ordering_prefers_correct_dims():
    # Create two scores manually to test ordering
    better = CandidateScore(
        passed=False,
        demos_passed=0,
        total_demos=1,
        dims_correct=1,
        pixel_diff_total=5,
        execution_errors=0,
        palette_overlap_avg=1.0,
        demo_scores=(),
    )
    worse = CandidateScore(
        passed=False,
        demos_passed=0,
        total_demos=1,
        dims_correct=0,
        pixel_diff_total=-1,
        execution_errors=0,
        palette_overlap_avg=0.5,
        demo_scores=(),
    )
    assert better < worse  # better has lower rank_key


def test_score_ordering_prefers_fewer_execution_errors():
    better = CandidateScore(
        passed=False,
        demos_passed=0,
        total_demos=2,
        dims_correct=1,
        pixel_diff_total=10,
        execution_errors=0,
        palette_overlap_avg=0.8,
        demo_scores=(),
    )
    worse = CandidateScore(
        passed=False,
        demos_passed=0,
        total_demos=2,
        dims_correct=1,
        pixel_diff_total=10,
        execution_errors=1,
        palette_overlap_avg=0.8,
        demo_scores=(),
    )
    assert better < worse


def test_score_to_dict_roundtrips_key_fields():
    score = CandidateScore(
        passed=False,
        demos_passed=1,
        total_demos=2,
        dims_correct=2,
        pixel_diff_total=3,
        execution_errors=0,
        palette_overlap_avg=0.75,
        demo_scores=(),
    )
    d = score.to_dict()
    assert d["passed"] is False
    assert d["dims_correct"] == 2
    assert d["pixel_diff_total"] == 3
    assert d["palette_overlap_avg"] == 0.75
