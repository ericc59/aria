"""Tests for diff-guided iterative program construction."""

from __future__ import annotations

import numpy as np

from aria.core.stepper import step_solve, beam_solve, _pixel_diff, _infer_color_map
from aria.types import DemoPair, grid_from_list
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Basic diff utilities
# ---------------------------------------------------------------------------


def test_pixel_diff_identical():
    a = grid_from_list([[1, 2], [3, 4]])
    assert _pixel_diff(a, a) == 0


def test_pixel_diff_counts():
    a = grid_from_list([[1, 2], [3, 4]])
    b = grid_from_list([[1, 9], [3, 9]])
    assert _pixel_diff(a, b) == 2


def test_infer_color_map():
    states = [grid_from_list([[1, 2], [3, 4]])]
    targets = [grid_from_list([[2, 3], [4, 5]])]
    cmap = _infer_color_map(states, targets)
    assert cmap is not None
    assert cmap[1] == 2
    assert cmap[2] == 3


def test_infer_color_map_identity_returns_none():
    states = [grid_from_list([[1, 2]])]
    targets = [grid_from_list([[1, 2]])]
    assert _infer_color_map(states, targets) is None


def test_infer_color_map_inconsistent_returns_none():
    states = [grid_from_list([[1, 1]])]
    targets = [grid_from_list([[2, 3]])]  # 1→2 and 1→3
    assert _infer_color_map(states, targets) is None


# ---------------------------------------------------------------------------
# Single-step solutions
# ---------------------------------------------------------------------------


def test_stepper_color_map():
    """Color remapping solved in one step."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved
    assert result.program is not None
    vr = verify(result.program, demos)
    assert vr.passed


def test_stepper_rotation():
    """Rotation solved in one step."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            output=grid_from_list([[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            output=grid_from_list([[0, 0, 1], [0, 2, 0], [3, 0, 0]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved


def test_stepper_reflection():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 8], [5, 6]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved


def test_stepper_fill_between():
    """Fill between objects of a color."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [8, 0, 0, 8], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [8, 3, 3, 8], [0, 0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 8, 0], [0, 0, 0], [0, 8, 0]]),
            output=grid_from_list([[0, 8, 0], [0, 3, 0], [0, 8, 0]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved


# ---------------------------------------------------------------------------
# Multi-step solutions
# ---------------------------------------------------------------------------


def test_stepper_multi_step():
    """Two-step solution: shift then color remap."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 3], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 4], [0, 0, 0]]),
        ),
    )
    result = step_solve(demos, max_steps=4)
    if result.solved:
        assert len(result.steps_description) >= 1
        vr = verify(result.program, demos)
        assert vr.passed


def test_stepper_verified_program():
    """Any program the stepper produces should pass verification."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 2, 0]]),
        ),
    )
    result = step_solve(demos, max_steps=4)
    if result.solved and result.program is not None:
        vr = verify(result.program, demos)
        assert vr.passed, "stepper program should always verify"


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


def test_stepper_dims_change_returns_unsolved():
    """Stepper requires same dims — dims change should fail cleanly."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[1, 2, 1, 2]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved is False


def test_stepper_empty_demos():
    result = step_solve(())
    assert result.solved is False


def test_stepper_identity():
    """Input == output → solved with 0 steps."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )
    result = step_solve(demos)
    assert result.solved


def test_stepper_reports_steps():
    """Steps description should be non-empty for multi-step solutions."""
    demos = (
        DemoPair(
            input=grid_from_list([[7, 6], [6, 7]]),
            output=grid_from_list([[6, 7], [7, 6]]),
        ),
    )
    result = step_solve(demos)
    if result.solved:
        assert len(result.steps_description) >= 1


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def test_beam_solves_simple():
    """Beam search should solve simple single-step tasks."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 8], [5, 6]]),
        ),
    )
    result = beam_solve(demos, beam_width=3)
    assert result.solved
    vr = verify(result.program, demos)
    assert vr.passed


def test_beam_handles_identity():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[1, 2]]),
        ),
    )
    result = beam_solve(demos, beam_width=3)
    assert result.solved


def test_beam_handles_dims_change():
    demos = (
        DemoPair(
            input=grid_from_list([[1]]),
            output=grid_from_list([[1, 1]]),
        ),
    )
    result = beam_solve(demos, beam_width=3)
    assert result.solved is False


def test_beam_finds_multi_step():
    """Beam should handle multi-step fill_between tasks."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [8, 0, 0, 8], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [8, 3, 3, 8], [0, 0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 8, 0], [0, 0, 0], [0, 8, 0]]),
            output=grid_from_list([[0, 8, 0], [0, 3, 0], [0, 8, 0]]),
        ),
    )
    result = beam_solve(demos, beam_width=3, max_steps=3)
    assert result.solved


def test_beam_object_move():
    """Beam should find object-level movement steps."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [1, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 1]]),
        ),
    )
    result = beam_solve(demos, beam_width=3, max_steps=3)
    assert result.solved
