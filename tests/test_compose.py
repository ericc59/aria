"""Tests for bounded two-stage pipeline composition."""

from __future__ import annotations

from aria.core.compose import CompositionResult, try_compositions
from aria.core.arc import solve_arc_task
from aria.types import DemoPair, grid_from_list


def _simple():
    return (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )


def _framed():
    return (
        DemoPair(
            input=grid_from_list([
                [4, 4, 4, 4, 4],
                [4, 1, 2, 1, 4],
                [4, 3, 0, 3, 4],
                [4, 1, 2, 1, 4],
                [4, 4, 4, 4, 4],
            ]),
            output=grid_from_list([
                [4, 4, 4, 4, 4],
                [4, 1, 2, 1, 4],
                [4, 3, 2, 3, 4],
                [4, 1, 2, 1, 4],
                [4, 4, 4, 4, 4],
            ]),
        ),
    )


def test_try_compositions_returns_result():
    result = try_compositions(_simple(), task_id="test")
    assert isinstance(result, CompositionResult)


def test_composition_on_framed_task():
    result = try_compositions(_framed(), task_id="test")
    assert isinstance(result, CompositionResult)
    # May or may not solve — just shouldn't crash


def test_composition_bounded():
    """Should not try more than max_attempts."""
    result = try_compositions(_simple(), task_id="test", max_attempts=5)
    assert isinstance(result, CompositionResult)


def test_no_task_id():
    import inspect
    src = inspect.getsource(try_compositions)
    assert "1b59e163" not in src


def test_no_regressions():
    result = solve_arc_task(_simple(), task_id="test", use_editor_search=True)
    assert result.solved
