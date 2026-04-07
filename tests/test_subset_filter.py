"""Tests for selective keep/erase ops."""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import has_op
from aria.runtime.ops.subset_filter import (
    _keep_by_min_size, _erase_by_max_size, _erase_color, _keep_color,
)
from aria.core.compose import try_compositions
from aria.core.arc import solve_arc_task
from aria.types import DemoPair, grid_from_list


def test_ops_registered():
    assert has_op("keep_by_min_size")
    assert has_op("erase_by_max_size")
    assert has_op("erase_color")
    assert has_op("keep_color")


def test_erase_by_max_size():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 5],  # 5 is single pixel
        [0, 0, 0, 0],
    ])
    result = _erase_by_max_size(grid, 1)
    assert result[2, 3] == 0  # singleton erased
    assert result[1, 1] == 1  # shape kept


def test_keep_by_min_size():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 5],
        [0, 0, 0, 0],
    ])
    result = _keep_by_min_size(grid, 1)
    assert result[1, 1] == 1  # shape kept
    assert result[2, 3] == 0  # singleton erased


def test_erase_color():
    grid = grid_from_list([[1, 2, 1], [3, 1, 2]])
    result = _erase_color(grid, 2)
    assert np.sum(result == 2) == 0


def test_keep_color():
    grid = grid_from_list([[0, 2, 0], [3, 0, 1]])
    result = _keep_color(grid, 1)
    assert np.sum(result == 2) == 0
    assert np.sum(result == 1) == 1


def test_erase_singletons_in_composition():
    """erase_singletons should be available as stage-1."""
    # Task: erase single-pixel markers, keep the rest
    inp = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 5],
        [0, 0, 0, 0],
    ])
    expected = _erase_by_max_size(inp, 1)
    demos = (DemoPair(input=inp, output=expected),)
    result = try_compositions(demos, task_id="test")
    assert result.solved


def test_no_task_id():
    import inspect
    for fn in [_keep_by_min_size, _erase_by_max_size]:
        assert "1b59e163" not in inspect.getsource(fn)


def test_no_regressions():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved
