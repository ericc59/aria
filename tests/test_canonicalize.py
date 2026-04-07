"""Tests for local canonicalization ops."""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import has_op
from aria.runtime.ops.canonicalize import _compact_to_origin, _normalize_to_grid
from aria.core.compose import try_compositions
from aria.core.arc import solve_arc_task
from aria.types import DemoPair, grid_from_list


def test_ops_registered():
    assert has_op("compact_to_origin")
    assert has_op("normalize_to_grid")


def test_compact_to_origin():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
    ])
    result = _compact_to_origin(grid)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_normalize_to_grid():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
    ])
    result = _normalize_to_grid(grid)
    assert result.shape == (4, 4)
    assert result[0, 0] == 1
    assert result[0, 1] == 2
    assert result[1, 0] == 3


def test_compact_preserves_content():
    grid = grid_from_list([
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ])
    result = _compact_to_origin(grid)
    assert result.shape == (1, 1)
    assert result[0, 0] == 5


def test_normalize_in_composition():
    """normalize_to_grid should verify for a simple shift task."""
    inp = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
    ])
    out = grid_from_list([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    demos = (DemoPair(input=inp, output=out),)
    result = try_compositions(demos, task_id="test")
    assert result.solved, f"Expected solved, got: {result.description}"


def test_no_task_id():
    import inspect
    for fn in [_compact_to_origin, _normalize_to_grid]:
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
