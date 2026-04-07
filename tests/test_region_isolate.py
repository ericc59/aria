"""Tests for region isolation ops."""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import has_op
from aria.runtime.ops.region_isolate import _crop_to_content, _crop_to_object, _crop_frame_interior
from aria.core.compose import try_compositions
from aria.core.arc import solve_arc_task
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_ops_registered():
    assert has_op("crop_to_content")
    assert has_op("crop_to_object")
    assert has_op("crop_frame_interior")


# ---------------------------------------------------------------------------
# crop_to_content
# ---------------------------------------------------------------------------


def test_crop_to_content_basic():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    result = _crop_to_content(grid)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_crop_to_content_full_grid():
    grid = grid_from_list([[1, 2], [3, 4]])
    result = _crop_to_content(grid)
    # bg is most common — if all different, bg=1 or similar
    assert result.size > 0


def test_crop_to_content_empty():
    grid = grid_from_list([[0, 0], [0, 0]])
    result = _crop_to_content(grid)
    np.testing.assert_array_equal(result, grid)


# ---------------------------------------------------------------------------
# crop_to_object
# ---------------------------------------------------------------------------


def test_crop_to_object():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    result = _crop_to_object(grid, 3)
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# crop_frame_interior
# ---------------------------------------------------------------------------


def test_crop_frame_interior():
    grid = grid_from_list([
        [4, 4, 4, 4, 4],
        [4, 1, 2, 3, 4],
        [4, 5, 6, 7, 4],
        [4, 4, 4, 4, 4],
    ])
    result = _crop_frame_interior(grid)
    assert result.shape == (2, 3)
    assert result[0, 0] == 1


def test_crop_frame_no_frame():
    grid = grid_from_list([[1, 2], [3, 4]])
    result = _crop_frame_interior(grid)
    np.testing.assert_array_equal(result, grid)


# ---------------------------------------------------------------------------
# Composition integration
# ---------------------------------------------------------------------------


def test_crop_available_in_composition():
    """crop_to_content should be tried as stage-1 for dims-change tasks."""
    # A task where output IS the cropped content
    inp = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    out = grid_from_list([[1, 2], [3, 4]])
    demos = (DemoPair(input=inp, output=out),)
    result = try_compositions(demos, task_id="test")
    assert result.solved, f"Expected solved, got: {result.description}"


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_no_task_id():
    import inspect
    for fn in [_crop_to_content, _crop_to_object, _crop_frame_interior]:
        assert "1b59e163" not in inspect.getsource(fn)


def test_solved_tasks_unaffected():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved
