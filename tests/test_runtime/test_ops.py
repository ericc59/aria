"""Tests for runtime operations."""

import numpy as np
import pytest

from aria.types import (
    Grid, ObjectNode, Shape, Symmetry, Dir, Axis, SortDir, Property,
    Type, grid_from_list, grid_eq, make_grid,
)
from aria.runtime.ops import get_op, has_op, all_ops


def test_registry_populated():
    """Core ops should be registered at import time."""
    import aria.runtime  # noqa: triggers registration
    ops = all_ops()
    assert len(ops) > 20, f"Expected 20+ ops, got {len(ops)}"
    # Spot-check key ops exist
    for name in ["find_objects", "new_grid", "dims_of", "count", "flood_fill"]:
        assert has_op(name), f"Missing op: {name}"


def test_new_grid():
    _, impl = get_op("new_grid")
    g = impl((3, 4), 5)
    assert g.shape == (3, 4)
    assert np.all(g == 5)


def test_dims_of():
    _, impl = get_op("dims_of")
    g = make_grid(7, 3)
    assert impl(g) == (7, 3)


def test_dims_make():
    _, impl = get_op("dims_make")
    assert impl(5, 10) == (5, 10)


def test_rows_of():
    _, impl = get_op("rows_of")
    assert impl((5, 10)) == 5


def test_cols_of():
    _, impl = get_op("cols_of")
    assert impl((5, 10)) == 10


def test_scale_dims():
    _, impl = get_op("scale_dims")
    assert impl((3, 4), 2) == (6, 8)


def test_find_objects():
    _, impl = get_op("find_objects")
    g = grid_from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    objects = impl(g)
    # Should find at least 1 non-background object
    assert len(objects) >= 1


def test_count():
    _, impl = get_op("count")
    assert impl({1, 2, 3}) == 3
    assert impl(set()) == 0


def test_length():
    _, impl = get_op("length")
    assert impl([1, 2, 3, 4]) == 4


def test_max_val():
    _, impl = get_op("max_val")
    assert impl([3, 1, 4, 1, 5]) == 5


def test_min_val():
    _, impl = get_op("min_val")
    assert impl([3, 1, 4, 1, 5]) == 1


def test_stack_h():
    _, impl = get_op("stack_h")
    a = grid_from_list([[1, 2], [3, 4]])
    b = grid_from_list([[5, 6], [7, 8]])
    result = impl(a, b)
    expected = grid_from_list([[1, 2, 5, 6], [3, 4, 7, 8]])
    assert grid_eq(result, expected)


def test_stack_v():
    _, impl = get_op("stack_v")
    a = grid_from_list([[1, 2], [3, 4]])
    b = grid_from_list([[5, 6], [7, 8]])
    result = impl(a, b)
    expected = grid_from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert grid_eq(result, expected)


def test_crop():
    _, impl = get_op("crop")
    g = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ])
    cropped = impl(g, (1, 1, 2, 2))
    expected = grid_from_list([[1, 2], [3, 4]])
    assert grid_eq(cropped, expected)


def test_overlay():
    _, impl = get_op("overlay")
    bottom = grid_from_list([[1, 1], [1, 1]])
    top = grid_from_list([[0, 2], [3, 0]])
    result = impl(top, bottom)
    expected = grid_from_list([[1, 2], [3, 1]])
    assert grid_eq(result, expected)


def test_apply_color_map():
    _, impl = get_op("apply_color_map")
    g = grid_from_list([[1, 2, 3], [3, 2, 1]])
    cmap = {1: 7, 3: 5}
    result = impl(cmap, g)
    expected = grid_from_list([[7, 2, 5], [5, 2, 7]])
    assert grid_eq(result, expected)


def test_flood_fill():
    _, impl = get_op("flood_fill")
    g = grid_from_list([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
    ])
    result = impl(g, (0, 0), 5)
    expected = grid_from_list([
        [5, 5, 1],
        [5, 5, 1],
        [1, 1, 1],
    ])
    assert grid_eq(result, expected)


def test_demo_count():
    _, impl = get_op("demo_count")
    from aria.types import DemoPair, TaskContext
    ctx = TaskContext(demos=(
        DemoPair(input=make_grid(1, 1), output=make_grid(1, 1)),
        DemoPair(input=make_grid(1, 1), output=make_grid(1, 1)),
    ))
    assert impl(ctx) == 2
