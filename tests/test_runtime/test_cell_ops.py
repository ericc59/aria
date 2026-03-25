"""Tests for cell-level operations."""

import numpy as np
from aria.types import grid_from_list, grid_eq, make_grid
from aria.runtime.ops import get_op


def test_conditional_fill():
    _, impl = get_op("conditional_fill")
    g = grid_from_list([[1, 0, 2], [0, 1, 0]])
    result = impl(g, 5, 0)  # replace 0 with 5
    expected = grid_from_list([[1, 5, 2], [5, 1, 5]])
    assert grid_eq(result, expected)


def test_fill_between():
    _, impl = get_op("fill_between")
    g = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ])
    result = impl(g, 3, 7)
    assert result[1, 2] == 7  # between the two 3s


def test_fill_enclosed():
    _, impl = get_op("fill_enclosed")
    # bg=0 (most frequent), walls are 1, interior 0s should get filled
    g = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    result = impl(g, 5)
    assert result[2, 2] == 5  # enclosed interior
    assert result[2, 3] == 5  # enclosed interior
    assert result[0, 0] == 0  # border bg, not enclosed


def test_propagate():
    _, impl = get_op("propagate")
    g = grid_from_list([
        [0, 0, 1, 0],
        [0, 3, 1, 0],
        [0, 0, 1, 0],
    ])
    result = impl(g, 3, 7, 0)  # spread from 3 through 0
    assert result[0, 0] == 7
    assert result[0, 1] == 7
    assert result[0, 3] == 0  # blocked by wall of 1s


def test_complete_symmetry_h():
    _, impl = get_op("complete_symmetry_h")
    g = grid_from_list([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
    ])
    result = impl(g)
    assert result[0, 3] == 1
    assert result[0, 2] == 2
    assert result[1, 3] == 3


def test_complete_symmetry_v():
    _, impl = get_op("complete_symmetry_v")
    g = grid_from_list([
        [1, 2],
        [3, 4],
        [0, 0],
        [0, 0],
    ])
    result = impl(g)
    assert result[3, 0] == 1
    assert result[2, 1] == 4


def test_grid_and():
    _, impl = get_op("grid_and")
    a = grid_from_list([[1, 0, 3], [0, 2, 0]])
    b = grid_from_list([[0, 1, 1], [1, 1, 0]])
    result = impl(a, b)
    expected = grid_from_list([[0, 0, 3], [0, 2, 0]])
    assert grid_eq(result, expected)


def test_grid_or():
    _, impl = get_op("grid_or")
    a = grid_from_list([[1, 0, 0], [0, 0, 3]])
    b = grid_from_list([[0, 2, 0], [4, 0, 0]])
    result = impl(a, b)
    expected = grid_from_list([[1, 2, 0], [4, 0, 3]])
    assert grid_eq(result, expected)


def test_grid_diff():
    _, impl = get_op("grid_diff")
    a = grid_from_list([[1, 2, 3], [4, 5, 6]])
    b = grid_from_list([[1, 0, 3], [0, 5, 0]])
    result = impl(a, b)
    expected = grid_from_list([[0, 2, 0], [4, 0, 6]])
    assert grid_eq(result, expected)


def test_find_pattern():
    _, impl = get_op("find_pattern")
    g = grid_from_list([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    pat = grid_from_list([[1, 1], [1, 1]])
    result = impl(g, pat)
    assert (0, 1) in result


def test_replace_pattern():
    _, impl = get_op("replace_pattern")
    g = grid_from_list([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    pat = grid_from_list([[1]])
    rep = grid_from_list([[3]])
    result = impl(g, pat, rep)
    expected = grid_from_list([
        [0, 3, 0],
        [3, 3, 3],
        [0, 3, 0],
    ])
    assert grid_eq(result, expected)


def test_get_row():
    _, impl = get_op("get_row")
    g = grid_from_list([[1, 2, 3], [4, 5, 6]])
    result = impl(g, 1)
    expected = grid_from_list([[4, 5, 6]])
    assert grid_eq(result, expected)


def test_most_common_color():
    _, impl = get_op("most_common_color")
    g = grid_from_list([[1, 2, 2], [2, 0, 1]])
    assert impl(g) == 2


def test_count_color():
    _, impl = get_op("count_color")
    g = grid_from_list([[1, 2, 2], [2, 0, 1]])
    assert impl(g, 2) == 3
    assert impl(g, 0) == 1


def test_sort_rows():
    _, impl = get_op("sort_rows")
    g = grid_from_list([[3, 2, 1], [1, 2, 3]])
    result = impl(g)
    assert result[0, 0] == 1  # [1,2,3] sorts before [3,2,1]
