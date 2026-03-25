"""Geometry-focused runtime operation tests."""

from __future__ import annotations

from aria.runtime.ops import get_op
from aria.types import Shape, grid_eq, grid_from_list


def _singleton_obj(color: int, x: int, y: int):
    import numpy as np
    from aria.types import ObjectNode

    return ObjectNode(
        id=color * 10 + x + y,
        color=color,
        mask=np.array([[True]], dtype=bool),
        bbox=(x, y, 1, 1),
        shape=Shape.DOT,
        symmetry=frozenset(),
        size=1,
    )


def test_nearest_to_uses_center_distance():
    _, impl = get_op("nearest_to")

    target = _singleton_obj(4, 5, 5)
    near = _singleton_obj(5, 7, 6)
    far = _singleton_obj(5, 1, 1)

    result = impl(target, {near, far})
    assert result == near


def test_square_region_and_fill_region_clip_cleanly():
    _, square_region = get_op("square_region")
    _, fill_region = get_op("fill_region")

    target = _singleton_obj(4, 0, 0)
    region = square_region(target, 1)
    result = fill_region(region, 2, grid_from_list([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]))

    expected = grid_from_list([
        [2, 2, 0],
        [2, 2, 0],
        [0, 0, 0],
    ])
    assert grid_eq(result, expected)
