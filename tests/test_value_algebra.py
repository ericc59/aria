"""Tests for the compositional value/slot algebra ops."""

from __future__ import annotations

import numpy as np
import pytest

from aria.types import (
    Bind,
    Call,
    ForEach,
    Grid,
    Lambda,
    Literal,
    ObjectNode,
    Program,
    Property,
    Ref,
    Shape,
    Symmetry,
    Type,
    make_grid,
    grid_eq,
)
from aria.runtime import execute, type_check
from aria.runtime.program import bind, call, lit, ref, make_program, lam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obj(
    id: int,
    color: int,
    x: int,
    y: int,
    w: int,
    h: int,
    mask: np.ndarray | None = None,
) -> ObjectNode:
    """Create a test ObjectNode."""
    if mask is None:
        mask = np.ones((h, w), dtype=np.bool_)
    return ObjectNode(
        id=id,
        color=color,
        mask=mask,
        bbox=(x, y, w, h),
        shape=Shape.RECT,
        symmetry=frozenset(),
        size=int(np.sum(mask)),
    )


def _grid(*rows: list[int]) -> Grid:
    return np.array(rows, dtype=np.uint8)


# ===================================================================
# B. Value functions
# ===================================================================


class TestObjBboxRegion:
    def test_returns_bbox(self):
        from aria.runtime.ops.value_algebra import _obj_bbox_region

        obj = _obj(1, 3, x=2, y=1, w=4, h=3)
        assert _obj_bbox_region(obj) == (2, 1, 4, 3)


class TestSubgridOf:
    def test_extracts_region(self):
        from aria.runtime.ops.value_algebra import _subgrid_of

        grid = _grid(
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 4, 5, 6, 0],
            [0, 0, 0, 0, 0],
        )
        obj = _obj(1, 1, x=1, y=1, w=3, h=2)
        sub = _subgrid_of(obj, grid)
        expected = _grid([1, 2, 3], [4, 5, 6])
        assert grid_eq(sub, expected)

    def test_clips_to_bounds(self):
        from aria.runtime.ops.value_algebra import _subgrid_of

        grid = _grid([1, 2], [3, 4])
        obj = _obj(1, 1, x=1, y=1, w=5, h=5)
        sub = _subgrid_of(obj, grid)
        expected = _grid([4])
        assert grid_eq(sub, expected)


class TestDominantColorIn:
    def test_finds_most_common(self):
        from aria.runtime.ops.value_algebra import _dominant_color_in

        grid = _grid([0, 1, 1, 2, 1])
        assert _dominant_color_in(grid, 0) == 1

    def test_all_bg(self):
        from aria.runtime.ops.value_algebra import _dominant_color_in

        grid = _grid([0, 0, 0])
        assert _dominant_color_in(grid, 0) == 0


class TestMinorityColorIn:
    def test_finds_least_common(self):
        from aria.runtime.ops.value_algebra import _minority_color_in

        grid = _grid([0, 1, 1, 2, 1])
        assert _minority_color_in(grid, 0) == 2


class TestNumColorsIn:
    def test_counts_non_bg(self):
        from aria.runtime.ops.value_algebra import _num_colors_in

        grid = _grid([0, 1, 2, 3, 0])
        assert _num_colors_in(grid, 0) == 3

    def test_empty(self):
        from aria.runtime.ops.value_algebra import _num_colors_in

        grid = _grid([0, 0, 0])
        assert _num_colors_in(grid, 0) == 0


class TestTouchesBorder:
    def test_touching(self):
        from aria.runtime.ops.value_algebra import _touches_border

        grid = make_grid(5, 5)
        obj = _obj(1, 1, x=0, y=0, w=2, h=2)
        assert _touches_border(obj, grid) is True

    def test_not_touching(self):
        from aria.runtime.ops.value_algebra import _touches_border

        grid = make_grid(5, 5)
        obj = _obj(1, 1, x=2, y=2, w=1, h=1)
        assert _touches_border(obj, grid) is False


class TestDisplacement:
    def test_basic(self):
        from aria.runtime.ops.value_algebra import _displacement

        a = _obj(1, 1, x=0, y=0, w=2, h=2)  # center at (0, 0)
        b = _obj(2, 2, x=4, y=6, w=2, h=2)  # center at (7, 5)
        dr, dc = _displacement(a, b)
        assert dr == 6  # 7 - 1... wait
        # a center_row = 0 + 2//2 = 1, center_col = 0 + 2//2 = 1
        # b center_row = 6 + 2//2 = 7, center_col = 4 + 2//2 = 5
        assert dr == 6
        assert dc == 4


# ===================================================================
# C. Aggregation combinators
# ===================================================================


class TestArgmaxBy:
    def test_by_size(self):
        from aria.runtime.ops.value_algebra import _argmax_by

        small = _obj(1, 1, x=0, y=0, w=1, h=1)
        big = _obj(2, 2, x=0, y=0, w=3, h=3)
        result = _argmax_by(Property.SIZE, {small, big})
        assert result.id == big.id

    def test_empty(self):
        from aria.runtime.ops.value_algebra import _argmax_by

        with pytest.raises(ValueError):
            _argmax_by(Property.SIZE, set())


class TestArgminBy:
    def test_by_size(self):
        from aria.runtime.ops.value_algebra import _argmin_by

        small = _obj(1, 1, x=0, y=0, w=1, h=1)
        big = _obj(2, 2, x=0, y=0, w=3, h=3)
        result = _argmin_by(Property.SIZE, {small, big})
        assert result.id == small.id


class TestMostCommonVal:
    def test_most_common_color(self):
        from aria.runtime.ops.value_algebra import _most_common_val

        a = _obj(1, 3, x=0, y=0, w=1, h=1)
        b = _obj(2, 3, x=1, y=0, w=1, h=1)
        c = _obj(3, 5, x=2, y=0, w=1, h=1)
        result = _most_common_val(Property.COLOR, {a, b, c})
        assert result == 3


class TestUnionBbox:
    def test_two_objects(self):
        from aria.runtime.ops.value_algebra import _union_bbox

        a = _obj(1, 1, x=1, y=2, w=3, h=2)
        b = _obj(2, 2, x=5, y=0, w=2, h=4)
        x, y, w, h = _union_bbox({a, b})
        assert x == 1
        assert y == 0
        assert w == 6  # 7 - 1
        assert h == 4  # max(2+2, 0+4) - 0


class TestMergeObjects:
    def test_merge_two(self):
        from aria.runtime.ops.value_algebra import _merge_objects

        a = _obj(1, 3, x=0, y=0, w=2, h=2)
        b = _obj(2, 3, x=3, y=0, w=2, h=2)
        merged = _merge_objects({a, b})
        assert merged.bbox == (0, 0, 5, 2)
        assert merged.mask[0, 0]  # from a
        assert merged.mask[0, 3]  # from b
        assert not merged.mask[0, 2]  # gap


# ===================================================================
# D. Generic actions
# ===================================================================


class TestStamp:
    def test_paints_at_bbox(self):
        from aria.runtime.ops.value_algebra import _stamp

        grid = make_grid(5, 5)
        obj = _obj(1, 7, x=1, y=1, w=2, h=2)
        result = _stamp(obj, grid)
        assert result[1, 1] == 7
        assert result[1, 2] == 7
        assert result[2, 1] == 7
        assert result[2, 2] == 7
        assert result[0, 0] == 0  # untouched

    def test_partial_mask(self):
        from aria.runtime.ops.value_algebra import _stamp

        grid = make_grid(5, 5)
        mask = np.array([[True, False], [False, True]], dtype=np.bool_)
        obj = _obj(1, 4, x=1, y=1, w=2, h=2, mask=mask)
        result = _stamp(obj, grid)
        assert result[1, 1] == 4
        assert result[1, 2] == 0
        assert result[2, 1] == 0
        assert result[2, 2] == 4


class TestCoverObj:
    def test_erases_object(self):
        from aria.runtime.ops.value_algebra import _cover_obj

        grid = _grid(
            [0, 0, 0],
            [0, 5, 5],
            [0, 5, 5],
        )
        obj = _obj(1, 5, x=1, y=1, w=2, h=2)
        result = _cover_obj(obj, 0, grid)
        assert result[1, 1] == 0
        assert result[1, 2] == 0
        assert result[2, 1] == 0
        assert result[2, 2] == 0
        assert result[0, 0] == 0  # unchanged


class TestPaintCells:
    def test_paints_cells(self):
        from aria.runtime.ops.value_algebra import _paint_cells

        grid = make_grid(5, 5)
        cells = {(0, 0), (1, 1), (2, 2)}
        result = _paint_cells(cells, 3, grid)
        assert result[0, 0] == 3
        assert result[1, 1] == 3
        assert result[2, 2] == 3
        assert result[0, 1] == 0

    def test_ignores_out_of_bounds(self):
        from aria.runtime.ops.value_algebra import _paint_cells

        grid = make_grid(3, 3)
        cells = {(0, 0), (10, 10)}
        result = _paint_cells(cells, 1, grid)
        assert result[0, 0] == 1


class TestConnectPaint:
    def test_draws_line(self):
        from aria.runtime.ops.value_algebra import _connect_paint

        grid = make_grid(5, 5)
        a = _obj(1, 1, x=0, y=0, w=1, h=1)  # center (0, 0)
        b = _obj(2, 2, x=4, y=4, w=1, h=1)  # center (4, 4)
        result = _connect_paint(a, b, 8, grid)
        # Diagonal line from (0,0) to (4,4)
        for i in range(5):
            assert result[i, i] == 8


class TestNegate:
    def test_basic(self):
        from aria.runtime.ops.value_algebra import _negate

        assert _negate(5) == -5
        assert _negate(-3) == 3
        assert _negate(0) == 0


# ===================================================================
# Integration: compose through Program IR
# ===================================================================


class TestProgramComposition:
    """Test that new ops compose correctly through the executor."""

    def test_move_by_negative_height(self):
        """Move each object up by its own height via for_each + stamp."""
        # 5x5 grid with a 2x2 object at (1,2)
        grid = _grid(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 0, 0, 0, 0],
        )

        # Program: find objects, for each: cover, translate up by height, stamp
        prog = make_program(
            [
                bind("objs", Type.OBJECT_SET, call("find_objects", ref("input"))),
                bind("canvas", Type.GRID, call("new_grid", call("dims_of", ref("input")), lit(0, Type.COLOR))),
                ForEach(
                    iter_name="obj",
                    source=Ref("objs"),
                    body=(
                        Bind("h", Type.INT, call("get_height", ref("obj"))),
                        Bind("neg_h", Type.INT, call("negate", ref("h"))),
                        Bind("moved", Type.OBJECT, call("translate_delta", ref("neg_h"), lit(0, Type.INT), ref("obj"))),
                        Bind("canvas", Type.GRID, call("stamp", ref("moved"), ref("canvas"))),
                    ),
                    accumulator="canvas",
                    output_name="result",
                ),
            ],
            output="result",
        )

        result = execute(prog, grid)
        # Object was at rows 2-3, height=2, so moves up by 2 to rows 0-1
        expected = _grid(
            [0, 3, 3, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        )
        assert grid_eq(result, expected)

    def test_crop_to_union_bbox(self):
        """Crop grid to the bounding box of filtered objects."""
        grid = _grid(
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        )

        prog = make_program(
            [
                bind("objs", Type.OBJECT_SET, call("find_objects", ref("input"))),
                bind("bbox", Type.REGION, call("union_bbox", ref("objs"))),
                bind("result", Type.GRID, call("crop", ref("input"), ref("bbox"))),
            ],
            output="result",
        )

        result = execute(prog, grid)
        # Objects at (1,1) and (4,1), both 1x1
        # union_bbox: x=1, y=1, w=4, h=1
        expected = _grid([1, 0, 0, 1])
        assert grid_eq(result, expected)

    def test_type_check_passes(self):
        """Verify type checking works for programs using new ops."""
        prog = make_program(
            [
                bind("objs", Type.OBJECT_SET, call("find_objects", ref("input"))),
                bind("big", Type.OBJECT, call("argmax_by", lit(Property.SIZE, Type.PROPERTY), ref("objs"))),
                bind("sub", Type.GRID, call("subgrid_of", ref("big"), ref("input"))),
                bind("dc", Type.COLOR, call("dominant_color_in", ref("sub"), lit(0, Type.COLOR))),
            ],
            output="dc",
        )

        errors = type_check(prog, {"input": Type.GRID})
        assert errors == [], f"Type errors: {errors}"
