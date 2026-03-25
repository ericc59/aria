"""Executor integration tests with multi-step programs.

Tests programs from DESIGN.md examples and other realistic scenarios
to verify the full execution pipeline works end-to-end.
"""

import numpy as np
import pytest

from aria.types import (
    Bind, Call, DemoPair, Literal, Program, Ref, Type, TaskContext,
    grid_from_list, grid_eq, make_grid, SortDir, Property,
)
from aria.runtime import execute, ExecutionError
from aria.runtime.program import ref, lit, call, bind, make_program


class TestBasicExecution:
    def test_identity(self):
        """Yield input directly."""
        prog = make_program([], "input")
        grid = grid_from_list([[1, 2], [3, 4]])
        result = execute(prog, grid)
        assert grid_eq(result, grid)

    def test_new_grid(self):
        """Create a new grid with computed dims."""
        prog = make_program([
            bind("dims", Type.DIMS, call("dims_make", lit(3, Type.INT), lit(2, Type.INT))),
            bind("result", Type.GRID, call("new_grid", ref("dims"), lit(7, Type.INT))),
        ], "result")
        result = execute(prog, make_grid(1, 1))
        expected = make_grid(3, 2, fill=7)
        assert grid_eq(result, expected)

    def test_dims_of_input(self):
        """Read dims from input grid."""
        prog = make_program([
            bind("d", Type.DIMS, call("dims_of", ref("input"))),
            bind("r", Type.INT, call("rows_of", ref("d"))),
            bind("c", Type.INT, call("cols_of", ref("d"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", ref("c"), ref("r")),  # transpose dims
                lit(0, Type.INT),
            )),
        ], "result")
        result = execute(prog, make_grid(3, 5))
        assert result.shape == (5, 3)


class TestArithmetic:
    def test_add(self):
        prog = make_program([
            bind("a", Type.INT, lit(3, Type.INT)),
            bind("b", Type.INT, lit(4, Type.INT)),
            bind("c", Type.INT, call("add", ref("a"), ref("b"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", ref("c"), lit(1, Type.INT)),
                lit(0, Type.INT),
            )),
        ], "result")
        result = execute(prog, make_grid(1, 1))
        assert result.shape == (7, 1)

    def test_isqrt(self):
        prog = make_program([
            bind("n", Type.INT, lit(16, Type.INT)),
            bind("s", Type.INT, call("isqrt", ref("n"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", ref("s"), ref("s")),
                lit(1, Type.INT),
            )),
        ], "result")
        result = execute(prog, make_grid(1, 1))
        assert result.shape == (4, 4)
        assert np.all(result == 1)

    def test_mul_dims(self):
        """Multiply to compute output dimensions."""
        prog = make_program([
            bind("d", Type.DIMS, call("dims_of", ref("input"))),
            bind("r", Type.INT, call("rows_of", ref("d"))),
            bind("r2", Type.INT, call("mul", ref("r"), lit(2, Type.INT))),
            bind("c", Type.INT, call("cols_of", ref("d"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", ref("r2"), ref("c")),
                lit(0, Type.INT),
            )),
        ], "result")
        result = execute(prog, make_grid(3, 5))
        assert result.shape == (6, 5)


class TestGridOps:
    def test_stack_v_two_copies(self):
        """Stack input vertically with itself."""
        prog = make_program([
            bind("result", Type.GRID, call("stack_v", ref("input"), ref("input"))),
        ], "result")
        grid = grid_from_list([[1, 2], [3, 4]])
        result = execute(prog, grid)
        expected = grid_from_list([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert grid_eq(result, expected)

    def test_overlay(self):
        """Overlay a pattern on the input."""
        prog = make_program([
            bind("bg", Type.GRID, call("new_grid",
                call("dims_of", ref("input")),
                lit(5, Type.INT),
            )),
            bind("result", Type.GRID, call("overlay", ref("input"), ref("bg"))),
        ], "result")
        grid = grid_from_list([[0, 1], [2, 0]])
        result = execute(prog, grid)
        # bg=0 cells get replaced by 5, non-zero stay
        expected = grid_from_list([[5, 1], [2, 5]])
        assert grid_eq(result, expected)

    def test_apply_color_map(self):
        """Apply a color mapping to the input."""
        prog = make_program([
            bind("result", Type.GRID, call("apply_color_map",
                lit({1: 3, 2: 7}, Type.COLOR_MAP),
                ref("input"),
            )),
        ], "result")
        grid = grid_from_list([[1, 2, 0], [0, 1, 2]])
        result = execute(prog, grid)
        expected = grid_from_list([[3, 7, 0], [0, 3, 7]])
        assert grid_eq(result, expected)

    def test_crop(self):
        prog = make_program([
            bind("result", Type.GRID, call("crop",
                ref("input"),
                lit((1, 1, 2, 2), Type.PAIR),
            )),
        ], "result")
        grid = grid_from_list([
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 7, 8, 0],
            [0, 0, 0, 0],
        ])
        result = execute(prog, grid)
        expected = grid_from_list([[5, 6], [7, 8]])
        assert grid_eq(result, expected)


class TestObjectOps:
    def test_find_and_count(self):
        """Find objects and count them."""
        prog = make_program([
            bind("objs", Type.OBJECT_SET, call("find_objects", ref("input"))),
            bind("n", Type.INT, call("count", ref("objs"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", ref("n"), lit(1, Type.INT)),
                lit(1, Type.INT),
            )),
        ], "result")
        # Grid with 2 distinct non-bg objects
        grid = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ])
        result = execute(prog, grid)
        assert result.shape[0] == 2  # 2 objects
        assert result.shape[1] == 1

    def test_get_color(self):
        """Extract color from an object."""
        prog = make_program([
            bind("objs", Type.OBJECT_SET, call("find_objects", ref("input"))),
            bind("obj", Type.OBJECT, call("singleton", ref("objs"))),
            bind("c", Type.INT, call("get_color", ref("obj"))),
            bind("result", Type.GRID, call("new_grid",
                call("dims_make", lit(1, Type.INT), lit(1, Type.INT)),
                ref("c"),
            )),
        ], "result")
        grid = grid_from_list([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
        ])
        result = execute(prog, grid)
        assert result[0, 0] == 3


class TestDesignExampleA:
    """DESIGN.md Example A: Computed output dimensions.

    Task: Input has N colored objects. Output is sqrt(N) x sqrt(N) grid,
    each cell colored by the corresponding object's color sorted by size.
    """

    def test_computed_dims(self):
        prog = make_program([
            bind("objects", Type.OBJECT_SET, call("find_objects", ref("input"))),
            bind("n", Type.INT, call("count", ref("objects"))),
            bind("sorted", Type.OBJECT_LIST, call("sort_by",
                lit(Property.SIZE, Type.PROPERTY),
                lit(SortDir.DESC, Type.SORT_DIR),
                ref("objects"),
            )),
            bind("colors", Type.INT_LIST, call("map_list",
                call("get_color"),  # This will be treated as a callable — actually need a ref
                ref("sorted"),
            )),
            bind("side", Type.INT, call("isqrt", ref("n"))),
            bind("out_dims", Type.DIMS, call("dims_make", ref("side"), ref("side"))),
            bind("canvas", Type.GRID, call("new_grid", ref("out_dims"), lit(0, Type.INT))),
            bind("result", Type.GRID, call("fill_cells", ref("canvas"), ref("colors"))),
        ], "result")
        # 4 objects of different sizes
        grid = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
        ])
        # 4 objects -> sqrt(4)=2 -> 2x2 grid
        # Object sizes: 1(9px), 2(4px), 3(1px), 4(12px)
        # Sorted desc: 4(12), 1(9), 2(4), 3(1)
        # Colors sorted: [4, 1, 2, 3]
        # get_color() with 0 args returns a partial (curried) function.
        # map_list applies it to each element of the sorted list.
        result = execute(prog, grid)
        assert result.shape == (2, 2)
        # Colors sorted by size desc: 4(12px), 1(9px), 2(4px), 3(1px)
        assert result[0, 0] == 4
        assert result[0, 1] == 1
        assert result[1, 0] == 2
        assert result[1, 1] == 3


class TestErrorHandling:
    def test_unbound_variable(self):
        prog = make_program([
            bind("result", Type.GRID, ref("nonexistent")),
        ], "result")
        with pytest.raises(ExecutionError, match="Unbound variable"):
            execute(prog, make_grid(1, 1))

    def test_assertion_failure(self):
        from aria.types import Assert
        prog = Program(
            steps=(Assert(pred=lit(False, Type.BOOL)),),
            output="input",
        )
        with pytest.raises(ExecutionError, match="assertion failed"):
            execute(prog, make_grid(1, 1))

    def test_missing_output(self):
        prog = make_program([], "nonexistent")
        with pytest.raises(ExecutionError, match="Output binding.*not found"):
            execute(prog, make_grid(1, 1))


class TestFloodFill:
    def test_fill_enclosed_region(self):
        prog = make_program([
            bind("result", Type.GRID, call("flood_fill",
                ref("input"),
                lit((1, 1), Type.PAIR),
                lit(5, Type.INT),
            )),
        ], "result")
        grid = grid_from_list([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        result = execute(prog, grid)
        expected = grid_from_list([
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1],
        ])
        assert grid_eq(result, expected)
