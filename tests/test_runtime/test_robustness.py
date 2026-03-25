"""Tests for robustness fixes — edge cases the proposer model hits.

These cover the integration-level issues discovered when Claude generates
programs that use ops in unexpected but reasonable ways.
"""

import numpy as np
import pytest

from aria.types import (
    DemoPair, TaskContext, Type, grid_from_list, grid_eq, make_grid,
)
from aria.runtime import execute, ExecutionError
from aria.runtime.type_system import type_check
from aria.runtime.program import ref, lit, call, bind, make_program
from aria.proposer.parser import parse_program


class TestCallableVariables:
    """bind step = infer_step(ctx) / bind result = step(input)"""

    def test_call_bound_function(self):
        prog = parse_program("""bind step = infer_step(ctx)
bind result = step(input)
yield result""")
        grid = make_grid(3, 3, fill=1)
        ctx = TaskContext(demos=(DemoPair(input=grid, output=grid),))
        result = execute(prog, grid, ctx)
        assert result.shape == (3, 3)

    def test_call_bound_with_arbitrary_name(self):
        prog = parse_program("""bind transform = infer_step(ctx)
bind result = transform(input)
yield result""")
        grid = make_grid(2, 2)
        ctx = TaskContext(demos=(DemoPair(input=grid, output=grid),))
        result = execute(prog, grid, ctx)
        assert result.shape == (2, 2)


class TestTupleLiterals:
    """flood_fill(input, (0, 0), 8) — tuple args"""

    def test_tuple_in_flood_fill(self):
        prog = parse_program("""bind result = flood_fill(input, (0, 0), 5)
yield result""")
        grid = grid_from_list([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        result = execute(prog, grid)
        expected = grid_from_list([[5, 5, 1], [5, 5, 1], [1, 1, 1]])
        assert grid_eq(result, expected)

    def test_tuple_in_crop(self):
        prog = parse_program("""bind result = crop(input, (1, 1, 2, 2))
yield result""")
        grid = grid_from_list([[0, 0, 0, 0], [0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0]])
        result = execute(prog, grid)
        assert result.shape == (2, 2)


class TestDictLiterals:
    """apply_color_map({1: 3, 2: 7}, input) — dict args"""

    def test_dict_color_map(self):
        prog = parse_program("""bind result = apply_color_map({1: 3, 2: 7}, input)
yield result""")
        grid = grid_from_list([[1, 2, 0], [0, 1, 2]])
        result = execute(prog, grid)
        expected = grid_from_list([[3, 7, 0], [0, 3, 7]])
        assert grid_eq(result, expected)

    def test_multi_step_with_dict(self):
        prog = parse_program("""bind filled = flood_fill(input, (0, 0), 8)
bind result = apply_color_map({0: 4, 8: 0}, filled)
yield result""")
        grid = grid_from_list([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        result = execute(prog, grid)
        # (0,0) region fills with 8, then 0->4 and 8->0
        expected = grid_from_list([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        # Original 0s at (0,0) region become 8 then 0; original 0 at other places become 4
        # Actually: flood_fill changes (0,0) region to 8, then color_map {0:4, 8:0}
        # After flood: [[8,8,1],[8,8,1],[1,1,1]]
        # After map: 8->0, 0->4: [[0,0,1],[0,0,1],[1,1,1]] (no 0s left to map to 4)
        assert result.shape == (3, 3)


class TestEnumLiteralParsing:
    """Variable names don't collide with enum literals"""

    def test_frame_is_variable(self):
        prog = parse_program("""bind objects = find_objects(input)
bind frame = by_size_rank(0, objects)
bind c = get_color(frame)
yield c""")
        # 'frame' should be Ref not ZoneRole.FRAME
        from aria.types import Ref
        assert prog.steps[2].expr.args[0] == Ref(name="frame")

    def test_color_is_variable(self):
        prog = parse_program("""bind color = get_color(obj)
yield color""")
        assert prog.output == "color"

    def test_uppercase_enums_still_work(self):
        prog = parse_program("""bind sorted = sort_by(SIZE, DESC, objects)
yield sorted""")
        from aria.types import Literal, Property, SortDir
        assert prog.steps[0].expr.args[0].value == Property.SIZE
        assert prog.steps[0].expr.args[1].value == SortDir.DESC


class TestLambdaSyntax:
    """Parsed lambda syntax reaches higher-order runtime ops."""

    def test_cell_map_with_multi_arg_lambda(self):
        prog = parse_program("""bind result = cell_map(input, |row: INT, col: INT, val: INT| add(val, 1))
yield result""")
        grid = grid_from_list([[0, 1], [2, 3]])
        result = execute(prog, grid)
        expected = grid_from_list([[1, 2], [3, 4]])
        assert grid_eq(result, expected)

    def test_neighbor_map_with_lambda(self):
        prog = parse_program("""bind result = neighbor_map(input, |val: INT, neighbors: INT_LIST| val)
yield result""")
        grid = grid_from_list([[0, 1], [2, 3]])
        result = execute(prog, grid)
        assert grid_eq(result, grid)

    def test_untyped_binds_type_check_cleanly(self):
        prog = parse_program("""bind objects = find_objects(input)
bind sorted = sort_by(SIZE, DESC, objects)
bind colors = map_list(get_color(), sorted)
bind dims = dims_make(count(objects), 1)
bind result = fill_cells(new_grid(dims, 0), colors)
yield result""")
        errors = type_check(prog, {"input": Type.GRID, "ctx": Type.TASK_CTX})
        assert errors == []


class TestOverlayRobustness:
    """overlay handles shape mismatches and ObjectSet args"""

    def test_overlay_different_sizes(self):
        """Smaller top overlaid onto larger bottom"""
        from aria.runtime.ops import get_op
        _, impl = get_op("overlay")
        top = grid_from_list([[1, 2], [3, 4]])
        bottom = make_grid(4, 4)
        result = impl(top, bottom)
        assert result.shape == (4, 4)
        assert result[0, 0] == 1
        assert result[1, 1] == 4
        assert result[3, 3] == 0

    def test_overlay_empty_top(self):
        from aria.runtime.ops import get_op
        _, impl = get_op("overlay")
        top = make_grid(2, 2, fill=0)
        bottom = grid_from_list([[5, 5], [5, 5]])
        result = impl(top, bottom)
        assert grid_eq(result, bottom)


class TestStackRobustness:
    """stack_v/stack_h handle width/height mismatches by padding"""

    def test_stack_v_different_widths(self):
        from aria.runtime.ops import get_op
        _, impl = get_op("stack_v")
        a = grid_from_list([[1, 2, 3]])
        b = grid_from_list([[4, 5]])
        result = impl(a, b)
        assert result.shape == (2, 3)

    def test_stack_h_different_heights(self):
        from aria.runtime.ops import get_op
        _, impl = get_op("stack_h")
        a = grid_from_list([[1], [2], [3]])
        b = grid_from_list([[4], [5]])
        result = impl(a, b)
        assert result.shape == (3, 2)
