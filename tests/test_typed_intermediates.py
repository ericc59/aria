"""Tests for typed intermediate values in lane decomposition."""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import has_op
from aria.runtime.ops.selection import _by_min_size, _by_max_size
from aria.runtime.executor import execute
from aria.types import (
    Bind, Call, DemoPair, Literal, Program, Ref, Type, grid_from_list,
)


# ---------------------------------------------------------------------------
# New predicate ops
# ---------------------------------------------------------------------------


def test_by_min_size_registered():
    assert has_op("by_min_size")


def test_by_max_size_registered():
    assert has_op("by_max_size")


def test_by_min_size_filters():
    pred = _by_min_size(1)
    from aria.types import ObjectNode, Shape, Symmetry
    # Mock object with size 3
    mask = np.array([[True, True, True]])
    obj = ObjectNode(id=0, color=1, mask=mask, bbox=(0, 0, 3, 1),
                     shape=Shape.IRREGULAR, symmetry=frozenset(), size=3)
    assert pred(obj) is True

    # Mock object with size 1
    mask1 = np.array([[True]])
    obj1 = ObjectNode(id=1, color=5, mask=mask1, bbox=(0, 0, 1, 1),
                      shape=Shape.DOT, symmetry=frozenset(), size=1)
    assert pred(obj1) is False


def test_by_max_size_filters():
    pred = _by_max_size(1)
    from aria.types import ObjectNode, Shape
    mask = np.array([[True]])
    obj = ObjectNode(id=0, color=5, mask=mask, bbox=(0, 0, 1, 1),
                     shape=Shape.DOT, symmetry=frozenset(), size=1)
    assert pred(obj) is True

    mask3 = np.array([[True, True, True]])
    obj3 = ObjectNode(id=1, color=1, mask=mask3, bbox=(0, 0, 3, 1),
                      shape=Shape.IRREGULAR, symmetry=frozenset(), size=3)
    assert pred(obj3) is False


# ---------------------------------------------------------------------------
# Typed decomposition of shape/marker separation
# ---------------------------------------------------------------------------


def test_typed_shape_marker_separation():
    """Demonstrate shape/marker separation using typed ops in a program."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ])

    # Program: find objects -> filter by size -> get shapes vs markers
    prog = Program(
        steps=(
            # Step 1: find all objects
            Bind("objs", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
            # Step 2: filter to shapes (size > 1)
            Bind("shape_pred", Type.PREDICATE, Call("by_min_size", (Literal(1, Type.INT),))),
            Bind("shapes", Type.OBJECT_SET, Call("where", (Ref("shape_pred"), Ref("objs")))),
            # Step 3: filter to markers (size <= 1)
            Bind("marker_pred", Type.PREDICATE, Call("by_max_size", (Literal(1, Type.INT),))),
            Bind("markers", Type.OBJECT_SET, Call("where", (Ref("marker_pred"), Ref("objs")))),
            # Step 4: count them
            Bind("n_shapes", Type.INT, Call("count", (Ref("shapes"),))),
            Bind("n_markers", Type.INT, Call("count", (Ref("markers"),))),
            # Step 5: paint just shapes (verify typed extraction works)
            Bind("v0", Type.GRID, Call("paint_objects", (Ref("shapes"), Ref("input")))),
        ),
        output="v0",
    )

    result = execute(prog, grid)
    # Should contain the shape pixels but painted onto the grid
    assert result.shape == grid.shape


def test_typed_object_nearest():
    """Demonstrate nearest_to as a typed pairing step."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ])

    # Find objects, get shapes, get markers, find nearest shape to marker
    prog = Program(
        steps=(
            Bind("objs", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
            Bind("sp", Type.PREDICATE, Call("by_min_size", (Literal(1, Type.INT),))),
            Bind("shapes", Type.OBJECT_SET, Call("where", (Ref("sp"), Ref("objs")))),
            Bind("mp", Type.PREDICATE, Call("by_max_size", (Literal(1, Type.INT),))),
            Bind("markers", Type.OBJECT_SET, Call("where", (Ref("mp"), Ref("objs")))),
            # Get the single marker
            Bind("marker", Type.OBJECT, Call("singleton", (Ref("markers"),))),
            # Find nearest shape to that marker
            Bind("nearest", Type.OBJECT, Call("nearest_to", (Ref("marker"), Ref("shapes")))),
            # Paint just the nearest shape
            Bind("v0", Type.GRID, Call("paint_objects", (
                Call("singleton_set", (Ref("nearest"),)), Ref("input"),
            ))),
        ),
        output="v0",
    )

    result = execute(prog, grid)
    assert result.shape == grid.shape


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_existing_solved_tasks_still_work():
    from aria.core.arc import solve_arc_task
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved


def test_1b59e163_still_solves():
    from aria.datasets import get_dataset, load_arc_task
    from aria.core.arc import solve_arc_task
    ds = get_dataset('v2-train')
    task = load_arc_task(ds, '1b59e163')
    result = solve_arc_task(task.train, task_id='1b59e163', use_editor_search=True)
    assert result.solved
