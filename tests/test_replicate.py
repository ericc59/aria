"""Tests for the template replication lane."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.graph import CompileSuccess
from aria.runtime.ops import has_op
from aria.runtime.ops.replicate import (
    KEY_ADJACENT_DIFF_COLOR, KEY_ADJACENT_ANY,
    SOURCE_ERASE, SOURCE_KEEP,
    PLACE_ANCHOR_OFFSET, PLACE_CENTER,
    _replicate_templates, _find_anchor,
)
from aria.runtime.ops.relate_paint import _get_shapes_and_markers
from aria.runtime.ops.selection import _find_objects
from aria.types import DemoPair, grid_from_list, Program, Bind, Call, Literal, Ref, Type
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_replicate_templates_registered():
    assert has_op("replicate_templates")


# ---------------------------------------------------------------------------
# Exemplar extraction + anchor finding
# ---------------------------------------------------------------------------


def test_find_anchor_diff_color():
    """Anchor must be adjacent singleton with different color than shape."""
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 5, 0],  # marker c=5 adjacent to shape c=1
    ])
    shapes, singletons, bg = _get_shapes_and_markers(grid)
    assert len(shapes) == 1
    anchor, offset = _find_anchor(shapes[0], singletons, KEY_ADJACENT_DIFF_COLOR)
    assert anchor is not None
    assert anchor.color == 5
    assert anchor.color != shapes[0].color


def test_find_anchor_skips_same_color():
    """Anchor with diff_color rule should skip same-color singletons."""
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 5, 0],  # c=1 singleton (same as shape) AND c=5
    ])
    shapes, singletons, bg = _get_shapes_and_markers(grid)
    anchor, offset = _find_anchor(shapes[0], singletons, KEY_ADJACENT_DIFF_COLOR)
    assert anchor is not None
    assert anchor.color == 5  # should pick c=5, not c=1


def test_find_anchor_any_color():
    """With adjacent_any rule, any adjacent singleton works."""
    grid = grid_from_list([
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])
    shapes, singletons, bg = _get_shapes_and_markers(grid)
    # No singletons adjacent — should return None
    anchor, offset = _find_anchor(shapes[0], singletons, KEY_ADJACENT_ANY)
    assert anchor is None


# ---------------------------------------------------------------------------
# One-to-many cloning
# ---------------------------------------------------------------------------


def test_replicate_one_to_many():
    """One exemplar + anchor + two targets = two clones."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],  # anchor c=5 @ (2,1), target c=5 @ (2,5)
        [0, 0, 0, 0, 0, 0, 0],
    ])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET)

    # The shape c=1 should be cloned near the target marker
    clone_pixels = np.sum(result == 1)
    assert clone_pixels >= 2  # at least one clone

    # Source shape should be erased
    assert result[1, 1] != 1 or result[1, 2] != 1  # at least partially erased


def test_replicate_target_markers_persist():
    """Target singletons (clone destinations) should remain in the output.
    Anchor singletons should be erased."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET)
    assert result[2, 1] == 0  # anchor erased
    assert result[2, 5] == 5  # target persists


def test_replicate_source_keep():
    """With SOURCE_KEEP, original shapes should remain."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_KEEP, PLACE_ANCHOR_OFFSET)
    # Source should be preserved
    assert result[1, 1] == 1
    assert result[1, 2] == 1


def test_replicate_anchor_offset_placement():
    """Clone placement should preserve anchor offset."""
    # Shape at (1,1)-(1,2), anchor c=5 at (2,1), offset=(1,0)
    # Target c=5 at (2,5)
    # Clone should be at: row=2-1=1, col=5-0=5 -> (1,5)-(1,6)
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET)
    assert result[1, 5] == 1
    assert result[1, 6] == 1


def test_replicate_no_shapes_returns_copy():
    grid = grid_from_list([[0, 5, 3]])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET)
    np.testing.assert_array_equal(result, grid)


def test_replicate_no_singletons_returns_copy():
    grid = grid_from_list([[1, 1], [1, 0]])
    result = _replicate_templates(grid, KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET)
    np.testing.assert_array_equal(result, grid)


# ---------------------------------------------------------------------------
# Compiler path
# ---------------------------------------------------------------------------


def test_compiler_compiles_replication_graph():
    from aria.sketch import SketchGraph, SketchNode, Primitive
    from aria.sketch_compile import compile_sketch_graph, CompileTaskProgram
    from aria.sketch import Specialization as AriaSpec
    from aria.runtime.executor import execute

    graph = SketchGraph(
        task_id="test",
        nodes={
            "select": SketchNode(id="select", primitive=Primitive.SELECT_SUBSET, inputs=("input",)),
            "relate": SketchNode(id="relate", primitive=Primitive.APPLY_RELATION, inputs=("select",)),
            "paint": SketchNode(id="paint", primitive=Primitive.PAINT, inputs=("relate",)),
        },
        output_id="paint",
    )
    spec = AriaSpec(task_id="test", bindings=())

    # Build a task that replicate_templates solves
    inp = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    prog = Program(
        steps=(Bind("v0", Type.GRID, Call("replicate_templates", (
            Ref("input"), Literal(0, Type.INT), Literal(0, Type.INT), Literal(0, Type.INT),
        ))),),
        output="v0",
    )
    expected = execute(prog, inp)
    demos = (DemoPair(input=inp, output=expected),)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram), f"Expected success, got {result}"
    assert "replicate" in result.description


# ---------------------------------------------------------------------------
# Diagnostic: replication vs relocation
# ---------------------------------------------------------------------------


def test_replication_compiles_before_relocation():
    """When replication works, it should be preferred over relocation."""
    from aria.sketch import SketchGraph, SketchNode, Primitive
    from aria.sketch_compile import compile_sketch_graph, CompileTaskProgram
    from aria.sketch import Specialization as AriaSpec
    from aria.runtime.executor import execute

    graph = SketchGraph(
        task_id="test",
        nodes={
            "select": SketchNode(id="select", primitive=Primitive.SELECT_SUBSET, inputs=("input",)),
            "paint": SketchNode(id="paint", primitive=Primitive.PAINT, inputs=("select",)),
        },
        output_id="paint",
    )
    spec = AriaSpec(task_id="test", bindings=())

    inp = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    prog = Program(
        steps=(Bind("v0", Type.GRID, Call("replicate_templates", (
            Ref("input"), Literal(0, Type.INT), Literal(0, Type.INT), Literal(0, Type.INT),
        ))),),
        output="v0",
    )
    expected = execute(prog, inp)
    demos = (DemoPair(input=inp, output=expected),)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    # Should be the replicate lane, not the relocate lane
    assert "replicate" in result.description


# ---------------------------------------------------------------------------
# No task-specific logic
# ---------------------------------------------------------------------------


def test_no_task_id_checks():
    import inspect
    from aria.runtime.ops.replicate import _replicate_templates
    src = inspect.getsource(_replicate_templates)
    assert "1b59e163" not in src
    assert "task_id" not in src


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_solved_tasks_unaffected():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is True
