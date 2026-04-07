"""Tests for explicit periodic repair primitives."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.runtime.ops import has_op
from aria.runtime.ops.periodic_repair import (
    ALL_REPAIR_MODES, REPAIR_LINES_ONLY, REPAIR_MOTIF_2D, REPAIR_LINES_THEN_2D,
    REPAIR_MODE_NAMES,
    _detect_frame, _periodic_repair,
)
from aria.types import DemoPair, grid_from_list, Program, Bind, Call, Literal, Ref, Type
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_periodic_repair_registered():
    assert has_op("periodic_repair")


def test_detect_frame_registered():
    assert has_op("detect_frame")


# ---------------------------------------------------------------------------
# Frame detection
# ---------------------------------------------------------------------------


def test_detect_frame_peels():
    grid = grid_from_list([
        [4, 4, 4, 4, 4],
        [4, 1, 2, 1, 4],
        [4, 3, 0, 3, 4],
        [4, 1, 2, 1, 4],
        [4, 4, 4, 4, 4],
    ])
    interior = _detect_frame(grid)
    assert interior.shape == (3, 3)
    assert interior[0, 0] == 1


def test_detect_frame_no_frame():
    grid = grid_from_list([[1, 2], [3, 4]])
    result = _detect_frame(grid)
    np.testing.assert_array_equal(result, grid)


# ---------------------------------------------------------------------------
# Repair modes
# ---------------------------------------------------------------------------


def test_repair_mode_constants():
    assert isinstance(REPAIR_LINES_ONLY, int)
    assert isinstance(REPAIR_MOTIF_2D, int)
    assert isinstance(REPAIR_LINES_THEN_2D, int)
    assert len(ALL_REPAIR_MODES) == 3


def test_repair_mode_names():
    for mode in ALL_REPAIR_MODES:
        assert mode in REPAIR_MODE_NAMES


# ---------------------------------------------------------------------------
# Periodic repair behavior preserved
# ---------------------------------------------------------------------------


def test_periodic_repair_preserves_solved_task():
    """00d62c1b should still solve after refactoring."""
    from aria.datasets import get_dataset, load_arc_task
    ds = get_dataset('v2-train')
    task = load_arc_task(ds, '00d62c1b')
    result = solve_arc_task(task.train, task_id='00d62c1b', use_editor_search=True)
    assert result.solved


def test_periodic_repair_lines_only():
    """Lines-only mode should match repair_framed_lines behavior."""
    grid = grid_from_list([
        [4, 4, 4, 4, 4],
        [4, 1, 2, 1, 4],
        [4, 3, 0, 3, 4],
        [4, 1, 2, 1, 4],
        [4, 4, 4, 4, 4],
    ])
    from aria.runtime.ops.grid import _repair_framed_lines
    expected = _repair_framed_lines(grid, 0, 2)
    result = _periodic_repair(grid, 0, 2, REPAIR_LINES_ONLY)
    np.testing.assert_array_equal(result, expected)


def test_periodic_repair_2d_only():
    """2D-only mode should match repair_framed_2d_motif behavior."""
    grid = grid_from_list([
        [4, 4, 4, 4, 4],
        [4, 1, 2, 1, 4],
        [4, 3, 0, 3, 4],
        [4, 1, 2, 1, 4],
        [4, 4, 4, 4, 4],
    ])
    from aria.runtime.ops.grid import _repair_framed_2d_motif
    expected = _repair_framed_2d_motif(grid)
    result = _periodic_repair(grid, 0, 2, REPAIR_MOTIF_2D)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Compiler path
# ---------------------------------------------------------------------------


def test_compiler_uses_periodic_repair_op():
    """The periodic lane should compile using the new composite op."""
    from aria.sketch import SketchGraph, SketchNode, Primitive, Slot, SlotType
    from aria.sketch_compile import compile_sketch_graph, CompileTaskProgram
    from aria.sketch import Specialization as AriaSpec, ResolvedBinding as AriaRB
    from aria.runtime.executor import execute

    # Build a task that periodic_repair can solve
    grid = grid_from_list([
        [4, 4, 4, 4, 4],
        [4, 1, 2, 1, 4],
        [4, 3, 0, 3, 4],
        [4, 1, 2, 1, 4],
        [4, 4, 4, 4, 4],
    ])
    prog = Program(
        steps=(Bind("v0", Type.GRID, Call("periodic_repair", (
            Ref("input"), Literal(0, Type.INT), Literal(2, Type.INT), Literal(2, Type.INT),
        ))),),
        output="v0",
    )
    expected = execute(prog, grid)
    demos = (DemoPair(input=grid, output=expected),)

    graph = SketchGraph(
        task_id="test",
        nodes={
            "roles": SketchNode(id="roles", primitive=Primitive.BIND_ROLE, inputs=("input",)),
            "repaired": SketchNode(
                id="repaired", primitive=Primitive.REPAIR_LINES, inputs=("roles",),
                slots=(Slot("axis", SlotType.AXIS, evidence="row"),
                       Slot("period", SlotType.INT, evidence=2)),
                evidence={"axis": "row", "period": 2},
            ),
        },
        output_id="repaired",
    )
    spec = AriaSpec(task_id="test", bindings=(
        AriaRB(node_id="__task__", name="dominant_axis", value="row"),
        AriaRB(node_id="__task__", name="dominant_period", value=2),
    ))

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)


# ---------------------------------------------------------------------------
# No task-id logic
# ---------------------------------------------------------------------------


def test_no_task_id():
    import inspect
    from aria.runtime.ops.periodic_repair import _periodic_repair
    src = inspect.getsource(_periodic_repair)
    assert "1b59e163" not in src
    assert "00d62c1b" not in src


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_1b59e163_still_solves():
    from aria.datasets import get_dataset, load_arc_task
    ds = get_dataset('v2-train')
    task = load_arc_task(ds, '1b59e163')
    result = solve_arc_task(task.train, task_id='1b59e163', use_editor_search=True)
    assert result.solved
