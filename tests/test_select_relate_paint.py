"""Tests for the explicit extract/match/place/paint lane."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.graph import CompileSuccess
from aria.runtime.ops.relate_paint import (
    ALL_MATCH_RULES, ALL_ALIGNS, MATCH_NAMES, ALIGN_NAMES,
    MATCH_SHAPE_NEAREST, MATCH_MARKER_NEAREST, MATCH_COLOR,
    MATCH_ORDERED_ROW, MATCH_MUTUAL_NEAREST, MATCH_SIZE,
    ALIGN_CENTER, ALIGN_AT_MARKER, ALIGN_MARKER_INTERIOR,
    _extract_shapes, _extract_markers, _relocate_objects,
    _match_and_place, _select_relate_paint,
    match_objects, _get_shapes_and_markers,
)
from aria.runtime.ops import has_op
from aria.types import DemoPair, grid_from_list, Program, Bind, Call, Literal, Ref, Type
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Ops registered
# ---------------------------------------------------------------------------


def test_relocate_objects_registered():
    assert has_op("relocate_objects")


def test_match_and_place_registered():
    assert has_op("match_and_place")


def test_extract_ops_registered():
    assert has_op("extract_shapes")
    assert has_op("extract_markers")


# ---------------------------------------------------------------------------
# Extract layer
# ---------------------------------------------------------------------------


def test_extract_shapes_erases_multi_pixel():
    grid = grid_from_list([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 5]])
    result = _extract_shapes(grid)
    assert result[3, 3] == 5
    assert result[1, 1] == 0


def test_extract_markers_erases_single_pixel():
    grid = grid_from_list([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 5]])
    result = _extract_markers(grid)
    assert result[1, 1] == 1
    assert result[3, 3] == 0


# ---------------------------------------------------------------------------
# Matching layer — explicit rules produce different pairings
# ---------------------------------------------------------------------------


def _sample_grid():
    return grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 3],
    ])


def test_match_shape_nearest():
    shapes, markers, bg = _get_shapes_and_markers(_sample_grid())
    pairs = match_objects(shapes, markers, MATCH_SHAPE_NEAREST)
    assert len(pairs) >= 1
    for s, m in pairs:
        assert s.size > 1
        assert m.size == 1


def test_match_marker_nearest():
    shapes, markers, bg = _get_shapes_and_markers(_sample_grid())
    pairs = match_objects(shapes, markers, MATCH_MARKER_NEAREST)
    assert len(pairs) >= 1


def test_match_ordered_row():
    shapes, markers, bg = _get_shapes_and_markers(_sample_grid())
    pairs = match_objects(shapes, markers, MATCH_ORDERED_ROW)
    assert len(pairs) >= 1
    # Shapes and markers should be paired in row order
    if len(pairs) >= 2:
        assert pairs[0][0].bbox[1] <= pairs[1][0].bbox[1]


def test_match_mutual_nearest():
    shapes, markers, bg = _get_shapes_and_markers(_sample_grid())
    pairs = match_objects(shapes, markers, MATCH_MUTUAL_NEAREST)
    # Mutual nearest is stricter — may produce fewer pairs
    for s, m in pairs:
        assert s.size > 1


def test_match_color():
    shapes, markers, bg = _get_shapes_and_markers(_sample_grid())
    pairs = match_objects(shapes, markers, MATCH_COLOR)
    assert len(pairs) >= 1


def test_different_match_rules_all_valid():
    """All match rules should produce valid output (same grid size)."""
    grid = _sample_grid()
    for rule in ALL_MATCH_RULES:
        r = _relocate_objects(grid, rule, ALIGN_CENTER)
        assert r.shape == grid.shape


# ---------------------------------------------------------------------------
# Match rules are explicit and typed
# ---------------------------------------------------------------------------


def test_match_rule_constants_are_ints():
    for r in ALL_MATCH_RULES:
        assert isinstance(r, int)


def test_match_names_cover_all():
    for r in ALL_MATCH_RULES:
        assert r in MATCH_NAMES


# ---------------------------------------------------------------------------
# Alignment layer (preserved from before)
# ---------------------------------------------------------------------------


def test_alignment_at_marker():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ])
    result = _relocate_objects(grid, MATCH_SHAPE_NEAREST, ALIGN_AT_MARKER)
    assert result[4, 4] == 1


def test_different_aligns_produce_different_output():
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
    ])
    r1 = _relocate_objects(grid, MATCH_SHAPE_NEAREST, ALIGN_CENTER)
    r2 = _relocate_objects(grid, MATCH_SHAPE_NEAREST, ALIGN_AT_MARKER)
    assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Compiler searches over match rules + alignments
# ---------------------------------------------------------------------------


def test_compiler_searches_match_and_alignment():
    """Compiler should find the right match rule + alignment combination."""
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

    # Use a task where relocate works: two shapes, two markers, 1:1 relocation
    # (replication won't match because output has same count as input)
    inp = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
    ])
    # relocate with at_marker: shape moves to marker pos
    prog = Program(
        steps=(Bind("v0", Type.GRID, Call("relocate_objects", (
            Ref("input"), Literal(MATCH_SHAPE_NEAREST, Type.INT), Literal(ALIGN_AT_MARKER, Type.INT),
        ))),),
        output="v0",
    )
    expected = execute(prog, inp)
    demos = (DemoPair(input=inp, output=expected),)

    result = compile_sketch_graph(graph, spec, demos)
    # Should compile via either replication or relocation lane
    assert isinstance(result, CompileTaskProgram), f"Expected success, got: {result}"


def test_compiler_failure_reports_match_info():
    from aria.sketch import SketchGraph, SketchNode, Primitive
    from aria.sketch_compile import compile_sketch_graph, CompileFailure
    from aria.sketch import Specialization as AriaSpec

    graph = SketchGraph(
        task_id="test",
        nodes={
            "select": SketchNode(id="select", primitive=Primitive.SELECT_SUBSET, inputs=("input",)),
            "paint": SketchNode(id="paint", primitive=Primitive.PAINT, inputs=("select",)),
        },
        output_id="paint",
    )
    spec = AriaSpec(task_id="test", bindings=())
    demos = (DemoPair(input=grid_from_list([[1, 2, 3]]), output=grid_from_list([[9, 8, 7]])),)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileFailure)
    assert "combinations" in result.reason or "verified failed" in result.reason


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def test_match_and_place_is_alias():
    grid = _sample_grid()
    np.testing.assert_array_equal(
        _match_and_place(grid, 0, 0),
        _relocate_objects(grid, 0, 0),
    )


def test_select_relate_paint_compat():
    grid = _sample_grid()
    np.testing.assert_array_equal(
        _select_relate_paint(grid),
        _relocate_objects(grid, MATCH_SHAPE_NEAREST, ALIGN_CENTER),
    )


# ---------------------------------------------------------------------------
# No family labels
# ---------------------------------------------------------------------------


def test_no_family_labels_in_dispatch():
    import inspect
    from aria.sketch_compile import _can_compile_select_relate_paint, _compile_select_relate_paint
    src = inspect.getsource(_can_compile_select_relate_paint) + inspect.getsource(_compile_select_relate_paint)
    assert "framed_periodic" not in src.lower()
    assert "composite_alignment" not in src.lower()


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
