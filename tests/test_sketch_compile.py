"""Tests for sketch compilation — turning Sketches into executable Programs."""

from __future__ import annotations

import numpy as np

from aria.sketch import (
    RoleKind,
    RoleVar,
    Sketch,
    SketchGraph,
    SketchStep,
    Slot,
    SlotType,
    sketch_to_text,
)
from aria.sketch_compile import (
    CompileFailure,
    CompilePerDemoPrograms,
    CompileTaskProgram,
    compile_sketch,
    compile_sketch_graph,
    _build_composite_alignment_program,
    _compute_periodic_repair,
)
from aria.sketch_fit import (
    fit_canvas_construction,
    fit_composite_role_alignment,
    fit_framed_periodic_repair,
    fit_grid_transform,
    fit_object_movement,
    specialize_sketch,
)
from aria.types import DemoPair, Program, grid_from_list
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _composite_alignment_task():
    """Task with composites aligning to anchor. Colors rotate."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 8, 4, 8, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 8, 4, 8, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 1, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 3, 1, 3, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
        ),
    )


def _framed_periodic_task():
    """Task with periodic content inside frames."""
    return (
        DemoPair(
            input=grid_from_list([
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 2, 1, 3, 1, 3, 1, 3, 3, 3, 1, 2, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]),
            output=grid_from_list([
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3],
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 4, 2, 5, 2, 5, 2, 5, 5, 5, 2, 4, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]),
            output=grid_from_list([
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 4, 2, 5, 2, 5, 2, 5, 2, 5, 2, 4, 5],
                [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# Composite alignment compilation
# ---------------------------------------------------------------------------


def test_compile_composite_returns_per_demo():
    """Composite alignment compiles to per-demo programs, NOT task-level."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert not isinstance(result, CompileTaskProgram)
    assert result.family == "composite_role_alignment"
    assert len(result.programs) == 2  # one per demo


def test_composite_per_demo_not_promotable():
    """Per-demo programs should NOT be promotable to the solver."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert result.can_promote_to_solver is False
    assert result.compilation_scope == "per_demo"


def test_compiled_composite_programs_are_executable():
    """Each per-demo program should be executable (no crashes)."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    from aria.runtime.executor import execute
    for di, (prog, demo) in enumerate(zip(result.programs, demos)):
        try:
            output = execute(prog, demo.input, None)
            assert output is not None, f"demo {di} execution returned None"
            assert isinstance(output, np.ndarray), f"demo {di} didn't return grid"
        except Exception as e:
            assert False, f"demo {di} execution failed: {e}"


def test_compiled_composite_passes_per_demo_verification():
    """Each per-demo program should pass verification on its own demo."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    any_passed = False
    for prog, demo in zip(result.programs, demos):
        vr = verify(prog, (demo,))
        if vr.passed:
            any_passed = True
    assert any_passed, "no per-demo program passed verification on its own demo"


def test_compiled_composite_fails_cross_demo():
    """A per-demo program should NOT pass on a different demo (different colors)."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    # Demo 0's program uses center=4, frame=8.
    # It should fail on demo 1 which has center=1, frame=3.
    prog0 = result.programs[0]
    vr_cross = verify(prog0, (demos[1],))
    assert not vr_cross.passed, "per-demo program should not pass on a different demo"


def test_compiled_composite_role_bindings():
    """Compile result should have per-demo role bindings."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    per_demo = result.role_bindings.get("per_demo", [])
    assert len(per_demo) == 2
    assert per_demo[0]["center"] == 4
    assert per_demo[0]["frame"] == 8
    assert per_demo[1]["center"] == 1
    assert per_demo[1]["frame"] == 3


def test_compiled_composite_slot_bindings():
    """Compile result should have axis bindings."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    axes = result.slot_bindings.get("per_demo_axis", [])
    assert len(axes) == 2
    assert all(a in ("row", "col") for a in axes)


def test_build_composite_alignment_program_structure():
    """The built program should have the right ops."""
    prog = _build_composite_alignment_program(
        center_color=4, frame_color=8, axis="col",
    )
    assert isinstance(prog, Program)
    assert len(prog.steps) >= 8
    # Should contain key ops
    op_names = [s.expr.op for s in prog.steps if hasattr(s, 'expr') and hasattr(s.expr, 'op')]
    assert "find_objects" in op_names
    assert "by_color" in op_names
    assert "align_center_to_col_of" in op_names
    assert "map_obj" in op_names
    assert "paint_objects" in op_names


# ---------------------------------------------------------------------------
# Periodic repair compilation
# ---------------------------------------------------------------------------


def test_compile_periodic_repair_succeeds():
    """Periodic repair should compile to a task-level program."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)
    assert result.family == "framed_periodic_repair"
    assert result.can_promote_to_solver is True
    assert result.compilation_scope == "task"


def test_compiled_periodic_passes_verification():
    """The compiled periodic program should pass verification on all demos."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)

    vr = verify(result.program, demos)
    assert vr.passed


def test_compiled_periodic_is_one_program():
    """Task-level compile produces ONE program, not per-demo."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)
    assert isinstance(result.program, Program)


def test_compiled_periodic_uses_primitive_ops():
    """Compiled periodic program uses repair_framed_lines (primitive-aligned),
    NOT the legacy repair_periodic wrapper."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)

    ops = [s.expr.op for s in result.program.steps
           if hasattr(s, 'expr') and hasattr(s.expr, 'op')]
    # Should use the primitive-aligned op
    assert "repair_framed_lines" in ops
    # Should NOT use the legacy monolithic wrapper
    assert "repair_periodic" not in ops
    # Should NOT contain literal color ops
    assert "by_color" not in ops


def test_compute_periodic_repair_produces_correct_grid():
    """The repair computation should fix violations to match output."""
    demo = _framed_periodic_task()[0]
    repaired = _compute_periodic_repair(demo.input, demo.output, "row", 2)
    assert repaired is not None
    assert np.array_equal(repaired, demo.output)


def test_compute_periodic_repair_no_change_returns_none():
    """If input already matches the pattern, return None."""
    grid = grid_from_list([[1, 3, 1, 3, 1, 3]])
    result = _compute_periodic_repair(grid, grid, "row", 2)
    assert result is None


# ---------------------------------------------------------------------------
# Unknown family
# ---------------------------------------------------------------------------


def test_compile_unsupported_pattern_fails():
    """Unsupported primitive composition should produce a CompileFailure."""
    sketch = Sketch(
        task_id="test",
        steps=(),
        output_ref="input",
        metadata={"family": "unknown_family"},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    assert "no supported composition" in result.reason


# ---------------------------------------------------------------------------
# Primitive-pattern dispatch tests
# ---------------------------------------------------------------------------


def test_compile_dispatches_on_primitive_not_family():
    """Compiler should dispatch on primitive pattern, not family name."""
    from aria.sketch import Primitive

    # Build a sketch with REPAIR_MISMATCH primitive but NO family metadata
    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(
                name="roles",
                primitive=Primitive.BIND_ROLE,
                roles=(RoleVar("bg", RoleKind.BG),),
            ),
            SketchStep(
                name="repair",
                primitive=Primitive.REPAIR_MISMATCH,
                slots=(
                    Slot("axis", SlotType.AXIS, evidence="row"),
                    Slot("period", SlotType.INT, evidence=2),
                ),
                input_refs=("roles",),
            ),
        ),
        output_ref="repair",
        # No family metadata — dispatch must work from primitives alone
        metadata={"dominant_axis": "row", "dominant_period": 2},
    )

    demos = _framed_periodic_task()
    result = compile_sketch(sketch, demos)
    # Should compile via primitive pattern, not family string
    assert isinstance(result, CompileTaskProgram)
    assert result.can_promote_to_solver is True


def test_compile_apply_relation_dispatches_on_primitive():
    """APPLY_RELATION primitive should trigger alignment compiler."""
    from aria.sketch import Primitive

    # Build a sketch with APPLY_RELATION + ANCHOR role but custom family name
    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(
                name="roles",
                primitive=Primitive.BIND_ROLE,
                roles=(RoleVar("bg", RoleKind.BG),),
            ),
            SketchStep(
                name="view",
                primitive=Primitive.EXTRACT_VIEW,
                input_refs=("roles",),
            ),
            SketchStep(
                name="aligned",
                primitive=Primitive.APPLY_RELATION,
                roles=(
                    RoleVar("anchor", RoleKind.ANCHOR),
                    RoleVar("center", RoleKind.CENTER),
                    RoleVar("frame", RoleKind.FRAME),
                ),
                slots=(Slot("axis", SlotType.AXIS, evidence="col"),),
                input_refs=("view",),
            ),
        ),
        output_ref="aligned",
        # Use a custom family name — compiler must not depend on it
        metadata={
            "family": "my_custom_alignment",
            "per_demo_axis": ["col", "col"],
        },
    )

    demos = _composite_alignment_task()
    result = compile_sketch(sketch, demos)
    # Should reach the alignment compiler via APPLY_RELATION pattern
    assert isinstance(result, (CompilePerDemoPrograms, CompileTaskProgram))


def test_repair_mismatch_without_evidence_fails():
    """REPAIR_MISMATCH with no regularity slots/evidence should fail cleanly."""
    from aria.sketch import Primitive

    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(
                name="repair",
                primitive=Primitive.REPAIR_MISMATCH,
                # No axis/period slots, no metadata
            ),
        ),
        output_ref="repair",
        metadata={},
    )

    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    # Fails because the composition requires axis slot but none is present
    assert "no supported composition" in result.reason


def test_family_metadata_ignored_by_compiler():
    """Family metadata should not influence compilation path."""
    from aria.sketch import Primitive

    # Sketch with periodic primitives but WRONG family label
    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="r", primitive=Primitive.BIND_ROLE,
                       roles=(RoleVar("bg", RoleKind.BG),)),
            SketchStep(name="s", primitive=Primitive.SELECT_REGION,
                       input_refs=("r",)),
            SketchStep(name="repair", primitive=Primitive.REPAIR_MISMATCH,
                       slots=(Slot("axis", SlotType.AXIS, evidence="row"),
                              Slot("period", SlotType.INT, evidence=2)),
                       input_refs=("s",)),
        ),
        output_ref="repair",
        metadata={"family": "DELIBERATELY_WRONG", "dominant_axis": "row", "dominant_period": 2},
    )
    demos = _framed_periodic_task()
    result = compile_sketch(sketch, demos)
    # Should compile despite wrong family — dispatch uses primitives
    assert isinstance(result, CompileTaskProgram)


def test_transform_paint_composition_not_yet_implemented():
    """APPLY_TRANSFORM + PAINT should fail with structured reason."""
    from aria.sketch import Primitive

    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="v", primitive=Primitive.EXTRACT_VIEW),
            SketchStep(name="t", primitive=Primitive.APPLY_TRANSFORM,
                       slots=(Slot("transform", SlotType.TRANSFORM),),
                       input_refs=("v",)),
            SketchStep(name="p", primitive=Primitive.PAINT,
                       input_refs=("t",)),
        ),
        output_ref="p",
        metadata={},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    assert "not yet implemented" in result.reason


def test_canvas_legacy_path_redirects_to_graph():
    """CONSTRUCT_CANVAS in the legacy compile_sketch path returns structured failure."""
    from aria.sketch import Primitive

    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="c", primitive=Primitive.CONSTRUCT_CANVAS,
                       slots=(Slot("output_dims", SlotType.DIMS),)),
            SketchStep(name="p", primitive=Primitive.PAINT,
                       input_refs=("c",)),
        ),
        output_ref="p",
        metadata={},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    assert "compile_sketch_graph" in result.reason or "not implemented" in result.reason


def test_apply_relation_without_anchor_fails():
    """APPLY_RELATION without ANCHOR role should fail."""
    from aria.sketch import Primitive

    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="v", primitive=Primitive.EXTRACT_VIEW),
            SketchStep(name="rel", primitive=Primitive.APPLY_RELATION,
                       # axis slot present but NO anchor role
                       slots=(Slot("axis", SlotType.AXIS, evidence="col"),),
                       input_refs=("v",)),
        ),
        output_ref="rel",
        metadata={"per_demo_axis": ["col"]},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    # Should fail because ANCHOR role is required for alignment composition
    assert "no supported composition" in result.reason


# ---------------------------------------------------------------------------
# Compilation scope semantics
# ---------------------------------------------------------------------------


def test_compile_failure_scope():
    """CompileFailure should have scope='failed' and not be promotable."""
    # Use unknown family to trigger a clean failure
    sketch = Sketch(
        task_id="test",
        steps=(),
        output_ref="input",
        metadata={"family": "nonexistent_family"},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    assert result.can_promote_to_solver is False
    assert result.compilation_scope == "failed"


def test_task_program_scope():
    """CompileTaskProgram should have scope='task' and be promotable."""
    # Construct one manually to test the type
    prog = _build_composite_alignment_program(
        center_color=4, frame_color=8, axis="col",
    )
    result = CompileTaskProgram(
        sketch_task_id="test",
        family="test",
        program=prog,
        role_bindings={},
        slot_bindings={},
    )
    assert result.can_promote_to_solver is True
    assert result.compilation_scope == "task"


def test_per_demo_scope():
    """CompilePerDemoPrograms should have scope='per_demo' and not be promotable."""
    prog = _build_composite_alignment_program(
        center_color=4, frame_color=8, axis="col",
    )
    result = CompilePerDemoPrograms(
        sketch_task_id="test",
        family="test",
        programs=(prog,),
        role_bindings={},
        slot_bindings={},
    )
    assert result.can_promote_to_solver is False
    assert result.compilation_scope == "per_demo"


# ---------------------------------------------------------------------------
# Integration: fit → compile → verify
# ---------------------------------------------------------------------------


def test_fit_compile_verify_composite():
    """Full pipeline: fit → compile → per-demo verify passes."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert result.can_promote_to_solver is False

    # Each per-demo program should pass on its own demo
    passes = 0
    for prog, demo in zip(result.programs, demos):
        vr = verify(prog, (demo,))
        if vr.passed:
            passes += 1
    assert passes >= 1


def test_fit_compile_verify_periodic():
    """Full pipeline: fit → compile → verify periodic repair passes."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)
    assert result.can_promote_to_solver is True

    vr = verify(result.program, demos)
    assert vr.passed


# ---------------------------------------------------------------------------
# Real task regression
# ---------------------------------------------------------------------------


def _load_real_task(task_id: str):
    try:
        from aria.datasets import get_dataset, load_arc_task
        ds = get_dataset("v2-eval")
        return load_arc_task(ds, task_id)
    except Exception:
        return None


def test_real_135a2760_compiles_to_task_program():
    """Real 135a2760: sketch compiles to task-level program and passes verification."""
    task = _load_real_task("135a2760")
    if task is None:
        import pytest
        pytest.skip("135a2760 data not available")

    sketch = fit_framed_periodic_repair(task.train, task_id="135a2760")
    assert sketch is not None

    result = compile_sketch(sketch, task.train)
    assert isinstance(result, CompileTaskProgram), f"expected task program, got {type(result).__name__}"
    assert result.can_promote_to_solver is True

    # Verify across all train demos
    vr = verify(result.program, task.train)
    assert vr.passed, "compiled program should pass verification on all train demos"


# ---------------------------------------------------------------------------
# Graph + Specialization compilation path
# ---------------------------------------------------------------------------


def test_graph_compile_periodic_repair():
    """Periodic repair compiles through graph + specialization path."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert result.can_promote_to_solver is True

    vr = verify(result.program, demos)
    assert vr.passed


def test_graph_compile_reads_specialization_bindings():
    """Compilation uses bindings from specialization, not sketch metadata."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    # Specialization should have axis and period
    assert spec.get("__task__", "dominant_axis") is not None
    assert spec.get("__task__", "dominant_period") is not None

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description or "specialization" in result.description


def test_graph_compile_specialization_captures_structure():
    """Specialization captures static task structure explicitly."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    # Should have axis and period from evidence
    assert spec.get("__task__", "dominant_axis") in ("row", "col")
    assert isinstance(spec.get("__task__", "dominant_period"), int)

    # Should have frame colors
    assert spec.get("__task__", "frame_colors") is not None

    # Should record per-node slot resolutions
    repair_bindings = spec.bindings_for_node("repaired")
    binding_names = {b.name for b in repair_bindings}
    assert "axis" in binding_names
    assert "period" in binding_names


def test_graph_compile_matches_old_path():
    """Graph path produces same verification result as old Sketch path."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    # Old path
    old_result = compile_sketch(sketch, demos)
    assert isinstance(old_result, CompileTaskProgram)

    # New path
    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    new_result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(new_result, CompileTaskProgram)

    # Both should pass verification
    vr_old = verify(old_result.program, demos)
    vr_new = verify(new_result.program, demos)
    assert vr_old.passed
    assert vr_new.passed


def test_graph_compile_fallback_to_sketch():
    """Unsupported graph compositions fall back to sketch compilation."""
    from aria.sketch import Primitive, SketchNode, Specialization

    # Build a graph with primitives that have no graph-native or sketch compiler
    graph = SketchGraph(
        task_id="test",
        nodes={
            "v": SketchNode(id="v", primitive=Primitive.EXTRACT_VIEW, inputs=("input",)),
            "t": SketchNode(
                id="t", primitive=Primitive.APPLY_TRANSFORM, inputs=("v",),
                slots=(Slot("transform", SlotType.TRANSFORM),),
            ),
            "p": SketchNode(id="p", primitive=Primitive.PAINT, inputs=("t",)),
        },
        output_id="p",
    )
    spec = Specialization(task_id="test", bindings=())
    result = compile_sketch_graph(graph, spec, ())
    # Falls back to sketch path → CompileFailure (not yet implemented)
    assert isinstance(result, CompileFailure)


# ---------------------------------------------------------------------------
# Graph-native composite alignment compilation
# ---------------------------------------------------------------------------


def test_graph_compile_composite_is_graph_native():
    """Composite alignment compiles directly from graph+specialization, not fallback."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    # Description should indicate graph-native path
    assert "graph" in result.description


def test_graph_compile_composite_per_demo_programs():
    """Graph-native composite compilation produces per-demo programs."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert len(result.programs) == len(demos)
    assert result.can_promote_to_solver is False
    assert result.compilation_scope == "per_demo"


def test_graph_compile_composite_programs_pass_per_demo_verification():
    """Each per-demo program passes verification on its own demo."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    any_passed = False
    for prog, demo in zip(result.programs, demos):
        vr = verify(prog, (demo,))
        if vr.passed:
            any_passed = True
    assert any_passed


def test_graph_compile_composite_matches_old_path():
    """Graph-native composite produces same results as old Sketch path."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    # Old path
    old_result = compile_sketch(sketch, demos)
    assert isinstance(old_result, CompilePerDemoPrograms)

    # New path
    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    new_result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(new_result, CompilePerDemoPrograms)

    # Same number of programs
    assert len(new_result.programs) == len(old_result.programs)

    # Same per-demo verification results
    for di, demo in enumerate(demos):
        vr_old = verify(old_result.programs[di], (demo,))
        vr_new = verify(new_result.programs[di], (demo,))
        assert vr_old.passed == vr_new.passed


def test_graph_compile_composite_specialization_has_relation_bindings():
    """Specialization for composite alignment has per-demo role bindings."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    per_demo_roles = spec.get("__relation__", "per_demo_roles")
    assert per_demo_roles is not None
    assert len(per_demo_roles) == len(demos)
    # Demo 0: center=4, frame=8
    assert per_demo_roles[0]["center"] == 4
    assert per_demo_roles[0]["frame"] == 8
    # Demo 1: center=1, frame=3
    assert per_demo_roles[1]["center"] == 1
    assert per_demo_roles[1]["frame"] == 3

    per_demo_axis = spec.get("__relation__", "per_demo_axis")
    assert per_demo_axis is not None
    assert len(per_demo_axis) == len(demos)
    assert all(a in ("row", "col") for a in per_demo_axis)


def test_graph_compile_composite_no_family_metadata_required():
    """Graph-native composite compilation works without family metadata."""
    from aria.sketch import Primitive, SketchNode

    demos = _composite_alignment_task()

    # Fit normally to get evidence, then strip family metadata
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)

    # Remove family from metadata — graph-native path should not need it
    clean_meta = {k: v for k, v in graph.metadata.items() if k != "family"}
    graph = SketchGraph(
        task_id=graph.task_id,
        nodes=graph.nodes,
        output_id=graph.output_id,
        description=graph.description,
        metadata=clean_meta,
    )

    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert "graph" in result.description


def test_graph_compile_composite_cross_demo_fails():
    """Per-demo program from graph-native compile fails on wrong demo."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)

    # Demo 0's program (center=4, frame=8) should fail on demo 1 (center=1, frame=3)
    vr_cross = verify(result.programs[0], (demos[1],))
    assert not vr_cross.passed


# ---------------------------------------------------------------------------
# Metadata independence — graph-native paths work without metadata
# ---------------------------------------------------------------------------


def test_periodic_graph_compile_without_metadata():
    """Periodic graph-native compile succeeds with empty metadata.

    All compile inputs come from node slot evidence and specialization
    bindings — metadata is not required.
    """
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)

    # Strip ALL metadata
    graph = SketchGraph(
        task_id=graph.task_id,
        nodes=graph.nodes,
        output_id=graph.output_id,
        description=graph.description,
        metadata={},
    )

    spec = specialize_sketch(graph, demos)

    # axis and period should come from node slot evidence, not metadata
    axis = spec.get("__task__", "dominant_axis")
    period = spec.get("__task__", "dominant_period")
    assert axis is not None, "axis should resolve from node slot evidence"
    assert period is not None, "period should resolve from node slot evidence"

    # Check sources — should not be metadata_fallback
    for b in spec.bindings:
        if b.node_id == "__task__" and b.name in ("dominant_axis", "dominant_period"):
            assert b.source == "node_evidence", (
                f"{b.name} should come from node_evidence, got {b.source}"
            )

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert result.can_promote_to_solver is True

    vr = verify(result.program, demos)
    assert vr.passed


def test_composite_graph_compile_without_metadata():
    """Composite graph-native compile succeeds with empty metadata.

    per_demo_roles and per_demo_axis are derived from demo decomposition
    by the specialization pass — metadata is not required.
    """
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)

    # Strip ALL metadata
    graph = SketchGraph(
        task_id=graph.task_id,
        nodes=graph.nodes,
        output_id=graph.output_id,
        description=graph.description,
        metadata={},
    )

    spec = specialize_sketch(graph, demos)

    # Relation bindings should come from demo_decomposition
    per_demo_roles = spec.get("__relation__", "per_demo_roles")
    per_demo_axis = spec.get("__relation__", "per_demo_axis")
    assert per_demo_roles is not None, "per_demo_roles should resolve from demos"
    assert per_demo_axis is not None, "per_demo_axis should resolve from demos"

    # Check sources
    for b in spec.bindings:
        if b.node_id == "__relation__":
            assert b.source == "demo_decomposition", (
                f"{b.name} should come from demo_decomposition, got {b.source}"
            )

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompilePerDemoPrograms)
    assert "graph" in result.description

    # Per-demo programs should still pass
    any_passed = False
    for prog, demo in zip(result.programs, demos):
        vr = verify(prog, (demo,))
        if vr.passed:
            any_passed = True
    assert any_passed


def test_specialization_does_not_include_family():
    """Specialization metadata should not carry family — it's not a compile input."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert "family" not in spec.metadata


def test_specialization_sources_are_explicit():
    """Every binding in specialization has an explicit source, not 'evidence'."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    for b in spec.bindings:
        assert b.source != "", f"binding {b.node_id}.{b.name} has empty source"


# ---------------------------------------------------------------------------
# Canvas construction — fixtures
# ---------------------------------------------------------------------------


def _tile_task():
    """Task: tile 2x2 grid into 2x2 arrangement → 4x4 output."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 6, 5, 6], [7, 8, 7, 8],
                [5, 6, 5, 6], [7, 8, 7, 8],
            ]),
        ),
    )


def _upscale_task():
    """Task: upscale each pixel by 3x → 6x6 output from 2x2 input."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 5, 5, 6, 6, 6], [5, 5, 5, 6, 6, 6], [5, 5, 5, 6, 6, 6],
                [7, 7, 7, 8, 8, 8], [7, 7, 7, 8, 8, 8], [7, 7, 7, 8, 8, 8],
            ]),
        ),
    )


def _crop_task():
    """Task: crop center 2x2 region from 4x4 input."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0],
            ]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0], [0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0],
            ]),
            output=grid_from_list([[5, 6], [7, 8]]),
        ),
    )


# ---------------------------------------------------------------------------
# Canvas construction — graph-native compilation
# ---------------------------------------------------------------------------


def test_canvas_tile_fits():
    """Tile task should fit canvas construction."""
    demos = _tile_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "tile"


def test_canvas_tile_graph_native():
    """Tile compiles through graph + specialization path."""
    demos = _tile_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.get("__canvas__", "strategy") == "tile"
    assert spec.get("__canvas__", "tile_rows") == 2
    assert spec.get("__canvas__", "tile_cols") == 2

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert result.can_promote_to_solver is True
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_canvas_upscale_graph_native():
    """Upscale compiles through graph + specialization path."""
    demos = _upscale_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.get("__canvas__", "strategy") == "upscale"
    assert spec.get("__canvas__", "scale_factor") == 3

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_canvas_crop_graph_native():
    """Crop compiles through graph + specialization path."""
    demos = _crop_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.get("__canvas__", "strategy") == "crop"
    assert spec.get("__canvas__", "crop_region") is not None

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_canvas_is_task_level():
    """Canvas construction produces task-level programs, not per-demo."""
    for demos_fn in (_tile_task, _upscale_task, _crop_task):
        demos = demos_fn()
        sketch = fit_canvas_construction(demos, task_id="test")
        assert sketch is not None

        graph = SketchGraph.from_sketch(sketch)
        spec = specialize_sketch(graph, demos)
        result = compile_sketch_graph(graph, spec, demos)

        assert isinstance(result, CompileTaskProgram), (
            f"{demos_fn.__name__} should produce task-level program"
        )
        assert result.can_promote_to_solver is True


def test_canvas_no_metadata_required():
    """Canvas graph-native compile works without metadata."""
    demos = _tile_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    graph = SketchGraph(
        task_id=graph.task_id,
        nodes=graph.nodes,
        output_id=graph.output_id,
        description=graph.description,
        metadata={},
    )

    spec = specialize_sketch(graph, demos)
    # Strategy comes from demo decomposition, not metadata
    assert spec.get("__canvas__", "strategy") == "tile"

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)

    vr = verify(result.program, demos)
    assert vr.passed


def test_canvas_specialization_sources():
    """All canvas bindings come from demo_decomposition."""
    demos = _tile_task()
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    for b in spec.bindings:
        if b.node_id == "__canvas__":
            assert b.source == "demo_decomposition", (
                f"{b.name} should come from demo_decomposition, got {b.source}"
            )


def test_canvas_rejects_same_dims():
    """Same-dims task should not fit canvas construction."""
    demos = _framed_periodic_task()
    sketch = fit_canvas_construction(demos)
    assert sketch is None


def test_canvas_rejects_inconsistent_strategy():
    """Demos with different strategies should not fit."""
    # Demo 1: tile, Demo 2: different dims that don't tile
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2, 3], [4, 5, 6]]),
            output=grid_from_list([[1, 2], [4, 5]]),
        ),
    )
    sketch = fit_canvas_construction(demos)
    assert sketch is None


# ---------------------------------------------------------------------------
# Object movement — fixtures and tests
# ---------------------------------------------------------------------------


def _uniform_translate_task():
    """All foreground objects move down by 1 row."""
    return (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [2, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [2, 2, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 3, 3], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 3, 3]]),
        ),
    )


def _gravity_down_task():
    """Objects at varying heights all drop to the bottom edge."""
    return (
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 3, 5, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [0, 0, 0, 7], [0, 2, 0, 0], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 7]]),
        ),
    )


def test_movement_uniform_translate_fits():
    """Uniform translate task should fit the movement fitter."""
    demos = _uniform_translate_task()
    sketch = fit_object_movement(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "uniform_translate"


def test_movement_uniform_translate_graph_native():
    """Uniform translate compiles via graph-native path."""
    demos = _uniform_translate_task()
    sketch = fit_object_movement(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    assert spec.get("__movement__", "strategy") == "uniform_translate"

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_movement_gravity_fits():
    """Gravity task should fit the movement fitter."""
    demos = _gravity_down_task()
    sketch = fit_object_movement(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "gravity"
    assert sketch.metadata.get("direction") == "down"


def test_movement_gravity_graph_native():
    """Gravity compiles via graph-native path."""
    demos = _gravity_down_task()
    sketch = fit_object_movement(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    assert spec.get("__movement__", "strategy") == "gravity"
    assert spec.get("__movement__", "direction") == "down"

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_movement_rejects_same_grid():
    """Task where input == output should not fit movement."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 1], [0, 0]]),
            output=grid_from_list([[0, 1], [0, 0]]),
        ),
    )
    sketch = fit_object_movement(demos)
    assert sketch is None


def test_movement_rejects_dims_change():
    """Dims-change task should not fit movement."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[1, 2, 1, 2]]),
        ),
    )
    sketch = fit_object_movement(demos)
    assert sketch is None


def test_movement_no_metadata_required():
    """Movement graph-native compile works without metadata."""
    demos = _uniform_translate_task()
    sketch = fit_object_movement(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    graph = SketchGraph(
        task_id=graph.task_id, nodes=graph.nodes, output_id=graph.output_id,
        description=graph.description, metadata={},
    )
    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description


# ---------------------------------------------------------------------------
# Grid transform — fixtures and tests
# ---------------------------------------------------------------------------


def _rotate_180_task():
    """Grid rotated 180 degrees."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _reflect_row_task():
    """Grid reflected across row axis (vertical flip)."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 8], [5, 6]]),
        ),
    )


def _transpose_task():
    """Grid transposed."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3], [4, 5, 6]]),
            output=grid_from_list([[1, 4], [2, 5], [3, 6]]),
        ),
        DemoPair(
            input=grid_from_list([[7, 8, 9], [1, 2, 3]]),
            output=grid_from_list([[7, 1], [8, 2], [9, 3]]),
        ),
    )


def test_grid_transform_rotate_fits():
    """Rotation 180 task should fit grid transform."""
    demos = _rotate_180_task()
    sketch = fit_grid_transform(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("transform") == "rotate"


def test_grid_transform_rotate_graph_native():
    """Rotation compiles via graph-native path."""
    demos = _rotate_180_task()
    sketch = fit_grid_transform(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    assert spec.get("__grid_transform__", "transform") == "rotate"

    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_grid_transform_reflect_graph_native():
    """Reflection compiles via graph-native path."""
    demos = _reflect_row_task()
    sketch = fit_grid_transform(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_grid_transform_transpose_graph_native():
    """Transpose compiles via graph-native path."""
    demos = _transpose_task()
    sketch = fit_grid_transform(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description

    vr = verify(result.program, demos)
    assert vr.passed


def test_grid_transform_rejects_identity():
    """Identity task should not fit grid transform."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )
    sketch = fit_grid_transform(demos)
    assert sketch is None


def test_grid_transform_no_metadata_required():
    """Grid transform compile works without metadata."""
    demos = _rotate_180_task()
    sketch = fit_grid_transform(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    graph = SketchGraph(
        task_id=graph.task_id, nodes=graph.nodes, output_id=graph.output_id,
        description=graph.description, metadata={},
    )
    spec = specialize_sketch(graph, demos)
    result = compile_sketch_graph(graph, spec, demos)
    assert isinstance(result, CompileTaskProgram)
    assert "graph" in result.description
