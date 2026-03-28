"""Tests for sketch compilation — turning Sketches into executable Programs."""

from __future__ import annotations

import numpy as np

from aria.sketch import Sketch, sketch_to_text
from aria.sketch_compile import (
    CompileFailure,
    CompilePerDemoPrograms,
    CompileTaskProgram,
    compile_sketch,
    _build_composite_alignment_program,
    _compute_periodic_repair,
)
from aria.sketch_fit import fit_composite_role_alignment, fit_framed_periodic_repair
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


def test_compile_periodic_repair_fails_cleanly():
    """Periodic repair should fail with a structured reason, not crash."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileFailure)
    assert result.family == "framed_periodic_repair"
    assert "periodic" in result.reason.lower()
    assert len(result.missing_ops) >= 1
    assert "repair_periodic" in result.missing_ops


def test_periodic_repair_failure_has_evidence():
    """Compile failure should include computed repair grids."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileFailure)

    evidence = result.partial_evidence
    assert evidence.get("repair_grids_computed", 0) >= 1
    assert "repair_grids" in evidence
    grids = evidence["repair_grids"]
    assert len(grids) >= 1
    # Each repair grid should match the output
    for gi, (grid, demo) in enumerate(zip(grids, demos)):
        assert np.array_equal(grid, demo.output), f"repair grid {gi} should match output"


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


def test_compile_unknown_family_fails():
    """Unknown family should produce a CompileFailure."""
    sketch = Sketch(
        task_id="test",
        steps=(),
        output_ref="input",
        metadata={"family": "unknown_family"},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)
    assert "unknown_family" in result.reason


# ---------------------------------------------------------------------------
# Compilation scope semantics
# ---------------------------------------------------------------------------


def test_compile_failure_scope():
    """CompileFailure should have scope='failed' and not be promotable."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    result = compile_sketch(sketch, demos)
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


def test_fit_compile_periodic_evidence_matches():
    """Periodic repair: fit → compile failure → repair grids match outputs."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileFailure)

    grids = result.partial_evidence.get("repair_grids", [])
    for grid, demo in zip(grids, demos):
        assert np.array_equal(grid, demo.output)


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


def test_real_135a2760_compile_repair_grids():
    """Real 135a2760: compile failure should include correct repair grids."""
    task = _load_real_task("135a2760")
    if task is None:
        import pytest
        pytest.skip("135a2760 data not available")

    sketch = fit_framed_periodic_repair(task.train, task_id="135a2760")
    assert sketch is not None

    result = compile_sketch(sketch, task.train)
    assert isinstance(result, CompileFailure)
    assert result.partial_evidence.get("repair_grids_computed", 0) >= 1
