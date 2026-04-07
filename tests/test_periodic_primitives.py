"""Tests for the decomposed periodic-repair primitives."""

from __future__ import annotations

import numpy as np

import aria.runtime  # register ops
from aria.runtime.ops import get_op, has_op
from aria.runtime.ops.grid import (
    _detect_frame_color,
    _find_best_separators_axis,
    _infer_line_motif,
    _partition_grid,
    _peel_frame,
    _repair_grid_periodic_lines,
    _repair_line_from_motif,
)
from aria.sketch import Primitive, Sketch, SketchStep, Slot, SlotType, RoleVar, RoleKind
from aria.sketch_compile import compile_sketch, CompileTaskProgram, CompileFailure
from aria.types import DemoPair, grid_from_list
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# peel_frame
# ---------------------------------------------------------------------------


def test_peel_frame_strips_border():
    grid = grid_from_list([
        [3, 3, 3, 3, 3],
        [3, 1, 2, 3, 3],
        [3, 4, 5, 6, 3],
        [3, 3, 3, 3, 3],
    ])
    result = _peel_frame(grid)
    assert result.shape == (2, 3)
    assert int(result[0, 0]) == 1
    assert int(result[1, 2]) == 6


def test_peel_frame_no_frame():
    grid = grid_from_list([[1, 2], [3, 4]])
    result = _peel_frame(grid)
    assert np.array_equal(result, grid)


def test_peel_frame_registered():
    assert has_op("peel_frame")
    _, fn = get_op("peel_frame")
    grid = grid_from_list([[5, 5, 5], [5, 1, 5], [5, 5, 5]])
    result = fn(grid)
    assert result.shape == (1, 1)
    assert int(result[0, 0]) == 1


# ---------------------------------------------------------------------------
# partition_grid
# ---------------------------------------------------------------------------


def test_partition_grid_no_separators():
    grid = grid_from_list([[1, 2], [3, 4]])
    cells = _partition_grid(grid)
    assert len(cells) == 1
    assert np.array_equal(cells[0][2], grid)


def test_partition_grid_row_separators():
    grid = grid_from_list([
        [5, 5, 5],
        [1, 2, 3],
        [4, 2, 6],
        [5, 5, 5],
        [7, 8, 9],
        [1, 2, 3],
        [5, 5, 5],
    ])
    cells = _partition_grid(grid)
    assert len(cells) >= 2
    # Should have sub-cells between separator rows


def test_partition_grid_col_separators():
    grid = grid_from_list([
        [5, 1, 2, 5, 3, 4, 5],
        [5, 5, 6, 5, 7, 8, 5],
    ])
    cells = _partition_grid(grid)
    assert len(cells) >= 2


# ---------------------------------------------------------------------------
# infer_line_motif
# ---------------------------------------------------------------------------


def test_infer_line_motif_period2():
    line = np.array([1, 3, 1, 3, 1, 3, 3, 3, 1], dtype=np.uint8)
    result = _infer_line_motif(line, 2)
    assert result is not None
    pattern, period, violations, lo, hi = result
    assert period == 2
    assert pattern == (3, 1)  # content starts at lo=1, so phase 0 is 3
    assert len(violations) >= 1


def test_infer_line_motif_period3():
    line = np.array([2, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 2], dtype=np.uint8)
    result = _infer_line_motif(line, 2)
    assert result is not None
    pattern, period, violations, lo, hi = result
    assert period == 3  # auto-detects period 3 even with hint 2


def test_infer_line_motif_perfect_no_violations():
    line = np.array([1, 3, 1, 3, 1, 3], dtype=np.uint8)
    result = _infer_line_motif(line, 2)
    assert result is None  # perfect pattern → nothing to repair


def test_infer_line_motif_too_short():
    line = np.array([1, 2], dtype=np.uint8)
    assert _infer_line_motif(line, 2) is None


# ---------------------------------------------------------------------------
# repair_line_from_motif
# ---------------------------------------------------------------------------


def test_repair_line_from_motif():
    line = np.array([0, 3, 1, 3, 1, 3, 3, 3, 1, 0], dtype=np.uint8)
    # Content at lo=1: [3,1,3,1,3,3,3,1]. Violation at content pos 5 (value 3, expected 1)
    result = _repair_line_from_motif(line, (3, 1), 2, [5], lo=1)
    assert int(result[6]) == 1  # position 1+5=6 repaired from 3 to 1


# ---------------------------------------------------------------------------
# repair_grid_lines
# ---------------------------------------------------------------------------


def test_repair_grid_lines_row():
    grid = grid_from_list([
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 3, 1, 3, 1, 3, 3, 3, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    expected = grid_from_list([
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 3, 1, 3, 1, 3, 1, 3, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    result = _repair_grid_periodic_lines(grid, 0, 2)
    assert np.array_equal(result, expected)


def test_repair_grid_lines_col_fallback():
    """If row repair finds nothing, column repair should fire."""
    grid = grid_from_list([
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],  # col 3 violation: should be 1
        [0, 0, 0, 0, 0, 0],
    ])
    result = _repair_grid_periodic_lines(grid, 0, 2)
    # Row repair: rows are alternating uniform [0,1,0,1,0,1] / [0,0,0,0,0,0]
    # Row 4 has violation at col 3: 0 instead of 1. Should be fixed.
    assert int(result[4, 3]) == 1


def test_repair_grid_lines_registered():
    assert has_op("repair_grid_lines")


# ---------------------------------------------------------------------------
# Compiler: decomposed primitive composition
# ---------------------------------------------------------------------------


def _periodic_demos():
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


def test_compile_decomposed_periodic_no_family():
    """Decomposed primitive composition compiles without family label."""
    demos = _periodic_demos()
    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="r", primitive=Primitive.BIND_ROLE,
                       roles=(RoleVar("bg", RoleKind.BG),)),
            SketchStep(name="f", primitive=Primitive.PEEL_FRAME,
                       input_refs=("r",)),
            SketchStep(name="p", primitive=Primitive.PARTITION_GRID,
                       input_refs=("f",)),
            SketchStep(name="repair", primitive=Primitive.REPAIR_LINES,
                       slots=(Slot("axis", SlotType.AXIS, evidence="row"),
                              Slot("period", SlotType.INT, evidence=2)),
                       input_refs=("p",)),
        ),
        output_ref="repair",
        metadata={"dominant_axis": "row", "dominant_period": 2},
    )
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)
    vr = verify(result.program, demos)
    assert vr.passed


def test_compile_repair_lines_without_axis_fails():
    """REPAIR_LINES without axis slot should not match any composition."""
    sketch = Sketch(
        task_id="test",
        steps=(
            SketchStep(name="repair", primitive=Primitive.REPAIR_LINES),
        ),
        output_ref="repair",
        metadata={},
    )
    result = compile_sketch(sketch, ())
    assert isinstance(result, CompileFailure)


def test_fitted_sketch_uses_decomposed_primitives():
    """The fitter now emits decomposed primitives, not monolithic families."""
    from aria.sketch_fit import fit_framed_periodic_repair
    demos = _periodic_demos()
    sketch = fit_framed_periodic_repair(demos, "test")
    assert sketch is not None
    pattern = sketch.primitive_pattern
    assert "PEEL_FRAME" in pattern
    assert "PARTITION_GRID" in pattern
    assert "REPAIR_LINES" in pattern
    # Old family names should not appear as primitives
    assert "REGION_PERIODIC_REPAIR" not in pattern


def test_fitted_decomposed_sketch_compiles_and_verifies():
    """Full pipeline: fit → compile → verify with decomposed primitives."""
    from aria.sketch_fit import fit_framed_periodic_repair
    demos = _periodic_demos()
    sketch = fit_framed_periodic_repair(demos, "test")
    result = compile_sketch(sketch, demos)
    assert isinstance(result, CompileTaskProgram)
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


def test_real_135a2760_compiles():
    """135a2760 should compile via decomposed primitives."""
    task = _load_real_task("135a2760")
    if task is None:
        import pytest; pytest.skip("data not available")
    from aria.sketch_fit import fit_framed_periodic_repair
    sketch = fit_framed_periodic_repair(task.train, "135a2760")
    assert sketch is not None
    result = compile_sketch(sketch, task.train)
    assert isinstance(result, CompileTaskProgram)


def test_real_2c181942_rejects():
    """2c181942 should still reject cleanly."""
    task = _load_real_task("2c181942")
    if task is None:
        import pytest; pytest.skip("data not available")
    from aria.sketch_fit import fit_framed_periodic_repair
    sketch = fit_framed_periodic_repair(task.train, "2c181942")
    if sketch is not None:
        result = compile_sketch(sketch, task.train)
        # Should either fail or produce a non-verifying program
        if isinstance(result, CompileTaskProgram):
            vr = verify(result.program, task.train)
            assert not vr.passed, "2c181942 should not pass verification"
