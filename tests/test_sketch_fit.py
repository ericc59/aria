"""Tests for sketch fitting — proposing sketches from demo decompositions."""

from __future__ import annotations

import numpy as np

from aria.sketch import PrimitiveFamily, Sketch, SketchGraph, sketch_to_text
from aria.sketch_fit import (
    fit_canvas_construction,
    fit_composite_role_alignment,
    fit_framed_periodic_repair,
    fit_sketches,
    specialize_sketch,
    _detect_row_period,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Helpers: synthetic task fixtures
# ---------------------------------------------------------------------------


def _framed_periodic_task():
    """Task with framed regions containing periodic patterns with violations.
    The output fixes the violations to complete the periodic pattern.
    Both demos have the same structure: row-periodic content inside nested frames."""
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


def _composite_alignment_task():
    """Task with composites (center+frame) that align to anchor.
    Colors rotate across demos."""
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


def _non_periodic_task():
    """Task with no periodic structure — should NOT fit periodic repair."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
            ]),
        ),
    )


def _no_composite_task():
    """Task with no composites — should NOT fit composite alignment."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 2],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 0],
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# Period detection tests
# ---------------------------------------------------------------------------


def test_detect_row_period_basic():
    row = np.array([1, 3, 1, 3, 1, 3, 3, 3, 1], dtype=np.uint8)
    ev = _detect_row_period(row)
    assert ev is not None
    assert ev.period == 2
    assert ev.n_violations >= 1
    assert ev.pattern == (1, 3)


def test_detect_row_period_perfect():
    """Perfect period has no violations — returns None (nothing to repair)."""
    row = np.array([1, 3, 1, 3, 1, 3], dtype=np.uint8)
    ev = _detect_row_period(row)
    assert ev is None


def test_detect_row_period_too_short():
    row = np.array([1, 2, 3], dtype=np.uint8)
    ev = _detect_row_period(row)
    assert ev is None


def test_detect_row_period_mostly_violations():
    """Too many violations — returns None."""
    row = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    ev = _detect_row_period(row)
    assert ev is None


# ---------------------------------------------------------------------------
# Framed periodic repair fitting
# ---------------------------------------------------------------------------


def test_fit_framed_periodic_repair():
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test_periodic")
    assert sketch is not None
    assert sketch.task_id == "test_periodic"
    # Should emit decomposed primitives: PEEL_FRAME, PARTITION_GRID, REPAIR_LINES
    from aria.sketch import Primitive
    primitives = sketch.primitive_pattern
    assert "PEEL_FRAME" in primitives
    assert "PARTITION_GRID" in primitives
    assert "REPAIR_LINES" in primitives
    # Should detect period 2
    repair_step = next(s for s in sketch.steps if s.primitive == Primitive.REPAIR_LINES)
    period_slot = next(s for s in repair_step.slots if s.name == "period")
    assert period_slot.evidence == 2


def test_fit_framed_periodic_has_roles():
    """Fitted sketch uses role variables, not literal colors."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None
    role_names = {rv.name for rv in sketch.role_vars}
    assert "bg" in role_names
    assert "frame" in role_names


def test_fit_framed_periodic_evidence():
    """Sketch carries structured evidence."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("family") == "framed_periodic_repair"
    assert sketch.metadata.get("dominant_period") is not None
    assert sketch.metadata.get("dominant_axis") is not None


def test_fit_framed_periodic_rejects_non_periodic():
    """Non-periodic task should not fit."""
    demos = _non_periodic_task()
    sketch = fit_framed_periodic_repair(demos)
    assert sketch is None


def test_fit_framed_periodic_rejects_dims_change():
    """Dims-change task should not fit."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[1], [2]]),
        ),
    )
    sketch = fit_framed_periodic_repair(demos)
    assert sketch is None


# ---------------------------------------------------------------------------
# Composite role alignment fitting
# ---------------------------------------------------------------------------


def test_fit_composite_alignment():
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test_composite")
    assert sketch is not None
    assert sketch.task_id == "test_composite"
    assert any(s.primitive == PrimitiveFamily.COMPOSITE_ROLE_ALIGNMENT for s in sketch.steps)
    assert any(s.primitive == PrimitiveFamily.FIND_COMPOSITES for s in sketch.steps)


def test_fit_composite_has_roles():
    """Fitted sketch uses role variables for center/frame/anchor."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None
    role_names = {rv.name for rv in sketch.role_vars}
    assert "center" in role_names
    assert "anchor" in role_names
    assert "frame_cc" in role_names


def test_fit_composite_detects_color_rotation():
    """Sketch metadata reports color rotation."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("roles_rotate") is True
    assert "colors rotate" in sketch.description


def test_fit_composite_evidence():
    """Sketch carries structured evidence about composites and axes."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("family") == "composite_role_alignment"
    assert sketch.metadata.get("n_composites") >= 1
    assert sketch.metadata.get("per_demo_axis") is not None


def test_fit_composite_rejects_no_composites():
    """Task without composites should not fit."""
    demos = _no_composite_task()
    sketch = fit_composite_role_alignment(demos)
    assert sketch is None


def test_fit_composite_rejects_no_anchor():
    """Composites without an isolated anchor should not fit."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 8, 8, 8, 0],
                [0, 8, 4, 8, 0],
                [0, 8, 8, 8, 0],
            ]),
            output=grid_from_list([
                [0, 8, 8, 8, 0],
                [0, 8, 4, 8, 0],
                [0, 8, 8, 8, 0],
            ]),
        ),
    )
    sketch = fit_composite_role_alignment(demos)
    assert sketch is None


# ---------------------------------------------------------------------------
# Combined fit_sketches
# ---------------------------------------------------------------------------


def test_fit_sketches_periodic():
    demos = _framed_periodic_task()
    sketches = fit_sketches(demos, task_id="test")
    families = [s.metadata.get("family") for s in sketches]
    assert "framed_periodic_repair" in families


def test_fit_sketches_composite():
    demos = _composite_alignment_task()
    sketches = fit_sketches(demos, task_id="test")
    families = [s.metadata.get("family") for s in sketches]
    assert "composite_role_alignment" in families


def test_fit_sketches_neither():
    demos = _non_periodic_task()
    sketches = fit_sketches(demos)
    assert len(sketches) == 0


# ---------------------------------------------------------------------------
# Validation and printing
# ---------------------------------------------------------------------------


def test_fitted_sketches_validate():
    """All fitted sketches should pass structural validation."""
    for demos in [_framed_periodic_task(), _composite_alignment_task()]:
        sketches = fit_sketches(demos, task_id="test")
        for s in sketches:
            errors = s.validate()
            assert errors == [], f"validation errors: {errors}"


def test_fitted_sketches_print():
    """Fitted sketches should produce readable text."""
    for demos in [_framed_periodic_task(), _composite_alignment_task()]:
        sketches = fit_sketches(demos, task_id="test")
        for s in sketches:
            text = sketch_to_text(s)
            assert len(text) > 50
            assert "sketch test" in text


# ---------------------------------------------------------------------------
# Real task regression (if data available)
# ---------------------------------------------------------------------------


def _load_real_task(task_id: str):
    """Try to load a real ARC task, skip if not available."""
    try:
        from aria.datasets import get_dataset, load_arc_task
        ds = get_dataset("v2-eval")
        return load_arc_task(ds, task_id)
    except Exception:
        return None


def test_real_135a2760_fits_periodic():
    """Real task 135a2760 should fit framed periodic repair."""
    task = _load_real_task("135a2760")
    if task is None:
        import pytest
        pytest.skip("135a2760 data not available")

    sketch = fit_framed_periodic_repair(task.train, task_id="135a2760")
    assert sketch is not None, "135a2760 should fit framed periodic repair"
    assert sketch.metadata["family"] == "framed_periodic_repair"
    assert sketch.metadata["dominant_period"] >= 2


def test_real_581f7754_has_composites():
    """Real task 581f7754 has composites in decomposition (prerequisite for fitting)."""
    task = _load_real_task("581f7754")
    if task is None:
        import pytest
        pytest.skip("581f7754 data not available")

    from aria.decomposition import decompose_composites, detect_bg
    for di, d in enumerate(task.train):
        bg = detect_bg(d.input)
        dec = decompose_composites(d.input, bg)
        assert len(dec.composites) >= 1, f"Demo {di} should have composites"
        assert dec.anchor is not None, f"Demo {di} should have anchor"
        assert dec.center_color is not None, f"Demo {di} should have center color"

    # Color rotation: center/frame colors differ across demos
    center_colors = set()
    frame_colors = set()
    for d in task.train:
        bg = detect_bg(d.input)
        dec = decompose_composites(d.input, bg)
        center_colors.add(dec.center_color)
        frame_colors.add(dec.frame_color)
    assert len(center_colors) >= 2, "Center colors should rotate"
    assert len(frame_colors) >= 2, "Frame colors should rotate"


# ---------------------------------------------------------------------------
# Specialization pass
# ---------------------------------------------------------------------------


def test_specialize_periodic_sketch():
    """Specialization extracts axis and period from periodic sketch."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.task_id == "test"
    assert spec.get("__task__", "dominant_axis") in ("row", "col")
    assert isinstance(spec.get("__task__", "dominant_period"), int)
    assert spec.get("__task__", "frame_colors") is not None


def test_specialize_resolves_slot_evidence():
    """Specialization resolves slot evidence into bindings."""
    demos = _framed_periodic_task()
    sketch = fit_framed_periodic_repair(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    # The "repaired" node has axis and period slots with evidence
    repair_bindings = spec.bindings_for_node("repaired")
    names = {b.name for b in repair_bindings}
    assert "axis" in names
    assert "period" in names


def test_specialize_composite_sketch():
    """Specialization works on composite alignment sketches too."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.task_id == "test"
    assert len(spec.bindings) > 0


def test_specialize_composite_resolves_per_demo_roles():
    """Specialization resolves per-demo center/frame/anchor for composites."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    per_demo_roles = spec.get("__relation__", "per_demo_roles")
    assert per_demo_roles is not None
    assert len(per_demo_roles) == 2

    # Demo 0: center=4, frame=8, bg=0
    assert per_demo_roles[0]["center"] == 4
    assert per_demo_roles[0]["frame"] == 8
    assert per_demo_roles[0]["bg"] == 0

    # Demo 1: center=1, frame=3, bg=0
    assert per_demo_roles[1]["center"] == 1
    assert per_demo_roles[1]["frame"] == 3
    assert per_demo_roles[1]["bg"] == 0


def test_specialize_composite_resolves_per_demo_axis():
    """Specialization resolves per-demo alignment axis for composites."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    per_demo_axis = spec.get("__relation__", "per_demo_axis")
    assert per_demo_axis is not None
    assert len(per_demo_axis) == 2
    assert all(a in ("row", "col") for a in per_demo_axis)


def test_specialize_composite_resolves_structural_invariants():
    """Specialization captures n_composites and roles_rotate."""
    demos = _composite_alignment_task()
    sketch = fit_composite_role_alignment(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    n_composites = spec.get("__relation__", "n_composites")
    assert n_composites is not None
    assert n_composites >= 1

    roles_rotate = spec.get("__relation__", "roles_rotate")
    assert roles_rotate is not None


def test_specialize_captures_bg_consensus():
    """When all demos have the same bg, specialization resolves it."""
    # Both demos have bg=3 in this fixture? No — different bgs.
    # Use a task where bg is consistent.
    demos = _composite_alignment_task()  # bg=0 for both demos
    sketch = fit_composite_role_alignment(demos, task_id="test")
    if sketch is None:
        return

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    # Check if bg was resolved for any node with BG role
    bg_bindings = [b for b in spec.bindings if b.name == "bg"]
    if bg_bindings:
        assert bg_bindings[0].value == 0
        assert bg_bindings[0].source == "consensus"


# ---------------------------------------------------------------------------
# Canvas construction fitting
# ---------------------------------------------------------------------------


def test_fit_canvas_tile():
    """Tile task fits canvas construction."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
    )
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "tile"
    from aria.sketch import Primitive
    assert any(s.primitive == Primitive.CONSTRUCT_CANVAS for s in sketch.steps)


def test_fit_canvas_upscale():
    """Upscale task fits canvas construction."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 1, 2, 2], [1, 1, 2, 2],
                [3, 3, 4, 4], [3, 3, 4, 4],
            ]),
        ),
    )
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "upscale"


def test_fit_canvas_crop():
    """Crop task fits canvas construction."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0],
            ]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None
    assert sketch.metadata.get("strategy") == "crop"


def test_fit_canvas_rejects_same_dims():
    """Same-dims tasks should not fit canvas construction."""
    demos = _framed_periodic_task()
    sketch = fit_canvas_construction(demos)
    assert sketch is None


def test_fit_canvas_rejects_no_pattern():
    """Random dims change with no detectable strategy should not fit."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8], [7, 6]]),
        ),
    )
    sketch = fit_canvas_construction(demos)
    assert sketch is None


def test_specialize_canvas_tile():
    """Canvas specialization resolves tile parameters from demos."""
    demos = (
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
    sketch = fit_canvas_construction(demos, task_id="test")
    assert sketch is not None

    graph = SketchGraph.from_sketch(sketch)
    spec = specialize_sketch(graph, demos)

    assert spec.get("__canvas__", "strategy") == "tile"
    assert spec.get("__canvas__", "tile_rows") == 2
    assert spec.get("__canvas__", "tile_cols") == 2
    assert spec.get("__canvas__", "output_dims") == (4, 4)
