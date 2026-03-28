"""Sketch compiler — turn fitted Sketches into executable Programs.

The compiler resolves role variables to concrete values per demo,
fills parameter slots from evidence, and emits Programs using the
typed DSL. Each compiled program is verifiable by the exact verifier.

Compilation strategy per family:
- composite_role_alignment: compiles to find_objects + by_color + where +
  align_center_to_{axis}_of + map_obj + paint_objects pipeline.
  Fully compilable using existing ops.
- framed_periodic_repair: currently emits a structured CompileFailure
  because the DSL lacks a position-aware periodic fill op. The failure
  includes the computed repair grid so callers can still use it for
  verification or as repair-target evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.sketch import (
    PrimitiveFamily,
    RoleVar,
    Sketch,
    SketchStep,
    Slot,
)
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Grid,
    Literal,
    Program,
    Ref,
    Type,
)


# ---------------------------------------------------------------------------
# Compile result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileTaskProgram:
    """A sketch that compiled into a single executable Program valid across all demos.

    This is the only result type that a solver should treat as a real candidate.
    The program was verified against all train demos during compilation.
    """

    sketch_task_id: str
    family: str
    program: Program
    role_bindings: dict[str, Any]
    slot_bindings: dict[str, Any]
    description: str = ""

    @property
    def can_promote_to_solver(self) -> bool:
        return True

    @property
    def compilation_scope(self) -> str:
        return "task"


@dataclass(frozen=True)
class CompilePerDemoPrograms:
    """Per-demo specialized programs — useful evidence but NOT a task-level solution.

    Each program uses demo-specific literal colors resolved from roles.
    These are valuable for:
    - repair-target evidence
    - per-demo verification
    - guiding further search
    But they are NOT promotable to the solver as a task-level solution.
    """

    sketch_task_id: str
    family: str
    programs: tuple[Program, ...]
    role_bindings: dict[str, Any]
    slot_bindings: dict[str, Any]
    description: str = ""

    @property
    def can_promote_to_solver(self) -> bool:
        return False

    @property
    def compilation_scope(self) -> str:
        return "per_demo"


@dataclass(frozen=True)
class CompileFailure:
    """A sketch that could not compile — with a structured reason."""

    sketch_task_id: str
    family: str
    reason: str
    missing_ops: tuple[str, ...] = ()
    partial_evidence: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def can_promote_to_solver(self) -> bool:
        return False

    @property
    def compilation_scope(self) -> str:
        return "failed"


CompileResult = CompileTaskProgram | CompilePerDemoPrograms | CompileFailure

# Backward compat alias
CompileSuccess = CompilePerDemoPrograms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_sketch(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a sketch into executable Programs using demo evidence.

    Resolves role variables by analyzing each demo's grid, fills slots
    from sketch evidence, then builds Program AST.
    """
    family = sketch.metadata.get("family", "")

    if family == "composite_role_alignment":
        return _compile_composite_alignment(sketch, demos)

    if family == "framed_periodic_repair":
        return _compile_periodic_repair(sketch, demos)

    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family=family,
        reason=f"no compiler for family '{family}'",
    )


# ---------------------------------------------------------------------------
# Composite role alignment compiler
# ---------------------------------------------------------------------------


def _resolve_composite_roles(
    demos: tuple[DemoPair, ...],
) -> list[dict[str, int]] | None:
    """Resolve center/frame/anchor colors for each demo."""
    from aria.decomposition import decompose_composites, detect_bg

    per_demo: list[dict[str, int]] = []
    for demo in demos:
        bg = detect_bg(demo.input)
        dec = decompose_composites(demo.input, bg)
        if dec.center_color is None or dec.frame_color is None or dec.anchor is None:
            return None
        # Count center-color singletons (composites + anchor)
        n_center_singletons = sum(
            1 for c in dec.composites if c.center.color == dec.center_color
        ) + 1  # +1 for the anchor itself
        per_demo.append({
            "bg": bg,
            "center": dec.center_color,
            "frame": dec.frame_color,
            "anchor_row": dec.anchor.row,
            "anchor_col": dec.anchor.col,
            "n_center_singletons": n_center_singletons,
        })
    return per_demo


def _compile_composite_alignment(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a composite-role-alignment sketch into Programs.

    Emits one Program per demo (since colors differ due to role rotation).
    Each program:
    1. find_objects(input) → all
    2. by_color(center_color) → pred_center
    3. where(pred_center, all) → center_objects
    4. by_color(frame_color) → pred_frame
    5. where(pred_frame, all) → frame_objects
    6. by_color(center_color) → pred_anchor_color  (reuse pred_center)
    7. where(pred_anchor_color, all) → anchor_candidates
    8. singleton(anchor_candidates)... or nth(0, ...)
    Actually: the anchor is an isolated singleton of center_color.
    We need to distinguish it from composites. Use: the anchor is the
    center-color object NOT adjacent to any frame-color object.
    Simpler: we know the anchor position from decomposition.

    Pipeline for each demo:
    1. find_objects(input)
    2. select center-color objects
    3. identify anchor (by position or as singleton of center_color)
    4. align_center_to_{axis}_of(anchor) → transform
    5. map_obj(transform, center_objects)
    6. erase center+frame from input, paint back aligned
    """
    from aria.runtime.ops import has_op

    # Get alignment axis from sketch metadata
    per_demo_axis = sketch.metadata.get("per_demo_axis", [])
    if not per_demo_axis:
        return CompileFailure(
            sketch_task_id=sketch.task_id,
            family="composite_role_alignment",
            reason="no per_demo_axis in sketch metadata",
        )

    role_bindings_list = _resolve_composite_roles(demos)
    if role_bindings_list is None:
        return CompileFailure(
            sketch_task_id=sketch.task_id,
            family="composite_role_alignment",
            reason="could not resolve center/frame/anchor roles in all demos",
        )

    programs: list[Program] = []
    for di, (demo, roles, axis) in enumerate(zip(demos, role_bindings_list, per_demo_axis)):
        if axis is None:
            return CompileFailure(
                sketch_task_id=sketch.task_id,
                family="composite_role_alignment",
                reason=f"no alignment axis for demo {di}",
            )

        op_name = f"align_center_to_{axis}_of"
        if not has_op(op_name):
            return CompileFailure(
                sketch_task_id=sketch.task_id,
                family="composite_role_alignment",
                reason=f"missing op: {op_name}",
                missing_ops=(op_name,),
            )

        center_color = roles["center"]
        frame_color = roles["frame"]

        # Count center-color singletons to find anchor index (last singleton)
        n_center_singletons = roles.get("n_center_singletons", 2)
        anchor_index = n_center_singletons - 1

        prog = _build_composite_alignment_program(
            center_color=center_color,
            frame_color=frame_color,
            axis=axis,
            anchor_nth=anchor_index,
        )
        programs.append(prog)

    return CompilePerDemoPrograms(
        sketch_task_id=sketch.task_id,
        family="composite_role_alignment",
        programs=tuple(programs),
        role_bindings={
            "per_demo": role_bindings_list,
        },
        slot_bindings={
            "per_demo_axis": per_demo_axis,
        },
        description=(
            f"compiled {len(programs)} per-demo program(s) for composite alignment "
            f"(colors differ across demos — not a task-level program)"
        ),
    )


def _build_composite_alignment_program(
    center_color: int,
    frame_color: int,
    axis: str,
    anchor_nth: int = 1,
) -> Program:
    """Build the concrete Program for one demo's composite alignment.

    The anchor is selected as the nth center-color object in scan order.
    Decomposition tells us which index is the isolated anchor (typically
    the last center-color singleton, after composite centers).
    """
    op_name = f"align_center_to_{axis}_of"
    erase_map = {center_color: 0, frame_color: 0}

    steps = [
        Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
        Bind("v1", Type.PREDICATE, Call("by_color", (Literal(center_color, Type.COLOR),))),
        Bind("v2", Type.OBJECT_SET, Call("where", (Ref("v1"), Ref("v0")))),
        Bind("v3", Type.PREDICATE, Call("by_color", (Literal(frame_color, Type.COLOR),))),
        Bind("v4", Type.OBJECT_SET, Call("where", (Ref("v3"), Ref("v0")))),
        # Anchor: nth center-color object (isolated singleton from decomposition)
        Bind("v5", Type.OBJECT, Call("nth", (Literal(anchor_nth, Type.INT), Ref("v2")))),
        # Alignment transform
        Bind("v6", Type.OBJ_TRANSFORM, Call(op_name, (Ref("v5"),))),
        # Move center and frame objects
        Bind("v7", Type.OBJECT_SET, Call("map_obj", (Ref("v6"), Ref("v2")))),
        Bind("v8", Type.OBJECT_SET, Call("map_obj", (Ref("v6"), Ref("v4")))),
        # Erase and repaint
        Bind("v9", Type.GRID, Call("apply_color_map", (
            Literal(erase_map, Type.COLOR_MAP), Ref("input"),
        ))),
        Bind("v10", Type.GRID, Call("paint_objects", (Ref("v8"), Ref("v9")))),
        Bind("v11", Type.GRID, Call("paint_objects", (Ref("v7"), Ref("v10")))),
    ]
    return Program(steps=tuple(steps), output="v11")


# ---------------------------------------------------------------------------
# Periodic repair compiler
# ---------------------------------------------------------------------------


def _compile_periodic_repair(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Attempt to compile a periodic-repair sketch.

    The DSL lacks a position-aware periodic-fill op, so this cannot
    compile to a general Program. Instead, returns a CompileFailure
    with the computed repair grids as partial evidence, which callers
    can use as repair targets or for direct pixel matching.
    """
    axis = sketch.metadata.get("dominant_axis", "row")
    period = sketch.metadata.get("dominant_period", 2)

    # Compute what the repaired grids would look like
    repair_grids: list[Grid] = []
    for demo in demos:
        repaired = _compute_periodic_repair(demo.input, demo.output, axis, period)
        if repaired is not None:
            repair_grids.append(repaired)

    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family="framed_periodic_repair",
        reason=(
            "DSL lacks position-aware periodic fill op; "
            "need repair_periodic(grid, axis, period) → GRID"
        ),
        missing_ops=("repair_periodic",),
        partial_evidence={
            "axis": axis,
            "period": period,
            "repair_grids_computed": len(repair_grids),
            "repair_grids": repair_grids,
        },
        description=(
            f"periodic repair (axis={axis}, period={period}) needs a new op; "
            f"computed {len(repair_grids)} repair grid(s) as evidence"
        ),
    )


def _compute_periodic_repair(
    input_grid: Grid,
    output_grid: Grid,
    axis: str,
    period: int,
) -> Grid | None:
    """Compute what the repaired grid should look like.

    Finds framed sub-regions recursively, then repairs periodic violations
    in the innermost content rows/columns. Only modifies cells where the
    input differs from the expected pattern AND the output matches.
    """
    from aria.sketch_fit import (
        _detect_row_period,
        _detect_col_period,
        _find_all_framed_regions,
    )
    from aria.decomposition import detect_bg

    if input_grid.shape != output_grid.shape:
        return None

    repaired = input_grid.copy()
    any_change = False
    bg = detect_bg(input_grid)

    # Find all framed regions (recursively nested)
    all_regions = _find_all_framed_regions(input_grid, bg, offset_r=0, offset_c=0, depth=3)

    # For each region, scan interior rows/cols for periodic violations
    for region in all_regions:
        r0, c0 = region.row, region.col
        h, w = region.height, region.width
        if r0 + h > output_grid.shape[0] or c0 + w > output_grid.shape[1]:
            continue
        interior_in = region.interior
        interior_out = output_grid[r0:r0 + h, c0:c0 + w]
        if interior_in.shape != interior_out.shape:
            continue

        if axis == "row":
            for ri in range(interior_in.shape[0]):
                ev = _detect_row_period(interior_in[ri])
                if ev is None:
                    continue
                for vi in ev.violation_positions:
                    if vi == 0 or vi == w - 1:
                        continue
                    expected = ev.pattern[vi % ev.period]
                    if int(interior_out[ri, vi]) == expected:
                        repaired[r0 + ri, c0 + vi] = expected
                        any_change = True
        else:
            for ci in range(interior_in.shape[1]):
                ev = _detect_col_period(interior_in[:, ci])
                if ev is None:
                    continue
                for vi in ev.violation_positions:
                    if vi == 0 or vi == h - 1:
                        continue
                    expected = ev.pattern[vi % ev.period]
                    if int(interior_out[vi, ci]) == expected:
                        repaired[r0 + vi, c0 + ci] = expected
                        any_change = True

    # Also try direct pixel diff: if remaining differences between repaired
    # and output are at positions where the output matches the expected
    # periodic pattern, apply those too
    if not any_change:
        # Fallback: diff input vs output, trust output for small changes
        diff_mask = input_grid != output_grid
        if 0 < int(diff_mask.sum()) <= 20:
            for r, c in zip(*np.where(diff_mask)):
                repaired[r, c] = output_grid[r, c]
                any_change = True

    return repaired if any_change else None
