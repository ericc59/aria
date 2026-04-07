"""Sketch compiler — compile sketch graphs into executable Programs.

Two entry points:

  compile_sketch_graph(graph, specialization, demos)
      Graph-native path.  Reads primitive structure from the SketchGraph
      DAG and resolved static bindings from the Specialization object.
      Three lanes are fully graph-native today:
        - REPAIR_LINES / REPAIR_2D_MOTIF  → task-level program
        - APPLY_RELATION + ANCHOR         → per-demo programs
        - CONSTRUCT_CANVAS (tile/upscale/crop) → task-level program

  compile_sketch(sketch, demos)
      Legacy/compatibility path.  Dispatches on the linear Sketch's
      primitive pattern + slots + roles via _COMPOSITIONS registry.
      Used as a fallback when compile_sketch_graph encounters an
      unrecognized graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.sketch import (
    PrimitiveFamily,
    RoleVar,
    Sketch,
    SketchGraph,
    SketchStep,
    Slot,
    Specialization,
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
    family: str              # reporting label, not used for dispatch
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
# Legacy compile_sketch path — _COMPOSITIONS registry
# ---------------------------------------------------------------------------
# Used by compile_sketch() as a fallback when compile_sketch_graph()
# does not recognize the graph structure.  New lanes should be added
# to compile_sketch_graph() directly, not here.

_COMPOSITIONS: list[tuple[
    frozenset[str],     # required primitive names (must all be present)
    frozenset[str],     # required slot names (on any step)
    frozenset[str],     # required role kinds (on any step)
    str,                # compiler function name
    str,                # human description
]] = [
    # 1. Full 2D repair: REPAIR_LINES + REPAIR_2D_MOTIF
    #    1D line repair first, then 2D motif repair on remaining cells.
    (
        frozenset({"REPAIR_LINES", "REPAIR_2D_MOTIF"}),
        frozenset({"axis"}),
        frozenset(),
        "_compile_periodic_2d_repair",
        "peel → partition → repair lines + 2D motif",
    ),
    # 2. 1D line repair only: REPAIR_LINES with axis slot
    (
        frozenset({"REPAIR_LINES"}),
        frozenset({"axis"}),
        frozenset(),
        "_compile_periodic_repair",
        "peel → partition → repair lines (periodic)",
    ),
    # 3. Legacy periodic repair: REPAIR_MISMATCH with axis slot
    (
        frozenset({"REPAIR_MISMATCH"}),
        frozenset({"axis"}),
        frozenset(),
        "_compile_periodic_repair",
        "region regularity repair (periodic, legacy)",
    ),
    # 3. Composite role alignment: APPLY_RELATION + axis + ANCHOR
    (
        frozenset({"APPLY_RELATION"}),
        frozenset({"axis"}),
        frozenset({"ANCHOR"}),
        "_compile_composite_alignment",
        "composite role alignment (axis-based)",
    ),
    # 4. Object transform + paint
    (
        frozenset({"APPLY_TRANSFORM", "PAINT"}),
        frozenset({"transform"}),
        frozenset(),
        "_compile_transform_paint",
        "object transform and paint",
    ),
    # 5. Canvas construction
    (
        frozenset({"CONSTRUCT_CANVAS"}),
        frozenset({"output_dims"}),
        frozenset(),
        "_compile_canvas_layout",
        "canvas construction",
    ),
]


def compile_sketch(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Legacy/compatibility compiler — dispatches on linear Sketch structure.

    Prefer compile_sketch_graph() for new work.  This entry point is
    used as the fallback when compile_sketch_graph encounters an
    unrecognized graph structure.
    """
    pattern_set = frozenset(sketch.primitive_pattern)
    all_slots = frozenset(s.name for step in sketch.steps for s in step.slots)
    all_role_kinds = frozenset(
        r.kind.name for step in sketch.steps for r in step.roles
    )

    # Also check metadata for evidence that slots reference
    meta_slots: set[str] = set()
    if sketch.metadata.get("dominant_axis") or sketch.metadata.get("per_demo_axis"):
        meta_slots.add("axis")
    if sketch.metadata.get("dominant_period"):
        meta_slots.add("period")

    effective_slots = all_slots | meta_slots

    # Try each supported composition in priority order
    for req_prims, req_slots, req_roles, fn_name, desc in _COMPOSITIONS:
        if not req_prims <= pattern_set:
            continue
        if not req_slots <= effective_slots:
            continue
        if not req_roles <= all_role_kinds:
            continue

        fn = _COMPILER_FNS.get(fn_name)
        if fn is not None:
            return fn(sketch, demos)

    # No supported composition matched — structured failure
    family = sketch.metadata.get("family", "")
    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family=family or "unknown",
        reason=(
            f"no supported composition for primitives={sorted(pattern_set)}, "
            f"slots={sorted(effective_slots)}, roles={sorted(all_role_kinds)}"
        ),
    )


# ---------------------------------------------------------------------------
# Legacy stubs — these lanes are handled graph-natively or not yet implemented
# ---------------------------------------------------------------------------


def _compile_transform_paint(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Legacy stub: APPLY_TRANSFORM + PAINT (not yet implemented)."""
    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family=sketch.metadata.get("family", "transform_paint"),
        reason="APPLY_TRANSFORM + PAINT compilation not yet implemented",
        missing_ops=("generic_object_transform",),
    )


def _compile_canvas_layout(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Legacy stub: CONSTRUCT_CANVAS (graph-native path handles this)."""
    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family=sketch.metadata.get("family", "canvas_layout"),
        reason="CONSTRUCT_CANVAS not implemented in legacy path — use compile_sketch_graph",
    )


# ---------------------------------------------------------------------------
# Composite role alignment compiler (legacy path — graph-native version is below)
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
    """Compile a composite-role-alignment sketch into per-demo Programs.

    Emits one Program per demo because colors differ due to role rotation.
    The anchor is the last center-color singleton in scan order (isolated
    from composites). Each program: find_objects → select by color →
    nth(anchor_index) → align_center_to_{axis}_of → map_obj → paint.
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
# Periodic repair compiler (legacy path — graph-native version is below)
# ---------------------------------------------------------------------------


def _compile_periodic_repair(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a periodic-repair sketch into a primitive-aligned program.

    Uses repair_framed_lines(grid, axis, period) which is explicitly composed
    from the primitives: peel_frame → partition_grid → repair_grid_lines.
    Unlike the legacy repair_periodic wrapper, this op's implementation
    delegates to the named primitive functions rather than containing its
    own duplicate logic.
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    axis_str = sketch.metadata.get("dominant_axis", "row")
    period = sketch.metadata.get("dominant_period", 2)

    if not has_op("repair_framed_lines"):
        return CompileFailure(
            sketch_task_id=sketch.task_id,
            family="framed_periodic_repair",
            reason="missing op: repair_framed_lines",
            missing_ops=("repair_framed_lines",),
        )

    axis_int = 0 if axis_str == "row" else 1

    # Build primitive-aligned program using repair_framed_lines which is
    # explicitly composed from: peel_frame → partition_grid → repair_grid_lines.
    # Unlike the legacy repair_periodic wrapper, this op uses the named
    # primitives as its implementation.
    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("repair_framed_lines", (
                Ref("input"),
                Literal(axis_int, Type.INT),
                Literal(period, Type.INT),
            ))),
        ),
        output="v0",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=sketch.task_id,
            family="framed_periodic_repair",
            program=prog,
            role_bindings={
                "note": "role-normalized: frame/bg detected at runtime, not compiled as literals",
            },
            slot_bindings={
                "axis": axis_str,
                "axis_int": axis_int,
                "period": period,
            },
            description=(
                f"repair_framed_lines(input, axis={axis_str}, period={period}) "
                f"[peel→partition→repair primitives] verified across {len(demos)} demo(s)"
            ),
        )

    # If verification fails, fall back to structured failure with evidence
    repair_grids: list[Grid] = []
    for demo in demos:
        repaired = _compute_periodic_repair(demo.input, demo.output, axis_str, period)
        if repaired is not None:
            repair_grids.append(repaired)

    return CompileFailure(
        sketch_task_id=sketch.task_id,
        family="framed_periodic_repair",
        reason=(
            f"repair_periodic compiled but failed verification "
            f"(axis={axis_str}, period={period})"
        ),
        partial_evidence={
            "axis": axis_str,
            "period": period,
            "repair_grids_computed": len(repair_grids),
            "repair_grids": repair_grids,
        },
        description=(
            f"periodic repair (axis={axis_str}, period={period}) compiled but "
            f"verification failed on {len(demos)} demo(s)"
        ),
    )


def _compile_periodic_2d_repair(
    sketch: Sketch,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a sketch with both REPAIR_LINES and REPAIR_2D_MOTIF.

    Emits a 2-step program:
    1. repair_framed_lines(input, axis, period) → handles 1D violations
    2. repair_framed_2d_motif(step1) → handles remaining 2D cell violations
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    axis_str = sketch.metadata.get("dominant_axis", "row")
    period = sketch.metadata.get("dominant_period", 2)

    for op in ("repair_framed_lines", "repair_framed_2d_motif"):
        if not has_op(op):
            return CompileFailure(
                sketch_task_id=sketch.task_id,
                family="framed_periodic_repair",
                reason=f"missing op: {op}",
                missing_ops=(op,),
            )

    axis_int = 0 if axis_str == "row" else 1

    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("repair_framed_lines", (
                Ref("input"),
                Literal(axis_int, Type.INT),
                Literal(period, Type.INT),
            ))),
            Bind("v1", Type.GRID, Call("repair_framed_2d_motif", (
                Ref("v0"),
            ))),
        ),
        output="v1",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=sketch.task_id,
            family="framed_periodic_repair",
            program=prog,
            role_bindings={"note": "role-normalized: 1D lines + 2D motif repair"},
            slot_bindings={"axis": axis_str, "period": period},
            description=(
                f"repair_framed_lines + repair_framed_2d_motif "
                f"verified across {len(demos)} demo(s)"
            ),
        )

    # Fall back to 1D only
    return _compile_periodic_repair(sketch, demos)


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


# ---------------------------------------------------------------------------
# Compiler function registry — maps names from _COMPOSITIONS to functions
# ---------------------------------------------------------------------------

_COMPILER_FNS: dict[str, Any] = {
    "_compile_periodic_repair": _compile_periodic_repair,
    "_compile_periodic_2d_repair": _compile_periodic_2d_repair,
    "_compile_composite_alignment": _compile_composite_alignment,
    "_compile_transform_paint": _compile_transform_paint,
    "_compile_canvas_layout": _compile_canvas_layout,
}


# ===========================================================================
# Early consensus — cross-demo consistency check before compilation
# ===========================================================================


def _check_specialization_consensus(
    graph: SketchGraph,
    specialization: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileFailure | None:
    """Check cross-demo consistency of specialization bindings.

    Returns a CompileFailure if the specialization is structurally
    inconsistent across demos, None if consistent.

    Checks:
    1. Per-demo role bindings exist for all demos (if the spec has per-demo data)
    2. Per-demo axis values are all present (if the spec has axis data)
    """
    # Check per-demo role bindings
    per_demo_roles = specialization.get("__relation__", "per_demo_roles")
    if per_demo_roles is not None and isinstance(per_demo_roles, (list, tuple)):
        if len(per_demo_roles) != len(demos):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="consensus",
                reason=(
                    f"per_demo_roles has {len(per_demo_roles)} entries "
                    f"but {len(demos)} demos"
                ),
            )
        # Check that all per-demo role dicts have the same keys
        if per_demo_roles:
            key_sets = [frozenset(r.keys()) for r in per_demo_roles if isinstance(r, dict)]
            if key_sets and len(set(key_sets)) > 1:
                return CompileFailure(
                    sketch_task_id=graph.task_id,
                    family="consensus",
                    reason=f"per_demo role binding keys differ: {[set(k) for k in key_sets]}",
                )

    # Check per-demo axis values
    per_demo_axis = specialization.get("__relation__", "per_demo_axis")
    if per_demo_axis is not None and isinstance(per_demo_axis, (list, tuple)):
        if len(per_demo_axis) != len(demos):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="consensus",
                reason=(
                    f"per_demo_axis has {len(per_demo_axis)} entries "
                    f"but {len(demos)} demos"
                ),
            )
        # All axis values should be non-None
        if any(a is None for a in per_demo_axis):
            missing = [i for i, a in enumerate(per_demo_axis) if a is None]
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="consensus",
                reason=f"per_demo_axis missing for demos {missing}",
            )

    return None  # consistent


# ===========================================================================
# Graph-native compilation — primary path
# ===========================================================================


def compile_sketch_graph(
    graph: SketchGraph,
    specialization: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a sketch graph + specialization into executable Programs.

    Primary compilation entry point.  Dispatches on graph primitive
    structure + specialization bindings.  Falls back to compile_sketch()
    for graph structures that are not yet graph-native.

    Early consensus: checks cross-demo role binding consistency before
    attempting compilation lanes.
    """
    # Early consensus: check role binding consistency from specialization
    try:
        from aria.core.scene_solve import get_consensus_enabled
        consensus_on = get_consensus_enabled()
    except ImportError:
        consensus_on = True
    if consensus_on:
        consensus_failure = _check_specialization_consensus(graph, specialization, demos)
        if consensus_failure is not None:
            return consensus_failure

    # Collect primitive names and role kinds from graph
    primitives = frozenset(n.primitive.name for n in graph.nodes.values())
    all_role_kinds = frozenset(
        r.kind.name for n in graph.nodes.values() for r in n.roles
    )

    # Read resolved bindings from specialization
    axis = specialization.get("__task__", "dominant_axis")
    period = specialization.get("__task__", "dominant_period")

    # --- Specialization-gated lanes (these have strong spec bindings) ---

    # Composite alignment (gated by ANCHOR+CENTER roles + per-demo bindings)
    if _can_compile_alignment_from_graph(primitives, all_role_kinds, specialization):
        return _compile_alignment_from_graph(graph, specialization, demos)

    # Canvas construction (gated by canvas strategy binding)
    if _can_compile_canvas_from_graph(primitives, specialization):
        return _compile_canvas_from_graph(graph, specialization, demos)

    # Object movement (gated by movement strategy binding)
    if _can_compile_movement_from_graph(primitives, specialization):
        return _compile_movement_from_graph(graph, specialization, demos)

    # Grid transforms (gated by transform binding)
    if _can_compile_grid_transform_from_graph(primitives, specialization):
        return _compile_grid_transform_from_graph(graph, specialization, demos)

    # --- Evidence-ranked mechanism lanes ---
    # Compute structural evidence from demos, rank lanes, try in order.
    # Exact verification remains the final arbiter.
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    evidence, ranking = compute_evidence_and_rank(demos)

    _lane_compilers = {
        "periodic_repair": (
            lambda: _can_compile_periodic_from_graph(primitives, axis, period),
            lambda: _compile_periodic_from_graph(graph, specialization, demos),
        ),
        "replication": (
            lambda: _can_compile_replicate(primitives),
            lambda: _compile_replicate(graph, specialization, demos),
        ),
        "relocation": (
            lambda: _can_compile_select_relate_paint(primitives),
            lambda: _compile_select_relate_paint(graph, specialization, demos),
        ),
    }

    best_failure = None
    for candidate in ranking.lanes:
        if candidate.name not in _lane_compilers:
            continue
        can_fn, compile_fn = _lane_compilers[candidate.name]
        if not can_fn():
            continue
        result = compile_fn()
        if isinstance(result, CompileTaskProgram):
            return result
        # Track best failure for diagnostics
        if best_failure is None:
            best_failure = result

    # Return best failure if any lane was tried
    if best_failure is not None:
        return best_failure

    # Multi-step graph compilation (compositions from library)
    if _can_compile_multistep_from_graph(graph):
        return _compile_multistep_from_graph(graph, specialization, demos)

    # Fallback: convert to Sketch and use existing path
    sketch = graph.to_sketch()
    return compile_sketch(sketch, demos)


def _can_compile_periodic_from_graph(
    primitives: frozenset[str],
    axis: Any,
    period: Any,
) -> bool:
    """Check if graph can be compiled via the periodic repair path."""
    has_repair = ("REPAIR_LINES" in primitives or "REPAIR_MISMATCH" in primitives
                  or "REPAIR_2D_MOTIF" in primitives)
    return has_repair and axis is not None and period is not None


def _compile_periodic_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile periodic repair directly from graph + specialization.

    Reads axis and period from the specialization bindings instead
    of metadata, then builds the same program structure.
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    axis_str = spec.get("__task__", "dominant_axis")
    period = spec.get("__task__", "dominant_period")
    if axis_str is None:
        axis_str = "row"
    if period is None:
        period = 2

    # Check for 2D repair path
    primitives = frozenset(n.primitive.name for n in graph.nodes.values())
    use_2d = "REPAIR_2D_MOTIF" in primitives

    if use_2d:
        for op in ("repair_framed_lines", "repair_framed_2d_motif"):
            if not has_op(op):
                return CompileFailure(
                    sketch_task_id=graph.task_id,
                    family="framed_periodic_repair",
                    reason=f"missing op: {op}",
                    missing_ops=(op,),
                )

        axis_int = 0 if axis_str == "row" else 1
        prog = Program(
            steps=(
                Bind("v0", Type.GRID, Call("repair_framed_lines", (
                    Ref("input"),
                    Literal(axis_int, Type.INT),
                    Literal(period, Type.INT),
                ))),
                Bind("v1", Type.GRID, Call("repair_framed_2d_motif", (
                    Ref("v0"),
                ))),
            ),
            output="v1",
        )
        vr = verify(prog, demos)
        if vr.passed:
            return CompileTaskProgram(
                sketch_task_id=graph.task_id,
                family="framed_periodic_repair",
                program=prog,
                role_bindings={"note": "compiled from graph+specialization: 1D+2D repair"},
                slot_bindings={"axis": axis_str, "period": period},
                description=(
                    f"graph→specialization→compile: repair_framed_lines + "
                    f"repair_framed_2d_motif (axis={axis_str}, period={period})"
                ),
            )
        # Fall through to 1D-only below

    if not has_op("repair_framed_lines"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="framed_periodic_repair",
            reason="missing op: repair_framed_lines",
            missing_ops=("repair_framed_lines",),
        )

    axis_int = 0 if axis_str == "row" else 1
    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("repair_framed_lines", (
                Ref("input"),
                Literal(axis_int, Type.INT),
                Literal(period, Type.INT),
            ))),
        ),
        output="v0",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="framed_periodic_repair",
            program=prog,
            role_bindings={"note": "compiled from graph+specialization"},
            slot_bindings={"axis": axis_str, "period": period},
            description=(
                f"graph→specialization→compile: repair_framed_lines"
                f"(axis={axis_str}, period={period})"
            ),
        )

    # Try all repair_mode variants via the composite periodic_repair op
    if has_op("periodic_repair"):
        from aria.runtime.ops.periodic_repair import ALL_REPAIR_MODES, REPAIR_MODE_NAMES
        best_diff = float("inf")
        best_mode = None
        best_axis_str = None
        for try_axis in ("row", "col"):
            try_axis_int = 0 if try_axis == "row" else 1
            for try_period in (period, 2, 3, 4, 5):
                for mode in ALL_REPAIR_MODES:
                    p = Program(
                        steps=(Bind("v0", Type.GRID, Call("periodic_repair", (
                            Ref("input"),
                            Literal(try_axis_int, Type.INT),
                            Literal(try_period, Type.INT),
                            Literal(mode, Type.INT),
                        ))),),
                        output="v0",
                    )
                    vr = verify(p, demos)
                    if vr.passed:
                        return CompileTaskProgram(
                            sketch_task_id=graph.task_id,
                            family="periodic_repair",
                            program=p,
                            role_bindings={},
                            slot_bindings={
                                "axis": try_axis, "period": try_period,
                                "repair_mode": mode,
                                "repair_mode_name": REPAIR_MODE_NAMES.get(mode, str(mode)),
                            },
                            description=(
                                f"graph→compile: periodic_repair("
                                f"axis={try_axis}, period={try_period}, "
                                f"mode={REPAIR_MODE_NAMES.get(mode, mode)})"
                            ),
                        )
                    try:
                        from aria.runtime.executor import execute
                        import numpy as np
                        diff = sum(int(np.sum(execute(p, d.input) != d.output)) for d in demos)
                        if diff < best_diff:
                            best_diff = diff
                            best_mode = mode
                            best_axis_str = try_axis
                    except Exception:
                        pass

        desc = f"periodic repair verified failed (tried all modes, best axis={best_axis_str} diff={best_diff})"
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="periodic_repair",
            reason=desc,
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="framed_periodic_repair",
        reason=f"periodic repair verified failed (axis={axis_str}, period={period})",
    )


# ---------------------------------------------------------------------------
# Graph-native composite alignment compilation
# ---------------------------------------------------------------------------


def _can_compile_alignment_from_graph(
    primitives: frozenset[str],
    role_kinds: frozenset[str],
    spec: Specialization,
) -> bool:
    """Check if graph can be compiled via the alignment path.

    Requires:
    - APPLY_RELATION primitive in graph
    - ANCHOR role in graph
    - per_demo_roles resolved in specialization
    - per_demo_axis resolved in specialization
    """
    has_relation = "APPLY_RELATION" in primitives
    has_anchor = "ANCHOR" in role_kinds
    has_roles = spec.get("__relation__", "per_demo_roles") is not None
    has_axes = spec.get("__relation__", "per_demo_axis") is not None
    return has_relation and has_anchor and has_roles and has_axes


def _compile_alignment_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile composite alignment directly from graph + specialization.

    Reads per-demo role bindings and alignment axes from the specialization
    instead of sketch metadata. Produces per-demo programs because
    center/frame colors differ across demos (role rotation).
    """
    from aria.runtime.ops import has_op

    per_demo_roles = spec.get("__relation__", "per_demo_roles")
    per_demo_axis = spec.get("__relation__", "per_demo_axis")

    if per_demo_roles is None or per_demo_axis is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="composite_role_alignment",
            reason="missing per-demo role or axis bindings in specialization",
        )

    if len(per_demo_roles) != len(demos) or len(per_demo_axis) != len(demos):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="composite_role_alignment",
            reason="per-demo binding count does not match demo count",
        )

    programs: list[Program] = []
    for di, (demo, roles, axis) in enumerate(zip(demos, per_demo_roles, per_demo_axis)):
        if axis is None:
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="composite_role_alignment",
                reason=f"no alignment axis for demo {di}",
            )

        op_name = f"align_center_to_{axis}_of"
        if not has_op(op_name):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="composite_role_alignment",
                reason=f"missing op: {op_name}",
                missing_ops=(op_name,),
            )

        center_color = roles["center"]
        frame_color = roles["frame"]
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
        sketch_task_id=graph.task_id,
        family="composite_role_alignment",
        programs=tuple(programs),
        role_bindings={
            "per_demo": per_demo_roles,
        },
        slot_bindings={
            "per_demo_axis": per_demo_axis,
        },
        description=(
            f"graph→specialization→compile: {len(programs)} per-demo program(s) "
            f"for composite alignment (colors differ across demos)"
        ),
    )


# ---------------------------------------------------------------------------
# Graph-native canvas construction compilation
# ---------------------------------------------------------------------------


def _can_compile_canvas_from_graph(
    primitives: frozenset[str],
    spec: Specialization,
) -> bool:
    """Check if graph can be compiled via the canvas construction path."""
    has_canvas = "CONSTRUCT_CANVAS" in primitives
    has_strategy = spec.get("__canvas__", "strategy") is not None
    return has_canvas and has_strategy


def _compile_canvas_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile canvas construction directly from graph + specialization.

    Reads strategy and parameters from specialization bindings.
    Produces a task-level program — canvas ops are role-normalized
    (no per-demo literal colors), so one program covers all demos.
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    strategy = spec.get("__canvas__", "strategy")

    if strategy == "tile":
        return _compile_canvas_tile(graph, spec, demos)
    elif strategy == "upscale":
        return _compile_canvas_upscale(graph, spec, demos)
    elif strategy == "crop":
        return _compile_canvas_crop(graph, spec, demos)

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="canvas_construction",
        reason=f"unknown canvas strategy: {strategy}",
    )


def _compile_canvas_tile(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    if not has_op("tile_grid"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing op: tile_grid",
            missing_ops=("tile_grid",),
        )

    tile_rows = spec.get("__canvas__", "tile_rows")
    tile_cols = spec.get("__canvas__", "tile_cols")
    if tile_rows is None or tile_cols is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing tile_rows or tile_cols in specialization",
        )

    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("tile_grid", (
                Ref("input"),
                Literal(tile_rows, Type.INT),
                Literal(tile_cols, Type.INT),
            ))),
        ),
        output="v0",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            program=prog,
            role_bindings={},
            slot_bindings={"strategy": "tile", "tile_rows": tile_rows, "tile_cols": tile_cols},
            description=(
                f"graph→specialization→compile: tile_grid(input, {tile_rows}, {tile_cols})"
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="canvas_construction",
        reason=f"tile_grid({tile_rows}, {tile_cols}) failed verification",
    )


def _compile_canvas_upscale(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    if not has_op("upscale_grid"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing op: upscale_grid",
            missing_ops=("upscale_grid",),
        )

    factor = spec.get("__canvas__", "scale_factor")
    if factor is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing scale_factor in specialization",
        )

    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("upscale_grid", (
                Ref("input"),
                Literal(factor, Type.INT),
            ))),
        ),
        output="v0",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            program=prog,
            role_bindings={},
            slot_bindings={"strategy": "upscale", "scale_factor": factor},
            description=(
                f"graph→specialization→compile: upscale_grid(input, {factor})"
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="canvas_construction",
        reason=f"upscale_grid({factor}) failed verification",
    )


def _compile_canvas_crop(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    if not has_op("crop"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing op: crop",
            missing_ops=("crop",),
        )

    region = spec.get("__canvas__", "crop_region")
    if region is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            reason="missing crop_region in specialization",
        )

    # crop expects (x, y, w, h) = (col, row, width, height)
    r0, c0, h, w = region
    crop_region = (c0, r0, w, h)

    prog = Program(
        steps=(
            Bind("v0", Type.GRID, Call("crop", (
                Ref("input"),
                Literal(crop_region, Type.REGION),
            ))),
        ),
        output="v0",
    )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="canvas_construction",
            program=prog,
            role_bindings={},
            slot_bindings={"strategy": "crop", "crop_region": region},
            description=(
                f"graph→specialization→compile: crop(input, {region})"
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="canvas_construction",
        reason=f"crop({region}) failed verification",
    )


# ---------------------------------------------------------------------------
# Graph-native object movement compilation
# ---------------------------------------------------------------------------


def _can_compile_movement_from_graph(
    primitives: frozenset[str],
    spec: Specialization,
) -> bool:
    """Check if graph can be compiled via the movement path."""
    has_transform = "APPLY_TRANSFORM" in primitives
    has_strategy = spec.get("__movement__", "strategy") is not None
    return has_transform and has_strategy


def _compile_movement_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile object movement directly from graph + specialization.

    Supports:
    - uniform_translate: all fg objects move by fixed (dr, dc)
    - gravity: all fg objects move to a grid edge
    Both are task-level programs.
    """
    strategy = spec.get("__movement__", "strategy")

    if strategy == "uniform_translate":
        return _compile_uniform_translate(graph, spec, demos)
    elif strategy == "gravity":
        return _compile_gravity(graph, spec, demos)

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="object_movement",
        reason=f"unknown movement strategy: {strategy}",
    )


def _compile_uniform_translate(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile uniform translation: erase fg objects, repaint translated."""
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    dr = spec.get("__movement__", "dr")
    dc = spec.get("__movement__", "dc")
    if dr is None or dc is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="object_movement",
            reason="missing dr or dc in specialization",
        )

    for op in ("find_objects", "map_obj", "translate_by", "paint_objects",
               "apply_color_map"):
        if not has_op(op):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="object_movement",
                reason=f"missing op: {op}",
                missing_ops=(op,),
            )

    # Determine bg color per demo for erase map (need consensus or per-demo)
    # Use consensus bg if available, else per-demo
    bg_binding = None
    for b in spec.bindings:
        if b.name == "bg" and b.source == "consensus":
            bg_binding = b.value
            break

    if bg_binding is not None:
        # Task-level: use consensus bg
        prog = _build_translate_program(dr, dc, bg_binding)
        vr = verify(prog, demos)
        if vr.passed:
            return CompileTaskProgram(
                sketch_task_id=graph.task_id,
                family="object_movement",
                program=prog,
                role_bindings={},
                slot_bindings={"strategy": "uniform_translate", "dr": dr, "dc": dc},
                description=(
                    f"graph→specialization→compile: translate_by({dr},{dc})"
                ),
            )

    # Try per-demo bg colors
    from aria.decomposition import detect_bg
    programs: list[Program] = []
    for demo in demos:
        bg = detect_bg(demo.input)
        prog = _build_translate_program(dr, dc, bg)
        programs.append(prog)

    # Verify as task-level (first program on all demos)
    vr = verify(programs[0], demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="object_movement",
            program=programs[0],
            role_bindings={},
            slot_bindings={"strategy": "uniform_translate", "dr": dr, "dc": dc},
            description=(
                f"graph→specialization→compile: translate_by({dr},{dc})"
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="object_movement",
        reason=f"translate_by({dr},{dc}) failed verification",
    )


def _build_translate_program(dr: int, dc: int, bg: int) -> Program:
    """Build a program: find_objects → translate_by → erase+repaint."""
    # Build color map to erase all non-bg colors to bg
    erase_map = {c: bg for c in range(10) if c != bg}

    steps = [
        Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
        Bind("v1", Type.OBJ_TRANSFORM, Call("translate_by", (
            Literal(dr, Type.INT), Literal(dc, Type.INT),
        ))),
        Bind("v2", Type.OBJECT_SET, Call("map_obj", (Ref("v1"), Ref("v0")))),
        Bind("v3", Type.GRID, Call("apply_color_map", (
            Literal(erase_map, Type.COLOR_MAP), Ref("input"),
        ))),
        Bind("v4", Type.GRID, Call("paint_objects", (Ref("v2"), Ref("v3")))),
    ]
    return Program(steps=tuple(steps), output="v4")


def _compile_gravity(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile gravity movement: all fg objects move to a grid edge."""
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    direction = spec.get("__movement__", "direction")
    if direction is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="object_movement",
            reason="missing direction in specialization",
        )

    for op in ("find_objects", "map_obj", "gravity_to_edge", "paint_objects",
               "apply_color_map"):
        if not has_op(op):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="object_movement",
                reason=f"missing op: {op}",
                missing_ops=(op,),
            )

    dir_map = {"up": 1, "down": 2, "left": 3, "right": 4}  # matches Dir enum values
    dir_int = dir_map.get(direction)
    if dir_int is None:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="object_movement",
            reason=f"unknown gravity direction: {direction}",
        )

    from aria.decomposition import detect_bg

    # Try task-level first (same dims → one program)
    dims_set = {d.input.shape for d in demos}
    if len(dims_set) == 1:
        h, w = demos[0].input.shape
        bg = detect_bg(demos[0].input)
        prog = _build_gravity_program(dir_int, h, w, bg)
        vr = verify(prog, demos)
        if vr.passed:
            return CompileTaskProgram(
                sketch_task_id=graph.task_id,
                family="object_movement",
                program=prog,
                role_bindings={},
                slot_bindings={"strategy": "gravity", "direction": direction},
                description=(
                    f"graph→specialization→compile: gravity_to_edge({direction}, {h}, {w})"
                ),
            )

    # Per-demo: dims may differ, compile one program per demo
    programs: list[Program] = []
    for demo in demos:
        h, w = demo.input.shape
        bg = detect_bg(demo.input)
        programs.append(_build_gravity_program(dir_int, h, w, bg))

    # Try promoting: if all programs are structurally identical, use one
    if len(dims_set) == 1:
        vr = verify(programs[0], demos)
        if vr.passed:
            return CompileTaskProgram(
                sketch_task_id=graph.task_id,
                family="object_movement",
                program=programs[0],
                role_bindings={},
                slot_bindings={"strategy": "gravity", "direction": direction},
                description=(
                    f"graph→specialization→compile: gravity_to_edge({direction})"
                ),
            )

    # Per-demo verification
    any_passed = False
    for prog, demo in zip(programs, demos):
        vr = verify(prog, (demo,))
        if vr.passed:
            any_passed = True

    if any_passed:
        return CompilePerDemoPrograms(
            sketch_task_id=graph.task_id,
            family="object_movement",
            programs=tuple(programs),
            role_bindings={},
            slot_bindings={"strategy": "gravity", "direction": direction},
            description=(
                f"graph→specialization→compile: {len(programs)} per-demo "
                f"gravity_to_edge({direction}) programs (dims vary)"
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="object_movement",
        reason=f"gravity({direction}) failed verification",
    )


def _build_gravity_program(dir_int: int, h: int, w: int, bg: int) -> Program:
    """Build a gravity program for specific grid dimensions."""
    erase_map = {c: bg for c in range(10) if c != bg}
    return Program(
        steps=(
            Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
            Bind("v1", Type.OBJ_TRANSFORM, Call("gravity_to_edge", (
                Literal(dir_int, Type.DIR),
                Literal(h, Type.INT),
                Literal(w, Type.INT),
            ))),
            Bind("v2", Type.OBJECT_SET, Call("map_obj", (Ref("v1"), Ref("v0")))),
            Bind("v3", Type.GRID, Call("apply_color_map", (
                Literal(erase_map, Type.COLOR_MAP), Ref("input"),
            ))),
            Bind("v4", Type.GRID, Call("paint_objects", (Ref("v2"), Ref("v3")))),
        ),
        output="v4",
    )


# ---------------------------------------------------------------------------
# Graph-native grid transform compilation
# ---------------------------------------------------------------------------


def _can_compile_grid_transform_from_graph(
    primitives: frozenset[str],
    spec: Specialization,
) -> bool:
    """Check if graph can be compiled via the grid transform path."""
    has_transform = "APPLY_TRANSFORM" in primitives
    has_grid_xform = spec.get("__grid_transform__", "transform") is not None
    return has_transform and has_grid_xform


def _compile_grid_transform_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a grid-level transform from graph + specialization.

    Supports rotate, reflect, transpose, and fill_enclosed.
    All produce task-level programs.
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    transform = spec.get("__grid_transform__", "transform")

    if transform == "rotate":
        degrees = spec.get("__grid_transform__", "degrees")
        if degrees is None:
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing degrees in specialization",
            )
        if not has_op("rotate_grid"):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing op: rotate_grid",
                missing_ops=("rotate_grid",),
            )
        prog = Program(
            steps=(Bind("v0", Type.GRID, Call("rotate_grid", (
                Literal(degrees, Type.INT), Ref("input"),
            ))),),
            output="v0",
        )

    elif transform == "reflect":
        axis = spec.get("__grid_transform__", "axis")
        if axis is None:
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing axis in specialization",
            )
        axis_int = 0 if axis == "row" else 1
        if not has_op("reflect_grid"):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing op: reflect_grid",
                missing_ops=("reflect_grid",),
            )
        prog = Program(
            steps=(Bind("v0", Type.GRID, Call("reflect_grid", (
                Literal(axis_int, Type.AXIS), Ref("input"),
            ))),),
            output="v0",
        )

    elif transform == "transpose":
        if not has_op("transpose_grid"):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing op: transpose_grid",
                missing_ops=("transpose_grid",),
            )
        prog = Program(
            steps=(Bind("v0", Type.GRID, Call("transpose_grid", (Ref("input"),))),),
            output="v0",
        )

    elif transform == "fill_enclosed":
        fill_color = spec.get("__grid_transform__", "fill_color")
        if fill_color is None:
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing fill_color in specialization",
            )
        if not has_op("fill_enclosed"):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="grid_transform",
                reason="missing op: fill_enclosed",
                missing_ops=("fill_enclosed",),
            )
        prog = Program(
            steps=(Bind("v0", Type.GRID, Call("fill_enclosed", (
                Ref("input"), Literal(fill_color, Type.COLOR),
            ))),),
            output="v0",
        )

    else:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="grid_transform",
            reason=f"unknown grid transform: {transform}",
        )

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="grid_transform",
            program=prog,
            role_bindings={},
            slot_bindings={"transform": transform},
            description=f"graph→specialization→compile: {transform}",
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="grid_transform",
        reason=f"{transform} failed verification",
    )


# ---------------------------------------------------------------------------
# Generalized multi-step graph compiler
# ---------------------------------------------------------------------------
# Walks the graph in topo order.  Each node with known evidence is
# compiled into a runtime call.  Nodes are chained: the output of one
# feeds as input to the next.  This handles compositions like
# fill_enclosed → rotate that no single-lane compiler covers.


_EVIDENCE_TO_PROGRAM_STEP: dict[str, Any] = {
    # transform -> (op_name, arg_builder)
    # arg_builder takes (evidence_dict, input_ref) -> (op, args) for a Bind
}


# ---------------------------------------------------------------------------
# Graph-native template replication compilation
# ---------------------------------------------------------------------------


def _can_compile_replicate(primitives: frozenset[str]) -> bool:
    """Check if graph can be compiled via the replication lane.

    Structural cue: output has MORE object instances than input,
    AND the graph contains selection + relation/paint primitives.
    Also matches graphs that explicitly contain replication-style ops.

    We check primitives only — the compiler verifies on demos.
    """
    has_select = "SELECT_SUBSET" in primitives
    has_relation = "APPLY_RELATION" in primitives
    has_paint = "PAINT" in primitives
    has_bind_role = "BIND_ROLE" in primitives

    # Replication pattern: selection + relation + paint
    if has_select and (has_relation or has_paint):
        return True
    # Simpler: just bind_role + paint
    if has_bind_role and has_paint:
        return True
    return False


def _compile_replicate(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a replication-style graph into replicate_templates(...).

    Searches over:
      key_rule (2): adjacent_diff_color, adjacent_any
      source_policy (2): erase_sources, keep_sources
      placement_rule (2): anchor_offset, center_on_target
    Total: 8 combinations (small, bounded).
    """
    from aria.runtime.ops import has_op
    from aria.runtime.ops.replicate import (
        ALL_KEY_RULES, KEY_NAMES,
        ALL_SOURCE_POLICIES, SOURCE_NAMES,
        ALL_PLACE_RULES, PLACE_NAMES,
    )
    from aria.verify.verifier import verify

    if not has_op("replicate_templates"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="replicate",
            reason="missing op: replicate_templates",
            missing_ops=("replicate_templates",),
        )

    bound_key = spec.get("__replicate__", "key_rule")
    bound_src = spec.get("__replicate__", "source_policy")
    bound_place = spec.get("__replicate__", "placement_rule")

    keys = [int(bound_key)] if bound_key is not None else list(ALL_KEY_RULES)
    srcs = [int(bound_src)] if bound_src is not None else list(ALL_SOURCE_POLICIES)
    places = [int(bound_place)] if bound_place is not None else list(ALL_PLACE_RULES)

    best_diff = float("inf")
    best_params = None

    for kr in keys:
        for sp in srcs:
            for pr in places:
                prog = Program(
                    steps=(
                        Bind("v0", Type.GRID, Call("replicate_templates", (
                            Ref("input"),
                            Literal(kr, Type.INT),
                            Literal(sp, Type.INT),
                            Literal(pr, Type.INT),
                        ))),
                    ),
                    output="v0",
                )

                vr = verify(prog, demos)
                if vr.passed:
                    return CompileTaskProgram(
                        sketch_task_id=graph.task_id,
                        family="replicate",
                        program=prog,
                        role_bindings={},
                        slot_bindings={
                            "key_rule": kr,
                            "key_rule_name": KEY_NAMES.get(kr, str(kr)),
                            "source_policy": sp,
                            "source_policy_name": SOURCE_NAMES.get(sp, str(sp)),
                            "placement_rule": pr,
                            "placement_rule_name": PLACE_NAMES.get(pr, str(pr)),
                        },
                        description=(
                            f"graph->compile: replicate_templates("
                            f"key={KEY_NAMES.get(kr, kr)}, "
                            f"src={SOURCE_NAMES.get(sp, sp)}, "
                            f"place={PLACE_NAMES.get(pr, pr)})"
                        ),
                    )

                try:
                    from aria.runtime.executor import execute
                    import numpy as np
                    diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
                    if diff < best_diff:
                        best_diff = diff
                        best_params = (kr, sp, pr)
                except Exception:
                    pass

    combos = len(keys) * len(srcs) * len(places)
    desc = f"replicate_templates verified failed (tried {combos} combinations"
    if best_params is not None:
        kr, sp, pr = best_params
        desc += (f", best key={KEY_NAMES.get(kr, kr)}"
                 f" src={SOURCE_NAMES.get(sp, sp)}"
                 f" place={PLACE_NAMES.get(pr, pr)}"
                 f" diff={best_diff}")
    desc += ")"

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="replicate",
        reason=desc,
    )


# ---------------------------------------------------------------------------
# Graph-native select-relate-paint compilation
# ---------------------------------------------------------------------------


def _can_compile_select_relate_paint(primitives: frozenset[str]) -> bool:
    """Check if graph can be compiled via the select-relate-paint lane.

    Matches graph patterns containing:
    - SELECT_SUBSET (object selection)
    - APPLY_RELATION or APPLY_TRANSFORM (relation-based transform)
    - optionally PAINT (object placement)

    This is the general structural lane for object-selection /
    relation-driven transform / paint.
    """
    has_select = "SELECT_SUBSET" in primitives
    has_relation = "APPLY_RELATION" in primitives
    has_paint = "PAINT" in primitives
    has_transform = "APPLY_TRANSFORM" in primitives

    # SELECT_SUBSET + (APPLY_RELATION or PAINT) is the canonical pattern
    if has_select and (has_relation or has_paint):
        return True
    # APPLY_RELATION + PAINT without explicit SELECT is also valid
    if has_relation and has_paint:
        return True
    # SELECT_SUBSET + APPLY_TRANSFORM (the simpler variant)
    if has_select and has_transform:
        return True
    return False


def _compile_select_relate_paint(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a select-relate-paint graph into an executable program.

    Searches over two explicit structural parameters:
      assignment rule (0-3): how shapes are matched to markers
      alignment mode (0-6): where shapes are placed relative to markers

    Both are visible in the compiled program as integer parameters
    to `match_and_place(grid, rule, align)`.

    Specialization bindings:
      __placement__/assignment_rule -> int
      __placement__/alignment_mode -> int
    If unbound, tries all combinations (bounded: 7 rules * 7 aligns = 49).
    """
    from aria.runtime.ops import has_op
    from aria.runtime.ops.relate_paint import ALL_MATCH_RULES, MATCH_NAMES, ALL_ALIGNS, ALIGN_NAMES
    from aria.verify.verifier import verify

    if not has_op("relocate_objects"):
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="select_relate_paint",
            reason="missing op: relocate_objects",
            missing_ops=("relocate_objects",),
        )

    bound_rule = spec.get("__placement__", "match_rule")
    if bound_rule is None:
        bound_rule = spec.get("__placement__", "assignment_rule")  # compat
    bound_align = spec.get("__placement__", "alignment_mode")

    # Use parameter priors for ordering
    if bound_rule is not None and bound_align is not None:
        pairs_to_try = [(int(bound_rule), int(bound_align))]
    elif bound_rule is not None:
        pairs_to_try = [(int(bound_rule), al) for al in ALL_ALIGNS]
    elif bound_align is not None:
        pairs_to_try = [(mr, int(bound_align)) for mr in ALL_MATCH_RULES]
    else:
        from aria.core.param_priors import rank_relocation_params
        pairs_to_try = rank_relocation_params(demos)

    best_diff = float("inf")
    best_rule = None
    best_align = None

    for rule, align in pairs_to_try:
            prog = Program(
                steps=(
                    Bind("v0", Type.GRID, Call("relocate_objects", (
                        Ref("input"),
                        Literal(rule, Type.INT),
                        Literal(align, Type.INT),
                    ))),
                ),
                output="v0",
            )

            vr = verify(prog, demos)
            if vr.passed:
                return CompileTaskProgram(
                    sketch_task_id=graph.task_id,
                    family="select_relate_paint",
                    program=prog,
                    role_bindings={},
                    slot_bindings={
                        "match_rule": rule,
                        "match_rule_name": MATCH_NAMES.get(rule, str(rule)),
                        "alignment_mode": align,
                        "alignment_mode_name": ALIGN_NAMES.get(align, str(align)),
                    },
                    description=(
                        f"graph->compile: relocate_objects("
                        f"match={MATCH_NAMES.get(rule, rule)}, "
                        f"align={ALIGN_NAMES.get(align, align)})"
                    ),
                )

            # Track best for diagnostic
            try:
                from aria.runtime.executor import execute
                import numpy as np
                diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
                if diff < best_diff:
                    best_diff = diff
                    best_rule = rule
                    best_align = align
            except Exception:
                pass

    combos = len(pairs_to_try)
    desc = f"relocate_objects verified failed (tried {combos} combinations"
    if best_rule is not None:
        desc += (f", best match={MATCH_NAMES.get(best_rule, best_rule)}"
                 f" align={ALIGN_NAMES.get(best_align, best_align)}"
                 f" diff={best_diff}")
    desc += ")"

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="select_relate_paint",
        reason=desc,
    )


def _can_compile_multistep_from_graph(
    graph: SketchGraph,
) -> bool:
    """Check if graph has multiple APPLY_TRANSFORM nodes with evidence."""
    transform_nodes = [
        n for n in graph.nodes.values()
        if n.primitive.name == "APPLY_TRANSFORM" and n.evidence.get("transform")
    ]
    return len(transform_nodes) >= 2


def _compile_multistep_from_graph(
    graph: SketchGraph,
    spec: Specialization,
    demos: tuple[DemoPair, ...],
) -> CompileResult:
    """Compile a multi-step graph by chaining single-op transforms.

    Walks topo order, compiles each APPLY_TRANSFORM node with evidence
    into a concrete runtime call, chains them sequentially.
    """
    from aria.runtime.ops import has_op
    from aria.verify.verifier import verify

    ordered = graph.topo_order()
    steps_list: list[tuple[str, str, tuple]] = []  # (bind_name, op, args)

    prev_ref = "input"
    step_idx = 0

    for nid in ordered:
        node = graph.nodes[nid]
        if node.primitive.name != "APPLY_TRANSFORM":
            continue

        transform = node.evidence.get("transform")
        if transform is None:
            # Check specialization
            transform = spec.get(nid, "transform")
        if transform is None:
            continue

        bind_name = f"v{step_idx}"
        step = _evidence_to_step(transform, node.evidence, spec, nid, prev_ref)
        if step is None:
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="multistep_transform",
                reason=f"cannot compile transform '{transform}' at node {nid}",
            )

        op_name, args = step
        if not has_op(op_name):
            return CompileFailure(
                sketch_task_id=graph.task_id,
                family="multistep_transform",
                reason=f"missing op: {op_name}",
                missing_ops=(op_name,),
            )

        steps_list.append((bind_name, op_name, args))
        prev_ref = bind_name
        step_idx += 1

    if not steps_list:
        return CompileFailure(
            sketch_task_id=graph.task_id,
            family="multistep_transform",
            reason="no compilable transform nodes found",
        )

    # Build program
    program_steps = []
    for bind_name, op_name, args in steps_list:
        program_steps.append(
            Bind(bind_name, Type.GRID, Call(op_name, args))
        )

    prog = Program(steps=tuple(program_steps), output=prev_ref)

    vr = verify(prog, demos)
    if vr.passed:
        return CompileTaskProgram(
            sketch_task_id=graph.task_id,
            family="multistep_transform",
            program=prog,
            role_bindings={},
            slot_bindings={
                "steps": [(op, str(args)) for _, op, args in steps_list],
            },
            description=(
                f"graph→specialization→compile: "
                + " → ".join(op for _, op, _ in steps_list)
            ),
        )

    return CompileFailure(
        sketch_task_id=graph.task_id,
        family="multistep_transform",
        reason=f"multi-step program failed verification",
    )


def _evidence_to_step(
    transform: str,
    evidence: dict,
    spec: Specialization,
    node_id: str,
    input_ref: str,
) -> tuple[str, tuple] | None:
    """Convert transform evidence into (op_name, args) for a Bind."""
    if transform == "rotate":
        degrees = evidence.get("degrees") or spec.get(node_id, "degrees")
        if degrees is None:
            return None
        return ("rotate_grid", (Literal(degrees, Type.INT), Ref(input_ref)))

    if transform == "reflect":
        axis = evidence.get("axis") or spec.get(node_id, "axis")
        if axis is None:
            return None
        axis_int = 0 if axis == "row" else 1
        return ("reflect_grid", (Literal(axis_int, Type.AXIS), Ref(input_ref)))

    if transform == "transpose":
        return ("transpose_grid", (Ref(input_ref),))

    if transform == "fill_enclosed":
        color = evidence.get("fill_color") or spec.get(node_id, "fill_color")
        if color is None:
            return None
        return ("fill_enclosed", (Ref(input_ref), Literal(color, Type.COLOR)))

    return None
