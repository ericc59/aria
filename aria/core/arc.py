"""ARC domain instantiation — bridges core framework to ARC grid reasoning.

This module provides the ARC instantiation of the domain-general pipeline:
  - ARCFitter: propose ComputationGraph hypotheses from demo pairs
  - ARCSpecializer: extract Specialization from graph + demos
  - ARCCompiler: compile graph + specialization into typed ARC programs
  - ARCVerifier: exact pixel-perfect grid comparison

The canonical solve path is:
  1. Fitter seeds (direct ComputationGraph where possible, SketchGraph adapter elsewhere)
  2. Library seeds (via proposer)
  3. Deterministic graph-edit search over seeds
  4. Return best verified program

SketchGraph remains as a transitional adapter for the specializer and compiler,
which still consume SketchGraph internally. The fitter now emits ComputationGraph
directly for grid_transform and object_movement, and via the SketchGraph adapter
for the more complex periodic/alignment/canvas fitters.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from aria.core.graph import (
    CompileFailure as CoreCompileFailure,
    CompileResult as CoreCompileResult,
    CompileSuccess,
    ComputationGraph,
    GraphFragment,
    GraphNode,
    NodeSlot,
    RegionResidual,
    RepairHint,
    ResolvedBinding,
    RoleBinding,
    SubgraphBlame,
    VerifyDiagnostic,
    Specialization,
)
from aria.core.protocol import SolveResult, SolveAttempt, solve


# ---------------------------------------------------------------------------
# Bridge: core ComputationGraph ↔ aria SketchGraph (transitional)
# ---------------------------------------------------------------------------


def sketch_graph_to_core(sg: Any) -> ComputationGraph:
    """Convert an aria SketchGraph to a core ComputationGraph."""
    nodes = {}
    for nid, node in sg.nodes.items():
        nodes[nid] = GraphNode(
            id=node.id,
            op=node.primitive.name,
            inputs=node.inputs,
            roles=tuple(
                RoleBinding(name=r.name, kind=r.kind.name, description=r.description)
                for r in node.roles
            ),
            slots=tuple(
                NodeSlot(name=s.name, typ=s.typ.name, constraint=s.constraint, evidence=s.evidence)
                for s in node.slots
            ),
            description=node.description,
            evidence=dict(node.evidence),
        )
    return ComputationGraph(
        task_id=sg.task_id,
        nodes=nodes,
        output_id=sg.output_id,
        description=sg.description,
        metadata=dict(sg.metadata),
    )


def core_to_specialization(spec: Any) -> Specialization:
    """Convert an aria Specialization to a core Specialization."""
    return Specialization(
        task_id=spec.task_id,
        bindings=tuple(
            ResolvedBinding(node_id=b.node_id, name=b.name, value=b.value, source=b.source)
            for b in spec.bindings
        ),
        metadata=dict(spec.metadata),
    )


# ---------------------------------------------------------------------------
# Direct ComputationGraph construction helpers
# ---------------------------------------------------------------------------


def _make_roles_node(bg_colors: list[int]) -> GraphNode:
    """Standard BIND_ROLE node with BG role."""
    return GraphNode(
        id="roles", op="BIND_ROLE", inputs=("input",),
        roles=(RoleBinding(name="bg", kind="BG"),),
        description="identify roles",
        evidence={"bg_colors": bg_colors},
    )


def _make_transform_node(
    description: str, evidence: dict[str, Any],
) -> GraphNode:
    """APPLY_TRANSFORM node with transform slot."""
    return GraphNode(
        id="transformed", op="APPLY_TRANSFORM", inputs=("roles",),
        slots=(NodeSlot(name="transform", typ="TRANSFORM", evidence=description),),
        description=description,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# ARC Fitter
# ---------------------------------------------------------------------------


class ARCFitter:
    """Propose computation graphs from ARC demo pairs.

    Emits ComputationGraph directly for grid_transform and object_movement.
    Uses the SketchGraph adapter for periodic repair, alignment, and canvas
    fitters (which have more complex graph structures).
    """

    def fit(self, examples: Sequence[Any], task_id: str = "") -> list[ComputationGraph]:
        demos = tuple(examples)
        graphs: list[ComputationGraph] = []

        # --- Direct ComputationGraph emission ---
        graphs.extend(self._fit_grid_transform_direct(demos, task_id))
        graphs.extend(self._fit_object_movement_direct(demos, task_id))

        # --- Frame-interior fitter ---
        graphs.extend(self._fit_frame_interior(demos, task_id))

        # --- SketchGraph adapter path (transitional) ---
        graphs.extend(self._fit_via_sketch_adapter(demos, task_id))

        return graphs

    def _fit_frame_interior(
        self, demos: tuple, task_id: str,
    ) -> list[ComputationGraph]:
        """Propose graphs for frame-interior edit tasks.

        Trigger: framed region detected in all demos, changes are
        concentrated inside frame interiors.

        Tries bounded operations scoped to frame interiors:
        - fill enclosed bg regions inside frame
        - color substitution inside frame
        - periodic repair inside frame (already handled by sketch adapter)
        """
        from aria.decomposition import detect_bg
        from aria.verify.verifier import verify
        from aria.types import Bind, Call, Literal, Program, Ref, Type as T
        from aria.core.grid_perception import perceive_grid

        if not demos:
            return []
        if not all(d.input.shape == d.output.shape for d in demos):
            return []

        # Check: all demos have framed regions
        states = []
        for d in demos:
            state = perceive_grid(d.input)
            if not state.framed_regions:
                return []
            states.append(state)

        bg_colors = [s.bg_color for s in states]

        # Check: changes are mostly inside frames
        for d, state in zip(demos, states):
            diff = d.input != d.output
            n_changed = int(np.sum(diff))
            if n_changed == 0:
                continue
            in_frame = 0
            for fr in state.framed_regions:
                r0, c0 = fr.row, fr.col
                in_frame += int(np.sum(diff[r0:r0+fr.height, c0:c0+fr.width]))
            if in_frame < n_changed * 0.6:
                return []

        # Build candidate programs
        candidates: list[tuple[Program, str, dict]] = []

        # 1. Fill enclosed regions inside frame with each candidate color
        diff_colors: set[int] = set()
        for d in demos:
            diff = d.input != d.output
            if np.any(diff):
                for r, c in zip(*np.where(diff)):
                    diff_colors.add(int(d.output[r, c]))

        for fc in sorted(diff_colors):
            prog = Program(
                steps=(Bind("v0", T.GRID, Call("fill_enclosed_regions", (Ref("input"), Literal(fc, T.INT)))),),
                output="v0",
            )
            candidates.append((prog, f"fill_enclosed_c{fc}", {"transform": "fill_enclosed", "fill_color": fc}))

        # 2. Fill enclosed auto
        prog = Program(
            steps=(Bind("v0", T.GRID, Call("fill_enclosed_regions_auto", (Ref("input"),))),),
            output="v0",
        )
        candidates.append((prog, "fill_enclosed_auto", {"transform": "fill_enclosed_auto"}))

        # 3. Frame-scoped color map
        color_map: dict[int, int] = {}
        cm_consistent = True
        for d in demos:
            for fr_state in perceive_grid(d.input).framed_regions:
                r0, c0 = fr_state.row, fr_state.col
                h, w = fr_state.height, fr_state.width
                int_in = d.input[r0:r0+h, c0:c0+w]
                int_out = d.output[r0:r0+h, c0:c0+w]
                for r in range(h):
                    for c in range(w):
                        ic, oc = int(int_in[r, c]), int(int_out[r, c])
                        if ic != oc:
                            if ic in color_map and color_map[ic] != oc:
                                cm_consistent = False
                            color_map[ic] = oc

        if cm_consistent and color_map and len(color_map) <= 5:
            pairs = sorted(color_map.items())
            n_pairs = len(pairs)
            padded = list(pairs) + [(-1, -1)] * (10 - n_pairs)
            cm_args = [Ref("input"), Literal(n_pairs, T.INT)]
            for fc, tc in padded:
                cm_args.extend([Literal(fc, T.INT), Literal(tc, T.INT)])
            prog = Program(
                steps=(Bind("v0", T.GRID, Call("apply_global_color_map", tuple(cm_args))),),
                output="v0",
            )
            candidates.append((prog, f"frame_color_map_{dict(pairs)}", {"transform": "frame_color_map"}))

        # Verify candidates
        results = []
        for prog, desc, evidence in candidates:
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    graph = ComputationGraph(
                        task_id=task_id,
                        nodes={
                            "roles": _make_roles_node(bg_colors),
                            "transformed": _make_transform_node(desc, evidence),
                        },
                        output_id="transformed",
                        description=f"frame interior: {desc}",
                        metadata={"family": "frame_interior", **evidence},
                    )
                    results.append(graph)
                    break  # first verified wins
            except Exception:
                continue

        return results

    def _fit_grid_transform_direct(
        self, demos: tuple, task_id: str,
    ) -> list[ComputationGraph]:
        """Directly emit ComputationGraph for grid transforms."""
        from aria.decomposition import detect_bg
        from aria.verify.verifier import verify
        from aria.types import Bind, Call, Literal, Program, Ref, Type as T

        if not demos:
            return []
        if all(np.array_equal(d.input, d.output) for d in demos):
            return []

        bg_colors = [detect_bg(d.input) for d in demos]

        candidates: list[tuple[str, tuple, str, dict]] = []
        for deg in (90, 180, 270):
            candidates.append((
                "rotate_grid",
                (Literal(deg, T.INT), Ref("input")),
                f"rotate {deg} degrees",
                {"transform": "rotate", "degrees": deg},
            ))
        for axis, axis_name in ((0, "row"), (1, "col")):
            candidates.append((
                "reflect_grid",
                (Literal(axis, T.AXIS), Ref("input")),
                f"reflect across {axis_name} axis",
                {"transform": "reflect", "axis": axis_name},
            ))
        candidates.append((
            "transpose_grid", (Ref("input"),),
            "transpose grid", {"transform": "transpose"},
        ))

        # Fill enclosed (old op: single color)
        palette: set[int] = set()
        for d in demos:
            for c in range(10):
                if int(np.sum(d.output == c)) > int(np.sum(d.input == c)):
                    palette.add(c)
        palette.update(bg_colors)
        for c in sorted(palette):
            candidates.append((
                "fill_enclosed", (Ref("input"), Literal(c, T.COLOR)),
                f"fill enclosed regions with color {c}",
                {"transform": "fill_enclosed", "fill_color": c},
            ))

        # Fill enclosed regions (new ops: fixed color + auto boundary color)
        for c in sorted(palette):
            candidates.append((
                "fill_enclosed_regions", (Ref("input"), Literal(c, T.INT)),
                f"fill enclosed bg regions with color {c}",
                {"transform": "fill_enclosed_regions", "fill_color": c},
            ))
        candidates.append((
            "fill_enclosed_regions_auto", (Ref("input"),),
            "fill enclosed bg regions with boundary color",
            {"transform": "fill_enclosed_regions_auto"},
        ))

        # Global color map: try small consistent maps from demo evidence
        from aria.core.output_stage1 import _infer_global_color_map_render_spec
        cm_spec = _infer_global_color_map_render_spec(demos)
        if cm_spec is not None:
            pairs = cm_spec.get("pairs", [])
            n_pairs = len(pairs)
            if 1 <= n_pairs <= 10:
                padded = list(pairs) + [(-1, -1)] * (10 - n_pairs)
                cm_args = [Ref("input"), Literal(n_pairs, T.INT)]
                for fc, tc in padded:
                    cm_args.extend([Literal(fc, T.INT), Literal(tc, T.INT)])
                candidates.append((
                    "apply_global_color_map", tuple(cm_args),
                    f"global color map: {dict(pairs)}",
                    {"transform": "global_color_map", "pairs": pairs},
                ))

        results = []
        for op_name, args, desc, evidence in candidates:
            prog = Program(
                steps=(Bind("v0", T.GRID, Call(op_name, args)),),
                output="v0",
            )
            try:
                vr = verify(prog, demos)
                if not vr.passed:
                    continue
            except Exception:
                continue

            graph = ComputationGraph(
                task_id=task_id,
                nodes={
                    "roles": _make_roles_node(bg_colors),
                    "transformed": _make_transform_node(desc, evidence),
                },
                output_id="transformed",
                description=f"grid transform: {desc}",
                metadata={"family": "grid_transform", **evidence},
            )
            results.append(graph)
            break  # first verified wins

        return results

    def _fit_object_movement_direct(
        self, demos: tuple, task_id: str,
    ) -> list[ComputationGraph]:
        """Directly emit ComputationGraph for object movement."""
        from aria.decomposition import detect_bg
        from aria.runtime.ops.selection import _find_objects

        if not demos:
            return []
        if not all(d.input.shape == d.output.shape for d in demos):
            return []

        bg_colors = []
        per_demo_matches = []

        for demo in demos:
            bg = detect_bg(demo.input)
            bg_colors.append(bg)
            inp_objs = [o for o in _find_objects(demo.input) if o.color != bg]
            out_objs = [o for o in _find_objects(demo.output) if o.color != bg]
            from aria.sketch_fit import _match_objects_by_color_size
            matches = _match_objects_by_color_size(inp_objs, out_objs)
            if matches is None:
                return []
            if not any((dr, dc) != (0, 0) for _, _, dr, dc in matches):
                return []
            per_demo_matches.append(matches)

        # Uniform translate
        all_deltas = set()
        for matches in per_demo_matches:
            for _, _, dr, dc in matches:
                if (dr, dc) != (0, 0):
                    all_deltas.add((dr, dc))

        if len(all_deltas) == 1:
            dr, dc = list(all_deltas)[0]
            all_uniform = all(
                all((mdr, mdc) == (dr, dc) or (mdr, mdc) == (0, 0)
                    for _, _, mdr, mdc in matches)
                for matches in per_demo_matches
            )
            if all_uniform:
                evidence = {"strategy": "uniform_translate", "dr": dr, "dc": dc}
                desc = f"translate all foreground objects by ({dr},{dc})"
                return [ComputationGraph(
                    task_id=task_id,
                    nodes={
                        "roles": _make_roles_node(bg_colors),
                        "moved": _make_transform_node(desc, evidence),
                    },
                    output_id="moved",
                    description=desc,
                    metadata={"family": "object_movement", **evidence},
                )]

        # Gravity
        from aria.sketch_fit import _detect_gravity_direction
        gravity_directions = []
        for demo, matches in zip(demos, per_demo_matches):
            h, w = demo.input.shape
            direction = _detect_gravity_direction(matches, h, w)
            if direction is None:
                break
            for io, oo, mdr, mdc in matches:
                if (mdr, mdc) == (0, 0):
                    continue
                ox, oy, ow, oh = oo.bbox
                if direction == "down" and oy + oh != h:
                    direction = None; break
                if direction == "up" and oy != 0:
                    direction = None; break
                if direction == "right" and ox + ow != w:
                    direction = None; break
                if direction == "left" and ox != 0:
                    direction = None; break
            if direction is None:
                break
            gravity_directions.append(direction)

        if len(gravity_directions) == len(demos) and len(set(gravity_directions)) == 1:
            direction = gravity_directions[0]
            evidence = {"strategy": "gravity", "direction": direction}
            desc = f"move all foreground objects to {direction} edge"
            return [ComputationGraph(
                task_id=task_id,
                nodes={
                    "roles": _make_roles_node(bg_colors),
                    "moved": _make_transform_node(desc, evidence),
                },
                output_id="moved",
                description=desc,
                metadata={"family": "object_movement", **evidence},
            )]

        return []

    def _fit_via_sketch_adapter(
        self, demos: tuple, task_id: str,
    ) -> list[ComputationGraph]:
        """Transitional: use SketchGraph adapter for complex fitters."""
        from aria.sketch import SketchGraph
        from aria.sketch_fit import (
            fit_framed_periodic_repair,
            fit_composite_role_alignment,
            fit_canvas_construction,
        )

        graphs = []
        for fitter in [
            fit_framed_periodic_repair,
            fit_composite_role_alignment,
            fit_canvas_construction,
        ]:
            try:
                sketch = fitter(demos, task_id=task_id)
                if sketch is not None:
                    sg = SketchGraph.from_sketch(sketch)
                    graphs.append(sketch_graph_to_core(sg))
            except Exception:
                pass

        return graphs


# ---------------------------------------------------------------------------
# ARC Specializer
# ---------------------------------------------------------------------------


class ARCSpecializer:
    """Extract specialization from an ARC computation graph + demos.

    Still uses SketchGraph adapter internally — the specialize_sketch
    function consumes SketchGraph. This is a migration boundary.
    """

    def specialize(self, graph: ComputationGraph, examples: Sequence[Any]) -> Specialization:
        from aria.sketch_fit import specialize_sketch

        sg = _core_to_sketch_graph(graph)
        demos = tuple(examples)
        aria_spec = specialize_sketch(sg, demos)
        return core_to_specialization(aria_spec)


# ---------------------------------------------------------------------------
# ARC Compiler
# ---------------------------------------------------------------------------


class ARCCompiler:
    """Compile a graph + specialization into an ARC typed program.

    When compile succeeds but verify fails, produces structured
    VerifyDiagnostic with per-demo diffs and repair hints.
    """

    def compile(
        self,
        graph: ComputationGraph,
        specialization: Specialization,
        examples: Sequence[Any],
    ) -> CoreCompileResult:
        from aria.sketch_compile import (
            compile_sketch_graph,
            CompileTaskProgram,
            CompilePerDemoPrograms,
            CompileFailure,
        )

        sg = _core_to_sketch_graph(graph)
        aria_spec = _core_to_aria_specialization(specialization)
        demos = tuple(examples)

        result = compile_sketch_graph(sg, aria_spec, demos)

        if isinstance(result, CompileTaskProgram):
            return CompileSuccess(
                task_id=graph.task_id,
                program=result.program,
                bindings_used=result.slot_bindings,
                description=result.description,
                scope="task",
            )
        elif isinstance(result, CompilePerDemoPrograms):
            return CompileSuccess(
                task_id=graph.task_id,
                program=result.programs,
                bindings_used=result.slot_bindings,
                description=result.description,
                scope="per_example",
            )
        else:
            # Build diagnostics for near-miss failures
            diagnostic = _build_near_miss_diagnostic(
                graph, specialization, result, demos,
            )
            return CoreCompileFailure(
                task_id=graph.task_id,
                reason=result.reason,
                missing_ops=result.missing_ops,
                diagnostic=diagnostic,
            )


# ---------------------------------------------------------------------------
# ARC Verifier
# ---------------------------------------------------------------------------


class _VR:
    def __init__(self, passed: bool):
        self.passed = passed


class ARCVerifier:
    """Exact pixel-perfect grid verification."""

    def verify(self, program: Any, examples: Sequence[Any]) -> _VR:
        from aria.verify.verifier import verify
        demos = tuple(examples)
        vr = verify(program, demos)
        return _VR(vr.passed)


# ---------------------------------------------------------------------------
# Near-miss diagnostics
# ---------------------------------------------------------------------------


def _build_near_miss_diagnostic(
    graph: ComputationGraph,
    spec: Specialization,
    failure: Any,
    demos: tuple,
) -> VerifyDiagnostic | None:
    """Build structured diagnostics when a graph nearly compiled.

    Produces:
    - per-demo pixel diffs
    - region-localized residuals (frame interior vs exterior)
    - subgraph-level blame (which nodes are the wrong mechanism)
    - replacement fragments (typed alternatives for blamed subgraphs)
    - parameter repair hints
    """
    reason = getattr(failure, "reason", "")
    if "verified failed" not in reason and "verify" not in reason.lower():
        return None

    per_demo_diffs, total_diff, total_pixels = _measure_residual(graph, spec, demos)
    if per_demo_diffs is None:
        return None

    diff_fraction = total_diff / max(total_pixels, 1)
    failed_demo = next((i for i, d in enumerate(per_demo_diffs) if d > 0), -1)

    # Parameter repair hints
    hints = _generate_repair_hints(graph, spec, demos, per_demo_diffs)
    blamed_bindings = list(dict.fromkeys(h.binding_name for h in hints))

    # Region-localized residuals
    region_residuals = _compute_region_residuals(graph, spec, demos)

    # Subgraph blame + static replacement fragments
    blames, static_fragments = _compute_subgraph_blame(
        graph, spec, demos, per_demo_diffs, hints,
    )

    # Dynamic task-conditioned fragment generation
    predicted_grids = _get_predicted_grids(graph, spec, demos)
    dynamic_fragments = _generate_dynamic_fragments(
        graph, spec, diagnostic_stub=None, demos=demos,
        predicted_grids=predicted_grids,
    )

    # Combine static + dynamic, dedup by label
    all_fragments = list(static_fragments)
    seen_labels = {f.label for f in all_fragments}
    for gf in dynamic_fragments:
        if gf.fragment.label not in seen_labels:
            all_fragments.append(gf.fragment)
            seen_labels.add(gf.fragment.label)

    # Add dynamic fragment labels to blame replacement_labels
    dynamic_labels = tuple(gf.fragment.label for gf in dynamic_fragments)
    if blames and dynamic_labels:
        old_blame = blames[0]
        combined_labels = old_blame.replacement_labels + dynamic_labels
        blames[0] = SubgraphBlame(
            node_ids=old_blame.node_ids,
            ops=old_blame.ops,
            residual_overlap=old_blame.residual_overlap,
            confidence=old_blame.confidence,
            replacement_labels=combined_labels,
            reason=old_blame.reason,
        )

    return VerifyDiagnostic(
        per_demo_diff=tuple(per_demo_diffs),
        total_diff=total_diff,
        failed_demo=failed_demo,
        diff_fraction=diff_fraction,
        repair_hints=tuple(hints),
        blamed_bindings=tuple(blamed_bindings),
        region_residuals=tuple(region_residuals),
        subgraph_blames=tuple(blames),
        replacement_fragments=tuple(all_fragments),
        description=f"near-miss: {total_diff}/{total_pixels} pixels wrong ({diff_fraction:.1%})",
    )


def _measure_residual(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
) -> tuple[list[int] | None, int, int]:
    """Build the program directly from spec parameters and measure per-demo diffs.

    Unlike _try_compile_variants, this builds the program without internal
    verification, so we can measure the diff even when verification fails.
    """
    try:
        from aria.runtime.executor import execute
    except Exception:
        return None, 0, 0

    prog = _build_program_from_spec(graph, spec)
    if prog is None:
        return None, 0, 0

    per_demo = []
    total = 0
    total_pixels = 0
    for demo in demos:
        try:
            predicted = execute(prog, demo.input)
            diff = int(np.sum(predicted != demo.output))
        except Exception:
            diff = demo.input.size
        per_demo.append(diff)
        total += diff
        total_pixels += demo.output.size

    return per_demo, total, total_pixels


def _compute_region_residuals(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
) -> list[RegionResidual]:
    """Compute where in the grid the residual is concentrated."""
    try:
        from aria.runtime.executor import execute
        from aria.decomposition import detect_bg, detect_framed_regions
    except Exception:
        return []

    prog = _build_program_from_spec(graph, spec)
    if prog is None:
        return []

    residuals = []
    frame_diff_total = 0
    frame_pixels_total = 0
    interior_diff_total = 0
    interior_pixels_total = 0

    for demo in demos:
        try:
            predicted = execute(prog, demo.input)
            diff_mask = predicted != demo.output
        except Exception:
            continue

        rows, cols = demo.input.shape
        bg = detect_bg(demo.input)

        # Try to detect framed regions
        try:
            regions = detect_framed_regions(demo.input)
        except Exception:
            regions = []

        if regions:
            # Compute frame vs interior residuals
            for region in regions:
                r0 = region.row
                c0 = region.col
                r1 = r0 + region.height
                c1 = c0 + region.width
                interior_mask = np.zeros_like(diff_mask)
                interior_mask[r0:r1, c0:c1] = True
                frame_mask = ~interior_mask

                frame_diff_total += int(np.sum(diff_mask & frame_mask))
                frame_pixels_total += int(np.sum(frame_mask))
                interior_diff_total += int(np.sum(diff_mask & interior_mask))
                interior_pixels_total += int(np.sum(interior_mask))
        else:
            # No framed regions — full grid
            interior_diff_total += int(np.sum(diff_mask))
            interior_pixels_total += diff_mask.size

    if frame_pixels_total > 0:
        residuals.append(RegionResidual(
            region_label="frame",
            diff_pixels=frame_diff_total,
            total_pixels=frame_pixels_total,
            diff_fraction=frame_diff_total / max(frame_pixels_total, 1),
        ))
    if interior_pixels_total > 0:
        residuals.append(RegionResidual(
            region_label="interior",
            diff_pixels=interior_diff_total,
            total_pixels=interior_pixels_total,
            diff_fraction=interior_diff_total / max(interior_pixels_total, 1),
        ))

    return residuals


# ---------------------------------------------------------------------------
# Subgraph blame + replacement fragments
# ---------------------------------------------------------------------------

# Structural replacement fragments: small typed graph shapes that can
# substitute for blamed subgraphs. Described structurally, not by family.

_REPLACEMENT_FRAGMENTS: dict[str, GraphFragment] = {}


def _init_replacement_fragments() -> None:
    """Initialize the small typed replacement fragment library."""
    global _REPLACEMENT_FRAGMENTS
    if _REPLACEMENT_FRAGMENTS:
        return

    # Fragment: unary transform (roles -> single transform)
    _REPLACEMENT_FRAGMENTS["unary_transform"] = GraphFragment(
        label="unary_transform",
        nodes={
            "_frag_t": GraphNode(
                id="_frag_t", op="APPLY_TRANSFORM", inputs=("_frag_in",),
                slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
                description="single transform",
            ),
        },
        input_id="_frag_in",
        output_id="_frag_t",
        description="replace with single transform",
    )

    # Fragment: 2D motif repair only (no 1D line step)
    _REPLACEMENT_FRAGMENTS["motif_repair_only"] = GraphFragment(
        label="motif_repair_only",
        nodes={
            "_frag_2d": GraphNode(
                id="_frag_2d", op="REPAIR_2D_MOTIF", inputs=("_frag_in",),
                description="2D motif repair only",
            ),
        },
        input_id="_frag_in",
        output_id="_frag_2d",
        description="replace with 2D motif repair only",
    )

    # Fragment: 1D line repair only (no 2D motif step)
    _REPLACEMENT_FRAGMENTS["line_repair_only"] = GraphFragment(
        label="line_repair_only",
        nodes={
            "_frag_1d": GraphNode(
                id="_frag_1d", op="REPAIR_LINES", inputs=("_frag_in",),
                slots=(
                    NodeSlot(name="axis", typ="AXIS"),
                    NodeSlot(name="period", typ="INT"),
                ),
                description="1D line repair only",
            ),
        },
        input_id="_frag_in",
        output_id="_frag_1d",
        description="replace with 1D line repair only",
    )

    # Fragment: relation alignment
    _REPLACEMENT_FRAGMENTS["relation_alignment"] = GraphFragment(
        label="relation_alignment",
        nodes={
            "_frag_rel": GraphNode(
                id="_frag_rel", op="APPLY_RELATION", inputs=("_frag_in",),
                description="relation alignment",
            ),
        },
        input_id="_frag_in",
        output_id="_frag_rel",
        description="replace with relation alignment",
    )

    # Fragment: partition -> local rule
    _REPLACEMENT_FRAGMENTS["partition_local_rule"] = GraphFragment(
        label="partition_local_rule",
        nodes={
            "_frag_part": GraphNode(
                id="_frag_part", op="PARTITION_GRID", inputs=("_frag_in",),
            ),
            "_frag_rule": GraphNode(
                id="_frag_rule", op="APPLY_TRANSFORM", inputs=("_frag_part",),
                slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
                description="local rule on partitioned cells",
            ),
        },
        input_id="_frag_in",
        output_id="_frag_rule",
        description="replace with partition + local rule",
    )


def _get_replacement_fragments() -> dict[str, GraphFragment]:
    _init_replacement_fragments()
    return _REPLACEMENT_FRAGMENTS


def _compute_subgraph_blame(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
    per_demo_diffs: list[int],
    hints: list[RepairHint],
) -> tuple[list[SubgraphBlame], list[GraphFragment]]:
    """Identify which subgraph is the wrong mechanism and suggest replacements.

    Heuristic: if parameter repair hints have low confidence (the axis/period
    alternatives don't materially reduce the diff), the repair mechanism
    itself is blamed, not just its parameters.
    """
    blames: list[SubgraphBlame] = []
    fragments: list[GraphFragment] = []
    all_fragments = _get_replacement_fragments()

    # Check if parameter repair is saturated (no high-confidence hints)
    max_hint_conf = max((h.confidence for h in hints), default=0.0)
    parameter_repair_saturated = max_hint_conf < 0.5

    if not parameter_repair_saturated:
        return blames, fragments  # parameter repair may still work

    # Identify "action" nodes (non-perception nodes that do the transform)
    ops = graph.op_set
    action_node_ids = []
    action_ops = []
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        if node.op in ("BIND_ROLE", "PEEL_FRAME", "PARTITION_GRID"):
            continue  # perception/setup nodes — not blamed
        action_node_ids.append(nid)
        action_ops.append(node.op)

    if not action_node_ids:
        return blames, fragments

    # Determine which replacement fragments are compatible
    compatible_labels = _compatible_replacements(action_ops)

    blame = SubgraphBlame(
        node_ids=tuple(action_node_ids),
        ops=tuple(action_ops),
        residual_overlap=1.0,  # the full residual is in the action subgraph
        confidence=0.7 if parameter_repair_saturated else 0.3,
        replacement_labels=tuple(compatible_labels),
        reason=(f"parameter repair saturated (max_conf={max_hint_conf:.2f}); "
                f"mechanism {'+'.join(action_ops)} likely wrong"),
    )
    blames.append(blame)

    # Collect the actual fragments
    for label in compatible_labels:
        frag = all_fragments.get(label)
        if frag is not None:
            fragments.append(frag)

    return blames, fragments


def _compatible_replacements(action_ops: list[str]) -> list[str]:
    """Determine which replacement fragments are structurally compatible
    with the current action subgraph."""
    labels = []
    ops_set = set(action_ops)

    # If current mechanism is repair-based, suggest transform alternatives
    is_repair = ops_set & {"REPAIR_LINES", "REPAIR_2D_MOTIF", "REPAIR_MISMATCH"}
    is_transform = "APPLY_TRANSFORM" in ops_set
    is_relation = "APPLY_RELATION" in ops_set

    if is_repair:
        labels.append("unary_transform")
        labels.append("relation_alignment")
        labels.append("partition_local_rule")
        # Also suggest repair variants
        if "REPAIR_LINES" in ops_set and "REPAIR_2D_MOTIF" in ops_set:
            labels.append("motif_repair_only")
            labels.append("line_repair_only")
        elif "REPAIR_LINES" in ops_set:
            labels.append("motif_repair_only")
        elif "REPAIR_2D_MOTIF" in ops_set:
            labels.append("line_repair_only")

    if is_transform:
        labels.append("relation_alignment")
        labels.append("partition_local_rule")

    if is_relation:
        labels.append("unary_transform")
        labels.append("partition_local_rule")

    # Always offer the simplest alternative if not already the mechanism
    if not is_transform:
        if "unary_transform" not in labels:
            labels.append("unary_transform")

    return labels


def _get_predicted_grids(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
) -> list[Any]:
    """Run the near-miss program on each demo input to get predicted grids."""
    try:
        from aria.runtime.executor import execute
    except Exception:
        return []

    prog = _build_program_from_spec(graph, spec)
    if prog is None:
        return []

    results = []
    for demo in demos:
        try:
            results.append(execute(prog, demo.input))
        except Exception:
            results.append(None)
    return results


def _generate_dynamic_fragments(
    graph: ComputationGraph,
    spec: Specialization,
    diagnostic_stub: VerifyDiagnostic | None,
    demos: tuple,
    predicted_grids: list[Any],
) -> list:
    """Generate dynamic fragments conditioned on the task's residual pattern."""
    try:
        from aria.core.fragment_gen import generate_fragments
    except Exception:
        return []

    if diagnostic_stub is None:
        # Build a minimal diagnostic stub for the generator
        diagnostic_stub = VerifyDiagnostic(total_diff=0, diff_fraction=0.0)

    try:
        return generate_fragments(
            graph, spec, diagnostic_stub, demos,
            predicted_grids=predicted_grids if predicted_grids else None,
        )
    except Exception:
        return []


def _build_program_from_spec(
    graph: ComputationGraph,
    spec: Specialization,
) -> Any:
    """Build a program directly from graph + specialization parameters.

    Handles the common ARC compilation patterns (periodic repair,
    grid transform, movement) without running internal verification.
    """
    from aria.types import Bind, Call, Literal, Program, Ref, Type
    from aria.runtime.ops import has_op

    ops = graph.op_set
    axis_str = spec.get("__task__", "dominant_axis")
    period = spec.get("__task__", "dominant_period")

    # Periodic repair path — prefer new composite op
    if ("REPAIR_LINES" in ops or "REPAIR_2D_MOTIF" in ops) and axis_str and period:
        axis_int = 0 if axis_str == "row" else 1
        repair_mode = spec.get("__periodic__", "repair_mode")
        if repair_mode is None:
            repair_mode = 2  # default: lines_then_2d
        if has_op("periodic_repair"):
            return Program(
                steps=(Bind("v0", Type.GRID, Call("periodic_repair", (
                    Ref("input"), Literal(axis_int, Type.INT),
                    Literal(period, Type.INT), Literal(int(repair_mode), Type.INT),
                ))),),
                output="v0",
            )
        # Fallback to old ops
        elif has_op("repair_framed_lines"):
            return Program(
                steps=(Bind("v0", Type.GRID, Call("repair_framed_lines", (
                    Ref("input"), Literal(axis_int, Type.INT),
                    Literal(period, Type.INT),
                ))),),
                output="v0",
            )

    # Grid transform path
    transform = spec.get("__grid_transform__", "transform")
    if transform and "APPLY_TRANSFORM" in ops:
        degrees = spec.get("__grid_transform__", "degrees")
        axis = spec.get("__grid_transform__", "axis")
        fill_color = spec.get("__grid_transform__", "fill_color")

        if transform == "rotate" and degrees and has_op("rotate_grid"):
            return Program(
                steps=(Bind("v0", Type.GRID, Call("rotate_grid", (
                    Literal(degrees, Type.INT), Ref("input"),
                ))),),
                output="v0",
            )
        if transform == "reflect" and axis is not None and has_op("reflect_grid"):
            axis_int = 0 if axis == "row" else 1
            return Program(
                steps=(Bind("v0", Type.GRID, Call("reflect_grid", (
                    Literal(axis_int, Type.AXIS), Ref("input"),
                ))),),
                output="v0",
            )
        if transform == "fill_enclosed" and fill_color is not None and has_op("fill_enclosed"):
            return Program(
                steps=(Bind("v0", Type.GRID, Call("fill_enclosed", (
                    Ref("input"), Literal(fill_color, Type.COLOR),
                ))),),
                output="v0",
            )

    # Replication path
    if (has_select or has_relation or has_paint) and has_op("replicate_templates"):
        kr = spec.get("__replicate__", "key_rule") or 0
        sp = spec.get("__replicate__", "source_policy") or 0
        pr = spec.get("__replicate__", "placement_rule") or 0
        return Program(
            steps=(Bind("v0", Type.GRID, Call("replicate_templates", (
                Ref("input"), Literal(int(kr), Type.INT),
                Literal(int(sp), Type.INT), Literal(int(pr), Type.INT),
            ))),),
            output="v0",
        )

    # Relocation path
    has_select = "SELECT_SUBSET" in ops
    has_relation = "APPLY_RELATION" in ops
    has_paint = "PAINT" in ops
    if (has_select or has_relation or has_paint) and has_op("relocate_objects"):
        rule = spec.get("__placement__", "match_rule")
        if rule is None:
            rule = spec.get("__placement__", "assignment_rule")
        align = spec.get("__placement__", "alignment_mode")
        if rule is None:
            rule = 0
        if align is None:
            align = 0
        return Program(
            steps=(Bind("v0", Type.GRID, Call("relocate_objects", (
                Ref("input"), Literal(int(rule), Type.INT), Literal(int(align), Type.INT),
            ))),),
            output="v0",
        )

    return None


def _try_compile_variants(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
    variants: list[tuple[Specialization, str]],
) -> list[tuple[Any, str, Specialization]]:
    """Try compiling with variant specializations, return (program, label, spec) for each that compiles."""
    from aria.sketch_compile import (
        compile_sketch_graph,
        CompileTaskProgram,
    )

    results = []
    for var_spec, label in variants:
        try:
            sg = _core_to_sketch_graph(graph)
            aria_spec = _core_to_aria_specialization(var_spec)
            result = compile_sketch_graph(sg, aria_spec, demos)
            if isinstance(result, CompileTaskProgram):
                results.append((result.program, label, var_spec))
        except Exception:
            pass
    return results


def _generate_repair_hints(
    graph: ComputationGraph,
    spec: Specialization,
    demos: tuple,
    per_demo_diffs: list[int],
) -> list[RepairHint]:
    """Generate repair hints by trying alternative parameter values."""
    from aria.verify.verifier import verify

    hints: list[RepairHint] = []

    # Identify mutable bindings (axis, period, strategy, etc.)
    _ALTERNATIVES: dict[str, list] = {
        "dominant_axis": ["row", "col"],
        "dominant_period": [2, 3, 4, 5, 6, 7, 8],
        "axis": ["row", "col"],
        "period": [2, 3, 4, 5, 6, 7, 8],
        "degrees": [90, 180, 270],
        "fill_color": list(range(10)),
        "strategy": ["tile", "upscale", "uniform_translate", "gravity"],
        "direction": ["up", "down", "left", "right"],
        "assignment_rule": [0, 1, 2, 3, 4, 5, 6],  # compat
        "match_rule": [0, 1, 2, 3, 4, 5, 6],  # all match rules
        "alignment_mode": [0, 1, 2, 3, 4, 5, 6],  # all alignment modes
    }

    for binding in spec.bindings:
        alts = _ALTERNATIVES.get(binding.name)
        if alts is None:
            continue
        # Filter to alternatives different from current
        candidates = [v for v in alts if v != binding.value]
        if not candidates:
            continue

        # Try each alternative
        best_alt = None
        best_diff = sum(per_demo_diffs)
        for alt_val in candidates:
            var_spec = _replace_binding(spec, binding.node_id, binding.name, alt_val)
            programs = _try_compile_variants(graph, var_spec, demos, [(var_spec, str(alt_val))])
            if not programs:
                continue

            prog = programs[0][0]
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    # This alternative solves it!
                    hints.append(RepairHint(
                        node_id=binding.node_id,
                        binding_name=binding.name,
                        current_value=binding.value,
                        alternatives=(alt_val,),
                        confidence=1.0,
                        reason=f"alternative {binding.name}={alt_val} passes verification",
                    ))
                    return hints  # early return: we found a fix
            except Exception:
                pass

            # Measure diff for this alternative
            try:
                from aria.runtime.executor import execute
                alt_diff = 0
                for demo in demos:
                    try:
                        predicted = execute(prog, demo.input)
                        alt_diff += int(np.sum(predicted != demo.output))
                    except Exception:
                        alt_diff += demo.input.size
                if alt_diff < best_diff:
                    best_diff = alt_diff
                    best_alt = alt_val
            except Exception:
                pass

        # Even if no alternative solves it, suggest the best one
        useful_alts = tuple(v for v in candidates
                           if v == best_alt) if best_alt is not None else ()
        if useful_alts or len(candidates) <= 5:
            hints.append(RepairHint(
                node_id=binding.node_id,
                binding_name=binding.name,
                current_value=binding.value,
                alternatives=useful_alts if useful_alts else tuple(candidates[:5]),
                confidence=0.5 if best_alt is not None else 0.1,
                reason=(f"best alternative reduces diff from {sum(per_demo_diffs)} to {best_diff}"
                        if best_alt else "enumerated alternatives"),
            ))

    return hints


def _replace_binding(
    spec: Specialization,
    node_id: str,
    name: str,
    new_value: Any,
) -> Specialization:
    """Return a new Specialization with one binding replaced."""
    new_bindings = []
    replaced = False
    for b in spec.bindings:
        if b.node_id == node_id and b.name == name:
            new_bindings.append(ResolvedBinding(
                node_id=b.node_id, name=b.name,
                value=new_value, source="repair_hint",
            ))
            replaced = True
        else:
            new_bindings.append(b)
    if not replaced:
        new_bindings.append(ResolvedBinding(
            node_id=node_id, name=name,
            value=new_value, source="repair_hint",
        ))
    return Specialization(
        task_id=spec.task_id,
        bindings=tuple(new_bindings),
        metadata=spec.metadata,
    )


def _verify_scene_program_as_solve(scene_prog, demos, verifier):
    """Verify a SceneProgram across all demos, return a Program wrapper if it passes."""
    from aria.core.scene_executor import execute_scene_program
    import numpy as np

    for d in demos:
        try:
            if hasattr(scene_prog, 'verify_on_demo'):
                if not scene_prog.verify_on_demo(d.input, d.output):
                    return None
                continue
            elif hasattr(scene_prog, 'execute'):
                result = scene_prog.execute(d.input)
            else:
                result = execute_scene_program(scene_prog, d.input)
        except Exception:
            return None
        if 'result' in dir() and result is not None:
            if result.shape != d.output.shape or not np.array_equal(result, d.output):
                return None

    # If it's a CorrespondenceProgram, wrap directly
    if hasattr(scene_prog, 'verify_on_demo') or (hasattr(scene_prog, 'execute') and not hasattr(scene_prog, 'steps')):
        from aria.types import Bind, Call, Literal, Program, Ref, Type
        return Program(
            steps=(Bind("out", Type.GRID, Call("correspondence_solve", (Ref("input"),))),),
            output="out",
        )

    # Wrap as a callable Program for compatibility
    from aria.types import Bind, Call, Literal, Program, Ref, Type

    # Serialize scene program steps as a JSON-like literal
    import json
    step_data = []
    for s in scene_prog.steps:
        step_data.append({
            "op": s.op.value,
            "inputs": list(s.inputs),
            "params": {k: _safe_serialize(v) for k, v in s.params.items()},
            "output_id": s.output_id,
        })
    steps_json = json.dumps(step_data)

    return Program(
        steps=(
            Bind(
                "v0",
                Type.GRID,
                Call("execute_scene_program_json", (Ref("input"), Literal(steps_json, Type.INT))),
            ),
        ),
        output="v0",
    )


def _safe_serialize(val):
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (tuple, list)):
        return [_safe_serialize(v) for v in val]
    return str(val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_arc_task(
    demos: tuple,
    task_id: str = "",
    *,
    use_editor_search: bool = True,
    use_learned_editor: bool = False,
    library: Any = None,
) -> SolveResult:
    """Solve an ARC task using the canonical pipeline.

    Order:
    1. Static fitter pipeline (fit -> specialize -> compile -> verify)
    2. If unsolved: deterministic graph-edit search over fitter + library seeds
    3. If unsolved and use_learned_editor: per-task CEM-trained graph editor
    """
    from aria.core.output_stage1 import (
        compile_stage1_program,
        infer_output_stage1_spec,
    )

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    # Hard gate: no downstream solving unless stage 1 verifies output size.
    # Exception: scene programs can bypass the gate since they infer their own size.
    stage1 = infer_output_stage1_spec(demos)
    if stage1 is None:
        try:
            from aria.core.scene_solve import infer_scene_programs
            scene_progs = infer_scene_programs(demos)
            for sp in scene_progs:
                scene_result = _verify_scene_program_as_solve(sp, demos, verifier)
                if scene_result is not None:
                    return SolveResult(
                        task_id=task_id,
                        solved=True,
                        winning_program=scene_result,
                    )
        except Exception:
            pass
        return SolveResult(task_id=task_id, solved=False)
    stage1_program = compile_stage1_program(stage1)
    if stage1_program is not None and verifier.verify(stage1_program, demos).passed:
        return SolveResult(task_id=task_id, solved=True, winning_program=stage1_program)

    # Phase 0.5: multi-step scene programs
    try:
        from aria.core.scene_solve import infer_scene_programs
        scene_progs = infer_scene_programs(demos)
        for sp in scene_progs:
            # Wrap scene program as a runtime Program for the verifier
            scene_result = _verify_scene_program_as_solve(sp, demos, verifier)
            if scene_result is not None:
                return SolveResult(
                    task_id=task_id,
                    solved=True,
                    winning_program=scene_result,
                )
    except Exception:
        pass

    # Phase 1: static pipeline
    result = solve(
        examples=demos,
        fitter=fitter,
        specializer=specializer,
        compiler=compiler,
        verifier=verifier,
        task_id=task_id,
    )

    if result.solved:
        return result

    # Collect seeds once for both editor phases
    seeds = None
    if use_editor_search or use_learned_editor:
        from aria.core.seeds import collect_seeds
        seeds = collect_seeds(
            examples=demos,
            fitter=fitter,
            specializer=specializer,
            compiler=compiler,
            verifier=verifier,
            task_id=task_id,
            library=library,
        )

    # Phase 2: deterministic graph-edit search
    if use_editor_search and seeds:
        from aria.core.editor_search import search_from_seeds

        edit_result = search_from_seeds(
            seeds=seeds,
            examples=demos,
            specializer=specializer,
            compiler=compiler,
            verifier=verifier,
            task_id=task_id,
        )

        if edit_result.solved and edit_result.program is not None:
            return SolveResult(
                task_id=task_id,
                solved=True,
                winning_program=edit_result.program,
                attempts=result.attempts,
                graphs_proposed=result.graphs_proposed,
                graphs_compiled=result.graphs_compiled + edit_result.compiles_attempted,
                graphs_verified=result.graphs_verified + 1,
            )

    # Phase 3: per-task learned graph editor
    if use_learned_editor and seeds:
        from aria.core.editor_train import train_and_solve

        learned_result = train_and_solve(
            seeds=seeds,
            examples=demos,
            specializer=specializer,
            compiler=compiler,
            verifier=verifier,
            task_id=task_id,
        )

        if learned_result.solved and learned_result.program is not None:
            return SolveResult(
                task_id=task_id,
                solved=True,
                winning_program=learned_result.program,
                attempts=result.attempts,
                graphs_proposed=result.graphs_proposed,
                graphs_compiled=result.graphs_compiled + learned_result.compiles_attempted,
                graphs_verified=result.graphs_verified + 1,
            )

    return result


# ---------------------------------------------------------------------------
# Internal: convert between core and aria representations (transitional)
# ---------------------------------------------------------------------------


def _core_to_sketch_graph(graph: ComputationGraph) -> Any:
    """Convert a core ComputationGraph back to an aria SketchGraph.

    Transitional: needed while specialize_sketch and compile_sketch_graph
    still consume SketchGraph. Will be removed when those are ported.
    """
    from aria.sketch import Primitive, SketchGraph, SketchNode, RoleVar, RoleKind, Slot, SlotType

    nodes = {}
    for nid, node in graph.nodes.items():
        roles = tuple(
            RoleVar(name=r.name, kind=RoleKind[r.kind], description=r.description)
            for r in node.roles
        )
        slots = tuple(
            Slot(name=s.name, typ=SlotType[s.typ], constraint=s.constraint, evidence=s.evidence)
            for s in node.slots
        )
        nodes[nid] = SketchNode(
            id=nid,
            primitive=Primitive[node.op],
            inputs=node.inputs,
            roles=roles,
            slots=slots,
            description=node.description,
            evidence=dict(node.evidence),
        )
    return SketchGraph(
        task_id=graph.task_id,
        nodes=nodes,
        output_id=graph.output_id,
        description=graph.description,
        metadata=dict(graph.metadata),
    )


def _core_to_aria_specialization(spec: Specialization) -> Any:
    """Convert a core Specialization back to an aria Specialization.

    Transitional: needed while compile_sketch_graph consumes aria Specialization.
    """
    from aria.sketch import Specialization as AriaSpec, ResolvedBinding as AriaRB
    return AriaSpec(
        task_id=spec.task_id,
        bindings=tuple(
            AriaRB(node_id=b.node_id, name=b.name, value=b.value, source=b.source)
            for b in spec.bindings
        ),
        metadata=dict(spec.metadata),
    )
