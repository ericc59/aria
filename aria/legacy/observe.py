"""Per-object observation: look at what happened to each object and find the rule.

The generic learning algorithm:
1. For each input object, observe what changed in the output at/around it
2. Group observations by entity key (color, then color+size if needed)
3. For each group, check if the change is consistent across all members and demos
4. Emit an ObjectRule for each consistent pattern
5. Express rules as flat-grid or object-pipeline programs and verify

Rule families:
- surround: spatial pattern added around each object of a group
- recolor: pixel-level color map
- move: objects in a group all moved by the same offset
- remove: objects in a group all disappeared
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.graph.extract import extract_with_delta
from aria.runtime.program import program_to_text
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Delta,
    Dir,
    Expr,
    Grid,
    Literal,
    ObjectNode,
    Program,
    Ref,
    StateGraph,
    Type,
)
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectObservation:
    """What happened to/around one input object."""

    obj_id: int
    color: int
    position: tuple[int, int]  # (row, col)
    size: int

    added_offsets: tuple[tuple[int, int], ...]
    added_color: int | None

    color_changed_to: int | None
    moved_to: tuple[int, int] | None
    removed: bool
    shape_transform: str | None = None  # "rot90", "rot180", "rot270", "flip_h", "flip_v", None


# Grouping key type: a hashable tuple describing how an observation was grouped
GroupKey = tuple  # (color,) or (color, size)


@dataclass(frozen=True)
class ObjectRule:
    """A generalized rule derived from multiple object observations."""

    kind: str
    input_color: int | None
    output_color: int | None
    offsets: tuple[tuple[int, int], ...] | None
    details: dict[str, Any]
    expressible: bool = True
    grouping_key: str = "color"
    expression_path: str = "flat"  # "flat", "pipeline", or "diagnosed"


@dataclass(frozen=True)
class ObservationSynthesisResult:
    """Result of per-object observation and rule inference."""

    solved: bool
    winning_program: Program | None
    candidates_tested: int
    rules: tuple[ObjectRule, ...]
    observations_per_demo: tuple[tuple[ObjectObservation, ...], ...]


@dataclass(frozen=True)
class CorrespondenceResult:
    """Result of correspondence-driven move analysis."""

    rules: tuple[ObjectRule, ...]
    programs: tuple[tuple[Program, ObjectRule], ...]
    matched_count: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DimsReconstructionResult:
    """Result of dims-change output reconstruction."""

    attempted: bool
    solved: bool
    winning_program: Program | None
    candidates_tested: int
    mode: str  # "none", "objects_on_blank", "scale_paint", "whole_grid_transform"
    inferred_output_dims_source: str  # "fixed", "scale_input", "transpose", "unknown"


def dims_change_reconstruct(
    demos: tuple[DemoPair, ...],
) -> DimsReconstructionResult:
    """Attempt to reconstruct outputs on a different-sized canvas.

    Strategies tried in order:
    1. whole_grid_transform — rotate/reflect/transpose/tile/upscale the entire input
    2. objects_on_blank — extract objects, paint on a blank canvas of inferred dims
    3. scale_paint — scale objects individually, paint on scaled canvas
    """
    if not demos:
        return DimsReconstructionResult(
            attempted=False, solved=False, winning_program=None,
            candidates_tested=0, mode="none",
            inferred_output_dims_source="unknown",
        )

    # Check if this is actually a dims-change task
    same_dims = all(d.input.shape == d.output.shape for d in demos)
    if same_dims:
        return DimsReconstructionResult(
            attempted=False, solved=False, winning_program=None,
            candidates_tested=0, mode="none",
            inferred_output_dims_source="fixed",
        )

    fixed_dims, dims_source = _infer_dims_source(demos)
    tested = 0

    # Strategy 1: whole-grid transforms (rotate, reflect, transpose, tile, upscale)
    prog, count = _try_whole_grid_transforms(demos, dims_source)
    tested += count
    if prog is not None:
        return DimsReconstructionResult(
            attempted=True, solved=True, winning_program=prog,
            candidates_tested=tested, mode="whole_grid_transform",
            inferred_output_dims_source=dims_source,
        )

    # Strategy 2: objects on blank canvas with fixed output dims
    if fixed_dims is not None:
        prog, count = _try_objects_on_blank(demos, fixed_dims)
        tested += count
        if prog is not None:
            return DimsReconstructionResult(
                attempted=True, solved=True, winning_program=prog,
                candidates_tested=tested, mode="objects_on_blank",
                inferred_output_dims_source=dims_source,
            )

    # Strategy 3: scale + paint (when dims_source is scale_input)
    if dims_source == "scale_input":
        prog, count = _try_scale_paint(demos)
        tested += count
        if prog is not None:
            return DimsReconstructionResult(
                attempted=True, solved=True, winning_program=prog,
                candidates_tested=tested, mode="scale_paint",
                inferred_output_dims_source=dims_source,
            )

    return DimsReconstructionResult(
        attempted=True, solved=False, winning_program=None,
        candidates_tested=tested,
        mode="none",
        inferred_output_dims_source=dims_source,
    )


def _try_whole_grid_transforms(
    demos: tuple[DemoPair, ...],
    dims_source: str,
) -> tuple[Program | None, int]:
    """Try single whole-grid transforms that change dimensions."""
    candidates: list[tuple[str, Program]] = []

    # Transpose
    candidates.append(("transpose", Program(
        steps=(Bind("v0", Type.GRID, Call("transpose_grid", (Ref("input"),))),),
        output="v0",
    )))

    # Rotations (90/270 change dims on non-square grids)
    for deg in (90, 270):
        candidates.append((f"rot{deg}", Program(
            steps=(Bind("v0", Type.GRID, Call("rotate_grid", (Literal(deg, Type.INT), Ref("input")))),),
            output="v0",
        )))

    # Reflections + transpose combos
    from aria.types import Axis
    for axis_val, axis_name in [(Axis.HORIZONTAL, "h"), (Axis.VERTICAL, "v")]:
        candidates.append((f"reflect_{axis_name}", Program(
            steps=(Bind("v0", Type.GRID, Call("reflect_grid", (Literal(axis_val, Type.AXIS), Ref("input")))),),
            output="v0",
        )))

    # Tile: try small tile factors when output is an integer multiple of input
    for rf in range(1, 5):
        for cf in range(1, 5):
            if rf == 1 and cf == 1:
                continue
            if all(
                d.output.shape == (d.input.shape[0] * rf, d.input.shape[1] * cf)
                for d in demos
            ):
                candidates.append((f"tile_{rf}x{cf}", Program(
                    steps=(Bind("v0", Type.GRID, Call("tile_grid", (
                        Ref("input"),
                        Literal(rf, Type.INT),
                        Literal(cf, Type.INT),
                    ))),),
                    output="v0",
                )))

    # Upscale: uniform factor
    for factor in range(2, 5):
        if all(
            d.output.shape == (d.input.shape[0] * factor, d.input.shape[1] * factor)
            for d in demos
        ):
            candidates.append((f"upscale_{factor}", Program(
                steps=(Bind("v0", Type.GRID, Call("upscale_grid", (
                    Ref("input"),
                    Literal(factor, Type.INT),
                ))),),
                output="v0",
            )))

    tested = 0
    for name, prog in candidates:
        tested += 1
        result = verify(prog, demos)
        if result.passed:
            return prog, tested

    return None, tested


def _try_objects_on_blank(
    demos: tuple[DemoPair, ...],
    output_dims: tuple[int, int],
) -> tuple[Program | None, int]:
    """Extract objects from input, paint on a blank canvas of fixed output dims.

    Tries two sub-strategies:
    a) paint_objects at original positions on blank canvas
    b) find_objects → filter by rules → paint on blank canvas
    """
    tested = 0

    # (a) All objects, original positions, blank canvas
    prog_all = Program(
        steps=(
            Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
            Bind("v1", Type.GRID, Call("new_grid", (
                Call("dims_make", (
                    Literal(output_dims[0], Type.INT),
                    Literal(output_dims[1], Type.INT),
                )),
                Literal(0, Type.COLOR),
            ))),
            Bind("v2", Type.GRID, Call("paint_objects", (Ref("v0"), Ref("v1")))),
        ),
        output="v2",
    )
    tested += 1
    result = verify(prog_all, demos)
    if result.passed:
        return prog_all, tested

    # (b) Per-color removal: for each color present in input but not in output,
    # try excluding that color from the object set
    from aria.graph.extract import extract
    input_colors = set()
    output_colors = set()
    for d in demos:
        input_colors |= set(int(v) for v in d.input.ravel() if v != 0)
        output_colors |= set(int(v) for v in d.output.ravel() if v != 0)

    removable = input_colors - output_colors
    for rm_color in sorted(removable):
        prog_rm = Program(
            steps=(
                Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
                Bind("v1", Type.PREDICATE, Call("by_color", (Literal(rm_color, Type.COLOR),))),
                Bind("v2", Type.OBJECT_SET, Call("where", (Ref("v1"), Ref("v0")))),
                Bind("v3", Type.OBJECT_SET, Call("excluding", (Ref("v2"), Ref("v0")))),
                Bind("v4", Type.GRID, Call("new_grid", (
                    Call("dims_make", (
                        Literal(output_dims[0], Type.INT),
                        Literal(output_dims[1], Type.INT),
                    )),
                    Literal(0, Type.COLOR),
                ))),
                Bind("v5", Type.GRID, Call("paint_objects", (Ref("v3"), Ref("v4")))),
            ),
            output="v5",
        )
        tested += 1
        result = verify(prog_rm, demos)
        if result.passed:
            return prog_rm, tested

    return None, tested


def _try_scale_paint(
    demos: tuple[DemoPair, ...],
) -> tuple[Program | None, int]:
    """For scale-factor dims-change: try scale_dims + new_grid + paint_objects."""
    # Determine the common scale factor
    factors = set()
    for d in demos:
        ir, ic = d.input.shape
        or_, oc = d.output.shape
        if ir == 0 or ic == 0 or or_ % ir != 0 or oc % ic != 0:
            return None, 0
        factors.add((or_ // ir, oc // ic))
    if len(factors) != 1:
        return None, 0
    rf, cf = next(iter(factors))

    tested = 0
    candidates: list[Program] = []

    if rf == cf:
        # Uniform scale: upscale_grid
        candidates.append(Program(
            steps=(Bind("v0", Type.GRID, Call("upscale_grid", (
                Ref("input"), Literal(rf, Type.INT),
            ))),),
            output="v0",
        ))

        # Objects on scaled blank
        candidates.append(Program(
            steps=(
                Bind("v0", Type.DIMS, Call("dims_of", (Ref("input"),))),
                Bind("v1", Type.DIMS, Call("scale_dims", (Ref("v0"), Literal(rf, Type.INT)))),
                Bind("v2", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
                Bind("v3", Type.GRID, Call("new_grid", (Ref("v1"), Literal(0, Type.COLOR)))),
                Bind("v4", Type.GRID, Call("paint_objects", (Ref("v2"), Ref("v3")))),
            ),
            output="v4",
        ))

    # Tile
    candidates.append(Program(
        steps=(Bind("v0", Type.GRID, Call("tile_grid", (
            Ref("input"), Literal(rf, Type.INT), Literal(cf, Type.INT),
        ))),),
        output="v0",
    ))

    for prog in candidates:
        tested += 1
        result = verify(prog, demos)
        if result.passed:
            return prog, tested

    return None, tested


@dataclass
class _CorrespondenceResult:
    """Result of correspondence-driven move analysis."""

    rules: tuple[ObjectRule, ...]
    programs: list[tuple[Program, ObjectRule]]


def _extract_objects_with_bg(grid: Grid) -> list[dict]:
    """Extract CC objects excluding the detected background color.

    Delegates to aria.decomposition.extract_objects for CC extraction,
    then converts to legacy dict format for backward compatibility.
    """
    from aria.decomposition import extract_objects, detect_bg

    if grid.size == 0:
        return []
    bg_color = detect_bg(grid)
    raw_objs = extract_objects(grid, bg_color)
    return [o.to_dict() for o in raw_objs]


def _correspondence_move_analysis(
    demos: tuple[DemoPair, ...],
) -> _CorrespondenceResult | None:
    """Infer object movement rules from pixel-level correspondence.

    Extracts 4-connected objects (with proper background detection),
    matches by color+mask, finds consistent per-color or color+size deltas.
    Also detects anchor-alignment patterns for non-uniform deltas.
    Returns rules and programs, or None if no correspondences found.
    """
    from aria.structural_edit import (
        _find_consistent_color_deltas,
        _find_consistent_color_size_deltas,
        _match_objects,
    )
    from aria.runtime.ops import has_op

    if not demos or demos[0].input.shape != demos[0].output.shape:
        return None

    per_demo_matches = []
    per_demo_inp_objs = []
    for demo in demos:
        if demo.input.shape != demo.output.shape:
            return None
        inp_objs = _extract_objects_with_bg(demo.input)
        out_objs = _extract_objects_with_bg(demo.output)
        matches = _match_objects(inp_objs, out_objs)
        if matches is None:
            return None
        per_demo_matches.append(matches)
        per_demo_inp_objs.append(inp_objs)

    color_consistent, color_inconsistent = _find_consistent_color_deltas(per_demo_matches)
    cs_refined = {}
    if color_inconsistent:
        cs_refined = _find_consistent_color_size_deltas(
            per_demo_matches, only_colors=color_inconsistent,
        )

    has_movement = (
        any(d != (0, 0) for d in color_consistent.values())
        or any(d != (0, 0) for d in cs_refined.values())
    )

    rules: list[ObjectRule] = []
    programs: list[tuple[Program, ObjectRule]] = []

    # Color-level rules
    for color, (dr, dc) in sorted(color_consistent.items()):
        if dr == 0 and dc == 0:
            continue
        rule = ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color,)},
            expressible=True, grouping_key="color",
            expression_path="pipeline",
        )
        rules.append(rule)
        prog = build_pipeline(rule)
        if prog is not None:
            programs.append((prog, rule))

    # Color+size refined rules
    for (color, size), (dr, dc) in sorted(cs_refined.items()):
        if dr == 0 and dc == 0:
            continue
        rule = ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color, size)},
            expressible=True, grouping_key="color_size",
            expression_path="pipeline",
        )
        rules.append(rule)
        prog = build_pipeline(rule)
        if prog is not None:
            programs.append((prog, rule))

    # Anchor-alignment detection for inconsistent colors
    if color_inconsistent and (
        has_op("align_center_to_row_of") or has_op("align_center_to_col_of")
    ):
        anchor_results = _detect_anchor_alignment(
            demos, per_demo_matches, per_demo_inp_objs,
            color_consistent, color_inconsistent,
        )
        for prog, rule in anchor_results:
            rules.append(rule)
            programs.append((prog, rule))
            has_movement = True

    # Shared composition
    if len(rules) >= 2:
        result = build_shared_composition(rules, demos)
        if result is not None:
            prog, meta = result
            programs.append((prog, meta))

    if not rules and not has_movement:
        return None
    if not rules:
        return None
    return _CorrespondenceResult(rules=tuple(rules), programs=programs)


def _detect_anchor_alignment(
    demos: tuple[DemoPair, ...],
    per_demo_matches: list,
    per_demo_inp_objs: list[list[dict]],
    color_consistent: dict[int, tuple[int, int]],
    color_inconsistent: set[int],
) -> list[tuple[Program, ObjectRule]]:
    """Detect anchor-alignment patterns from correspondence data.

    For each inconsistent color, checks whether a stationary single-pixel
    object (anchor) in another color explains the per-object deltas via
    center-alignment on one axis (row or col).

    Returns (program, rule) pairs for detected patterns.
    """
    from aria.runtime.ops import has_op

    results: list[tuple[Program, ObjectRule]] = []

    # Find anchor candidates: colors where all objects are stationary and size=1
    anchor_colors: set[int] = set()
    for color, (dr, dc) in color_consistent.items():
        if dr == 0 and dc == 0:
            # Check if all objects of this color are size 1 across all demos
            all_size_1 = True
            for matches in per_demo_matches:
                for m in matches:
                    if m.input_color == color and m.input_size != 1:
                        all_size_1 = False
                        break
                if not all_size_1:
                    break
            if all_size_1:
                # Also need exactly 1 object of this color per demo
                counts = []
                for matches in per_demo_matches:
                    c = sum(1 for m in matches if m.input_color == color)
                    counts.append(c)
                if all(c == 1 for c in counts):
                    anchor_colors.add(color)

    if not anchor_colors:
        return results

    for anchor_color in sorted(anchor_colors):
        for moved_color in sorted(color_inconsistent):
            # Get anchor position and moved object data per demo
            for axis, op_name in [
                ("col", "align_center_to_col_of"),
                ("row", "align_center_to_row_of"),
            ]:
                if not has_op(op_name):
                    continue
                if _check_anchor_alignment_axis(
                    per_demo_matches, per_demo_inp_objs,
                    anchor_color, moved_color, axis,
                    demos=demos,
                ):
                    rule = ObjectRule(
                        kind="move",
                        input_color=moved_color,
                        output_color=None,
                        offsets=None,
                        details={
                            "anchor_color": anchor_color,
                            "anchor_axis": axis,
                            "correspondence_source": True,
                            "anchor_alignment": True,
                            "group_key": (moved_color,),
                        },
                        expressible=True,
                        grouping_key="color",
                        expression_path="correspondence_anchor_pipeline",
                    )
                    prog = _anchor_alignment_pipeline(
                        moved_color, anchor_color, axis,
                    )
                    if prog is not None:
                        results.append((prog, rule))

    return results


def _check_anchor_alignment_axis(
    per_demo_matches: list,
    per_demo_inp_objs: list[list[dict]],
    anchor_color: int,
    moved_color: int,
    axis: str,
    demos: tuple[DemoPair, ...] | None = None,
) -> bool:
    """Check if all moved_color objects align their center to the anchor on axis.

    Uses input-side geometry: for each input object of moved_color,
    computes the expected displacement to align its center to the anchor.
    If a strict 1:1 match exists, validates against it. Otherwise (e.g.
    output-side merge), validates that the expected output pixel has
    moved_color in the output grid.

    axis="col": center col aligns to anchor, dr=0.
    axis="row": center row aligns to anchor, dc=0.
    """
    for di, matches in enumerate(per_demo_matches):
        # Find anchor from matches (anchor must be stationary and matched)
        anchor_matches = [m for m in matches if m.input_color == anchor_color]
        if len(anchor_matches) != 1:
            return False
        anchor = anchor_matches[0]
        anchor_center_row = anchor.input_row
        anchor_center_col = anchor.input_col

        # Gather input objects of moved_color from inp_objs (not from matches)
        moved_objs = [
            o for o in per_demo_inp_objs[di]
            if o["color"] == moved_color
        ]
        if not moved_objs:
            return False

        for obj in moved_objs:
            shape = obj["mask_shape"]
            obj_center_row = obj["row"] + shape[0] // 2
            obj_center_col = obj["col"] + shape[1] // 2

            if axis == "col":
                expected_dc = anchor_center_col - obj_center_col
                expected_dr = 0
            elif axis == "row":
                expected_dr = anchor_center_row - obj_center_row
                expected_dc = 0
            else:
                return False

            # Check: is this displacement consistent with the match if one exists?
            matched = [
                m for m in matches
                if m.input_color == moved_color
                and m.input_row == obj["row"]
                and m.input_col == obj["col"]
            ]
            if matched:
                m = matched[0]
                if m.dr != expected_dr or m.dc != expected_dc:
                    return False
            elif demos is not None:
                # No strict match (output-side merge). Verify expected
                # landing position has moved_color in output grid.
                out_grid = demos[di].output
                landing_row = obj["row"] + expected_dr
                landing_col = obj["col"] + expected_dc
                if not (0 <= landing_row < out_grid.shape[0]
                        and 0 <= landing_col < out_grid.shape[1]):
                    return False
                if int(out_grid[landing_row, landing_col]) != moved_color:
                    return False
            # If no match and no demos, accept geometric plausibility

    return True


def _anchor_alignment_pipeline(
    moved_color: int,
    anchor_color: int,
    axis: str,
) -> Program | None:
    """Build a pipeline for anchor-aligned movement.

    Pipeline:
    1. find_objects(input) → all
    2. select anchor: singleton(where(by_color(anchor_color),
                               where(by_size(1), all)))
    3. select moved objects: where(by_color(moved_color), all)
    4. build transform: align_center_to_{axis}_of(anchor)
    5. map_obj(transform, moved_set) → moved
    6. erase moved_color from input
    7. paint_objects(moved, erased)
    """
    from aria.runtime.ops import has_op

    op_name = f"align_center_to_{axis}_of"
    if not has_op("paint_objects") or not has_op(op_name):
        return None

    steps = [
        Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
        Bind("v1", Type.PREDICATE, Call("by_size", (Literal(1, Type.INT),))),
        Bind("v2", Type.OBJECT_SET, Call("where", (Ref("v1"), Ref("v0")))),
        Bind("v3", Type.PREDICATE, Call("by_color", (
            Literal(anchor_color, Type.COLOR),))),
        Bind("v4", Type.OBJECT_SET, Call("where", (Ref("v3"), Ref("v2")))),
        Bind("v5", Type.OBJECT, Call("singleton", (Ref("v4"),))),
        Bind("v6", Type.PREDICATE, Call("by_color", (
            Literal(moved_color, Type.COLOR),))),
        Bind("v7", Type.OBJECT_SET, Call("where", (Ref("v6"), Ref("v0")))),
        Bind("v8", Type.OBJ_TRANSFORM, Call(op_name, (Ref("v5"),))),
        Bind("v9", Type.OBJECT_SET, Call("map_obj", (Ref("v8"), Ref("v7")))),
        Bind("v10", Type.GRID, Call("apply_color_map", (
            Literal({moved_color: 0}, Type.COLOR_MAP), Ref("input"),
        ))),
        Bind("v11", Type.GRID, Call("paint_objects", (Ref("v9"), Ref("v10")))),
    ]
    return Program(steps=tuple(steps), output="v11")


# ---------------------------------------------------------------------------
# Composite role-normalized correspondence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompositeMotif:
    """A multi-color composite: singleton center + adjacent frame CCs."""

    center: dict          # the singleton CC (center role)
    frames: tuple[dict, ...]  # adjacent CCs of the frame color
    center_row: int       # row of the center pixel
    center_col: int       # col of the center pixel


@dataclass(frozen=True)
class _DemoRoles:
    """Role assignment for one demo grid."""

    bg_color: int
    center_color: int
    frame_color: int
    composites: tuple[_CompositeMotif, ...]
    anchor: dict | None   # isolated singleton of center_color


def _assemble_composites(
    objects: list[dict], grid: Grid,
) -> tuple[list[_CompositeMotif], list[dict]]:
    """Group singleton CCs with adjacent different-color CCs into composites.

    A composite is formed when a size-1 CC of color A is 4-adjacent to
    at least one CC of a different non-background color B.

    Returns (composites, isolated) where isolated are singletons with
    no adjacent different-color foreground neighbor.
    """
    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])
    rows, cols = grid.shape

    singletons = [o for o in objects if o["size"] == 1]
    non_singletons = [o for o in objects if o["size"] > 1]

    # Build a spatial lookup: (row, col) → object for non-singletons
    # by painting their masks onto a map
    cell_to_obj: dict[tuple[int, int], dict] = {}
    for obj in non_singletons:
        mask = obj["mask"]
        for dr in range(mask.shape[0]):
            for dc in range(mask.shape[1]):
                if mask[dr, dc]:
                    cell_to_obj[(obj["row"] + dr, obj["col"] + dc)] = obj

    composites: list[_CompositeMotif] = []
    isolated: list[dict] = []

    for s in singletons:
        sr, sc = s["row"], s["col"]
        adj_frames: dict[int, dict] = {}  # id(obj) → obj
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_color = int(grid[nr, nc])
            if cell_color == bg or cell_color == s["color"]:
                continue
            # Check non-singleton neighbors
            neighbor = cell_to_obj.get((nr, nc))
            if neighbor is not None and neighbor["color"] != s["color"]:
                adj_frames[id(neighbor)] = neighbor
            # Also check other singletons of different color
            for other in singletons:
                if other is s or other["color"] == s["color"]:
                    continue
                if other["row"] == nr and other["col"] == nc:
                    adj_frames[id(other)] = other

        if adj_frames:
            composites.append(_CompositeMotif(
                center=s,
                frames=tuple(adj_frames.values()),
                center_row=sr,
                center_col=sc,
            ))
        else:
            isolated.append(s)

    return composites, isolated


def _identify_demo_roles(
    objects: list[dict], grid: Grid,
) -> _DemoRoles | None:
    """Identify frame/center/anchor color roles for one demo.

    Returns None if no clear center-frame pattern is found.
    Requires:
    - At least one composite (singleton enclosed/adjacent to different-color CCs)
    - A dominant (center_color, frame_color) pair
    - At least one isolated singleton of center_color (the anchor)
    """
    composites, isolated = _assemble_composites(objects, grid)
    if not composites:
        return None

    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])

    # Count (center_color, frame_color) pairs across composites
    pair_counts: Counter = Counter()
    for comp in composites:
        frame_colors = {f["color"] for f in comp.frames}
        for fc in frame_colors:
            if fc != comp.center["color"]:
                pair_counts[(comp.center["color"], fc)] += 1

    if not pair_counts:
        return None

    (center_color, frame_color), _ = pair_counts.most_common(1)[0]

    # Filter composites to only those matching the dominant pair
    valid_composites = [
        c for c in composites
        if c.center["color"] == center_color
        and any(f["color"] == frame_color for f in c.frames)
    ]
    if not valid_composites:
        return None

    # Anchor: isolated singleton of center_color
    anchors = [s for s in isolated if s["color"] == center_color]
    anchor = anchors[0] if len(anchors) == 1 else None

    return _DemoRoles(
        bg_color=bg,
        center_color=center_color,
        frame_color=frame_color,
        composites=tuple(valid_composites),
        anchor=anchor,
    )


def _composite_structural_signature(comp: _CompositeMotif) -> tuple:
    """Color-invariant structural signature for a composite motif.

    Includes: component count, sorted frame sizes, center relative position
    within the composite's overall bounding box.
    """
    all_rows = [comp.center_row]
    all_cols = [comp.center_col]
    for f in comp.frames:
        mask = f["mask"]
        for dr in range(mask.shape[0]):
            for dc in range(mask.shape[1]):
                if mask[dr, dc]:
                    all_rows.append(f["row"] + dr)
                    all_cols.append(f["col"] + dc)

    min_r, max_r = min(all_rows), max(all_rows)
    min_c, max_c = min(all_cols), max(all_cols)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1

    center_rel_r = comp.center_row - min_r
    center_rel_c = comp.center_col - min_c

    frame_sizes = tuple(sorted(f["size"] for f in comp.frames))
    n_components = 1 + len(comp.frames)

    return (n_components, frame_sizes, bbox_h, bbox_w, center_rel_r, center_rel_c)


def _composite_correspondence_analysis(
    demos: tuple[DemoPair, ...],
) -> _CorrespondenceResult | None:
    """Role-normalized composite correspondence for same-dims movement tasks.

    1. Per demo: extract CCs, assemble composites, identify color roles
    2. Across demos: verify structural role pattern is consistent
    3. Match composites in input to output using role-normalized signatures
    4. Infer movement from composite displacement
    5. Check anchor alignment on center pixels
    6. Attempt program synthesis

    Returns _CorrespondenceResult with rules and programs, or None.
    """
    from aria.runtime.ops import has_op

    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    # Phase 1: per-demo role identification (input side only)
    demo_roles: list[_DemoRoles] = []
    for demo in demos:
        inp_objs = _extract_objects_with_bg(demo.input)
        inp_roles = _identify_demo_roles(inp_objs, demo.input)
        if inp_roles is None or inp_roles.anchor is None:
            return None
        demo_roles.append(inp_roles)

    # Phase 2: verify structural consistency across demos
    n_composites = len(demo_roles[0].composites)
    for dr in demo_roles:
        if len(dr.composites) != n_composites:
            return None

    # Phase 3: match input center pixels to output center pixels per demo
    # Find center-color singletons in output, match to input by proximity
    per_demo_deltas: list[list[tuple[int, int]]] = []
    for di, (demo, inp_r) in enumerate(zip(demos, demo_roles)):
        out_objs = _extract_objects_with_bg(demo.output)
        # Gather output singletons of the input's center color
        out_centers = [
            o for o in out_objs
            if o["color"] == inp_r.center_color and o["size"] == 1
        ]
        # Input center pixels from composites
        inp_centers = [(c.center_row, c.center_col) for c in inp_r.composites]

        # Match by proximity (greedy nearest)
        used: set[int] = set()
        deltas: list[tuple[int, int]] = []
        for ir, ic in inp_centers:
            best_idx, best_dist = -1, float("inf")
            for oi, oc in enumerate(out_centers):
                if oi in used:
                    continue
                dist = abs(oc["row"] - ir) + abs(oc["col"] - ic)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = oi
            if best_idx < 0:
                return None
            used.add(best_idx)
            out_c = out_centers[best_idx]
            deltas.append((out_c["row"] - ir, out_c["col"] - ic))

        per_demo_deltas.append(deltas)

    # Phase 4: check anchor alignment on center pixels
    rules: list[ObjectRule] = []
    programs: list[tuple[Program, ObjectRule]] = []
    diag: dict[str, Any] = {
        "source": "composite_correspondence",
        "n_demos": len(demos),
        "n_composites": n_composites,
        "role_map": [
            {
                "center_color": dr.center_color,
                "frame_color": dr.frame_color,
                "bg": dr.bg_color,
                "anchor_pos": (dr.anchor["row"], dr.anchor["col"]) if dr.anchor else None,
            }
            for dr in demo_roles
        ],
    }

    # Check anchor alignment per demo — the axis may differ across demos
    # (col-align when anchor is above/below, row-align when left/right).
    per_demo_axis: list[str | None] = []
    all_aligned = True
    for di, inp_r in enumerate(demo_roles):
        anchor = inp_r.anchor
        assert anchor is not None
        anchor_row, anchor_col = anchor["row"], anchor["col"]
        found_axis = None
        for axis in ("col", "row"):
            ok = True
            for delta, comp in zip(per_demo_deltas[di], inp_r.composites):
                dr, dc = delta
                if axis == "col":
                    if dc != anchor_col - comp.center_col:
                        ok = False
                        break
                else:
                    if dr != anchor_row - comp.center_row:
                        ok = False
                        break
            if ok:
                found_axis = axis
                break
        per_demo_axis.append(found_axis)
        if found_axis is None:
            all_aligned = False

    if all_aligned and all(a is not None for a in per_demo_axis):
        diag["per_demo_axis"] = per_demo_axis
        diag["alignment_verified"] = True

        rule = ObjectRule(
            kind="move",
            input_color=None,
            output_color=None,
            offsets=None,
            details={
                "composite_correspondence": True,
                "anchor_alignment": True,
                "per_demo_axis": per_demo_axis,
                "n_composites": n_composites,
                "diagnostics": diag,
            },
            expressible=True,
            grouping_key="composite_role",
            expression_path="composite_anchor_pipeline",
        )
        rules.append(rule)

    if not rules:
        # Even without anchor alignment, emit diagnostic rules
        # about composite movement
        rule = ObjectRule(
            kind="move",
            input_color=None,
            output_color=None,
            offsets=None,
            details={
                "composite_correspondence": True,
                "n_composites": n_composites,
                "per_demo_deltas": [list(d) for d in per_demo_deltas],
                "diagnostics": diag,
            },
            expressible=False,
            grouping_key="composite_role",
            expression_path="diagnosed",
        )
        rules.append(rule)

    if not rules:
        return None
    return _CorrespondenceResult(rules=tuple(rules), programs=programs)


def observe_and_synthesize(
    demos: tuple[DemoPair, ...],
) -> ObservationSynthesisResult:
    """Observe per-object changes, find common rules, synthesize programs."""
    all_observations: list[tuple[ObjectObservation, ...]] = []
    for demo in demos:
        obs = _observe_objects(demo)
        all_observations.append(obs)

    rules = _infer_rules(all_observations, demos)

    tested = 0
    for program, rule in _rules_to_programs(rules, demos):
        tested += 1
        result = verify(program, demos)
        if result.passed:
            return ObservationSynthesisResult(
                solved=True,
                winning_program=program,
                candidates_tested=tested,
                rules=tuple(rules),
                observations_per_demo=tuple(
                    tuple(obs) for obs in all_observations
                ),
            )

    # Fallback 1: strict correspondence-driven move synthesis
    corr = _correspondence_move_analysis(demos)
    if corr is not None:
        rules = list(rules) + list(corr.rules)
        for program, rule in corr.programs:
            tested += 1
            vr = verify(program, demos)
            if vr.passed:
                return ObservationSynthesisResult(
                    solved=True,
                    winning_program=program,
                    candidates_tested=tested,
                    rules=tuple(rules),
                    observations_per_demo=tuple(
                        tuple(obs) for obs in all_observations
                    ),
                )

    # Fallback 2: composite role-normalized correspondence
    comp_corr = _composite_correspondence_analysis(demos)
    if comp_corr is not None:
        rules = list(rules) + list(comp_corr.rules)
        for program, rule in comp_corr.programs:
            tested += 1
            vr = verify(program, demos)
            if vr.passed:
                return ObservationSynthesisResult(
                    solved=True,
                    winning_program=program,
                    candidates_tested=tested,
                    rules=tuple(rules),
                    observations_per_demo=tuple(
                        tuple(obs) for obs in all_observations
                    ),
                )

    return ObservationSynthesisResult(
        solved=False,
        winning_program=None,
        candidates_tested=tested,
        rules=tuple(rules),
        observations_per_demo=tuple(
            tuple(obs) for obs in all_observations
        ),
    )


# ---------------------------------------------------------------------------
# Phase 1: Per-object observation
# ---------------------------------------------------------------------------


def _observe_objects(demo: DemoPair) -> tuple[ObjectObservation, ...]:
    sg_in, sg_out, delta = extract_with_delta(demo.input, demo.output)
    inp, out = demo.input, demo.output

    observations: list[ObjectObservation] = []
    for obj in sg_in.objects:
        obj_col, obj_row = obj.bbox[0], obj.bbox[1]
        removed = obj.id in delta.removed

        color_changed_to = None
        moved_to = None
        for mod_id, field, old, new in delta.modified:
            if mod_id == obj.id:
                if field == "color":
                    color_changed_to = new
                elif field == "bbox":
                    moved_to = (new[1], new[0])

        added_offsets, added_color = _find_added_near(obj, sg_in.objects, inp, out)

        # Detect rigid shape transform by comparing masks
        shape_transform = None
        if not removed:
            for mod_id, field, old_mask, new_mask in delta.modified:
                if mod_id == obj.id and field == "mask":
                    shape_transform = _detect_rigid_transform(old_mask, new_mask)
                    break

        observations.append(ObjectObservation(
            obj_id=obj.id, color=obj.color,
            position=(obj_row, obj_col), size=obj.size,
            added_offsets=tuple(added_offsets), added_color=added_color,
            color_changed_to=color_changed_to, moved_to=moved_to,
            removed=removed, shape_transform=shape_transform,
        ))
    return tuple(observations)


def _detect_rigid_transform(old_mask, new_mask) -> str | None:
    """Detect if new_mask is a rigid transform of old_mask."""
    if not isinstance(old_mask, np.ndarray) or not isinstance(new_mask, np.ndarray):
        return None
    transforms = [
        ("rot90", lambda m: np.rot90(m, k=3)),   # 90° CW = 3x CCW
        ("rot180", lambda m: np.rot90(m, k=2)),
        ("rot270", lambda m: np.rot90(m, k=1)),   # 270° CW = 1x CCW
        ("flip_h", lambda m: np.flip(m, axis=0)),  # flip rows
        ("flip_v", lambda m: np.flip(m, axis=1)),  # flip cols
    ]
    for name, fn in transforms:
        try:
            transformed = fn(old_mask)
            if transformed.shape == new_mask.shape and np.array_equal(transformed, new_mask):
                return name
        except Exception:
            continue
    return None


def _find_added_near(
    obj: ObjectNode, all_input_objects: tuple[ObjectNode, ...],
    inp: Grid, out: Grid,
) -> tuple[list[tuple[int, int]], int | None]:
    if inp.shape != out.shape:
        return [], None
    obj_col, obj_row = obj.bbox[0], obj.bbox[1]
    rows, cols = inp.shape
    changed_pixels = [
        (r, c, int(out[r, c]))
        for r in range(rows) for c in range(cols)
        if int(inp[r, c]) == 0 and int(out[r, c]) != 0
    ]
    if not changed_pixels:
        return [], None
    offsets, colors_added = [], []
    for r, c, new_color in changed_pixels:
        closest_id, closest_dist = None, float("inf")
        for other in all_input_objects:
            oc, or_ = other.bbox[0], other.bbox[1]
            d = max(abs(r - or_), abs(c - oc))
            if d < closest_dist:
                closest_dist, closest_id = d, other.id
        if closest_id == obj.id:
            offsets.append((r - obj_row, c - obj_col))
            colors_added.append(new_color)
    if not colors_added:
        return [], None
    return offsets, Counter(colors_added).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Phase 2: Rule inference with generic grouping
# ---------------------------------------------------------------------------


def _group_by(
    all_observations: list[tuple[ObjectObservation, ...]],
    key_fn,
) -> dict[Any, list[ObjectObservation]]:
    groups: dict[Any, list[ObjectObservation]] = {}
    for demo_obs in all_observations:
        for obs in demo_obs:
            k = key_fn(obs)
            groups.setdefault(k, []).append(obs)
    return groups


def _color_key(obs: ObjectObservation) -> tuple:
    return (obs.color,)


def _color_size_key(obs: ObjectObservation) -> tuple:
    return (obs.color, obs.size)


def _make_size_rank_key(
    all_observations: list[tuple[ObjectObservation, ...]],
):
    """Build a pairwise grouping key: (color, is_largest_in_color_group).

    Semantics (matches by_size_rank execution):
    - Within each color group per demo, the object(s) with maximum size
      are marked True. All others are False.
    - If all objects in a color group have the same size, all are True
      (no useful split — the grouping degenerates).
    - Tie behavior: all objects at max size are True.
    """
    # Precompute per-demo per-color largest
    per_demo: dict[int, dict[int, bool]] = {}
    for di, demo_obs in enumerate(all_observations):
        by_color: dict[int, list[ObjectObservation]] = {}
        for o in demo_obs:
            by_color.setdefault(o.color, []).append(o)
        demo_map: dict[int, bool] = {}
        for color, obs_list in by_color.items():
            if len(obs_list) < 2:
                for o in obs_list:
                    demo_map[o.obj_id] = True
                continue
            max_size = max(o.size for o in obs_list)
            for o in obs_list:
                demo_map[o.obj_id] = (o.size == max_size)
        per_demo[di] = demo_map

    def key_fn(obs: ObjectObservation) -> tuple:
        for di, demo_obs in enumerate(all_observations):
            for o in demo_obs:
                if o is obs:
                    return (obs.color, per_demo.get(di, {}).get(obs.obj_id, True))
        return (obs.color, True)

    return key_fn


def _make_proximity_key(
    all_observations: list[tuple[ObjectObservation, ...]],
    demos: tuple[DemoPair, ...],
):
    """Build a relational grouping key: (color, is_nearest_to_marker).

    Semantics (matches pipeline execution exactly):
    - Marker = rarest non-background color across all demos.
    - Reference point = center of the first marker object (sorted by id).
    - For each color group, the ONE object whose center is closest to the
      reference point (Chebyshev distance, ties broken by Manhattan then
      obj_id ascending) is marked True. All others are False.
    - This is per-color-group nearest, not global nearest.
    - This matches what the pipeline executes:
        nth(0, where(by_color(marker), objects))  →  marker reference
        nearest_to(marker_ref, where(by_color(C), objects))  →  nearest

    Proximity mode: per_color_nearest_to_first_marker_object
    """
    from aria.graph.extract import extract

    color_counts: Counter = Counter()
    for demo in demos:
        for v in demo.input.ravel():
            v = int(v)
            if v != 0:
                color_counts[v] += 1

    if not color_counts:
        return None, None

    marker_color = color_counts.most_common()[-1][0]

    # For each demo, find the reference marker object and compute per-color nearest
    per_demo_near: dict[int, dict[int, bool]] = {}
    for di, demo in enumerate(demos):
        sg = extract(demo.input)
        marker_objs = sorted(
            [obj for obj in sg.objects if obj.color == marker_color],
            key=lambda o: o.id,
        )
        if not marker_objs:
            continue
        ref = marker_objs[0]
        ref_cx = ref.bbox[0] + ref.bbox[2] // 2
        ref_cy = ref.bbox[1] + ref.bbox[3] // 2

        obs_tuple = all_observations[di] if di < len(all_observations) else ()

        # Build a map from obj_id to the actual ObjectNode for center computation
        obj_by_id = {obj.id: obj for obj in sg.objects}

        # Group observations by color
        by_color: dict[int, list[ObjectObservation]] = {}
        for o in obs_tuple:
            if o.color == marker_color:
                continue
            by_color.setdefault(o.color, []).append(o)

        demo_near: dict[int, bool] = {}
        for color, obs_list in by_color.items():
            if not obs_list:
                continue
            # Find nearest using same metric as nearest_to runtime op:
            # center-to-center Chebyshev, ties by Manhattan, then by obj_id
            def _sort_key(o):
                real_obj = obj_by_id.get(o.obj_id)
                if real_obj is None:
                    return (float("inf"), float("inf"), o.obj_id)
                cx = real_obj.bbox[0] + real_obj.bbox[2] // 2
                cy = real_obj.bbox[1] + real_obj.bbox[3] // 2
                cheb = max(abs(cx - ref_cx), abs(cy - ref_cy))
                manh = abs(cx - ref_cx) + abs(cy - ref_cy)
                return (cheb, manh, o.obj_id)

            sorted_obs = sorted(obs_list, key=_sort_key)
            nearest_id = sorted_obs[0].obj_id
            for o in obs_list:
                demo_near[o.obj_id] = (o.obj_id == nearest_id)

        per_demo_near[di] = demo_near

    def key_fn(obs: ObjectObservation) -> tuple:
        for di, demo_obs in enumerate(all_observations):
            for o in demo_obs:
                if o is obs:
                    near = per_demo_near.get(di, {}).get(obs.obj_id, False)
                    return (obs.color, near)
        return (obs.color, False)

    return key_fn, marker_color


def _make_directional_keys(
    all_observations: list[tuple[ObjectObservation, ...]],
    demos: tuple[DemoPair, ...],
):
    """Build directional grouping keys: (color, direction_from_marker).

    Semantics (matches by_relative_pos execution exactly):
    - Marker = rarest non-background color (same as proximity).
    - Reference = center of first marker object (sorted by id).
    - For each non-marker object, direction is determined by comparing
      object center to reference center using strict inequality:
        UP:    center_y(obj) < center_y(ref)
        DOWN:  center_y(obj) > center_y(ref)
        LEFT:  center_x(obj) < center_x(ref)
        RIGHT: center_x(obj) > center_x(ref)
    - Objects at the same row/col as reference get direction=None.
    - Returns four key functions, one per direction.

    Directional mode: per_color_direction_from_first_marker_object
    """
    from aria.graph.extract import extract

    color_counts: Counter = Counter()
    for demo in demos:
        for v in demo.input.ravel():
            v = int(v)
            if v != 0:
                color_counts[v] += 1
    if not color_counts:
        return [], None

    marker_color = color_counts.most_common()[-1][0]

    # Precompute per-demo direction labels for each object
    per_demo_dir: dict[int, dict[int, dict[str, bool]]] = {}
    for di, demo in enumerate(demos):
        sg = extract(demo.input)
        marker_objs = sorted(
            [obj for obj in sg.objects if obj.color == marker_color],
            key=lambda o: o.id,
        )
        if not marker_objs:
            continue
        ref = marker_objs[0]
        ref_cx = ref.bbox[0] + ref.bbox[2] // 2
        ref_cy = ref.bbox[1] + ref.bbox[3] // 2

        obj_by_id = {obj.id: obj for obj in sg.objects}
        demo_dirs: dict[int, dict[str, bool]] = {}
        obs_tuple = all_observations[di] if di < len(all_observations) else ()
        for o in obs_tuple:
            if o.color == marker_color:
                continue
            real_obj = obj_by_id.get(o.obj_id)
            if real_obj is None:
                demo_dirs[o.obj_id] = {"above": False, "below": False, "left": False, "right": False}
                continue
            cx = real_obj.bbox[0] + real_obj.bbox[2] // 2
            cy = real_obj.bbox[1] + real_obj.bbox[3] // 2
            demo_dirs[o.obj_id] = {
                "above": cy < ref_cy,
                "below": cy > ref_cy,
                "left": cx < ref_cx,
                "right": cx > ref_cx,
            }
        per_demo_dir[di] = demo_dirs

    key_fns = []
    for direction in ("above", "below", "left", "right"):
        def _make_fn(d=direction):
            def key_fn(obs: ObjectObservation) -> tuple:
                for di, demo_obs in enumerate(all_observations):
                    for o in demo_obs:
                        if o is obs:
                            dirs = per_demo_dir.get(di, {}).get(obs.obj_id, {})
                            return (obs.color, dirs.get(d, False))
                return (obs.color, False)
            return key_fn
        key_fns.append((_make_fn(), direction))

    return key_fns, marker_color


def _infer_output_dims(demos: tuple[DemoPair, ...]) -> tuple[int, int] | None:
    """If all demos have the same output shape, return it. Otherwise None."""
    if not demos:
        return None
    shapes = set(d.output.shape for d in demos)
    if len(shapes) == 1:
        r, c = next(iter(shapes))
        return (int(r), int(c))
    return None


def _infer_dims_source(
    demos: tuple[DemoPair, ...],
) -> tuple[tuple[int, int] | None, str]:
    """Infer output dims and classify _how_ they were inferred.

    Returns (dims, source) where source is one of:
      "fixed"         – all outputs share the same constant shape
      "scale_input"   – output = input * constant factor per axis
      "transpose"     – output rows/cols are swapped input cols/rows
      "unknown"       – could not determine a consistent rule
    """
    if not demos:
        return None, "unknown"

    # Fixed: all outputs same shape
    out_shapes = [d.output.shape for d in demos]
    if len(set(out_shapes)) == 1:
        r, c = out_shapes[0]
        return (int(r), int(c)), "fixed"

    # Scale: output = input * (row_factor, col_factor), same factors across demos
    factors = set()
    for d in demos:
        ir, ic = d.input.shape
        or_, oc = d.output.shape
        if ir == 0 or ic == 0:
            break
        if or_ % ir != 0 or oc % ic != 0:
            break
        factors.add((or_ // ir, oc // ic))
    else:
        if len(factors) == 1:
            rf, cf = next(iter(factors))
            return None, f"scale_input"

    # Transpose: output shape = (input_cols, input_rows)
    if all(d.output.shape == (d.input.shape[1], d.input.shape[0]) for d in demos):
        return None, "transpose"

    return None, "unknown"


def _infer_rules(
    all_observations: list[tuple[ObjectObservation, ...]],
    demos: tuple[DemoPair, ...],
) -> list[ObjectRule]:
    rules: list[ObjectRule] = []
    # Layer 1: color-only grouping (simplest)
    rules.extend(_infer_surround_rules(all_observations, _color_key, "color"))
    rules.extend(_infer_recolor_rules(demos))
    rules.extend(_infer_move_rules(all_observations, _color_key, "color"))
    rules.extend(_infer_remove_rules(all_observations, _color_key, "color"))
    rules.extend(_infer_rigid_transform_rules(all_observations, _color_key, "color"))

    # Layer 2: (color, size) grouping
    rules.extend(_infer_surround_rules(all_observations, _color_size_key, "color_size"))
    rules.extend(_infer_move_rules(all_observations, _color_size_key, "color_size"))
    rules.extend(_infer_remove_rules(all_observations, _color_size_key, "color_size"))
    rules.extend(_infer_rigid_transform_rules(all_observations, _color_size_key, "color_size"))

    # Layer 2.5: size-rank pairwise grouping (largest vs others in color group)
    size_rank_key = _make_size_rank_key(all_observations)
    rules.extend(_infer_move_rules(all_observations, size_rank_key, "size_rank"))
    rules.extend(_infer_remove_rules(all_observations, size_rank_key, "size_rank"))
    rules.extend(_infer_rigid_transform_rules(all_observations, size_rank_key, "size_rank"))

    # Layer 3: proximity-to-marker relational grouping
    prox_key_fn, marker_color = _make_proximity_key(all_observations, demos)
    if prox_key_fn is not None:
        prox_rules = []
        prox_rules.extend(_infer_move_rules(all_observations, prox_key_fn, "proximity"))
        prox_rules.extend(_infer_remove_rules(all_observations, prox_key_fn, "proximity"))
        prox_rules.extend(_infer_surround_rules(all_observations, prox_key_fn, "proximity"))
        for r in prox_rules:
            r.details["marker_color"] = marker_color
            r.details["proximity_mode"] = "per_color_nearest_to_first_marker_object"
        rules.extend(prox_rules)

    # Layer 4: directional relative-to-marker grouping
    dir_key_fns, dir_marker = _make_directional_keys(all_observations, demos)
    if dir_key_fns:
        for key_fn, direction in dir_key_fns:
            gkey_name = f"direction_{direction}"
            dir_rules = []
            dir_rules.extend(_infer_move_rules(all_observations, key_fn, gkey_name))
            dir_rules.extend(_infer_remove_rules(all_observations, key_fn, gkey_name))
            dir_rules.extend(_infer_surround_rules(all_observations, key_fn, gkey_name))
            for r in dir_rules:
                r.details["marker_color"] = dir_marker
                r.details["direction"] = direction
                r.details["direction_mode"] = "per_color_direction_from_first_marker_object"
            rules.extend(dir_rules)

    return rules


def _infer_surround_rules(all_observations, key_fn, key_name) -> list[ObjectRule]:
    if not all_observations:
        return []
    rules = []
    for gkey, obs_list in _group_by(all_observations, key_fn).items():
        offset_sets, output_colors = [], []
        for obs in obs_list:
            if obs.added_offsets and obs.added_color is not None:
                offset_sets.append(frozenset(obs.added_offsets))
                output_colors.append(obs.added_color)
        if not offset_sets or len(set(offset_sets)) != 1 or len(set(output_colors)) != 1:
            continue
        rules.append(ObjectRule(
            kind="surround", input_color=gkey[0],
            output_color=output_colors[0],
            offsets=tuple(sorted(offset_sets[0])),
            details={"count": len(obs_list), "group_key": gkey},
            grouping_key=key_name,
        ))
    return rules


def _infer_recolor_rules(demos) -> list[ObjectRule]:
    if not demos:
        return []
    inp, out = demos[0].input, demos[0].output
    if inp.shape != out.shape:
        return []
    color_map: dict[int, int] = {}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            iv, ov = int(inp[r, c]), int(out[r, c])
            if iv in color_map:
                if color_map[iv] != ov:
                    return []
            else:
                color_map[iv] = ov
    nontrivial = {k: v for k, v in color_map.items() if k != v}
    if not nontrivial:
        return []
    for demo in demos[1:]:
        i2, o2 = demo.input, demo.output
        if i2.shape != o2.shape:
            return []
        for r in range(i2.shape[0]):
            for c in range(i2.shape[1]):
                if color_map.get(int(i2[r, c])) != int(o2[r, c]):
                    return []
    return [ObjectRule(
        kind="recolor", input_color=None, output_color=None, offsets=None,
        details={"color_map": nontrivial, "full_map": color_map},
        grouping_key="pixel",
    )]


def _infer_move_rules(all_observations, key_fn, key_name) -> list[ObjectRule]:
    if not all_observations:
        return []
    rules = []
    for gkey, obs_list in _group_by(all_observations, key_fn).items():
        deltas = []
        for obs in obs_list:
            if obs.moved_to is not None and not obs.removed:
                deltas.append((obs.moved_to[0] - obs.position[0],
                               obs.moved_to[1] - obs.position[1]))
        if not deltas or len(deltas) != len(obs_list) or len(set(deltas)) != 1:
            continue
        dr, dc = deltas[0]
        all_same = all(
            obs.moved_to is None or
            (obs.moved_to[0] - obs.position[0], obs.moved_to[1] - obs.position[1]) == (dr, dc)
            for demo_obs in all_observations for obs in demo_obs
        )
        axis_aligned = (dr == 0) != (dc == 0)
        rules.append(ObjectRule(
            kind="move", input_color=gkey[0], output_color=None,
            offsets=((dr, dc),),
            details={"dr": dr, "dc": dc, "count": len(deltas),
                     "global_shift": all_same, "axis_aligned": axis_aligned,
                     "group_key": gkey},
            expressible=True,  # all consistent-delta moves via translate_delta
            grouping_key=key_name,
        ))
    return rules


def _infer_remove_rules(all_observations, key_fn, key_name) -> list[ObjectRule]:
    if not all_observations:
        return []
    rules = []
    for gkey, obs_list in _group_by(all_observations, key_fn).items():
        removed = sum(1 for o in obs_list if o.removed)
        if removed > 0 and removed == len(obs_list):
            rules.append(ObjectRule(
                kind="remove", input_color=gkey[0], output_color=None,
                offsets=None,
                details={"count": removed, "group_key": gkey},
                expressible=True, grouping_key=key_name,
            ))
    return rules


def _infer_rigid_transform_rules(all_observations, key_fn, key_name) -> list[ObjectRule]:
    """If all objects in a group had the same rigid shape transform, emit a rule."""
    if not all_observations:
        return []
    rules = []
    for gkey, obs_list in _group_by(all_observations, key_fn).items():
        transforms = []
        for obs in obs_list:
            if obs.shape_transform is not None:
                transforms.append(obs.shape_transform)
        if not transforms or len(transforms) != len(obs_list):
            continue
        if len(set(transforms)) == 1:
            rules.append(ObjectRule(
                kind="rigid_transform", input_color=gkey[0], output_color=None,
                offsets=None,
                details={"transform": transforms[0], "count": len(obs_list), "group_key": gkey},
                expressible=True, grouping_key=key_name,
            ))
    return rules


# ---------------------------------------------------------------------------
# Phase 3: Rule → Program synthesis
# ---------------------------------------------------------------------------


def _rules_to_programs(rules, demos):
    programs: list[tuple[Program, ObjectRule]] = []

    # Flat-grid programs
    surround_progs = []
    for r in rules:
        if r.kind == "surround":
            p = _surround_rule_to_program(r)
            if p:
                programs.append((p, r))
                surround_progs.append(p)
    if len(surround_progs) >= 2:
        c = _compose_programs(surround_progs)
        if c:
            programs.append((c, ObjectRule(
                kind="surround_composed", input_color=None, output_color=None,
                offsets=None, details={"rule_count": len(surround_progs)},
            )))

    for r in rules:
        if r.kind == "recolor":
            p = _recolor_rule_to_program(r)
            if p:
                programs.append((p, r))

    for r in rules:
        if r.kind == "remove" and r.expressible:
            p = _remove_rule_to_program(r)
            if p:
                programs.append((p, r))

    for r in rules:
        if r.kind == "move" and r.expressible and r.details.get("global_shift"):
            p = _move_flat_program(r)
            if p:
                programs.append((p, r))

    # Object-pipeline programs
    programs.extend(_object_pipeline_programs(rules))

    # Object-pipeline programs with inferred output dims (for dims-change tasks)
    out_dims = _infer_output_dims(demos)
    if out_dims is not None and demos and demos[0].input.shape != demos[0].output.shape:
        programs.extend(_object_pipeline_programs(rules, output_dims=out_dims))

    # Shared object-composition: compose compatible rules over the original object set
    composed = _shared_object_composition(rules, demos)
    if composed is not None:
        prog, meta = composed
        programs.append((prog, meta))

    return programs


# ---------------------------------------------------------------------------
# Shared pipeline builder
# ---------------------------------------------------------------------------


def _build_predicate(rule: ObjectRule) -> list[Bind]:
    """Build predicate step(s) from a rule's grouping key."""
    gkey = rule.details.get("group_key", (rule.input_color,))

    if len(gkey) >= 2:
        # (color, size) → combine two predicates with AND via nested where
        # For now: use by_color only, rely on by_size after where
        # Actually we can chain: where(by_color(C)) then where(by_size(S))
        color, size = gkey[0], gkey[1]
        return [
            Bind("pred_color", Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))),
            Bind("pred_size", Type.PREDICATE, Call("by_size", (Literal(size, Type.INT),))),
        ]
    else:
        color = gkey[0] if gkey else rule.input_color
        return [
            Bind("pred", Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))),
        ]


def _build_selection(rule: ObjectRule, step_idx: int) -> tuple[list[Bind], str, int]:
    """Build find + filter steps. Returns (steps, selected_set_name, next_step_idx)."""
    from aria.runtime.ops import has_op

    steps: list[Bind] = []
    all_name = f"v{step_idx}"; step_idx += 1
    steps.append(Bind(all_name, Type.OBJECT_SET, Call("find_objects", (Ref("input"),))))

    gkey = rule.details.get("group_key", (rule.input_color,))
    color = gkey[0] if gkey else rule.input_color

    if rule.grouping_key == "proximity" and has_op("nearest_to"):
        # Proximity selection: find objects of this color, then select
        # the one nearest to the marker (rarest color)
        marker_color = rule.details.get("marker_color")
        near_marker = gkey[1] if len(gkey) >= 2 else True

        pred_c = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred_c, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
        color_set = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(color_set, Type.OBJECT_SET, Call("where", (Ref(pred_c), Ref(all_name)))))

        if near_marker and marker_color is not None:
            # Select the one nearest to the marker
            pred_m = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(pred_m, Type.PREDICATE, Call("by_color", (Literal(marker_color, Type.COLOR),))))
            marker_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(marker_set, Type.OBJECT_SET, Call("where", (Ref(pred_m), Ref(all_name)))))
            # Get the first marker object as reference
            marker_obj = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(marker_obj, Type.OBJECT, Call("nth", (Literal(0, Type.INT), Ref(marker_set)))))
            # Select nearest from color set
            selected = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(selected, Type.OBJECT, Call("nearest_to", (Ref(marker_obj), Ref(color_set)))))
            # Wrap in singleton set for pipeline compatibility
            selected_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(selected_set, Type.OBJECT_SET, Call("singleton_set", (Ref(selected),))))
            return steps, selected_set, step_idx
        else:
            # "Not nearest" = excluding the nearest one
            if marker_color is not None:
                pred_m = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(pred_m, Type.PREDICATE, Call("by_color", (Literal(marker_color, Type.COLOR),))))
                marker_set = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(marker_set, Type.OBJECT_SET, Call("where", (Ref(pred_m), Ref(all_name)))))
                marker_obj = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(marker_obj, Type.OBJECT, Call("nth", (Literal(0, Type.INT), Ref(marker_set)))))
                nearest = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(nearest, Type.OBJECT, Call("nearest_to", (Ref(marker_obj), Ref(color_set)))))
                nearest_set = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(nearest_set, Type.OBJECT_SET, Call("singleton_set", (Ref(nearest),))))
                selected = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(selected, Type.OBJECT_SET, Call("excluding", (Ref(nearest_set), Ref(color_set)))))
                return steps, selected, step_idx

        return steps, color_set, step_idx

    if rule.grouping_key.startswith("direction_") and has_op("by_relative_pos"):
        # Directional selection: by_color(C) then by_relative_pos(DIR, marker_obj)
        marker_color = rule.details.get("marker_color")
        direction_name = rule.details.get("direction")
        in_direction = gkey[1] if len(gkey) >= 2 else True

        dir_map = {"above": Dir.UP, "below": Dir.DOWN, "left": Dir.LEFT, "right": Dir.RIGHT}
        dsl_dir = dir_map.get(direction_name)
        if dsl_dir is not None and marker_color is not None:
            pred_c = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(pred_c, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
            color_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(color_set, Type.OBJECT_SET, Call("where", (Ref(pred_c), Ref(all_name)))))

            # Get marker reference
            pred_m = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(pred_m, Type.PREDICATE, Call("by_color", (Literal(marker_color, Type.COLOR),))))
            marker_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(marker_set, Type.OBJECT_SET, Call("where", (Ref(pred_m), Ref(all_name)))))
            marker_obj = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(marker_obj, Type.OBJECT, Call("nth", (Literal(0, Type.INT), Ref(marker_set)))))

            # Filter by direction
            dir_pred = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(dir_pred, Type.PREDICATE, Call("by_relative_pos", (
                Literal(dsl_dir, Type.DIR), Ref(marker_obj),
            ))))

            if in_direction:
                selected = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(selected, Type.OBJECT_SET, Call("where", (Ref(dir_pred), Ref(color_set)))))
            else:
                in_dir = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(in_dir, Type.OBJECT_SET, Call("where", (Ref(dir_pred), Ref(color_set)))))
                selected = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(selected, Type.OBJECT_SET, Call("excluding", (Ref(in_dir), Ref(color_set)))))

            return steps, selected, step_idx

    if rule.grouping_key == "size_rank" and len(gkey) >= 2 and has_op("by_size_rank"):
        is_largest = gkey[1]
        pred_c = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred_c, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
        color_set = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(color_set, Type.OBJECT_SET, Call("where", (Ref(pred_c), Ref(all_name)))))
        if is_largest:
            # Select the largest: by_size_rank(0, color_set) → singleton
            largest = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(largest, Type.OBJECT, Call("by_size_rank", (Literal(0, Type.INT), Ref(color_set)))))
            selected = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(selected, Type.OBJECT_SET, Call("singleton_set", (Ref(largest),))))
        else:
            # Select all except the largest
            largest = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(largest, Type.OBJECT, Call("by_size_rank", (Literal(0, Type.INT), Ref(color_set)))))
            largest_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(largest_set, Type.OBJECT_SET, Call("singleton_set", (Ref(largest),))))
            selected = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(selected, Type.OBJECT_SET, Call("excluding", (Ref(largest_set), Ref(color_set)))))
        return steps, selected, step_idx

    if rule.grouping_key == "color_size" and len(gkey) >= 2 and has_op("by_size"):
        size = gkey[1]
        pred_c = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred_c, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
        filtered1 = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(filtered1, Type.OBJECT_SET, Call("where", (Ref(pred_c), Ref(all_name)))))
        pred_s = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred_s, Type.PREDICATE, Call("by_size", (Literal(size, Type.INT),))))
        selected = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(selected, Type.OBJECT_SET, Call("where", (Ref(pred_s), Ref(filtered1)))))
    else:
        pred = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
        selected = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(selected, Type.OBJECT_SET, Call("where", (Ref(pred), Ref(all_name)))))

    return steps, selected, step_idx


def _single_pipeline(rule: ObjectRule, output_dims: tuple[int, int] | None = None) -> Program | None:
    """Build one pipeline program for a single rule, or None if not expressible.

    output_dims: if provided, use a blank canvas of this size instead of input dims.
    """
    from aria.runtime.ops import has_op
    if not has_op("paint_objects"):
        return None

    if rule.kind == "move" and not rule.details.get("global_shift"):
        dr, dc = rule.details.get("dr", 0), rule.details.get("dc", 0)
        if dr == 0 and dc == 0:
            return None
        steps, selected, idx = _build_selection(rule, 0)
        transform = f"v{idx}"; idx += 1
        steps.append(Bind(transform, Type.OBJ_TRANSFORM,
                          Call("translate_delta", (Literal(dr, Type.INT), Literal(dc, Type.INT)))))
        moved = f"v{idx}"; idx += 1
        steps.append(Bind(moved, Type.OBJECT_SET, Call("map_obj", (Ref(transform), Ref(selected)))))
        color = rule.input_color
        erased = f"v{idx}"; idx += 1
        steps.append(Bind(erased, Type.GRID,
                          Call("apply_color_map", (Literal({color: 0}, Type.COLOR_MAP), Ref("input")))))
        result = f"v{idx}"; idx += 1
        steps.append(Bind(result, Type.GRID, Call("paint_objects", (Ref(moved), Ref(erased)))))
        return Program(steps=tuple(steps), output=result)

    if rule.kind == "remove":
        steps, selected, idx = _build_selection(rule, 0)
        all_name = steps[0].name
        remaining = f"v{idx}"; idx += 1
        steps.append(Bind(remaining, Type.OBJECT_SET, Call("excluding", (Ref(selected), Ref(all_name)))))
        blank = f"v{idx}"; idx += 1
        if output_dims is not None:
            steps.append(Bind(blank, Type.GRID, Call("new_grid", (
                Call("dims_make", (Literal(output_dims[0], Type.INT), Literal(output_dims[1], Type.INT))),
                Literal(0, Type.COLOR),
            ))))
        else:
            steps.append(Bind(blank, Type.GRID,
                              Call("new_grid", (Call("dims_of", (Ref("input"),)), Literal(0, Type.COLOR)))))
        result = f"v{idx}"; idx += 1
        steps.append(Bind(result, Type.GRID, Call("paint_objects", (Ref(remaining), Ref(blank)))))
        return Program(steps=tuple(steps), output=result)

    if rule.kind == "recolor" and rule.details.get("color_map"):
        cmap = rule.details["color_map"]
        # Per-color pipelines
        progs = []
        for old_c, new_c in cmap.items():
            if old_c == new_c:
                continue
            steps = [
                Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
                Bind("v1", Type.PREDICATE, Call("by_color", (Literal(old_c, Type.COLOR),))),
                Bind("v2", Type.OBJECT_SET, Call("where", (Ref("v1"), Ref("v0")))),
                Bind("v3", Type.OBJ_TRANSFORM, Call("recolor", (Literal(new_c, Type.COLOR),))),
                Bind("v4", Type.OBJECT_SET, Call("map_obj", (Ref("v3"), Ref("v2")))),
                Bind("v5", Type.GRID, Call("paint_objects", (Ref("v4"), Ref("input")))),
            ]
            progs.append(Program(steps=tuple(steps), output="v5"))
        if len(progs) >= 2:
            return _compose_programs(progs)
        elif progs:
            return progs[0]

    if rule.kind == "rigid_transform":
        transform_name = rule.details.get("transform")
        transform_expr = _rigid_transform_expr(transform_name)
        if transform_expr is None:
            return None
        steps, selected, idx = _build_selection(rule, 0)
        tf = f"v{idx}"; idx += 1
        steps.append(Bind(tf, Type.OBJ_TRANSFORM, transform_expr))
        transformed = f"v{idx}"; idx += 1
        steps.append(Bind(transformed, Type.OBJECT_SET, Call("map_obj", (Ref(tf), Ref(selected)))))
        result = f"v{idx}"; idx += 1
        steps.append(Bind(result, Type.GRID, Call("paint_objects", (Ref(transformed), Ref("input")))))
        return Program(steps=tuple(steps), output=result)

    return None


def _rigid_transform_expr(transform_name: str) -> Expr | None:
    """Build a partial-application expression for a rigid transform."""
    from aria.types import Axis
    if transform_name == "rot90":
        return Call("rotate", (Literal(90, Type.INT),))
    if transform_name == "rot180":
        return Call("rotate", (Literal(180, Type.INT),))
    if transform_name == "rot270":
        return Call("rotate", (Literal(270, Type.INT),))
    if transform_name == "flip_h":
        return Call("reflect", (Literal(Axis.HORIZONTAL, Type.AXIS),))
    if transform_name == "flip_v":
        return Call("reflect", (Literal(Axis.VERTICAL, Type.AXIS),))
    return None


# ---------------------------------------------------------------------------
# Shared object-composition: multi-rule programs over one object set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubgroupAction:
    """One action in a shared composition plan."""

    rule: ObjectRule
    kind: str  # "remove", "recolor", "move"
    # For building the selection from the shared all-objects set
    selection_builder: Any  # callable(all_name, marker_obj_name, step_idx) -> (steps, selected_name, step_idx)
    # Transform details
    transform_details: dict[str, Any]


def _shared_object_composition(
    rules: list[ObjectRule],
    demos: tuple[DemoPair, ...] = (),
) -> tuple[Program, ObjectRule] | None:
    """Compose multiple object-level rules into one shared-object program.

    Semantics:
    1. find_objects(input) once → all_objects
    2. For each rule, select its subgroup from all_objects
    3. Validate subgroups are disjoint across all demos (reject if overlap)
    4. Apply each transform to its subgroup
    5. Compute untouched = all_objects - union(subgroups)
    6. Paint: blank grid ← untouched ← transformed groups

    Safety: subgroup overlap is checked by actually running the selection
    logic on each demo's objects, not by heuristic key comparison.
    """
    from aria.runtime.ops import has_op
    if not has_op("paint_objects") or not has_op("excluding"):
        return None

    # Collect composable rules
    composable = []
    for r in rules:
        if not r.expressible:
            continue
        if r.kind == "remove" and r.input_color is not None:
            composable.append(r)
        elif r.kind == "move" and r.input_color is not None:
            dr, dc = r.details.get("dr", 0), r.details.get("dc", 0)
            if r.details.get("global_shift"):
                continue
            if dr != 0 or dc != 0:
                composable.append(r)
        elif r.kind == "recolor" and r.details.get("color_map"):
            if r.input_color is not None:
                composable.append(r)
        elif r.kind == "rigid_transform" and r.input_color is not None:
            composable.append(r)

    if len(composable) < 2:
        return None

    # Deduplicate: prefer simplest grouping key for same (kind, input_color)
    key_priority = {"color": 0, "color_size": 1, "proximity": 2,
                    "direction_above": 3, "direction_below": 3,
                    "direction_left": 3, "direction_right": 3}
    best: dict[tuple, ObjectRule] = {}
    for r in composable:
        k = (r.kind, r.input_color)
        prev = best.get(k)
        if prev is None or key_priority.get(r.grouping_key, 9) < key_priority.get(prev.grouping_key, 9):
            best[k] = r
    composable = list(best.values())

    if len(composable) < 2:
        return None

    # Same color + same grouping key = always conflict
    color_counts = Counter(r.input_color for r in composable)
    for color, count in color_counts.items():
        if count > 1:
            same_rules = [r for r in composable if r.input_color == color]
            gkeys = {r.grouping_key for r in same_rules}
            if len(gkeys) == 1:
                return None

    # Demo-validated disjointness check
    if demos and not _validate_disjoint_selections(composable, demos):
        return None

    return _build_shared_program(composable)


def _validate_disjoint_selections(
    rules: list[ObjectRule],
    demos: tuple[DemoPair, ...],
) -> bool:
    """Check that rules select non-overlapping object IDs on every demo.

    Runs the actual selection logic (same as pipeline execution) on each
    demo to get concrete object IDs, then checks pairwise overlap.
    """
    from aria.graph.extract import extract
    from aria.runtime.ops import get_op

    try:
        _, find_fn = get_op("find_objects")
        _, by_color_fn = get_op("by_color")
        _, where_fn = get_op("where")
        _, by_size_fn = get_op("by_size")
        _, nth_fn = get_op("nth")
        _, nearest_fn = get_op("nearest_to")
        _, by_rel_fn = get_op("by_relative_pos")
    except KeyError:
        return True  # can't validate, allow optimistically

    dir_map = {"above": Dir.UP, "below": Dir.DOWN, "left": Dir.LEFT, "right": Dir.RIGHT}

    for demo in demos:
        all_objects = find_fn(demo.input)
        if not all_objects:
            continue

        # Compute selected IDs for each rule
        rule_ids: list[frozenset[int]] = []
        for rule in rules:
            try:
                ids = _select_ids_for_rule(
                    rule, all_objects, demo.input,
                    by_color_fn, where_fn, by_size_fn, nth_fn,
                    nearest_fn, by_rel_fn, dir_map,
                )
            except Exception:
                ids = frozenset()
            rule_ids.append(ids)

        # Pairwise overlap check
        for i in range(len(rule_ids)):
            for j in range(i + 1, len(rule_ids)):
                overlap = rule_ids[i] & rule_ids[j]
                if overlap:
                    return False

    return True


def _select_ids_for_rule(
    rule, all_objects, grid,
    by_color_fn, where_fn, by_size_fn, nth_fn,
    nearest_fn, by_rel_fn, dir_map,
) -> frozenset[int]:
    """Compute the set of object IDs a rule selects, using runtime ops."""
    gkey = rule.details.get("group_key", (rule.input_color,))
    color = gkey[0] if gkey else rule.input_color

    color_set = where_fn(by_color_fn(color), all_objects)

    if rule.grouping_key == "color":
        return frozenset(o.id for o in color_set)

    if rule.grouping_key == "color_size" and len(gkey) >= 2:
        size = gkey[1]
        filtered = where_fn(by_size_fn(size), color_set)
        return frozenset(o.id for o in filtered)

    if rule.grouping_key == "size_rank" and len(gkey) >= 2:
        is_largest = gkey[1]
        from aria.runtime.ops import get_op as _get_op
        try:
            _, size_rank_fn = _get_op("by_size_rank")
            largest_obj = size_rank_fn(0, color_set)
            if is_largest:
                return frozenset({largest_obj.id})
            else:
                return frozenset(o.id for o in color_set if o.id != largest_obj.id)
        except Exception:
            return frozenset(o.id for o in color_set)

    marker_color = rule.details.get("marker_color")
    if marker_color is None:
        return frozenset(o.id for o in color_set)

    marker_set = where_fn(by_color_fn(marker_color), all_objects)
    if not marker_set:
        return frozenset(o.id for o in color_set)
    marker_obj = nth_fn(0, marker_set)

    if rule.grouping_key == "proximity":
        near = gkey[1] if len(gkey) >= 2 else True
        nearest = nearest_fn(marker_obj, color_set)
        if near:
            return frozenset({nearest.id})
        else:
            return frozenset(o.id for o in color_set if o.id != nearest.id)

    if rule.grouping_key.startswith("direction_"):
        direction_name = rule.details.get("direction")
        in_direction = gkey[1] if len(gkey) >= 2 else True
        dsl_dir = dir_map.get(direction_name)
        if dsl_dir is not None:
            dir_pred = by_rel_fn(dsl_dir, marker_obj)
            in_dir = where_fn(dir_pred, color_set)
            if in_direction:
                return frozenset(o.id for o in in_dir)
            else:
                return frozenset(o.id for o in color_set) - frozenset(o.id for o in in_dir)

    return frozenset(o.id for o in color_set)


def _build_shared_program(
    rules: list[ObjectRule],
) -> tuple[Program, ObjectRule] | None:
    """Build the actual shared-object composition program."""
    steps: list[Bind] = []
    idx = 0

    # Step 1: find_objects(input)
    all_name = f"v{idx}"; idx += 1
    steps.append(Bind(all_name, Type.OBJECT_SET, Call("find_objects", (Ref("input"),))))

    # Step 2: build marker reference if any rule needs it
    marker_colors = {r.details.get("marker_color") for r in rules} - {None}
    marker_refs: dict[int, str] = {}
    for mc in marker_colors:
        pred = f"v{idx}"; idx += 1
        steps.append(Bind(pred, Type.PREDICATE, Call("by_color", (Literal(mc, Type.COLOR),))))
        mset = f"v{idx}"; idx += 1
        steps.append(Bind(mset, Type.OBJECT_SET, Call("where", (Ref(pred), Ref(all_name)))))
        mobj = f"v{idx}"; idx += 1
        steps.append(Bind(mobj, Type.OBJECT, Call("nth", (Literal(0, Type.INT), Ref(mset)))))
        marker_refs[mc] = mobj

    # Step 3: for each rule, build selection and transform
    selected_names: list[str] = []
    transformed_names: list[str] = []
    erased_colors: set[int] = set()

    for rule in rules:
        # Build selection from the shared all-objects set
        sel_steps, sel_name, idx = _build_shared_selection(
            rule, all_name, marker_refs, idx,
        )
        steps.extend(sel_steps)
        selected_names.append(sel_name)

        if rule.kind == "remove":
            # Nothing to transform — this subgroup is excluded
            erased_colors.add(rule.input_color)
        elif rule.kind == "move":
            dr = rule.details.get("dr", 0)
            dc = rule.details.get("dc", 0)
            if dr == 0 and dc == 0:
                return None
            tf = f"v{idx}"; idx += 1
            steps.append(Bind(tf, Type.OBJ_TRANSFORM,
                              Call("translate_delta", (Literal(dr, Type.INT), Literal(dc, Type.INT)))))
            moved = f"v{idx}"; idx += 1
            steps.append(Bind(moved, Type.OBJECT_SET, Call("map_obj", (Ref(tf), Ref(sel_name)))))
            transformed_names.append(moved)
            erased_colors.add(rule.input_color)
        elif rule.kind == "recolor":
            cmap = rule.details.get("color_map", {})
            new_color = cmap.get(rule.input_color)
            if new_color is None:
                return None
            tf = f"v{idx}"; idx += 1
            steps.append(Bind(tf, Type.OBJ_TRANSFORM, Call("recolor", (Literal(new_color, Type.COLOR),))))
            recolored = f"v{idx}"; idx += 1
            steps.append(Bind(recolored, Type.OBJECT_SET, Call("map_obj", (Ref(tf), Ref(sel_name)))))
            transformed_names.append(recolored)
        elif rule.kind == "rigid_transform":
            transform_expr = _rigid_transform_expr(rule.details.get("transform", ""))
            if transform_expr is None:
                return None
            tf = f"v{idx}"; idx += 1
            steps.append(Bind(tf, Type.OBJ_TRANSFORM, transform_expr))
            xformed = f"v{idx}"; idx += 1
            steps.append(Bind(xformed, Type.OBJECT_SET, Call("map_obj", (Ref(tf), Ref(sel_name)))))
            transformed_names.append(xformed)

    # Step 4: compute untouched = all - union(selected)
    excluded_union = selected_names[0]
    for sn in selected_names[1:]:
        # Merge via successive excluding — but we need union.
        # Simpler: exclude each selected set from remaining
        pass
    # Build iterative exclusion: remaining = all - sel1 - sel2 - ...
    remaining = all_name
    for sn in selected_names:
        new_remaining = f"v{idx}"; idx += 1
        steps.append(Bind(new_remaining, Type.OBJECT_SET, Call("excluding", (Ref(sn), Ref(remaining)))))
        remaining = new_remaining

    # Step 5: base grid — erase colors that were moved/removed
    if erased_colors:
        erase_map = {c: 0 for c in erased_colors}
        base = f"v{idx}"; idx += 1
        steps.append(Bind(base, Type.GRID,
                          Call("apply_color_map", (Literal(erase_map, Type.COLOR_MAP), Ref("input")))))
    else:
        base = "input"

    # Step 6: paint untouched objects onto base
    grid_with_untouched = f"v{idx}"; idx += 1
    steps.append(Bind(grid_with_untouched, Type.GRID, Call("paint_objects", (Ref(remaining), Ref(base)))))

    # Step 7: paint each transformed subgroup
    current_grid = grid_with_untouched
    for tn in transformed_names:
        next_grid = f"v{idx}"; idx += 1
        steps.append(Bind(next_grid, Type.GRID, Call("paint_objects", (Ref(tn), Ref(current_grid)))))
        current_grid = next_grid

    composed_rule = ObjectRule(
        kind="shared_composition", input_color=None, output_color=None,
        offsets=None,
        details={
            "rule_count": len(rules),
            "rule_kinds": [r.kind for r in rules],
            "rule_colors": [r.input_color for r in rules],
            "disjointness_validated": True,
            "rule_groupings": [r.grouping_key for r in rules],
        },
        expressible=True,
        expression_path="shared_object_composition",
    )
    return Program(steps=tuple(steps), output=current_grid), composed_rule


def _build_shared_selection(
    rule: ObjectRule,
    all_name: str,
    marker_refs: dict[int, str],
    step_idx: int,
) -> tuple[list[Bind], str, int]:
    """Build selection steps from a shared all-objects set.

    Unlike _build_selection which starts with find_objects, this
    assumes all_name already holds the full object set.
    """
    steps: list[Bind] = []
    gkey = rule.details.get("group_key", (rule.input_color,))
    color = gkey[0] if gkey else rule.input_color

    # Color filter
    pred_c = f"v{step_idx}"; step_idx += 1
    steps.append(Bind(pred_c, Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))))
    color_set = f"v{step_idx}"; step_idx += 1
    steps.append(Bind(color_set, Type.OBJECT_SET, Call("where", (Ref(pred_c), Ref(all_name)))))

    # Additional filters based on grouping key
    if rule.grouping_key == "size_rank" and len(gkey) >= 2:
        is_largest = gkey[1]
        largest = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(largest, Type.OBJECT, Call("by_size_rank", (Literal(0, Type.INT), Ref(color_set)))))
        if is_largest:
            selected = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(selected, Type.OBJECT_SET, Call("singleton_set", (Ref(largest),))))
            return steps, selected, step_idx
        else:
            largest_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(largest_set, Type.OBJECT_SET, Call("singleton_set", (Ref(largest),))))
            rest = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(rest, Type.OBJECT_SET, Call("excluding", (Ref(largest_set), Ref(color_set)))))
            return steps, rest, step_idx

    if rule.grouping_key == "color_size" and len(gkey) >= 2:
        size = gkey[1]
        pred_s = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(pred_s, Type.PREDICATE, Call("by_size", (Literal(size, Type.INT),))))
        filtered = f"v{step_idx}"; step_idx += 1
        steps.append(Bind(filtered, Type.OBJECT_SET, Call("where", (Ref(pred_s), Ref(color_set)))))
        return steps, filtered, step_idx

    if rule.grouping_key == "proximity":
        marker_color = rule.details.get("marker_color")
        near = gkey[1] if len(gkey) >= 2 else True
        marker_obj = marker_refs.get(marker_color)
        if marker_obj:
            nearest = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(nearest, Type.OBJECT, Call("nearest_to", (Ref(marker_obj), Ref(color_set)))))
            nearest_set = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(nearest_set, Type.OBJECT_SET, Call("singleton_set", (Ref(nearest),))))
            if near:
                return steps, nearest_set, step_idx
            else:
                rest = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(rest, Type.OBJECT_SET, Call("excluding", (Ref(nearest_set), Ref(color_set)))))
                return steps, rest, step_idx

    if rule.grouping_key.startswith("direction_"):
        direction_name = rule.details.get("direction")
        marker_color = rule.details.get("marker_color")
        in_direction = gkey[1] if len(gkey) >= 2 else True
        dir_map = {"above": Dir.UP, "below": Dir.DOWN, "left": Dir.LEFT, "right": Dir.RIGHT}
        dsl_dir = dir_map.get(direction_name)
        marker_obj = marker_refs.get(marker_color)
        if dsl_dir is not None and marker_obj:
            dir_pred = f"v{step_idx}"; step_idx += 1
            steps.append(Bind(dir_pred, Type.PREDICATE,
                              Call("by_relative_pos", (Literal(dsl_dir, Type.DIR), Ref(marker_obj)))))
            if in_direction:
                selected = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(selected, Type.OBJECT_SET, Call("where", (Ref(dir_pred), Ref(color_set)))))
                return steps, selected, step_idx
            else:
                in_dir = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(in_dir, Type.OBJECT_SET, Call("where", (Ref(dir_pred), Ref(color_set)))))
                rest = f"v{step_idx}"; step_idx += 1
                steps.append(Bind(rest, Type.OBJECT_SET, Call("excluding", (Ref(in_dir), Ref(color_set)))))
                return steps, rest, step_idx

    # Default: just the color set
    return steps, color_set, step_idx


# ---------------------------------------------------------------------------
# Flat program builders (unchanged)
# ---------------------------------------------------------------------------


def _surround_rule_to_program(rule):
    if rule.input_color is None or rule.output_color is None or rule.offsets is None:
        return None
    from aria.runtime.ops import has_op
    offset_set = frozenset(rule.offsets)
    for radius in range(1, 4):
        cardinal = frozenset((dr, dc) for dr in range(-radius, radius + 1)
                             for dc in range(-radius, radius + 1)
                             if (dr == 0) != (dc == 0) and abs(dr) + abs(dc) <= radius)
        if offset_set == cardinal and has_op("fill_cardinal"):
            return _make_fill_program("fill_cardinal", rule.input_color, radius, rule.output_color)
        diagonal = frozenset((dr, dc) for dr in range(-radius, radius + 1)
                             for dc in range(-radius, radius + 1)
                             if abs(dr) == abs(dc) and dr != 0 and abs(dr) <= radius)
        if offset_set == diagonal and has_op("fill_diagonal"):
            return _make_fill_program("fill_diagonal", rule.input_color, radius, rule.output_color)
        chebyshev = frozenset((dr, dc) for dr in range(-radius, radius + 1)
                              for dc in range(-radius, radius + 1)
                              if (dr, dc) != (0, 0) and max(abs(dr), abs(dc)) <= radius)
        if offset_set == chebyshev and has_op("fill_around"):
            return _make_fill_program("fill_around", rule.input_color, radius, rule.output_color)
    return None


def _make_fill_program(fill_op, input_color, radius, output_color):
    return Program(steps=(
        Bind("v0", Type.PREDICATE, Call("by_color", (Literal(input_color, Type.COLOR),))),
        Bind("v1", Type.GRID, Call(fill_op, (
            Ref("v0"), Literal(radius, Type.INT),
            Literal(output_color, Type.COLOR), Ref("input")))),
    ), output="v1")


def _recolor_rule_to_program(rule):
    from aria.runtime.ops import has_op
    if not has_op("apply_color_map"):
        return None
    cmap = rule.details.get("color_map")
    if not cmap:
        return None
    return Program(steps=(
        Bind("v0", Type.GRID, Call("apply_color_map", (Literal(cmap, Type.COLOR_MAP), Ref("input")))),
    ), output="v0")


def _remove_rule_to_program(rule):
    from aria.runtime.ops import has_op
    if rule.input_color is None or not has_op("apply_color_map"):
        return None
    return Program(steps=(
        Bind("v0", Type.GRID, Call("apply_color_map", (
            Literal({rule.input_color: 0}, Type.COLOR_MAP), Ref("input")))),
    ), output="v0")


def _move_flat_program(rule):
    from aria.runtime.ops import has_op
    if not has_op("shift_grid") or not rule.details.get("global_shift"):
        return None
    dr, dc = rule.details.get("dr", 0), rule.details.get("dc", 0)
    return Program(steps=(
        Bind("v0", Type.GRID, Call("shift_grid", (
            Literal(dr, Type.INT), Literal(dc, Type.INT),
            Literal(0, Type.COLOR), Ref("input")))),
    ), output="v0")


# ---------------------------------------------------------------------------
# Object-pipeline programs (legacy compat — delegates to _single_pipeline)
# ---------------------------------------------------------------------------


def _object_pipeline_programs(rules, output_dims=None):
    programs = []
    for rule in rules:
        if rule.kind in ("recolor", "remove", "move", "rigid_transform") and rule.expressible:
            prog = _single_pipeline(rule, output_dims=output_dims)
            if prog is not None:
                programs.append((prog, ObjectRule(
                    kind=rule.kind, input_color=rule.input_color,
                    output_color=rule.output_color, offsets=rule.offsets,
                    details=rule.details, expressible=True,
                    grouping_key=rule.grouping_key, expression_path="pipeline",
                )))
    return programs


# ---------------------------------------------------------------------------
# Composition and helpers
# ---------------------------------------------------------------------------


def _compose_programs(programs):
    if not programs:
        return None
    all_steps, prev_grid, step_idx = [], "input", 0
    for prog in programs:
        rename = {"input": prev_grid}
        for step in prog.steps:
            if isinstance(step, Bind):
                rename[step.name] = f"v{step_idx}"
                step_idx += 1
        for step in prog.steps:
            if isinstance(step, Bind):
                all_steps.append(Bind(
                    name=rename[step.name], typ=step.typ,
                    expr=_rewrite_refs(step.expr, rename), declared=step.declared))
        prev_grid = all_steps[-1].name
    return Program(steps=tuple(all_steps), output=prev_grid)


def _rewrite_refs(expr, rename):
    if isinstance(expr, Ref):
        return Ref(rename.get(expr.name, expr.name))
    if isinstance(expr, Call):
        return Call(op=expr.op, args=tuple(_rewrite_refs(a, rename) for a in expr.args))
    return expr


def _delta_to_dir(dr, dc):
    if dc == 0 and dr > 0: return Dir.DOWN, dr
    if dc == 0 and dr < 0: return Dir.UP, -dr
    if dr == 0 and dc > 0: return Dir.RIGHT, dc
    if dr == 0 and dc < 0: return Dir.LEFT, -dc
    return None, 0


# ---------------------------------------------------------------------------
# Public API: rebuild programs from (possibly mutated) rules
# ---------------------------------------------------------------------------


def rules_to_programs(
    rules: list[ObjectRule],
    demos: tuple[DemoPair, ...],
) -> list[tuple[Program, ObjectRule]]:
    """Public wrapper around _rules_to_programs for structural edit use."""
    return _rules_to_programs(rules, demos)


def build_pipeline(rule: ObjectRule) -> Program | None:
    """Public wrapper: build a single object-pipeline program from a rule."""
    return _single_pipeline(rule)


def build_shared_composition(
    rules: list[ObjectRule],
    demos: tuple[DemoPair, ...],
) -> tuple[Program, ObjectRule] | None:
    """Public wrapper: build a shared-object composition from multiple rules."""
    return _shared_object_composition(rules, demos)
