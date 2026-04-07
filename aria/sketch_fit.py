"""Sketch fitting and specialization.

Two responsibilities:

1. **Fitting** — propose Sketch hypotheses from demo evidence.
   Each fitter inspects decomposition structure and pixel diffs,
   then emits a Sketch or returns None.

   Fitters today:
     - fit_framed_periodic_repair  (regularity repair)
     - fit_composite_role_alignment (spatial relation alignment)
     - fit_canvas_construction     (tile / upscale / crop)

2. **Specialization** — extract resolved static task structure.
   specialize_sketch(graph, demos) walks graph node evidence and
   demo decomposition to produce a Specialization bundle of
   ResolvedBindings the compiler reads directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.decomposition import (
    CompositeDecomposition,
    FramedRegion,
    decompose_composites,
    decompose_grid,
    detect_bg,
    detect_framed_regions,
)
from aria.sketch import (
    PrimitiveFamily,
    ResolvedBinding,
    RoleKind,
    RoleVar,
    Sketch,
    SketchGraph,
    SketchStep,
    Slot,
    SlotType,
    Specialization,
    make_composite_role_alignment,
    make_identify_roles,
    make_region_periodic_repair,
)
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Map: decomposition view name → sketch fitter function name
_VIEW_TO_FITTER: list[tuple[str, str]] = [
    ("framed_regions", "fit_framed_periodic_repair"),
    ("composites", "fit_composite_role_alignment"),
]


@dataclass(frozen=True)
class FitResult:
    """Result of fit_sketches with decomposition ranking metadata."""

    sketches: list
    decomp_views_tried: tuple[str, ...]
    decomp_ranking_applied: bool = False
    decomp_ranking_changed_order: bool = False
    decomp_ranking_policy: str = "none"


def fit_sketches(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
    *,
    decomp_ranker=None,
    task_signatures: tuple[str, ...] = (),
) -> list[Sketch]:
    """Try sketch fitters against a task's demos, in decomposition-ranked order.

    Each fitter depends on a specific decomposition view. When a
    decomp_ranker is provided, the views (and thus fitters) are reordered
    so the most promising run first. Without a ranker, the default order
    is [framed_regions, composites].

    Returns a list of fitted sketches (may be empty, may have multiple).
    """
    result = fit_sketches_with_report(
        demos, task_id,
        decomp_ranker=decomp_ranker,
        task_signatures=task_signatures,
    )
    return result.sketches


def fit_sketches_with_report(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
    *,
    decomp_ranker=None,
    task_signatures: tuple[str, ...] = (),
) -> FitResult:
    """Like fit_sketches but also returns decomposition ranking metadata."""
    from aria.sketch_rank import rank_decompositions

    # Determine available views and their natural order
    view_names = [view for view, _ in _VIEW_TO_FITTER]

    # Apply decomposition ranking
    same_dims = all(d.input.shape == d.output.shape for d in demos) if demos else True
    bg_colors = [detect_bg(d.input) for d in demos] if demos else []
    bg_rotates = len(set(bg_colors)) > 1

    ranked_views, decomp_report = rank_decompositions(
        view_names,
        task_signatures,
        {
            "same_dims": same_dims,
            "bg_rotates": bg_rotates,
            "n_demos": len(demos),
        },
        ranker=decomp_ranker,
    )

    # Build a lookup from view name to fitter
    fitter_lookup = dict(_VIEW_TO_FITTER)
    fitters_by_name = {
        "fit_framed_periodic_repair": fit_framed_periodic_repair,
        "fit_composite_role_alignment": fit_composite_role_alignment,
    }

    # Try fitters in ranked view order
    results: list[Sketch] = []
    for view in ranked_views:
        fitter_name = fitter_lookup.get(view)
        if fitter_name is None:
            continue
        fitter_fn = fitters_by_name.get(fitter_name)
        if fitter_fn is None:
            continue
        s = fitter_fn(demos, task_id)
        if s is not None:
            results.append(s)

    # View-independent fitters (dims-change detection, movement, etc.)
    canvas = fit_canvas_construction(demos, task_id)
    if canvas is not None:
        results.append(canvas)

    movement = fit_object_movement(demos, task_id)
    if movement is not None:
        results.append(movement)

    grid_xform = fit_grid_transform(demos, task_id)
    if grid_xform is not None:
        results.append(grid_xform)

    return FitResult(
        sketches=results,
        decomp_views_tried=tuple(ranked_views),
        decomp_ranking_applied=decomp_report.policy_name != "none",
        decomp_ranking_changed_order=decomp_report.order_changed,
        decomp_ranking_policy=decomp_report.policy_name,
    )


# ---------------------------------------------------------------------------
# Fitter: periodic regularity repair
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PeriodicEvidence:
    """Evidence that a row/column has a periodic pattern with violations."""

    axis: str           # "row" or "col"
    period: int         # detected period length
    n_violations: int   # cells that break the pattern
    total_cells: int
    pattern: tuple[int, ...]   # the inferred repeating unit
    violation_positions: tuple[int, ...]  # indices of violating cells


def _detect_row_period(row: np.ndarray) -> _PeriodicEvidence | None:
    """Detect the best periodic pattern in a 1D array.

    Tries periods 2..len//2, picks the one with fewest violations
    (ties broken by smallest period). Returns None if best has 0 violations
    (already perfect, nothing to repair) or if all periods have too many.
    """
    n = len(row)
    if n < 3:
        return None

    best: _PeriodicEvidence | None = None

    for period in range(2, n // 2 + 1):
        # Infer pattern by majority vote at each phase position
        pattern = []
        for phase in range(period):
            vals = [int(row[i]) for i in range(phase, n, period)]
            # Majority vote
            from collections import Counter
            counts = Counter(vals)
            pattern.append(counts.most_common(1)[0][0])

        # Count violations
        violations = []
        for i in range(n):
            expected = pattern[i % period]
            if int(row[i]) != expected:
                violations.append(i)

        if not violations:
            continue  # perfect already — nothing to repair
        if len(violations) > n // 3:
            continue  # too many violations — not a real period

        evidence = _PeriodicEvidence(
            axis="row",
            period=period,
            n_violations=len(violations),
            total_cells=n,
            pattern=tuple(pattern),
            violation_positions=tuple(violations),
        )
        if best is None or len(violations) < best.n_violations:
            best = evidence
        elif len(violations) == best.n_violations and period < best.period:
            best = evidence

    return best


def _detect_col_period(col: np.ndarray) -> _PeriodicEvidence | None:
    """Same as _detect_row_period but for a column vector."""
    result = _detect_row_period(col)
    if result is not None:
        return _PeriodicEvidence(
            axis="col",
            period=result.period,
            n_violations=result.n_violations,
            total_cells=result.total_cells,
            pattern=result.pattern,
            violation_positions=result.violation_positions,
        )
    return None


def _check_periodic_repair_for_region(
    region: FramedRegion,
    input_grid: Grid,
    output_grid: Grid,
) -> dict | None:
    """Check if the framed region has periodic content with repairable violations.

    Returns evidence dict or None.
    """
    interior_in = region.interior
    r, c, h, w = region.row, region.col, region.height, region.width
    if r + h > output_grid.shape[0] or c + w > output_grid.shape[1]:
        return None
    interior_out = output_grid[r:r + h, c:c + w]
    if interior_in.shape != interior_out.shape:
        return None

    return _check_periodic_content(interior_in, interior_out, region.frame_color, r, c, h, w)


def _check_periodic_content(
    interior_in: Grid,
    interior_out: Grid,
    frame_color: int,
    abs_row: int,
    abs_col: int,
    h: int,
    w: int,
) -> dict | None:
    """Check rows and columns of a content grid for repairable periodic violations.

    For each row/col, detects the best period, filters out violations at
    frame-border positions (first/last elements where value matches an
    adjacent uniform row/col), then checks the output fixes remaining violations.
    """
    row_evidence: list[_PeriodicEvidence] = []
    col_evidence: list[_PeriodicEvidence] = []

    rows_h, cols_w = interior_in.shape

    for ri in range(rows_h):
        ev = _detect_row_period(interior_in[ri])
        if ev is None:
            continue
        # Filter: skip violations at border positions (col 0 or col w-1)
        # where the value matches a uniform border value
        content_violations = []
        for vi in ev.violation_positions:
            if vi == 0 or vi == cols_w - 1:
                continue  # likely sub-frame border
            content_violations.append(vi)
        if not content_violations:
            continue
        # Verify output fixes the content violations
        fixed = True
        for vi in content_violations:
            expected = ev.pattern[vi % ev.period]
            if int(interior_out[ri, vi]) != expected:
                fixed = False
                break
        if fixed:
            row_evidence.append(_PeriodicEvidence(
                axis="row",
                period=ev.period,
                n_violations=len(content_violations),
                total_cells=ev.total_cells,
                pattern=ev.pattern,
                violation_positions=tuple(content_violations),
            ))

    for ci in range(cols_w):
        ev = _detect_col_period(interior_in[:, ci])
        if ev is None:
            continue
        content_violations = []
        for vi in ev.violation_positions:
            if vi == 0 or vi == rows_h - 1:
                continue
            content_violations.append(vi)
        if not content_violations:
            continue
        fixed = True
        for vi in content_violations:
            expected = ev.pattern[vi % ev.period]
            if int(interior_out[vi, ci]) != expected:
                fixed = False
                break
        if fixed:
            col_evidence.append(_PeriodicEvidence(
                axis="col",
                period=ev.period,
                n_violations=len(content_violations),
                total_cells=ev.total_cells,
                pattern=ev.pattern,
                violation_positions=tuple(content_violations),
            ))

    if not row_evidence and not col_evidence:
        return None

    if len(row_evidence) >= len(col_evidence):
        axis = "row"
        evidence_list = row_evidence
    else:
        axis = "col"
        evidence_list = col_evidence

    periods = [e.period for e in evidence_list]
    violations = sum(e.n_violations for e in evidence_list)

    from collections import Counter
    dominant_period = Counter(periods).most_common(1)[0][0]

    return {
        "axis": axis,
        "period": dominant_period,
        "n_periodic_lines": len(evidence_list),
        "total_violations": violations,
        "frame_color": frame_color,
        "region_pos": (abs_row, abs_col),
        "region_size": (h, w),
    }


def _scan_raw_rows_cols(
    input_grid: Grid,
    output_grid: Grid,
    bg: int,
) -> list[dict]:
    """Directly scan all grid rows/cols for periodic repair evidence.

    This handles cases where framed regions aren't detected (e.g., separator
    grids where content rows are interspersed with uniform rows).
    """
    if input_grid.shape != output_grid.shape:
        return []

    results = []
    h, w = input_grid.shape
    ev = _check_periodic_content(input_grid, output_grid, bg, 0, 0, h, w)
    if ev is not None:
        results.append(ev)
    return results


def _find_all_framed_regions(
    grid: Grid, bg: int, *, offset_r: int = 0, offset_c: int = 0, depth: int = 3,
) -> list[FramedRegion]:
    """Recursively find framed regions up to `depth` levels."""
    if depth <= 0:
        return []
    regions = detect_framed_regions(grid, bg)
    result = []
    for r in regions:
        # Add with adjusted coordinates
        adjusted = FramedRegion(
            frame_color=r.frame_color,
            interior=r.interior,
            row=offset_r + r.row,
            col=offset_c + r.col,
            height=r.height,
            width=r.width,
            interior_colors=r.interior_colors,
            interior_bg=r.interior_bg,
        )
        result.append(adjusted)
        # Recurse into interior
        sub = _find_all_framed_regions(
            r.interior, r.interior_bg,
            offset_r=offset_r + r.row,
            offset_c=offset_c + r.col,
            depth=depth - 1,
        )
        result.extend(sub)
    return result


def fit_framed_periodic_repair(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> Sketch | None:
    """Try to fit a framed-periodic-repair sketch.

    Evidence required:
    1. At least one framed region detected in every demo
    2. Interior has periodic content with violations that the output fixes
    3. Consistent axis and period across demos
    """
    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    per_demo_evidence: list[list[dict]] = []
    bg_colors: list[int] = []

    for demo in demos:
        bg = detect_bg(demo.input)
        bg_colors.append(bg)

        # Recursively find all framed regions up to 3 levels deep
        all_regions = _find_all_framed_regions(demo.input, bg, offset_r=0, offset_c=0, depth=3)

        demo_ev: list[dict] = []
        for region in all_regions:
            ev = _check_periodic_repair_for_region(region, demo.input, demo.output)
            if ev is not None:
                demo_ev.append(ev)

        # Fallback: scan interiors of found regions for periodic content
        # (handles cases with nested separators that don't form clean sub-frames)
        if not demo_ev:
            for region in all_regions:
                r, c, h, w = region.row, region.col, region.height, region.width
                if r + h > demo.output.shape[0] or c + w > demo.output.shape[1]:
                    continue
                int_in = region.interior
                int_out = demo.output[r:r + h, c:c + w]
                if int_in.shape != int_out.shape:
                    continue
                ev = _check_periodic_content(int_in, int_out, region.frame_color, r, c, h, w)
                if ev is not None:
                    demo_ev.append(ev)

        per_demo_evidence.append(demo_ev)

    # Check: every demo must have at least one periodic-repair region
    if any(not ev_list for ev_list in per_demo_evidence):
        return None

    # Check consistency: same axis and period across demos
    all_axes = set()
    all_periods = set()
    total_violations = 0
    for ev_list in per_demo_evidence:
        for ev in ev_list:
            all_axes.add(ev["axis"])
            all_periods.add(ev["period"])
            total_violations += ev["total_violations"]

    # Allow mixed axes (different regions may have different axes)
    # but require at least one consistent axis+period pair
    dominant_axis = max(all_axes, key=lambda a: sum(
        1 for evl in per_demo_evidence for ev in evl if ev["axis"] == a
    ))
    dominant_period = max(all_periods, key=lambda p: sum(
        1 for evl in per_demo_evidence for ev in evl if ev["period"] == p
    ))

    # Build the sketch
    frame_colors = set()
    for evl in per_demo_evidence:
        for ev in evl:
            frame_colors.add(ev["frame_color"])

    from aria.sketch import Primitive

    steps = [make_identify_roles(bg_colors=bg_colors)]

    # Step 2: peel outermost frame(s) to expose interior
    steps.append(SketchStep(
        name="interior",
        primitive=Primitive.PEEL_FRAME,
        roles=(RoleVar("frame", RoleKind.FRAME, "frame/border color"),),
        input_refs=("roles",),
        description="strip outermost frame border(s) to expose interior content",
        evidence={"frame_colors_observed": sorted(frame_colors)},
    ))

    # Step 3: partition interior into sub-cells by separators
    steps.append(SketchStep(
        name="cells",
        primitive=Primitive.PARTITION_GRID,
        input_refs=("interior",),
        description="split interior into sub-cells by row/column separators",
    ))

    # Step 4: repair periodic violations per line in each cell
    steps.append(SketchStep(
        name="repaired",
        primitive=Primitive.REPAIR_LINES,
        slots=(
            Slot("axis", SlotType.AXIS, evidence=dominant_axis),
            Slot("period", SlotType.INT, constraint="positive", evidence=dominant_period),
        ),
        input_refs=("cells",),
        description=(
            f"infer motif + repair {dominant_axis}-periodic mismatches "
            f"(period≈{dominant_period}), {total_violations} violation(s) across {len(demos)} demo(s)"
        ),
        evidence={
            "axis": dominant_axis,
            "period": dominant_period,
            "total_violations": total_violations,
            "per_demo_region_count": [len(evl) for evl in per_demo_evidence],
        },
    ))

    # Step 5: 2D motif repair for remaining violations
    steps.append(SketchStep(
        name="motif_repaired",
        primitive=Primitive.REPAIR_2D_MOTIF,
        input_refs=("repaired",),
        description="infer 2D tile motif per cell, repair remaining mismatches",
    ))

    output_ref = "motif_repaired"

    return Sketch(
        task_id=task_id,
        steps=tuple(steps),
        output_ref=output_ref,
        description=(
            f"inside framed regions, detect {dominant_axis}-periodic pattern "
            f"(period {dominant_period}) and repair {total_violations} violation(s)"
        ),
        metadata={
            "family": "framed_periodic_repair",
            "bg_colors": bg_colors,
            "frame_colors": sorted(frame_colors),
            "dominant_axis": dominant_axis,
            "dominant_period": dominant_period,
        },
    )


# ---------------------------------------------------------------------------
# Fitter: composite role alignment
# ---------------------------------------------------------------------------


def _check_composite_alignment(
    demos: tuple[DemoPair, ...],
) -> dict | None:
    """Check if composites move to align their center to an anchor axis.

    Returns evidence dict or None.
    """
    per_demo_roles: list[CompositeDecomposition] = []
    per_demo_out_roles: list[CompositeDecomposition] = []

    for demo in demos:
        bg = detect_bg(demo.input)
        inp_dec = decompose_composites(demo.input, bg)
        out_dec = decompose_composites(demo.output, bg)

        if not inp_dec.composites or inp_dec.anchor is None:
            return None
        if inp_dec.center_color is None:
            return None

        per_demo_roles.append(inp_dec)
        per_demo_out_roles.append(out_dec)

    # Verify structural consistency: same number of composites
    n_composites = len(per_demo_roles[0].composites)
    if any(len(d.composites) != n_composites for d in per_demo_roles):
        return None

    # Check cross-demo structural signature consistency
    sig0 = tuple(c.structural_signature for c in per_demo_roles[0].composites)
    for dr in per_demo_roles[1:]:
        sig_i = tuple(c.structural_signature for c in dr.composites)
        if sig_i != sig0:
            return None

    # Match input composites to output by proximity of center pixels
    per_demo_deltas: list[list[tuple[int, int]]] = []
    for di, (demo, inp_dec) in enumerate(zip(demos, per_demo_roles)):
        out_dec = per_demo_out_roles[di]
        # Find output center-color singletons
        out_centers = [
            o for o in out_dec.composites
        ]
        # Also check isolated singletons as output centers (composites may not form in output)
        out_center_positions = [(c.center_row, c.center_col) for c in out_centers]
        # Add isolated singletons of center color
        for iso in out_dec.isolated:
            if iso.color == inp_dec.center_color:
                out_center_positions.append((iso.row, iso.col))

        inp_centers = [(c.center_row, c.center_col) for c in inp_dec.composites]

        if len(out_center_positions) < len(inp_centers):
            # Try matching via output grid pixel scan
            out_grid = demo.output
            for r in range(out_grid.shape[0]):
                for c in range(out_grid.shape[1]):
                    if int(out_grid[r, c]) == inp_dec.center_color:
                        pos = (r, c)
                        if pos not in out_center_positions:
                            out_center_positions.append(pos)

        # Greedy nearest matching
        used: set[int] = set()
        deltas: list[tuple[int, int]] = []
        for ir, ic in inp_centers:
            best_idx, best_dist = -1, float("inf")
            for oi, (orow, ocol) in enumerate(out_center_positions):
                if oi in used:
                    continue
                dist = abs(orow - ir) + abs(ocol - ic)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = oi
            if best_idx < 0:
                return None
            used.add(best_idx)
            orow, ocol = out_center_positions[best_idx]
            deltas.append((orow - ir, ocol - ic))

        per_demo_deltas.append(deltas)

    # Check anchor alignment: for each demo, is there an axis where
    # all composite centers move to match the anchor's coordinate?
    per_demo_axis: list[str | None] = []
    for di, inp_dec in enumerate(per_demo_roles):
        anchor = inp_dec.anchor
        assert anchor is not None
        anchor_row, anchor_col = anchor.row, anchor.col

        found_axis = None
        for axis in ("col", "row"):
            ok = True
            for delta, comp in zip(per_demo_deltas[di], inp_dec.composites):
                dr, dc = delta
                if axis == "col":
                    expected_dc = anchor_col - comp.center_col
                    if dc != expected_dc:
                        ok = False
                        break
                else:
                    expected_dr = anchor_row - comp.center_row
                    if dr != expected_dr:
                        ok = False
                        break
            if ok:
                found_axis = axis
                break
        per_demo_axis.append(found_axis)

    all_aligned = all(a is not None for a in per_demo_axis)
    if not all_aligned:
        return None

    # Color rotation detection
    bg_colors = [d.bg_color for d in per_demo_roles]
    center_colors = [d.center_color for d in per_demo_roles]
    frame_colors = [d.frame_color for d in per_demo_roles]
    roles_rotate = len(set(center_colors)) > 1 or len(set(frame_colors)) > 1

    return {
        "n_composites": n_composites,
        "per_demo_axis": per_demo_axis,
        "bg_colors": bg_colors,
        "center_colors": center_colors,
        "frame_colors": frame_colors,
        "roles_rotate": roles_rotate,
        "structural_signatures": [list(s) for s in [sig0]],
        "anchor_positions": [
            (d.anchor.row, d.anchor.col) for d in per_demo_roles
        ],
    }


def fit_composite_role_alignment(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> Sketch | None:
    """Try to fit a composite-role-alignment sketch.

    Evidence required:
    1. Every demo has composites (center+frame) with a singleton anchor
    2. Structural signatures are consistent across demos
    3. Composites move to align center to anchor axis
    """
    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    evidence = _check_composite_alignment(demos)
    if evidence is None:
        return None

    axes = evidence["per_demo_axis"]
    dominant_axis = max(set(axes), key=axes.count)

    steps = [make_identify_roles(bg_colors=evidence["bg_colors"])]

    # Find composites step
    steps.append(SketchStep(
        name="composites",
        primitive=PrimitiveFamily.FIND_COMPOSITES,
        roles=(
            RoleVar("center", RoleKind.CENTER, "singleton center pixel"),
            RoleVar("frame_cc", RoleKind.FRAME, "frame CCs enclosing center"),
        ),
        input_refs=("roles",),
        description=f"find {evidence['n_composites']} composite motif(s) per demo",
        evidence={
            "center_colors": evidence["center_colors"],
            "frame_colors": evidence["frame_colors"],
            "roles_rotate": evidence["roles_rotate"],
        },
    ))

    # Alignment step
    steps.append(make_composite_role_alignment(
        composites_ref="composites",
        axis=dominant_axis,
    ))

    # Build paint step
    steps.append(SketchStep(
        name="output",
        primitive=PrimitiveFamily.COMPOSE_SEQUENTIAL,
        input_refs=("aligned",),
        slots=(
            Slot("paint_rule", SlotType.TRANSFORM,
                 constraint="erase original composite positions, paint at aligned positions"),
        ),
        description="erase composites from input, repaint at aligned positions",
    ))

    return Sketch(
        task_id=task_id,
        steps=tuple(steps),
        output_ref="output",
        description=(
            f"align {evidence['n_composites']} composite motif(s) to anchor "
            f"on {dominant_axis} axis"
            + (" (colors rotate across demos)" if evidence["roles_rotate"] else "")
        ),
        confidence=0.9 if evidence["roles_rotate"] else 1.0,
        metadata={
            "family": "composite_role_alignment",
            **evidence,
        },
    )


# ---------------------------------------------------------------------------
# Fitter: canvas construction (dims-change via tile / upscale / crop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CanvasEvidence:
    """Evidence that the input→output transformation is a canvas construction."""
    strategy: str          # "tile", "upscale", or "crop"
    output_dims: tuple[int, int]  # (rows, cols) of the output
    # Strategy-specific parameters
    tile_rows: int = 0
    tile_cols: int = 0
    scale_factor: int = 0
    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)  # (row, col, h, w)
    bg_color: int = 0


def _detect_canvas_strategy(
    demo: DemoPair,
) -> _CanvasEvidence | None:
    """Detect the canvas construction strategy for one demo pair."""
    ih, iw = demo.input.shape
    oh, ow = demo.output.shape

    if ih == oh and iw == ow:
        return None  # same dims — not a canvas task

    bg = detect_bg(demo.input)

    # Check upscale: output is an integer multiple of input in both dims
    if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
        fr = oh // ih
        fc = ow // iw
        if fr == fc:
            expected = np.repeat(np.repeat(demo.input, fr, axis=0), fc, axis=1)
            if np.array_equal(expected, demo.output):
                return _CanvasEvidence(
                    strategy="upscale",
                    output_dims=(oh, ow),
                    scale_factor=fr,
                    bg_color=bg,
                )

    # Check tile: output is an integer multiple via np.tile
    if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
        tr = oh // ih
        tc = ow // iw
        expected = np.tile(demo.input, (tr, tc))
        if np.array_equal(expected, demo.output):
            return _CanvasEvidence(
                strategy="tile",
                output_dims=(oh, ow),
                tile_rows=tr,
                tile_cols=tc,
                bg_color=bg,
            )

    # Check tile with different aspect: output rows are multiple of input rows
    # but cols might differ, or vice versa
    if oh >= ih and ow >= iw:
        if oh % ih == 0 and ow % iw == 0:
            tr = oh // ih
            tc = ow // iw
            expected = np.tile(demo.input, (tr, tc))
            if np.array_equal(expected, demo.output):
                return _CanvasEvidence(
                    strategy="tile",
                    output_dims=(oh, ow),
                    tile_rows=tr,
                    tile_cols=tc,
                    bg_color=bg,
                )

    # Check crop: output is smaller than input and matches a sub-region
    if oh <= ih and ow <= iw and (oh < ih or ow < iw):
        for r0 in range(ih - oh + 1):
            for c0 in range(iw - ow + 1):
                if np.array_equal(demo.input[r0:r0 + oh, c0:c0 + ow], demo.output):
                    return _CanvasEvidence(
                        strategy="crop",
                        output_dims=(oh, ow),
                        crop_region=(r0, c0, oh, ow),
                        bg_color=bg,
                    )

    return None


def fit_canvas_construction(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> Sketch | None:
    """Try to fit a canvas-construction sketch.

    Evidence required:
    1. Input and output have different dimensions
    2. The dimension relationship is consistent across demos
    3. The strategy (tile, upscale, or crop) is the same for all demos
    """
    if not demos:
        return None
    # Must have dims-change
    if all(d.input.shape == d.output.shape for d in demos):
        return None

    per_demo: list[_CanvasEvidence] = []
    for demo in demos:
        ev = _detect_canvas_strategy(demo)
        if ev is None:
            return None
        per_demo.append(ev)

    # All demos must use the same strategy
    strategies = {ev.strategy for ev in per_demo}
    if len(strategies) != 1:
        return None
    strategy = per_demo[0].strategy

    # Strategy-specific consistency checks
    if strategy == "tile":
        tile_dims = {(ev.tile_rows, ev.tile_cols) for ev in per_demo}
        if len(tile_dims) != 1:
            return None
        tile_rows, tile_cols = per_demo[0].tile_rows, per_demo[0].tile_cols
    elif strategy == "upscale":
        factors = {ev.scale_factor for ev in per_demo}
        if len(factors) != 1:
            return None
    elif strategy == "crop":
        regions = {ev.crop_region for ev in per_demo}
        if len(regions) != 1:
            return None

    from aria.sketch import Primitive

    bg_colors = [ev.bg_color for ev in per_demo]

    steps = [make_identify_roles(bg_colors=bg_colors)]

    if strategy == "tile":
        steps.append(SketchStep(
            name="canvas",
            primitive=Primitive.CONSTRUCT_CANVAS,
            slots=(
                Slot("output_dims", SlotType.DIMS,
                     constraint="fixed",
                     evidence=per_demo[0].output_dims),
                Slot("layout_rule", SlotType.TRANSFORM,
                     constraint="tile input grid",
                     evidence=f"tile({tile_rows},{tile_cols})"),
            ),
            input_refs=("roles",),
            description=f"tile input grid {tile_rows}x{tile_cols}",
            evidence={
                "strategy": "tile",
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
            },
        ))
    elif strategy == "upscale":
        factor = per_demo[0].scale_factor
        steps.append(SketchStep(
            name="canvas",
            primitive=Primitive.CONSTRUCT_CANVAS,
            slots=(
                Slot("output_dims", SlotType.DIMS,
                     constraint="fixed",
                     evidence=per_demo[0].output_dims),
                Slot("layout_rule", SlotType.TRANSFORM,
                     constraint="upscale input grid",
                     evidence=f"upscale({factor})"),
            ),
            input_refs=("roles",),
            description=f"upscale input grid by factor {factor}",
            evidence={
                "strategy": "upscale",
                "scale_factor": factor,
            },
        ))
    elif strategy == "crop":
        region = per_demo[0].crop_region
        steps.append(SketchStep(
            name="canvas",
            primitive=Primitive.CONSTRUCT_CANVAS,
            slots=(
                Slot("output_dims", SlotType.DIMS,
                     constraint="fixed",
                     evidence=per_demo[0].output_dims),
                Slot("layout_rule", SlotType.TRANSFORM,
                     constraint="crop input grid",
                     evidence=f"crop({region})"),
            ),
            input_refs=("roles",),
            description=f"crop input at region {region}",
            evidence={
                "strategy": "crop",
                "crop_region": region,
            },
        ))

    return Sketch(
        task_id=task_id,
        steps=tuple(steps),
        output_ref="canvas",
        description=f"canvas construction: {strategy} input to {per_demo[0].output_dims}",
        metadata={
            "family": "canvas_construction",
            "strategy": strategy,
            "output_dims": per_demo[0].output_dims,
            "bg_colors": bg_colors,
        },
    )


# ---------------------------------------------------------------------------
# Fitter: object movement (uniform translate / gravity)
# ---------------------------------------------------------------------------


def _match_objects_by_color_size(
    inp_objs: list,
    out_objs: list,
) -> list[tuple] | None:
    """Match input objects to output objects by color+size, return (inp, out, dr, dc) tuples."""
    if len(inp_objs) != len(out_objs):
        return None
    used: set[int] = set()
    matches = []
    for io in inp_objs:
        best_j = None
        best_dist = float("inf")
        for j, oo in enumerate(out_objs):
            if j not in used and oo.color == io.color and oo.size == io.size:
                dist = abs(oo.bbox[1] - io.bbox[1]) + abs(oo.bbox[0] - io.bbox[0])
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
        if best_j is None:
            return None
        used.add(best_j)
        oo = out_objs[best_j]
        dr = oo.bbox[1] - io.bbox[1]
        dc = oo.bbox[0] - io.bbox[0]
        matches.append((io, oo, dr, dc))
    return matches


def _detect_gravity_direction(
    matches: list[tuple],
    grid_h: int,
    grid_w: int,
) -> str | None:
    """Check if all moved objects ended up at a grid edge."""
    moved = [(io, oo, dr, dc) for io, oo, dr, dc in matches if (dr, dc) != (0, 0)]
    if not moved:
        return None

    for direction, check in [
        ("down", lambda oo: oo.bbox[1] + oo.bbox[3] == grid_h),
        ("up", lambda oo: oo.bbox[1] == 0),
        ("right", lambda oo: oo.bbox[0] + oo.bbox[2] == grid_w),
        ("left", lambda oo: oo.bbox[0] == 0),
    ]:
        if all(check(oo) for _, oo, _, _ in moved):
            return direction
    return None


def fit_object_movement(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> Sketch | None:
    """Try to fit an object movement sketch.

    Detects two patterns:
    1. Uniform translate: all objects move by the same (dr, dc) in every demo
    2. Gravity: all objects move to the same grid edge in every demo
    """
    from aria.runtime.ops.selection import _find_objects

    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    from aria.sketch import Primitive

    bg_colors = []
    per_demo_matches = []

    for demo in demos:
        bg = detect_bg(demo.input)
        bg_colors.append(bg)
        inp_objs = [o for o in _find_objects(demo.input) if o.color != bg]
        out_objs = [o for o in _find_objects(demo.output) if o.color != bg]

        matches = _match_objects_by_color_size(inp_objs, out_objs)
        if matches is None:
            return None
        # At least one object must have moved
        if not any((dr, dc) != (0, 0) for _, _, dr, dc in matches):
            return None
        per_demo_matches.append(matches)

    # --- Check for uniform translate ---
    all_deltas = set()
    for matches in per_demo_matches:
        for _, _, dr, dc in matches:
            if (dr, dc) != (0, 0):
                all_deltas.add((dr, dc))

    if len(all_deltas) == 1:
        dr, dc = list(all_deltas)[0]
        # All moved objects use the same delta across all demos
        all_uniform = all(
            all((mdr, mdc) == (dr, dc) or (mdr, mdc) == (0, 0)
                for _, _, mdr, mdc in matches)
            for matches in per_demo_matches
        )
        if all_uniform:
            steps = [make_identify_roles(bg_colors=bg_colors)]
            steps.append(SketchStep(
                name="moved",
                primitive=Primitive.APPLY_TRANSFORM,
                slots=(
                    Slot("transform", SlotType.TRANSFORM,
                         evidence=f"translate_by({dr},{dc})"),
                ),
                input_refs=("roles",),
                description=f"translate all foreground objects by ({dr},{dc})",
                evidence={"strategy": "uniform_translate", "dr": dr, "dc": dc},
            ))
            return Sketch(
                task_id=task_id,
                steps=tuple(steps),
                output_ref="moved",
                description=f"uniform translate ({dr},{dc})",
                metadata={
                    "family": "object_movement",
                    "strategy": "uniform_translate",
                    "dr": dr, "dc": dc,
                    "bg_colors": bg_colors,
                },
            )

    # --- Check for gravity ---
    # Detect gravity direction AND verify it explains all movement
    gravity_directions = []
    for demo, matches in zip(demos, per_demo_matches):
        h, w = demo.input.shape
        direction = _detect_gravity_direction(matches, h, w)
        if direction is None:
            break
        # Verify: stationary objects must not have moved
        for io, oo, dr, dc in matches:
            if (dr, dc) == (0, 0):
                continue
            # Moved object should have ended up at the edge
            ox, oy, ow, oh = oo.bbox
            if direction == "down" and oy + oh != h:
                direction = None
                break
            if direction == "up" and oy != 0:
                direction = None
                break
            if direction == "right" and ox + ow != w:
                direction = None
                break
            if direction == "left" and ox != 0:
                direction = None
                break
        if direction is None:
            break
        gravity_directions.append(direction)

    if len(gravity_directions) == len(demos) and len(set(gravity_directions)) == 1:
        direction = gravity_directions[0]
        steps = [make_identify_roles(bg_colors=bg_colors)]
        steps.append(SketchStep(
            name="moved",
            primitive=Primitive.APPLY_TRANSFORM,
            slots=(
                Slot("transform", SlotType.TRANSFORM,
                     evidence=f"gravity({direction})"),
            ),
            input_refs=("roles",),
            description=f"move all foreground objects to {direction} edge",
            evidence={"strategy": "gravity", "direction": direction},
        ))
        return Sketch(
            task_id=task_id,
            steps=tuple(steps),
            output_ref="moved",
            description=f"gravity {direction}",
            metadata={
                "family": "object_movement",
                "strategy": "gravity",
                "direction": direction,
                "bg_colors": bg_colors,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Fitter: grid-level transforms (rotate, reflect, transpose, fill_enclosed)
# ---------------------------------------------------------------------------


def fit_grid_transform(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> Sketch | None:
    """Try to fit a single grid-level transform that explains all demos.

    Tries: rotate(90/180/270), reflect(row/col), transpose, fill_enclosed(c).
    Returns the first that passes exact verification on all train demos.
    """
    from aria.sketch import Primitive
    from aria.verify.verifier import verify
    from aria.runtime.executor import execute
    from aria.types import Bind, Call, Literal, Program, Ref, Type as T

    if not demos:
        return None
    # Skip trivial identity tasks
    if all(np.array_equal(d.input, d.output) for d in demos):
        return None

    bg_colors = [detect_bg(d.input) for d in demos]

    # Candidate transforms: (op_name, args_fn, description, evidence_dict)
    candidates: list[tuple[str, tuple, str, dict]] = []

    # Rotations
    for deg in (90, 180, 270):
        candidates.append((
            "rotate_grid",
            (Literal(deg, T.INT), Ref("input")),
            f"rotate {deg} degrees",
            {"transform": "rotate", "degrees": deg},
        ))

    # Reflections
    for axis, axis_name in ((0, "row"), (1, "col")):
        candidates.append((
            "reflect_grid",
            (Literal(axis, T.AXIS), Ref("input")),
            f"reflect across {axis_name} axis",
            {"transform": "reflect", "axis": axis_name},
        ))

    # Transpose
    candidates.append((
        "transpose_grid",
        (Ref("input"),),
        "transpose grid",
        {"transform": "transpose"},
    ))

    # Fill enclosed regions (try each color 0-9)
    palette: set[int] = set()
    for d in demos:
        for c in range(10):
            if int(np.sum(d.output == c)) > int(np.sum(d.input == c)):
                palette.add(c)
    # Also try bg colors
    palette.update(bg_colors)
    for c in sorted(palette):
        candidates.append((
            "fill_enclosed",
            (Ref("input"), Literal(c, T.COLOR)),
            f"fill enclosed regions with color {c}",
            {"transform": "fill_enclosed", "fill_color": c},
        ))

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

        steps = [make_identify_roles(bg_colors=bg_colors)]
        steps.append(SketchStep(
            name="transformed",
            primitive=Primitive.APPLY_TRANSFORM,
            slots=(
                Slot("transform", SlotType.TRANSFORM, evidence=desc),
            ),
            input_refs=("roles",),
            description=desc,
            evidence=evidence,
        ))
        return Sketch(
            task_id=task_id,
            steps=tuple(steps),
            output_ref="transformed",
            description=f"grid transform: {desc}",
            metadata={
                "family": "grid_transform",
                **evidence,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Specialization pass — extract static task structure from evidence
# ---------------------------------------------------------------------------


def specialize_sketch(
    graph: SketchGraph,
    demos: tuple[DemoPair, ...],
) -> Specialization:
    """Extract resolved static bindings from a sketch graph and demo evidence.

    Primary sources (in priority order):
    1. Graph node slot evidence — the authoritative per-node source
    2. Demo decomposition — derives per-demo role bindings and axes
    3. Graph metadata — compatibility fallback only, not the primary source

    This is a representation change — no new capabilities, but static
    structure becomes explicit and inspectable.
    """
    bindings: list[ResolvedBinding] = []

    # Detect which graph roles are present to drive relation specialization
    all_role_kinds: set[str] = set()
    for node in graph.nodes.values():
        for role in node.roles:
            all_role_kinds.add(role.kind.name)

    # --- Per-node bindings from slot evidence and role grounding ---
    node_slot_axis: str | None = None
    node_slot_period: int | None = None

    for nid in graph.topo_order():
        node = graph.nodes[nid]

        # Resolve slots that carry evidence
        for slot in node.slots:
            if slot.evidence is not None:
                bindings.append(ResolvedBinding(
                    node_id=nid,
                    name=slot.name,
                    value=slot.evidence,
                    source="evidence",
                ))
                # Track axis/period from node slots for task-level bindings
                if slot.name == "axis" and isinstance(slot.evidence, str):
                    node_slot_axis = slot.evidence
                if slot.name == "period" and isinstance(slot.evidence, int):
                    node_slot_period = slot.evidence

        # Resolve roles whose kind can be grounded from demos
        for role in node.roles:
            if role.kind == RoleKind.BG and demos:
                bg_colors = [detect_bg(d.input) for d in demos]
                if len(set(bg_colors)) == 1:
                    bindings.append(ResolvedBinding(
                        node_id=nid,
                        name=role.name,
                        value=bg_colors[0],
                        source="consensus",
                    ))

    # --- Task-level bindings: prefer node slot evidence, metadata as fallback ---
    dominant_axis = node_slot_axis or graph.metadata.get("dominant_axis")
    if dominant_axis is not None:
        bindings.append(ResolvedBinding(
            node_id="__task__",
            name="dominant_axis",
            value=dominant_axis,
            source="node_evidence" if node_slot_axis else "metadata_fallback",
        ))

    dominant_period = node_slot_period or graph.metadata.get("dominant_period")
    if dominant_period is not None:
        bindings.append(ResolvedBinding(
            node_id="__task__",
            name="dominant_period",
            value=dominant_period,
            source="node_evidence" if node_slot_period else "metadata_fallback",
        ))

    # Frame colors: from node evidence if available, else metadata fallback
    frame_colors = None
    frame_colors_from_node = False
    for node in graph.nodes.values():
        obs_fc = node.evidence.get("frame_colors_observed")
        if obs_fc is not None:
            frame_colors = obs_fc
            frame_colors_from_node = True
            break
    if frame_colors is None:
        frame_colors = graph.metadata.get("frame_colors")
    if frame_colors is not None:
        bindings.append(ResolvedBinding(
            node_id="__task__",
            name="frame_colors",
            value=frame_colors,
            source="node_evidence" if frame_colors_from_node else "metadata_fallback",
        ))

    # --- Relation / composite-alignment specialization ---
    # If graph has ANCHOR + CENTER roles, resolve per-demo composite roles
    # directly from demo decomposition so the compiler never needs to re-derive.
    if {"ANCHOR", "CENTER"} <= all_role_kinds and demos:
        _specialize_composite_roles(graph, demos, bindings)

    # --- Canvas construction specialization ---
    all_primitives = {n.primitive.name for n in graph.nodes.values()}
    if "CONSTRUCT_CANVAS" in all_primitives and demos:
        _specialize_canvas(graph, demos, bindings)

    # --- Movement specialization ---
    if "APPLY_TRANSFORM" in all_primitives and demos:
        _specialize_movement(graph, demos, bindings)

    return Specialization(
        task_id=graph.task_id,
        bindings=tuple(bindings),
        metadata={
            "n_demos": len(demos),
        },
    )


def _specialize_composite_roles(
    graph: SketchGraph,
    demos: tuple[DemoPair, ...],
    bindings: list[ResolvedBinding],
) -> None:
    """Resolve per-demo composite role bindings into specialization.

    Derives everything from demo decomposition — no metadata reads.
    Extracts center/frame/bg colors, anchor position, singleton index,
    alignment axis, composite count, and role rotation flag.
    """
    per_demo_roles: list[dict[str, Any]] = []
    per_demo_axis: list[str | None] = []

    for demo in demos:
        bg = detect_bg(demo.input)
        dec = decompose_composites(demo.input, bg)
        if dec.center_color is None or dec.frame_color is None or dec.anchor is None:
            return  # cannot resolve; skip relation specialization entirely
        n_center_singletons = sum(
            1 for c in dec.composites if c.center.color == dec.center_color
        ) + 1  # +1 for the anchor itself
        per_demo_roles.append({
            "bg": bg,
            "center": dec.center_color,
            "frame": dec.frame_color,
            "anchor_row": dec.anchor.row,
            "anchor_col": dec.anchor.col,
            "n_center_singletons": n_center_singletons,
        })

        # Derive alignment axis from input/output decomposition
        out_dec = decompose_composites(demo.output, bg)
        axis = _derive_alignment_axis(dec, out_dec, demo)
        per_demo_axis.append(axis)

    bindings.append(ResolvedBinding(
        node_id="__relation__",
        name="per_demo_roles",
        value=per_demo_roles,
        source="demo_decomposition",
    ))

    bindings.append(ResolvedBinding(
        node_id="__relation__",
        name="per_demo_axis",
        value=per_demo_axis,
        source="demo_decomposition",
    ))

    # Structural invariants derived from decomposition, not metadata
    n_composites = len(per_demo_roles[0]) if per_demo_roles else 0
    # Actually count composites from first demo's decomposition
    if demos:
        bg0 = detect_bg(demos[0].input)
        dec0 = decompose_composites(demos[0].input, bg0)
        n_composites = len(dec0.composites)
    bindings.append(ResolvedBinding(
        node_id="__relation__",
        name="n_composites",
        value=n_composites,
        source="demo_decomposition",
    ))

    center_colors = [r["center"] for r in per_demo_roles]
    frame_colors = [r["frame"] for r in per_demo_roles]
    roles_rotate = len(set(center_colors)) > 1 or len(set(frame_colors)) > 1
    bindings.append(ResolvedBinding(
        node_id="__relation__",
        name="roles_rotate",
        value=roles_rotate,
        source="demo_decomposition",
    ))


def _derive_alignment_axis(
    inp_dec: CompositeDecomposition,
    out_dec: CompositeDecomposition,
    demo: DemoPair,
) -> str | None:
    """Derive the alignment axis for one demo from input/output decomposition.

    Checks whether composites moved to align their center to the anchor's
    row or column coordinate.
    """
    if not inp_dec.composites or inp_dec.anchor is None or inp_dec.center_color is None:
        return None

    # Collect output center positions
    out_center_positions = [(c.center_row, c.center_col) for c in out_dec.composites]
    for iso in out_dec.isolated:
        if iso.color == inp_dec.center_color:
            out_center_positions.append((iso.row, iso.col))

    inp_centers = [(c.center_row, c.center_col) for c in inp_dec.composites]

    if len(out_center_positions) < len(inp_centers):
        out_grid = demo.output
        for r in range(out_grid.shape[0]):
            for c in range(out_grid.shape[1]):
                if int(out_grid[r, c]) == inp_dec.center_color:
                    pos = (r, c)
                    if pos not in out_center_positions:
                        out_center_positions.append(pos)

    # Greedy nearest matching to get deltas
    used: set[int] = set()
    deltas: list[tuple[int, int]] = []
    for ir, ic in inp_centers:
        best_idx, best_dist = -1, float("inf")
        for oi, (orow, ocol) in enumerate(out_center_positions):
            if oi in used:
                continue
            dist = abs(orow - ir) + abs(ocol - ic)
            if dist < best_dist:
                best_dist = dist
                best_idx = oi
        if best_idx < 0:
            return None
        used.add(best_idx)
        orow, ocol = out_center_positions[best_idx]
        deltas.append((orow - ir, ocol - ic))

    anchor = inp_dec.anchor
    anchor_row, anchor_col = anchor.row, anchor.col

    for axis in ("col", "row"):
        ok = True
        for delta, comp in zip(deltas, inp_dec.composites):
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
            return axis

    return None


def _specialize_canvas(
    graph: SketchGraph,
    demos: tuple[DemoPair, ...],
    bindings: list[ResolvedBinding],
) -> None:
    """Resolve canvas construction bindings from demo evidence.

    Derives strategy, dimensions, and strategy-specific parameters
    directly from input/output demo pairs.
    """
    per_demo: list[_CanvasEvidence] = []
    for demo in demos:
        ev = _detect_canvas_strategy(demo)
        if ev is None:
            return
        per_demo.append(ev)

    strategies = {ev.strategy for ev in per_demo}
    if len(strategies) != 1:
        return
    strategy = per_demo[0].strategy

    bindings.append(ResolvedBinding(
        node_id="__canvas__",
        name="strategy",
        value=strategy,
        source="demo_decomposition",
    ))
    bindings.append(ResolvedBinding(
        node_id="__canvas__",
        name="output_dims",
        value=per_demo[0].output_dims,
        source="demo_decomposition",
    ))

    if strategy == "tile":
        bindings.append(ResolvedBinding(
            node_id="__canvas__",
            name="tile_rows",
            value=per_demo[0].tile_rows,
            source="demo_decomposition",
        ))
        bindings.append(ResolvedBinding(
            node_id="__canvas__",
            name="tile_cols",
            value=per_demo[0].tile_cols,
            source="demo_decomposition",
        ))
    elif strategy == "upscale":
        bindings.append(ResolvedBinding(
            node_id="__canvas__",
            name="scale_factor",
            value=per_demo[0].scale_factor,
            source="demo_decomposition",
        ))
    elif strategy == "crop":
        bindings.append(ResolvedBinding(
            node_id="__canvas__",
            name="crop_region",
            value=per_demo[0].crop_region,
            source="demo_decomposition",
        ))


def _specialize_movement(
    graph: SketchGraph,
    demos: tuple[DemoPair, ...],
    bindings: list[ResolvedBinding],
) -> None:
    """Resolve movement and grid transform bindings from node evidence."""
    from aria.runtime.ops.selection import _find_objects

    # Look for APPLY_TRANSFORM nodes with grid transform evidence
    for node in graph.nodes.values():
        transform_type = node.evidence.get("transform")
        if transform_type is not None:
            bindings.append(ResolvedBinding(
                node_id="__grid_transform__",
                name="transform",
                value=transform_type,
                source="node_evidence",
            ))
            for key in ("degrees", "axis", "fill_color"):
                val = node.evidence.get(key)
                if val is not None:
                    bindings.append(ResolvedBinding(
                        node_id="__grid_transform__",
                        name=key,
                        value=val,
                        source="node_evidence",
                    ))
            return

    # Look for APPLY_TRANSFORM nodes with movement evidence
    for node in graph.nodes.values():
        strategy = node.evidence.get("strategy")
        if strategy == "uniform_translate":
            dr = node.evidence.get("dr")
            dc = node.evidence.get("dc")
            if dr is not None and dc is not None:
                bindings.append(ResolvedBinding(
                    node_id="__movement__",
                    name="strategy",
                    value="uniform_translate",
                    source="node_evidence",
                ))
                bindings.append(ResolvedBinding(
                    node_id="__movement__",
                    name="dr",
                    value=dr,
                    source="node_evidence",
                ))
                bindings.append(ResolvedBinding(
                    node_id="__movement__",
                    name="dc",
                    value=dc,
                    source="node_evidence",
                ))
                return

        if strategy == "gravity":
            direction = node.evidence.get("direction")
            if direction is not None:
                bindings.append(ResolvedBinding(
                    node_id="__movement__",
                    name="strategy",
                    value="gravity",
                    source="node_evidence",
                ))
                bindings.append(ResolvedBinding(
                    node_id="__movement__",
                    name="direction",
                    value=direction,
                    source="node_evidence",
                ))
                return

    # Fallback: derive from demos if no node evidence
    for demo in demos:
        bg = detect_bg(demo.input)
        inp_objs = [o for o in _find_objects(demo.input) if o.color != bg]
        out_objs = [o for o in _find_objects(demo.output) if o.color != bg]
        matches = _match_objects_by_color_size(inp_objs, out_objs)
        if matches is None:
            return
        h, w = demo.input.shape
        direction = _detect_gravity_direction(matches, h, w)
        if direction is not None:
            bindings.append(ResolvedBinding(
                node_id="__movement__",
                name="strategy",
                value="gravity",
                source="demo_decomposition",
            ))
            bindings.append(ResolvedBinding(
                node_id="__movement__",
                name="direction",
                value=direction,
                source="demo_decomposition",
            ))
            return
