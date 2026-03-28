"""Sketch fitting — propose sketches from demo decompositions.

Given a task's train demos, decomposes each grid and tries to fit
a sketch that explains the input→output transformation. Each fitter
is bounded and evidence-backed: it inspects decomposition structure
and pixel diffs, then emits a Sketch with role variables and parameter
slots, or returns None if the evidence doesn't support the family.

Currently implements two families:
1. Framed periodic repair — detect period in framed interior, fix violations
2. Composite role alignment — align composite motifs to anchor axis
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
    RoleKind,
    RoleVar,
    Sketch,
    SketchStep,
    Slot,
    SlotType,
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
    """Try sketch families against a task's demos, in decomposition-ranked order.

    Each sketch family depends on a specific decomposition view. When a
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

    return FitResult(
        sketches=results,
        decomp_views_tried=tuple(ranked_views),
        decomp_ranking_applied=decomp_report.policy_name != "none",
        decomp_ranking_changed_order=decomp_report.order_changed,
        decomp_ranking_policy=decomp_report.policy_name,
    )


# ---------------------------------------------------------------------------
# Family 1: Framed periodic repair
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

    steps = [make_identify_roles(bg_colors=bg_colors)]

    # Add frame extraction step
    steps.append(SketchStep(
        name="regions",
        primitive=PrimitiveFamily.EXTRACT_REGION,
        roles=(RoleVar("frame", RoleKind.FRAME, "frame/border color"),),
        input_refs=("roles",),
        description=f"extract {sum(len(evl) for evl in per_demo_evidence) // len(demos)} framed interior region(s)",
        evidence={"frame_colors_observed": sorted(frame_colors)},
    ))

    # Add periodic repair step
    steps.append(SketchStep(
        name="repaired",
        primitive=PrimitiveFamily.REGION_PERIODIC_REPAIR,
        roles=(RoleVar("frame", RoleKind.FRAME),),
        slots=(
            Slot("axis", SlotType.AXIS, evidence=dominant_axis),
            Slot("period", SlotType.INT, constraint="positive", evidence=dominant_period),
        ),
        input_refs=("regions",),
        description=(
            f"detect {dominant_axis}-periodic pattern (period={dominant_period}), "
            f"fix {total_violations} violation(s) across {len(demos)} demo(s)"
        ),
        evidence={
            "axis": dominant_axis,
            "period": dominant_period,
            "total_violations": total_violations,
            "per_demo_region_count": [len(evl) for evl in per_demo_evidence],
        },
    ))

    # If multiple regions per demo, wrap in FOR_EACH_REGION
    max_regions = max(len(evl) for evl in per_demo_evidence)
    if max_regions > 1:
        steps.append(SketchStep(
            name="all_repaired",
            primitive=PrimitiveFamily.FOR_EACH_REGION,
            input_refs=("repaired",),
            description=f"apply periodic repair to each of {max_regions} framed regions",
        ))
        output_ref = "all_repaired"
    else:
        output_ref = "repaired"

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
# Family 2: Composite role alignment
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
