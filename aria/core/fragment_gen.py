"""Dynamic task-conditioned fragment generation from residual structure.

Given a near-miss diagnostic (region residuals, subgraph blame, residual
pattern), generates candidate graph fragments that are inferred from the
task's actual structure — not from a fixed replacement list.

The generation is compositional: it uses a small typed primitive grammar
and conditions the proposals on observed residual characteristics.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from aria.core.graph import (
    ComputationGraph,
    GraphFragment,
    GraphNode,
    NodeSlot,
    ResidualPattern,
    RoleBinding,
    Specialization,
    SubgraphBlame,
    VerifyDiagnostic,
)


# ---------------------------------------------------------------------------
# Residual pattern analysis
# ---------------------------------------------------------------------------


def analyze_residual_pattern(
    demos: Sequence[Any],
    predicted_grids: list[Any],
) -> ResidualPattern:
    """Characterize the spatial structure of the residual.

    Looks at the diff between predicted and target to determine
    what kind of structural repair is needed.
    """
    total_clusters = 0
    total_isolated = 0
    has_movement = False
    colors_in_diff: set[int] = set()
    local_count = 0
    distributed_count = 0

    for demo, predicted in zip(demos, predicted_grids):
        if predicted is None:
            continue
        target = demo.output
        diff_mask = predicted != target

        if not np.any(diff_mask):
            continue

        # Count connected components in diff mask
        n_clusters = _count_clusters(diff_mask)
        total_clusters += n_clusters

        # Count isolated single-pixel diffs (potential markers/attractors)
        n_iso = _count_isolated_pixels(diff_mask)
        total_isolated += n_iso

        # Colors involved in diff
        diff_positions = np.where(diff_mask)
        for r, c in zip(*diff_positions):
            if int(target[r, c]) != _detect_bg_fast(target):
                colors_in_diff.add(int(target[r, c]))
            if int(demo.input[r, c]) != _detect_bg_fast(demo.input):
                colors_in_diff.add(int(demo.input[r, c]))

        # Movement detection: check if non-bg pixel positions differ
        # between input and output but pixel counts are similar
        inp_bg = _detect_bg_fast(demo.input)
        out_bg = _detect_bg_fast(target)
        inp_nonbg = set(zip(*np.where(demo.input != inp_bg)))
        out_nonbg = set(zip(*np.where(target != out_bg)))
        if inp_nonbg and out_nonbg:
            overlap = inp_nonbg & out_nonbg
            # Movement: positions changed but total non-bg count is similar
            if (len(overlap) < max(len(inp_nonbg), len(out_nonbg))
                    and abs(len(inp_nonbg) - len(out_nonbg)) <= max(len(inp_nonbg), len(out_nonbg)) // 2):
                has_movement = True

        # Local vs distributed
        rows_with_diff = len(set(diff_positions[0]))
        cols_with_diff = len(set(diff_positions[1]))
        total_cells = diff_mask.size
        diff_area = rows_with_diff * cols_with_diff
        if diff_area < total_cells * 0.25:
            local_count += 1
        else:
            distributed_count += 1

    return ResidualPattern(
        has_scattered_objects=total_clusters >= 2,
        n_diff_clusters=total_clusters,
        has_isolated_pixels=total_isolated > 0,
        n_isolated_pixels=total_isolated,
        has_shape_movement=has_movement,
        n_colors_in_diff=len(colors_in_diff),
        diff_is_local=local_count > distributed_count,
        diff_is_distributed=distributed_count >= local_count,
    )


def _detect_bg_fast(grid: Any) -> int:
    unique, counts = np.unique(grid, return_counts=True)
    return int(unique[np.argmax(counts)])


def _count_clusters(mask: Any) -> int:
    """Count connected components in a boolean mask (4-connected)."""
    if not np.any(mask):
        return 0
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labels[r, c] == 0:
                current_label += 1
                _flood_fill(mask, labels, r, c, current_label)
    return current_label


def _flood_fill(mask: Any, labels: Any, r: int, c: int, label: int) -> None:
    h, w = mask.shape
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if 0 <= cr < h and 0 <= cc < w and mask[cr, cc] and labels[cr, cc] == 0:
            labels[cr, cc] = label
            stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])


def _count_isolated_pixels(mask: Any) -> int:
    """Count diff pixels with no 4-connected diff neighbors."""
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    has_neighbor = True
                    break
            if not has_neighbor:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Dynamic fragment generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedFragment:
    """A dynamically generated fragment with its generation rationale."""
    fragment: GraphFragment
    rationale: str          # why this was generated (residual pattern features)
    pattern_match: float    # 0-1, how well the residual pattern matches this fragment


def generate_fragments(
    graph: ComputationGraph,
    spec: Specialization,
    diagnostic: VerifyDiagnostic,
    demos: Sequence[Any],
    predicted_grids: list[Any] | None = None,
    *,
    max_fragments: int = 8,
) -> list[GeneratedFragment]:
    """Generate candidate graph fragments conditioned on the task's residual structure.

    Uses the residual pattern to decide what kind of structural repairs
    might fix the near-miss. Each fragment is compositionally built from
    the primitive grammar, not looked up from a fixed table.
    """
    # Compute residual pattern if we have predictions
    if predicted_grids is not None:
        pattern = analyze_residual_pattern(demos, predicted_grids)
    else:
        pattern = _estimate_pattern_from_diagnostic(diagnostic)

    # Generate fragments conditioned on the pattern
    fragments: list[GeneratedFragment] = []

    # --- Pattern: scattered object clusters + movement → object-level transform ---
    if pattern.has_scattered_objects or pattern.has_shape_movement:
        fragments.append(_gen_object_select_transform(pattern))

    # --- Pattern: isolated pixels (markers/attractors) → marker-guided placement ---
    if pattern.has_isolated_pixels and pattern.n_isolated_pixels > 0:
        fragments.append(_gen_marker_guided_transform(pattern))

    # --- Pattern: multi-color diff → per-color transform ---
    if pattern.n_colors_in_diff >= 2:
        fragments.append(_gen_color_conditioned_transform(pattern))

    # --- Pattern: distributed diff → global grid transform ---
    if pattern.diff_is_distributed:
        fragments.append(_gen_global_grid_transform(pattern))

    # --- Pattern: local diff → select + local transform ---
    if pattern.diff_is_local:
        fragments.append(_gen_local_select_transform(pattern))

    # --- Pattern: many clusters → partition + per-cell transform ---
    if pattern.n_diff_clusters >= 3:
        fragments.append(_gen_partition_per_cell_transform(pattern))

    # --- Pattern: movement + markers → extract + relocate ---
    if pattern.has_shape_movement and pattern.has_isolated_pixels:
        fragments.append(_gen_extract_relocate(pattern))

    # --- Always offer: simplest alternative (identity-like) ---
    if len(fragments) < max_fragments:
        fragments.append(_gen_select_paint(pattern))

    return fragments[:max_fragments]


def _estimate_pattern_from_diagnostic(diag: VerifyDiagnostic) -> ResidualPattern:
    """Estimate a residual pattern from diagnostic fields when predictions aren't available."""
    total = diag.total_diff
    frac = diag.diff_fraction
    return ResidualPattern(
        has_scattered_objects=total > 10,
        n_diff_clusters=max(1, total // 20),
        has_isolated_pixels=frac < 0.2,
        n_isolated_pixels=max(0, total // 10),
        has_shape_movement=frac < 0.5 and total > 5,
        n_colors_in_diff=2,
        diff_is_local=frac < 0.25,
        diff_is_distributed=frac >= 0.25,
    )


# ---------------------------------------------------------------------------
# Fragment generators — each builds from the primitive grammar
# ---------------------------------------------------------------------------


def _gen_object_select_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff has scattered object clusters: select objects then transform."""
    nodes = {
        "_fg_select": GraphNode(
            id="_fg_select", op="SELECT_SUBSET", inputs=("_fg_in",),
            description="select non-bg objects",
        ),
        "_fg_transform": GraphNode(
            id="_fg_transform", op="APPLY_TRANSFORM", inputs=("_fg_select",),
            slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            description="transform selected objects",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label=f"select_transform_c{pattern.n_diff_clusters}",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_transform",
            description=f"select objects + transform (clusters={pattern.n_diff_clusters})",
        ),
        rationale=f"residual has {pattern.n_diff_clusters} scattered clusters",
        pattern_match=min(1.0, pattern.n_diff_clusters / 5.0),
    )


def _gen_marker_guided_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff has isolated pixels (markers): use markers to guide transform."""
    nodes = {
        "_fg_roles": GraphNode(
            id="_fg_roles", op="BIND_ROLE", inputs=("_fg_in",),
            roles=(
                RoleBinding(name="marker", kind="MARKER"),
                RoleBinding(name="target", kind="TARGET"),
            ),
            description="bind marker and target roles",
        ),
        "_fg_rel": GraphNode(
            id="_fg_rel", op="APPLY_RELATION", inputs=("_fg_roles",),
            description="apply marker-guided relation",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label=f"marker_guided_n{pattern.n_isolated_pixels}",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_rel",
            description=f"marker-guided transform ({pattern.n_isolated_pixels} markers)",
        ),
        rationale=f"residual has {pattern.n_isolated_pixels} isolated pixels (markers)",
        pattern_match=min(1.0, pattern.n_isolated_pixels / 4.0),
    )


def _gen_color_conditioned_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff involves multiple colors: per-color-role transform."""
    nodes = {
        "_fg_roles": GraphNode(
            id="_fg_roles", op="BIND_ROLE", inputs=("_fg_in",),
            roles=(RoleBinding(name="color_role", kind="COLOR_ROLE"),),
        ),
        "_fg_transform": GraphNode(
            id="_fg_transform", op="APPLY_TRANSFORM", inputs=("_fg_roles",),
            slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            description="color-conditioned transform",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label=f"color_transform_c{pattern.n_colors_in_diff}",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_transform",
            description=f"color-conditioned transform ({pattern.n_colors_in_diff} colors)",
        ),
        rationale=f"diff involves {pattern.n_colors_in_diff} distinct colors",
        pattern_match=min(1.0, pattern.n_colors_in_diff / 4.0),
    )


def _gen_global_grid_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff is distributed: whole-grid transform."""
    nodes = {
        "_fg_t": GraphNode(
            id="_fg_t", op="APPLY_TRANSFORM", inputs=("_fg_in",),
            slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            description="global grid transform",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label="global_transform",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_t",
            description="global grid transform (distributed diff)",
        ),
        rationale="diff is distributed across the grid",
        pattern_match=0.4,
    )


def _gen_local_select_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff is local: select region then transform."""
    nodes = {
        "_fg_sel": GraphNode(
            id="_fg_sel", op="SELECT_REGION", inputs=("_fg_in",),
            description="select local diff region",
        ),
        "_fg_t": GraphNode(
            id="_fg_t", op="APPLY_TRANSFORM", inputs=("_fg_sel",),
            slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            description="local region transform",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label="local_select_transform",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_t",
            description="select local region + transform",
        ),
        rationale="diff is concentrated in a local region",
        pattern_match=0.5,
    )


def _gen_partition_per_cell_transform(pattern: ResidualPattern) -> GeneratedFragment:
    """When diff has many clusters: partition grid then per-cell transform."""
    nodes = {
        "_fg_part": GraphNode(
            id="_fg_part", op="PARTITION_GRID", inputs=("_fg_in",),
            description="partition into cells",
        ),
        "_fg_rule": GraphNode(
            id="_fg_rule", op="APPLY_TRANSFORM", inputs=("_fg_part",),
            slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            description="per-cell transform rule",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label=f"partition_percell_n{pattern.n_diff_clusters}",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_rule",
            description=f"partition + per-cell transform ({pattern.n_diff_clusters} clusters)",
        ),
        rationale=f"{pattern.n_diff_clusters} diff clusters suggest cell-level rules",
        pattern_match=min(1.0, pattern.n_diff_clusters / 6.0),
    )


def _gen_extract_relocate(pattern: ResidualPattern) -> GeneratedFragment:
    """When objects move to marker positions: extract + relocate."""
    nodes = {
        "_fg_extract": GraphNode(
            id="_fg_extract", op="SELECT_SUBSET", inputs=("_fg_in",),
            description="extract objects and markers",
        ),
        "_fg_rel": GraphNode(
            id="_fg_rel", op="APPLY_RELATION", inputs=("_fg_extract",),
            description="compute object-marker relations",
        ),
        "_fg_paint": GraphNode(
            id="_fg_paint", op="PAINT", inputs=("_fg_rel",),
            description="paint relocated objects",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label=f"extract_relocate_m{pattern.n_isolated_pixels}",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_paint",
            description=f"extract + relocate objects to markers ({pattern.n_isolated_pixels} markers)",
        ),
        rationale=f"movement + {pattern.n_isolated_pixels} isolated marker pixels",
        pattern_match=min(1.0, 0.3 + pattern.n_isolated_pixels * 0.2),
    )


def _gen_select_paint(pattern: ResidualPattern) -> GeneratedFragment:
    """Fallback: select + paint (minimal structural repair)."""
    nodes = {
        "_fg_sel": GraphNode(
            id="_fg_sel", op="SELECT_SUBSET", inputs=("_fg_in",),
            description="select elements to modify",
        ),
        "_fg_paint": GraphNode(
            id="_fg_paint", op="PAINT", inputs=("_fg_sel",),
            description="paint/recolor selected elements",
        ),
    }
    return GeneratedFragment(
        fragment=GraphFragment(
            label="select_paint",
            nodes=nodes,
            input_id="_fg_in",
            output_id="_fg_paint",
            description="select + paint (fallback repair)",
        ),
        rationale="fallback structural alternative",
        pattern_match=0.2,
    )
