"""Delta computation between two StateGraphs.

Matches objects across input/output grids and identifies additions,
removals, and modifications.
"""

from __future__ import annotations

import numpy as np

from aria.types import Delta, ObjectNode, StateGraph


def compute_delta(sg_in: StateGraph, sg_out: StateGraph) -> Delta:
    """Compute the delta between an input and output StateGraph.

    Matching strategy:
    1. Try to match by overlapping position and same color (strongest signal).
    2. Fall back to matching by color + similar size.
    3. Unmatched input objects are "removed"; unmatched output objects are "added".
    4. Matched pairs are compared field-by-field for modifications.

    Parameters
    ----------
    sg_in : StateGraph
        The input state graph.
    sg_out : StateGraph
        The output state graph.

    Returns
    -------
    Delta
        Describes what changed between input and output.
    """
    # Detect dimension changes
    dims_changed = None
    if sg_in.grid.shape != sg_out.grid.shape:
        dims_changed = (
            (int(sg_in.grid.shape[0]), int(sg_in.grid.shape[1])),
            (int(sg_out.grid.shape[0]), int(sg_out.grid.shape[1])),
        )

    # Build matching
    matched: list[tuple[ObjectNode, ObjectNode]] = []
    unmatched_in: set[int] = {obj.id for obj in sg_in.objects}
    unmatched_out: set[int] = {obj.id for obj in sg_out.objects}

    out_by_id: dict[int, ObjectNode] = {obj.id: obj for obj in sg_out.objects}
    in_by_id: dict[int, ObjectNode] = {obj.id: obj for obj in sg_in.objects}

    # Pass 1: match by overlap + same color
    for obj_in in sg_in.objects:
        if obj_in.id not in unmatched_in:
            continue
        best_match = _find_best_overlap_match(obj_in, sg_out.objects, unmatched_out)
        if best_match is not None:
            matched.append((obj_in, best_match))
            unmatched_in.discard(obj_in.id)
            unmatched_out.discard(best_match.id)

    # Pass 2: match remaining by color + closest size
    remaining_in = [in_by_id[i] for i in sorted(unmatched_in)]
    remaining_out = [out_by_id[i] for i in sorted(unmatched_out)]

    for obj_in in remaining_in:
        best = _find_best_color_size_match(obj_in, remaining_out, unmatched_out)
        if best is not None:
            matched.append((obj_in, best))
            unmatched_in.discard(obj_in.id)
            unmatched_out.discard(best.id)

    # Compute results
    added = tuple(out_by_id[i] for i in sorted(unmatched_out))
    removed = tuple(sorted(unmatched_in))
    modified = _compute_modifications(matched)

    return Delta(
        added=added,
        removed=removed,
        modified=modified,
        dims_changed=dims_changed,
    )


def _bbox_overlap(a: ObjectNode, b: ObjectNode) -> int:
    """Compute pixel overlap area between two bounding boxes."""
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox

    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    return x_overlap * y_overlap


def _find_best_overlap_match(
    obj_in: ObjectNode,
    out_objects: tuple[ObjectNode, ...],
    available: set[int],
) -> ObjectNode | None:
    """Find the best matching output object by position overlap + color."""
    best: ObjectNode | None = None
    best_overlap = 0

    for obj_out in out_objects:
        if obj_out.id not in available:
            continue
        if obj_out.color != obj_in.color:
            continue
        overlap = _bbox_overlap(obj_in, obj_out)
        if overlap > best_overlap:
            best_overlap = overlap
            best = obj_out

    return best


def _find_best_color_size_match(
    obj_in: ObjectNode,
    candidates: list[ObjectNode],
    available: set[int],
) -> ObjectNode | None:
    """Find best match by same color, then closest size."""
    best: ObjectNode | None = None
    best_diff = float("inf")

    for obj_out in candidates:
        if obj_out.id not in available:
            continue
        if obj_out.color != obj_in.color:
            continue
        diff = abs(obj_out.size - obj_in.size)
        if diff < best_diff:
            best_diff = diff
            best = obj_out

    return best


def _compute_modifications(
    matched: list[tuple[ObjectNode, ObjectNode]],
) -> tuple[tuple[int, str, object, object], ...]:
    """Compare matched pairs and report field-level differences."""
    mods: list[tuple[int, str, object, object]] = []

    for obj_in, obj_out in matched:
        if obj_in.color != obj_out.color:
            mods.append((obj_in.id, "color", obj_in.color, obj_out.color))

        if obj_in.bbox != obj_out.bbox:
            mods.append((obj_in.id, "bbox", obj_in.bbox, obj_out.bbox))

        if obj_in.shape != obj_out.shape:
            mods.append((obj_in.id, "shape", obj_in.shape, obj_out.shape))

        if obj_in.size != obj_out.size:
            mods.append((obj_in.id, "size", obj_in.size, obj_out.size))

        if not np.array_equal(obj_in.mask, obj_out.mask):
            mods.append((obj_in.id, "mask", obj_in.mask, obj_out.mask))

        if obj_in.symmetry != obj_out.symmetry:
            mods.append((obj_in.id, "symmetry", obj_in.symmetry, obj_out.symmetry))

    return tuple(mods)
