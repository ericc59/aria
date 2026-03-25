"""Pairwise spatial relation computation between ObjectNodes.

Computes spatial, topological, alignment, and match relations for
every ordered pair of objects.
"""

from __future__ import annotations

import numpy as np

from aria.types import (
    AlignRel,
    MatchRel,
    ObjectNode,
    RelationEdge,
    SpatialRel,
    TopoRel,
)


def compute_relations(objects: list[ObjectNode]) -> list[RelationEdge]:
    """Compute pairwise relations between all object pairs.

    For n objects, produces up to n*(n-1) directed edges (both directions
    are computed since spatial relations are directional).

    Parameters
    ----------
    objects : list[ObjectNode]
        Objects to compute relations between.

    Returns
    -------
    list[RelationEdge]
        One edge per ordered pair with non-empty relation sets.
    """
    edges: list[RelationEdge] = []

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue

            spatial = _spatial_rels(a, b)
            topo = _topo_rels(a, b)
            align = _align_rels(a, b)
            match = _match_rels(a, b)

            # Only emit edge if at least one relation exists
            if spatial or topo or align or match:
                edges.append(RelationEdge(
                    src=a.id,
                    dst=b.id,
                    spatial=frozenset(spatial),
                    topo=frozenset(topo),
                    align=frozenset(align),
                    match=frozenset(match),
                ))

    return edges


def _center(obj: ObjectNode) -> tuple[float, float]:
    """Return (center_row, center_col) of an object's bbox."""
    x, y, w, h = obj.bbox
    return y + h / 2.0, x + w / 2.0


def _spatial_rels(a: ObjectNode, b: ObjectNode) -> set[SpatialRel]:
    """Determine directional spatial relations from a to b."""
    rels: set[SpatialRel] = set()
    ar, ac = _center(a)
    br, bc = _center(b)

    if ar < br:
        rels.add(SpatialRel.ABOVE)
    elif ar > br:
        rels.add(SpatialRel.BELOW)

    if ac < bc:
        rels.add(SpatialRel.LEFT)
    elif ac > bc:
        rels.add(SpatialRel.RIGHT)

    # Diagonal: both row and col differ
    if SpatialRel.ABOVE in rels or SpatialRel.BELOW in rels:
        if SpatialRel.LEFT in rels or SpatialRel.RIGHT in rels:
            rels.add(SpatialRel.DIAGONAL)

    return rels


def _bbox_to_slices(obj: ObjectNode) -> tuple[int, int, int, int]:
    """Return (r_min, r_max, c_min, c_max) inclusive from bbox (x, y, w, h)."""
    x, y, w, h = obj.bbox
    return y, y + h - 1, x, x + w - 1


def _topo_rels(a: ObjectNode, b: ObjectNode) -> set[TopoRel]:
    """Determine topological relations between a and b."""
    rels: set[TopoRel] = set()

    ar0, ar1, ac0, ac1 = _bbox_to_slices(a)
    br0, br1, bc0, bc1 = _bbox_to_slices(b)

    # Check bbox overlap first (fast reject for disjoint)
    bboxes_overlap = ar0 <= br1 and ar1 >= br0 and ac0 <= bc1 and ac1 >= bc0

    if not bboxes_overlap:
        # Check adjacency: bboxes are 1 pixel apart
        expanded_overlap = (
            ar0 - 1 <= br1 and ar1 + 1 >= br0
            and ac0 - 1 <= bc1 and ac1 + 1 >= bc0
        )
        if expanded_overlap:
            rels.add(TopoRel.ADJACENT)
        else:
            rels.add(TopoRel.DISJOINT)
        return rels

    # Bboxes overlap — check pixel-level relations using global masks
    # Build global coordinate space covering both objects
    r_min = min(ar0, br0)
    r_max = max(ar1, br1)
    c_min = min(ac0, bc0)
    c_max = max(ac1, bc1)
    gh = r_max - r_min + 1
    gw = c_max - c_min + 1

    mask_a = np.zeros((gh, gw), dtype=np.bool_)
    mask_b = np.zeros((gh, gw), dtype=np.bool_)

    mask_a[ar0 - r_min:ar1 - r_min + 1, ac0 - c_min:ac1 - c_min + 1] = a.mask
    mask_b[br0 - r_min:br1 - r_min + 1, bc0 - c_min:bc1 - c_min + 1] = b.mask

    intersection = mask_a & mask_b
    has_overlap = intersection.any()

    if has_overlap:
        # Check containment: a contains b if all of b's pixels are in a
        if np.array_equal(intersection, mask_b):
            rels.add(TopoRel.CONTAINS)
        else:
            rels.add(TopoRel.OVERLAPS)
    else:
        # Bboxes overlap but masks don't — check adjacency via dilation
        from scipy.ndimage import binary_dilation
        dilated_a = binary_dilation(mask_a)
        if (dilated_a & mask_b).any():
            rels.add(TopoRel.ADJACENT)
        else:
            rels.add(TopoRel.DISJOINT)

    return rels


def _align_rels(a: ObjectNode, b: ObjectNode) -> set[AlignRel]:
    """Check alignment relations between a and b."""
    rels: set[AlignRel] = set()
    ar, ac = _center(a)
    br, bc = _center(b)

    # Horizontal alignment: same center row
    if abs(ar - br) < 0.5:
        rels.add(AlignRel.ALIGN_H)

    # Vertical alignment: same center column
    if abs(ac - bc) < 0.5:
        rels.add(AlignRel.ALIGN_V)

    # Diagonal alignment: centers on a 45-degree line
    if abs(abs(ar - br) - abs(ac - bc)) < 0.5:
        if ar != br:  # avoid degenerate case where they're at the same spot
            rels.add(AlignRel.ALIGN_DIAG)

    return rels


def _match_rels(a: ObjectNode, b: ObjectNode) -> set[MatchRel]:
    """Check property-match relations between a and b."""
    rels: set[MatchRel] = set()

    if a.color == b.color:
        rels.add(MatchRel.SAME_COLOR)

    if a.shape == b.shape:
        rels.add(MatchRel.SAME_SHAPE)

    if a.size == b.size:
        rels.add(MatchRel.SAME_SIZE)

    # Mirror: one is the horizontal or vertical flip of the other
    if a.mask.shape == b.mask.shape:
        if np.array_equal(a.mask, np.fliplr(b.mask)):
            rels.add(MatchRel.MIRROR)
        elif np.array_equal(a.mask, np.flipud(b.mask)):
            rels.add(MatchRel.MIRROR)

    return rels
