"""Fill-enclosed-regions runtime ops.

Two variants:
- fill_enclosed_regions(grid, fill_color): fill all interior bg regions with a fixed color
- fill_enclosed_regions_auto(grid): fill each interior bg region with its boundary color
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


def _fill_enclosed_regions(grid: Grid, fill_color: int) -> Grid:
    """Fill enclosed background regions with a fixed color."""
    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    mask = grid == bg

    labeled, n_regions = ndimage.label(mask)
    if n_regions < 2:
        return grid.copy()

    result = grid.copy()
    h, w = grid.shape

    # Border-touching labels are exterior
    border_labels: set[int] = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    for region_id in range(1, n_regions + 1):
        if region_id not in border_labels:
            result[labeled == region_id] = fill_color

    return result


def _fill_enclosed_regions_auto(grid: Grid) -> Grid:
    """Fill enclosed bg regions with their enclosing boundary color."""
    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    mask = grid == bg

    labeled, n_regions = ndimage.label(mask)
    if n_regions < 2:
        return grid.copy()

    result = grid.copy()
    h, w = grid.shape

    border_labels: set[int] = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    for region_id in range(1, n_regions + 1):
        if region_id in border_labels:
            continue
        region_mask = labeled == region_id
        dilated = ndimage.binary_dilation(region_mask)
        boundary = dilated & ~region_mask
        boundary_colors = grid[boundary]
        boundary_non_bg = boundary_colors[boundary_colors != bg]
        if len(boundary_non_bg) == 0:
            continue
        vals, counts = np.unique(boundary_non_bg, return_counts=True)
        fill_color = int(vals[np.argmax(counts)])
        result[region_mask] = fill_color

    return result


register(
    "fill_enclosed_regions",
    OpSignature(
        params=(("grid", Type.GRID), ("fill_color", Type.INT)),
        return_type=Type.GRID,
    ),
    _fill_enclosed_regions,
)

register(
    "fill_enclosed_regions_auto",
    OpSignature(
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
    ),
    _fill_enclosed_regions_auto,
)


def _fill_enclosed_marker_guided(grid: Grid) -> Grid:
    """Fill enclosed bg regions using the color of the nearest singleton marker.

    For each enclosed bg region:
    1. Find the enclosing boundary structure
    2. Find singleton markers (size=1 non-bg) adjacent to or within that structure
    3. Fill with the nearest marker's color

    Only fills regions where a singleton marker can be found nearby.
    """
    from aria.graph.cc_label import label_4conn

    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    mask = grid == bg

    labeled, n_regions = ndimage.label(mask)
    if n_regions < 2:
        return grid.copy()

    h, w = grid.shape

    # Find border-touching labels
    border_labels: set[int] = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    # Find singleton markers
    objs = label_4conn(grid, ignore_color=bg)
    singletons = [o for o in objs if o.size == 1]
    if not singletons:
        return grid.copy()

    result = grid.copy()

    for region_id in range(1, n_regions + 1):
        if region_id in border_labels:
            continue

        region_mask = labeled == region_id
        region_positions = np.argwhere(region_mask)
        if len(region_positions) == 0:
            continue

        # Find the enclosing boundary by dilating
        dilated = ndimage.binary_dilation(region_mask)
        boundary = dilated & ~region_mask

        # Get boundary pixel positions and the enclosing structure's bbox
        boundary_positions = np.argwhere(boundary)
        if len(boundary_positions) == 0:
            continue

        # Find the nearest singleton marker to this enclosed region
        region_center_r = region_positions[:, 0].mean()
        region_center_c = region_positions[:, 1].mean()

        best_marker = None
        best_dist = float("inf")
        for marker in singletons:
            mx, my = marker.bbox[0], marker.bbox[1]
            mr, mc = my, mx  # row, col
            d = abs(mr - region_center_r) + abs(mc - region_center_c)
            if d < best_dist:
                best_dist = d
                best_marker = marker

        if best_marker is not None:
            result[region_mask] = best_marker.color

    return result


register(
    "fill_enclosed_marker_guided",
    OpSignature(
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
    ),
    _fill_enclosed_marker_guided,
)


def _fill_enclosed_rarest_singleton(grid: Grid) -> Grid:
    """Fill enclosed bg regions with the rarest singleton color.

    Finds the singleton (size=1) color that appears least often,
    then fills all enclosed bg regions with that color.
    """
    from collections import Counter
    from aria.graph.cc_label import label_4conn

    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    objs = label_4conn(grid, ignore_color=bg)
    singletons = [o for o in objs if o.size == 1]
    if not singletons:
        return grid.copy()

    # Find the rarest singleton color
    color_counts = Counter(o.color for o in singletons)
    rarest_color = min(color_counts, key=color_counts.get)

    return _fill_enclosed_regions(grid, rarest_color)


register(
    "fill_enclosed_rarest_singleton",
    OpSignature(
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
    ),
    _fill_enclosed_rarest_singleton,
)


def _fill_enclosed_per_composite(grid: Grid) -> Grid:
    """Fill enclosed bg per 8-conn composite structure using its singleton color.

    1. Label all non-bg pixels with 8-connectivity (ignoring color) to find
       multi-color composite structures.
    2. For each composite that contains a singleton-colored pixel (a color
       appearing exactly once in the entire grid), fill enclosed bg regions
       within that composite's bbox with the singleton color.
    3. Composites without singleton colors are left unchanged.
    """
    from aria.graph.cc_label import label_4conn

    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])

    # Find singleton colors (pixel count == 1 in entire grid)
    color_counts = np.bincount(grid.ravel(), minlength=10)
    singleton_colors = {c for c in range(10) if c != bg and color_counts[c] == 1}
    if not singleton_colors:
        return grid.copy()

    # Label 8-conn composites (all non-bg pixels, ignoring color)
    non_bg = grid != bg
    composite_labeled, n_composites = ndimage.label(non_bg, structure=np.ones((3, 3)))

    result = grid.copy()
    h, w = grid.shape

    for cid in range(1, n_composites + 1):
        comp_mask = composite_labeled == cid
        comp_colors = set(int(grid[r, c]) for r, c in zip(*np.where(comp_mask)))

        # Check if this composite contains a singleton color
        fill_color = None
        for sc in singleton_colors:
            if sc in comp_colors:
                fill_color = sc
                break

        if fill_color is None:
            continue  # no singleton → skip this composite

        # Find enclosed bg within this composite's bbox
        positions = np.argwhere(comp_mask)
        r0, c0 = int(positions[:, 0].min()), int(positions[:, 1].min())
        r1, c1 = int(positions[:, 0].max()), int(positions[:, 1].max())

        bbox_bg = grid[r0:r1 + 1, c0:c1 + 1] == bg
        if not np.any(bbox_bg):
            continue

        bg_labeled, n_bg = ndimage.label(bbox_bg)
        bh, bw = bbox_bg.shape

        border_labels: set[int] = set()
        for r in range(bh):
            if bg_labeled[r, 0] > 0:
                border_labels.add(bg_labeled[r, 0])
            if bg_labeled[r, bw - 1] > 0:
                border_labels.add(bg_labeled[r, bw - 1])
        for c in range(bw):
            if bg_labeled[0, c] > 0:
                border_labels.add(bg_labeled[0, c])
            if bg_labeled[bh - 1, c] > 0:
                border_labels.add(bg_labeled[bh - 1, c])

        out_region = result[r0:r1 + 1, c0:c1 + 1]
        for lbl in range(1, n_bg + 1):
            if lbl not in border_labels:
                out_region[bg_labeled == lbl] = fill_color

    return result


register(
    "fill_enclosed_per_composite",
    OpSignature(
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
    ),
    _fill_enclosed_per_composite,
)
