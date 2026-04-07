"""Per-entity operations for ForEach body steps.

These ops take an ObjectNode (bound by ForEach iteration) and perform
entity-local grid mutations — the building blocks for entity-conditional
programs.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from aria.types import Grid, ObjectNode, Type
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Entity property extraction
# ---------------------------------------------------------------------------


def _entity_color(obj: ObjectNode) -> int:
    """Return the color of an entity."""
    return obj.color


def _entity_size(obj: ObjectNode) -> int:
    """Return the pixel count of an entity."""
    return obj.size


def _entity_row(obj: ObjectNode) -> int:
    """Return the top row of an entity's bbox."""
    return obj.bbox[1]  # bbox is (x, y, w, h)


def _entity_col(obj: ObjectNode) -> int:
    """Return the left column of an entity's bbox."""
    return obj.bbox[0]


# ---------------------------------------------------------------------------
# Per-entity grid mutations
# ---------------------------------------------------------------------------


def _fill_entity(obj: ObjectNode, color: int, grid: Grid) -> Grid:
    """Fill all pixels of an entity's mask with a color."""
    result = grid.copy()
    x, y, w, h = obj.bbox
    mask = obj.mask
    for dr in range(h):
        for dc in range(w):
            if mask[dr, dc]:
                r, c = y + dr, x + dc
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    result[r, c] = color
    return result


def _erase_entity(obj: ObjectNode, grid: Grid) -> Grid:
    """Erase an entity (fill with background color = most frequent)."""
    bg = int(np.bincount(grid.ravel()).argmax())
    return _fill_entity(obj, bg, grid)


def _fill_entity_enclosed(obj: ObjectNode, color: int, grid: Grid) -> Grid:
    """Fill enclosed background regions within an entity's bbox."""
    result = grid.copy()
    x, y, w, h = obj.bbox
    r0, c0 = y, x
    r1, c1 = y + h, x + w
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(grid.shape[0], r1)
    c1 = min(grid.shape[1], c1)

    if r1 <= r0 or c1 <= c0:
        return result

    region = grid[r0:r1, c0:c1]
    bg = int(np.bincount(grid.ravel()).argmax())
    bg_mask = region == bg

    if not np.any(bg_mask):
        return result

    labeled, n = ndimage.label(bg_mask)
    rh, rw = region.shape

    # Find border-touching labels
    border_labels = set()
    for r in range(rh):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, rw - 1] > 0:
            border_labels.add(labeled[r, rw - 1])
    for c in range(rw):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[rh - 1, c] > 0:
            border_labels.add(labeled[rh - 1, c])

    out_region = result[r0:r1, c0:c1]
    for lbl in range(1, n + 1):
        if lbl not in border_labels:
            out_region[labeled == lbl] = color

    return result


def _recolor_entity(obj: ObjectNode, from_color: int, to_color: int, grid: Grid) -> Grid:
    """Recolor pixels of from_color to to_color within entity bbox."""
    result = grid.copy()
    x, y, w, h = obj.bbox
    r0, c0 = y, x
    r1, c1 = y + h, x + w
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(grid.shape[0], r1)
    c1 = min(grid.shape[1], c1)

    region = result[r0:r1, c0:c1]
    orig = grid[r0:r1, c0:c1]
    region[orig == from_color] = to_color
    return result


def _fill_entity_bbox_bg(obj: ObjectNode, color: int, grid: Grid) -> Grid:
    """Fill background pixels within entity bbox with a color."""
    result = grid.copy()
    bg = int(np.bincount(grid.ravel()).argmax())
    x, y, w, h = obj.bbox
    r0, c0 = y, x
    r1, c1 = y + h, x + w
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(grid.shape[0], r1)
    c1 = min(grid.shape[1], c1)

    region = result[r0:r1, c0:c1]
    orig = grid[r0:r1, c0:c1]
    region[orig == bg] = color
    return result


# ---------------------------------------------------------------------------
# Entity filtering
# ---------------------------------------------------------------------------


def _filter_by_min_size(objs: set, threshold: int) -> set:
    """Keep objects with size >= threshold."""
    return {o for o in objs if o.size >= threshold}


def _filter_by_max_size(objs: set, threshold: int) -> set:
    """Keep objects with size <= threshold."""
    return {o for o in objs if o.size <= threshold}


def _filter_by_color(objs: set, color: int) -> set:
    """Keep objects of a specific color."""
    return {o for o in objs if o.color == color}


def _filter_by_not_color(objs: set, color: int) -> set:
    """Keep objects NOT of a specific color."""
    return {o for o in objs if o.color != color}


def _filter_singletons(objs: set) -> set:
    """Keep only singleton (size==1) objects."""
    return {o for o in objs if o.size == 1}


def _filter_non_singletons(objs: set) -> set:
    """Keep only non-singleton objects."""
    return {o for o in objs if o.size > 1}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


# Entity properties (OBJECT -> scalar)
register("entity_color", OpSignature(
    params=(("obj", Type.OBJECT),),
    return_type=Type.COLOR), _entity_color)

register("entity_size", OpSignature(
    params=(("obj", Type.OBJECT),),
    return_type=Type.INT), _entity_size)

register("entity_row", OpSignature(
    params=(("obj", Type.OBJECT),),
    return_type=Type.INT), _entity_row)

register("entity_col", OpSignature(
    params=(("obj", Type.OBJECT),),
    return_type=Type.INT), _entity_col)

# Per-entity grid mutations (OBJECT x ... x GRID -> GRID)
register("fill_entity", OpSignature(
    params=(("obj", Type.OBJECT), ("color", Type.COLOR), ("grid", Type.GRID)),
    return_type=Type.GRID), _fill_entity)

register("erase_entity", OpSignature(
    params=(("obj", Type.OBJECT), ("grid", Type.GRID)),
    return_type=Type.GRID), _erase_entity)

register("fill_entity_enclosed", OpSignature(
    params=(("obj", Type.OBJECT), ("color", Type.COLOR), ("grid", Type.GRID)),
    return_type=Type.GRID), _fill_entity_enclosed)

register("recolor_entity", OpSignature(
    params=(("obj", Type.OBJECT), ("from_color", Type.COLOR),
            ("to_color", Type.COLOR), ("grid", Type.GRID)),
    return_type=Type.GRID), _recolor_entity)

register("fill_entity_bbox_bg", OpSignature(
    params=(("obj", Type.OBJECT), ("color", Type.COLOR), ("grid", Type.GRID)),
    return_type=Type.GRID), _fill_entity_bbox_bg)

# Entity set filtering (OBJECT_SET x ... -> OBJECT_SET)
register("filter_by_min_size", OpSignature(
    params=(("objs", Type.OBJECT_SET), ("threshold", Type.INT)),
    return_type=Type.OBJECT_SET), _filter_by_min_size)

register("filter_by_max_size", OpSignature(
    params=(("objs", Type.OBJECT_SET), ("threshold", Type.INT)),
    return_type=Type.OBJECT_SET), _filter_by_max_size)

register("filter_by_color", OpSignature(
    params=(("objs", Type.OBJECT_SET), ("color", Type.COLOR)),
    return_type=Type.OBJECT_SET), _filter_by_color)

register("filter_by_not_color", OpSignature(
    params=(("objs", Type.OBJECT_SET), ("color", Type.COLOR)),
    return_type=Type.OBJECT_SET), _filter_by_not_color)

register("filter_singletons", OpSignature(
    params=(("objs", Type.OBJECT_SET),),
    return_type=Type.OBJECT_SET), _filter_singletons)

register("filter_non_singletons", OpSignature(
    params=(("objs", Type.OBJECT_SET),),
    return_type=Type.OBJECT_SET), _filter_non_singletons)
