"""Compositional value/slot algebra for structure-derived parameterization.

This module adds the missing bridge between aria's high-level typed scene
programs and the grid-conditioned parameterization needed by real ARC tasks.

The ops here let programs express patterns like:
  - move object by -height(object)
  - fill with dominant_color of adjacent object's subgrid
  - crop to union_bbox of filtered objects
  - connect selected objects with their own color
  - erase an object, modify it, repaint it

All ops are typed, bounded, and compose through the existing Program IR.
No new types are introduced — everything uses existing Type variants.

Categories:
  B. Value functions — extract derived values from objects/grids
  C. Aggregation combinators — reduce object sets to single values
  D. Generic actions — parameterized grid mutations
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from aria.types import Grid, ObjectNode, Property, Shape, Type
from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.analysis import _get_property


# ===================================================================
# B. Value functions
# ===================================================================


def _obj_bbox_region(obj: ObjectNode) -> tuple[int, int, int, int]:
    """Return the object's bbox as a (x, y, w, h) region.

    Bridges object → region so crop/fill_region can use it directly.
    """
    return obj.bbox


def _subgrid_of(obj: ObjectNode, grid: Grid) -> Grid:
    """Extract the rectangular subgrid under an object's bounding box.

    Useful for analyzing the color content, pattern, or structure
    within a specific object's region. Clips to grid bounds.
    """
    x, y, w, h = obj.bbox
    rows, cols = grid.shape
    r0 = max(0, y)
    r1 = min(rows, y + h)
    c0 = max(0, x)
    c1 = min(cols, x + w)
    if r0 >= r1 or c0 >= c1:
        return np.zeros((0, 0), dtype=np.uint8)
    return grid[r0:r1, c0:c1].copy()


def _dominant_color_in(grid: Grid, bg: int) -> int:
    """Most common non-bg color in a grid region.

    Returns bg if no non-bg colors exist.
    """
    if grid.size == 0:
        return bg
    flat = grid.ravel()
    mask = flat != bg
    nonbg = flat[mask]
    if len(nonbg) == 0:
        return bg
    vals, counts = np.unique(nonbg, return_counts=True)
    return int(vals[np.argmax(counts)])


def _minority_color_in(grid: Grid, bg: int) -> int:
    """Least common non-bg color in a grid region.

    Returns bg if no non-bg colors exist.
    """
    if grid.size == 0:
        return bg
    flat = grid.ravel()
    mask = flat != bg
    nonbg = flat[mask]
    if len(nonbg) == 0:
        return bg
    vals, counts = np.unique(nonbg, return_counts=True)
    return int(vals[np.argmin(counts)])


def _num_colors_in(grid: Grid, bg: int) -> int:
    """Count of unique non-bg colors in a grid region."""
    if grid.size == 0:
        return 0
    unique = np.unique(grid)
    return int(np.sum(unique != bg))


def _touches_border(obj: ObjectNode, grid: Grid) -> bool:
    """True if any mask cell of the object is on the grid edge."""
    x, y, w, h = obj.bbox
    rows, cols = grid.shape
    mask = obj.mask

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                gr, gc = y + r, x + c
                if gr == 0 or gr == rows - 1 or gc == 0 or gc == cols - 1:
                    return True
    return False


def _displacement(a: ObjectNode, b: ObjectNode) -> tuple[int, int]:
    """Center-to-center displacement vector from a to b.

    Returns (dr, dc) where dr = row_b - row_a, dc = col_b - col_a.
    Returned as (rows, cols) to match DIMS convention.
    """
    ar = a.bbox[1] + a.bbox[3] // 2
    ac = a.bbox[0] + a.bbox[2] // 2
    br = b.bbox[1] + b.bbox[3] // 2
    bc = b.bbox[0] + b.bbox[2] // 2
    return (br - ar, bc - ac)


# ===================================================================
# C. Aggregation combinators
# ===================================================================


def _argmax_by(prop: Property, objs: set[ObjectNode]) -> ObjectNode:
    """Object with the maximum property value. Ties broken by id (lowest)."""
    if not objs:
        raise ValueError("argmax_by: empty object set")
    return max(objs, key=lambda o: (_get_property(prop, o), -o.id))


def _argmin_by(prop: Property, objs: set[ObjectNode]) -> ObjectNode:
    """Object with the minimum property value. Ties broken by id (lowest)."""
    if not objs:
        raise ValueError("argmin_by: empty object set")
    return min(objs, key=lambda o: (_get_property(prop, o), o.id))


def _most_common_val(prop: Property, objs: set[ObjectNode]) -> int:
    """Most common value of a property across the object set.

    Returns an int (works for COLOR, SIZE, POS_X, POS_Y).
    Raises ValueError if set is empty.
    """
    if not objs:
        raise ValueError("most_common_val: empty object set")
    counter: Counter = Counter()
    for obj in objs:
        val = _get_property(prop, obj)
        counter[val] += 1
    return counter.most_common(1)[0][0]


def _union_bbox(objs: set[ObjectNode]) -> tuple[int, int, int, int]:
    """Smallest bounding box enclosing all objects.

    Returns (x, y, w, h) in the same convention as ObjectNode.bbox.
    """
    if not objs:
        raise ValueError("union_bbox: empty object set")
    min_x = min(o.bbox[0] for o in objs)
    min_y = min(o.bbox[1] for o in objs)
    max_x = max(o.bbox[0] + o.bbox[2] for o in objs)
    max_y = max(o.bbox[1] + o.bbox[3] for o in objs)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _merge_objects(objs: set[ObjectNode]) -> ObjectNode:
    """Merge all objects into one combined-mask ObjectNode.

    The union bbox encloses all objects. The mask is the union of all
    individual masks. Color is the most common color across the set.
    """
    if not objs:
        raise ValueError("merge_objects: empty object set")

    bbox = _union_bbox(objs)
    bx, by, bw, bh = bbox
    mask = np.zeros((bh, bw), dtype=np.bool_)

    color_counter: Counter = Counter()
    total_size = 0

    for obj in objs:
        ox, oy, ow, oh = obj.bbox
        color_counter[obj.color] += obj.size
        total_size += obj.size
        for r in range(obj.mask.shape[0]):
            for c in range(obj.mask.shape[1]):
                if obj.mask[r, c]:
                    mr = (oy + r) - by
                    mc = (ox + c) - bx
                    if 0 <= mr < bh and 0 <= mc < bw:
                        mask[mr, mc] = True

    dominant_color = color_counter.most_common(1)[0][0]
    min_id = min(o.id for o in objs)

    return ObjectNode(
        id=min_id,
        color=dominant_color,
        mask=mask,
        bbox=bbox,
        shape=Shape.IRREGULAR,
        symmetry=frozenset(),
        size=total_size,
    )


# ===================================================================
# D. Generic actions
# ===================================================================


def _stamp(obj: ObjectNode, grid: Grid) -> Grid:
    """Paint an object at its own bbox position onto a grid.

    Only writes cells where the mask is True. Returns a new grid.
    This is the self-positioning variant of place_at.
    """
    result = grid.copy()
    x, y, w, h = obj.bbox
    mask = obj.mask
    mh, mw = mask.shape
    gr, gc = result.shape

    r_start = max(0, y)
    r_end = min(gr, y + mh)
    c_start = max(0, x)
    c_end = min(gc, x + mw)

    mr_start = r_start - y
    mr_end = r_end - y
    mc_start = c_start - x
    mc_end = c_end - x

    if mr_start >= mr_end or mc_start >= mc_end:
        return result

    sub_mask = mask[mr_start:mr_end, mc_start:mc_end]
    result[r_start:r_end, c_start:c_end][sub_mask] = obj.color
    return result


def _cover_obj(obj: ObjectNode, bg: int, grid: Grid) -> Grid:
    """Erase an object by filling its mask cells with bg color.

    Returns a new grid with the object's footprint replaced by bg.
    """
    result = grid.copy()
    x, y, w, h = obj.bbox
    mask = obj.mask
    mh, mw = mask.shape
    gr, gc = result.shape

    for r in range(mh):
        for c in range(mw):
            if mask[r, c]:
                ar, ac = y + r, x + c
                if 0 <= ar < gr and 0 <= ac < gc:
                    result[ar, ac] = bg
    return result


def _paint_cells(
    cells: set[tuple[int, int]], color: int, grid: Grid
) -> Grid:
    """Fill a set of (row, col) cells with a color.

    Bridges topo ops (boundary, interior, connect) to grid mutations.
    Cells outside grid bounds are silently ignored.
    """
    result = grid.copy()
    rows, cols = result.shape
    for r, c in cells:
        if 0 <= r < rows and 0 <= c < cols:
            result[r, c] = color
    return result


def _connect_paint(
    a: ObjectNode, b: ObjectNode, color: int, grid: Grid
) -> Grid:
    """Draw a line between the centers of two objects, painted with color.

    Combines connect (topo) + paint_cells in one step for convenience.
    """
    from aria.runtime.ops.topo import _connect

    cells = _connect(a, b)
    return _paint_cells(cells, color, grid)


def _negate(n: int) -> int:
    """Return -n. Shorthand for sub(0, n)."""
    return -n


# ===================================================================
# Registration
# ===================================================================

# -- B. Value functions --

register(
    "obj_bbox_region",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.REGION),
    _obj_bbox_region,
)

register(
    "subgrid_of",
    OpSignature(
        params=(("obj", Type.OBJECT), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _subgrid_of,
)

register(
    "dominant_color_in",
    OpSignature(
        params=(("grid", Type.GRID), ("bg", Type.COLOR)),
        return_type=Type.COLOR,
    ),
    _dominant_color_in,
)

register(
    "minority_color_in",
    OpSignature(
        params=(("grid", Type.GRID), ("bg", Type.COLOR)),
        return_type=Type.COLOR,
    ),
    _minority_color_in,
)

register(
    "num_colors_in",
    OpSignature(
        params=(("grid", Type.GRID), ("bg", Type.COLOR)),
        return_type=Type.INT,
    ),
    _num_colors_in,
)

register(
    "touches_border",
    OpSignature(
        params=(("obj", Type.OBJECT), ("grid", Type.GRID)),
        return_type=Type.BOOL,
    ),
    _touches_border,
)

register(
    "displacement",
    OpSignature(
        params=(("a", Type.OBJECT), ("b", Type.OBJECT)),
        return_type=Type.DIMS,
    ),
    _displacement,
)

# -- C. Aggregation combinators --

register(
    "argmax_by",
    OpSignature(
        params=(("prop", Type.PROPERTY), ("objs", Type.OBJECT_SET)),
        return_type=Type.OBJECT,
    ),
    _argmax_by,
)

register(
    "argmin_by",
    OpSignature(
        params=(("prop", Type.PROPERTY), ("objs", Type.OBJECT_SET)),
        return_type=Type.OBJECT,
    ),
    _argmin_by,
)

register(
    "most_common_val",
    OpSignature(
        params=(("prop", Type.PROPERTY), ("objs", Type.OBJECT_SET)),
        return_type=Type.INT,
    ),
    _most_common_val,
)

register(
    "union_bbox",
    OpSignature(
        params=(("objs", Type.OBJECT_SET),),
        return_type=Type.REGION,
    ),
    _union_bbox,
)

register(
    "merge_objects",
    OpSignature(
        params=(("objs", Type.OBJECT_SET),),
        return_type=Type.OBJECT,
    ),
    _merge_objects,
)

# -- D. Generic actions --

register(
    "stamp",
    OpSignature(
        params=(("obj", Type.OBJECT), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _stamp,
)

register(
    "cover_obj",
    OpSignature(
        params=(("obj", Type.OBJECT), ("bg", Type.COLOR), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _cover_obj,
)

register(
    "paint_cells",
    OpSignature(
        params=(("cells", Type.REGION), ("color", Type.COLOR), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _paint_cells,
)

register(
    "connect_paint",
    OpSignature(
        params=(
            ("a", Type.OBJECT),
            ("b", Type.OBJECT),
            ("color", Type.COLOR),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _connect_paint,
)

register(
    "negate",
    OpSignature(params=(("n", Type.INT),), return_type=Type.INT),
    _negate,
)
