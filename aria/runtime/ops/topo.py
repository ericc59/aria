"""Topological operations: flood fill, boundary, hull, connectivity."""

from __future__ import annotations

from collections import deque

import numpy as np

from aria.types import Grid, ObjectNode, Shape, Type
from aria.runtime.ops import OpSignature, register
from aria.graph.cc_label import label_4conn


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _flood_fill(grid: Grid, pos: tuple[int, int], color: int) -> Grid:
    """Flood fill from a position, replacing connected cells with `color`.

    Uses 4-connectivity. Returns a NEW grid.
    pos is (col, row).
    """
    result = grid.copy()
    col, row = pos
    rows, cols = result.shape

    if row < 0 or row >= rows or col < 0 or col >= cols:
        return result

    target_color = int(result[row, col])
    if target_color == color:
        return result

    queue: deque[tuple[int, int]] = deque()
    queue.append((row, col))
    visited = np.zeros((rows, cols), dtype=np.bool_)
    visited[row, col] = True

    while queue:
        r, c = queue.popleft()
        result[r, c] = color
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not visited[nr, nc]
                and int(result[nr, nc]) == target_color
            ):
                visited[nr, nc] = True
                queue.append((nr, nc))

    return result


def _boundary(obj: ObjectNode) -> set[tuple[int, int]]:
    """Return the set of mask cells that border a False cell or the mask edge.

    Coordinates are (row, col) relative to the mask origin.
    """
    mask = obj.mask
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            # Check if any neighbor is outside or False.
            is_border = False
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    is_border = True
                    break
            if is_border:
                # Convert to absolute coords using bbox.
                x, y, _, _ = obj.bbox
                cells.add((y + r, x + c))

    return cells


def _interior(obj: ObjectNode) -> set[tuple[int, int]]:
    """Return the set of mask cells that are NOT on the boundary.

    Coordinates are absolute (row, col).
    """
    mask = obj.mask
    h, w = mask.shape
    x, y, _, _ = obj.bbox
    boundary = _boundary(obj)
    cells: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                abs_coord = (y + r, x + c)
                if abs_coord not in boundary:
                    cells.add(abs_coord)

    return cells


def _connect(obj1: ObjectNode, obj2: ObjectNode) -> set[tuple[int, int]]:
    """Return the set of cells forming a straight-line connection between
    the centers of two objects.

    Uses a simple Bresenham-like line between bounding-box centers.
    Coordinates are absolute (row, col).
    """
    x1, y1, w1, h1 = obj1.bbox
    x2, y2, w2, h2 = obj2.bbox
    cr1 = y1 + h1 // 2
    cc1 = x1 + w1 // 2
    cr2 = y2 + h2 // 2
    cc2 = x2 + w2 // 2

    cells: set[tuple[int, int]] = set()

    dr = cr2 - cr1
    dc = cc2 - cc1
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        cells.add((cr1, cc1))
        return cells

    r_inc = dr / steps
    c_inc = dc / steps
    r, c = float(cr1), float(cc1)

    for _ in range(steps + 1):
        cells.add((round(r), round(c)))
        r += r_inc
        c += c_inc

    return cells


def _hull(obj: ObjectNode) -> ObjectNode:
    """Return the convex hull of an object as a filled rectangular ObjectNode.

    This is a simplified hull: the axis-aligned bounding box filled solidly.
    """
    x, y, w, h = obj.bbox
    mask = np.ones((h, w), dtype=np.bool_)
    return ObjectNode(
        id=obj.id,
        color=obj.color,
        mask=mask,
        bbox=(x, y, w, h),
        shape=Shape.RECT,
        symmetry=obj.symmetry,
        size=h * w,
    )


def _connected_components(grid: Grid) -> set[ObjectNode]:
    """Extract all connected components from a grid (no background exclusion).

    Uses 4-connectivity.
    """
    objects = label_4conn(grid, ignore_color=None)
    return set(objects)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "flood_fill",
    OpSignature(
        params=(("grid", Type.GRID), ("pos", Type.DIMS), ("color", Type.COLOR)),
        return_type=Type.GRID,
    ),
    _flood_fill,
)

register(
    "boundary",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.REGION),
    _boundary,
)

register(
    "interior",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.REGION),
    _interior,
)

register(
    "connect",
    OpSignature(
        params=(("obj1", Type.OBJECT), ("obj2", Type.OBJECT)),
        return_type=Type.REGION,
    ),
    _connect,
)

register(
    "hull",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.OBJECT),
    _hull,
)

register(
    "connected_components",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.OBJECT_SET),
    _connected_components,
)
