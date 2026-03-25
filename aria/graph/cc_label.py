"""Connected component labeling via flood fill.

Extracts ObjectNode instances from a grid by finding connected regions
of the same color. Supports both 4-connectivity and 8-connectivity.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from aria.types import ObjectNode, Shape, Symmetry


# 4-connected neighbor offsets (row, col)
_NEIGHBORS_4 = ((0, 1), (0, -1), (1, 0), (-1, 0))

# 8-connected neighbor offsets
_NEIGHBORS_8 = (
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def _flood_fill(
    grid: NDArray[np.uint8],
    visited: NDArray[np.bool_],
    start_r: int,
    start_c: int,
    neighbors: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    """BFS flood fill returning list of (row, col) for one component."""
    rows, cols = grid.shape
    color = grid[start_r, start_c]
    queue: deque[tuple[int, int]] = deque()
    queue.append((start_r, start_c))
    visited[start_r, start_c] = True
    pixels: list[tuple[int, int]] = []

    while queue:
        r, c = queue.popleft()
        pixels.append((r, c))
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == color:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return pixels


def _pixels_to_object(
    obj_id: int,
    color: int,
    pixels: list[tuple[int, int]],
) -> ObjectNode:
    """Convert a list of pixel coordinates into an ObjectNode.

    Shape and symmetry are set to placeholder values here.
    The caller should patch them via shapes.classify_shape and
    symmetry.detect_obj_symmetry after construction.
    """
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = np.zeros((h, w), dtype=np.bool_)
    for r, c in pixels:
        mask[r - min_r, c - min_c] = True

    # bbox uses (x, y, w, h) where x=col, y=row
    bbox = (min_c, min_r, w, h)

    return ObjectNode(
        id=obj_id,
        color=int(color),
        mask=mask,
        bbox=bbox,
        shape=Shape.IRREGULAR,  # placeholder; classified later
        symmetry=frozenset(),   # placeholder; detected later
        size=len(pixels),
    )


def _label_components(
    grid: NDArray[np.uint8],
    neighbors: tuple[tuple[int, int], ...],
    ignore_color: int | None = None,
) -> list[ObjectNode]:
    """Core labeling routine shared by 4-conn and 8-conn."""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)

    if ignore_color is not None:
        visited[grid == ignore_color] = True

    objects: list[ObjectNode] = []
    obj_id = 0

    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                pixels = _flood_fill(grid, visited, r, c, neighbors)
                objects.append(_pixels_to_object(obj_id, int(grid[r, c]), pixels))
                obj_id += 1

    return objects


def label_4conn(
    grid: NDArray[np.uint8],
    ignore_color: int | None = None,
) -> list[ObjectNode]:
    """Extract connected components using 4-connectivity.

    Parameters
    ----------
    grid : Grid
        2D array of color values (0-9).
    ignore_color : int or None
        Color to skip (typically the background).

    Returns
    -------
    list[ObjectNode]
        One node per connected component. Shape and symmetry fields are
        placeholders; use shapes.classify_shape / symmetry.detect_obj_symmetry
        to fill them.
    """
    return _label_components(grid, _NEIGHBORS_4, ignore_color)


def label_8conn(
    grid: NDArray[np.uint8],
    ignore_color: int | None = None,
) -> list[ObjectNode]:
    """Extract connected components using 8-connectivity.

    Parameters
    ----------
    grid : Grid
        2D array of color values (0-9).
    ignore_color : int or None
        Color to skip (typically the background).

    Returns
    -------
    list[ObjectNode]
        One node per connected component. Shape and symmetry fields are
        placeholders; use shapes.classify_shape / symmetry.detect_obj_symmetry
        to fill them.
    """
    return _label_components(grid, _NEIGHBORS_8, ignore_color)
