"""Region isolation ops — extract subgrids for downstream composition.

  crop_to_content(grid) -> grid
    Crop to the tight bounding box of all non-background content.

  crop_to_object(grid, color) -> grid
    Crop to the bounding box of the largest object of the given color.

  crop_frame_interior(grid) -> grid
    If the grid has a frame border, return the interior. Otherwise return as-is.

All ops return grids that can be fed to downstream lanes.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.selection import _find_objects
from aria.types import Grid, Type


def _crop_to_content(grid: Grid) -> Grid:
    """Crop to the tight bounding box of all non-background pixels."""
    if grid.size == 0:
        return grid.copy()

    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])

    mask = grid != bg
    if not mask.any():
        return grid.copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    return grid[r_min:r_max + 1, c_min:c_max + 1].copy()


def _crop_to_object(grid: Grid, color: int) -> Grid:
    """Crop to the bounding box of the largest object of the given color."""
    if grid.size == 0:
        return grid.copy()

    objs = [o for o in _find_objects(grid) if o.color == color]
    if not objs:
        return _crop_to_content(grid)

    largest = max(objs, key=lambda o: o.size)
    x, y, w, h = largest.bbox  # col, row, w, h
    return grid[y:y + h, x:x + w].copy()


def _crop_frame_interior(grid: Grid) -> Grid:
    """If grid has a uniform frame border, return the interior."""
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return grid.copy()

    # Check if outermost border is uniform
    border_vals = set()
    border_vals.update(int(v) for v in grid[0, :])
    border_vals.update(int(v) for v in grid[-1, :])
    border_vals.update(int(v) for v in grid[:, 0])
    border_vals.update(int(v) for v in grid[:, -1])

    if len(border_vals) == 1:
        return grid[1:rows - 1, 1:cols - 1].copy()

    return grid.copy()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "crop_to_content",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _crop_to_content,
)

register(
    "crop_to_object",
    OpSignature(params=(("grid", Type.GRID), ("color", Type.COLOR)), return_type=Type.GRID),
    _crop_to_object,
)

register(
    "crop_frame_interior",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _crop_frame_interior,
)
