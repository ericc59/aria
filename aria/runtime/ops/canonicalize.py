"""Local canonicalization ops — normalize position/structure before downstream lanes.

  compact_to_origin(grid) -> grid
    Translate all non-background content so its bounding box starts at (0,0).
    Grid is resized to the tight content bbox. This is equivalent to
    crop_to_content but explicitly framed as a normalization step.

  normalize_to_grid(grid) -> grid
    Same as compact_to_origin but preserves the original grid dimensions
    by filling the rest with background. Content is moved to top-left corner.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


def _compact_to_origin(grid: Grid) -> Grid:
    """Translate all non-bg content to start at (0,0), crop to tight bbox."""
    if grid.size == 0:
        return grid.copy()

    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])
    mask = grid != bg

    if not mask.any():
        return grid.copy()

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()


def _normalize_to_grid(grid: Grid) -> Grid:
    """Move all non-bg content to top-left corner, preserve grid dims."""
    if grid.size == 0:
        return grid.copy()

    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])
    mask = grid != bg

    if not mask.any():
        return grid.copy()

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    content = grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    result = np.full_like(grid, bg)
    ch, cw = content.shape
    rh, rw = result.shape
    ph, pw = min(ch, rh), min(cw, rw)
    result[:ph, :pw] = content[:ph, :pw]
    return result


register("compact_to_origin",
         OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
         _compact_to_origin)

register("normalize_to_grid",
         OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
         _normalize_to_grid)
