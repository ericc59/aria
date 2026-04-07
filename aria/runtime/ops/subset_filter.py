"""Selective keep/erase ops — filter objects by predicate before downstream lanes.

  keep_by_min_size(grid, threshold) -> grid
    Keep objects with size > threshold, erase smaller ones to background.

  erase_by_max_size(grid, threshold) -> grid
    Erase objects with size <= threshold to background, keep larger ones.

  erase_color(grid, color) -> grid
    Erase all pixels of the given color to background.

  keep_color(grid, color) -> grid
    Keep only pixels of the given color (plus background), erase others.

These are general pre-processing transforms usable as stage-1 in compositions.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.selection import _find_objects
from aria.types import Grid, Type


def _detect_bg(grid: Grid) -> int:
    unique, counts = np.unique(grid, return_counts=True)
    return int(unique[np.argmax(counts)])


def _keep_by_min_size(grid: Grid, threshold: int) -> Grid:
    """Keep objects with size > threshold, erase smaller to bg."""
    bg = _detect_bg(grid)
    result = np.full_like(grid, bg)
    for obj in _find_objects(grid):
        if obj.color == bg:
            continue
        if obj.size > threshold:
            x, y, w, h = obj.bbox
            mask = obj.mask
            for dr in range(mask.shape[0]):
                for dc in range(mask.shape[1]):
                    if mask[dr, dc]:
                        result[y + dr, x + dc] = obj.color
    return result


def _erase_by_max_size(grid: Grid, threshold: int) -> Grid:
    """Erase objects with size <= threshold to bg, keep larger."""
    bg = _detect_bg(grid)
    result = grid.copy()
    for obj in _find_objects(grid):
        if obj.color == bg:
            continue
        if obj.size <= threshold:
            x, y, w, h = obj.bbox
            mask = obj.mask
            for dr in range(mask.shape[0]):
                for dc in range(mask.shape[1]):
                    if mask[dr, dc]:
                        result[y + dr, x + dc] = bg
    return result


def _erase_color(grid: Grid, color: int) -> Grid:
    """Erase all pixels of the given color to background."""
    bg = _detect_bg(grid)
    result = grid.copy()
    result[result == color] = bg
    return result


def _keep_color(grid: Grid, color: int) -> Grid:
    """Keep only pixels of the given color, erase others to background."""
    bg = _detect_bg(grid)
    result = np.full_like(grid, bg)
    result[grid == color] = color
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register("keep_by_min_size",
         OpSignature(params=(("grid", Type.GRID), ("threshold", Type.INT)), return_type=Type.GRID),
         _keep_by_min_size)

register("erase_by_max_size",
         OpSignature(params=(("grid", Type.GRID), ("threshold", Type.INT)), return_type=Type.GRID),
         _erase_by_max_size)

register("erase_color",
         OpSignature(params=(("grid", Type.GRID), ("color", Type.COLOR)), return_type=Type.GRID),
         _erase_color)

register("keep_color",
         OpSignature(params=(("grid", Type.GRID), ("color", Type.COLOR)), return_type=Type.GRID),
         _keep_color)
