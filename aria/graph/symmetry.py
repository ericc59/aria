"""Symmetry detection for object masks and full grids.

Checks rotational and reflective symmetries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aria.types import GlobalSymmetry, Grid, Symmetry


def detect_obj_symmetry(mask: NDArray[np.bool_]) -> frozenset[Symmetry]:
    """Detect symmetries of a boolean mask.

    Checks: 90-degree rotation, 180-degree rotation, horizontal
    reflection, vertical reflection, and main diagonal reflection.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        Boolean mask of shape (h, w).

    Returns
    -------
    frozenset[Symmetry]
        Set of detected symmetries.
    """
    syms: set[Symmetry] = set()

    # Horizontal reflection (flip left-right)
    if np.array_equal(mask, np.fliplr(mask)):
        syms.add(Symmetry.REFL_V)

    # Vertical reflection (flip top-bottom)
    if np.array_equal(mask, np.flipud(mask)):
        syms.add(Symmetry.REFL_H)

    # 180-degree rotation
    if np.array_equal(mask, np.rot90(mask, 2)):
        syms.add(Symmetry.ROT180)

    # 90-degree rotation (requires square mask)
    h, w = mask.shape
    if h == w and np.array_equal(mask, np.rot90(mask, 1)):
        syms.add(Symmetry.ROT90)

    # Diagonal reflection (transpose, only meaningful for square)
    if h == w and np.array_equal(mask, mask.T):
        syms.add(Symmetry.REFL_D)

    return frozenset(syms)


def detect_global_symmetry(grid: Grid) -> frozenset[GlobalSymmetry]:
    """Detect global symmetries of a grid.

    Checks: global rotation (90 or 180), global reflection (any axis),
    and periodicity (tiling).

    Parameters
    ----------
    grid : Grid
        2D array of color values.

    Returns
    -------
    frozenset[GlobalSymmetry]
        Set of detected global symmetries.
    """
    syms: set[GlobalSymmetry] = set()
    rows, cols = grid.shape

    # Global reflection: horizontal or vertical
    if np.array_equal(grid, np.fliplr(grid)):
        syms.add(GlobalSymmetry.GLOBAL_REFL)
    elif np.array_equal(grid, np.flipud(grid)):
        syms.add(GlobalSymmetry.GLOBAL_REFL)

    # Global rotation: 180 always works, 90 requires square
    if np.array_equal(grid, np.rot90(grid, 2)):
        syms.add(GlobalSymmetry.GLOBAL_ROT)
    elif rows == cols and np.array_equal(grid, np.rot90(grid, 1)):
        syms.add(GlobalSymmetry.GLOBAL_ROT)

    # Periodicity: check small tile sizes
    if _has_periodicity(grid, rows, cols):
        syms.add(GlobalSymmetry.PERIODIC)

    return frozenset(syms)


def _has_periodicity(grid: Grid, rows: int, cols: int) -> bool:
    """Check if the grid is periodic (tiles perfectly with a smaller sub-grid)."""
    # Try all tile sizes that evenly divide the grid
    for th in range(1, rows // 2 + 1):
        if rows % th != 0:
            continue
        for tw in range(1, cols // 2 + 1):
            if cols % tw != 0:
                continue
            # At least one dimension must actually repeat
            if th == rows and tw == cols:
                continue
            tile = grid[:th, :tw]
            if _tiles_match(grid, tile, rows, cols, th, tw):
                return True
    return False


def _tiles_match(
    grid: Grid,
    tile: Grid,
    rows: int,
    cols: int,
    th: int,
    tw: int,
) -> bool:
    """Check if the entire grid is tiled by the given tile."""
    for r in range(0, rows, th):
        for c in range(0, cols, tw):
            if not np.array_equal(grid[r:r + th, c:c + tw], tile):
                return False
    return True
