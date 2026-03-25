"""Multi-resolution zone parsing.

Detects tiling patterns and decomposes grids into rectangular zones
separated by uniform-color borders.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aria.types import Grid, Zone


def detect_tiling(grid: Grid) -> tuple[int, int] | None:
    """Detect if the grid is evenly tiled by identical sub-grids.

    Tries all tile sizes that evenly divide both dimensions, smallest
    first. Returns the first (smallest) tile size found.

    Parameters
    ----------
    grid : Grid
        2D array of color values.

    Returns
    -------
    tuple[int, int] or None
        (tile_rows, tile_cols) if tiling detected, else None.
    """
    rows, cols = grid.shape

    # Collect candidate tile heights and widths (proper divisors)
    tile_hs = [h for h in range(1, rows) if rows % h == 0]
    tile_ws = [w for w in range(1, cols) if cols % w == 0]

    for th in tile_hs:
        for tw in tile_ws:
            tile = grid[:th, :tw]
            if _all_tiles_match(grid, tile, rows, cols, th, tw):
                return (th, tw)

    return None


def _all_tiles_match(
    grid: Grid, tile: Grid, rows: int, cols: int, th: int, tw: int,
) -> bool:
    """Check if every tile-sized block in the grid equals the given tile."""
    for r in range(0, rows, th):
        for c in range(0, cols, tw):
            if not np.array_equal(grid[r:r + th, c:c + tw], tile):
                return False
    return True


def find_zones(grid: Grid) -> list[Zone]:
    """Decompose a grid into rectangular zones separated by uniform borders.

    Looks for full rows and full columns of a single color that act
    as separators. The regions between these separators become zones.

    If no separators are found, returns a single zone covering the
    entire grid.

    Parameters
    ----------
    grid : Grid
        2D array of color values.

    Returns
    -------
    list[Zone]
        List of rectangular zones found in the grid.
    """
    rows, cols = grid.shape

    # Find separator rows: rows where all pixels are the same color
    sep_rows = _find_separator_rows(grid, rows, cols)
    # Find separator cols: cols where all pixels are the same color
    sep_cols = _find_separator_cols(grid, rows, cols)

    # If we found separators, use them to define regions
    if sep_rows or sep_cols:
        return _extract_zones_from_separators(grid, rows, cols, sep_rows, sep_cols)

    # No separators found: whole grid is one zone
    return [Zone(grid=grid.copy(), x=0, y=0, w=cols, h=rows)]


def _find_separator_rows(grid: Grid, rows: int, cols: int) -> list[int]:
    """Find rows where all values are the same color."""
    seps: list[int] = []
    for r in range(rows):
        if np.all(grid[r, :] == grid[r, 0]):
            seps.append(r)
    return _filter_contiguous_separators(seps)


def _find_separator_cols(grid: Grid, rows: int, cols: int) -> list[int]:
    """Find columns where all values are the same color."""
    seps: list[int] = []
    for c in range(cols):
        if np.all(grid[:, c] == grid[0, c]):
            seps.append(c)
    return _filter_contiguous_separators(seps)


def _filter_contiguous_separators(seps: list[int]) -> list[int]:
    """Group contiguous separator indices and return all of them.

    Only return separators if they don't span the entire dimension
    (i.e., not every row/col is a separator).
    """
    if not seps:
        return []
    return seps


def _extract_zones_from_separators(
    grid: Grid,
    rows: int,
    cols: int,
    sep_rows: list[int],
    sep_cols: list[int],
) -> list[Zone]:
    """Cut the grid along separator rows/cols and extract zone sub-grids."""
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    # Find contiguous bands of non-separator rows
    row_bands = _find_bands(rows, sep_row_set)
    col_bands = _find_bands(cols, sep_col_set)

    # If everything is a separator or no bands found, return whole grid
    if not row_bands or not col_bands:
        return [Zone(grid=grid.copy(), x=0, y=0, w=cols, h=rows)]

    zones: list[Zone] = []
    for r_start, r_end in row_bands:
        for c_start, c_end in col_bands:
            sub = grid[r_start:r_end, c_start:c_end].copy()
            zones.append(Zone(
                grid=sub,
                x=c_start,
                y=r_start,
                w=c_end - c_start,
                h=r_end - r_start,
            ))

    return zones


def _find_bands(length: int, sep_set: set[int]) -> list[tuple[int, int]]:
    """Find contiguous ranges of non-separator indices.

    Returns list of (start, end) where end is exclusive.
    """
    bands: list[tuple[int, int]] = []
    i = 0
    while i < length:
        if i in sep_set:
            i += 1
            continue
        start = i
        while i < length and i not in sep_set:
            i += 1
        bands.append((start, i))
    return bands
