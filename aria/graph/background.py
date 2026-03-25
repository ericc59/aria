"""Background color detection.

Uses frequency analysis with a border heuristic as tiebreaker.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def detect_bg(grid: NDArray[np.uint8]) -> int:
    """Detect the background color of a grid.

    Strategy:
    1. Count frequency of each color across the entire grid.
    2. If there is a unique most-frequent color, return it.
    3. On ties, prefer the color most frequent on the grid border.
    4. If still tied, return the smallest color value.

    Parameters
    ----------
    grid : Grid
        2D array of color values (0-9).

    Returns
    -------
    int
        The detected background color.
    """
    rows, cols = grid.shape

    # Global frequency (colors are 0-9)
    counts = np.bincount(grid.ravel(), minlength=10)
    max_count = counts.max()
    candidates = np.where(counts == max_count)[0]

    if len(candidates) == 1:
        return int(candidates[0])

    # Tiebreak: border frequency
    border_pixels = _extract_border(grid, rows, cols)
    border_counts = np.bincount(border_pixels, minlength=10)

    best_color = int(candidates[0])
    best_border = border_counts[best_color]

    for c in candidates[1:]:
        c = int(c)
        bc = border_counts[c]
        if bc > best_border or (bc == best_border and c < best_color):
            best_color = c
            best_border = bc

    return best_color


def _extract_border(
    grid: NDArray[np.uint8], rows: int, cols: int
) -> NDArray[np.uint8]:
    """Return a 1D array of all border pixel values (no duplicates of position)."""
    if rows <= 2 or cols <= 2:
        # Entire grid is border
        return grid.ravel()

    top = grid[0, :]
    bottom = grid[rows - 1, :]
    left = grid[1:rows - 1, 0]
    right = grid[1:rows - 1, cols - 1]
    return np.concatenate([top, bottom, left, right])
