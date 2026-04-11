"""Grid structure detection for separator-defined cell grids.

Detects regular grids from separator positions and enumerates
the cells with their positions. NOT an execution layer — used
at derive time to expose grid structure to search strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.guided.perceive import GridFacts


@dataclass(frozen=True)
class GridCell:
    """One cell in a detected grid."""
    grid_row: int      # row index in the grid (0-based)
    grid_col: int      # col index in the grid (0-based)
    r0: int            # top pixel row
    c0: int            # left pixel col
    height: int        # cell height in pixels
    width: int         # cell width in pixels


@dataclass(frozen=True)
class DetectedGrid:
    """A regular grid detected from separator positions."""
    n_rows: int        # number of cell rows
    n_cols: int        # number of cell columns
    cell_height: int   # uniform cell height (0 if variable)
    cell_width: int    # uniform cell width (0 if variable)
    sep_color: int     # separator color
    cells: tuple[GridCell, ...]

    def cell_at(self, grid_row: int, grid_col: int) -> GridCell | None:
        for c in self.cells:
            if c.grid_row == grid_row and c.grid_col == grid_col:
                return c
        return None


def detect_separator_grid(facts: GridFacts) -> DetectedGrid | None:
    """Detect a regular grid from row/col separators.

    Requires at least one row separator OR one col separator.
    All separators must share the same color.
    Cell sizes must be uniform (within 1 pixel tolerance).

    Returns None if no regular grid is found.
    """
    if not facts.separators:
        return None

    # Group separators by color
    sep_colors = set(s.color for s in facts.separators)
    if len(sep_colors) != 1:
        return None
    sep_color = next(iter(sep_colors))

    row_seps = sorted([s.index for s in facts.separators if s.axis == 'row'])
    col_seps = sorted([s.index for s in facts.separators if s.axis == 'col'])

    # Compute row boundaries
    row_bounds = [0]
    for s in row_seps:
        row_bounds.append(s)
        row_bounds.append(s + 1)
    row_bounds.append(facts.rows)

    # Compute col boundaries
    col_bounds = [0]
    for s in col_seps:
        col_bounds.append(s)
        col_bounds.append(s + 1)
    col_bounds.append(facts.cols)

    # Extract cell regions (between separators)
    cell_rows = []
    i = 0
    while i < len(row_bounds) - 1:
        r0 = row_bounds[i]
        r1 = row_bounds[i + 1]
        if r1 > r0:
            # Skip if this region is a separator row
            if r0 in row_seps:
                i += 1
                continue
            cell_rows.append((r0, r1 - r0))
        i += 1

    cell_cols = []
    i = 0
    while i < len(col_bounds) - 1:
        c0 = col_bounds[i]
        c1 = col_bounds[i + 1]
        if c1 > c0:
            if c0 in col_seps:
                i += 1
                continue
            cell_cols.append((c0, c1 - c0))
        i += 1

    if not cell_rows or not cell_cols:
        return None

    # Check uniformity
    heights = [h for _, h in cell_rows]
    widths = [w for _, w in cell_cols]
    uniform_h = max(heights) - min(heights) <= 1
    uniform_w = max(widths) - min(widths) <= 1

    if not uniform_h or not uniform_w:
        return None

    # Build cells
    cells = []
    for gi, (r0, h) in enumerate(cell_rows):
        for gj, (c0, w) in enumerate(cell_cols):
            cells.append(GridCell(
                grid_row=gi, grid_col=gj,
                r0=r0, c0=c0, height=h, width=w,
            ))

    return DetectedGrid(
        n_rows=len(cell_rows),
        n_cols=len(cell_cols),
        cell_height=heights[0],
        cell_width=widths[0],
        sep_color=sep_color,
        cells=tuple(cells),
    )


def cell_content(grid: np.ndarray, cell: GridCell, bg: int) -> np.ndarray:
    """Extract the content of a cell (the sub-grid)."""
    return grid[cell.r0:cell.r0 + cell.height,
                cell.c0:cell.c0 + cell.width].copy()


def cell_has_content(grid: np.ndarray, cell: GridCell, bg: int) -> bool:
    """Check if a cell has any non-bg pixels."""
    sub = grid[cell.r0:cell.r0 + cell.height, cell.c0:cell.c0 + cell.width]
    return bool(np.any(sub != bg))


def cell_content_color(grid: np.ndarray, cell: GridCell, bg: int) -> int | None:
    """Return the dominant non-bg color in a cell, or None if empty."""
    sub = grid[cell.r0:cell.r0 + cell.height, cell.c0:cell.c0 + cell.width]
    non_bg = sub[sub != bg]
    if len(non_bg) == 0:
        return None
    from collections import Counter
    return int(Counter(non_bg.tolist()).most_common(1)[0][0])
