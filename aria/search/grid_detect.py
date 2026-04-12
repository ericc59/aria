"""Grid structure detection for cell grids.

Detects regular grids from separator positions or implicit
object lattices and enumerates the cells with their positions.
NOT an execution layer — used at derive time to expose grid
structure to search strategies.
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


def cell_center(cell: GridCell) -> tuple[float, float]:
    return (cell.r0 + cell.height / 2.0, cell.c0 + cell.width / 2.0)


def assign_cells_by_nearest(
    sources: list[GridCell],
    targets: list[GridCell],
) -> list[tuple[int, int]] | None:
    """Assign source cells to target cells by nearest centers.

    Requires compatible cell sizes. Returns list of (src_idx, tgt_idx).
    """
    if not sources or not targets:
        return None
    if len(sources) > len(targets):
        return None

    cost = np.zeros((len(sources), len(targets)), dtype=float)
    for si, src in enumerate(sources):
        sr, sc = cell_center(src)
        for ti, tgt in enumerate(targets):
            if src.height != tgt.height or src.width != tgt.width:
                cost[si, ti] = 1e6
                continue
            tr, tc = cell_center(tgt)
            cost[si, ti] = abs(sr - tr) + abs(sc - tc)

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    if any(cost[r, c] >= 1e6 for r, c in zip(rows, cols)):
        return None
    return list(zip(rows.tolist(), cols.tolist()))


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


def _gcd_list(values: list[int]) -> int:
    if not values:
        return 0
    from math import gcd
    g = values[0]
    for v in values[1:]:
        g = gcd(g, v)
    return g


def detect_implicit_grid(facts: GridFacts, *, min_objects: int = 2) -> DetectedGrid | None:
    """Detect a regular grid implied by repeated object placement.

    Uses the most common object (height, width) as the cell template,
    then infers row/col steps from top-left coordinates.
    """
    if not facts.objects:
        return None

    size_groups: dict[tuple[int, int], list] = {}
    for obj in facts.objects:
        size_groups.setdefault((obj.height, obj.width), []).append(obj)

    (cell_h, cell_w), group = max(size_groups.items(), key=lambda kv: len(kv[1]))
    if len(group) < min_objects:
        return None

    rows = sorted({obj.row for obj in group})
    cols = sorted({obj.col for obj in group})
    if len(rows) < 2 or len(cols) < 2:
        return None

    row_diffs = [b - a for a, b in zip(rows, rows[1:]) if b - a > 0]
    col_diffs = [b - a for a, b in zip(cols, cols[1:]) if b - a > 0]
    row_step = _gcd_list(row_diffs)
    col_step = _gcd_list(col_diffs)
    if row_step <= 0 or col_step <= 0:
        return None

    # Build grid positions from min to max by step
    row0 = rows[0]
    col0 = cols[0]
    row_positions = list(range(row0, rows[-1] + 1, row_step))
    col_positions = list(range(col0, cols[-1] + 1, col_step))
    if not row_positions or not col_positions:
        return None

    # Ensure all objects align to grid positions
    row_set = set(row_positions)
    col_set = set(col_positions)
    for obj in group:
        if obj.row not in row_set or obj.col not in col_set:
            return None

    cells = []
    for gi, r0 in enumerate(row_positions):
        for gj, c0 in enumerate(col_positions):
            cells.append(GridCell(
                grid_row=gi, grid_col=gj,
                r0=r0, c0=c0, height=row_step, width=col_step,
            ))

    return DetectedGrid(
        n_rows=len(row_positions),
        n_cols=len(col_positions),
        cell_height=row_step,
        cell_width=col_step,
        sep_color=facts.bg,
        cells=tuple(cells),
    )


def detect_grid(facts: GridFacts) -> DetectedGrid | None:
    """Detect a grid via separators first, then implicit object lattice."""
    grid = detect_separator_grid(facts)
    if grid is not None:
        return grid
    return detect_implicit_grid(facts)


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
