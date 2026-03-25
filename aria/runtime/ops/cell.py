"""Cell-level computation ops.

These operate at the individual cell level — per-pixel transforms, neighborhood
analysis, pattern matching, and conditional fills. This closes the expressiveness
gap for tasks that require reasoning about individual cells and their local context.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

from aria.types import Grid, Type, make_grid
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Neighborhood helpers
# ---------------------------------------------------------------------------

def _neighbors_4(grid: Grid, r: int, c: int) -> list[int]:
    """4-connected neighbors (up, down, left, right). Returns list of color values."""
    rows, cols = grid.shape
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append(int(grid[nr, nc]))
    return result


def _neighbors_8(grid: Grid, r: int, c: int) -> list[int]:
    """8-connected neighbors. Returns list of color values."""
    rows, cols = grid.shape
    result = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append(int(grid[nr, nc]))
    return result


# ---------------------------------------------------------------------------
# Cell-level map operations
# ---------------------------------------------------------------------------

def _cell_map(grid: Grid, fn: Callable) -> Grid:
    """Apply fn(row, col, value) to every cell. Returns new grid.

    fn receives (row: int, col: int, val: int) and returns an int (color).
    """
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            result[r, c] = int(fn(r, c, int(grid[r, c])))
    return result


def _neighbor_map(grid: Grid, fn: Callable) -> Grid:
    """Apply fn(value, neighbor_colors_4) to every cell. Returns new grid.

    fn receives (val: int, neighbors: list[int]) and returns an int.
    """
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            nbrs = _neighbors_4(grid, r, c)
            result[r, c] = int(fn(int(grid[r, c]), nbrs))
    return result


def _neighbor_map_8(grid: Grid, fn: Callable) -> Grid:
    """Apply fn(value, neighbor_colors_8) to every cell. Returns new grid."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            nbrs = _neighbors_8(grid, r, c)
            result[r, c] = int(fn(int(grid[r, c]), nbrs))
    return result


# ---------------------------------------------------------------------------
# Conditional fills
# ---------------------------------------------------------------------------

def _conditional_fill(grid: Grid, color: int, target_color: int) -> Grid:
    """Replace all cells of target_color with color. Simple but common."""
    result = grid.copy()
    result[grid == target_color] = color
    return result


def _fill_where_neighbor_count(grid: Grid, neighbor_color: int,
                                min_count: int, fill_color: int) -> Grid:
    """Fill cells where the count of 4-connected neighbors matching
    neighbor_color is >= min_count."""
    rows, cols = grid.shape
    result = grid.copy()
    for r in range(rows):
        for c in range(cols):
            nbrs = _neighbors_4(grid, r, c)
            if sum(1 for n in nbrs if n == neighbor_color) >= min_count:
                result[r, c] = fill_color
    return result


def _fill_between(grid: Grid, color: int, fill_color: int) -> Grid:
    """Fill cells that are horizontally or vertically between two cells of `color`.

    For each row, find leftmost and rightmost `color` cell, fill between them.
    Same for each column. Common ARC pattern: "connect the dots."
    """
    result = grid.copy()
    rows, cols = grid.shape

    # Horizontal fill
    for r in range(rows):
        positions = [c for c in range(cols) if grid[r, c] == color]
        if len(positions) >= 2:
            for c in range(positions[0] + 1, positions[-1]):
                if result[r, c] != color:
                    result[r, c] = fill_color

    # Vertical fill
    for c in range(cols):
        positions = [r for r in range(rows) if grid[r, c] == color]
        if len(positions) >= 2:
            for r in range(positions[0] + 1, positions[-1]):
                if result[r, c] != color:
                    result[r, c] = fill_color

    return result


def _propagate(grid: Grid, source_color: int, fill_color: int,
               bg_color: int = 0) -> Grid:
    """BFS propagation from all cells of source_color through bg_color cells.

    Fills reachable bg_color cells with fill_color. Stops at non-bg cells.
    Like flood fill but from multiple sources simultaneously.
    """
    rows, cols = grid.shape
    result = grid.copy()
    queue: deque[tuple[int, int]] = deque()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == source_color:
                queue.append((r, c))

    visited = set()
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr, nc] == bg_color):
                visited.add((nr, nc))
                result[nr, nc] = fill_color
                queue.append((nr, nc))

    return result


# ---------------------------------------------------------------------------
# Pattern matching and replacement
# ---------------------------------------------------------------------------

def _find_pattern(grid: Grid, pattern: Grid) -> list[tuple[int, int]]:
    """Find all positions where pattern matches in grid.

    A 0 in the pattern is a wildcard (matches anything).
    Returns list of (row, col) top-left positions.
    """
    gr, gc = grid.shape
    pr, pc = pattern.shape
    matches: list[tuple[int, int]] = []

    for r in range(gr - pr + 1):
        for c in range(gc - pc + 1):
            match = True
            for dr in range(pr):
                for dc in range(pc):
                    pval = int(pattern[dr, dc])
                    if pval != 0 and pval != int(grid[r + dr, c + dc]):
                        match = False
                        break
                if not match:
                    break
            if match:
                matches.append((r, c))

    return matches


def _replace_pattern(grid: Grid, pattern: Grid,
                     replacement: Grid) -> Grid:
    """Find pattern in grid and replace with replacement.

    Pattern 0s are wildcards for matching. Replacement 0s are transparent
    (don't overwrite). Replaces ALL occurrences.
    """
    result = grid.copy()
    positions = _find_pattern(grid, pattern)
    pr, pc = replacement.shape

    for r, c in positions:
        for dr in range(pr):
            for dc in range(pc):
                rval = int(replacement[dr, dc])
                if rval != 0:
                    if r + dr < result.shape[0] and c + dc < result.shape[1]:
                        result[r + dr, c + dc] = rval

    return result


# ---------------------------------------------------------------------------
# Symmetry completion
# ---------------------------------------------------------------------------

def _complete_symmetry_h(grid: Grid) -> Grid:
    """Complete horizontal symmetry: mirror left half to right (or vice versa).

    Copies non-bg cells from whichever half has more content.
    """
    rows, cols = grid.shape
    result = grid.copy()
    mid = cols // 2

    # Count non-zero in each half
    left_count = np.count_nonzero(grid[:, :mid])
    right_count = np.count_nonzero(grid[:, mid + (cols % 2):])

    if left_count >= right_count:
        # Mirror left to right
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                if result[r, c] != 0 and result[r, mirror_c] == 0:
                    result[r, mirror_c] = result[r, c]
    else:
        # Mirror right to left
        for r in range(rows):
            for c in range(mid + (cols % 2), cols):
                mirror_c = cols - 1 - c
                if result[r, c] != 0 and result[r, mirror_c] == 0:
                    result[r, mirror_c] = result[r, c]

    return result


def _complete_symmetry_v(grid: Grid) -> Grid:
    """Complete vertical symmetry: mirror top half to bottom (or vice versa)."""
    rows, cols = grid.shape
    result = grid.copy()
    mid = rows // 2

    top_count = np.count_nonzero(grid[:mid, :])
    bot_count = np.count_nonzero(grid[mid + (rows % 2):, :])

    if top_count >= bot_count:
        for r in range(mid):
            mirror_r = rows - 1 - r
            for c in range(cols):
                if result[r, c] != 0 and result[mirror_r, c] == 0:
                    result[mirror_r, c] = result[r, c]
    else:
        for r in range(mid + (rows % 2), rows):
            mirror_r = rows - 1 - r
            for c in range(cols):
                if result[r, c] != 0 and result[mirror_r, c] == 0:
                    result[mirror_r, c] = result[r, c]

    return result


# ---------------------------------------------------------------------------
# Grid arithmetic (cell-wise)
# ---------------------------------------------------------------------------

def _grid_and(a: Grid, b: Grid) -> Grid:
    """Cell-wise AND: result cell is non-zero only where both a and b are non-zero.
    Takes the color from a."""
    if a.shape != b.shape:
        raise ValueError(f"grid_and: shape mismatch {a.shape} vs {b.shape}")
    result = a.copy()
    result[b == 0] = 0
    return result


def _grid_or(a: Grid, b: Grid) -> Grid:
    """Cell-wise OR: non-zero cells from a, then fill gaps from b."""
    if a.shape != b.shape:
        raise ValueError(f"grid_or: shape mismatch {a.shape} vs {b.shape}")
    result = a.copy()
    gaps = result == 0
    result[gaps] = b[gaps]
    return result


def _grid_xor(a: Grid, b: Grid) -> Grid:
    """Cell-wise XOR: cells that differ between a and b."""
    if a.shape != b.shape:
        raise ValueError(f"grid_xor: shape mismatch {a.shape} vs {b.shape}")
    result = np.zeros_like(a)
    diff = a != b
    # Where they differ, take the non-zero value (or a's value)
    result[diff] = np.where(a[diff] != 0, a[diff], b[diff])
    return result


def _grid_diff(a: Grid, b: Grid) -> Grid:
    """Cells that are in a but not in b (a minus b). Non-zero in a, zero in b."""
    if a.shape != b.shape:
        raise ValueError(f"grid_diff: shape mismatch {a.shape} vs {b.shape}")
    result = a.copy()
    result[b != 0] = 0
    return result


# ---------------------------------------------------------------------------
# Row/column operations
# ---------------------------------------------------------------------------

def _most_common_color(grid: Grid) -> int:
    """Most common non-zero color in the grid."""
    nonzero = grid[grid != 0]
    if len(nonzero) == 0:
        return 0
    vals, counts = np.unique(nonzero, return_counts=True)
    return int(vals[np.argmax(counts)])


def _count_color(grid: Grid, color: int) -> int:
    """Count cells of a given color."""
    return int(np.sum(grid == color))


def _get_row(grid: Grid, idx: int) -> Grid:
    """Extract a single row as a 1xN grid."""
    return grid[idx:idx + 1, :].copy()


def _get_col(grid: Grid, idx: int) -> Grid:
    """Extract a single column as an Nx1 grid."""
    return grid[:, idx:idx + 1].copy()


def _set_row(grid: Grid, idx: int, row: Grid) -> Grid:
    """Set a row in the grid. row should be 1xN or a flat array."""
    result = grid.copy()
    if row.ndim == 2:
        result[idx, :row.shape[1]] = row[0, :]
    else:
        result[idx, :len(row)] = row
    return result


def _set_col(grid: Grid, idx: int, col: Grid) -> Grid:
    """Set a column in the grid."""
    result = grid.copy()
    if col.ndim == 2:
        result[:col.shape[0], idx] = col[:, 0]
    else:
        result[:len(col), idx] = col
    return result


def _sort_rows(grid: Grid) -> Grid:
    """Sort rows by their content (lexicographic on non-zero cells)."""
    rows = [grid[r, :] for r in range(grid.shape[0])]
    rows.sort(key=lambda r: tuple(int(x) for x in r))
    return np.stack(rows).astype(np.uint8)


def _sort_cols(grid: Grid) -> Grid:
    """Sort columns by their content."""
    cols = [grid[:, c] for c in range(grid.shape[1])]
    cols.sort(key=lambda c: tuple(int(x) for x in c))
    return np.stack(cols, axis=1).astype(np.uint8)


def _unique_rows(grid: Grid) -> Grid:
    """Remove duplicate rows."""
    _, idx = np.unique(grid, axis=0, return_index=True)
    return grid[np.sort(idx)].astype(np.uint8)


def _unique_cols(grid: Grid) -> Grid:
    """Remove duplicate columns."""
    _, idx = np.unique(grid, axis=1, return_index=True)
    return grid[:, np.sort(idx)].astype(np.uint8)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Cell-level maps
register("cell_map", OpSignature(
    params=(("grid", Type.GRID), ("fn", Type.CALLABLE)),
    return_type=Type.GRID), _cell_map)

register("neighbor_map", OpSignature(
    params=(("grid", Type.GRID), ("fn", Type.CALLABLE)),
    return_type=Type.GRID), _neighbor_map)

register("neighbor_map_8", OpSignature(
    params=(("grid", Type.GRID), ("fn", Type.CALLABLE)),
    return_type=Type.GRID), _neighbor_map_8)

# Conditional fills
register("conditional_fill", OpSignature(
    params=(("grid", Type.GRID), ("color", Type.COLOR), ("target", Type.COLOR)),
    return_type=Type.GRID), _conditional_fill)

register("fill_where_neighbor_count", OpSignature(
    params=(("grid", Type.GRID), ("neighbor_color", Type.COLOR),
            ("min_count", Type.INT), ("fill_color", Type.COLOR)),
    return_type=Type.GRID), _fill_where_neighbor_count)

register("fill_between", OpSignature(
    params=(("grid", Type.GRID), ("color", Type.COLOR), ("fill_color", Type.COLOR)),
    return_type=Type.GRID), _fill_between)

register("propagate", OpSignature(
    params=(("grid", Type.GRID), ("source_color", Type.COLOR),
            ("fill_color", Type.COLOR), ("bg_color", Type.COLOR)),
    return_type=Type.GRID), _propagate)

# Pattern matching
register("find_pattern", OpSignature(
    params=(("grid", Type.GRID), ("pattern", Type.GRID)),
    return_type=Type.INT_LIST), _find_pattern)

register("replace_pattern", OpSignature(
    params=(("grid", Type.GRID), ("pattern", Type.GRID), ("replacement", Type.GRID)),
    return_type=Type.GRID), _replace_pattern)

# Symmetry completion
register("complete_symmetry_h", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _complete_symmetry_h)

register("complete_symmetry_v", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _complete_symmetry_v)

# Grid arithmetic
register("grid_and", OpSignature(
    params=(("a", Type.GRID), ("b", Type.GRID)), return_type=Type.GRID), _grid_and)

register("grid_or", OpSignature(
    params=(("a", Type.GRID), ("b", Type.GRID)), return_type=Type.GRID), _grid_or)

register("grid_xor", OpSignature(
    params=(("a", Type.GRID), ("b", Type.GRID)), return_type=Type.GRID), _grid_xor)

register("grid_diff", OpSignature(
    params=(("a", Type.GRID), ("b", Type.GRID)), return_type=Type.GRID), _grid_diff)

# Color counting
register("most_common_color", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.COLOR), _most_common_color)

register("count_color", OpSignature(
    params=(("grid", Type.GRID), ("color", Type.COLOR)), return_type=Type.INT), _count_color)

# Row/column ops
register("get_row", OpSignature(
    params=(("grid", Type.GRID), ("idx", Type.INT)), return_type=Type.GRID), _get_row)

register("get_col", OpSignature(
    params=(("grid", Type.GRID), ("idx", Type.INT)), return_type=Type.GRID), _get_col)

register("set_row", OpSignature(
    params=(("grid", Type.GRID), ("idx", Type.INT), ("row", Type.GRID)),
    return_type=Type.GRID), _set_row)

register("set_col", OpSignature(
    params=(("grid", Type.GRID), ("idx", Type.INT), ("col", Type.GRID)),
    return_type=Type.GRID), _set_col)

register("sort_rows", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _sort_rows)

register("sort_cols", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _sort_cols)

register("unique_rows", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _unique_rows)

register("unique_cols", OpSignature(
    params=(("grid", Type.GRID),), return_type=Type.GRID), _unique_cols)
