"""Separator-aware partition detection for panel/grid tasks."""

from __future__ import annotations

import numpy as np

from aria.graph.background import detect_bg
from aria.graph.cc_label import label_4conn
from aria.types import Grid, PartitionCell, PartitionScene


def detect_partition(grid: Grid, *, background: int | None = None) -> PartitionScene | None:
    """Detect grids split by uniform separator rows/cols into cell panels."""
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return None

    bg = detect_bg(grid) if background is None else int(background)
    row_candidates: dict[int, list[int]] = {}
    col_candidates: dict[int, list[int]] = {}

    for row in range(rows):
        values = {int(value) for value in grid[row, :]}
        if len(values) == 1:
            color = next(iter(values))
            row_candidates.setdefault(color, []).append(row)

    for col in range(cols):
        values = {int(value) for value in grid[:, col]}
        if len(values) == 1:
            color = next(iter(values))
            col_candidates.setdefault(color, []).append(col)

    best: PartitionScene | None = None
    for color in sorted(set(row_candidates) & set(col_candidates)):
        scene = _try_partition(
            grid,
            sep_color=color,
            sep_rows=row_candidates[color],
            sep_cols=col_candidates[color],
            background=bg,
        )
        if scene is None:
            continue
        if best is None or len(scene.cells) > len(best.cells):
            best = scene

    return best


def _try_partition(
    grid: Grid,
    *,
    sep_color: int,
    sep_rows: list[int],
    sep_cols: list[int],
    background: int,
) -> PartitionScene | None:
    rows, cols = grid.shape

    for row in sep_rows:
        for col in sep_cols:
            if int(grid[row, col]) != sep_color:
                return None

    row_intervals = _compute_intervals(sep_rows, rows)
    col_intervals = _compute_intervals(sep_cols, cols)
    if not row_intervals or not col_intervals:
        return None
    if len(row_intervals) * len(col_intervals) < 2:
        return None

    inner_sep_rows = [row for row in sep_rows if 0 < row < rows - 1]
    inner_sep_cols = [col for col in sep_cols if 0 < col < cols - 1]
    if not inner_sep_rows and not inner_sep_cols:
        return None

    cells: list[PartitionCell] = []
    cell_shapes: list[tuple[int, int]] = []
    for row_idx, (r0, r1) in enumerate(row_intervals):
        for col_idx, (c0, c1) in enumerate(col_intervals):
            cell_grid = grid[r0 : r1 + 1, c0 : c1 + 1]
            if cell_grid.size == 0:
                return None
            cell_bg = background
            objects = label_4conn(cell_grid, ignore_color=cell_bg)
            cell_shapes.append((int(cell_grid.shape[0]), int(cell_grid.shape[1])))
            cells.append(PartitionCell(
                row_idx=row_idx,
                col_idx=col_idx,
                bbox=(r0, c0, r1, c1),
                dims=(int(cell_grid.shape[0]), int(cell_grid.shape[1])),
                background=int(cell_bg),
                palette=frozenset(int(value) for value in np.unique(cell_grid)),
                obj_count=len(objects),
            ))

    is_uniform = len(set(cell_shapes)) == 1
    if sep_color == background:
        separator_support = _mean_separator_support(
            grid,
            inner_sep_rows=inner_sep_rows,
            inner_sep_cols=inner_sep_cols,
            background=background,
        )
        if not is_uniform and len(set(cell_shapes)) > 2 and separator_support < 0.35:
            return None

        nonempty_cells = [cell for cell in cells if cell.obj_count > 0]
        sparse_marker_cells = (
            nonempty_cells
            and len(nonempty_cells) <= 2
            and all(cell.obj_count <= 1 for cell in nonempty_cells)
            and all((cell.dims[0] * cell.dims[1]) <= 2 for cell in nonempty_cells)
        )
        if sparse_marker_cells and separator_support < 0.35:
            return None

    return PartitionScene(
        separator_color=int(sep_color),
        separator_rows=tuple(sorted(sep_rows)),
        separator_cols=tuple(sorted(sep_cols)),
        cells=tuple(cells),
        n_rows=len(row_intervals),
        n_cols=len(col_intervals),
        cell_shapes=tuple(cell_shapes),
        is_uniform_partition=is_uniform,
    )


def _compute_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    seps = sorted(set(separators))
    if not seps:
        return [(0, size - 1)]

    intervals: list[tuple[int, int]] = []
    if seps[0] > 0:
        intervals.append((0, seps[0] - 1))

    for idx in range(len(seps) - 1):
        start = seps[idx] + 1
        end = seps[idx + 1] - 1
        if start <= end:
            intervals.append((start, end))

    if seps[-1] < size - 1:
        intervals.append((seps[-1] + 1, size - 1))

    return intervals


def _mean_separator_support(
    grid: Grid,
    *,
    inner_sep_rows: list[int],
    inner_sep_cols: list[int],
    background: int,
) -> float:
    supports: list[float] = []

    for row in inner_sep_rows:
        above = grid[row - 1, :] != background
        below = grid[row + 1, :] != background
        supports.append(float(np.mean(np.logical_or(above, below))))

    for col in inner_sep_cols:
        left = grid[:, col - 1] != background
        right = grid[:, col + 1] != background
        supports.append(float(np.mean(np.logical_or(left, right))))

    if not supports:
        return 1.0
    return float(np.mean(supports))
