"""Scene-level geometric transforms and partition cell selection.

These ops represent reusable world-model operations:
- geometric transforms (rot90, flip, transpose)
- partition cell selection by property
"""

from __future__ import annotations

import numpy as np

from aria.core.grid_perception import perceive_grid
from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

TRANSFORM_CODES = {
    0: "rot90",
    1: "rot180",
    2: "rot270",
    3: "flip_lr",
    4: "flip_ud",
    5: "transpose",
}


def _apply_geometric_transform(grid: Grid, transform_code: int) -> Grid:
    """Apply a geometric transform to the grid."""
    if transform_code == 0:
        return np.rot90(grid, 1)
    elif transform_code == 1:
        return np.rot90(grid, 2)
    elif transform_code == 2:
        return np.rot90(grid, 3)
    elif transform_code == 3:
        return np.fliplr(grid).copy()
    elif transform_code == 4:
        return np.flipud(grid).copy()
    elif transform_code == 5:
        return grid.T.copy()
    return grid.copy()


register(
    "apply_geometric_transform",
    OpSignature(
        params=(("grid", Type.GRID), ("transform_code", Type.INT)),
        return_type=Type.GRID,
    ),
    _apply_geometric_transform,
)


# ---------------------------------------------------------------------------
# Partition cell selection by property
# ---------------------------------------------------------------------------

CELL_SELECTOR_CODES = {
    0: "most_non_bg",
    1: "fewest_non_bg_gt0",
    2: "most_colors",
    3: "unique_non_empty",
}


def _select_partition_cell_by_property(
    grid: Grid,
    selector_code: int,
) -> Grid:
    """Select a partition cell by a structural property and return it."""
    state = perceive_grid(grid)
    p = state.partition
    if p is None or len(p.cells) < 2:
        return grid.copy()

    bg = state.bg_color
    cell_infos = []
    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        cg = grid[r0 : r1 + 1, c0 : c1 + 1]
        non_bg = cg[cg != bg]
        n_non_bg = len(non_bg)
        n_colors = len(set(int(v) for v in non_bg)) if n_non_bg > 0 else 0
        cell_infos.append((cell, cg, n_non_bg, n_colors))

    selected = None
    if selector_code == 0:  # most_non_bg
        selected = max(cell_infos, key=lambda x: x[2])
    elif selector_code == 1:  # fewest_non_bg_gt0
        candidates = [ci for ci in cell_infos if ci[2] > 0]
        if candidates:
            selected = min(candidates, key=lambda x: x[2])
    elif selector_code == 2:  # most_colors
        selected = max(cell_infos, key=lambda x: x[3])
    elif selector_code == 3:  # unique_non_empty
        non_empty = [ci for ci in cell_infos if ci[2] > 0]
        if len(non_empty) == 1:
            selected = non_empty[0]

    if selected is None:
        return grid.copy()
    return selected[1].copy()


register(
    "select_partition_cell_by_property",
    OpSignature(
        params=(("grid", Type.GRID), ("selector_code", Type.INT)),
        return_type=Type.GRID,
    ),
    _select_partition_cell_by_property,
)
