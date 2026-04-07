"""Zone/partition summary grid runtime op.

Renders a summary grid where each cell is a scalar property of a zone
or partition cell (dominant color, count, etc.). Used by stage-1 when
a verified summary mapping is found.
"""

from __future__ import annotations

import numpy as np

from aria.core.grid_perception import perceive_grid
from aria.core.relations import (
    _ZONE_PROPERTY_EXTRACTORS,
    _match_zone_order_to_grid_layout,
)
from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


_PROPERTY_INDEX = {name: idx for idx, (name, _) in enumerate(_ZONE_PROPERTY_EXTRACTORS)}

# source codes: 0 = partition, 1 = zone
SOURCE_PARTITION = 0
SOURCE_ZONE = 1


def _render_zone_summary_grid(
    grid: Grid,
    property_code: int,
    source_code: int,
) -> Grid:
    """Render a summary grid from zone or partition cell properties.

    Args:
        grid: input grid
        property_code: index into _ZONE_PROPERTY_EXTRACTORS
        source_code: 0 = partition cells, 1 = zones
    """
    if property_code < 0 or property_code >= len(_ZONE_PROPERTY_EXTRACTORS):
        return grid.copy()

    _prop_name, extractor = _ZONE_PROPERTY_EXTRACTORS[property_code]
    state = perceive_grid(grid)

    if source_code == SOURCE_PARTITION:
        return _render_from_partition(state, extractor)
    elif source_code == SOURCE_ZONE:
        return _render_from_zones(state, extractor)
    return grid.copy()


def _render_from_partition(state, extractor):
    p = state.partition
    if p is None:
        return np.zeros((1, 1), dtype=np.int32)

    cell_map = {}
    for cell in p.cells:
        cell_map[(cell.row_idx, cell.col_idx)] = cell

    result = np.zeros((p.n_rows, p.n_cols), dtype=np.int32)
    for ri in range(p.n_rows):
        for ci in range(p.n_cols):
            cell = cell_map.get((ri, ci))
            if cell is None:
                continue
            r0, c0, r1, c1 = cell.bbox
            cell_grid = state.grid[r0 : r1 + 1, c0 : c1 + 1]
            val = extractor(cell_grid, state.bg_color)
            if val is not None:
                result[ri, ci] = val

    return result


def _render_from_zones(state, extractor):
    zones = state.zones
    content_zones = [z for z in zones if z.h * z.w < state.dims[0] * state.dims[1]]
    n = len(content_zones)
    if n < 2:
        return np.zeros((1, 1), dtype=np.int32)

    # Try all grid layouts
    for r in range(1, n + 1):
        if n % r != 0:
            continue
        c = n // r
        zone_order = _match_zone_order_to_grid_layout(content_zones, r, c)
        if zone_order is None:
            continue
        result = np.zeros((r, c), dtype=np.int32)
        for idx, zi in enumerate(zone_order):
            z = content_zones[zi]
            ri, ci = divmod(idx, c)
            val = extractor(z.grid, state.bg_color)
            if val is not None:
                result[ri, ci] = val
        return result

    return np.zeros((1, 1), dtype=np.int32)


register(
    "render_zone_summary_grid",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("property_code", Type.INT),
            ("source_code", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _render_zone_summary_grid,
)
