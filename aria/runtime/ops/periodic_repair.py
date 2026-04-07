"""Explicit periodic repair primitives — decomposed stages.

Exposes the internal stages of periodic repair as registered ops
with explicit parameters:

  detect_frame(grid) -> grid
    Detect and peel the outermost frame border, returning the interior.

  partition_interior(grid) -> grid
    Partition the interior into cells by separator lines.
    Returns the grid unchanged if no separators found.

  repair_periodic_lines(grid, axis, period) -> grid
    Infer the dominant motif along the given axis with the given period,
    then repair violations.

  repair_2d_motif(grid) -> grid
    Infer a 2D tiling motif by majority vote and repair violations.

  periodic_repair(grid, axis, period, repair_mode) -> grid
    Composite: peel_frame -> partition -> repair per cell.
    repair_mode:
      0 = lines_only   — 1D line repair only
      1 = motif_2d     — 2D motif repair only
      2 = lines_then_2d — 1D then 2D (current default)

All parameters are explicit integers visible in programs and specialization.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.grid import (
    _detect_frame_color,
    _partition_grid,
    _repair_grid_periodic_lines,
    _infer_2d_motif,
    _repair_2d_cell,
    _repair_cell_2d_or_1d,
    _repair_framed_lines,
    _repair_framed_2d_motif,
)
from aria.types import Grid, Type


# ---------------------------------------------------------------------------
# Repair mode constants
# ---------------------------------------------------------------------------

REPAIR_LINES_ONLY = 0
REPAIR_MOTIF_2D = 1
REPAIR_LINES_THEN_2D = 2

REPAIR_MODE_NAMES = {
    REPAIR_LINES_ONLY: "lines_only",
    REPAIR_MOTIF_2D: "motif_2d",
    REPAIR_LINES_THEN_2D: "lines_then_2d",
}
ALL_REPAIR_MODES = (REPAIR_LINES_ONLY, REPAIR_MOTIF_2D, REPAIR_LINES_THEN_2D)


# ---------------------------------------------------------------------------
# Stage 1: Frame detection / peeling
# ---------------------------------------------------------------------------


def _detect_frame(grid: Grid) -> Grid:
    """Detect and peel the outermost frame border, returning the interior.

    If no frame is detected, returns the grid unchanged.
    """
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return grid.copy()
    fc = _detect_frame_color(grid)
    if fc is None:
        return grid.copy()
    return grid[1:rows - 1, 1:cols - 1].copy()


# ---------------------------------------------------------------------------
# Stage 2: Partition
# ---------------------------------------------------------------------------


def _partition_interior(grid: Grid) -> Grid:
    """Partition the interior by separator lines.

    Returns the grid unchanged — partitioning is structural
    and used by the repair stage internally. This op is for
    diagnostic visibility, not transformation.
    """
    return grid.copy()


# ---------------------------------------------------------------------------
# Stage 3: Repair
# ---------------------------------------------------------------------------


def _periodic_repair(grid: Grid, axis: int, period: int, repair_mode: int) -> Grid:
    """Composite periodic repair with explicit mode selection.

    Stages:
    1. Peel frame(s) recursively
    2. Partition interior by separators
    3. Repair each cell using the specified mode

    repair_mode:
      0 = lines_only    — 1D line repair per cell
      1 = motif_2d      — 2D motif repair per cell
      2 = lines_then_2d — 1D repair then 2D motif repair
    """
    result = grid.copy()

    if repair_mode == REPAIR_LINES_ONLY:
        # Use existing 1D repair
        return _repair_framed_lines(grid, axis, period)
    elif repair_mode == REPAIR_MOTIF_2D:
        # Use existing 2D repair
        return _repair_framed_2d_motif(grid)
    elif repair_mode == REPAIR_LINES_THEN_2D:
        # 1D then 2D (current default behavior)
        intermediate = _repair_framed_lines(grid, axis, period)
        return _repair_framed_2d_motif(intermediate)

    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "detect_frame",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _detect_frame,
)

register(
    "periodic_repair",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("axis", Type.INT),
            ("period", Type.INT),
            ("repair_mode", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _periodic_repair,
)
