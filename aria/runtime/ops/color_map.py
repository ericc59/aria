"""Global color map runtime op.

Applies a fixed color→color substitution table to the input grid.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


def _apply_global_color_map(
    grid: Grid,
    n_pairs: int,
    c0_from: int, c0_to: int,
    c1_from: int, c1_to: int,
    c2_from: int, c2_to: int,
    c3_from: int, c3_to: int,
    c4_from: int, c4_to: int,
    c5_from: int, c5_to: int,
    c6_from: int, c6_to: int,
    c7_from: int, c7_to: int,
    c8_from: int, c8_to: int,
    c9_from: int, c9_to: int,
) -> Grid:
    """Apply up to 10 color substitutions."""
    result = grid.copy()
    pairs = [
        (c0_from, c0_to), (c1_from, c1_to), (c2_from, c2_to),
        (c3_from, c3_to), (c4_from, c4_to), (c5_from, c5_to),
        (c6_from, c6_to), (c7_from, c7_to), (c8_from, c8_to),
        (c9_from, c9_to),
    ]
    # Build substitution in a temp buffer to handle swaps correctly
    # (e.g. 5→8, 8→5 must not clobber)
    out = grid.copy()
    for i in range(min(n_pairs, 10)):
        fc, tc = pairs[i]
        if fc < 0:
            continue
        out[grid == fc] = tc
    return out


register(
    "apply_global_color_map",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("n_pairs", Type.INT),
            ("c0_from", Type.INT), ("c0_to", Type.INT),
            ("c1_from", Type.INT), ("c1_to", Type.INT),
            ("c2_from", Type.INT), ("c2_to", Type.INT),
            ("c3_from", Type.INT), ("c3_to", Type.INT),
            ("c4_from", Type.INT), ("c4_to", Type.INT),
            ("c5_from", Type.INT), ("c5_to", Type.INT),
            ("c6_from", Type.INT), ("c6_to", Type.INT),
            ("c7_from", Type.INT), ("c7_to", Type.INT),
            ("c8_from", Type.INT), ("c8_to", Type.INT),
            ("c9_from", Type.INT), ("c9_to", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _apply_global_color_map,
)
