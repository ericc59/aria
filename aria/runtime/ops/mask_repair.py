"""Masked-region symmetry repair op.

Detects a solid-color marker rectangle (the mask), infers the scene's
symmetry center from input-only signals, computes the source position
by reflecting the mask through that center, and applies the transform.

Shared program: one transform code parameter, source derived at runtime
from inferred symmetry center. No per-demo bindings needed.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


def _find_solid_marker(grid: Grid, bg: int) -> tuple[int, int, int, int, int] | None:
    """Find the largest solid rectangular marker that is not bg.

    Returns (color, r0, c0, height, width) or None.
    """
    best = None
    for color in range(10):
        if color == bg:
            continue
        mask = grid == color
        if not np.any(mask):
            continue
        positions = np.argwhere(mask)
        r0, c0 = int(positions[:, 0].min()), int(positions[:, 1].min())
        r1, c1 = int(positions[:, 0].max()), int(positions[:, 1].max())
        rh, rw = r1 - r0 + 1, c1 - c0 + 1

        if rh * rw != int(np.sum(mask)):
            continue
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(region == color):
            continue

        area = rh * rw
        if area < 4:
            continue
        if best is None or area > best[0]:
            best = (area, color, r0, c0, rh, rw)

    if best is None:
        return None
    return (best[1], best[2], best[3], best[4], best[5])


_TRANSFORMS = [
    ("rot180", lambda s: np.rot90(s, 2)),
    ("flipLR", lambda s: np.fliplr(s)),
    ("flipUD", lambda s: np.flipud(s)),
    ("identity", lambda s: s.copy()),
]


def _infer_symmetry_center(grid: Grid, mask_region: np.ndarray, transform_code: int) -> tuple[float, float] | None:
    """Infer the scene's symmetry center from input-only signals.

    Searches for the half-integer center that maximizes the number of
    pixel pairs agreeing under the given transform, excluding mask pixels.

    Search range scales with grid size:
    - Small grids (≤10): exhaustive search over all centers
    - Large grids: ±min(ih//2, 15) around the midpoint
    """
    ih, iw = grid.shape
    mid_r2 = ih - 1
    mid_c2 = iw - 1

    # Adaptive search range
    if ih <= 10:
        r_lo, r_hi = 0, 2 * ih
    else:
        search_r = min(ih // 2, 15)
        r_lo = max(0, mid_r2 - search_r)
        r_hi = min(2 * ih, mid_r2 + search_r + 1)

    if iw <= 10:
        c_lo, c_hi = 0, 2 * iw
    else:
        search_c = min(iw // 2, 15)
        c_lo = max(0, mid_c2 - search_c)
        c_hi = min(2 * iw, mid_c2 + search_c + 1)

    best_center = None
    best_sym = 0

    for cr2 in range(r_lo, r_hi):
        for cc2 in range(c_lo, c_hi):
            sym = 0
            for r in range(ih):
                if transform_code == 0:  # rot180
                    r2 = cr2 - r
                elif transform_code == 1:  # flipLR
                    r2 = r
                elif transform_code == 2:  # flipUD
                    r2 = cr2 - r
                else:
                    r2 = r

                if r2 < 0 or r2 >= ih:
                    continue

                for c in range(iw):
                    if transform_code == 0:  # rot180
                        c2 = cc2 - c
                    elif transform_code == 1:  # flipLR
                        c2 = cc2 - c
                    elif transform_code == 2:  # flipUD
                        c2 = c
                    else:
                        c2 = c

                    if c2 < 0 or c2 >= iw:
                        continue
                    if mask_region[r, c] or mask_region[r2, c2]:
                        continue
                    if grid[r, c] == grid[r2, c2]:
                        sym += 1

            if sym > best_sym:
                best_sym = sym
                best_center = (cr2 / 2.0, cc2 / 2.0)

    return best_center


def _source_from_center(
    mr0: int, mc0: int, mh: int, mw: int,
    cr: float, cc: float,
    transform_code: int,
) -> tuple[int, int] | None:
    """Compute source position from symmetry center and transform."""
    mr1 = mr0 + mh - 1
    mc1 = mc0 + mw - 1

    if transform_code == 0:  # rot180
        sr0 = int(round(2 * cr - mr1))
        sc0 = int(round(2 * cc - mc1))
    elif transform_code == 1:  # flipLR
        sr0 = mr0
        sc0 = int(round(2 * cc - mc1))
    elif transform_code == 2:  # flipUD
        sr0 = int(round(2 * cr - mr1))
        sc0 = mc0
    else:
        return None

    return (sr0, sc0)


def _repair_masked_region_op(grid: Grid, transform_code: int) -> Grid:
    """Repair masked region using inferred symmetry center.

    One shared program: the transform is a constant parameter.
    The source position is deterministically derived at runtime from
    the input grid's symmetry structure. No per-demo bindings.

    1. Detect the solid marker (mask).
    2. Infer the scene's symmetry center (excluding mask pixels).
    3. Compute the source position by reflecting the mask through the center.
    4. Apply the transform to the source to produce the repair patch.
    """
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    marker = _find_solid_marker(grid, bg)
    if marker is None:
        return grid.copy()

    mask_color, mr0, mc0, mh, mw = marker
    ih, iw = grid.shape

    if transform_code < 0 or transform_code >= len(_TRANSFORMS):
        return grid.copy()

    # Build mask region
    mask_region = np.zeros((ih, iw), dtype=bool)
    mask_region[mr0 : mr0 + mh, mc0 : mc0 + mw] = True

    # Infer symmetry center from input
    center = _infer_symmetry_center(grid, mask_region, transform_code)
    if center is None:
        return grid.copy()

    cr, cc = center
    pos = _source_from_center(mr0, mc0, mh, mw, cr, cc, transform_code)
    if pos is None:
        return grid.copy()

    sr0, sc0 = pos
    if sr0 < 0 or sc0 < 0 or sr0 + mh > ih or sc0 + mw > iw:
        return grid.copy()

    source = grid[sr0 : sr0 + mh, sc0 : sc0 + mw]
    if np.any(source == mask_color):
        return grid.copy()

    _, fn = _TRANSFORMS[transform_code]
    result = fn(source)
    if result.shape == (mh, mw):
        return result

    return grid.copy()


register(
    "repair_masked_region",
    OpSignature(
        params=(("grid", Type.GRID), ("transform_code", Type.INT)),
        return_type=Type.GRID,
    ),
    _repair_masked_region_op,
)
