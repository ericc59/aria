"""Shape classification for object masks.

Classifies a boolean mask (cropped to bounding box) into one of the
Shape enum variants based on geometry and pattern matching.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aria.types import Shape


def classify_shape(mask: NDArray[np.bool_]) -> Shape:
    """Classify a boolean mask into a Shape.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        Boolean mask of shape (h, w), cropped to the object's bounding box.
        True = object pixel.

    Returns
    -------
    Shape
        The classified shape.
    """
    h, w = mask.shape

    # Dot: single pixel
    if h == 1 and w == 1:
        return Shape.DOT

    # Line: single row or single column
    if h == 1 or w == 1:
        if mask.all():
            return Shape.LINE
        # Fragmented single-row/col still counts as line if contiguous
        return Shape.LINE

    # Rectangle: all pixels filled
    if mask.all():
        return Shape.RECT

    size = int(mask.sum())

    # Cross pattern: + shape
    if _is_cross(mask, h, w, size):
        return Shape.CROSS

    # T-shape
    if _is_t_shape(mask, h, w, size):
        return Shape.T

    # L-shape
    if _is_l_shape(mask, h, w, size):
        return Shape.L

    return Shape.IRREGULAR


def _is_cross(mask: NDArray[np.bool_], h: int, w: int, size: int) -> bool:
    """Check if mask is a + (cross) shape.

    A cross has a center row and center column that are fully filled,
    and all other pixels are False. Both dimensions must be odd.
    """
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False

    mid_r = h // 2
    mid_c = w // 2

    expected_size = h + w - 1
    if size != expected_size:
        return False

    # Center row and center column must be fully True
    if not mask[mid_r, :].all():
        return False
    if not mask[:, mid_c].all():
        return False

    # Everything else must be False
    test = np.zeros_like(mask)
    test[mid_r, :] = True
    test[:, mid_c] = True
    return np.array_equal(mask, test)


def _is_t_shape(mask: NDArray[np.bool_], h: int, w: int, size: int) -> bool:
    """Check if mask is a T-shape (in any of 4 rotations).

    A T-shape is a bar along one edge with a perpendicular stem from
    its midpoint.
    """
    for rot in range(4):
        m = np.rot90(mask, rot)
        rh, rw = m.shape
        if rh < 2 or rw < 3 or rw % 2 == 0:
            continue
        mid_c = rw // 2
        expected = rw + (rh - 1)
        if size != expected:
            continue
        # Top row fully filled
        if not m[0, :].all():
            continue
        # Stem from center going down
        if not m[:, mid_c].all():
            continue
        # Check nothing else is set
        test = np.zeros_like(m)
        test[0, :] = True
        test[:, mid_c] = True
        if np.array_equal(m, test):
            return True

    return False


def _is_l_shape(mask: NDArray[np.bool_], h: int, w: int, size: int) -> bool:
    """Check if mask is an L-shape (in any of 4 rotations).

    An L-shape has one full row along one edge and one full column
    along one edge, meeting at a corner, with no extra pixels.
    """
    for rot in range(4):
        m = np.rot90(mask, rot)
        rh, rw = m.shape
        if rh < 2 or rw < 2:
            continue
        expected = rh + rw - 1
        if size != expected:
            continue
        # Bottom row fully filled and left column fully filled
        if not m[rh - 1, :].all():
            continue
        if not m[:, 0].all():
            continue
        test = np.zeros_like(m)
        test[rh - 1, :] = True
        test[:, 0] = True
        if np.array_equal(m, test):
            return True

    return False
