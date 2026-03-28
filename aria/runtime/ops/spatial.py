"""Spatial transform operations on objects."""

from __future__ import annotations

import numpy as np

from aria.types import Axis, Dir, ObjectNode, Shape, Type
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replace_obj(obj: ObjectNode, **kwargs) -> ObjectNode:  # type: ignore[no-untyped-def]
    """Return a new ObjectNode with the given fields replaced."""
    return ObjectNode(
        id=kwargs.get("id", obj.id),
        color=kwargs.get("color", obj.color),
        mask=kwargs.get("mask", obj.mask),
        bbox=kwargs.get("bbox", obj.bbox),
        shape=kwargs.get("shape", obj.shape),
        symmetry=kwargs.get("symmetry", obj.symmetry),
        size=kwargs.get("size", obj.size),
    )


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _translate_delta(dr: int, dc: int, obj: ObjectNode) -> ObjectNode:
    """Translate an object by arbitrary (dr, dc).

    dr = row delta (positive = down), dc = col delta (positive = right).
    Returns a new ObjectNode with updated bbox. The mask stays the same.
    """
    x, y, w, h = obj.bbox
    return _replace_obj(obj, bbox=(x + dc, y + dr, w, h))


def _translate(direction: Dir, amount: int, obj: ObjectNode) -> ObjectNode:
    """Translate an object by `amount` cells in the given direction.

    Returns a new ObjectNode with an updated bbox. The mask stays the same.
    """
    x, y, w, h = obj.bbox
    dx, dy = {
        Dir.UP: (0, -amount),
        Dir.DOWN: (0, amount),
        Dir.LEFT: (-amount, 0),
        Dir.RIGHT: (amount, 0),
    }[direction]
    return _replace_obj(obj, bbox=(x + dx, y + dy, w, h))


def _rotate(degrees: int, obj: ObjectNode) -> ObjectNode:
    """Rotate an object's mask by 90, 180, or 270 degrees clockwise.

    Returns a new ObjectNode. The bbox dimensions are swapped for
    90/270 rotations; the top-left corner stays the same.
    """
    if degrees not in (90, 180, 270):
        raise ValueError(f"rotate: degrees must be 90, 180, or 270, got {degrees}")
    k = degrees // 90  # number of 90-degree clockwise rotations
    # np.rot90 rotates counter-clockwise, so k=1 CCW = 3 CW
    new_mask = np.rot90(obj.mask, k=4 - k)
    x, y, w, h = obj.bbox
    if degrees in (90, 270):
        new_bbox = (x, y, h, w)
    else:
        new_bbox = (x, y, w, h)
    return _replace_obj(obj, mask=new_mask.copy(), bbox=new_bbox, size=obj.size)


def _reflect(axis: Axis, obj: ObjectNode) -> ObjectNode:
    """Reflect an object's mask across the given axis.

    The bbox position stays the same; only the mask content is flipped.
    """
    if axis == Axis.HORIZONTAL:
        new_mask = np.flip(obj.mask, axis=0)  # flip rows (top/bottom)
    elif axis == Axis.VERTICAL:
        new_mask = np.flip(obj.mask, axis=1)  # flip cols (left/right)
    elif axis == Axis.DIAG_MAIN:
        new_mask = obj.mask.T
        x, y, w, h = obj.bbox
        return _replace_obj(obj, mask=new_mask.copy(), bbox=(x, y, h, w))
    elif axis == Axis.DIAG_ANTI:
        new_mask = np.flip(np.flip(obj.mask, axis=0).T, axis=0)
        x, y, w, h = obj.bbox
        return _replace_obj(obj, mask=new_mask.copy(), bbox=(x, y, h, w))
    else:
        raise ValueError(f"reflect: unknown axis {axis}")
    return _replace_obj(obj, mask=new_mask.copy())


def _resize_obj(factor: int, obj: ObjectNode) -> ObjectNode:
    """Scale an object's mask by an integer factor.

    Each pixel becomes a factor x factor block.
    """
    if factor < 1:
        raise ValueError(f"resize_obj: factor must be >= 1, got {factor}")
    new_mask = np.repeat(np.repeat(obj.mask, factor, axis=0), factor, axis=1)
    x, y, w, h = obj.bbox
    new_bbox = (x, y, w * factor, h * factor)
    new_size = obj.size * factor * factor
    return _replace_obj(obj, mask=new_mask, bbox=new_bbox, size=new_size)


def _recolor(color: int, obj: ObjectNode) -> ObjectNode:
    """Change an object's color."""
    return _replace_obj(obj, color=color)


def _gravity(direction: Dir, obj: ObjectNode) -> ObjectNode:
    """Move an object to the grid edge in the given direction.

    Since we don't have grid dimensions here, we move the object's
    bbox origin to 0 on the relevant axis. The caller should use
    this with grid-aware placement afterward.
    """
    x, y, w, h = obj.bbox
    if direction == Dir.UP:
        return _replace_obj(obj, bbox=(x, 0, w, h))
    if direction == Dir.DOWN:
        # Without grid height, move to y=0 as a convention.
        # Real usage should compose with grid dims.
        return _replace_obj(obj, bbox=(x, 0, w, h))
    if direction == Dir.LEFT:
        return _replace_obj(obj, bbox=(0, y, w, h))
    if direction == Dir.RIGHT:
        return _replace_obj(obj, bbox=(0, y, w, h))
    raise ValueError(f"gravity: unknown direction {direction}")


def _extend(direction: Dir, amount: int, obj: ObjectNode) -> ObjectNode:
    """Extend an object's mask by `amount` cells in the given direction.

    Adds rows or columns of True to the mask on the specified side.
    """
    mask = obj.mask
    x, y, w, h = obj.bbox

    if direction == Dir.DOWN:
        extension = np.ones((amount, mask.shape[1]), dtype=np.bool_)
        new_mask = np.vstack([mask, extension])
        return _replace_obj(
            obj, mask=new_mask, bbox=(x, y, w, h + amount),
            size=obj.size + amount * mask.shape[1],
        )
    if direction == Dir.UP:
        extension = np.ones((amount, mask.shape[1]), dtype=np.bool_)
        new_mask = np.vstack([extension, mask])
        return _replace_obj(
            obj, mask=new_mask, bbox=(x, y - amount, w, h + amount),
            size=obj.size + amount * mask.shape[1],
        )
    if direction == Dir.RIGHT:
        extension = np.ones((mask.shape[0], amount), dtype=np.bool_)
        new_mask = np.hstack([mask, extension])
        return _replace_obj(
            obj, mask=new_mask, bbox=(x, y, w + amount, h),
            size=obj.size + amount * mask.shape[0],
        )
    if direction == Dir.LEFT:
        extension = np.ones((mask.shape[0], amount), dtype=np.bool_)
        new_mask = np.hstack([extension, mask])
        return _replace_obj(
            obj, mask=new_mask, bbox=(x - amount, y, w + amount, h),
            size=obj.size + amount * mask.shape[0],
        )
    raise ValueError(f"extend: unknown direction {direction}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "translate_delta",
    OpSignature(
        params=(("dr", Type.INT), ("dc", Type.INT), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _translate_delta,
)

register(
    "translate",
    OpSignature(
        params=(("dir", Type.DIR), ("amount", Type.INT), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _translate,
)

register(
    "rotate",
    OpSignature(
        params=(("degrees", Type.INT), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _rotate,
)

register(
    "reflect",
    OpSignature(
        params=(("axis", Type.AXIS), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _reflect,
)

register(
    "resize_obj",
    OpSignature(
        params=(("factor", Type.INT), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _resize_obj,
)

register(
    "recolor",
    OpSignature(
        params=(("color", Type.COLOR), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _recolor,
)

register(
    "gravity",
    OpSignature(
        params=(("dir", Type.DIR), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _gravity,
)

register(
    "extend",
    OpSignature(
        params=(("dir", Type.DIR), ("amount", Type.INT), ("obj", Type.OBJECT)),
        return_type=Type.OBJECT,
    ),
    _extend,
)


# ---------------------------------------------------------------------------
# Anchor-aligned movement (OBJ_TRANSFORM factories)
# ---------------------------------------------------------------------------


def _align_center_to_row_of(anchor: ObjectNode) -> callable:
    """Return a transform that moves obj so its bbox center row matches anchor's."""
    target_row = anchor.bbox[1] + anchor.bbox[3] // 2

    def transform(obj: ObjectNode) -> ObjectNode:
        obj_center_row = obj.bbox[1] + obj.bbox[3] // 2
        dr = target_row - obj_center_row
        if dr == 0:
            return obj
        x, y, w, h = obj.bbox
        return _replace_obj(obj, bbox=(x, y + dr, w, h))

    return transform


def _align_center_to_col_of(anchor: ObjectNode) -> callable:
    """Return a transform that moves obj so its bbox center col matches anchor's."""
    target_col = anchor.bbox[0] + anchor.bbox[2] // 2

    def transform(obj: ObjectNode) -> ObjectNode:
        obj_center_col = obj.bbox[0] + obj.bbox[2] // 2
        dc = target_col - obj_center_col
        if dc == 0:
            return obj
        x, y, w, h = obj.bbox
        return _replace_obj(obj, bbox=(x + dc, y, w, h))

    return transform


register(
    "align_center_to_row_of",
    OpSignature(
        params=(("anchor", Type.OBJECT),),
        return_type=Type.OBJ_TRANSFORM,
    ),
    _align_center_to_row_of,
)

register(
    "align_center_to_col_of",
    OpSignature(
        params=(("anchor", Type.OBJECT),),
        return_type=Type.OBJ_TRANSFORM,
    ),
    _align_center_to_col_of,
)
