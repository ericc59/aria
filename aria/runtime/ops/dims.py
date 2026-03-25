"""Dimension query and construction operations."""

from __future__ import annotations

from aria.types import Grid, ObjectNode, Type
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _dims_of(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) of a grid."""
    return (int(grid.shape[0]), int(grid.shape[1]))


def _dims_make(r: int, c: int) -> tuple[int, int]:
    """Construct a (rows, cols) pair."""
    return (r, c)


def _scale_dims(dims: tuple[int, int], factor: int) -> tuple[int, int]:
    """Scale dimensions by an integer factor."""
    return (dims[0] * factor, dims[1] * factor)


def _obj_dims(obj: ObjectNode) -> tuple[int, int]:
    """Return the (height, width) of an object's bounding box."""
    _, _, w, h = obj.bbox
    return (h, w)


def _rows_of(dims: tuple[int, int]) -> int:
    """Extract the row count from a dims pair."""
    return dims[0]


def _cols_of(dims: tuple[int, int]) -> int:
    """Extract the column count from a dims pair."""
    return dims[1]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "dims_of",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.DIMS),
    _dims_of,
)

register(
    "dims_make",
    OpSignature(params=(("r", Type.INT), ("c", Type.INT)), return_type=Type.DIMS),
    _dims_make,
)

register(
    "scale_dims",
    OpSignature(
        params=(("dims", Type.DIMS), ("factor", Type.INT)),
        return_type=Type.DIMS,
    ),
    _scale_dims,
)

register(
    "obj_dims",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.DIMS),
    _obj_dims,
)

register(
    "rows_of",
    OpSignature(params=(("dims", Type.DIMS),), return_type=Type.INT),
    _rows_of,
)

register(
    "cols_of",
    OpSignature(params=(("dims", Type.DIMS),), return_type=Type.INT),
    _cols_of,
)
