"""Curried higher-order object factories for search.

Each factory partially applies a spatial/selection op, returning
a PREDICATE or OBJ_TRANSFORM that search can bind and pass to
higher-order consumers like ``where`` or ``map_obj``.

Only a small, explicit set is registered here so that the search
space stays bounded and auditable.
"""

from __future__ import annotations

from typing import Callable

from aria.types import ObjectNode, Type
from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.spatial import _replace_obj


# ---------------------------------------------------------------------------
# OBJ_TRANSFORM factories (partial application → Object → Object)
# ---------------------------------------------------------------------------


def _recolor_to(color: int) -> Callable[[ObjectNode], ObjectNode]:
    """Partially applied recolor: recolor_to(c)(obj) == recolor(c, obj)."""
    def transform(obj: ObjectNode) -> ObjectNode:
        return _replace_obj(obj, color=color)
    return transform


register(
    "recolor_to",
    OpSignature(params=(("color", Type.COLOR),), return_type=Type.OBJ_TRANSFORM),
    _recolor_to,
)


def _translate_by(dr: int, dc: int) -> Callable[[ObjectNode], ObjectNode]:
    """Partially applied translate_delta: translate_by(dr, dc)(obj).

    dr = row delta (positive = down), dc = col delta (positive = right).
    """
    def transform(obj: ObjectNode) -> ObjectNode:
        x, y, w, h = obj.bbox
        return _replace_obj(obj, bbox=(x + dc, y + dr, w, h))
    return transform


register(
    "translate_by",
    OpSignature(
        params=(("dr", Type.INT), ("dc", Type.INT)),
        return_type=Type.OBJ_TRANSFORM,
    ),
    _translate_by,
)
