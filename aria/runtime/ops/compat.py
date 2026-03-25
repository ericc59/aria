"""Compatibility wrappers and coercion helpers.

Makes ops more forgiving when the model passes slightly wrong types —
e.g. ObjectSet where a single Object is expected, or ObjectSet where
Grid is expected. This dramatically improves proposer success rate
without compromising the type system's safety.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import Grid, ObjectNode, Type, make_grid


def coerce_to_grid(val: Any) -> Grid | None:
    """Try to coerce a value to a Grid."""
    if isinstance(val, np.ndarray) and val.ndim == 2:
        return val

    # ObjectNode -> render as grid
    if isinstance(val, ObjectNode):
        from aria.runtime.ops.grid import _from_object
        return _from_object(val)

    # set of ObjectNode -> render all onto a grid
    if isinstance(val, (set, frozenset)) and val and isinstance(next(iter(val)), ObjectNode):
        return objects_to_grid(val)

    # list of ObjectNode
    if isinstance(val, list) and val and isinstance(val[0], ObjectNode):
        return objects_to_grid(set(val))

    return None


def objects_to_grid(objects: set[ObjectNode], bg: int = 0) -> Grid:
    """Render a set of ObjectNodes onto a minimal grid."""
    if not objects:
        return make_grid(1, 1, bg)

    # Find bounding box of all objects
    min_r, min_c = float("inf"), float("inf")
    max_r, max_c = 0, 0
    for obj in objects:
        x, y, w, h = obj.bbox
        min_r = min(min_r, y)
        min_c = min(min_c, x)
        max_r = max(max_r, y + h)
        max_c = max(max_c, x + w)

    rows = int(max_r - min_r)
    cols = int(max_c - min_c)
    if rows <= 0 or cols <= 0:
        return make_grid(1, 1, bg)

    grid = make_grid(rows, cols, bg)
    for obj in objects:
        x, y, w, h = obj.bbox
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    gr = y - int(min_r) + r
                    gc = x - int(min_c) + c
                    if 0 <= gr < rows and 0 <= gc < cols:
                        grid[gr, gc] = obj.color
    return grid
