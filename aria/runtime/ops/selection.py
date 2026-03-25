"""Selection operations: finding and filtering objects in grids."""

from __future__ import annotations

from typing import Callable

import numpy as np

from aria.types import Color, Grid, ObjectNode, Shape, Symmetry, Type, make_grid
from aria.runtime.ops import OpSignature, register
from aria.graph.cc_label import label_4conn


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _find_objects(grid: Grid) -> set[ObjectNode]:
    """Run connected-component labeling, returning all non-background objects.

    Uses the most common color as background.
    """
    if grid.size == 0:
        return set()
    # Detect background as the most frequent color.
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = int(unique[np.argmax(counts)])
    objects = label_4conn(grid, ignore_color=bg_color)
    return set(objects)


def _by_color(color: int) -> Callable[[ObjectNode], bool]:
    """Return a predicate that tests for the given color."""
    def pred(obj: ObjectNode) -> bool:
        return obj.color == color
    return pred


def _by_shape(shape: Shape) -> Callable[[ObjectNode], bool]:
    """Return a predicate that tests for the given shape."""
    def pred(obj: ObjectNode) -> bool:
        return obj.shape == shape
    return pred


def _by_size_rank(rank: int, objects: set[ObjectNode]) -> ObjectNode:
    """Return the object at the given size rank.

    rank=0 is the largest, rank=-1 (or last) is the smallest.
    Objects are sorted descending by size.
    """
    if not objects:
        raise ValueError("by_size_rank: empty object set")
    sorted_objs = sorted(objects, key=lambda o: o.size, reverse=True)
    if rank < 0:
        rank = len(sorted_objs) + rank
    if rank < 0 or rank >= len(sorted_objs):
        raise IndexError(f"by_size_rank: rank {rank} out of range for {len(sorted_objs)} objects")
    return sorted_objs[rank]


def _where(pred: Callable[[ObjectNode], bool], objects: set[ObjectNode]) -> set[ObjectNode]:
    """Filter objects by predicate."""
    return {obj for obj in objects if pred(obj)}


def _excluding(to_remove: set[ObjectNode], from_set: set[ObjectNode]) -> set[ObjectNode]:
    """Return from_set minus to_remove."""
    remove_ids = {obj.id for obj in to_remove}
    return {obj for obj in from_set if obj.id not in remove_ids}


def _related_to(obj: ObjectNode, rel_type: str) -> set[ObjectNode]:
    """Return objects related to obj by rel_type. STUB: returns empty set."""
    return set()


def _background_obj(grid: Grid) -> ObjectNode:
    """Return an ObjectNode representing the background of the grid.

    The background is defined as the most frequent color, and the
    resulting object's mask covers every cell of that color.
    """
    if grid.size == 0:
        raise ValueError("background_obj: empty grid")
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = int(unique[np.argmax(counts)])
    rows, cols = grid.shape
    mask = grid == bg_color
    pixel_count = int(np.sum(mask))
    return ObjectNode(
        id=-1,
        color=bg_color,
        mask=mask,
        bbox=(0, 0, cols, rows),
        shape=Shape.IRREGULAR,
        symmetry=frozenset(),
        size=pixel_count,
    )


def _singleton(objects: set[ObjectNode]) -> ObjectNode:
    """Assert exactly one object, return it."""
    if len(objects) != 1:
        raise ValueError(f"singleton: expected 1 object, got {len(objects)}")
    return next(iter(objects))


def _nth(idx: int, obj_list: set[ObjectNode] | list[ObjectNode]) -> ObjectNode:
    """Return the nth object, sorted by id for determinism."""
    if isinstance(obj_list, set):
        ordered = sorted(obj_list, key=lambda o: o.id)
    else:
        ordered = list(obj_list)
    if idx < 0 or idx >= len(ordered):
        raise IndexError(f"nth: index {idx} out of range for {len(ordered)} objects")
    return ordered[idx]


def _nearest_to(target: ObjectNode, objects: set[ObjectNode] | list[ObjectNode]) -> ObjectNode:
    """Return the nearest distinct object by center-to-center distance."""
    if isinstance(objects, set):
        candidates = sorted(objects, key=lambda obj: obj.id)
    else:
        candidates = list(objects)

    filtered = [obj for obj in candidates if obj.id != target.id]
    if not filtered:
        raise ValueError("nearest_to: no distinct candidate objects")

    tx = target.bbox[0] + (target.bbox[2] // 2)
    ty = target.bbox[1] + (target.bbox[3] // 2)

    return min(
        filtered,
        key=lambda obj: (
            max(
                abs((obj.bbox[0] + (obj.bbox[2] // 2)) - tx),
                abs((obj.bbox[1] + (obj.bbox[3] // 2)) - ty),
            ),
            abs((obj.bbox[0] + (obj.bbox[2] // 2)) - tx)
            + abs((obj.bbox[1] + (obj.bbox[3] // 2)) - ty),
            obj.id,
        ),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "find_objects",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.OBJECT_SET),
    _find_objects,
)

register(
    "by_color",
    OpSignature(params=(("color", Type.COLOR),), return_type=Type.PREDICATE),
    _by_color,
)

register(
    "by_shape",
    OpSignature(params=(("shape", Type.SHAPE),), return_type=Type.PREDICATE),
    _by_shape,
)

register(
    "by_size_rank",
    OpSignature(
        params=(("rank", Type.INT), ("objects", Type.OBJECT_SET)),
        return_type=Type.OBJECT,
    ),
    _by_size_rank,
)

register(
    "where",
    OpSignature(
        params=(("pred", Type.PREDICATE), ("objects", Type.OBJECT_SET)),
        return_type=Type.OBJECT_SET,
    ),
    _where,
)

register(
    "excluding",
    OpSignature(
        params=(("to_remove", Type.OBJECT_SET), ("from_set", Type.OBJECT_SET)),
        return_type=Type.OBJECT_SET,
    ),
    _excluding,
)

register(
    "related_to",
    OpSignature(
        params=(("obj", Type.OBJECT), ("rel_type", Type.INT)),
        return_type=Type.OBJECT_SET,
    ),
    _related_to,
)

register(
    "background_obj",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.OBJECT),
    _background_obj,
)

register(
    "singleton",
    OpSignature(params=(("objects", Type.OBJECT_SET),), return_type=Type.OBJECT),
    _singleton,
)

register(
    "nth",
    OpSignature(
        params=(("idx", Type.INT), ("obj_list", Type.OBJECT_SET)),
        return_type=Type.OBJECT,
    ),
    _nth,
)

register(
    "nearest_to",
    OpSignature(
        params=(("target", Type.OBJECT), ("objects", Type.OBJECT_SET)),
        return_type=Type.OBJECT,
    ),
    _nearest_to,
)
