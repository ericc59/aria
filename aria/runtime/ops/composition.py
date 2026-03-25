"""Higher-order composition operations."""

from __future__ import annotations

from typing import Any, Callable

from aria.types import Grid, ObjectNode, Type
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _compose(
    f: Callable[..., Any], g: Callable[..., Any]
) -> Callable[..., Any]:
    """Compose two functions: compose(f, g)(x) = f(g(x))."""
    def composed(x: Any) -> Any:
        return f(g(x))
    return composed


def _map_obj(
    f: Callable[[ObjectNode], ObjectNode], objects: set[ObjectNode]
) -> set[ObjectNode]:
    """Apply a transform to every object in a set."""
    return {f(obj) for obj in objects}


def _map_list(f: Callable[[Any], Any], lst: list[Any]) -> list[Any]:
    """Apply a function to every element of a list."""
    return [f(item) for item in lst]


def _fold(
    f: Callable[[Any, Any], Any], init: Any, lst: list[Any]
) -> Any:
    """Left fold over a list: fold(f, z, [a,b,c]) = f(f(f(z,a),b),c).

    The callable `f` receives (accumulator, element) but since our
    Lambda nodes are single-parameter, we wrap: f is called as
    f(element) and is expected to return a function of the accumulator,
    OR the caller passes a two-arg closure.
    """
    acc = init
    for item in lst:
        acc = f(acc, item)
    return acc


def _if_then_else(cond: bool, a: Any, b: Any) -> Any:
    """Conditional expression."""
    return a if cond else b


def _repeat_apply(n: int, f: Callable[[Grid], Grid], grid: Grid) -> Grid:
    """Apply a grid transform n times in sequence."""
    result = grid
    for _ in range(n):
        result = f(result)
    return result


def _for_each_place(
    objects,
    pos_fn: Callable[[ObjectNode], tuple[int, int]],
    grid: Grid,
) -> Grid:
    """Place each object at a position determined by pos_fn, folding into the grid.

    Accepts ObjectSet (set), ObjectList (list), or a single ObjectNode.
    Processes objects in id order for determinism.
    """
    from aria.runtime.ops.grid import _place_at

    if isinstance(objects, ObjectNode):
        objects = [objects]
    elif isinstance(objects, set):
        objects = sorted(objects, key=lambda o: o.id)
    elif isinstance(objects, list):
        pass
    else:
        raise ValueError(f"for_each_place: expected objects, got {type(objects).__name__}")

    result = grid.copy()
    for obj in objects:
        if not isinstance(obj, ObjectNode):
            continue
        pos = pos_fn(obj)
        result = _place_at(obj, pos, result)
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "compose",
    OpSignature(
        params=(("f", Type.CALLABLE), ("g", Type.CALLABLE)),
        return_type=Type.CALLABLE,
    ),
    _compose,
)

register(
    "map_obj",
    OpSignature(
        params=(("f", Type.OBJ_TRANSFORM), ("objects", Type.OBJECT_SET)),
        return_type=Type.OBJECT_SET,
    ),
    _map_obj,
)

register(
    "map_list",
    OpSignature(
        params=(("f", Type.CALLABLE), ("lst", Type.OBJECT_LIST)),
        return_type=Type.OBJECT_LIST,
    ),
    _map_list,
)

register(
    "fold",
    OpSignature(
        params=(
            ("f", Type.CALLABLE),
            ("init", Type.GRID),
            ("lst", Type.OBJECT_LIST),
        ),
        return_type=Type.GRID,
    ),
    _fold,
)

register(
    "if_then_else",
    OpSignature(
        params=(("cond", Type.BOOL), ("a", Type.GRID), ("b", Type.GRID)),
        return_type=Type.GRID,
    ),
    _if_then_else,
)

register(
    "repeat_apply",
    OpSignature(
        params=(
            ("n", Type.INT),
            ("f", Type.GRID_TRANSFORM),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _repeat_apply,
)

register(
    "for_each_place",
    OpSignature(
        params=(
            ("objects", Type.OBJECT_SET),
            ("pos_fn", Type.CALLABLE),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _for_each_place,
)
