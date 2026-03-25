"""Arithmetic and utility operations.

These are referenced in the DESIGN.md examples but were not in the
original ~40 op catalog. They're essential for computed dimensions,
indexing, and general numeric work in programs.
"""

from __future__ import annotations

import math

from aria.types import ObjectNode, Type
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def _add(a: int, b: int) -> int:
    return a + b

def _sub(a: int, b: int) -> int:
    return a - b

def _mul(a: int, b: int) -> int:
    return a * b

def _div(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("div: division by zero")
    return a // b

def _mod(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("mod: division by zero")
    return a % b

def _isqrt(n: int) -> int:
    """Integer square root."""
    if n < 0:
        raise ValueError("isqrt: negative input")
    return int(math.isqrt(n))

def _abs_val(n: int) -> int:
    return abs(n)

def _eq(a: int, b: int) -> bool:
    return a == b

def _neq(a: int, b: int) -> bool:
    return a != b

def _lt(a: int, b: int) -> bool:
    return a < b

def _gt(a: int, b: int) -> bool:
    return a > b

def _lte(a: int, b: int) -> bool:
    return a <= b

def _gte(a: int, b: int) -> bool:
    return a >= b

def _and(a: bool, b: bool) -> bool:
    return a and b

def _or(a: bool, b: bool) -> bool:
    return a or b

def _not(a: bool) -> bool:
    return not a


# ---------------------------------------------------------------------------
# Object property accessors
# ---------------------------------------------------------------------------

def _get_color(obj: ObjectNode) -> int:
    return obj.color

def _get_size(obj: ObjectNode) -> int:
    return obj.size

def _get_shape(obj: ObjectNode) -> str:
    return obj.shape

def _get_pos_x(obj: ObjectNode) -> int:
    return obj.bbox[0]

def _get_pos_y(obj: ObjectNode) -> int:
    return obj.bbox[1]

def _get_width(obj: ObjectNode) -> int:
    return obj.bbox[2]

def _get_height(obj: ObjectNode) -> int:
    return obj.bbox[3]

def _center_x(obj: ObjectNode) -> int:
    return obj.bbox[0] + (obj.bbox[2] // 2)

def _center_y(obj: ObjectNode) -> int:
    return obj.bbox[1] + (obj.bbox[3] // 2)

def _chebyshev_distance(a: ObjectNode, b: ObjectNode) -> int:
    return max(abs(_center_x(a) - _center_x(b)), abs(_center_y(a) - _center_y(b)))

def _manhattan_distance(a: ObjectNode, b: ObjectNode) -> int:
    return abs(_center_x(a) - _center_x(b)) + abs(_center_y(a) - _center_y(b))


# ---------------------------------------------------------------------------
# Tuple construction
# ---------------------------------------------------------------------------

def _make_tuple(*args) -> tuple:
    return tuple(args)


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------

def _sum_list(ints: list[int]) -> int:
    return sum(ints)

def _range_list(n: int) -> list[int]:
    return list(range(n))

def _reverse_list(lst: list) -> list:
    return list(reversed(lst))

def _index_of(val: int, lst: list[int]) -> int:
    try:
        return lst.index(val)
    except ValueError:
        raise ValueError(f"index_of: {val} not in list")

def _contains(val: int, lst: list[int]) -> bool:
    return val in lst

def _at(idx: int, lst: list) -> object:
    if idx < 0 or idx >= len(lst):
        raise IndexError(f"at: index {idx} out of range for list of length {len(lst)}")
    return lst[idx]


# ---------------------------------------------------------------------------
# Registration (make_tuple is special — variadic, registered with PAIR return)
# ---------------------------------------------------------------------------

register("make_tuple", OpSignature(params=(), return_type=Type.PAIR), _make_tuple)


# ---------------------------------------------------------------------------

# Arithmetic
register("add", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.INT), _add)
register("sub", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.INT), _sub)
register("mul", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.INT), _mul)
register("div", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.INT), _div)
register("mod", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.INT), _mod)
register("isqrt", OpSignature(params=(("n", Type.INT),), return_type=Type.INT), _isqrt)
register("abs", OpSignature(params=(("n", Type.INT),), return_type=Type.INT), _abs_val)

# Comparison
register("eq", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _eq)
register("neq", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _neq)
register("lt", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _lt)
register("gt", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _gt)
register("lte", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _lte)
register("gte", OpSignature(params=(("a", Type.INT), ("b", Type.INT)), return_type=Type.BOOL), _gte)

# Logic
register("and", OpSignature(params=(("a", Type.BOOL), ("b", Type.BOOL)), return_type=Type.BOOL), _and)
register("or", OpSignature(params=(("a", Type.BOOL), ("b", Type.BOOL)), return_type=Type.BOOL), _or)
register("not", OpSignature(params=(("a", Type.BOOL),), return_type=Type.BOOL), _not)

# Object accessors
register("get_color", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.COLOR), _get_color)
register("get_size", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _get_size)
register("get_shape", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.SHAPE), _get_shape)
register("get_pos_x", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _get_pos_x)
register("get_pos_y", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _get_pos_y)
register("get_width", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _get_width)
register("get_height", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _get_height)
register("center_x", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _center_x)
register("center_y", OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.INT), _center_y)
register("chebyshev_distance", OpSignature(params=(("a", Type.OBJECT), ("b", Type.OBJECT)), return_type=Type.INT), _chebyshev_distance)
register("manhattan_distance", OpSignature(params=(("a", Type.OBJECT), ("b", Type.OBJECT)), return_type=Type.INT), _manhattan_distance)

# List operations
register("sum_list", OpSignature(params=(("ints", Type.INT_LIST),), return_type=Type.INT), _sum_list)
register("range_list", OpSignature(params=(("n", Type.INT),), return_type=Type.INT_LIST), _range_list)
register("reverse_list", OpSignature(params=(("lst", Type.INT_LIST),), return_type=Type.INT_LIST), _reverse_list)
register("index_of", OpSignature(params=(("val", Type.INT), ("lst", Type.INT_LIST)), return_type=Type.INT), _index_of)
register("contains", OpSignature(params=(("val", Type.INT), ("lst", Type.INT_LIST)), return_type=Type.BOOL), _contains)
register("at", OpSignature(params=(("idx", Type.INT), ("lst", Type.INT_LIST)), return_type=Type.INT), _at)
