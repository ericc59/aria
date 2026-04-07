"""Explicit object extraction / matching / placement / paint primitives.

The relocation lane is decomposed into four typed layers, each with
explicit structural parameters:

  Layer 1 — Extraction:
    extract_shapes(grid) -> list[ObjectNode]  (multi-pixel objects)
    extract_markers(grid) -> list[ObjectNode]  (single-pixel markers)

  Layer 2 — Matching (explicit rule):
    match_objects(shapes, markers, match_rule) -> list[(shape, marker)]
    Match rules:
      0 = shape_nearest     — each shape finds its nearest marker
      1 = marker_nearest    — each marker finds its nearest shape
      2 = color_match       — prefer same-color pairs, break ties by distance
      3 = ordered_row       — sort both by row, pair 1:1
      4 = ordered_col       — sort both by column, pair 1:1
      5 = size_match        — pair shapes/markers by size similarity
      6 = mutual_nearest    — only pair if each is the other's nearest

  Layer 3 — Placement (explicit alignment):
    compute_placement(shape, marker, align_mode) -> (row, col)
    Alignment modes:
      0 = center, 1 = at_marker, 2 = above_left, 3 = above_right,
      4 = below_left, 5 = below_right, 6 = marker_interior

  Layer 4 — Paint:
    paint_matched(grid, pairs, align_mode) -> grid

  Composite:
    relocate_objects(grid, match_rule, align_mode) -> grid
      All-in-one: extract + match + place + paint.
    match_and_place(grid, match_rule, align_mode) -> grid
      Compatibility alias for relocate_objects.

All parameters are explicit integers visible in specialization/programs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.selection import _find_objects
from aria.types import Grid, ObjectNode, Type


# ---------------------------------------------------------------------------
# Match rule constants
# ---------------------------------------------------------------------------

MATCH_SHAPE_NEAREST = 0
MATCH_MARKER_NEAREST = 1
MATCH_COLOR = 2
MATCH_ORDERED_ROW = 3
MATCH_ORDERED_COL = 4
MATCH_SIZE = 5
MATCH_MUTUAL_NEAREST = 6

MATCH_NAMES = {
    MATCH_SHAPE_NEAREST: "shape_nearest",
    MATCH_MARKER_NEAREST: "marker_nearest",
    MATCH_COLOR: "color_match",
    MATCH_ORDERED_ROW: "ordered_row",
    MATCH_ORDERED_COL: "ordered_col",
    MATCH_SIZE: "size_match",
    MATCH_MUTUAL_NEAREST: "mutual_nearest",
}
ALL_MATCH_RULES = tuple(MATCH_NAMES.keys())


# ---------------------------------------------------------------------------
# Alignment mode constants
# ---------------------------------------------------------------------------

ALIGN_CENTER = 0
ALIGN_AT_MARKER = 1
ALIGN_ABOVE_LEFT = 2
ALIGN_ABOVE_RIGHT = 3
ALIGN_BELOW_LEFT = 4
ALIGN_BELOW_RIGHT = 5
ALIGN_MARKER_INTERIOR = 6

ALIGN_NAMES = {
    ALIGN_CENTER: "center",
    ALIGN_AT_MARKER: "at_marker",
    ALIGN_ABOVE_LEFT: "above_left",
    ALIGN_ABOVE_RIGHT: "above_right",
    ALIGN_BELOW_LEFT: "below_left",
    ALIGN_BELOW_RIGHT: "below_right",
    ALIGN_MARKER_INTERIOR: "marker_interior",
}
ALL_ALIGNS = tuple(ALIGN_NAMES.keys())

# Backward compat aliases
ALL_RULES = ALL_MATCH_RULES
RULE_NEAREST = MATCH_SHAPE_NEAREST
RULE_COLOR_MATCH = MATCH_COLOR
RULE_ORDERED_ROW = MATCH_ORDERED_ROW
RULE_ORDERED_COL = MATCH_ORDERED_COL
RULE_NAMES = {
    RULE_NEAREST: "nearest",
    RULE_COLOR_MATCH: "color_match",
    RULE_ORDERED_ROW: "ordered_row",
    RULE_ORDERED_COL: "ordered_col",
}


# ---------------------------------------------------------------------------
# Layer 1: Extraction
# ---------------------------------------------------------------------------


def _detect_bg(grid: Grid) -> int:
    unique, counts = np.unique(grid, return_counts=True)
    return int(unique[np.argmax(counts)])


def _get_shapes_and_markers(grid: Grid) -> tuple[list[ObjectNode], list[ObjectNode], int]:
    bg = _detect_bg(grid)
    all_objs = list(_find_objects(grid))
    shapes = [o for o in all_objs if o.color != bg and o.size > 1]
    markers = [o for o in all_objs if o.color != bg and o.size == 1]
    return shapes, markers, bg


def _extract_shapes(grid: Grid) -> Grid:
    """Erase multi-pixel objects, keep markers + bg."""
    shapes, markers, bg = _get_shapes_and_markers(grid)
    result = np.full_like(grid, bg)
    for m in markers:
        mx, my = m.bbox[0], m.bbox[1]
        if 0 <= my < result.shape[0] and 0 <= mx < result.shape[1]:
            result[my, mx] = m.color
    return result


def _extract_markers(grid: Grid) -> Grid:
    """Erase single-pixel markers, keep shapes + bg."""
    shapes, markers, bg = _get_shapes_and_markers(grid)
    result = np.full_like(grid, bg)
    for s in shapes:
        sx, sy, sw, sh = s.bbox
        _paint_shape_at(result, s, sy, sx)
    return result


# ---------------------------------------------------------------------------
# Layer 2: Matching (explicit rule)
# ---------------------------------------------------------------------------


def _obj_center(o: ObjectNode) -> tuple[float, float]:
    x, y, w, h = o.bbox
    return (x + w / 2.0, y + h / 2.0)


def _l1(a: ObjectNode, b: ObjectNode) -> float:
    ac, bc = _obj_center(a), _obj_center(b)
    return abs(ac[0] - bc[0]) + abs(ac[1] - bc[1])


def match_objects(
    shapes: list[ObjectNode],
    markers: list[ObjectNode],
    rule: int,
) -> list[tuple[ObjectNode, ObjectNode]]:
    """Match shapes to markers under the given rule. Returns (shape, marker) pairs."""
    fn = _MATCH_FNS.get(rule, _match_shape_nearest)
    return fn(shapes, markers)


def _match_shape_nearest(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    """Each shape finds its nearest unused marker."""
    used: set[int] = set()
    pairs = []
    for shape in shapes:
        best_j, best_d = -1, float("inf")
        for j, m in enumerate(markers):
            if j in used:
                continue
            d = _l1(shape, m)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            pairs.append((shape, markers[best_j]))
            used.add(best_j)
    return pairs


def _match_marker_nearest(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    """Each marker finds its nearest unused shape."""
    used: set[int] = set()
    pairs = []
    for marker in markers:
        best_i, best_d = -1, float("inf")
        for i, s in enumerate(shapes):
            if i in used:
                continue
            d = _l1(s, marker)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            pairs.append((shapes[best_i], marker))
            used.add(best_i)
    return pairs


def _match_color(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    """Prefer same-color pairs, break ties by distance."""
    used_s: set[int] = set()
    used_m: set[int] = set()
    pairs = []
    # First pass: same-color, nearest
    for i, s in enumerate(shapes):
        best_j, best_d = -1, float("inf")
        for j, m in enumerate(markers):
            if j in used_m:
                continue
            if m.color == s.color:
                d = _l1(s, m)
                if d < best_d:
                    best_d = d
                    best_j = j
        if best_j >= 0:
            pairs.append((s, markers[best_j]))
            used_s.add(i)
            used_m.add(best_j)
    # Second pass: remaining shapes, any marker
    for i, s in enumerate(shapes):
        if i in used_s:
            continue
        best_j, best_d = -1, float("inf")
        for j, m in enumerate(markers):
            if j in used_m:
                continue
            d = _l1(s, m)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            pairs.append((s, markers[best_j]))
            used_s.add(i)
            used_m.add(best_j)
    return pairs


def _match_ordered_row(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    s_sorted = sorted(shapes, key=lambda o: (o.bbox[1], o.bbox[0]))
    m_sorted = sorted(markers, key=lambda o: (o.bbox[1], o.bbox[0]))
    return list(zip(s_sorted, m_sorted))


def _match_ordered_col(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    s_sorted = sorted(shapes, key=lambda o: (o.bbox[0], o.bbox[1]))
    m_sorted = sorted(markers, key=lambda o: (o.bbox[0], o.bbox[1]))
    return list(zip(s_sorted, m_sorted))


def _match_size(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    """Match by size similarity — largest shape with largest marker, etc."""
    s_sorted = sorted(shapes, key=lambda o: -o.size)
    m_sorted = sorted(markers, key=lambda o: -o.size)
    return list(zip(s_sorted, m_sorted))


def _match_mutual_nearest(shapes: list[ObjectNode], markers: list[ObjectNode]) -> list[tuple[ObjectNode, ObjectNode]]:
    """Only pair if shape is marker's nearest AND marker is shape's nearest."""
    pairs = []
    used_s: set[int] = set()
    used_m: set[int] = set()
    for i, s in enumerate(shapes):
        # Find s's nearest marker
        best_j = min(range(len(markers)), key=lambda j: _l1(s, markers[j]), default=-1)
        if best_j < 0 or best_j in used_m:
            continue
        # Check: is s also that marker's nearest shape?
        m = markers[best_j]
        best_i = min((ii for ii in range(len(shapes)) if ii not in used_s),
                     key=lambda ii: _l1(shapes[ii], m), default=-1)
        if best_i == i:
            pairs.append((s, m))
            used_s.add(i)
            used_m.add(best_j)
    return pairs


_MATCH_FNS = {
    MATCH_SHAPE_NEAREST: _match_shape_nearest,
    MATCH_MARKER_NEAREST: _match_marker_nearest,
    MATCH_COLOR: _match_color,
    MATCH_ORDERED_ROW: _match_ordered_row,
    MATCH_ORDERED_COL: _match_ordered_col,
    MATCH_SIZE: _match_size,
    MATCH_MUTUAL_NEAREST: _match_mutual_nearest,
}


# ---------------------------------------------------------------------------
# Layer 3: Placement (explicit alignment)
# ---------------------------------------------------------------------------


def _compute_placement(shape: ObjectNode, marker: ObjectNode, align: int) -> tuple[int, int]:
    """Compute (target_row, target_col) for shape top-left."""
    mx, my = marker.bbox[0], marker.bbox[1]
    sx, sy, sw, sh = shape.bbox

    if align == ALIGN_CENTER:
        return my - sh // 2, mx - sw // 2
    elif align == ALIGN_AT_MARKER:
        return my, mx
    elif align == ALIGN_ABOVE_LEFT:
        return my - sh, mx - sw
    elif align == ALIGN_ABOVE_RIGHT:
        return my - sh, mx + 1
    elif align == ALIGN_BELOW_LEFT:
        return my + 1, mx - sw
    elif align == ALIGN_BELOW_RIGHT:
        return my + 1, mx + 1
    elif align == ALIGN_MARKER_INTERIOR:
        mask = shape.mask
        mh, mw = mask.shape
        best_dr, best_dc = mh // 2, mw // 2
        best_dist = float("inf")
        cr, cc = mh / 2, mw / 2
        for dr in range(mh):
            for dc in range(mw):
                if mask[dr, dc]:
                    d = abs(dr - cr) + abs(dc - cc)
                    if d < best_dist:
                        best_dist = d
                        best_dr, best_dc = dr, dc
        return my - best_dr, mx - best_dc
    return my - sh // 2, mx - sw // 2


# ---------------------------------------------------------------------------
# Layer 4: Paint
# ---------------------------------------------------------------------------


def _paint_shape_at(result: Grid, shape: ObjectNode, target_row: int, target_col: int) -> None:
    mask = shape.mask
    mh, mw = mask.shape
    rows, cols = result.shape
    for dr in range(mh):
        for dc in range(mw):
            if mask[dr, dc]:
                r, c = target_row + dr, target_col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    result[r, c] = shape.color


# ---------------------------------------------------------------------------
# Composite: relocate_objects
# ---------------------------------------------------------------------------


def _relocate_objects(grid: Grid, match_rule: int, align: int) -> Grid:
    """All-in-one: extract + match + place + paint.

    Parameters are explicit integers, not hidden heuristics.
    """
    shapes, markers, bg = _get_shapes_and_markers(grid)
    if not shapes or not markers:
        return grid.copy()

    pairs = match_objects(shapes, markers, match_rule)

    result = np.full_like(grid, bg)
    for m in markers:
        mx, my = m.bbox[0], m.bbox[1]
        if 0 <= my < result.shape[0] and 0 <= mx < result.shape[1]:
            result[my, mx] = m.color

    for shape, marker in pairs:
        tr, tc = _compute_placement(shape, marker, align)
        _paint_shape_at(result, shape, tr, tc)

    return result


def _match_and_place(grid: Grid, match_rule: int, align: int = ALIGN_CENTER) -> Grid:
    """Compatibility alias for relocate_objects."""
    return _relocate_objects(grid, match_rule, align)


def _select_relate_paint(grid: Grid) -> Grid:
    """Legacy compatibility: relocate_objects with shape_nearest/center."""
    return _relocate_objects(grid, MATCH_SHAPE_NEAREST, ALIGN_CENTER)


def _select_and_separate(grid: Grid, size_threshold: int) -> Grid:
    if grid.size == 0:
        return grid.copy()
    bg = _detect_bg(grid)
    all_objs = list(_find_objects(grid))
    result = np.full_like(grid, bg)
    for obj in all_objs:
        if obj.size <= size_threshold:
            x, y, _, _ = obj.bbox
            if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                result[y, x] = obj.color
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "relocate_objects",
    OpSignature(params=(("grid", Type.GRID), ("match_rule", Type.INT), ("align", Type.INT)),
                return_type=Type.GRID),
    _relocate_objects,
)

register(
    "match_and_place",
    OpSignature(params=(("grid", Type.GRID), ("match_rule", Type.INT), ("align", Type.INT)),
                return_type=Type.GRID),
    _match_and_place,
)

register(
    "extract_shapes",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _extract_shapes,
)

register(
    "extract_markers",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _extract_markers,
)

register(
    "select_relate_paint",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _select_relate_paint,
)

register(
    "select_and_separate",
    OpSignature(params=(("grid", Type.GRID), ("size_threshold", Type.INT)),
                return_type=Type.GRID),
    _select_and_separate,
)
