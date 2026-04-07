"""Generalized rule induction: express rules in terms of structural roles,
not literal values. Accumulate across demos and apply to any new input.

A generalized rule says:
  "recolor target to marker's color" (not "recolor 3 to 6")
  "fill enclosed with frame's color" (not "fill with 4")
  "move object to nearest enclosed region" (not "move by (1,0)")

This generalizes to test inputs with unseen colors/positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from collections import Counter, deque

import numpy as np
from scipy import ndimage

from aria.guided.construct import construct_canvas, ConstructedCanvas
from aria.guided.residual_objects import analyze_residual_objects, ResidualObject
from aria.guided.workspace import _detect_bg, _extract_objects, ObjectInfo
from aria.types import Grid


# ---------------------------------------------------------------------------
# Generalized rule: a function described structurally
# ---------------------------------------------------------------------------

@dataclass
class GeneralizedRule:
    description: str
    apply_fn: Callable[[Grid], Grid]


def verify_rule(rule: GeneralizedRule, demos: list[tuple[Grid, Grid]]) -> tuple[bool, int]:
    total_diff = 0
    for inp, out in demos:
        try:
            pred = rule.apply_fn(inp)
        except Exception:
            return False, sum(o.size for _, o in demos)
        if pred.shape != out.shape:
            return False, sum(o.size for _, o in demos)
        total_diff += int(np.sum(pred != out))
    return total_diff == 0, total_diff


# ---------------------------------------------------------------------------
# Main entry: induce generalized rules from demos
# ---------------------------------------------------------------------------

def induce_generalized(
    demos: list[tuple[Grid, Grid]],
) -> GeneralizedRule | None:
    """Induce a generalized rule from train demos."""
    if not demos:
        return None

    candidates = []

    # Analyze residual objects
    try:
        residuals = analyze_residual_objects(demos)
    except Exception:
        return None

    # Collect cause types
    causes = set()
    for dr in residuals:
        for obj in dr.objects:
            causes.add(obj.cause)

    # Generate candidate generalized rules based on observed causes
    if 'SAME_POS_RECOLOR' in causes:
        candidates.extend(_gen_recolor_rules(residuals, demos))

    if 'FILLED' in causes:
        candidates.extend(_gen_fill_rules(residuals, demos))

    if 'MOVED' in causes:
        candidates.extend(_gen_move_rules(residuals, demos))

    # Also try composite: canvas + residual placement
    candidates.extend(_gen_canvas_plus_residual(residuals, demos))

    # Verify each candidate
    for rule in candidates:
        ok, diff = verify_rule(rule, demos)
        if ok:
            return rule

    # Return best partial match
    if candidates:
        best = min(candidates, key=lambda r: verify_rule(r, demos)[1])
        _, diff = verify_rule(best, demos)
        orig = sum(
            int(np.sum(inp != out)) if inp.shape == out.shape else out.size
            for inp, out in demos
        )
        if diff < orig:
            return best

    return None


# ---------------------------------------------------------------------------
# Recolor rules (generalized)
# ---------------------------------------------------------------------------

def _gen_recolor_rules(residuals, demos):
    rules = []

    # Rule: recolor target object to singleton marker's color
    if _check_singleton_determines_color(residuals, demos):
        rules.append(GeneralizedRule(
            "recolor target to marker color",
            _apply_recolor_to_marker,
        ))

    # Rule: recolor target to adjacent object's color
    if _check_adjacent_determines_color(residuals, demos):
        rules.append(GeneralizedRule(
            "recolor target to adjacent color",
            _apply_recolor_to_adjacent,
        ))

    # Rule: recolor target to enclosing object's color
    if _check_enclosing_determines_color(residuals, demos):
        rules.append(GeneralizedRule(
            "recolor target to enclosing color",
            _apply_recolor_to_enclosing,
        ))

    # Fallback: literal color map (accumulated across demos)
    cmap = _accumulate_color_map(residuals)
    if cmap:
        rules.append(GeneralizedRule(
            f"literal recolor {cmap}",
            lambda inp, m=dict(cmap): _apply_literal_recolor(inp, m),
        ))

    return rules


def _check_singleton_determines_color(residuals, demos):
    for dr, (inp, out) in zip(residuals, demos):
        bg = _detect_bg(inp)
        objs = _extract_objects(inp, bg)
        singletons = [o for o in objs if o.is_singleton]
        for robj in dr.objects:
            if robj.cause != 'SAME_POS_RECOLOR':
                continue
            if not any(s.color == robj.new_color for s in singletons):
                return False
    return True


def _check_adjacent_determines_color(residuals, demos):
    for dr, (inp, out) in zip(residuals, demos):
        bg = _detect_bg(inp)
        for robj in dr.objects:
            if robj.cause != 'SAME_POS_RECOLOR':
                continue
            adj = _get_adjacent_colors(robj, inp, bg)
            if robj.new_color not in adj:
                return False
    return True


def _check_enclosing_determines_color(residuals, demos):
    for dr, (inp, out) in zip(residuals, demos):
        bg = _detect_bg(inp)
        objs = _extract_objects(inp, bg)
        for robj in dr.objects:
            if robj.cause != 'SAME_POS_RECOLOR':
                continue
            enc_color = _find_enclosing_color(robj, objs)
            if enc_color != robj.new_color:
                return False
    return True


def _accumulate_color_map(residuals):
    cmap = {}
    for dr in residuals:
        for obj in dr.objects:
            if obj.cause == 'SAME_POS_RECOLOR' and obj.source_color >= 0:
                if obj.source_color in cmap and cmap[obj.source_color] != obj.new_color:
                    return None
                cmap[obj.source_color] = obj.new_color
    return cmap if cmap else None


# Recolor application functions

def _apply_recolor_to_marker(inp):
    bg = _detect_bg(inp)
    objs = _extract_objects(inp, bg)
    singletons = [o for o in objs if o.is_singleton]
    non_singletons = [o for o in objs if not o.is_singleton]
    if not singletons:
        return inp.copy()
    marker_color = singletons[0].color
    result = np.full_like(inp, bg)
    # Place non-singleton objects recolored to marker color
    for obj in non_singletons:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = marker_color
    return result


def _apply_recolor_to_adjacent(inp):
    bg = _detect_bg(inp)
    objs = _extract_objects(inp, bg)
    singletons = {(o.row, o.col): o for o in objs if o.is_singleton}
    result = inp.copy()
    for obj in objs:
        if obj.is_singleton:
            continue
        adj_c = set()
        for r in range(obj.height):
            for c in range(obj.width):
                if not obj.mask[r, c]:
                    continue
                gr, gc = obj.row + r, obj.col + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (gr + dr, gc + dc) in singletons:
                        adj_c.add(singletons[(gr + dr, gc + dc)].color)
        if len(adj_c) == 1:
            tc = next(iter(adj_c))
            if tc != obj.color:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            result[obj.row + r, obj.col + c] = tc
    return result


def _apply_recolor_to_enclosing(inp):
    bg = _detect_bg(inp)
    objs = _extract_objects(inp, bg)
    result = inp.copy()
    for obj in objs:
        enc_c = _find_enclosing_color(
            type('O', (), {'row': obj.row, 'col': obj.col,
                           'height': obj.height, 'width': obj.width})(),
            objs,
        )
        if enc_c >= 0 and enc_c != obj.color:
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.mask[r, c]:
                        result[obj.row + r, obj.col + c] = enc_c
    return result


def _apply_literal_recolor(inp, cmap):
    lut = np.arange(256, dtype=np.uint8)
    for fc, tc in cmap.items():
        lut[fc] = tc
    return lut[inp]


# ---------------------------------------------------------------------------
# Fill rules (generalized)
# ---------------------------------------------------------------------------

def _gen_fill_rules(residuals, demos):
    rules = []

    # Rule: fill enclosed with enclosing frame's color
    rules.append(GeneralizedRule(
        "fill enclosed with frame color",
        _apply_fill_with_frame,
    ))

    # Rule: fill enclosed with a literal color (accumulated)
    fill_colors = set()
    for dr in residuals:
        for obj in dr.objects:
            if obj.cause == 'FILLED':
                fill_colors.add(obj.new_color)
    if len(fill_colors) == 1:
        fc = next(iter(fill_colors))
        rules.append(GeneralizedRule(
            f"fill enclosed with {fc}",
            lambda inp, c=fc: _apply_fill_literal(inp, c),
        ))

    return rules


def _apply_fill_with_frame(inp):
    bg = _detect_bg(inp)
    out = inp.copy()
    enclosed = _get_enclosed(inp, bg)
    if not np.any(enclosed):
        return out
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for lid in range(1, n + 1):
        comp = labeled == lid
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        vals = inp[border]
        non_bg = vals[vals != bg]
        if len(non_bg) > 0:
            out[comp] = int(Counter(non_bg.tolist()).most_common(1)[0][0])
    return out


def _apply_fill_literal(inp, fill_color):
    bg = _detect_bg(inp)
    out = inp.copy()
    enclosed = _get_enclosed(inp, bg)
    out[enclosed] = fill_color
    return out


# ---------------------------------------------------------------------------
# Move rules (generalized)
# ---------------------------------------------------------------------------

def _gen_move_rules(residuals, demos):
    rules = []

    # Check: consistent offset across all moved objects in all demos
    offsets = set()
    for dr in residuals:
        for obj in dr.objects:
            if obj.cause == 'MOVED' and obj.source_obj is not None:
                off = (obj.row - obj.source_obj.row, obj.col - obj.source_obj.col)
                offsets.add(off)

    if len(offsets) == 1:
        dr, dc = next(iter(offsets))
        rules.append(GeneralizedRule(
            f"move all by ({dr},{dc})",
            lambda inp, d=dr, c=dc: _apply_move(inp, d, c),
        ))

    return rules


def _apply_move(inp, dr, dc):
    bg = _detect_bg(inp)
    objs = _extract_objects(inp, bg)
    result = np.full_like(inp, bg)
    for obj in objs:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    nr, nc = obj.row + r + dr, obj.col + c + dc
                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                        result[nr, nc] = obj.color
    return result


# ---------------------------------------------------------------------------
# Canvas + residual placement (generalized)
# ---------------------------------------------------------------------------

def _gen_canvas_plus_residual(residuals, demos):
    """Build canvas, then place residual objects using structural rules."""
    rules = []

    # Accumulate: for each residual object, what's the relationship between
    # source_color and new_color in structural terms?
    # If new_color = some_singleton's color across all demos → marker rule
    all_marker = True
    for dr, (inp, out) in zip(residuals, demos):
        bg = _detect_bg(inp)
        objs = _extract_objects(inp, bg)
        singletons = {o.color for o in objs if o.is_singleton}
        for robj in dr.objects:
            if robj.cause == 'SAME_POS_RECOLOR':
                if robj.new_color not in singletons:
                    all_marker = False

    if all_marker and any(
        any(o.cause == 'SAME_POS_RECOLOR' for o in dr.objects)
        for dr in residuals
    ):
        rules.append(GeneralizedRule(
            "canvas + recolor targets to marker color",
            _apply_canvas_recolor_marker,
        ))

    return rules


def _apply_canvas_recolor_marker(inp):
    bg = _detect_bg(inp)
    objs = _extract_objects(inp, bg)
    singletons = [o for o in objs if o.is_singleton]
    if not singletons:
        return inp.copy()
    marker_color = singletons[0].color

    # Build canvas: bg + preserved objects (everything that isn't a target or marker)
    result = np.full_like(inp, bg)
    # Place non-singleton objects recolored to marker color
    for obj in objs:
        if obj.is_singleton:
            continue  # marker — don't copy
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = marker_color
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_adjacent_colors(robj, inp, bg):
    obj_mask = np.zeros(inp.shape, dtype=bool)
    r0, c0 = robj.row, robj.col
    h, w = robj.height, robj.width
    r1 = min(r0 + h, inp.shape[0])
    c1 = min(c0 + w, inp.shape[1])
    mask = robj.mask[:r1 - r0, :c1 - c0]
    obj_mask[r0:r1, c0:c1] |= mask
    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    dilated = ndimage.binary_dilation(obj_mask, structure=struct4)
    border = dilated & ~obj_mask
    return set(int(v) for v in np.unique(inp[border])) - {bg}


def _find_enclosing_color(robj, objs):
    for obj in objs:
        if (obj.row <= robj.row and obj.col <= robj.col and
                obj.row + obj.height >= robj.row + robj.height and
                obj.col + obj.width >= robj.col + robj.width and
                hasattr(obj, 'size') and obj.size > getattr(robj, 'size', 0)):
            return obj.color
    return -1


def _get_enclosed(grid, bg):
    rows, cols = grid.shape
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return (grid == bg) & ~reachable
