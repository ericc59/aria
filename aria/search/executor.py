"""Canonical AST executor for the search engine.

Executes ASTNode trees against input grids using aria.guided primitives.
No closures — all behavior is determined by the AST structure.

Execution is recursive: each node evaluates its children, then applies its op.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import Grid
from aria.search.ast import ASTNode, Op


def execute_ast(node: ASTNode, inp: Grid, ctx: dict = None) -> Any:
    """Execute an AST node against an input grid.

    Returns the value produced by the node (Grid, Region, Object, Color, etc.)
    Returns None on failure (missing object, bad selection, etc.)
    """
    if ctx is None:
        ctx = {}

    op = node.op

    # --- Leaves ---
    if op == Op.INPUT:
        return inp
    if op == Op.CONST_COLOR or op == Op.CONST_INT:
        return node.param

    # --- Perception ---
    if op == Op.PERCEIVE:
        from aria.guided.perceive import perceive
        grid = _child(node, 0, inp, ctx)
        return perceive(grid) if grid is not None else None

    if op == Op.SELECT:
        from aria.guided.dsl import prim_select
        facts = _child(node, 0, inp, ctx)
        if facts is None:
            return None
        targets = prim_select(facts, node.param)
        return targets[0] if len(targets) == 1 else None

    if op == Op.SELECT_IDX:
        facts = _child(node, 0, inp, ctx)
        if facts is None:
            return None
        idx = node.param
        return facts.objects[idx] if 0 <= idx < len(facts.objects) else None

    # --- Extraction ---
    if op == Op.CROP_BBOX:
        from aria.guided.dsl import prim_crop_bbox
        grid = _child(node, 0, inp, ctx)
        obj = _child(node, 1, inp, ctx)
        return prim_crop_bbox(grid, obj) if grid is not None and obj is not None else None

    if op == Op.CROP_INTERIOR:
        from aria.guided.dsl import prim_find_frame, prim_crop_interior
        grid = _child(node, 0, inp, ctx)
        obj = _child(node, 1, inp, ctx)
        if grid is None or obj is None:
            return None
        frame = prim_find_frame(obj, grid)
        return prim_crop_interior(grid, frame) if frame else None

    if op == Op.SPLIT:
        from aria.guided.dsl import prim_split_at_separator
        from aria.guided.perceive import perceive
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        facts = perceive(grid)
        if not facts.separators:
            return None
        return prim_split_at_separator(grid, facts, 0)

    if op == Op.FIRST:
        pair = _child(node, 0, inp, ctx)
        return pair[0] if pair else None

    if op == Op.SECOND:
        pair = _child(node, 0, inp, ctx)
        return pair[1] if pair else None

    # --- Region ops ---
    if op == Op.COMBINE:
        from aria.guided.dsl import prim_combine
        from aria.guided.perceive import perceive
        pair = _child(node, 0, inp, ctx)
        if pair is None:
            return None
        a, b = pair
        bg = perceive(a).bg if hasattr(a, 'shape') and a.size > 0 else 0
        return prim_combine(a, b, node.param, bg)

    if op == Op.RENDER:
        from aria.guided.dsl import prim_render_mask
        from aria.guided.perceive import perceive
        mask = _child(node, 0, inp, ctx)
        color = _child(node, 1, inp, ctx)
        if mask is None:
            return None
        bg = perceive(inp).bg
        return prim_render_mask(mask, color, bg)

    # --- Transforms ---
    _XFORM_FNS = {
        Op.FLIP_H: lambda g: g[:, ::-1],
        Op.FLIP_V: lambda g: g[::-1, :],
        Op.FLIP_HV: lambda g: g[::-1, ::-1],
        Op.ROT90: lambda g: np.rot90(g),
        Op.ROT180: lambda g: np.rot90(g, 2),
        Op.TRANSPOSE: lambda g: g.T,
    }
    if op in _XFORM_FNS:
        grid = _child(node, 0, inp, ctx)
        return _XFORM_FNS[op](grid) if grid is not None else None

    # --- Grid constructors ---
    if op == Op.REPAIR_FRAMES:
        from aria.guided.synthesize import _repair_all_frames
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        result = _repair_all_frames(grid)
        return result if result is not None else grid

    # --- Panel operations ---
    if op == Op.PANEL_ODD_SELECT:
        from aria.search.panel_ops import panel_odd_select
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        return panel_odd_select(grid)

    if op == Op.PANEL_MAJORITY_SELECT:
        from aria.search.panel_ops import panel_majority_select
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        return panel_majority_select(grid)

    if op == Op.PANEL_BOOLEAN:
        from aria.search.panel_ops import panel_boolean_combine
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        param = node.param or {}
        if isinstance(param, tuple) and len(param) >= 2:
            op_name, color = param[0], param[1]
            sep_rows = param[2] if len(param) > 2 else None
            sep_cols = param[3] if len(param) > 3 else None
        else:
            op_name, color, sep_rows, sep_cols = 'xor', 0, None, None
        return panel_boolean_combine(grid, op_name, color, sep_rows=sep_rows, sep_cols=sep_cols)

    if op == Op.PANEL_REPAIR:
        from aria.search.panel_ops import panel_periodic_repair
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        result = panel_periodic_repair(grid)
        return result if result is not None else grid

    # --- Repair ---
    if op == Op.SYMMETRY_REPAIR:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        damage_color = node.param
        return _exec_symmetry_repair(grid, damage_color)

    # --- Object repacking ---
    if op == Op.OBJECT_REPACK:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_object_repack(grid, params)

    # --- Object-level actions ---
    if op == Op.RECOLOR:
        return _exec_recolor(node, inp, ctx)
    if op == Op.REMOVE:
        return _exec_remove(node, inp, ctx)
    if op == Op.MOVE:
        return _exec_move(node, inp, ctx)
    if op == Op.GRAVITY:
        return _exec_gravity(node, inp, ctx)

    # --- Composition ---
    if op == Op.COMPOSE:
        result = inp
        for child in node.children:
            result = execute_ast(child, result, ctx)
            if result is None:
                return None
        return result

    if op == Op.MIRROR:
        from aria.guided.dsl import prim_mirror_across
        from aria.guided.perceive import perceive
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return grid
        facts = perceive(grid)
        if not facts.separators:
            return grid
        sep = facts.separators[0]
        mode = node.param
        if mode and mode.startswith('col') and sep.axis != 'col':
            return grid
        if mode and mode.startswith('row') and sep.axis != 'row':
            return grid
        return prim_mirror_across(grid, mode, sep.index)

    if op == Op.HOLE:
        return None

    raise ValueError(f"Unimplemented AST op: {op}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _child(node, idx, inp, ctx):
    """Execute the idx-th child, returning None if missing."""
    if idx < len(node.children):
        return execute_ast(node.children[idx], inp, ctx)
    return None


def _exec_object_repack(grid, params):
    """Repack objects into a new layout.

    params:
        ordering: 'chain' (spatial adjacency), 'spatial' (top-left to bottom-right), 'size' (largest first)
        layout: 'column' (vertical stack), 'row' (horizontal stack)
        payload: 'color_by_area' (color repeated by pixel count), 'bbox' (crop of each object)
    """
    from aria.guided.perceive import perceive
    facts = perceive(grid)
    if not facts.objects:
        return None

    ordering = params.get('ordering', 'chain')
    layout = params.get('layout', 'column')
    payload = params.get('payload', 'color_by_area')

    # Order objects
    if ordering == 'chain':
        ordered = _chain_order_objects(list(facts.objects))
    elif ordering == 'spatial':
        ordered = sorted(facts.objects, key=lambda o: (o.row, o.col))
    elif ordering == 'size':
        ordered = sorted(facts.objects, key=lambda o: -o.size)
    else:
        ordered = list(facts.objects)

    if not ordered:
        return None

    # Build payload per object
    if payload == 'color_by_area':
        segments = [np.full((obj.size, 1), obj.color, dtype=grid.dtype) for obj in ordered]
    elif payload == 'bbox':
        from aria.guided.dsl import prim_crop_bbox
        segments = [prim_crop_bbox(grid, obj) for obj in ordered]
    else:
        return None

    # Layout
    if layout == 'column':
        return np.vstack(segments)
    elif layout == 'row':
        return np.hstack(segments)

    return None


def _chain_order_objects(objs):
    """Order objects by spatial chain: start topmost-leftmost, follow by proximity."""
    if not objs:
        return []
    remaining = set(range(len(objs)))
    current = min(remaining, key=lambda i: (objs[i].row, objs[i].col))
    ordered = [objs[current]]
    remaining.remove(current)

    while remaining:
        cur = ordered[-1]
        best = None
        best_dist = 9999
        for j in remaining:
            o = objs[j]
            dr = max(0, max(cur.row, o.row) - min(cur.row + cur.height, o.row + o.height))
            dc = max(0, max(cur.col, o.col) - min(cur.col + cur.width, o.col + o.width))
            dist = dr + dc
            if dist < best_dist:
                best_dist = dist
                best = j
        if best is not None:
            ordered.append(objs[best])
            remaining.remove(best)

    return ordered


def _exec_symmetry_repair(grid, damage_color):
    """Repair cells of damage_color using the grid's symmetry.

    For each damaged cell, copy from the closest undamaged symmetric position.
    Tries all D4 symmetry positions (flips + rotations).
    """
    h, w = grid.shape
    damage = (grid == damage_color)
    result = grid.copy()

    # All D4 symmetric positions for (r, c) in an h×w grid
    def sym_positions(r, c):
        positions = [
            (r, w - 1 - c),       # flip horizontal
            (h - 1 - r, c),       # flip vertical
            (h - 1 - r, w - 1 - c),  # flip both
        ]
        if h == w:
            positions.extend([
                (c, r),               # transpose
                (w - 1 - c, r),       # rot90
                (c, h - 1 - r),       # rot270
                (w - 1 - c, h - 1 - r),  # transpose + flip
            ])
        return positions

    for r in range(h):
        for c in range(w):
            if damage[r, c]:
                for mr, mc in sym_positions(r, c):
                    if 0 <= mr < h and 0 <= mc < w and not damage[mr, mc]:
                        result[r, c] = grid[mr, mc]
                        break

    return result


def _exec_recolor(node, inp, ctx):
    from aria.guided.dsl import prim_select
    from aria.guided.perceive import perceive
    grid = _child(node, 0, inp, ctx)
    if grid is None:
        return None
    facts = perceive(grid)
    sel_preds, new_color = node.param
    result = grid.copy()
    for obj in prim_select(facts, sel_preds):
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = new_color
    return result


def _exec_remove(node, inp, ctx):
    from aria.guided.dsl import prim_select
    from aria.guided.perceive import perceive
    grid = _child(node, 0, inp, ctx)
    if grid is None:
        return None
    facts = perceive(grid)
    result = grid.copy()
    for obj in prim_select(facts, node.param):
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = facts.bg
    return result


def _exec_move(node, inp, ctx):
    from aria.guided.dsl import prim_select
    from aria.guided.perceive import perceive
    grid = _child(node, 0, inp, ctx)
    if grid is None:
        return None
    facts = perceive(grid)
    sel_preds, dr, dc = node.param
    result = grid.copy()
    rows, cols = grid.shape
    targets = prim_select(facts, sel_preds)
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = facts.bg
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    nr, nc = obj.row + r + dr, obj.col + c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = obj.color
    return result


def _exec_gravity(node, inp, ctx):
    from aria.guided.dsl import prim_select
    from aria.guided.perceive import perceive
    grid = _child(node, 0, inp, ctx)
    if grid is None:
        return None
    facts = perceive(grid)
    sel_preds, direction = node.param
    result = grid.copy()
    rows, cols = grid.shape
    targets = prim_select(facts, sel_preds)
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = facts.bg
    for obj in targets:
        h, w = obj.height, obj.width
        if direction == 'down':
            nr, nc = rows - h, obj.col
        elif direction == 'up':
            nr, nc = 0, obj.col
        elif direction == 'right':
            nr, nc = obj.row, cols - w
        elif direction == 'left':
            nr, nc = obj.row, 0
        else:
            nr, nc = obj.row, obj.col
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    pr, pc = nr + r, nc + c
                    if 0 <= pr < rows and 0 <= pc < cols:
                        result[pr, pc] = obj.color
    return result
