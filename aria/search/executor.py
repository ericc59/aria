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

    # --- Region decode ---
    if op == Op.ANOMALY_HALO:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_anomaly_halo(grid, params)

    if op == Op.OBJECT_HIGHLIGHT:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_object_highlight_ast(grid, params)

    if op == Op.LEGEND_CHAIN_CONNECT:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_legend_chain_connect(grid, params)

    if op == Op.DIAGONAL_COLLISION_TRACE:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_diagonal_collision_trace(grid, params)

    if op == Op.MASKED_PATCH_TRANSFER:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_masked_patch_transfer(grid, params)

    if op == Op.SEPARATOR_MOTIF_BROADCAST:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_separator_motif_broadcast(grid, params)

    if op == Op.LINE_ARITH_BROADCAST:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_line_arith_broadcast(grid, params)

    if op == Op.BARRIER_PORT_TRANSFER:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_barrier_port_transfer(grid, params)

    if op == Op.CAVITY_TRANSFER:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        return _exec_cavity_transfer(grid, node.param or {})

    if op == Op.LEGEND_FRAME_FILL:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_legend_frame_fill(grid, params)

    if op == Op.CROSS_STENCIL_RECOLOR:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_cross_stencil_recolor(grid, params)

    if op == Op.FRAME_BBOX_PACK:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_frame_bbox_pack_ast(grid, params)

    if op == Op.QUADRANT_TEMPLATE_DECODE:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        params = node.param or {}
        return _exec_quadrant_template_decode(grid, params)

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
    # param is either: sel_preds (old format) or (sel_preds, bg_override)
    param = node.param
    if isinstance(param, tuple) and len(param) == 2 and (param[1] is None or isinstance(param[1], int)):
        sel_preds, bg_override = param
    else:
        sel_preds = param
        bg_override = None

    facts = perceive(grid)
    bg = bg_override if bg_override is not None else facts.bg
    result = grid.copy()
    for obj in prim_select(facts, sel_preds):
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg
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


def _exec_anomaly_halo(grid, params):
    """Decorate anomaly cells with a halo of a new color."""
    halo_color = params.get('halo_color')
    if halo_color is None:
        return grid

    c1 = params.get('c1')
    c2 = params.get('c2')
    h, w = grid.shape
    if c1 is None or c2 is None:
        from collections import Counter
        counts = Counter(int(grid[r, c]) for r in range(h) for c in range(w))
        top2 = counts.most_common(2)
        if len(top2) < 2:
            return grid
        c1, c2 = top2[0][0], top2[1][0]

    h, w = grid.shape
    result = grid.copy()

    # Detect per-row dominant color to identify band structure
    row_dominant = []
    for r in range(h):
        n1 = sum(1 for c in range(w) if int(grid[r, c]) == c1)
        n2 = sum(1 for c in range(w) if int(grid[r, c]) == c2)
        row_dominant.append(c1 if n1 > n2 else c2)

    # Anomaly = cell whose color differs from its row's dominant color
    anom_minor = []  # c2 cells in c1-dominant rows → holes (get halos)
    anom_major = []  # c1 cells in c2-dominant rows → noise (get removed)
    for r in range(h):
        dom = row_dominant[r]
        for c in range(w):
            v = int(grid[r, c])
            if v == c2 and dom == c1:
                anom_minor.append((r, c))
            elif v == c1 and dom == c2:
                anom_major.append((r, c))

    # Remove c1-in-c2 noise
    for r, c in anom_major:
        result[r, c] = c2

    # Halo around c2-in-c1 anomalies (color all neighbors except other anomalies)
    anomaly_set = set(anom_minor)
    for r, c in anom_minor:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in anomaly_set:
                    result[nr, nc] = halo_color

    return result


def _exec_cavity_transfer_auto_QUARANTINED(grid):
    """QUARANTINED: monolithic auto-detect solver. Use _exec_cavity_transfer with
    explicit pairs derived at search time instead.

    Kept for reference only — not called from any live code path.
    """
    from aria.guided.perceive import perceive
    from collections import defaultdict, deque
    from aria.search.geometry import axis_ray, bbox, boundary_cells, closest_boundary_anchor, component_center

    def _components8(cells):
        remaining = set(cells)
        comps = []
        while remaining:
            start = remaining.pop()
            comp = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nxt = (r + dr, c + dc)
                        if nxt in remaining:
                            remaining.remove(nxt)
                            comp.add(nxt)
                            stack.append(nxt)
            comps.append(comp)
        return comps

    def _marker_leaks_outside(inside, outside):
        if not outside:
            return False
        for r, c in inside:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in outside:
                    return True
        return False

    def _cavity_open_sides(host_cells, inside_cells):
        bounds = bbox(host_cells)
        top, bot = bounds.top, bounds.bottom
        left, right = bounds.left, bounds.right
        q = deque(inside_cells)
        seen = set(inside_cells)
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (top <= nr <= bot and left <= nc <= right):
                    continue
                if (nr, nc) in host_cells or (nr, nc) in seen:
                    continue
                if grid[nr, nc] != bg and (nr, nc) not in inside_cells:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))

        open_sides = set()
        if any(r == top for r, _ in seen):
            open_sides.add('up')
        if any(r == bot for r, _ in seen):
            open_sides.add('down')
        if any(c == left for _, c in seen):
            open_sides.add('left')
        if any(c == right for _, c in seen):
            open_sides.add('right')
        return open_sides, bounds

    def _direction_order(host_cells, inside_cells, open_sides, bounds):
        top, bot, left, right = bounds.top, bounds.bottom, bounds.left, bounds.right
        m_cr, m_cc = component_center(inside_cells)
        opposite = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left',
        }

        def _dist_to_side(direction):
            if direction == 'up':
                return m_cr - top
            if direction == 'down':
                return bot - m_cr
            if direction == 'left':
                return m_cc - left
            return right - m_cc

        if len(open_sides) == 1:
            candidate_dirs = [opposite[next(iter(open_sides))]]
        else:
            candidate_dirs = [d for d in ['up', 'down', 'left', 'right'] if d not in open_sides]
        if not candidate_dirs:
            candidate_dirs = ['up', 'down', 'left', 'right']
        return sorted(
            candidate_dirs,
            key=lambda d: (len(boundary_cells(host_cells, d)), -_dist_to_side(d)),
        )

    def _tip_anchor(host_cells, inside_cells, direction, bounds):
        m_cr, m_cc = component_center(inside_cells)
        return closest_boundary_anchor(
            host_cells,
            direction,
            target_row=m_cr,
            target_col=m_cc,
        )

    def _project_cells(direction, anchor, bounds, inside_cells):
        if anchor is None:
            return []
        if direction == 'up':
            start = (bounds.top, anchor)
        elif direction == 'down':
            start = (bounds.bottom, anchor)
        elif direction == 'left':
            start = (anchor, bounds.left)
        else:
            start = (anchor, bounds.right)

        projected = []
        for nr, nc in axis_ray(start, direction, (h, w))[:len(inside_cells)]:
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                projected.append((nr, nc))
            else:
                break
        return projected

    facts = perceive(grid)
    bg = facts.bg
    h, w = grid.shape

    color_cells = defaultdict(set)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                color_cells[int(grid[r, c])].add((r, c))

    host_groups = []
    for color, cells in color_cells.items():
        for comp in _components8(cells):
            if len(comp) >= 4:
                host_groups.append((color, comp))
    host_groups.sort(key=lambda item: -len(item[1]))

    result = grid.copy()
    used_marker_cells = set()

    for host_color, host_cells in host_groups:
        host_bounds = bbox(host_cells)
        top, bot = host_bounds.top, host_bounds.bottom
        left, right = host_bounds.left, host_bounds.right

        for mc, mc_cells in color_cells.items():
            if mc == host_color:
                continue
            inside = [(r, c) for r, c in mc_cells
                      if top <= r <= bot and left <= c <= right and (r, c) not in used_marker_cells]
            if not inside or len(inside) < 1:
                continue
            if len(inside) >= len(host_cells):
                continue
            inside_set = set(inside)
            outside_set = mc_cells - inside_set
            if _marker_leaks_outside(inside, outside_set):
                continue

            open_sides, bounds = _cavity_open_sides(host_cells, inside_set)
            dir_order = _direction_order(host_cells, inside_set, open_sides, bounds)

            for direction in dir_order:
                anchor = _tip_anchor(host_cells, inside_set, direction, bounds)
                new_cells = _project_cells(direction, anchor, bounds, inside_set)
                if not new_cells:
                    continue

                # Apply: remove inside markers, place at new positions
                for r, c in inside:
                    result[r, c] = bg
                for nr, nc in new_cells:
                    result[nr, nc] = mc
                used_marker_cells.update(inside_set)
                break  # found direction for this marker/host pair

    return result


def _exec_cavity_transfer(grid, params):
    """Transfer marker cells from inside concave host to the opposite open end.

    Params:
        pairs: list of (marker_color, host_color, direction) tuples
        bg: optional background color (derived from grid border if absent)
    """
    pairs = params.get('pairs', [])
    if not pairs:
        mc = params.get('marker_color')
        hc = params.get('host_color')
        d = params.get('direction')
        if mc is not None and hc is not None and d:
            pairs = [(mc, hc, d)]
        else:
            return grid

    bg = params.get('bg')
    if bg is None:
        border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
        bg = int(np.bincount(border).argmax())
    h, w = grid.shape
    result = grid.copy()

    for marker_color, host_color, direction in pairs:
        host = set((r, c) for r in range(h) for c in range(w) if grid[r, c] == host_color)
        markers = set((r, c) for r in range(h) for c in range(w) if grid[r, c] == marker_color)
        if not host or not markers:
            continue

        hr = [r for r, c in host]
        hc = [c for r, c in host]
        top, bot = min(hr), max(hr)
        left, right = min(hc), max(hc)

        inside = [(r, c) for r, c in markers if top <= r <= bot and left <= c <= right]
        if not inside:
            continue

        # Remove marker cells inside the host
        for r, c in inside:
            result[r, c] = bg

        # Compute new positions
        if direction == 'up':
            top_cells = sorted([(r, c) for r, c in host if r == top])
            tip_col = top_cells[0][1] if top_cells else left
            n = len(set(r for r, c in inside))
            for i in range(n):
                nr = top - 1 - i
                if 0 <= nr < h:
                    result[nr, tip_col] = marker_color
        elif direction == 'down':
            bot_cells = sorted([(r, c) for r, c in host if r == bot])
            tip_col = bot_cells[0][1] if bot_cells else left
            n = len(set(r for r, c in inside))
            for i in range(n):
                nr = bot + 1 + i
                if 0 <= nr < h:
                    result[nr, tip_col] = marker_color
        elif direction == 'right':
            tip_row = max(r for r, c in host if c == right)
            n = len(set(c for r, c in inside))
            for i in range(n):
                nc = right + 1 + i
                if 0 <= nc < w:
                    result[tip_row, nc] = marker_color
        elif direction == 'left':
            tip_row = max(r for r, c in host if c == left)
            n = len(set(c for r, c in inside))
            for i in range(n):
                nc = left - 1 - i
                if 0 <= nc < w:
                    result[tip_row, nc] = marker_color

    return result


def _exec_legend_frame_fill(grid, params):
    """Fill enclosed bg regions with derived colors.

    Params (derived at search time for enclosed_fill strategy):
        strategy: 'enclosed_fill' or 'color_map'
        legend_map: dict mapping wall_color -> fill_color
        wall_colors: list of wall color ints

    Falls back to auto-detect if legend_map/wall_colors not provided.
    """
    from aria.guided.perceive import perceive
    from collections import deque

    strategy = params.get('strategy', 'color_map')

    if strategy == 'enclosed_fill':
        legend_map = params.get('legend_map')
        wall_colors_list = params.get('wall_colors')
        if legend_map is not None and wall_colors_list is not None:
            return _exec_enclosed_fill_parameterized(grid, legend_map, set(wall_colors_list))
        # Legacy fallback
        return _exec_enclosed_fill_auto_QUARANTINED(grid)

    # Original color_map strategy
    color_map = params.get('color_map', {})
    if not color_map:
        return grid
    from aria.guided.dsl import prim_find_frame
    cmap = {int(k): int(v) for k, v in color_map.items()}
    facts = perceive(grid)
    bg = facts.bg
    result = grid.copy()
    for obj in facts.objects:
        frame = prim_find_frame(obj, grid)
        if not frame:
            continue
        r0, c0, r1, c1 = frame
        fcolor = obj.color
        if fcolor not in cmap:
            continue
        fill = cmap[fcolor]
        for r in range(r0, r1):
            for c in range(c0, c1):
                if result[r, c] == bg:
                    result[r, c] = fill
    return result


def _exec_enclosed_fill_parameterized(grid, legend_map, wall_colors):
    """Fill enclosed bg regions using pre-computed legend_map and wall_colors.

    Args:
        grid: input grid
        legend_map: dict mapping wall_color -> fill_color (from derive)
        wall_colors: set of wall colors that enclose bg regions
    """
    from aria.guided.perceive import perceive
    from collections import deque

    facts = perceive(grid)
    bg = facts.bg
    h, w = grid.shape

    def _flood_border(g, bg_val):
        gh, gw = g.shape
        reachable = np.zeros((gh, gw), dtype=bool)
        q = deque()
        for r in range(gh):
            for c in [0, gw - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        for c in range(gw):
            for r in [0, gh - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gh and 0 <= nc < gw and not reachable[nr, nc] and g[nr, nc] == bg_val:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        return ~reachable & (g == bg_val)

    # Normalize legend_map keys to int
    lmap = {int(k): int(v) for k, v in legend_map.items()}
    reverse_map = {v: k for k, v in lmap.items()}

    result = grid.copy()
    for wc in wall_colors:
        wg = np.where(grid == wc, wc, bg)
        enclosed = _flood_border(wg, bg)
        if enclosed.sum() > 0:
            if wc in lmap:
                fill = lmap[wc]
            elif wc in reverse_map:
                fill = reverse_map[wc]
            else:
                fill = wc  # fallback
            result[enclosed] = fill

    return result


def _exec_enclosed_fill_auto_QUARANTINED(grid):
    """QUARANTINED: monolithic auto-detect enclosed fill. Use _exec_legend_frame_fill
    with explicit legend_map/wall_colors derived at search time instead.

    Kept for reference only -- not called from any live code path.
    """
    from aria.guided.perceive import perceive
    from collections import deque

    facts = perceive(grid)
    bg = facts.bg
    h, w = grid.shape

    def _flood_border(g, bg_val):
        gh, gw = g.shape
        reachable = np.zeros((gh, gw), dtype=bool)
        q = deque()
        for r in range(gh):
            for c in [0, gw - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        for c in range(gw):
            for r in [0, gh - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gh and 0 <= nc < gw and not reachable[nr, nc] and g[nr, nc] == bg_val:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        return ~reachable & (g == bg_val)

    from aria.guided.dsl import prim_find_frame
    non_bg_colors = set(int(grid[r, c]) for r in range(h) for c in range(w) if grid[r, c] != bg)
    wall_colors = set()
    frame_colors = set()
    for obj in facts.objects:
        if prim_find_frame(obj, grid):
            frame_colors.add(obj.color)
    for wc in non_bg_colors:
        wg = np.where(grid == wc, wc, bg)
        enc = _flood_border(wg, bg)
        if enc.sum() > 0:
            wall_colors.add(wc)

    legend_map = {}
    best_legend_size = 0

    for r0 in range(h - 1):
        for c0 in range(w):
            for c1 in range(c0 + 2, min(c0 + 12, w + 1)):
                block = grid[r0:r0 + 2, c0:c1]
                if np.any(block == bg):
                    continue
                nc = len(set(int(block[r, c]) for r in range(2) for c in range(c1 - c0)))
                if nc < 3:
                    continue
                cand_map = {}
                for c in range(c1 - c0):
                    a, b = int(block[0, c]), int(block[1, c])
                    cand_map[a] = b
                if len(cand_map) > best_legend_size and any(k in wall_colors for k in cand_map):
                    legend_map = cand_map
                    best_legend_size = len(cand_map)

    for c0 in range(w - 1):
        for r0 in range(h):
            for r1 in range(r0 + 2, min(r0 + 12, h + 1)):
                block = grid[r0:r1, c0:c0 + 2]
                if np.any(block == bg):
                    continue
                nc = len(set(int(block[r, c]) for r in range(r1 - r0) for c in range(2)))
                if nc < 3:
                    continue
                cand_map = {}
                for r in range(r1 - r0):
                    a, b = int(block[r, 0]), int(block[r, 1])
                    cand_map[a] = b
                if len(cand_map) > best_legend_size and any(k in wall_colors for k in cand_map):
                    legend_map = cand_map
                    best_legend_size = len(cand_map)

    reverse_map = {v: k for k, v in legend_map.items()}

    result = grid.copy()
    for wc in wall_colors:
        if wc not in frame_colors:
            continue
        wg = np.where(grid == wc, wc, bg)
        enclosed = _flood_border(wg, bg)
        if enclosed.sum() > 0:
            if wc in legend_map:
                fill = legend_map[wc]
            elif wc in reverse_map:
                fill = reverse_map[wc]
            else:
                fill = wc
            result[enclosed] = fill

    return result


def _extract_enclosed_fill_params(grid):
    """Extract legend_map and wall_colors from a grid for enclosed_fill.

    Returns (legend_map, wall_colors) or None if extraction fails.
    Used by derive to pre-compute params at search time.
    """
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_find_frame
    from collections import deque

    facts = perceive(grid)
    bg = facts.bg
    h, w = grid.shape

    def _flood_border(g, bg_val):
        gh, gw = g.shape
        reachable = np.zeros((gh, gw), dtype=bool)
        q = deque()
        for r in range(gh):
            for c in [0, gw - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        for c in range(gw):
            for r in [0, gh - 1]:
                if g[r, c] == bg_val and not reachable[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gh and 0 <= nc < gw and not reachable[nr, nc] and g[nr, nc] == bg_val:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        return ~reachable & (g == bg_val)

    non_bg_colors = set(int(grid[r, c]) for r in range(h) for c in range(w) if grid[r, c] != bg)
    wall_colors = set()
    frame_colors = set()
    for obj in facts.objects:
        if prim_find_frame(obj, grid):
            frame_colors.add(obj.color)
    for wc in non_bg_colors:
        wg = np.where(grid == wc, wc, bg)
        enc = _flood_border(wg, bg)
        if enc.sum() > 0:
            wall_colors.add(wc)

    # Only keep wall_colors that are also frame colors
    wall_colors = wall_colors & frame_colors
    if not wall_colors:
        return None

    legend_map = {}
    best_legend_size = 0

    for r0 in range(h - 1):
        for c0 in range(w):
            for c1 in range(c0 + 2, min(c0 + 12, w + 1)):
                block = grid[r0:r0 + 2, c0:c1]
                if np.any(block == bg):
                    continue
                nc = len(set(int(block[r, c]) for r in range(2) for c in range(c1 - c0)))
                if nc < 3:
                    continue
                cand_map = {}
                for c in range(c1 - c0):
                    a, b = int(block[0, c]), int(block[1, c])
                    cand_map[a] = b
                if len(cand_map) > best_legend_size and any(k in wall_colors for k in cand_map):
                    legend_map = cand_map
                    best_legend_size = len(cand_map)

    for c0 in range(w - 1):
        for r0 in range(h):
            for r1 in range(r0 + 2, min(r0 + 12, h + 1)):
                block = grid[r0:r1, c0:c0 + 2]
                if np.any(block == bg):
                    continue
                nc = len(set(int(block[r, c]) for r in range(r1 - r0) for c in range(2)))
                if nc < 3:
                    continue
                cand_map = {}
                for r in range(r1 - r0):
                    a, b = int(block[r, 0]), int(block[r, 1])
                    cand_map[a] = b
                if len(cand_map) > best_legend_size and any(k in wall_colors for k in cand_map):
                    legend_map = cand_map
                    best_legend_size = len(cand_map)

    return (legend_map, wall_colors)


def _exec_cross_stencil_recolor(grid, params):
    """Recolor plus/cross patterns where all 4 neighbors match."""
    old_color = params.get('old_color')
    new_color = params.get('new_color')
    if old_color is None or new_color is None:
        return grid

    h, w = grid.shape
    result = grid.copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != old_color:
                continue
            if (grid[r-1, c] == old_color and grid[r+1, c] == old_color and
                grid[r, c-1] == old_color and grid[r, c+1] == old_color):
                result[r, c] = new_color
                result[r-1, c] = new_color
                result[r+1, c] = new_color
                result[r, c-1] = new_color
                result[r, c+1] = new_color
    return result


def _exec_frame_bbox_pack_ast(grid, params):
    """Execute frame-bbox pack from AST."""
    from aria.search.sketch import _do_frame_bbox_pack
    return _do_frame_bbox_pack(grid, params)


def _exec_object_highlight_ast(grid, params):
    """Execute object highlight from AST."""
    from aria.search.sketch import _exec_object_highlight_full
    return _exec_object_highlight_full(grid, params)


def _exec_legend_chain_connect(grid, params):
    """Connect workspace motifs following a sparse legend/control strip order.

    The control strip provides an ordered color chain. Each occurrence is matched
    to one connected component of that color in the workspace. Consecutive
    matched components must align horizontally or vertically; the gap between
    their bboxes is then painted with the source color.
    """
    from aria.guided.perceive import perceive
    from aria.search.geometry import bridge_area, orthogonal_bridge

    def _detect_control_row(arr, bg, side):
        h, w = arr.shape
        rows = range(h - 1, -1, -1) if side == 'bottom' else range(h)
        for r in rows:
            cols = [c for c in range(w) if arr[r, c] != bg]
            if len(cols) < 3:
                continue
            # Control rows are sparse singleton marks, not full objects.
            if any((r > 0 and arr[r - 1, c] != bg) or (r + 1 < h and arr[r + 1, c] != bg) for c in cols):
                continue
            return r, cols
        return None

    def _components_of_color(arr, color, row_limit):
        h, w = arr.shape
        seen = set()
        comps = []
        for r in range(row_limit):
            for c in range(w):
                if arr[r, c] != color or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < row_limit and 0 <= nc < w and arr[nr, nc] == color and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                comps.append(tuple(sorted(comp)))
        comps.sort(key=lambda cells: (min(r for r, _ in cells), min(c for _, c in cells)))
        return comps

    def _assign_chain(sequence, pools):
        best_assignment = None
        best_cost = None

        def _dfs(i, chosen, used, cost):
            nonlocal best_assignment, best_cost
            if best_cost is not None and cost >= best_cost:
                return
            if i == len(sequence):
                best_assignment = list(chosen)
                best_cost = cost
                return

            color = sequence[i]
            for idx, comp in enumerate(pools[color]):
                key = (color, idx)
                if key in used:
                    continue
                added_cost = 0
                if chosen:
                    bridge = orthogonal_bridge(chosen[-1], comp)
                    if bridge is None:
                        continue
                    added_cost = bridge_area(bridge)
                used.add(key)
                chosen.append(comp)
                _dfs(i + 1, chosen, used, cost + added_cost)
                chosen.pop()
                used.remove(key)

        _dfs(0, [], set(), 0)
        return best_assignment

    facts = perceive(grid)
    bg = params.get('bg', facts.bg)
    side = params.get('control_side', 'bottom')
    control = _detect_control_row(grid, bg, side)
    if control is None:
        return grid

    control_row, control_cols = control
    sequence = [int(grid[control_row, c]) for c in control_cols]
    if len(sequence) < 2:
        return grid

    pools = {color: _components_of_color(grid, color, control_row) for color in set(sequence)}
    if any(not pools[color] for color in sequence):
        return grid

    assignment = _assign_chain(sequence, pools)
    if assignment is None:
        return grid

    result = grid.copy()
    for color, cells_a, cells_b in zip(sequence, assignment, assignment[1:]):
        bridge = orthogonal_bridge(cells_a, cells_b)
        if bridge is None:
            return grid
        _, r0, r1, c0, c1 = bridge
        if r0 > r1 or c0 > c1:
            continue
        patch = result[r0:r1 + 1, c0:c1 + 1]
        result[r0:r1 + 1, c0:c1 + 1] = np.where(patch == bg, color, patch)

    return result


def _exec_diagonal_collision_trace(grid, params):
    """Trace diagonal rays from corner/point emitters through bar/point reflectors."""
    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    point_dir_name = params.get('point_dir', 'up_right')
    include_direct_hit = bool(params.get('include_direct_hit', True))
    point_dir = {
        'up_left': (-1, -1),
        'up_right': (-1, 1),
        'down_left': (1, -1),
        'down_right': (1, 1),
    }.get(point_dir_name, (-1, 1))

    def _classify(coords):
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        if len(coords) == 1:
            return 'point'
        if len(coords) == 3 and h == 1 and w == 3:
            return 'hbar'
        if len(coords) == 3 and h == 3 and w == 1:
            return 'vbar'
        if len(coords) == 3 and h == 2 and w == 2:
            return 'corner'
        return None

    def _components(arr):
        comps = []
        for color in sorted(int(v) for v in np.unique(arr) if v != 0):
            labels, count = ndimage.label(arr == color, structure=structure)
            for idx in range(1, count + 1):
                coords = [tuple(x) for x in np.argwhere(labels == idx).tolist()]
                kind = _classify(coords)
                if kind is None:
                    return None
                rs = [r for r, _ in coords]
                cs = [c for _, c in coords]
                comps.append({
                    'color': color,
                    'coords': coords,
                    'cells': set(coords),
                    'kind': kind,
                    'r0': min(rs),
                    'r1': max(rs),
                    'c0': min(cs),
                    'c1': max(cs),
                })
        return comps

    def _source_spec(comp):
        if comp['kind'] == 'point':
            r, c = comp['coords'][0]
            return (r - 1, c), point_dir

        if comp['kind'] != 'corner':
            return None

        r0, r1 = comp['r0'], comp['r1']
        c0, c1 = comp['c0'], comp['c1']
        offsets = {(r - r0, c - c0) for r, c in comp['coords']}
        missing = next(iter({(0, 0), (0, 1), (1, 0), (1, 1)} - offsets))
        mapping = {
            (0, 0): ((r1, c1), (1, 1)),
            (0, 1): ((r1, c0), (1, -1)),
            (1, 0): ((r0, c1), (-1, 1)),
            (1, 1): ((r0, c0), (-1, -1)),
        }
        return mapping[missing]

    def _hit_component(arr, comps, r, c, dr, dc):
        h, w = arr.shape
        hit_cells = [(r, c + dc), (r + dr, c)]
        if include_direct_hit:
            hit_cells.append((r + dr, c + dc))
        candidates = []
        for rr, cc in hit_cells:
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if arr[rr, cc] == 0:
                continue
            for comp in comps:
                if (rr, cc) in comp['cells']:
                    side_rank = 0 if (rr, cc) != (r + dr, c + dc) else 1
                    candidates.append((side_rank, rr, cc, comp))
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def _reflect(dr, dc, kind):
        if kind == 'vbar':
            return dr, -dc
        if kind in ('hbar', 'point'):
            return -dr, dc
        return None

    comps = _components(grid)
    if not comps:
        return grid

    emitters = []
    for comp in comps:
        spec = _source_spec(comp)
        if spec is None:
            continue
        (sr, sc), (dr, dc) = spec
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            emitters.append((comp['color'], nr, nc, dr, dc, 1))
    if not emitters:
        return grid

    result = grid.copy()
    best: dict[tuple[int, int], tuple[int, int]] = {}
    queue = list(emitters)
    while queue:
        color, r, c, dr, dc, dist = queue.pop(0)
        steps = 0
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and steps < grid.size:
            steps += 1
            if grid[r, c] == 0:
                prev = best.get((r, c))
                if prev is None or dist < prev[0]:
                    best[(r, c)] = (dist, color)

            hit = _hit_component(grid, comps, r, c, dr, dc)
            if hit is not None:
                prev = best.get((r, c))
                if prev is None or dist <= prev[0]:
                    best[(r, c)] = (dist, hit['color'])
                reflected = _reflect(dr, dc, hit['kind'])
                if reflected is not None:
                    ndr, ndc = reflected
                    queue.append((hit['color'], r + ndr, c + ndc, ndr, ndc, 1))
                break

            r += dr
            c += dc
            dist += 1

    for (r, c), (_, color) in best.items():
        result[r, c] = color
    return result


def _masked_patch_xforms():
    return [
        ("id", lambda a: a),
        ("rot90", lambda a: np.rot90(a, 1)),
        ("rot180", lambda a: np.rot90(a, 2)),
        ("rot270", lambda a: np.rot90(a, 3)),
        ("flip_h", lambda a: a[:, ::-1]),
        ("flip_v", lambda a: a[::-1, :]),
        ("transpose", lambda a: a.T),
    ]


def _masked_patch_source_shape(target_h: int, target_w: int, name: str) -> tuple[int, int]:
    if name in ("rot90", "rot270", "transpose"):
        return (target_w, target_h)
    return (target_h, target_w)


def _masked_patch_window_score(
    grid: Grid,
    mask_bbox: tuple[int, int, int, int],
    src_bbox: tuple[int, int, int, int],
    xform,
    ring: int,
) -> tuple[float, int, int] | None:
    mr0, mc0, mr1, mc1 = mask_bbox
    sr0, sc0, sr1, sc1 = src_bbox
    wr0 = max(0, mr0 - ring)
    wc0 = max(0, mc0 - ring)
    wr1 = min(grid.shape[0], mr1 + ring + 1)
    wc1 = min(grid.shape[1], mc1 + ring + 1)
    swr0 = max(0, sr0 - ring)
    swc0 = max(0, sc0 - ring)
    swr1 = min(grid.shape[0], sr1 + ring + 1)
    swc1 = min(grid.shape[1], sc1 + ring + 1)
    mask_window = grid[wr0:wr1, wc0:wc1]
    src_window = grid[swr0:swr1, swc0:swc1]
    transformed_window = xform(src_window)
    if transformed_window.shape != mask_window.shape:
        return None

    inner_r0 = mr0 - wr0
    inner_c0 = mc0 - wc0
    inner_r1 = inner_r0 + (mr1 - mr0)
    inner_c1 = inner_c0 + (mc1 - mc0)

    eq = 0
    total = 0
    for r in range(mask_window.shape[0]):
        for c in range(mask_window.shape[1]):
            inside = inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1
            if inside:
                continue
            total += 1
            if transformed_window[r, c] == mask_window[r, c]:
                eq += 1
    if total == 0:
        return None
    return (eq / total, eq, total)


def _masked_patch_best_for_bbox(
    grid: Grid,
    mask_bbox: tuple[int, int, int, int],
    *,
    mask_color: int,
    ring: int,
) -> tuple[np.ndarray, tuple[float, int, int]] | None:
    mr0, mc0, mr1, mc1 = mask_bbox
    target_h = mr1 - mr0 + 1
    target_w = mc1 - mc0 + 1

    best_score: tuple[float, int, int] | None = None
    best_outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for name, xform in _masked_patch_xforms():
        source_h, source_w = _masked_patch_source_shape(target_h, target_w, name)
        for r in range(grid.shape[0] - source_h + 1):
            for c in range(grid.shape[1] - source_w + 1):
                sr1 = r + source_h - 1
                sc1 = c + source_w - 1
                overlaps = not (sr1 < mr0 or r > mr1 or sc1 < mc0 or c > mc1)
                if overlaps:
                    continue
                source = grid[r:r + source_h, c:c + source_w]
                if np.any(source == mask_color):
                    continue
                score = _masked_patch_window_score(
                    grid,
                    mask_bbox,
                    (r, c, sr1, sc1),
                    xform,
                    ring,
                )
                if score is None:
                    continue
                transformed = xform(source)
                key = (transformed.shape, transformed.tobytes())
                if best_score is None or score > best_score:
                    best_score = score
                    best_outputs = {key: transformed.copy()}
                elif score == best_score:
                    best_outputs.setdefault(key, transformed.copy())

    if best_score is None or len(best_outputs) != 1:
        return None
    return next(iter(best_outputs.values())), best_score


def _exec_masked_patch_transfer(grid, params):
    """Recover a rectangular masked patch from a transformed source elsewhere.

    The op detects solid rectangular mask objects of a learned mask color,
    searches the rest of the grid for same-size (up to rotation/transpose)
    source patches, and selects the patch whose transformed surrounding ring
    best matches the visible context around the mask.
    """
    from aria.guided.perceive import perceive

    mask_color = int(params.get("mask_color", -1))
    ring = int(params.get("ring", 1))
    if mask_color < 0:
        return None

    facts = perceive(grid)
    mask_objects = [
        obj for obj in facts.objects
        if obj.color == mask_color and obj.is_rectangular and np.all(obj.mask)
    ]
    if not mask_objects:
        return None

    best_score: tuple[float, int, int] | None = None
    best_outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for obj in mask_objects:
        bbox = (obj.row, obj.col, obj.row + obj.height - 1, obj.col + obj.width - 1)
        found = _masked_patch_best_for_bbox(grid, bbox, mask_color=mask_color, ring=ring)
        if found is None:
            continue
        patch, score = found
        key = (patch.shape, patch.tobytes())
        if best_score is None or score > best_score:
            best_score = score
            best_outputs = {key: patch}
        elif score == best_score:
            best_outputs.setdefault(key, patch)

    if best_score is None or len(best_outputs) != 1:
        return None
    return next(iter(best_outputs.values()))


def _broadcast_run_score(grid: Grid, axis: str, sep_idx: int, bg: int) -> int:
    """Count orthogonal lines with content on exactly one side of the separator."""
    if axis == "col":
        total = 0
        for r in range(grid.shape[0]):
            left = grid[r, :sep_idx]
            right = grid[r, sep_idx + 1:]
            left_non = np.count_nonzero(left != bg)
            right_non = np.count_nonzero(right != bg)
            if (left_non > 0 and right_non == 0) or (right_non > 0 and left_non == 0):
                total += 1
        return total

    total = 0
    for c in range(grid.shape[1]):
        top = grid[:sep_idx, c]
        bottom = grid[sep_idx + 1:, c]
        top_non = np.count_nonzero(top != bg)
        bottom_non = np.count_nonzero(bottom != bg)
        if (top_non > 0 and bottom_non == 0) or (bottom_non > 0 and top_non == 0):
            total += 1
    return total


def _mono_separator_candidates(grid: Grid, axis: str, bg: int) -> list[int]:
    """Separator-like lines with a single non-background color and bg gaps allowed."""
    n = grid.shape[0] if axis == "row" else grid.shape[1]
    out: list[int] = []
    for idx in range(n):
        line = grid[idx, :] if axis == "row" else grid[:, idx]
        vals = {int(v) for v in line if int(v) != bg}
        if len(vals) == 1 and np.count_nonzero(line != bg) >= 2:
            out.append(idx)
    return out


def _non_bg_runs(line: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Contiguous non-background runs in natural line order."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if int(line[i]) == bg:
            i += 1
            continue
        color = int(line[i])
        j = i + 1
        while j < n and int(line[j]) == color:
            j += 1
        runs.append((color, j - i))
        i = j
    return runs


def _paint_periodic_runs(dest_len: int, runs: list[tuple[int, int]], dtype) -> np.ndarray:
    """Overlay periodic colors using run lengths as periods."""
    dest = np.zeros(dest_len, dtype=dtype)
    for color, period in runs:
        for q in range(0, dest_len, period):
            dest[q] = color
    return dest


def _exec_separator_motif_broadcast_auto_QUARANTINED(grid: Grid, params: dict[str, Any]) -> Grid:
    """QUARANTINED: monolithic auto-detect solver. Use _exec_separator_motif_broadcast
    with explicit sep_axis/sep_idx derived at search time instead.

    Kept for reference only -- not called from any live code path.
    """
    from aria.guided.perceive import perceive

    axis = str(params.get("axis", "auto"))
    facts = perceive(grid)
    bg = int(facts.bg)
    row_seps = sorted({s.index for s in facts.separators if s.axis == "row"})
    col_seps = sorted({s.index for s in facts.separators if s.axis == "col"})
    row_candidates = sorted(set(row_seps) | set(_mono_separator_candidates(grid, "row", bg)))
    col_candidates = sorted(set(col_seps) | set(_mono_separator_candidates(grid, "col", bg)))

    candidates: list[tuple[str, int, int]] = []
    if axis in {"auto", "row"}:
        for idx in row_candidates:
            candidates.append(("row", idx, _broadcast_run_score(grid, "row", idx, bg)))
    if axis in {"auto", "col"}:
        for idx in col_candidates:
            candidates.append(("col", idx, _broadcast_run_score(grid, "col", idx, bg)))

    candidates = [cand for cand in candidates if cand[2] > 0]
    if not candidates:
        return grid.copy()

    sep_axis, sep_idx, _ = max(candidates, key=lambda item: (item[2], item[0] == "col"))
    out = grid.copy()

    if sep_axis == "col":
        for r in range(grid.shape[0]):
            left = grid[r, :sep_idx]
            right = grid[r, sep_idx + 1:]
            left_non = np.count_nonzero(left != bg)
            right_non = np.count_nonzero(right != bg)
            if left_non > 0 and right_non == 0:
                run_order = _non_bg_runs(left, bg)
                if run_order:
                    out[r, sep_idx + 1:] = _paint_periodic_runs(len(right), run_order, grid.dtype)
            elif right_non > 0 and left_non == 0:
                run_order = list(reversed(_non_bg_runs(right, bg)))
                if run_order:
                    out[r, :sep_idx] = _paint_periodic_runs(len(left), run_order, grid.dtype)[::-1]
        return out

    for c in range(grid.shape[1]):
        top = grid[:sep_idx, c]
        bottom = grid[sep_idx + 1:, c]
        top_non = np.count_nonzero(top != bg)
        bottom_non = np.count_nonzero(bottom != bg)
        if top_non > 0 and bottom_non == 0:
            run_order = _non_bg_runs(top, bg)
            if run_order:
                out[sep_idx + 1:, c] = _paint_periodic_runs(len(bottom), run_order, grid.dtype)
        elif bottom_non > 0 and top_non == 0:
            run_order = list(reversed(_non_bg_runs(bottom, bg)))
            if run_order:
                out[:sep_idx, c] = _paint_periodic_runs(len(top), run_order, grid.dtype)[::-1]
    return out


def _exec_separator_motif_broadcast(grid: Grid, params: dict[str, Any]) -> Grid:
    """Broadcast separator-side line motifs into the empty opposite side.

    Params (derived at search time):
        sep_axis: 'row' or 'col'
        sep_idx: integer index of the separator line

    Falls back to auto-detect if sep_axis/sep_idx not provided (legacy path).
    """
    from aria.guided.perceive import perceive

    sep_axis = params.get("sep_axis")
    sep_idx = params.get("sep_idx")

    if sep_axis is None or sep_idx is None:
        # Legacy fallback: auto-detect
        return _exec_separator_motif_broadcast_auto_QUARANTINED(grid, params)

    facts = perceive(grid)
    bg = int(facts.bg)
    out = grid.copy()

    if sep_axis == "col":
        for r in range(grid.shape[0]):
            left = grid[r, :sep_idx]
            right = grid[r, sep_idx + 1:]
            left_non = np.count_nonzero(left != bg)
            right_non = np.count_nonzero(right != bg)
            if left_non > 0 and right_non == 0:
                run_order = _non_bg_runs(left, bg)
                if run_order:
                    out[r, sep_idx + 1:] = _paint_periodic_runs(len(right), run_order, grid.dtype)
            elif right_non > 0 and left_non == 0:
                run_order = list(reversed(_non_bg_runs(right, bg)))
                if run_order:
                    out[r, :sep_idx] = _paint_periodic_runs(len(left), run_order, grid.dtype)[::-1]
        return out

    for c in range(grid.shape[1]):
        top = grid[:sep_idx, c]
        bottom = grid[sep_idx + 1:, c]
        top_non = np.count_nonzero(top != bg)
        bottom_non = np.count_nonzero(bottom != bg)
        if top_non > 0 and bottom_non == 0:
            run_order = _non_bg_runs(top, bg)
            if run_order:
                out[sep_idx + 1:, c] = _paint_periodic_runs(len(bottom), run_order, grid.dtype)
        elif bottom_non > 0 and top_non == 0:
            run_order = list(reversed(_non_bg_runs(bottom, bg)))
            if run_order:
                out[:sep_idx, c] = _paint_periodic_runs(len(top), run_order, grid.dtype)[::-1]
    return out


def _line_repeat_score(grid: Grid, axis: str) -> int:
    """Total repeated-color evidence along rows or columns."""
    n_lines = grid.shape[0] if axis == "row" else grid.shape[1]
    total = 0
    for idx in range(n_lines):
        line = grid[idx, :] if axis == "row" else grid[:, idx]
        for color in (int(v) for v in np.unique(line) if int(v) != 0):
            count = int(np.sum(line == color))
            if count >= 2:
                total += count
    return total


def _broadcast_axis(grid: Grid, axis: str) -> Grid:
    """Broadcast sparse arithmetic scaffolds along the given axis."""
    import math

    h, w = grid.shape
    out = np.zeros_like(grid)
    n_lines = h if axis == "row" else w
    line_len = w if axis == "row" else h

    def _get_line(i: int) -> np.ndarray:
        return grid[i, :] if axis == "row" else grid[:, i]

    def _put_line(i: int, arr: np.ndarray) -> None:
        if axis == "row":
            out[i, :] = arr
        else:
            out[:, i] = arr

    for idx in range(n_lines):
        line = _get_line(idx)
        result = np.zeros_like(line)
        infos: list[tuple[int, list[int], int | None, int | None]] = []
        for color in (int(v) for v in np.unique(line) if int(v) != 0):
            positions = np.flatnonzero(line == color).tolist()
            if len(positions) >= 2:
                diffs = [b - a for a, b in zip(positions, positions[1:]) if b > a]
                step = diffs[0]
                for diff in diffs[1:]:
                    step = math.gcd(step, diff)
                residue = positions[0] % step if step else 0
                infos.append((color, positions, step, residue))
            else:
                infos.append((color, positions, None, None))

        scaffolds = [info for info in infos if info[2] is not None]
        claimed = np.zeros(line_len, dtype=bool)
        if len(scaffolds) == 1:
            scaffold_color, scaffold_pos, scaffold_step, scaffold_residue = scaffolds[0]
            aligned_override = False
            for color, positions, step, _ in infos:
                if step is not None:
                    continue
                pos = positions[0]
                if pos % scaffold_step != scaffold_residue:
                    result[pos] = color
                    claimed[pos] = True
                    continue
                aligned_override = True
                if max(scaffold_pos) < pos:
                    search = range(0, pos + 1)
                elif min(scaffold_pos) > pos:
                    search = range(pos, line_len)
                else:
                    search = [pos]
                for q in search:
                    if q % scaffold_step == scaffold_residue:
                        result[q] = color
                        claimed[q] = True
            if not aligned_override:
                for q in range(line_len):
                    if q % scaffold_step == scaffold_residue and not claimed[q]:
                        result[q] = scaffold_color
        else:
            for color, positions, step, residue in infos:
                if step is None:
                    result[positions[0]] = color
                    continue
                for q in range(line_len):
                    if q % step == residue:
                        result[q] = color
        _put_line(idx, result)

    return out


def _exec_line_arith_broadcast(grid: Grid, params: dict[str, Any]) -> Grid:
    """Broadcast sparse arithmetic line scaffolds along the stronger axis."""
    axis = str(params.get("axis", "auto"))
    if axis == "auto":
        row_score = _line_repeat_score(grid, "row")
        col_score = _line_repeat_score(grid, "col")
        axis = "row" if row_score >= col_score else "col"
    if axis not in {"row", "col"}:
        return grid.copy()
    return _broadcast_axis(grid, axis)


def _detect_barrier_params(grid: Grid) -> tuple[int, str, int] | None:
    """Extract barrier_color, barrier_orient, bg from a grid.

    Returns (barrier_color, barrier_orient, bg) or None if no barrier found.
    Used by derive to pre-compute params at search time.
    """
    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    best: tuple[int, dict[str, Any]] | None = None
    for color in sorted(int(v) for v in np.unique(grid) if v != 0):
        labels, count = ndimage.label(grid == color, structure=structure)
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            rs = coords[:, 0]
            cs = coords[:, 1]
            r0, r1 = int(rs.min()), int(rs.max())
            c0, c1 = int(cs.min()), int(cs.max())
            area = int(len(coords))
            w = c1 - c0 + 1
            h = r1 - r0 + 1
            spans_w = c0 == 0 and c1 == grid.shape[1] - 1
            spans_h = r0 == 0 and r1 == grid.shape[0] - 1
            if not (spans_w or spans_h):
                continue
            orient = "horizontal" if spans_w and w >= h else "vertical"
            if best is None or area > best[0]:
                best = (area, {"color": color, "orient": orient})

    if best is None:
        return None

    barrier_color = best[1]["color"]
    barrier_orient = best[1]["orient"]

    # Dominant bg: most frequent color excluding barrier
    vals, counts = np.unique(grid, return_counts=True)
    bg_candidates = [
        (int(c), int(n))
        for c, n in zip(vals, counts, strict=False)
        if int(c) != barrier_color
    ]
    bg = max(bg_candidates, key=lambda item: item[1])[0] if bg_candidates else 0

    return (barrier_color, barrier_orient, bg)


def _exec_barrier_port_transfer_auto_QUARANTINED(grid: Grid, params: dict[str, Any]) -> Grid:
    """QUARANTINED: monolithic auto-detect solver. Use _exec_barrier_port_transfer
    with explicit barrier_color/barrier_orient/bg derived at search time instead.

    Kept for reference only -- not called from any live code path.
    """
    from collections import defaultdict
    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    def _components_of_color(color: int) -> list[dict[str, Any]]:
        labels, count = ndimage.label(grid == color, structure=structure)
        comps: list[dict[str, Any]] = []
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            rs = coords[:, 0]
            cs = coords[:, 1]
            mask = np.zeros((rs.max() - rs.min() + 1, cs.max() - cs.min() + 1), dtype=bool)
            mask[rs - rs.min(), cs - cs.min()] = True
            comps.append({
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in coords],
                "r0": int(rs.min()),
                "r1": int(rs.max()),
                "c0": int(cs.min()),
                "c1": int(cs.max()),
                "h": int(rs.max() - rs.min() + 1),
                "w": int(cs.max() - cs.min() + 1),
                "area": int(len(coords)),
                "mask": mask,
            })
        return comps

    def _detect_barrier() -> dict[str, Any] | None:
        best: tuple[int, dict[str, Any]] | None = None
        for color in sorted(int(v) for v in np.unique(grid) if v != 0):
            for comp in _components_of_color(color):
                spans_w = comp["c0"] == 0 and comp["c1"] == grid.shape[1] - 1
                spans_h = comp["r0"] == 0 and comp["r1"] == grid.shape[0] - 1
                if not (spans_w or spans_h):
                    continue
                orient = "horizontal" if spans_w and comp["w"] >= comp["h"] else "vertical"
                candidate = {"orient": orient, **comp}
                if best is None or comp["area"] > best[0]:
                    best = (comp["area"], candidate)
        return None if best is None else best[1]

    def _dominant_bg(barrier_color: int) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        candidates = [
            (int(color), int(count))
            for color, count in zip(vals, counts, strict=False)
            if int(color) != barrier_color
        ]
        return max(candidates, key=lambda item: item[1])[0] if candidates else 0

    barrier = _detect_barrier()
    if barrier is None:
        return grid
    bg = _dominant_bg(int(barrier["color"]))
    return _exec_barrier_port_transfer_core(grid, barrier, bg)


def _exec_barrier_port_transfer(grid: Grid, params: dict[str, Any]) -> Grid:
    """Pack objects into barrier-adjacent port families and slide them through openings.

    Params (derived at search time):
        barrier_color: int color of the barrier
        barrier_orient: 'horizontal' or 'vertical'
        bg: int background color

    Falls back to auto-detect if params not provided (legacy path).
    """
    barrier_color = params.get("barrier_color")
    barrier_orient = params.get("barrier_orient")
    bg_param = params.get("bg")

    if barrier_color is None or barrier_orient is None or bg_param is None:
        # Legacy fallback: auto-detect
        return _exec_barrier_port_transfer_auto_QUARANTINED(grid, params)

    # Build barrier dict from pre-computed params + grid geometry
    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, count = ndimage.label(grid == barrier_color, structure=structure)

    # Find the largest component of the barrier color
    best_comp = None
    best_area = 0
    for idx in range(1, count + 1):
        coords = np.argwhere(labels == idx)
        if len(coords) > best_area:
            best_area = len(coords)
            rs = coords[:, 0]
            cs = coords[:, 1]
            mask = np.zeros((rs.max() - rs.min() + 1, cs.max() - cs.min() + 1), dtype=bool)
            mask[rs - rs.min(), cs - cs.min()] = True
            best_comp = {
                "color": int(barrier_color),
                "orient": barrier_orient,
                "coords": [(int(r), int(c)) for r, c in coords],
                "r0": int(rs.min()),
                "r1": int(rs.max()),
                "c0": int(cs.min()),
                "c1": int(cs.max()),
                "h": int(rs.max() - rs.min() + 1),
                "w": int(cs.max() - cs.min() + 1),
                "area": int(len(coords)),
                "mask": mask,
            }

    if best_comp is None:
        return grid

    return _exec_barrier_port_transfer_core(grid, best_comp, int(bg_param))


def _exec_barrier_port_transfer_core(grid: Grid, barrier: dict[str, Any], bg: int) -> Grid:
    """Core barrier-port-transfer logic shared by parameterized and quarantined paths."""
    from collections import defaultdict

    from scipy import ndimage

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    def _components_of_color(color: int) -> list[dict[str, Any]]:
        labels, count = ndimage.label(grid == color, structure=structure)
        comps: list[dict[str, Any]] = []
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            rs = coords[:, 0]
            cs = coords[:, 1]
            mask = np.zeros((rs.max() - rs.min() + 1, cs.max() - cs.min() + 1), dtype=bool)
            mask[rs - rs.min(), cs - cs.min()] = True
            comps.append({
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in coords],
                "r0": int(rs.min()),
                "r1": int(rs.max()),
                "c0": int(cs.min()),
                "c1": int(cs.max()),
                "h": int(rs.max() - rs.min() + 1),
                "w": int(cs.max() - cs.min() + 1),
                "area": int(len(coords)),
                "mask": mask,
            })
        return comps

    def _port_candidates(barrier: dict[str, Any], bg: int) -> list[dict[str, Any]]:
        r0, r1, c0, c1 = barrier["r0"], barrier["r1"], barrier["c0"], barrier["c1"]
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        labels, count = ndimage.label(sub == bg, structure=structure)
        ports: list[dict[str, Any]] = []
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            rs = coords[:, 0] + r0
            cs = coords[:, 1] + c0
            if barrier["orient"] == "horizontal":
                side = "top" if int(rs.min()) == r0 else "bottom" if int(rs.max()) == r1 else None
                if side is None:
                    continue
                extent = int(cs.max() - cs.min() + 1)
                axis_center = (float(cs.min()) + float(cs.max())) / 2.0
            else:
                side = "left" if int(cs.min()) == c0 else "right" if int(cs.max()) == c1 else None
                if side is None:
                    continue
                extent = int(rs.max() - rs.min() + 1)
                axis_center = (float(rs.min()) + float(rs.max())) / 2.0

            family_extent = extent + 2
            lane_start = int(np.floor(axis_center - (family_extent - 1) / 2.0))
            ports.append({
                "side": side,
                "family_extent": family_extent,
                "axis_center": axis_center,
                "lane_start": lane_start,
                "hole_bbox": (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())),
                "hole_cells": {(int(r), int(c)) for r, c in zip(rs, cs, strict=False)},
            })
        return ports

    def _objects(barrier_color: int, bg: int) -> list[dict[str, Any]]:
        objs: list[dict[str, Any]] = []
        for color in sorted(int(v) for v in np.unique(grid) if v not in (bg, barrier_color)):
            objs.extend(_components_of_color(color))
        return objs

    def _source_side(obj: dict[str, Any], barrier: dict[str, Any]) -> str:
        if barrier["orient"] == "horizontal":
            if obj["r1"] < barrier["r0"]:
                return "top"
            if obj["r0"] > barrier["r1"]:
                return "bottom"
        else:
            if obj["c1"] < barrier["c0"]:
                return "left"
            if obj["c0"] > barrier["c1"]:
                return "right"
        return "interior"

    def _can_place(canvas: np.ndarray, obj: dict[str, Any], top: int, left: int, bg: int) -> bool:
        h, w = obj["mask"].shape
        H, W = canvas.shape
        if top < 0 or left < 0 or top + h > H or left + w > W:
            return False
        patch = canvas[top:top + h, left:left + w]
        return bool(np.all(~obj["mask"] | (patch == bg)))

    def _place(canvas: np.ndarray, obj: dict[str, Any], top: int, left: int) -> None:
        patch = canvas[top:top + obj["mask"].shape[0], left:left + obj["mask"].shape[1]]
        patch[obj["mask"]] = obj["color"]

    def _settled_position(canvas: np.ndarray, obj: dict[str, Any], port: dict[str, Any], bg: int) -> tuple[int, int] | None:
        if barrier["orient"] == "horizontal":
            left = port["lane_start"]
            if port["side"] == "top":
                top = 0
                while _can_place(canvas, obj, top + 1, left, bg):
                    top += 1
            else:
                top = canvas.shape[0] - obj["h"]
                while _can_place(canvas, obj, top - 1, left, bg):
                    top -= 1
            return (top, left) if _can_place(canvas, obj, top, left, bg) else None

        top = port["lane_start"]
        if port["side"] == "left":
            left = 0
            while _can_place(canvas, obj, top, left + 1, bg):
                left += 1
        else:
            left = canvas.shape[1] - obj["w"]
            while _can_place(canvas, obj, top, left - 1, bg):
                left -= 1
        return (top, left) if _can_place(canvas, obj, top, left, bg) else None

    def _hole_occupancy(obj: dict[str, Any], port: dict[str, Any], top: int, left: int) -> int:
        return sum(
            1
            for dr, dc in zip(*np.where(obj["mask"]), strict=False)
            if (top + int(dr), left + int(dc)) in port["hole_cells"]
        )

    ports = _port_candidates(barrier, bg)
    if not ports:
        return grid
    objs = _objects(int(barrier["color"]), bg)
    if not objs:
        return grid

    result = grid.copy()
    for obj in objs:
        for r, c in obj["coords"]:
            result[r, c] = bg

    axis_key = "w" if barrier["orient"] == "horizontal" else "h"
    groups: dict[int, list[dict[str, Any]]] = {}
    for obj in objs:
        groups.setdefault(int(obj[axis_key]), []).append(obj)

    assignments: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for extent, group in groups.items():
        candidates = [port for port in ports if port["family_extent"] == extent]
        if not candidates:
            return grid
        source_axis_center = {
            id(obj): (
                (obj["c0"] + obj["c1"]) / 2.0 if barrier["orient"] == "horizontal"
                else (obj["r0"] + obj["r1"]) / 2.0
            )
            for obj in group
        }
        previews: dict[tuple[int, int], tuple[int, int] | None] = {}
        occupancy: dict[tuple[int, int], int] = {}
        for obj_idx, obj in enumerate(group):
            for port_idx, port in enumerate(candidates):
                pos = _settled_position(result, obj, port, bg)
                previews[(obj_idx, port_idx)] = pos
                occupancy[(obj_idx, port_idx)] = 0 if pos is None else _hole_occupancy(obj, port, *pos)

        per_port: dict[int, list[dict[str, Any]]] = defaultdict(list)
        active_port_indices: set[int] = set()
        assigned_indices: set[int] = set()
        for obj_idx, obj in enumerate(group):
            best_occ = max(occupancy[(obj_idx, port_idx)] for port_idx in range(len(candidates)))
            if best_occ <= 0:
                continue
            chosen_idx = min(
                [port_idx for port_idx in range(len(candidates)) if occupancy[(obj_idx, port_idx)] == best_occ],
                key=lambda port_idx: (
                    abs(source_axis_center[id(obj)] - candidates[port_idx]["axis_center"]),
                    0 if _source_side(obj, barrier) == candidates[port_idx]["side"] else 1,
                    -obj["area"],
                ),
            )
            enriched = dict(obj)
            enriched["_anchor_occ"] = best_occ
            per_port[chosen_idx].append(enriched)
            active_port_indices.add(chosen_idx)
            assigned_indices.add(obj_idx)

        if not active_port_indices:
            centers = [source_axis_center[id(obj)] for obj in group]
            mean_center = sum(centers) / len(centers)
            chosen_idx = min(
                range(len(candidates)),
                key=lambda port_idx: (
                    abs(candidates[port_idx]["axis_center"] - mean_center),
                    0 if candidates[port_idx]["side"] in ("top", "left") else 1,
                    candidates[port_idx]["hole_bbox"],
                ),
            )
            active_port_indices.add(chosen_idx)

        for obj_idx, obj in enumerate(group):
            if obj_idx in assigned_indices:
                continue
            same_side_ports = [
                port_idx for port_idx in active_port_indices
                if candidates[port_idx]["side"] == _source_side(obj, barrier)
            ]
            choices = same_side_ports or list(active_port_indices)
            chosen_idx = min(
                choices,
                key=lambda port_idx: (
                    abs(source_axis_center[id(obj)] - candidates[port_idx]["axis_center"]),
                    candidates[port_idx]["hole_bbox"],
                ),
            )
            enriched = dict(obj)
            enriched["_anchor_occ"] = 0
            per_port[chosen_idx].append(enriched)

        for port_idx in sorted(active_port_indices):
            assignments.append((candidates[port_idx], per_port[port_idx]))

    assignments.sort(key=lambda item: (item[0]["family_extent"], item[0]["axis_center"]))
    for port, group in assignments:
        axis_center = lambda obj: (
            (obj["c0"] + obj["c1"]) / 2.0
            if barrier["orient"] == "horizontal"
            else (obj["r0"] + obj["r1"]) / 2.0
        )
        anchor = max(
            group,
            key=lambda obj: (
                int(obj.get("_anchor_occ", 0)),
                -abs(axis_center(obj) - port["axis_center"]),
                obj["area"],
            ),
        )
        opposite = sorted(
            [obj for obj in group if obj is not anchor and _source_side(obj, barrier) != port["side"]],
            key=lambda obj: (abs(axis_center(obj) - port["axis_center"]), -obj["area"]),
        )
        same_side = sorted(
            [obj for obj in group if obj is not anchor and _source_side(obj, barrier) == port["side"]],
            key=lambda obj: (abs(axis_center(obj) - port["axis_center"]), -obj["area"]),
        )
        ordered = [anchor, *opposite, *same_side]

        for obj in ordered:
            pos = _settled_position(result, obj, port, bg)
            if pos is None:
                return grid
            _place(result, obj, *pos)

    return result


def _exec_quadrant_template_decode(grid, params):
    """Execute quadrant template decode.

    One quadrant is a spatial template; others have seed blocks that
    get the template applied with appropriate mirroring.
    """
    from aria.guided.perceive import perceive
    from aria.search.decode import (
        _extract_quadrant_pattern, _apply_quadrant_template,
    )

    facts = perceive(grid)
    bg = facts.bg
    tmpl_idx = params.get('template_quadrant', 0)
    seed_color = params.get('seed_color')

    row_seps = sorted([s for s in facts.separators if s.axis == 'row'], key=lambda s: s.index)
    col_seps = sorted([s for s in facts.separators if s.axis == 'col'], key=lambda s: s.index)
    if len(row_seps) != 1 or len(col_seps) != 1:
        return grid

    rs, cs = row_seps[0].index, col_seps[0].index
    h, w = grid.shape

    quads_rc = [(0, rs, 0, cs), (0, rs, cs + 1, w),
                (rs + 1, h, 0, cs), (rs + 1, h, cs + 1, w)]
    quads = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]

    tmpl = quads[tmpl_idx]
    tmpl_pattern = _extract_quadrant_pattern(tmpl, bg, seed_color)
    if tmpl_pattern is None:
        return grid

    pat_colors, pat_bbox, pat_relative = tmpl_pattern
    central_color = seed_color if seed_color is not None else pat_colors.get('center')
    if central_color is None:
        return grid

    result = grid.copy()
    tmpl_row = 0 if tmpl_idx < 2 else 1
    tmpl_col = tmpl_idx % 2

    for qi in range(4):
        if qi == tmpl_idx:
            continue
        q = quads[qi]
        seed_cells = [(r, c) for r in range(q.shape[0]) for c in range(q.shape[1])
                       if q[r, c] == central_color]
        if not seed_cells:
            continue

        qi_row = 0 if qi < 2 else 1
        qi_col = qi % 2
        flip_v = (qi_row != tmpl_row)
        flip_h = (qi_col != tmpl_col)

        applied = _apply_quadrant_template(q, pat_relative, seed_cells,
                                            central_color, bg, flip_h, flip_v)
        if applied is not None:
            r0, r1, c0, c1 = quads_rc[qi]
            result[r0:r1, c0:c1] = applied

    return result
