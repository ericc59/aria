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

    if op == Op.CAVITY_TRANSFER:
        grid = _child(node, 0, inp, ctx)
        if grid is None:
            return None
        return _exec_cavity_transfer_auto(grid)

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


def _exec_cavity_transfer_auto(grid):
    """Auto-detect concave hosts with markers and transfer them through openings.

    Hosts are local 8-connected color groups, not global same-color unions.
    The marker cavity defines an opening side inside the host bbox; markers
    then exit toward the complementary tip side.
    """
    from aria.guided.perceive import perceive
    from collections import defaultdict, deque

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
        hr = [r for r, _ in host_cells]
        hc = [c for _, c in host_cells]
        top, bot = min(hr), max(hr)
        left, right = min(hc), max(hc)
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
        return open_sides, (top, bot, left, right)

    def _boundary_cells(host_cells, direction, top, bot, left, right):
        if direction == 'up':
            return [(r, c) for r, c in host_cells if r == top]
        if direction == 'down':
            return [(r, c) for r, c in host_cells if r == bot]
        if direction == 'left':
            return [(r, c) for r, c in host_cells if c == left]
        return [(r, c) for r, c in host_cells if c == right]

    def _direction_order(host_cells, inside_cells, open_sides, bounds):
        top, bot, left, right = bounds
        m_cr = sum(r for r, _ in inside_cells) / len(inside_cells)
        m_cc = sum(c for _, c in inside_cells) / len(inside_cells)
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
            key=lambda d: (len(_boundary_cells(host_cells, d, top, bot, left, right)), -_dist_to_side(d)),
        )

    def _tip_anchor(host_cells, inside_cells, direction, bounds):
        top, bot, left, right = bounds
        m_cr = sum(r for r, _ in inside_cells) / len(inside_cells)
        m_cc = sum(c for _, c in inside_cells) / len(inside_cells)
        cells = _boundary_cells(host_cells, direction, top, bot, left, right)
        if direction in ('up', 'down'):
            cells = sorted(cells, key=lambda rc: abs(rc[1] - m_cc))
            return cells[0][1] if cells else None
        cells = sorted(cells, key=lambda rc: abs(rc[0] - m_cr))
        return cells[0][0] if cells else None

    def _project_cells(direction, anchor, bounds, inside_cells):
        top, bot, left, right = bounds
        if anchor is None:
            return []
        n = len(inside_cells)
        if direction == 'up':
            candidates = [(top - 1 - i, anchor) for i in range(n)]
        elif direction == 'down':
            candidates = [(bot + 1 + i, anchor) for i in range(n)]
        elif direction == 'left':
            candidates = [(anchor, left - 1 - i) for i in range(n)]
        else:
            candidates = [(anchor, right + 1 + i) for i in range(n)]

        projected = []
        for nr, nc in candidates:
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
        hr = [r for r, c in host_cells]
        hc = [c for r, c in host_cells]
        top, bot = min(hr), max(hr)
        left, right = min(hc), max(hc)

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
    """Transfer marker cells from inside concave host to the opposite open end."""
    from aria.guided.perceive import perceive

    pairs = params.get('pairs', [])
    if not pairs:
        # Single-pair fallback
        mc = params.get('marker_color')
        hc = params.get('host_color')
        d = params.get('direction')
        if mc is not None and hc is not None and d:
            pairs = [(mc, hc, d)]
        else:
            return grid

    facts = perceive(grid)
    bg = facts.bg
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

    For each non-bg color, finds bg cells enclosed by that color's boundary
    (not reachable from the grid border via bg cells). If a legend/color_map
    is provided, uses it. Otherwise derives fill color from the enclosed
    cells' neighborhood.
    """
    from aria.guided.perceive import perceive
    from collections import deque

    strategy = params.get('strategy', 'color_map')

    if strategy == 'enclosed_fill':
        return _exec_enclosed_fill(grid)

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


def _exec_enclosed_fill(grid):
    """Fill enclosed bg cells using a legend-derived color map.

    1. Find enclosed bg regions for each wall color
    2. Extract a color-pair legend from the input (small dense block)
    3. Fill enclosed cells using the legend mapping
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

    # Find colors that enclose bg regions AND form frame-like boundaries
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

    # Extract legend: scan for 2-row or 2-col dense rectangles with diverse colors
    legend_map = {}
    best_legend_size = 0

    # Try 2-row blocks
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

    # Try 2-col blocks
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

    # Build reverse map too: value→key
    reverse_map = {v: k for k, v in legend_map.items()}

    # Fill enclosed regions using legend (bidirectional lookup)
    # Only fill colors that also form proper frames
    result = grid.copy()
    for wc in wall_colors:
        if wc not in frame_colors:
            continue  # skip non-frame boundaries
        wg = np.where(grid == wc, wc, bg)
        enclosed = _flood_border(wg, bg)
        if enclosed.sum() > 0:
            if wc in legend_map:
                fill = legend_map[wc]
            elif wc in reverse_map:
                fill = reverse_map[wc]
            else:
                fill = wc  # fallback
            result[enclosed] = fill

    return result


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
