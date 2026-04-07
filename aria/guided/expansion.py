"""Expansion guidance: learned next-step and binding prediction.

Given a workspace state + partial explanation, predict:
1. The next rewrite operation to try
2. Key parameter bindings (color, axis, object ID)
3. Whether to stop

Uses the selector model from prompt 02 for support/op prediction,
plus new binding prediction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from aria.guided.workspace import Workspace, build_workspace, ObjectInfo, _detect_bg
from aria.guided.training_data import (
    featurize_example, SelectionExample,
    SUPPORT_TYPES, REWRITE_OPS, SUPPORT_TO_IDX, OP_TO_IDX,
)
from aria.guided.selector_model import SelectorMLP
from aria.types import Grid


# ---------------------------------------------------------------------------
# Expansion supervision extraction
# ---------------------------------------------------------------------------

@dataclass
class ExpansionTarget:
    """Supervised target for one expansion step."""
    task_id: str
    rule_type: str

    # Next-step targets
    support_type_idx: int
    rewrite_op_idx: int

    # Binding targets
    bind_color: int | None       # target color for FILL/RECOLOR
    bind_from_color: int | None  # source color for RECOLOR
    bind_axis: str | None        # "row"/"col" for PERIODIC, "h"/"v" for REFLECT
    bind_stamp_color: int | None # stamp source color for COPY
    bind_marker_color: int | None  # marker color for COPY

    # Features
    features: dict[str, Any]


def extract_expansion_targets(tasks) -> list[ExpansionTarget]:
    """Extract supervised expansion targets from synthetic tasks."""
    from aria.guided.synthetic import SyntheticTask
    targets = []

    for task in tasks:
        if not task.latent_graph.steps:
            continue
        step = task.latent_graph.steps[0]
        ws = build_workspace(task.train[0][0], task.train[0][1])

        support_idx = SUPPORT_TO_IDX.get(step.support_desc, 0)
        op_idx = OP_TO_IDX.get(step.rewrite_op.name, 0)

        params = task.latent_params
        targets.append(ExpansionTarget(
            task_id=task.task_id,
            rule_type=task.rule_type,
            support_type_idx=support_idx,
            rewrite_op_idx=op_idx,
            bind_color=params.get("fill_color", params.get("to_color")),
            bind_from_color=params.get("from_color"),
            bind_axis=params.get("axis", params.get("sym_axis")),
            bind_stamp_color=params.get("stamp_color"),
            bind_marker_color=params.get("marker_color"),
            features=ws.serialize(),
        ))

    return targets


# ---------------------------------------------------------------------------
# Binding predictor: simple heuristic rules from workspace features
# ---------------------------------------------------------------------------

def predict_bindings(
    ws: Workspace,
    support_type: str,
    rewrite_op: str,
) -> list[dict[str, Any]]:
    """Predict likely parameter bindings given support type and op.

    Returns a ranked list of binding candidates.
    """
    candidates = []

    if rewrite_op == "FILL" and support_type == "enclosed_bg":
        # Fill color candidates: each non-bg color present in the grid
        for color in ws.palette:
            if color != ws.bg:
                candidates.append({"fill_color": color})
        # Also try: most common adjacent color of residual units
        adj_colors = set()
        for u in ws.residual_units:
            adj_colors.update(u.adjacent_colors)
        for c in sorted(adj_colors):
            if c != ws.bg and {"fill_color": c} not in candidates:
                candidates.insert(0, {"fill_color": c})

    elif rewrite_op == "RECOLOR" and support_type == "all_objects":
        # Identify candidate from_color and to_color from residual
        for u in ws.residual_units:
            if u.change_type == "recolor" and u.input_colors and u.output_colors:
                for fc in u.input_colors:
                    for tc in u.output_colors:
                        if fc != tc:
                            candidates.append({"from_color": fc, "to_color": tc})

    elif rewrite_op == "PERIODIC_REPAIR":
        candidates.append({"axis": "row"})
        candidates.append({"axis": "col"})

    elif rewrite_op in ("REFLECT_H", "REFLECT_V"):
        candidates.append({"sym_axis": "h" if rewrite_op == "REFLECT_H" else "v",
                           "prefer_bg": True})
        candidates.append({"sym_axis": "h" if rewrite_op == "REFLECT_H" else "v",
                           "prefer_bg": False})

    elif rewrite_op == "COPY" and support_type == "stamp_source":
        # Identify stamp source: largest non-singleton object
        # Marker: singleton of different color
        non_singletons = [o for o in ws.objects if not o.is_singleton]
        singletons = [o for o in ws.objects if o.is_singleton]
        for src in sorted(non_singletons, key=lambda o: -o.size):
            for marker in singletons:
                if marker.color != src.color:
                    candidates.append({
                        "stamp_color": src.color,
                        "marker_color": marker.color,
                        "stamp_obj": src,
                        "marker_obj": marker,
                    })

    if not candidates:
        candidates.append({})

    return candidates


# ---------------------------------------------------------------------------
# Guided search: uses selector + binding predictor to order candidates
# ---------------------------------------------------------------------------

def guided_expansion_search(
    train: list[tuple[Grid, Grid]],
    model: SelectorMLP,
    max_candidates: int = 300,
) -> tuple[bool, Any, int, int]:
    """Search guided by the selector model.

    Returns: (solved, op_fn, candidates_tried, train_diff)
    """
    if not train:
        return False, None, 0, 0

    ws = build_workspace(train[0][0], train[0][1])
    feat = _quick_featurize(ws)
    topk_s, topk_o = model.predict_topk(feat["x_global"], feat["x_objects"], k=3)

    candidates_tried = 0
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)

    # Phase 1: guided candidates (selector-ordered)
    for op_fn in _guided_candidates(ws, topk_s, topk_o):
        candidates_tried += 1
        ok, diff = _verify_on_train(op_fn, train)
        if ok:
            return True, op_fn, candidates_tried, 0
        if diff < best_diff:
            best_diff = diff
        if candidates_tried >= max_candidates:
            return False, None, candidates_tried, best_diff

    # Phase 2: always try symmetry repair and copy_stamp (new operations)
    for op_fn in _structural_candidates(ws):
        candidates_tried += 1
        ok, diff = _verify_on_train(op_fn, train)
        if ok:
            return True, op_fn, candidates_tried, 0
        if diff < best_diff:
            best_diff = diff
        if candidates_tried >= max_candidates:
            return False, None, candidates_tried, best_diff

    # Phase 3: unguided fallback
    for op_fn in _unguided_fallback(ws.bg):
        candidates_tried += 1
        ok, diff = _verify_on_train(op_fn, train)
        if ok:
            return True, op_fn, candidates_tried, 0
        if diff < best_diff:
            best_diff = diff
        if candidates_tried >= max_candidates:
            break

    return False, None, candidates_tried, best_diff


def _verify_on_train(op_fn, train):
    """Verify an operation on all train demos."""
    all_ok = True
    total_diff = 0
    for inp, out in train:
        bg = _detect_bg(inp)
        try:
            pred = op_fn(inp, bg)
        except Exception:
            return False, sum(out.size for _, out in train)
        if pred is None or not np.array_equal(pred, out):
            all_ok = False
            if pred is not None and pred.shape == out.shape:
                total_diff += int(np.sum(pred != out))
            else:
                total_diff += out.size
    return all_ok, total_diff


def _quick_featurize(ws):
    ex = SelectionExample("", 0, ws.serialize(), "", "", "", {})
    return featurize_example(ex)


def _guided_candidates(ws, topk_s, topk_o):
    """Yield operations ordered by selector's top-K predictions."""
    support_names = [SUPPORT_TYPES[i] for i in topk_s if i < len(SUPPORT_TYPES)]
    op_names = [REWRITE_OPS[i] for i in topk_o if i < len(REWRITE_OPS)]

    for sn in support_names:
        for on in op_names:
            bindings = predict_bindings(ws, sn, on)
            for binding in bindings:
                op = _make_operation(sn, on, binding, ws)
                if op is not None:
                    yield op


def _make_operation(support_type, rewrite_op, binding, ws):
    """Create a callable operation from support type + op + binding."""

    if support_type == "enclosed_bg" and rewrite_op == "FILL":
        color = binding.get("fill_color")
        if color is not None:
            def _f(inp, bg, c=color):
                return _op_fill_enclosed(inp, bg, c)
            return _f

    if support_type == "all_objects" and rewrite_op == "RECOLOR":
        fc = binding.get("from_color")
        tc = binding.get("to_color")
        if fc is not None and tc is not None:
            def _r(inp, bg, f=fc, t=tc):
                out = inp.copy()
                out[inp == f] = t
                return out
            return _r

    if support_type == "full_grid" and rewrite_op == "PERIODIC_REPAIR":
        axis = binding.get("axis", "row")
        def _p(inp, bg, ax=axis):
            from aria.guided.grammar import _repair_periodic_region
            return _repair_periodic_region(inp, ax)
        return _p

    if support_type == "per_object" and rewrite_op in ("REFLECT_H", "REFLECT_V"):
        prefer_bg = binding.get("prefer_bg", True)
        sym = "h" if rewrite_op == "REFLECT_H" else "v"
        def _sym(inp, bg, s=sym, pb=prefer_bg):
            return _op_symmetry_repair(inp, bg, s, pb)
        return _sym

    if support_type == "stamp_source" and rewrite_op == "COPY":
        stamp_obj = binding.get("stamp_obj")
        marker_obj = binding.get("marker_obj")
        if stamp_obj is not None and marker_obj is not None:
            def _copy(inp, bg, so=stamp_obj, mo=marker_obj):
                return _op_copy_stamp(inp, bg, so.color, mo.color)
            return _copy

    if support_type == "all_objects" and rewrite_op == "DELETE":
        fc = binding.get("from_color")
        if fc is not None:
            def _d(inp, bg, c=fc):
                out = inp.copy()
                out[inp == c] = bg
                return out
            return _d

    if support_type == "enclosed_bg" and rewrite_op == "FILL":
        # Majority fill variant
        def _fm(inp, bg):
            return _op_fill_enclosed_majority(inp, bg)
        return _fm

    return None


def _structural_candidates(ws):
    """Always-tried candidates: symmetry repair + copy stamp + cross-demo ops."""
    bg = ws.bg

    # Symmetry repair
    for sym in ["h", "v"]:
        for pbg in [False, True]:
            def _s(inp, dbg, s=sym, p=pbg):
                return _op_symmetry_repair(inp, dbg, s, p)
            yield _s
    for sym in ["h", "v"]:
        def _sa(inp, dbg, s=sym):
            return _op_symmetry_repair_auto(inp, dbg, s)
        yield _sa

    # Copy stamp
    non_singleton_colors = set(o.color for o in ws.objects if not o.is_singleton)
    singleton_colors = set(o.color for o in ws.objects if o.is_singleton)
    for sc in non_singleton_colors:
        for mc in singleton_colors:
            if sc != mc:
                def _c(inp, dbg, s=sc, m=mc):
                    return _op_copy_stamp(inp, dbg, s, m)
                yield _c

    # Cross-demo: fill enclosed with enclosing frame's own color
    yield _op_fill_enclosed_with_frame_color

    # Cross-demo: recolor target to adjacent singleton's color
    yield _op_recolor_to_adjacent_singleton


def _unguided_fallback(bg):
    """Yield operations not covered by guided candidates."""
    from aria.guided.grammar import _repair_periodic_region

    # Periodic repair
    for axis in ["row", "col"]:
        def _p(inp, dbg, ax=axis):
            return _repair_periodic_region(inp, ax)
        yield _p

    # Simple recolors
    for fc in range(10):
        for tc in range(10):
            if fc == tc:
                continue
            def _r(inp, dbg, f=fc, t=tc):
                out = inp.copy()
                out[inp == f] = t
                return out
            yield _r

    # Enclosed fills
    for c in range(10):
        def _f(inp, dbg, color=c):
            return _op_fill_enclosed(inp, dbg, color)
        yield _f

    # Majority fill
    yield _op_fill_enclosed_majority

    # Delete by color
    for c in range(10):
        def _d(inp, dbg, color=c):
            out = inp.copy()
            out[inp == color] = dbg
            return out
        yield _d


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def _extract_objects_8conn(grid, bg):
    """Extract objects with 8-connectivity."""
    from aria.guided.workspace import ObjectInfo
    rows, cols = grid.shape
    objects = []
    oid = 0
    struct8 = np.ones((3, 3), dtype=np.uint8)
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary, structure=struct8)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labeled == label_id)
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            mask = labeled[r0:r1 + 1, c0:c1 + 1] == label_id
            size = int(mask.sum())
            objects.append(ObjectInfo(
                oid=oid, color=color, row=r0, col=c0,
                height=r1 - r0 + 1, width=c1 - c0 + 1,
                size=size, mask=mask, is_singleton=(size == 1),
            ))
            oid += 1
    return objects


def _op_fill_enclosed(grid, bg, fill_color):
    from collections import deque
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
    out = grid.copy()
    out[(grid == bg) & ~reachable] = fill_color
    return out


def _op_fill_enclosed_majority(grid, bg):
    from collections import deque, Counter
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
    enclosed = (grid == bg) & ~reachable
    if not np.any(enclosed):
        return grid.copy()
    out = grid.copy()
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for lid in range(1, n + 1):
        comp = labeled == lid
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        vals = grid[border]
        non_bg = vals[vals != bg]
        if len(non_bg) > 0:
            out[comp] = int(Counter(non_bg.tolist()).most_common(1)[0][0])
    return out


def _op_symmetry_repair(grid, bg, sym_axis, prefer_bg):
    """Repair each non-bg region to be symmetric along the given axis.

    Uses bounding box of all non-bg connected components (8-conn),
    so isolated defect pixels are included.
    """
    out = grid.copy()
    rows, cols = grid.shape
    non_bg = grid != bg
    if not np.any(non_bg):
        return out

    labeled, n = ndimage.label(non_bg, structure=np.ones((3, 3)))
    for lid in range(1, n + 1):
        comp = labeled == lid
        ys, xs = np.where(comp)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1
        sub = out[r0:r0 + h, c0:c0 + w].copy()
        obj_vals = grid[comp]
        obj_color = int(np.bincount(obj_vals[obj_vals != bg]).argmax()) if np.any(obj_vals != bg) else 0
        size = int(np.sum(comp))

        if sym_axis == "h":
            asym = int(np.sum(sub != sub[:, ::-1]))
        else:
            asym = int(np.sum(sub != sub[::-1, :]))

        if asym == 0 or asym > max(2, size // 3):
            continue

        repaired = sub.copy()
        if sym_axis == "h":
            for r in range(h):
                for c in range(w // 2):
                    mc = w - 1 - c
                    if repaired[r, c] != repaired[r, mc]:
                        winner = _sym_pick(repaired[r, c], repaired[r, mc], bg, obj_color, prefer_bg)
                        repaired[r, c] = winner
                        repaired[r, mc] = winner
            new_asym = int(np.sum(repaired != repaired[:, ::-1]))
        else:
            for r in range(h // 2):
                mr = h - 1 - r
                for c in range(w):
                    if repaired[r, c] != repaired[mr, c]:
                        winner = _sym_pick(repaired[r, c], repaired[mr, c], bg, obj_color, prefer_bg)
                        repaired[r, c] = winner
                        repaired[mr, c] = winner
            new_asym = int(np.sum(repaired != repaired[::-1, :]))

        if new_asym == 0:
            out[r0:r0 + h, c0:c0 + w] = repaired

    return out


def _op_symmetry_repair_auto(grid, bg, sym_axis):
    """Per-region symmetry repair: find non-bg bounding boxes and symmetrize.

    Uses the bounding box of all non-bg pixels (not per-object), so isolated
    defect pixels that disconnected from the main shape are still covered.
    """
    out = grid.copy()
    rows, cols = grid.shape

    # Find all non-bg connected regions using 8-connectivity on non-bg mask
    non_bg = grid != bg
    if not np.any(non_bg):
        return out

    labeled, n = ndimage.label(non_bg, structure=np.ones((3, 3)))

    for lid in range(1, n + 1):
        comp = labeled == lid
        ys, xs = np.where(comp)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1

        sub = out[r0:r0 + h, c0:c0 + w].copy()
        obj_color_vals = grid[comp]
        obj_color = int(np.bincount(obj_color_vals[obj_color_vals != bg]).argmax()) if np.any(obj_color_vals != bg) else 0
        size = int(np.sum(comp))

        if sym_axis == "h":
            asym = int(np.sum(sub != sub[:, ::-1]))
        else:
            asym = int(np.sum(sub != sub[::-1, :]))

        if asym == 0 or asym > max(2, size // 3):
            continue

        best_sub = None
        best_changes = h * w + 1

        for prefer_bg in [False, True]:
            candidate = sub.copy()
            if sym_axis == "h":
                for r in range(h):
                    for c in range(w // 2):
                        mc = w - 1 - c
                        if candidate[r, c] != candidate[r, mc]:
                            winner = _sym_pick(candidate[r, c], candidate[r, mc], bg, obj_color, prefer_bg)
                            candidate[r, c] = winner
                            candidate[r, mc] = winner
                new_asym = int(np.sum(candidate != candidate[:, ::-1]))
            else:
                for r in range(h // 2):
                    mr = h - 1 - r
                    for c in range(w):
                        if candidate[r, c] != candidate[mr, c]:
                            winner = _sym_pick(candidate[r, c], candidate[mr, c], bg, obj_color, prefer_bg)
                            candidate[r, c] = winner
                            candidate[mr, c] = winner
                new_asym = int(np.sum(candidate != candidate[::-1, :]))

            if new_asym == 0:
                changes = int(np.sum(candidate != sub))
                if changes < best_changes:
                    best_changes = changes
                    best_sub = candidate
                break

        if best_sub is not None:
            out[r0:r0 + h, c0:c0 + w] = best_sub

    return out


def _op_fill_enclosed_with_frame_color(grid, bg):
    """Fill each enclosed bg region with the color of its enclosing frame.

    Cross-demo: the fill color is determined by the input structure,
    not a fixed parameter. Different demos → different fill colors.
    """
    from collections import deque, Counter
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
    enclosed = (grid == bg) & ~reachable
    if not np.any(enclosed):
        return grid.copy()

    out = grid.copy()
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for lid in range(1, n + 1):
        comp = labeled == lid
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        vals = grid[border]
        non_bg = vals[vals != bg]
        if len(non_bg) > 0:
            fill_c = int(Counter(non_bg.tolist()).most_common(1)[0][0])
            out[comp] = fill_c
    return out


def _op_recolor_to_adjacent_singleton(grid, bg):
    """Recolor each non-singleton object to match its adjacent singleton's color.

    Cross-demo: the target color comes from the input structure.
    """
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    singletons = {(o.row, o.col): o for o in objs if o.is_singleton}

    out = grid.copy()
    for obj in objs:
        if obj.is_singleton:
            continue
        # Find adjacent singletons
        adj_singleton_colors = set()
        for r in range(obj.height):
            for c in range(obj.width):
                if not obj.mask[r, c]:
                    continue
                gr, gc = obj.row + r, obj.col + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = gr + dr, gc + dc
                    if (nr, nc) in singletons:
                        adj_singleton_colors.add(singletons[(nr, nc)].color)
        if len(adj_singleton_colors) == 1:
            target = next(iter(adj_singleton_colors))
            if target != obj.color:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            out[obj.row + r, obj.col + c] = target
    return out


def _sym_pick(a, b, bg, obj_color, prefer_bg):
    if prefer_bg:
        if a == bg or b == bg:
            return bg
        return a if a == obj_color else b
    else:
        if a != bg and b == bg:
            return a
        if b != bg and a == bg:
            return b
        return a if a == obj_color else b


def _op_copy_stamp(grid, bg, stamp_color, marker_color):
    """Copy the stamp-colored object to the marker-colored singleton's location."""
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)

    # Find stamp source (largest non-singleton of stamp_color)
    stamp_objs = sorted(
        [o for o in objs if o.color == stamp_color and not o.is_singleton],
        key=lambda o: -o.size,
    )
    if not stamp_objs:
        return grid.copy()
    stamp = stamp_objs[0]
    stamp_grid = grid[stamp.row:stamp.row + stamp.height,
                      stamp.col:stamp.col + stamp.width].copy()

    # Find marker (singleton of marker_color)
    markers = [o for o in objs if o.color == marker_color and o.is_singleton]
    if not markers:
        return grid.copy()

    out = grid.copy()
    for marker in markers:
        mr, mc = marker.row, marker.col
        sh, sw = stamp.height, stamp.width
        # Overlay non-bg stamp pixels at marker position
        for r in range(sh):
            for c in range(sw):
                if stamp_grid[r, c] != bg:
                    tr, tc = mr + r, mc + c
                    if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                        out[tr, tc] = stamp_grid[r, c]
    return out
