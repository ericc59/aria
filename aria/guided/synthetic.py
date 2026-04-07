"""Synthetic task generator — multi-step compositional tasks.

Each task has a PROGRAM (sequence of grammar actions) as its latent rule.
Programs compose multiple steps: select targets, apply rewrites, move on.
Different demos instantiate the same abstract program with different
concrete values (colors, sizes, positions vary; structure is shared).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from collections import Counter, deque

from aria.guided.grammar import (
    Program, Action, Act, Support, Target, Rewrite, BindSource,
    execute_program,
)
from aria.types import Grid


@dataclass
class SyntheticTask:
    task_id: str
    train: list[tuple[Grid, Grid]]
    test: list[tuple[Grid, Grid]]
    rule_type: str
    latent_program: Program
    latent_params: dict[str, Any]


def generate_benchmark(n_tasks: int = 200, seed: int = 42, n_train: int = 3, n_test: int = 1) -> list[SyntheticTask]:
    rng = np.random.RandomState(seed)
    tasks = []
    generators = [
        _gen_fill_enclosed,
        _gen_recolor_by_color,
        _gen_periodic_repair,
        _gen_symmetry_repair,
        _gen_copy_stamp,
        _gen_fill_with_frame_color,
        _gen_recolor_to_neighbor,
        _gen_fill_then_recolor,
        _gen_delete_then_symmetrize,
    ]
    attempts = 0
    i = 0
    while len(tasks) < n_tasks and attempts < n_tasks * 5:
        gen_fn = generators[i % len(generators)]
        task = gen_fn(rng, attempts, n_train, n_test)
        attempts += 1
        i += 1
        # Validate: latent program must reproduce all train outputs
        ok = True
        for inp, out in task.train:
            bg = _detect_bg_simple(inp)
            try:
                pred = execute_program(task.latent_program, inp, bg)
            except Exception:
                ok = False
                break
            if not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            tasks.append(task)
    return tasks


def _detect_bg_simple(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def split_benchmark(tasks, train_frac=0.7):
    n = int(len(tasks) * train_frac)
    return tasks[:n], tasks[n:]


# ===========================================================================
# Helpers
# ===========================================================================

def _make_framed_grid(rng, bg=0):
    size = rng.randint(7, 15)
    frame_color = rng.randint(1, 10)
    while frame_color == bg:
        frame_color = rng.randint(1, 10)
    grid = np.full((size, size), bg, dtype=np.uint8)
    fr, fc = rng.randint(1, size // 3), rng.randint(1, size // 3)
    fh = rng.randint(3, size - fr - 1)
    fw = rng.randint(3, size - fc - 1)
    grid[fr, fc:fc + fw] = frame_color
    grid[fr + fh - 1, fc:fc + fw] = frame_color
    grid[fr:fr + fh, fc] = frame_color
    grid[fr:fr + fh, fc + fw - 1] = frame_color
    return grid, frame_color, (fr, fc, fh, fw)


def _make_symmetric_object(rng, bg=0):
    obj_color = rng.randint(1, 10)
    while obj_color == bg:
        obj_color = rng.randint(1, 10)
    oh = rng.randint(2, 4) * 2
    ow = rng.randint(2, 4) * 2
    full = np.full((oh, ow), obj_color, dtype=np.uint8)
    for _ in range(rng.randint(0, oh * ow // 4)):
        full[rng.randint(0, oh), rng.randint(0, ow // 2)] = bg
    for r in range(oh):
        for c in range(ow // 2):
            full[r, ow - 1 - c] = full[r, c]
    return full, obj_color, oh, ow


def _do_fill_enclosed(grid, bg, color_val, color_src):
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
        return
    if color_src == "from_context:frame":
        labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
        struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        for lid in range(1, n + 1):
            comp = labeled == lid
            dilated = ndimage.binary_dilation(comp, structure=struct4)
            border = dilated & ~comp
            vals = grid[border]
            non_bg = vals[vals != bg]
            if len(non_bg) > 0:
                grid[comp] = int(Counter(non_bg.tolist()).most_common(1)[0][0])
    else:
        grid[enclosed] = color_val


# ===========================================================================
# Single-step generators
# ===========================================================================

def _gen_fill_enclosed(rng, idx, n_train, n_test):
    fill_color = rng.randint(1, 10)
    demos = []
    for _ in range(n_train + n_test):
        grid, fc, _ = _make_framed_grid(rng)
        while fill_color == fc:
            fill_color = rng.randint(1, 10)
        out = grid.copy()
        _do_fill_enclosed(out, 0, fill_color, "literal")
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
        Action(Act.REWRITE, Rewrite.FILL),
        Action(Act.BIND, param_name="color", param_value=fill_color),
        Action(Act.BIND, param_name="color_source", param_value="literal"),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_fill_{idx:04d}", demos[:n_train], demos[n_train:],
                         "fill_enclosed", prog, {"fill_color": fill_color})


def _gen_recolor_by_color(rng, idx, n_train, n_test):
    from_c = rng.randint(1, 8)
    to_c = rng.randint(1, 10)
    while to_c == from_c:
        to_c = rng.randint(1, 10)
    other_c = rng.randint(1, 10)
    while other_c in (from_c, to_c):
        other_c = rng.randint(1, 10)
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(8, 14)
        grid = np.zeros((size, size), dtype=np.uint8)
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, size - 2), rng.randint(0, size - 2)
            grid[r:r + rng.randint(1, 3), c:c + rng.randint(1, 3)] = from_c
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, size - 2), rng.randint(0, size - 2)
            mask = grid[r:r + 2, c:c + 2] == 0
            grid[r:r + 2, c:c + 2][mask] = other_c
        out = grid.copy()
        out[out == from_c] = to_c
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.BY_COLOR),
        Action(Act.REWRITE, Rewrite.RECOLOR),
        Action(Act.BIND, param_name="from_color", param_value=from_c),
        Action(Act.BIND, param_name="color", param_value=to_c),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_recolor_{idx:04d}", demos[:n_train], demos[n_train:],
                         "recolor_by_color", prog, {"from_color": from_c, "to_color": to_c})


def _gen_periodic_repair(rng, idx, n_train, n_test):
    axis = "row" if rng.rand() < 0.5 else "col"
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(6, 14)
        period = rng.randint(2, min(4, size // 2 + 1))
        colors = rng.choice(range(1, 10), size=min(period, 4), replace=False).tolist()
        tile = np.array([colors[i % len(colors)] for i in range(period)], dtype=np.uint8)
        if axis == "row":
            clean = np.tile(tile, (size, size // period + 1))[:size, :size]
        else:
            clean = np.tile(tile.reshape(-1, 1), (size // period + 1, size))[:size, :size]
        inp = clean.copy()
        for _ in range(rng.randint(1, max(2, size // 3))):
            r, c = rng.randint(0, size), rng.randint(0, size)
            inp[r, c] = rng.randint(0, 10)
        demos.append((inp, clean))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ANOMALY),
        Action(Act.REWRITE, Rewrite.PERIODIC_REPAIR),
        Action(Act.BIND, param_name="axis", param_value=axis),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_periodic_{idx:04d}", demos[:n_train], demos[n_train:],
                         "periodic_repair", prog, {"axis": axis})


def _gen_symmetry_repair(rng, idx, n_train, n_test):
    sym_axis = "h" if rng.rand() < 0.5 else "v"
    bg = int(rng.choice([0, 1, 2, 3]))
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(8, 16)
        full, obj_color, oh, ow = _make_symmetric_object(rng, bg)
        or_ = rng.randint(1, max(2, size - oh))
        oc = rng.randint(1, max(2, size - ow))
        out_grid = np.full((size, size), bg, dtype=np.uint8)
        out_grid[or_:or_ + oh, oc:oc + ow] = full
        inp_grid = out_grid.copy()
        if sym_axis == "h":
            fixable = [(r, c) for r in range(oh) for c in range(ow // 2 + 1, ow)
                        if full[r, c] == obj_color and full[r, ow - 1 - c] == obj_color]
        else:
            fixable = [(r, c) for r in range(oh // 2 + 1, oh) for c in range(ow)
                        if full[r, c] == obj_color and full[oh - 1 - r, c] == obj_color]
        if fixable:
            n_d = rng.randint(1, max(2, len(fixable) // 3))
            for i2 in rng.choice(len(fixable), min(n_d, len(fixable)), replace=False):
                dr, dc = fixable[i2]
                inp_grid[or_ + dr, oc + dc] = bg
        demos.append((inp_grid, out_grid))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ASYMMETRIC),
        Action(Act.REWRITE, Rewrite.SYMMETRIZE),
        Action(Act.BIND, param_name="axis", param_value=sym_axis),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_sym_{idx:04d}", demos[:n_train], demos[n_train:],
                         "symmetry_repair", prog, {"axis": sym_axis})


def _gen_copy_stamp(rng, idx, n_train, n_test):
    stamp_color = rng.randint(1, 8)
    marker_color = rng.randint(1, 10)
    while marker_color == stamp_color:
        marker_color = rng.randint(1, 10)
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(10, 18)
        sh, sw = rng.randint(2, 4), rng.randint(2, 4)
        grid = np.zeros((size, size), dtype=np.uint8)
        sr, sc = rng.randint(0, size - sh), rng.randint(0, size - sw)
        stamp = np.full((sh, sw), stamp_color, dtype=np.uint8)
        for _ in range(rng.randint(0, sh * sw // 3)):
            stamp[rng.randint(0, sh), rng.randint(0, sw)] = 0
        if np.sum(stamp != 0) < 2:
            stamp[0, 0] = stamp_color; stamp[-1, -1] = stamp_color
        grid[sr:sr + sh, sc:sc + sw] = stamp
        mr, mc = rng.randint(0, size - sh), rng.randint(0, size - sw)
        while abs(mr - sr) < sh + 1 and abs(mc - sc) < sw + 1:
            mr, mc = rng.randint(0, size - sh), rng.randint(0, size - sw)
        grid[mr, mc] = marker_color
        out = grid.copy()
        for r in range(sh):
            for c in range(sw):
                if stamp[r, c] != 0:
                    tr, tc = mr + r, mc + c
                    if 0 <= tr < size and 0 <= tc < size:
                        out[tr, tc] = stamp[r, c]
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.SINGLETONS),
        Action(Act.REWRITE, Rewrite.COPY_STAMP),
        Action(Act.BIND, param_name="stamp_color", param_value=stamp_color),
        Action(Act.BIND, param_name="marker_color", param_value=marker_color),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_copy_{idx:04d}", demos[:n_train], demos[n_train:],
                         "copy_stamp", prog, {"stamp_color": stamp_color, "marker_color": marker_color})


# ===========================================================================
# Cross-demo varying
# ===========================================================================

def _gen_fill_with_frame_color(rng, idx, n_train, n_test):
    demos = []
    for _ in range(n_train + n_test):
        grid, fc, _ = _make_framed_grid(rng)
        out = grid.copy()
        _do_fill_enclosed(out, 0, fc, "from_context:frame")
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
        Action(Act.REWRITE, Rewrite.FILL),
        Action(Act.BIND, param_name="color_source", param_value="from_context:frame"),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_fillframe_{idx:04d}", demos[:n_train], demos[n_train:],
                         "fill_with_frame_color", prog, {"color_rule": "frame_color"})


def _gen_recolor_to_neighbor(rng, idx, n_train, n_test):
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(8, 14)
        tc = rng.randint(1, 10)
        sc = rng.randint(1, 10)
        while sc == tc:
            sc = rng.randint(1, 10)
        grid = np.zeros((size, size), dtype=np.uint8)
        th, tw = rng.randint(2, 4), rng.randint(2, 4)
        tr, tcc = rng.randint(1, size - th - 1), rng.randint(1, size - tw - 1)
        grid[tr:tr + th, tcc:tcc + tw] = tc
        adj = []
        for r in range(max(0, tr - 1), min(size, tr + th + 1)):
            for c in range(max(0, tcc - 1), min(size, tcc + tw + 1)):
                if grid[r, c] == 0 and ((r in (tr - 1, tr + th) and tcc <= c < tcc + tw) or (c in (tcc - 1, tcc + tw) and tr <= r < tr + th)):
                    adj.append((r, c))
        if adj:
            sr, scc = adj[rng.randint(0, len(adj))]
            grid[sr, scc] = sc
        out = grid.copy()
        out[tr:tr + th, tcc:tcc + tw] = sc
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ADJACENT_TO),
        Action(Act.REWRITE, Rewrite.RECOLOR),
        Action(Act.BIND, param_name="color_source", param_value="from_context:adjacent_singleton"),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_recolneigh_{idx:04d}", demos[:n_train], demos[n_train:],
                         "recolor_to_neighbor", prog, {"color_rule": "adjacent_singleton"})


# ===========================================================================
# MULTI-STEP compositions
# ===========================================================================

def _gen_fill_then_recolor(rng, idx, n_train, n_test):
    """Step 1: fill enclosed bg with frame color. Step 2: recolor specific color."""
    recolor_from = rng.randint(1, 8)
    recolor_to = rng.randint(1, 10)
    while recolor_to == recolor_from:
        recolor_to = rng.randint(1, 10)
    demos = []
    for _ in range(n_train + n_test):
        grid, fc, _ = _make_framed_grid(rng)
        for _ in range(rng.randint(1, 3)):
            r, c = rng.randint(0, grid.shape[0] - 1), rng.randint(0, grid.shape[1] - 1)
            if grid[r, c] == 0:
                grid[r, c] = recolor_from
        out = grid.copy()
        _do_fill_enclosed(out, 0, fc, "from_context:frame")
        out[out == recolor_from] = recolor_to
        demos.append((grid, out))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
        Action(Act.REWRITE, Rewrite.FILL),
        Action(Act.BIND, param_name="color_source", param_value="from_context:frame"),
        Action(Act.NEXT),
        Action(Act.SELECT_TARGET, Target.BY_COLOR),
        Action(Act.REWRITE, Rewrite.RECOLOR),
        Action(Act.BIND, param_name="from_color", param_value=recolor_from),
        Action(Act.BIND, param_name="color", param_value=recolor_to),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_fillrecol_{idx:04d}", demos[:n_train], demos[n_train:],
                         "fill_then_recolor", prog,
                         {"recolor_from": recolor_from, "recolor_to": recolor_to})


def _gen_delete_then_symmetrize(rng, idx, n_train, n_test):
    """Step 1: delete singletons of a color. Step 2: symmetrize remaining objects."""
    bg = 0
    delete_color = rng.randint(1, 10)
    sym_axis = "h" if rng.rand() < 0.5 else "v"
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(10, 16)
        full, obj_color, oh, ow = _make_symmetric_object(rng, bg)
        while obj_color == delete_color:
            full, obj_color, oh, ow = _make_symmetric_object(rng, bg)
        or_ = rng.randint(1, max(2, size - oh))
        oc = rng.randint(1, max(2, size - ow))
        target = np.full((size, size), bg, dtype=np.uint8)
        target[or_:or_ + oh, oc:oc + ow] = full
        inp = target.copy()
        if sym_axis == "h":
            fixable = [(r, c) for r in range(oh) for c in range(ow // 2 + 1, ow)
                        if full[r, c] == obj_color and full[r, ow - 1 - c] == obj_color]
        else:
            fixable = [(r, c) for r in range(oh // 2 + 1, oh) for c in range(ow)
                        if full[r, c] == obj_color and full[oh - 1 - r, c] == obj_color]
        if fixable:
            n_d = rng.randint(1, max(2, len(fixable) // 3))
            for i2 in rng.choice(len(fixable), min(n_d, len(fixable)), replace=False):
                dr, dc = fixable[i2]
                inp[or_ + dr, oc + dc] = bg
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, size), rng.randint(0, size)
            if inp[r, c] == bg:
                inp[r, c] = delete_color
        demos.append((inp, target))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.BY_COLOR),
        Action(Act.REWRITE, Rewrite.DELETE),
        Action(Act.BIND, param_name="from_color", param_value=delete_color),
        Action(Act.NEXT),
        Action(Act.SELECT_TARGET, Target.ASYMMETRIC),
        Action(Act.REWRITE, Rewrite.SYMMETRIZE),
        Action(Act.BIND, param_name="axis", param_value=sym_axis),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_delsym_{idx:04d}", demos[:n_train], demos[n_train:],
                         "delete_then_symmetrize", prog,
                         {"delete_color": delete_color, "axis": sym_axis})


def _gen_periodic_then_fill(rng, idx, n_train, n_test):
    """Step 1: repair periodic pattern. Step 2: fill enclosed regions."""
    axis = "row" if rng.rand() < 0.5 else "col"
    fill_color = rng.randint(1, 10)
    demos = []
    for _ in range(n_train + n_test):
        size = rng.randint(8, 14)
        period = rng.randint(2, min(4, size // 2 + 1))
        colors = rng.choice(range(1, 10), size=min(period, 4), replace=False).tolist()
        while fill_color in colors:
            fill_color = rng.randint(1, 10)
        tile = np.array([colors[i % len(colors)] for i in range(period)], dtype=np.uint8)
        if axis == "row":
            clean = np.tile(tile, (size, size // period + 1))[:size, :size]
        else:
            clean = np.tile(tile.reshape(-1, 1), (size // period + 1, size))[:size, :size]
        fc = colors[0]
        fr, fcc = 1, 1
        fh, fw = min(5, size - 3), min(5, size - 3)
        clean[fr, fcc:fcc + fw] = fc
        clean[fr + fh - 1, fcc:fcc + fw] = fc
        clean[fr:fr + fh, fcc] = fc
        clean[fr:fr + fh, fcc + fw - 1] = fc
        target = clean.copy()
        for r in range(fr + 1, fr + fh - 1):
            for c in range(fcc + 1, fcc + fw - 1):
                target[r, c] = fill_color
        inp = clean.copy()
        for _ in range(rng.randint(1, max(2, size // 4))):
            r, c = rng.randint(0, size), rng.randint(0, size)
            if not (fr <= r < fr + fh and fcc <= c < fcc + fw):
                inp[r, c] = rng.randint(0, 10)
        demos.append((inp, target))
    prog = Program([
        Action(Act.SELECT_TARGET, Target.ANOMALY),
        Action(Act.REWRITE, Rewrite.PERIODIC_REPAIR),
        Action(Act.BIND, param_name="axis", param_value=axis),
        Action(Act.NEXT),
        Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
        Action(Act.REWRITE, Rewrite.FILL),
        Action(Act.BIND, param_name="color", param_value=fill_color),
        Action(Act.BIND, param_name="color_source", param_value="literal"),
        Action(Act.STOP),
    ])
    return SyntheticTask(f"syn_periodfill_{idx:04d}", demos[:n_train], demos[n_train:],
                         "periodic_then_fill", prog,
                         {"axis": axis, "fill_color": fill_color})
