"""LEGACY — superseded by induce.py (clause-based induction).

This module hard-routes to REPAIR/COMPLETE/PROPAGATE/DECODE branches,
each with named transforms (symmetry_repair, periodic_repair, etc.).
That is a parallel solver with handcrafted strategy menus, not the
clause-based architecture. Its transforms should become clause primitives
or clause derivations.

Do not add new functionality here.

---
Answer program induction conditioned on scene interpretation.

Each answer mode routes to a different program construction strategy.
The interpretation tells us WHAT to do; this module figures out HOW.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from collections import Counter

from aria.guided.interpret import (
    SceneInterpretation, Role, AnswerMode, SemanticRel,
)
from aria.guided.workspace import Workspace, build_workspace, _detect_bg, ObjectInfo
from aria.types import Grid


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def induce_answer(
    demos: list[tuple[Grid, Grid]],
    interp: SceneInterpretation,
) -> AnswerProgram | None:
    """Induce an answer program from demos + interpretation."""
    mode = interp.answer_mode

    if mode == AnswerMode.REPAIR:
        return _induce_repair(demos, interp)
    if mode == AnswerMode.COMPLETE:
        return _induce_complete(demos, interp)
    if mode == AnswerMode.PROPAGATE:
        return _induce_propagate(demos, interp)
    if mode == AnswerMode.DECODE:
        return _induce_decode(demos, interp)

    return None


class AnswerProgram:
    """An executable answer program."""
    def __init__(self, mode: str, apply_fn, description: str = ""):
        self.mode = mode
        self._apply = apply_fn
        self.description = description

    def apply(self, inp: Grid) -> Grid:
        return self._apply(inp)

    def __repr__(self):
        return f"AnswerProgram({self.mode}: {self.description})"


def _orig_diff(demos):
    total = 0
    for inp, out in demos:
        if inp.shape == out.shape:
            total += int(np.sum(inp != out))
        else:
            total += out.size
    return total


def verify_on_train(prog: AnswerProgram, demos: list[tuple[Grid, Grid]]) -> tuple[bool, int]:
    """Verify program on all train demos. Returns (exact_match, total_diff)."""
    total_diff = 0
    for inp, out in demos:
        try:
            pred = prog.apply(inp)
        except Exception:
            return False, sum(out.size for _, out in demos)
        if pred.shape != out.shape:
            return False, sum(out.size for _, out in demos)
        d = int(np.sum(pred != out))
        total_diff += d
    return total_diff == 0, total_diff


# ---------------------------------------------------------------------------
# REPAIR: fix anomalies inside scaffolds
# ---------------------------------------------------------------------------

def _induce_repair(demos, interp):
    """Repair TARGET objects that are anomalies of their SCAFFOLD.

    Strategy: for each scaffold, infer the pattern its children should
    follow (from non-target children), then repair targets to match.
    """
    candidates = []

    # Try symmetry repair (per-object, both axes)
    for axis in ['h', 'v']:
        def _fn(inp, ax=axis):
            return _apply_symmetry_repair(inp, ax)
        prog = AnswerProgram("repair", _fn, f"symmetry_repair_{axis}")
        ok, diff = verify_on_train(prog, demos)
        candidates.append((ok, diff, prog))

    # Try periodic repair (rows and cols)
    for axis in ['row', 'col']:
        def _fn(inp, ax=axis):
            return _apply_periodic_repair(inp, ax)
        prog = AnswerProgram("repair", _fn, f"periodic_repair_{axis}")
        ok, diff = verify_on_train(prog, demos)
        candidates.append((ok, diff, prog))

    # Take best
    candidates.sort(key=lambda x: (not x[0], x[1]))
    if candidates and candidates[0][1] < _orig_diff(demos):
        return candidates[0][2]
    return None


def _apply_symmetry_repair(grid, axis):
    bg = _detect_bg(grid)
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
        sub = out[r0:r0+h, c0:c0+w].copy()
        vals = grid[comp]
        obj_c = int(Counter(vals[vals != bg].tolist()).most_common(1)[0][0]) if np.any(vals != bg) else 0
        size = int(np.sum(comp))
        if axis == 'h':
            asym = int(np.sum(sub != sub[:, ::-1]))
        else:
            asym = int(np.sum(sub != sub[::-1, :]))
        if asym == 0 or asym > max(2, size // 3):
            continue
        repaired = sub.copy()
        if axis == 'h':
            for r in range(h):
                for c in range(w // 2):
                    mc = w - 1 - c
                    if repaired[r, c] != repaired[r, mc]:
                        if repaired[r, c] != bg:
                            repaired[r, mc] = repaired[r, c]
                        else:
                            repaired[r, c] = repaired[r, mc]
            if int(np.sum(repaired != repaired[:, ::-1])) == 0:
                out[r0:r0+h, c0:c0+w] = repaired
        else:
            for r in range(h // 2):
                mr = h - 1 - r
                for c in range(w):
                    if repaired[r, c] != repaired[mr, c]:
                        if repaired[r, c] != bg:
                            repaired[mr, c] = repaired[r, c]
                        else:
                            repaired[r, c] = repaired[mr, c]
            if int(np.sum(repaired != repaired[::-1, :])) == 0:
                out[r0:r0+h, c0:c0+w] = repaired
    return out


def _apply_periodic_repair(grid, axis):
    out = grid.copy()
    if axis == 'row':
        for r in range(grid.shape[0]):
            out[r, :] = _repair_line(grid[r, :])
    else:
        for c in range(grid.shape[1]):
            out[:, c] = _repair_line(grid[:, c])
    return out


def _repair_line(line):
    n = len(line)
    if n <= 1:
        return line.copy()
    for p in range(1, n // 2 + 1):
        tile = line[:p]
        if all(np.array_equal(line[i:i+p], tile[:min(p, n-i)]) for i in range(p, n, p)):
            return line.copy()
    best, best_v = None, n
    for p in range(1, n // 2 + 1):
        n_reps = n // p
        for ref in range(0, n - p + 1, p):
            tile = line[ref:ref+p]
            repaired = np.array([tile[i % p] for i in range(n)], dtype=line.dtype)
            v = int(np.sum(repaired != line))
            if 0 < v <= max(1, n_reps // 2) and v < best_v:
                best_v = v
                best = repaired
        pattern = np.zeros(p, dtype=line.dtype)
        for phase in range(p):
            vals = [int(line[i]) for i in range(phase, n, p)]
            pattern[phase] = Counter(vals).most_common(1)[0][0]
        repaired = np.array([pattern[i % p] for i in range(n)], dtype=line.dtype)
        v = int(np.sum(repaired != line))
        if 0 < v <= max(1, n_reps // 2) and v < best_v:
            best_v = v
            best = repaired
    return best if best is not None else line.copy()


# ---------------------------------------------------------------------------
# COMPLETE: fill in missing parts following a pattern
# ---------------------------------------------------------------------------

def _induce_complete(demos, interp):
    """Complete a pattern by filling gaps.

    Strategy: identify the repeating pattern from preserved regions,
    then fill target/gap regions to match.
    """
    candidates = []

    # Try periodic repair (often the pattern IS periodic)
    for axis in ['row', 'col']:
        def _fn(inp, ax=axis):
            return _apply_periodic_repair(inp, ax)
        prog = AnswerProgram("complete", _fn, f"complete_periodic_{axis}")
        ok, diff = verify_on_train(prog, demos)
        candidates.append((ok, diff, prog))

    # Try fill enclosed with frame color
    def _fn_fill_frame(inp):
        return _apply_fill_enclosed_frame(inp)
    prog = AnswerProgram("complete", _fn_fill_frame, "fill_enclosed_with_frame_color")
    ok, diff = verify_on_train(prog, demos)
    candidates.append((ok, diff, prog))

    # Try fill enclosed with each color from output residual
    for demo_inp, demo_out in demos[:1]:
        diff_mask = demo_inp != demo_out if demo_inp.shape == demo_out.shape else np.ones(demo_out.shape, dtype=bool)
        out_colors = set(int(v) for v in np.unique(demo_out[diff_mask]))
        for c in out_colors:
            def _fn(inp, color=c):
                return _apply_fill_enclosed(inp, color)
            prog = AnswerProgram("complete", _fn, f"fill_enclosed_{c}")
            ok, diff = verify_on_train(prog, demos)
            candidates.append((ok, diff, prog))

    candidates.sort(key=lambda x: (not x[0], x[1]))
    if candidates and candidates[0][1] < _orig_diff(demos):
        return candidates[0][2]
    return None


def _apply_fill_enclosed_frame(grid):
    bg = _detect_bg(grid)
    out = grid.copy()
    rows, cols = grid.shape
    from collections import deque
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))
    enclosed = (grid == bg) & ~reachable
    if not np.any(enclosed):
        return out
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
    struct4 = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
    for lid in range(1, n+1):
        comp = labeled == lid
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        vals = grid[border]
        non_bg = vals[vals != bg]
        if len(non_bg) > 0:
            out[comp] = int(Counter(non_bg.tolist()).most_common(1)[0][0])
    return out


def _apply_fill_enclosed(grid, fill_color):
    bg = _detect_bg(grid)
    out = grid.copy()
    rows, cols = grid.shape
    from collections import deque
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))
    enclosed = (grid == bg) & ~reachable
    out[enclosed] = fill_color
    return out


# ---------------------------------------------------------------------------
# PROPAGATE: extend a rule from data to targets
# ---------------------------------------------------------------------------

def _induce_propagate(demos, interp):
    """Propagate a rule observed in DATA objects to TARGET objects.

    Strategy: find what distinguishes target from data objects,
    then apply the reverse transformation.
    """
    candidates = []

    # Try: delete objects by color (for each color that appears only in targets)
    ws = build_workspace(demos[0][0], demos[0][1])
    target_oids = set(interp.target_objects)
    target_colors = set()
    data_colors = set()
    for obj in ws.objects:
        if obj.oid in target_oids:
            target_colors.add(obj.color)
        else:
            data_colors.add(obj.color)

    exclusive_target_colors = target_colors - data_colors
    for c in exclusive_target_colors:
        def _fn(inp, color=c):
            bg = _detect_bg(inp)
            out = inp.copy()
            out[inp == color] = bg
            return out
        prog = AnswerProgram("propagate", _fn, f"delete_color_{c}")
        ok, diff = verify_on_train(prog, demos)
        candidates.append((ok, diff, prog))

    # Try: recolor target objects to adjacent singleton color
    def _fn_adj(inp):
        return _apply_recolor_to_adjacent(inp)
    prog = AnswerProgram("propagate", _fn_adj, "recolor_to_adjacent_singleton")
    ok, diff = verify_on_train(prog, demos)
    candidates.append((ok, diff, prog))

    # Try symmetry repair
    for axis in ['h', 'v']:
        def _fn(inp, ax=axis):
            return _apply_symmetry_repair(inp, ax)
        prog = AnswerProgram("propagate", _fn, f"symmetry_repair_{axis}")
        ok, diff = verify_on_train(prog, demos)
        candidates.append((ok, diff, prog))

    candidates.sort(key=lambda x: (not x[0], x[1]))
    if candidates and candidates[0][1] < _orig_diff(demos):
        return candidates[0][2]
    return None


def _apply_recolor_to_adjacent(grid):
    bg = _detect_bg(grid)
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    singletons = {(o.row, o.col): o for o in objs if o.is_singleton}
    out = grid.copy()
    for obj in objs:
        if obj.is_singleton:
            continue
        adj_colors = set()
        for r in range(obj.height):
            for c in range(obj.width):
                if not obj.mask[r, c]:
                    continue
                gr, gc = obj.row + r, obj.col + c
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if (gr+dr, gc+dc) in singletons:
                        adj_colors.add(singletons[(gr+dr, gc+dc)].color)
        if len(adj_colors) == 1:
            tc = next(iter(adj_colors))
            if tc != obj.color:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            out[obj.row + r, obj.col + c] = tc
    return out


# ---------------------------------------------------------------------------
# DECODE: use legend to translate
# ---------------------------------------------------------------------------

def _induce_decode(demos, interp):
    """Use a LEGEND region to decode/translate the scene.

    Strategy: extract the key→value mapping from the legend,
    then apply it to data regions to produce the answer.
    """
    # For now, try the simpler variant: the legend defines a color map
    # Extract color map from the legend region
    candidates = []

    # Try: the legend maps frame colors to fill colors
    def _fn_frame(inp):
        return _apply_fill_enclosed_frame(inp)
    prog = AnswerProgram("decode", _fn_frame, "decode_fill_frame")
    ok, diff = verify_on_train(prog, demos)
    candidates.append((ok, diff, prog))

    candidates.sort(key=lambda x: (not x[0], x[1]))
    if candidates and candidates[0][1] < _orig_diff(demos):
        return candidates[0][2]
    return None
