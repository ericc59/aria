"""Primitive-graph expansion grammar — sequential graph construction.

A program is built step-by-step by choosing actions from the grammar.
Each action extends the current partial program. The sequence of
actions IS the program. Execution replays the action sequence.

Action vocabulary:
  SELECT_UNIT(uid)         — pick which residual unit to explain
  SELECT_SUPPORT(method)   — pick how to find input support for that unit
  SELECT_TARGET(method)    — pick which cells/objects within support to act on
  REWRITE(op)              — pick the primitive rewrite for the target
  BIND(param, value)       — bind a parameter (color, axis, etc.)
  NEXT                     — move to the next unit
  STOP                     — declare the program complete

A complete program looks like:
  SELECT_UNIT(0) → SELECT_SUPPORT(same_bbox) → SELECT_TARGET(enclosed_bg) →
  REWRITE(fill) → BIND(color, from_context:frame) → NEXT →
  SELECT_UNIT(1) → SELECT_SUPPORT(per_object) → SELECT_TARGET(asymmetric) →
  REWRITE(symmetrize) → BIND(axis, h) → STOP

This is the construction language. It composes naturally — multi-step
programs emerge from multiple NEXT cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from scipy import ndimage
from collections import Counter, deque

from aria.types import Grid


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class Act(Enum):
    SELECT_UNIT = auto()
    SELECT_SUPPORT = auto()
    SELECT_TARGET = auto()
    REWRITE = auto()
    BIND = auto()
    NEXT = auto()
    STOP = auto()


# Support methods
class Support(Enum):
    SAME_BBOX = auto()       # same position in input
    FULL_GRID = auto()       # entire input grid
    PER_OBJECT = auto()      # each object independently
    ENCLOSED_REGIONS = auto() # bg regions enclosed by non-bg


# Target methods
class Target(Enum):
    ALL_CELLS = auto()       # all cells in support
    BY_COLOR = auto()        # cells of a specific color
    ENCLOSED_BG = auto()     # bg cells enclosed by non-bg
    ASYMMETRIC = auto()      # cells that break symmetry
    ANOMALY = auto()         # cells that break periodicity
    SINGLETONS = auto()      # singleton objects
    ADJACENT_TO = auto()     # cells adjacent to specific objects


# Rewrite operations
class Rewrite(Enum):
    FILL = auto()            # fill target with a color
    RECOLOR = auto()         # change color of target cells
    DELETE = auto()           # set target to bg
    SYMMETRIZE = auto()      # repair symmetry
    PERIODIC_REPAIR = auto() # repair periodic pattern
    COPY_STAMP = auto()      # copy from source to target
    # Context-parameterized operations
    FILL_WITH_FRAME = auto()     # fill enclosed bg with enclosing frame's color
    RECOLOR_TO_ADJ = auto()      # recolor object to adjacent singleton's color
    SWAP_COLORS = auto()         # swap two colors (both directions)
    MOVE_TO_ENCLOSED = auto()    # move selected objects into nearest enclosed region
    SWAP_ENCLOSED = auto()       # swap colors of enclosed object pairs
    COLOR_MAP = auto()           # apply per-pixel color map (simultaneous)


# Bind value sources
class BindSource(Enum):
    LITERAL = auto()         # a fixed value (color=3, axis=h)
    FROM_CONTEXT = auto()    # derived from structure (frame_color, adjacent_color)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    act: Act
    choice: Any = None       # the enum variant or value chosen
    param_name: str = ""     # for BIND: which param
    param_value: Any = None  # for BIND: the value or source

    def __repr__(self):
        if self.act == Act.BIND:
            return f"BIND({self.param_name}={self.param_value})"
        if self.choice is not None:
            name = self.choice.name if hasattr(self.choice, 'name') else str(self.choice)
            return f"{self.act.name}({name})"
        return self.act.name


# ---------------------------------------------------------------------------
# Program: a sequence of actions
# ---------------------------------------------------------------------------

@dataclass
class Program:
    actions: list[Action] = field(default_factory=list)

    def append(self, action: Action):
        self.actions.append(action)

    def copy(self) -> Program:
        return Program(actions=list(self.actions))

    def __repr__(self):
        return " → ".join(str(a) for a in self.actions)


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------

def execute_program(prog: Program, inp: Grid, bg: int) -> Grid:
    """Execute a program on an input grid."""
    canvas = inp.copy()
    state = _ExecState(inp=inp, canvas=canvas, bg=bg)

    for action in prog.actions:
        _exec_action(state, action)

    return state.canvas


@dataclass
class _ExecState:
    inp: Grid
    canvas: Grid
    bg: int
    current_support: str = ""
    current_target: str = ""
    current_rewrite: str = ""
    params: dict = field(default_factory=dict)


def _exec_action(state: _ExecState, action: Action):
    if action.act == Act.SELECT_SUPPORT:
        state.current_support = action.choice.name if action.choice else ""
    elif action.act == Act.SELECT_TARGET:
        state.current_target = action.choice.name if action.choice else ""
    elif action.act == Act.REWRITE:
        state.current_rewrite = action.choice.name if action.choice else ""
    elif action.act == Act.BIND:
        state.params[action.param_name] = action.param_value
    elif action.act in (Act.NEXT, Act.STOP):
        # Execute the accumulated step
        _execute_step(state)
        state.current_support = ""
        state.current_target = ""
        state.current_rewrite = ""
        state.params = {}
    elif action.act == Act.SELECT_UNIT:
        pass  # just metadata


def _execute_step(state: _ExecState):
    """Execute one complete step (support + target + rewrite + params)."""
    rewrite = state.current_rewrite
    params = state.params
    inp = state.inp
    canvas = state.canvas
    bg = state.bg

    obj_pred = params.get("object_predicate")

    if rewrite == "FILL":
        color_src = params.get("color_source", "literal")
        color_val = params.get("color", 0)

        if state.current_target == "ENCLOSED_BG":
            _fill_enclosed(canvas, bg, color_val, color_src)
        elif state.current_target == "BY_COLOR":
            from_c = params.get("from_color", -1)
            if from_c >= 0:
                canvas[canvas == from_c] = color_val
        elif state.current_target == "ALL_CELLS":
            pass

    elif rewrite == "RECOLOR":
        from_c = params.get("from_color", -1)
        to_src = params.get("color_source", "literal")
        to_val = params.get("color", 0)

        if obj_pred:
            _recolor_by_predicate(canvas, bg, obj_pred, to_val, to_src)
        elif state.current_target == "BY_COLOR" and from_c >= 0:
            canvas[canvas == from_c] = to_val
        elif state.current_target == "SINGLETONS":
            _recolor_singletons(canvas, bg, from_c, to_val)
        elif state.current_target == "ADJACENT_TO":
            _recolor_adjacent_to_singleton(canvas, bg)

    elif rewrite == "DELETE":
        from_c = params.get("from_color", -1)
        if obj_pred:
            _delete_by_predicate(canvas, bg, obj_pred)
        elif state.current_target == "BY_COLOR" and from_c >= 0:
            canvas[canvas == from_c] = bg
        elif state.current_target == "SINGLETONS":
            _delete_singletons(canvas, bg, from_c)

    elif rewrite == "SYMMETRIZE":
        axis = params.get("axis", "h")
        _symmetrize_objects(canvas, bg, axis)

    elif rewrite == "PERIODIC_REPAIR":
        axis = params.get("axis", "row")
        repaired = _periodic_repair(canvas, axis)
        canvas[:] = repaired

    elif rewrite == "COPY_STAMP":
        stamp_color = params.get("stamp_color", -1)
        marker_color = params.get("marker_color", -1)
        if stamp_color >= 0 and marker_color >= 0:
            _copy_stamp(canvas, bg, stamp_color, marker_color)

    elif rewrite == "FILL_WITH_FRAME":
        _fill_enclosed(canvas, bg, 0, "from_context:frame")

    elif rewrite == "RECOLOR_TO_ADJ":
        obj_pred = params.get("object_predicate")
        if obj_pred:
            _recolor_by_predicate(canvas, bg, obj_pred, 0, "from_context:adjacent_singleton")
        else:
            _recolor_adjacent_to_singleton(canvas, bg)

    elif rewrite == "SWAP_COLORS":
        color_a = params.get("color_a", -1)
        color_b = params.get("color_b", -1)
        if color_a >= 0 and color_b >= 0:
            _swap_colors(canvas, color_a, color_b, params.get("object_predicate"))

    elif rewrite == "MOVE_TO_ENCLOSED":
        obj_pred = params.get("object_predicate")
        if obj_pred:
            _move_to_enclosed(canvas, bg, obj_pred)

    elif rewrite == "SWAP_ENCLOSED":
        _swap_enclosed_objects(canvas, bg)

    elif rewrite == "COLOR_MAP":
        _apply_per_demo_color_map(canvas, bg, params.get("color_map"))


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------

def _fill_enclosed(grid, bg, color_val, color_src):
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
        # Fill each enclosed region with its enclosing frame color
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


def _matches_predicate(obj, pred, bg, all_objs, grid):
    """Check if an object matches a predicate dict."""
    if "color" in pred and obj.color != pred["color"]:
        return False
    if "singleton" in pred:
        if pred["singleton"] and not obj.is_singleton:
            return False
        if not pred["singleton"] and obj.is_singleton:
            return False
    if "contained" in pred and pred["contained"]:
        # Check if this object's bbox is inside any larger object's bbox
        for other in all_objs:
            if other.oid == obj.oid:
                continue
            if (other.row <= obj.row and other.col <= obj.col and
                other.row + other.height >= obj.row + obj.height and
                other.col + other.width >= obj.col + obj.width and
                other.size > obj.size):
                return True
        return False
    return True


def _recolor_by_predicate(grid, bg, pred, to_val, to_src):
    """Recolor objects matching predicate."""
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    for obj in objs:
        if not _matches_predicate(obj, pred, bg, objs, grid):
            continue
        # Determine target color
        if to_src == "from_context:adjacent_singleton":
            singletons = {(o.row, o.col): o for o in objs if o.is_singleton and o.oid != obj.oid}
            adj_colors = set()
            for r in range(obj.height):
                for c in range(obj.width):
                    if not obj.mask[r, c]:
                        continue
                    gr, gc = obj.row + r, obj.col + c
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (gr + dr, gc + dc) in singletons:
                            adj_colors.add(singletons[(gr + dr, gc + dc)].color)
            if len(adj_colors) == 1:
                target_c = next(iter(adj_colors))
            else:
                continue
        else:
            target_c = to_val

        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    grid[obj.row + r, obj.col + c] = target_c


def _delete_by_predicate(grid, bg, pred):
    """Delete objects matching predicate."""
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    for obj in objs:
        if _matches_predicate(obj, pred, bg, objs, grid):
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.mask[r, c]:
                        grid[obj.row + r, obj.col + c] = bg


def _recolor_singletons(grid, bg, from_c, to_c):
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    for obj in objs:
        if obj.is_singleton and (from_c < 0 or obj.color == from_c):
            grid[obj.row, obj.col] = to_c


def _delete_singletons(grid, bg, from_c):
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    for obj in objs:
        if obj.is_singleton and (from_c < 0 or obj.color == from_c):
            grid[obj.row, obj.col] = bg


def _recolor_adjacent_to_singleton(grid, bg):
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    singletons = {(o.row, o.col): o for o in objs if o.is_singleton}
    for obj in objs:
        if obj.is_singleton:
            continue
        adj_colors = set()
        for r in range(obj.height):
            for c in range(obj.width):
                if not obj.mask[r, c]:
                    continue
                gr, gc = obj.row + r, obj.col + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (gr + dr, gc + dc) in singletons:
                        adj_colors.add(singletons[(gr + dr, gc + dc)].color)
        if len(adj_colors) == 1:
            tc = next(iter(adj_colors))
            if tc != obj.color:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            grid[obj.row + r, obj.col + c] = tc


def _symmetrize_objects(grid, bg, axis):
    rows, cols = grid.shape
    non_bg = grid != bg
    if not np.any(non_bg):
        return
    labeled, n = ndimage.label(non_bg, structure=np.ones((3, 3)))
    for lid in range(1, n + 1):
        comp = labeled == lid
        ys, xs = np.where(comp)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1
        sub = grid[r0:r0 + h, c0:c0 + w].copy()
        obj_vals = grid[comp]
        obj_color = int(Counter(obj_vals[obj_vals != bg].tolist()).most_common(1)[0][0]) if np.any(obj_vals != bg) else 0
        size = int(np.sum(comp))

        if axis == "h":
            asym = int(np.sum(sub != sub[:, ::-1]))
        else:
            asym = int(np.sum(sub != sub[::-1, :]))
        if asym == 0 or asym > max(2, size // 3):
            continue

        repaired = sub.copy()
        if axis == "h":
            for r in range(h):
                for c in range(w // 2):
                    mc = w - 1 - c
                    if repaired[r, c] != repaired[r, mc]:
                        # prefer non-bg (fill mode)
                        if repaired[r, c] != bg:
                            repaired[r, mc] = repaired[r, c]
                        else:
                            repaired[r, c] = repaired[r, mc]
            new_asym = int(np.sum(repaired != repaired[:, ::-1]))
        else:
            for r in range(h // 2):
                mr = h - 1 - r
                for c in range(w):
                    if repaired[r, c] != repaired[mr, c]:
                        if repaired[r, c] != bg:
                            repaired[mr, c] = repaired[r, c]
                        else:
                            repaired[r, c] = repaired[mr, c]
            new_asym = int(np.sum(repaired != repaired[::-1, :]))

        if new_asym == 0:
            grid[r0:r0 + h, c0:c0 + w] = repaired


def _periodic_repair(grid, axis):
    out = grid.copy()
    if axis == "row":
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
    # Check already periodic
    for p in range(1, n // 2 + 1):
        tile = line[:p]
        if all(np.array_equal(line[i:i + p], tile[:min(p, n - i)]) for i in range(p, n, p)):
            return line.copy()
    # Find best repair
    best = None
    best_v = n
    for p in range(1, n // 2 + 1):
        n_reps = n // p
        # Try each repetition as reference
        for ref in range(0, n - p + 1, p):
            tile = line[ref:ref + p]
            repaired = np.array([tile[i % p] for i in range(n)], dtype=line.dtype)
            v = int(np.sum(repaired != line))
            if 0 < v <= max(1, n_reps // 2) and v < best_v:
                best_v = v
                best = repaired
        # Also majority vote
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


def _copy_stamp(grid, bg, stamp_color, marker_color):
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    stamps = sorted([o for o in objs if o.color == stamp_color and not o.is_singleton], key=lambda o: -o.size)
    markers = [o for o in objs if o.color == marker_color and o.is_singleton]
    if not stamps or not markers:
        return
    stamp = stamps[0]
    sg = grid[stamp.row:stamp.row + stamp.height, stamp.col:stamp.col + stamp.width].copy()
    for m in markers:
        for r in range(stamp.height):
            for c in range(stamp.width):
                if sg[r, c] != bg:
                    tr, tc = m.row + r, m.col + c
                    if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]:
                        grid[tr, tc] = sg[r, c]


def _swap_colors(grid, color_a, color_b, obj_pred=None):
    """Swap two colors. If obj_pred given, only within matching objects."""
    from aria.guided.workspace import _extract_objects
    if obj_pred:
        objs = _extract_objects(grid, int(np.bincount(grid.flatten()).argmax()))
        for obj in objs:
            if _matches_predicate(obj, obj_pred, int(np.bincount(grid.flatten()).argmax()), objs, grid):
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            gr, gc = obj.row + r, obj.col + c
                            if grid[gr, gc] == color_a:
                                grid[gr, gc] = color_b
                            elif grid[gr, gc] == color_b:
                                grid[gr, gc] = color_a
    else:
        mask_a = grid == color_a
        mask_b = grid == color_b
        grid[mask_a] = color_b
        grid[mask_b] = color_a


def _move_to_enclosed(grid, bg, obj_pred):
    """Move matching objects into the nearest enclosed bg region.

    For each object matching the predicate:
    1. Find enclosed bg regions
    2. For each enclosed region, find the position closest to the object
    3. Move the object there (erase old, paint new)
    """
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)
    rows, cols = grid.shape

    # Find enclosed bg cells
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

    # Label enclosed regions
    labeled, n_regions = ndimage.label(enclosed, structure=np.ones((3, 3)))

    for obj in objs:
        if not _matches_predicate(obj, obj_pred, bg, objs, grid):
            continue
        if not obj.is_singleton:
            continue  # only move singletons for now

        # Find nearest enclosed region
        obj_r, obj_c = obj.row, obj.col
        best_r, best_c = -1, -1
        best_dist = float('inf')

        for lid in range(1, n_regions + 1):
            comp = labeled == lid
            ys, xs = np.where(comp)
            for yr, xc in zip(ys, xs):
                dist = abs(yr - obj_r) + abs(xc - obj_c)
                if dist < best_dist:
                    best_dist = dist
                    best_r, best_c = yr, xc

        if best_r >= 0:
            # Move: erase old, paint new
            grid[obj_r, obj_c] = bg
            grid[best_r, best_c] = obj.color
            # Mark this enclosed cell as used
            labeled[best_r, best_c] = 0


def _swap_enclosed_objects(grid, bg):
    """Swap colors between pairs of small objects enclosed within larger objects.

    For each large object that contains exactly 2 small enclosed sub-objects
    of different colors, swap those sub-objects' colors.
    """
    from aria.guided.workspace import _extract_objects
    objs = _extract_objects(grid, bg)

    # Find containment relationships
    for host in objs:
        enclosed = []
        for inner in objs:
            if inner.oid == host.oid:
                continue
            if (host.row <= inner.row and host.col <= inner.col and
                host.row + host.height >= inner.row + inner.height and
                host.col + host.width >= inner.col + inner.width and
                host.size > inner.size):
                enclosed.append(inner)

        if len(enclosed) == 2 and enclosed[0].color != enclosed[1].color:
            a, b = enclosed[0], enclosed[1]
            # Swap their colors in the grid
            for obj, new_color in [(a, b.color), (b, a.color)]:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            grid[obj.row + r, obj.col + c] = new_color
