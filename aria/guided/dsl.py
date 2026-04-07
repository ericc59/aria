"""Typed DSL for ARC program synthesis.

Primitives are composable operations over workspace entities.
The synthesizer searches for programs (compositions of primitives)
that reconstruct the output from input facts, verified across all demos.

The clause engine is one fast-path subset.

Primitive admission rules:
- Composable: can combine with many other things
- Parameterizable: takes typed arguments (object, region, color, etc.)
- Domain-general: applies across many tasks
- Cluster-justified: addresses a repeated capability gap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.guided.perceive import perceive, GridFacts, ObjFact
from aria.types import Grid


# ---------------------------------------------------------------------------
# Primitives: composable typed operations
# ---------------------------------------------------------------------------

def prim_select(facts: GridFacts, predicate) -> list[ObjFact]:
    """Select objects matching a predicate."""
    return [o for o in facts.objects
            if all(p.test(o, facts.objects) for p in predicate)]


def prim_crop_bbox(grid: Grid, obj: ObjFact) -> Grid:
    """Crop the grid to an object's bounding box."""
    return grid[obj.row:obj.row + obj.height,
                obj.col:obj.col + obj.width].copy()


def prim_crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Crop a rectangular region from the grid."""
    return grid[r0:r1, c0:c1].copy()


def prim_find_frame(obj: ObjFact, grid: Grid) -> tuple[int, int, int, int] | None:
    """Find the rectangular frame within an object (top/bottom/left/right walls).

    Returns (r0, c0, r1, c1) of the frame corners, or None.
    """
    color = obj.color
    sub = grid[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width]
    h, w = sub.shape

    # Find rows and cols with contiguous spans of the frame color
    def _spans(line, val):
        spans = []
        i = 0
        while i < len(line):
            if line[i] == val:
                start = i
                while i < len(line) and line[i] == val:
                    i += 1
                spans.append((start, i - 1))
            else:
                i += 1
        return spans

    # Find the best rectangular frame
    best = None
    # Candidate top/bottom rows: rows with long spans of the color
    row_spans = [(r, _spans(sub[r], color)) for r in range(h)]
    col_spans = [(c, _spans(sub[:, c], color)) for c in range(w)]

    for r_top in range(h):
        for sp_top in row_spans[r_top][1]:
            c_left, c_right = sp_top
            for r_bot in range(r_top + 2, h):
                for sp_bot in row_spans[r_bot][1]:
                    if sp_bot[0] > c_left or sp_bot[1] < c_right:
                        continue
                    # Check left and right columns span from r_top to r_bot
                    left_ok = all(sub[r, c_left] == color for r in range(r_top, r_bot + 1))
                    right_ok = all(sub[r, c_right] == color for r in range(r_top, r_bot + 1))
                    if left_ok and right_ok:
                        area = (r_bot - r_top) * (c_right - c_left)
                        if best is None or area > best[4]:
                            best = (obj.row + r_top, obj.col + c_left,
                                    obj.row + r_bot, obj.col + c_right, area)

    return best[:4] if best else None


def prim_crop_interior(grid: Grid, frame: tuple[int, int, int, int]) -> Grid:
    """Crop the interior of a rectangular frame (1px border)."""
    r0, c0, r1, c1 = frame
    return grid[r0 + 1:r1, c0 + 1:c1].copy()


def prim_split_at_separator(grid: Grid, facts: GridFacts, sep_idx: int = 0
                             ) -> tuple[Grid, Grid] | None:
    """Split grid at a separator into two regions."""
    if sep_idx >= len(facts.separators):
        return None
    sep = facts.separators[sep_idx]
    if sep.axis == 'col':
        sc = sep.index
        left = grid[:, :sc]
        right = grid[:, sc + 1:]
        # Equalize widths
        min_w = min(left.shape[1], right.shape[1])
        return left[:, :min_w], right[:, :min_w]
    elif sep.axis == 'row':
        sr = sep.index
        top = grid[:sr, :]
        bottom = grid[sr + 1:, :]
        min_h = min(top.shape[0], bottom.shape[0])
        return top[:min_h, :], bottom[:min_h, :]
    return None


def prim_combine(region_a: Grid, region_b: Grid, op: str, bg: int) -> np.ndarray:
    """Binary operation on two regions. Returns boolean mask."""
    a = (region_a != bg)
    b = (region_b != bg)
    if op == 'and':
        return a & b
    elif op == 'or':
        return a | b
    elif op == 'xor':
        return a ^ b
    elif op == 'diff':
        return a & ~b
    elif op == 'rdiff':
        return ~a & b
    return a


# ---------------------------------------------------------------------------
# Core trace primitive — all geometry ops are built on this
# ---------------------------------------------------------------------------

_DIR_MAP = {
    'up': [(-1, 0)],
    'down': [(1, 0)],
    'up_left': [(-1, -1)],
    'up_right': [(-1, 1)],
    'down_left': [(1, -1)],
    'down_right': [(1, 1)],
    'left': [(0, -1)],
    'right': [(0, 1)],
    'cardinal': [(-1, 0), (1, 0), (0, -1), (0, 1)],
    'diagonal': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    'all': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
}


def _ray_cells(row, col, dr, dc, max_steps, shape):
    """Cell sequence for a ray from (row,col) in direction (dr,dc)."""
    cells = []
    r, c = row + dr, col + dc
    for _ in range(max_steps):
        if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _line_cells(r0, c0, r1, c1):
    """Cell sequence for a straight line from (r0,c0) to (r1,c1), inclusive."""
    cells = []
    dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        r += dr
        c += dc
        if len(cells) > 2000:
            break
    return cells


def _path_cells(r0, c0, r1, c1, order='hv'):
    """Cell sequence for an L-shaped manhattan path (excludes start)."""
    cells = []
    if order == 'hv':
        dc = 1 if c1 >= c0 else -1
        c = c0
        while c != c1:
            c += dc
            cells.append((r0, c))
        dr = 1 if r1 >= r0 else -1
        r = r0
        while r != r1:
            r += dr
            cells.append((r, c1))
    else:  # vh
        dr = 1 if r1 >= r0 else -1
        r = r0
        while r != r1:
            r += dr
            cells.append((r, c0))
        dc = 1 if c1 >= c0 else -1
        c = c0
        while c != c1:
            c += dc
            cells.append((r1, c))
    return cells


def prim_trace(grid: Grid, cells, color: int, bg: int,
               write: str = 'bg_only', stop_colors: set = None) -> Grid:
    """Core geometry primitive: write color along a cell sequence.

    write='bg_only':  write only bg cells, stop at first non-bg
    write='skip':     write only bg cells, continue past non-bg
    write='overwrite': write all cells
    stop_colors: if set, stop when hitting any of these (overrides write-mode stop)
    """
    result = grid.copy()
    for r, c in cells:
        if r < 0 or r >= result.shape[0] or c < 0 or c >= result.shape[1]:
            break
        cell = result[r, c]
        if stop_colors and cell in stop_colors:
            break
        if write == 'bg_only':
            if cell == bg:
                result[r, c] = color
            else:
                break
        elif write == 'skip':
            if cell == bg:
                result[r, c] = color
        elif write == 'overwrite':
            result[r, c] = color
    return result


# ---------------------------------------------------------------------------
# Convenience wrappers over prim_trace
# ---------------------------------------------------------------------------

def prim_cast_ray(grid: Grid, row: int, col: int, dr: int, dc: int,
                   color: int, distance: int, bg: int,
                   collision: str = 'stop') -> Grid:
    """Cast a ray from (row,col) in direction (dr,dc)."""
    write = {'stop': 'bg_only', 'skip': 'skip', 'overwrite': 'overwrite'}[collision]
    cells = _ray_cells(row, col, dr, dc, distance, grid.shape)
    return prim_trace(grid, cells, color, bg, write=write)


def prim_cast_rays(grid: Grid, row: int, col: int, color: int,
                    directions: str, distance: int, bg: int,
                    collision: str = 'stop') -> Grid:
    """Cast rays from (row,col) in multiple directions."""
    result = grid
    for dr, dc in _DIR_MAP.get(directions, []):
        result = prim_cast_ray(result, row, col, dr, dc, color, distance, bg,
                                collision=collision)
    return result


def prim_draw_line(grid: Grid, r0: int, c0: int, r1: int, c1: int,
                    color: int, bg: int) -> Grid:
    """Draw a straight line between two points (writes bg cells, skips non-bg)."""
    return prim_trace(grid, _line_cells(r0, c0, r1, c1), color, bg, write='skip')


def prim_fill_toward(grid: Grid, row: int, col: int, dr: int, dc: int,
                      color: int, bg: int, stop_colors: set = None) -> Grid:
    """Extend color in direction until hitting stop or border."""
    cells = _ray_cells(row, col, dr, dc, max(grid.shape), grid.shape)
    if stop_colors:
        return prim_trace(grid, cells, color, bg, write='overwrite', stop_colors=stop_colors)
    return prim_trace(grid, cells, color, bg, write='bg_only')


def prim_draw_path(grid: Grid, r0: int, c0: int, r1: int, c1: int,
                    color: int, bg: int, order: str = 'hv') -> Grid:
    """Draw an L-shaped manhattan path between two points (writes bg cells, skips non-bg)."""
    return prim_trace(grid, _path_cells(r0, c0, r1, c1, order), color, bg, write='skip')


def prim_mirror_across(grid: Grid, axis: str, index: int) -> Grid:
    """Mirror one half of the grid across a separator axis.

    Tries: copy left→right, right→left, top→bottom, bottom→top.
    Returns all 4 variants so the synthesizer can test which one matches.
    """
    result = grid.copy()
    rows, cols = grid.shape
    if axis == 'col':
        # Mirror left half to right
        left = grid[:, :index]
        right_w = cols - index - 1
        if left.shape[1] >= right_w and right_w > 0:
            result[:, index + 1:] = left[:, :right_w][:, ::-1]
    elif axis == 'col_rtl':
        # Mirror right half to left
        right = grid[:, index + 1:]
        left_w = index
        if right.shape[1] >= left_w and left_w > 0:
            result[:, :index] = right[:, :left_w][:, ::-1]
    elif axis == 'row':
        # Mirror top half to bottom
        top = grid[:index, :]
        bottom_h = rows - index - 1
        if top.shape[0] >= bottom_h and bottom_h > 0:
            result[index + 1:, :] = top[:bottom_h, :][::-1, :]
    elif axis == 'row_btt':
        # Mirror bottom half to top
        bottom = grid[index + 1:, :]
        top_h = index
        if bottom.shape[0] >= top_h and top_h > 0:
            result[:index, :] = bottom[:top_h, :][::-1, :]
    return result


def prim_repair_region(region: Grid, threshold: float = 0.7) -> Grid:
    """Find the smallest repeating tile in a region and fix defective cells.

    Works on any size region. Finds the NxM tile with the highest match
    rate via majority vote, then replaces the entire region with the tiled
    result.
    """
    from collections import Counter
    rows, cols = region.shape
    if rows == 0 or cols == 0:
        return region

    best_result = None
    best_score = (-float('inf'), -float('inf'))

    for rp in range(1, rows + 1):
        for cp in range(1, cols + 1):
            if rp == rows and cp == cols:
                continue  # trivial: tile = whole region
            # Majority vote each tile position
            # Global color frequency for tie-breaking
            global_counts = Counter(int(v) for v in region.flat)
            tile = np.zeros((rp, cp), dtype=region.dtype)
            valid_tile = True
            for tr in range(rp):
                for tc in range(cp):
                    vals = [int(region[r, c])
                            for r in range(tr, rows, rp)
                            for c in range(tc, cols, cp)]
                    if len(vals) < 3:
                        valid_tile = False
                        break
                    vc = Counter(vals)
                    top_count = vc.most_common(1)[0][1]
                    if top_count < 2:
                        valid_tile = False
                        break
                    candidates = [v for v, c in vc.items() if c == top_count]
                    if len(candidates) > 1:
                        tile[tr, tc] = min(candidates,
                                           key=lambda v: -global_counts.get(v, 0))
                    else:
                        tile[tr, tc] = candidates[0]
                if not valid_tile:
                    break
            if not valid_tile:
                continue

            # Count changes needed
            changes = sum(1 for r in range(rows) for c in range(cols)
                          if region[r, c] != tile[r % rp, c % cp])

            if changes == 0:
                continue  # already perfect at this period
            rate = 1.0 - changes / (rows * cols)
            if rate < threshold:
                continue
            # Must repeat at least 2x (partial repetition at edges is ok)
            n_repeats = (rows * cols) / (rp * cp)
            if n_repeats < 2:
                continue
            # Highest rate wins among tiles that repeat ≥ 2x
            # Tie-break: most repetitions (simplest explanation)
            score = (rate, n_repeats)
            if score > best_score:
                best_score = score
                result = np.zeros_like(region)
                for r in range(rows):
                    for c in range(cols):
                        result[r, c] = tile[r % rp, c % cp]
                best_result = result
            if score > best_score:
                best_score = score
                result = np.zeros_like(region)
                for r in range(rows):
                    for c in range(cols):
                        result[r, c] = tile[r % rp, c % cp]
                best_result = result

    return best_result


def _repair_2d_periodic(grid, threshold=0.7):
    """Repair by detecting a 2D repeating tile and majority-voting each cell.

    Finds the period with the HIGHEST match rate (fewest changes needed).
    """
    from collections import Counter
    rows, cols = grid.shape

    best_result = None
    best_rate = 0
    best_changes = rows * cols

    max_rp = max(rows // 2, 1)
    max_cp = max(cols // 2, 1)
    for rp in range(1, max_rp + 1):
        for cp in range(1, max_cp + 1):
            tile = np.zeros((rp, cp), dtype=grid.dtype)
            for tr in range(rp):
                for tc in range(cp):
                    vals = [int(grid[r, c])
                            for r in range(tr, rows, rp)
                            for c in range(tc, cols, cp)]
                    tile[tr, tc] = Counter(vals).most_common(1)[0][0]

            matches = sum(1 for r in range(rows) for c in range(cols)
                          if grid[r, c] == tile[r % rp, c % cp])
            total = rows * cols
            changes = total - matches

            if matches == total:
                continue  # already perfect for this period
            rate = matches / total
            if rate >= threshold and changes < best_changes:
                best_changes = changes
                best_rate = rate
                result = np.zeros_like(grid)
                for r in range(rows):
                    for c in range(cols):
                        result[r, c] = tile[r % rp, c % cp]
                best_result = result

    return best_result


def _repair_line_periodic(line, bg, threshold=0.7):
    """Repair a 1D line by majority-vote period detection.

    Automatically trims border cells (matching the first/last cell if they're
    the same and appear to be a frame), repairs the interior, then reassembles.
    """
    from collections import Counter
    n = len(line)
    vals = set(int(v) for v in line)
    if len(vals) <= 1:
        return None

    # Trim border cells: strip matching first/last colors, up to 2 levels.
    # Only trim if border color differs from adjacent interior AND doesn't
    # appear frequently in the interior (i.e., it's a frame, not part of pattern).
    trim_start = 0
    trim_end = n
    for _ in range(2):
        if trim_end - trim_start < 4:
            break
        seg = line[trim_start:trim_end]
        border_color = int(seg[0])
        if border_color != int(seg[-1]):
            break  # first and last don't match — no frame
        if border_color == int(seg[1]):
            break  # border same as interior — it's pattern, not frame
        # Check: border color should be rare in the interior
        interior_seg = seg[1:-1]
        border_in_interior = np.sum(interior_seg == border_color)
        if border_in_interior > len(interior_seg) * 0.3:
            break  # border color is common in interior — it's pattern
        trim_start += 1
        trim_end -= 1

    # Only repair the trimmed interior — don't touch border cells
    segment = line[trim_start:trim_end]
    seg_n = len(segment)
    if seg_n < 2:
        return None
    seg_vals = set(int(v) for v in segment)
    if len(seg_vals) <= 1:
        return None

    # Find the period with the highest match rate
    best_p = None
    best_tile = None
    best_rate = 0

    for p in range(2, seg_n // 2 + 1):
        tile = []
        for phase in range(p):
            phase_vals = [int(segment[i]) for i in range(phase, seg_n, p)]
            tile.append(Counter(phase_vals).most_common(1)[0][0])

        matches = sum(1 for i in range(seg_n) if int(segment[i]) == tile[i % p])
        rate = matches / seg_n
        if matches == seg_n:
            return None  # already perfect
        if rate > best_rate:
            best_rate = rate
            best_p = p
            best_tile = tile

    if best_rate >= threshold and best_tile is not None:
        result = line.copy()
        for i in range(seg_n):
            result[trim_start + i] = best_tile[i % best_p]
        return result

    return None


def prim_replace_interior(grid: Grid, obj: 'ObjFact', new_interior: Grid) -> Grid:
    """Replace the interior of an object's frame with new content.

    Keeps the frame pixels, swaps the inside.
    """
    frame = prim_find_frame(obj, grid)
    if frame is None:
        return grid
    r0, c0, r1, c1 = frame
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    if new_interior.shape != (ih, iw):
        return grid
    result = grid.copy()
    result[r0 + 1:r1, c0 + 1:c1] = new_interior
    return result


def prim_render_mask(mask: np.ndarray, color: int, bg: int) -> Grid:
    """Render a boolean mask as a grid with given color."""
    result = np.full(mask.shape, bg, dtype=np.uint8)
    result[mask] = color
    return result


# ---------------------------------------------------------------------------
# Program representation
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """A DSL program."""
    steps: list = field(default_factory=list)
    description: str = ""
    _execute_fn: Any = field(default=None, repr=False)

    def execute(self, inp: Grid) -> Grid:
        if self._execute_fn:
            return self._execute_fn(inp)
        return inp


def _make_program(fn, desc) -> Program:
    """Create a program from an execute function."""
    return Program(steps=[], description=desc, _execute_fn=fn)


# ---------------------------------------------------------------------------
# Synthesizer: search over primitive compositions
# ---------------------------------------------------------------------------

def synthesize_program(
    demos: list[tuple[Grid, Grid]],
) -> Program | None:
    """Synthesize a program from demo pairs.

    Phase 1: clause engine (fast path for known clause patterns)
    Phase 2: typed bottom-up synthesis over DSL primitives
    """
    from aria.guided.synthesize import synthesize
    return synthesize(demos)


def _verify(prog: Program, demos: list[tuple[Grid, Grid]]) -> bool:
    for inp, out in demos:
        try:
            if not np.array_equal(prog.execute(inp), out):
                return False
        except Exception:
            return False
    return True


