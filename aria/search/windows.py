"""Extraction and lightweight role classification for bar-window scenes.

These tasks are not ordinary rectangular frames. They often consist of a
uniform straight bar attached to a filled rectangular window interior. The bar
acts like an anchor/role marker, while the interior carries the content that
later gets transferred or rewritten.

This module keeps that representation explicit at derive time so tasks do not
have to rediscover it from raw connected components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import label


WindowSide = Literal["top", "bottom", "left", "right"]


@dataclass(frozen=True)
class BarWindow:
    """A rectangular component with a uniform straight anchor bar on one side."""

    bbox: tuple[int, int, int, int]
    side: WindowSide
    bar_color: int
    bar_thickness: int
    full_grid: np.ndarray
    interior_grid: np.ndarray
    content_bbox: tuple[int, int, int, int] | None
    content_colors: frozenset[int]

    @property
    def height(self) -> int:
        return self.full_grid.shape[0]

    @property
    def width(self) -> int:
        return self.full_grid.shape[1]

    @property
    def content_area(self) -> int:
        if self.content_bbox is None:
            return 0
        r0, c0, r1, c1 = self.content_bbox
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    @property
    def is_empty(self) -> bool:
        return self.content_bbox is None

    @property
    def is_mixed(self) -> bool:
        return len(self.content_colors) >= 2


def _dominant_border_color(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0] if grid.shape[0] > 2 else np.array([], dtype=grid.dtype),
            grid[1:-1, -1] if grid.shape[0] > 2 else np.array([], dtype=grid.dtype),
        ]
    )
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _candidate_bgs(grid: np.ndarray, preferred_bg: int | None) -> list[int]:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0] if grid.shape[0] > 2 else np.array([], dtype=grid.dtype),
            grid[1:-1, -1] if grid.shape[0] > 2 else np.array([], dtype=grid.dtype),
        ]
    )
    border_vals, border_counts = np.unique(border, return_counts=True)
    global_vals, global_counts = np.unique(grid, return_counts=True)

    ordered: list[int] = []
    if preferred_bg is not None:
        ordered.append(int(preferred_bg))
    ordered.extend(int(v) for v in border_vals[np.argsort(-border_counts)])
    ordered.extend(int(v) for v in global_vals[np.argsort(-global_counts)])

    deduped: list[int] = []
    seen: set[int] = set()
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped[:6]


def _side_candidate(
    sub: np.ndarray,
    side: WindowSide,
    thickness: int,
    bg: int,
) -> tuple[int, int, int] | None:
    """Return (score, bar_color, thickness) when a side is a valid uniform bar."""
    h, w = sub.shape
    if side == "top":
        strip = sub[:thickness, :]
    elif side == "bottom":
        strip = sub[h - thickness :, :]
    elif side == "left":
        strip = sub[:, :thickness]
    else:
        strip = sub[:, w - thickness :]

    colors = np.unique(strip)
    if len(colors) != 1:
        return None
    bar_color = int(colors[0])
    if bar_color in (0, bg):
        return None
    extent = w if side in ("top", "bottom") else h
    return (thickness * extent, bar_color, thickness)


def _strip_interior(sub: np.ndarray, side: WindowSide, thickness: int) -> np.ndarray:
    if side == "top":
        return sub[thickness:, :]
    if side == "bottom":
        return sub[:-thickness, :]
    if side == "left":
        return sub[:, thickness:]
    return sub[:, :-thickness]


def _content_bbox(interior: np.ndarray, bg: int, bar_color: int) -> tuple[int, int, int, int] | None:
    mask = (interior != 0) & (interior != bg) & (interior != bar_color)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def extract_bar_windows(
    grid: np.ndarray,
    bg: int | None = None,
    *,
    max_bar_thickness: int = 3,
    min_side_extent: int = 3,
) -> list[BarWindow]:
    """Extract rectangular bar-window entities from a grid.

    A valid bar-window:
    - is a 4-connected non-bg component
    - fills its bounding box completely with non-bg cells
    - has a uniform strip on one side that acts as the anchor bar
    """
    if bg is None:
        bg = _dominant_border_color(grid)

    mask = grid != bg
    labels, n = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))

    windows: list[BarWindow] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        r0, c0, r1, c1 = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        h, w = sub.shape
        if max(h, w) < min_side_extent:
            continue

        best: tuple[int, WindowSide, int, int] | None = None
        for side in ("top", "left", "right", "bottom"):
            limit = min(max_bar_thickness, h if side in ("top", "bottom") else w)
            for thickness in range(1, limit + 1):
                cand = _side_candidate(sub, side, thickness, bg)
                if cand is None:
                    continue
                score, bar_color, thickness = cand
                interior = _strip_interior(sub, side, thickness)
                if interior.size == 0:
                    score += 1
                elif interior.shape[0] < 1 or interior.shape[1] < 1:
                    continue
                choice = (score, side, bar_color, thickness)
                if best is None or choice[0] > best[0]:
                    best = choice

        if best is None:
            continue

        _, side, bar_color, thickness = best
        interior = _strip_interior(sub, side, thickness)
        content_box = _content_bbox(interior, bg, bar_color)
        content_colors = frozenset(
            int(v)
            for v in np.unique(interior)
            if v not in (0, bg, bar_color)
        )
        windows.append(
            BarWindow(
                bbox=(r0, c0, r1, c1),
                side=side,
                bar_color=bar_color,
                bar_thickness=thickness,
                full_grid=sub,
                interior_grid=interior,
                content_bbox=content_box,
                content_colors=content_colors,
            )
        )

    windows.sort(key=lambda w: (w.bbox[0], w.bbox[1], w.height, w.width))
    return windows


def extract_bar_windows_best(
    grid: np.ndarray,
    preferred_bg: int | None = None,
    *,
    max_bar_thickness: int = 3,
    min_side_extent: int = 3,
) -> list[BarWindow]:
    """Try several plausible backgrounds and keep the cleanest window set."""
    best: list[BarWindow] = []
    best_key = (-1, -1)
    for bg in _candidate_bgs(grid, preferred_bg):
        windows = extract_bar_windows(
            grid,
            bg=bg,
            max_bar_thickness=max_bar_thickness,
            min_side_extent=min_side_extent,
        )
        key = (len(windows), sum(w.height * w.width for w in windows))
        if key > best_key:
            best = windows
            best_key = key
    return best


def classify_bar_window_roles(windows: list[BarWindow]) -> list[tuple[int, str, float, str]]:
    """Assign conservative scene roles for extracted bar windows.

    This is intentionally weak:
    - mixed-content windows tend to be sources/prototypes
    - empty windows tend to be targets/anchors
    - single-color content windows are workspaces/fillers
    """
    if len(windows) < 2:
        return []

    roles: list[tuple[int, str, float, str]] = []
    mixed = [i for i, w in enumerate(windows) if w.is_mixed]
    empty = [i for i, w in enumerate(windows) if w.is_empty]
    filled = [i for i, w in enumerate(windows) if not w.is_empty]

    for i in mixed:
        roles.append((i, "SOURCE", 0.55, "bar-window with mixed interior content"))

    for i in empty:
        roles.append((i, "TARGET", 0.45, "empty bar-window"))

    assigned = {idx for idx, _, _, _ in roles}
    for i in filled:
        if i in assigned:
            continue
        roles.append((i, "WORKSPACE", 0.35, "filled bar-window"))

    return roles


# ---------------------------------------------------------------------------
# Progressive fill
# ---------------------------------------------------------------------------

def _fill_corner(side: WindowSide, rows: int, cols: int) -> tuple[int, int]:
    """Manhattan staircase corner for a given bar side.

    The corner is on the bar-adjacent edge, 90deg counter-clockwise from bar
    outward direction:
        left  -> bottom-left,  right -> top-right,
        top   -> top-left,     bottom -> bottom-right
    """
    if side == "left":
        return (rows - 1, 0)
    if side == "right":
        return (0, cols - 1)
    if side == "top":
        return (0, 0)
    return (rows - 1, cols - 1)  # bottom


def _staircase_extent(N: int, idx: int, corner_idx: int, max_ext: int) -> int:
    return min(max(0, N - abs(idx - corner_idx)), max_ext)


def _stack_and_fill_dims(
    side: WindowSide, rows: int, cols: int
) -> tuple[int, int, int]:
    """Return (stack_dim, fill_dim, corner_index_along_stack)."""
    cr, cc = _fill_corner(side, rows, cols)
    if side in ("left", "right"):
        return rows, cols, cr
    return cols, rows, cc


def get_fill_extents(content: np.ndarray, side: WindowSide) -> list[int]:
    """Per-stack-slot extent of color-7 cells from the bar side."""
    rows, cols = content.shape
    if side in ("left", "right"):
        extents = []
        for r in range(rows):
            if side == "left":
                ext = 0
                for c in range(cols):
                    if content[r, c] == 7:
                        ext = c + 1
                    else:
                        break
            else:
                ext = 0
                for c in range(cols - 1, -1, -1):
                    if content[r, c] == 7:
                        ext = cols - c
                    else:
                        break
            extents.append(ext)
        return extents
    # top / bottom
    extents = []
    for c in range(cols):
        if side == "top":
            ext = 0
            for r in range(rows):
                if content[r, c] == 7:
                    ext = r + 1
                else:
                    break
        else:
            ext = 0
            for r in range(rows - 1, -1, -1):
                if content[r, c] == 7:
                    ext = rows - r
                else:
                    break
        extents.append(ext)
    return extents


def _compute_fill_K(
    content: np.ndarray,
    side: WindowSide,
    N_source: int,
    is_transfer: bool,
) -> int:
    """Determine the fill step count K.

    K=1 for in-place.  For transfer: K=2 if 7-cells span >=2 positions
    along the stack axis, else K = fill_dim - N_source (fills to edge).
    """
    if not is_transfer:
        return 1
    rows, cols = content.shape
    sdim, fdim, _ = _stack_and_fill_dims(side, rows, cols)

    # Count distinct stack positions containing 7
    if side in ("left", "right"):
        n_active = sum(1 for r in range(rows) if 7 in content[r])
    else:
        n_active = sum(1 for c in range(cols) if 7 in content[:, c])

    if n_active >= 2:
        return 2
    return max(1, fdim - N_source)


def _edge_correction(extents: list[int], idx: int) -> int:
    """Return 1 if *idx* is at the interior edge of a flat run before a drop."""
    n = len(extents)
    if idx <= 0 or idx >= n - 1:
        return 0
    if extents[idx] == extents[idx - 1] and extents[idx] > extents[idx + 1]:
        return 1
    return 0


def progressive_fill(
    content: np.ndarray,
    side: WindowSide,
    N_out: int,
    *,
    source_extents: list[int] | None = None,
    N_source: int | None = None,
    source_side: WindowSide | None = None,
    source_shape: tuple[int, int] | None = None,
    fill_color: int = 7,
    base_color: int = 5,
    K: int | None = None,
) -> np.ndarray:
    """Apply progressive fill to a content grid.

    Two modes depending on whether source metadata enables the ``source+K``
    model or falls back to the Manhattan staircase:

    1. **source+K** (same bar-side + content-shape): each slot's extent =
       ``source_extent + K - edge_correction``, capped at fill_dim.
    2. **staircase** (different shape / fresh workspace): pure Manhattan
       staircase from the bar-corner with ``N_out`` steps.
    """
    rows, cols = content.shape
    sdim, fdim, corner_idx = _stack_and_fill_dims(side, rows, cols)

    use_source_plus_k = (
        source_extents is not None
        and K is not None
        and side == source_side
        and content.shape == source_shape
    )

    result = content.copy()
    for i in range(sdim):
        if use_source_plus_k and i < len(source_extents):
            ext = source_extents[i] + K - _edge_correction(source_extents, i)
            base = min(ext, fdim)
        else:
            base = _staircase_extent(N_out, i, corner_idx, fdim)

        if base <= 0:
            continue

        if side in ("left", "right"):
            if side == "left":
                for c in range(min(base, cols)):
                    if result[i, c] == base_color:
                        result[i, c] = fill_color
            else:
                for c in range(cols - 1, max(cols - 1 - base, -1), -1):
                    if result[i, c] == base_color:
                        result[i, c] = fill_color
        else:  # top / bottom
            if side == "top":
                for r in range(min(base, rows)):
                    if result[r, i] == base_color:
                        result[r, i] = fill_color
            else:
                for r in range(rows - 1, max(rows - 1 - base, -1), -1):
                    if result[r, i] == base_color:
                        result[r, i] = fill_color
    return result


# ---------------------------------------------------------------------------
# Window-pair transfer
# ---------------------------------------------------------------------------

def _get_content(interior: np.ndarray) -> np.ndarray:
    """Strip 0-border from an interior grid to get the content area."""
    h, w = interior.shape
    if h <= 2 or w <= 2:
        return interior
    return interior[1:-1, 1:-1]


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _row_band(bbox: tuple) -> tuple[int, int]:
    return (bbox[0], bbox[2])


def _col_band(bbox: tuple) -> tuple[int, int]:
    return (bbox[1], bbox[3])


def _bands_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _build_window_at_target(
    bg: int,
    bar_color: int,
    target_bbox: tuple,
    target_side: WindowSide,
    content_rows: int,
    content_cols: int,
    filled_content: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    """Build a full window grid (bar + border + content) at the target position.

    Returns (window_grid, placement_bbox).
    """
    # Content wrapped in 0-border
    bordered_h = content_rows + 2
    bordered_w = content_cols + 2
    bordered = np.zeros((bordered_h, bordered_w), dtype=filled_content.dtype)
    bordered[1:-1, 1:-1] = filled_content

    # Add bar
    if target_side in ("left", "right"):
        full_h = bordered_h
        full_w = bordered_w + 1  # 1 col for bar
        full = np.full((full_h, full_w), bg, dtype=filled_content.dtype)
        if target_side == "left":
            full[:, 0] = bar_color
            full[:, 1:] = bordered
        else:
            full[:, -1] = bar_color
            full[:, :-1] = bordered
    else:  # top / bottom
        full_h = bordered_h + 1
        full_w = bordered_w
        full = np.full((full_h, full_w), bg, dtype=filled_content.dtype)
        if target_side == "top":
            full[0, :] = bar_color
            full[1:, :] = bordered
        else:
            full[-1, :] = bar_color
            full[:-1, :] = bordered

    # Placement: anchor the bar at the target's bar position
    tr0, tc0, tr1, tc1 = target_bbox
    if target_side == "left":
        pr0, pc0 = tr0, tc0
    elif target_side == "right":
        pr0 = tr0
        pc0 = tc1 - full_w + 1
    elif target_side == "top":
        pr0, pc0 = tr0, tc0
    else:
        pr0 = tr1 - full_h + 1
        pc0 = tc0

    return full, (pr0, pc0, pr0 + full_h - 1, pc0 + full_w - 1)


def _place_window(
    result: np.ndarray,
    bg: int,
    bar_color: int,
    side: WindowSide,
    interior: np.ndarray,
    bar_row_or_col: int,
) -> None:
    """Stamp a 1-thick bar + interior into *result* in-place.

    *bar_row_or_col* is the absolute row (top/bottom) or column (left/right)
    where the single-row/col bar goes.
    """
    ih, iw = interior.shape
    if side == "top":
        result[bar_row_or_col, : iw] = bar_color  # won't work if col offset needed
    # We need column offset too; generalise below.


def _rebuild_window(
    result: np.ndarray,
    bg: int,
    bar_color: int,
    side: WindowSide,
    content: np.ndarray,
    anchor_r: int,
    anchor_c: int,
) -> None:
    """Draw bar(1) + 0-border + content into *result* at anchor position.

    Anchor is the top-left of the bar row/col.  For bar=top the bar is at
    row anchor_r; for bar=left it's at col anchor_c.
    """
    cr, cc = content.shape
    # Build interior (0-border + content)
    interior = np.zeros((cr + 2, cc + 2), dtype=result.dtype)
    interior[1:-1, 1:-1] = content
    ih, iw = interior.shape

    H, W = result.shape
    if side == "top":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + 1 + ih)
        c_end = min(W, c0 + iw)
        result[r0, c0:c_end] = bar_color
        h_avail = r_end - (r0 + 1)
        w_avail = c_end - c0
        if h_avail > 0 and w_avail > 0:
            result[r0 + 1:r0 + 1 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
    elif side == "bottom":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih + 1)
        c_end = min(W, c0 + iw)
        h_avail = min(ih, r_end - r0 - 1)
        w_avail = c_end - c0
        if h_avail > 0 and w_avail > 0:
            result[r0:r0 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
        bar_r = r0 + h_avail
        if bar_r < H:
            result[bar_r, c0:c_end] = bar_color
    elif side == "left":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih)
        c_end = min(W, c0 + 1 + iw)
        h_avail = r_end - r0
        w_avail = c_end - (c0 + 1)
        result[r0:r_end, c0] = bar_color
        if h_avail > 0 and w_avail > 0:
            result[r0:r0 + h_avail, c0 + 1:c0 + 1 + w_avail] = interior[:h_avail, :w_avail]
    else:  # right
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih)
        c_end = min(W, c0 + iw + 1)
        h_avail = r_end - r0
        w_avail = min(iw, c_end - c0 - 1)
        if h_avail > 0 and w_avail > 0:
            result[r0:r0 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
        bar_c = c0 + w_avail
        if bar_c < W:
            result[r0:r_end, bar_c] = bar_color


def _source_params(source: BarWindow) -> tuple[np.ndarray, list[int], int]:
    """Extract (content, fill_extents, N_source) for a source window."""
    content = _get_content(source.interior_grid)
    ext = get_fill_extents(content, source.side)
    _, _, ci = _stack_and_fill_dims(source.side, *content.shape)
    N = ext[ci]
    return content, ext, N


def solve_window_pair_transfer(
    grid: np.ndarray,
    bg: int = 6,
) -> np.ndarray | None:
    """Solve a bar-window progressive-fill transfer task.

    Detects source (mixed 5+7 content), empty targets, and all-5 workspaces.
    Applies progressive fill and spatial transfer.  Returns the solved output
    grid or None if the task doesn't match this pattern.
    """
    windows = extract_bar_windows(grid, bg=bg)
    if len(windows) < 2:
        return None

    sources = [w for w in windows if w.is_mixed and 7 in w.content_colors and 5 in w.content_colors]
    targets = [w for w in windows if w.is_empty]

    if not sources:
        return None

    bar_color = sources[0].bar_color
    is_transfer = len(targets) > 0
    result = grid.copy()

    if is_transfer:
        return _solve_transfer(result, bg, bar_color, windows, sources, targets)

    # In-place: use first source
    source = sources[0]
    s_content, s_ext, N_src = _source_params(source)
    return _solve_inplace(result, bg, bar_color, windows, source,
                          s_content, s_ext, N_src)


def _solve_inplace(result, bg, bar_color, windows, source,
                    s_content, s_ext, N_src):
    """Each window shrinks bar by 1, retracts far edge by 1, fills +1 step."""
    for w in windows:
        content = _get_content(w.interior_grid)
        cr, cc = content.shape
        if cr == 0 or cc == 0:
            continue

        # Determine this window's current N
        if w.is_mixed:
            own_ext = s_ext
            own_N = N_src
        else:
            own_N = 0
            own_ext = None

        new_content = progressive_fill(
            np.full_like(content, 5), w.side, own_N + 1,
            source_extents=own_ext, N_source=own_N,
            source_side=w.side if own_ext else None,
            source_shape=content.shape if own_ext else None,
            K=1,  # in-place always K=1
        )

        # Erase old window
        r0, c0, r1, c1 = w.bbox
        result[r0:r1 + 1, c0:c1 + 1] = bg

        # Draw new window: bar at bar-adjacent edge, bar_thickness reduced by 1
        # The bar stays at its original outermost position, interior shifts toward bar
        if w.side == "top":
            _rebuild_window(result, bg, bar_color, w.side, new_content,
                            anchor_r=r0, anchor_c=c0)
        elif w.side == "bottom":
            # Bar stays at r1, interior above: anchor = r1 - (cr+2)
            _rebuild_window(result, bg, bar_color, w.side, new_content,
                            anchor_r=r1 - (cr + 2), anchor_c=c0)
        elif w.side == "left":
            _rebuild_window(result, bg, bar_color, w.side, new_content,
                            anchor_r=r0, anchor_c=c0)
        else:  # right
            _rebuild_window(result, bg, bar_color, w.side, new_content,
                            anchor_r=r0, anchor_c=c1 - (cc + 2))

    return result


def _window_center(w: BarWindow) -> tuple[float, float]:
    r0, c0, r1, c1 = w.bbox
    return ((r0 + r1) / 2, (c0 + c1) / 2)


def _is_fully_filled(w: BarWindow, fill_color: int = 7, base_color: int = 5) -> bool:
    """Content has no base_color left — fully filled already."""
    content = _get_content(w.interior_grid)
    return content.size > 0 and base_color not in content


def _solve_transfer(result, bg, bar_color, windows, sources, targets):
    """Move content from source/workspace to empty targets with progressive fill."""
    content_windows = [
        w for w in windows if not w.is_empty and not _is_fully_filled(w)
    ]

    # Pair targets ↔ content by the MATCHING axis band, then distance.
    # Horizontal targets (bar=top/bottom) → match by COLUMN band.
    # Vertical targets (bar=left/right) → match by ROW band.
    used_t: set[int] = set()
    used_c: set[int] = set()

    pairs: list[tuple[BarWindow, BarWindow]] = []
    # Two passes: (1) vertical targets by row band, (2) horizontal by col band.
    # This prevents horizontal targets from stealing vertical-target content.
    for pass_horizontal in (False, True):
        for ti, t in enumerate(targets):
            if ti in used_t:
                continue
            is_horiz = t.side in ("top", "bottom")
            if is_horiz != pass_horizontal:
                continue

            if is_horiz:
                t_band = _col_band(t.bbox)
                band_fn = _col_band
            else:
                t_band = _row_band(t.bbox)
                band_fn = _row_band

            best_ci, best_dist = -1, float("inf")
            for ci, cw in enumerate(content_windows):
                if ci in used_c:
                    continue
                if not _bands_overlap(t_band, band_fn(cw.bbox)):
                    continue
                cr, cc = _window_center(cw)
                tr, tc = _window_center(t)
                dist = abs(tr - cr) + abs(tc - cc)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci

            # Fallback: try the other axis if no primary match
            if best_ci < 0:
                alt_band = _row_band(t.bbox) if is_horiz else _col_band(t.bbox)
                alt_fn = _row_band if is_horiz else _col_band
                for ci, cw in enumerate(content_windows):
                    if ci in used_c:
                        continue
                    if not _bands_overlap(alt_band, alt_fn(cw.bbox)):
                        continue
                    cr, cc = _window_center(cw)
                    tr, tc = _window_center(t)
                    dist = abs(tr - cr) + abs(tc - cc)
                    if dist < best_dist:
                        best_dist = dist
                        best_ci = ci

            if best_ci >= 0:
                pairs.append((t, content_windows[best_ci]))
                used_t.add(ti)
                used_c.add(best_ci)

    for target, paired in pairs:
        p_content = _get_content(paired.interior_grid)
        pr, pc = p_content.shape
        if pr == 0 or pc == 0:
            continue

        # Determine output bar side: content extends toward paired's position
        tr0, tc0, tr1, tc1 = target.bbox
        pr0, pc0, pr1, pc1 = paired.bbox

        # Use the axis with more separation
        row_sep = abs((pr0 + pr1) / 2 - (tr0 + tr1) / 2)
        col_sep = abs((pc0 + pc1) / 2 - (tc0 + tc1) / 2)

        if col_sep >= row_sep:
            # Horizontal transfer
            if pc0 < tc0:
                out_side: WindowSide = "right"
            else:
                out_side = "left"
        else:
            # Vertical transfer
            if pr0 < tr0:
                out_side = "bottom"
            else:
                out_side = "top"

        # Find the nearest source for this pair's band
        pair_center = _window_center(paired)
        best_src = min(sources, key=lambda s: abs(_window_center(s)[0] - pair_center[0]) + abs(_window_center(s)[1] - pair_center[1]))
        s_content, s_ext, N_src = _source_params(best_src)

        K = _compute_fill_K(s_content, best_src.side, N_src, is_transfer=True)
        sdim = pr if out_side in ("left", "right") else pc
        N_out = N_src + sdim

        filled = progressive_fill(
            np.full((pr, pc), 5, dtype=result.dtype),
            out_side, N_out,
            source_extents=s_ext, N_source=N_src,
            source_side=best_src.side, source_shape=s_content.shape,
            K=K,
        )

        # Erase paired window
        er0, ec0, er1, ec1 = paired.bbox
        result[er0:er1 + 1, ec0:ec1 + 1] = bg

        # Place new window: bar anchored at target position
        iw = pc + 2
        ih = pr + 2

        # Check axis compatibility: target bar axis vs output bar axis
        target_vertical = target.side in ("left", "right")
        output_vertical = out_side in ("left", "right")
        axes_match = target_vertical == output_vertical

        if axes_match:
            # For single-slot (K>2) horizontal targets with bar=top,
            # the target bar stays separate and the new window starts after it.
            # All other cases: bar merges with the target.
            separate_bar = (K is not None and K > 2 and out_side == "top")

            if out_side == "left":
                anchor_r, anchor_c = tr0, tc0
            elif out_side == "right":
                anchor_r = tr0
                anchor_c = tc1 - iw
            elif out_side == "top":
                if separate_bar:
                    anchor_r = tr1 + 1
                else:
                    anchor_r = tr0
                anchor_c = tc0
            else:
                anchor_r = tr1 - ih
                anchor_c = tc0
        else:
            # Different axes: place adjacent with 1-cell bg gap
            if out_side == "left":
                anchor_r = tr0
                anchor_c = tc1 + 2  # target end + gap + bar
            elif out_side == "right":
                anchor_r = tr0
                anchor_c = tc0 - 2 - iw  # gap before target
            elif out_side == "top":
                anchor_r = tr1 + 2
                anchor_c = tc0
            else:
                anchor_r = tr0 - 2 - ih
                anchor_c = tc0

        _rebuild_window(result, bg, bar_color, out_side, filled,
                        anchor_r=anchor_r, anchor_c=anchor_c)

    return result
