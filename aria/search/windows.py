"""Extraction, classification, and geometric primitives for bar-window scenes.

Bar-windows are rectangular components with a uniform straight bar on one side.
The bar acts as an anchor/role marker; the interior carries content.

This module provides:
  - BarWindow extraction and role classification (perception)
  - progressive_fill: parameterized staircase fill (content transform)
  - stamp_bar_window: place bar+border+content into a grid (spatial primitive)
  - erase_region: clear a bbox to background (spatial primitive)

Solver logic that orchestrates these primitives lives in the derive layer
(binding_derive.py), NOT here.
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


# ---------------------------------------------------------------------------
# Extraction internals
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Extraction public API
# ---------------------------------------------------------------------------

def extract_bar_windows(
    grid: np.ndarray,
    bg: int | None = None,
    *,
    max_bar_thickness: int = 3,
    min_side_extent: int = 3,
) -> list[BarWindow]:
    """Extract rectangular bar-window entities from a grid."""
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
    """Assign conservative scene roles for extracted bar windows."""
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
# Content utilities
# ---------------------------------------------------------------------------

def strip_border(interior: np.ndarray) -> np.ndarray:
    """Strip the 0-valued border from an interior grid to get content."""
    h, w = interior.shape
    if h <= 2 or w <= 2:
        return interior
    return interior[1:-1, 1:-1]


def source_fill_params(source: BarWindow) -> tuple[np.ndarray, list[int], int]:
    """Extract (content, fill_extents, N_source) for a source window."""
    content = strip_border(source.interior_grid)
    ext = get_fill_extents(content, source.side)
    _, _, ci = _stack_and_fill_dims(source.side, *content.shape)
    N = ext[ci]
    return content, ext, N


# ---------------------------------------------------------------------------
# Progressive fill (content transform primitive)
# ---------------------------------------------------------------------------

def _fill_corner(side: WindowSide, rows: int, cols: int) -> tuple[int, int]:
    """Manhattan staircase corner for a given bar side."""
    if side == "left":
        return (rows - 1, 0)
    if side == "right":
        return (0, cols - 1)
    if side == "top":
        return (0, 0)
    return (rows - 1, cols - 1)


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


def compute_fill_K(
    content: np.ndarray,
    side: WindowSide,
    N_source: int,
    is_transfer: bool,
) -> int:
    """Determine fill step count K.

    K=1 for in-place.  For transfer: K=2 if 7-cells span >=2 positions
    along the stack axis, else K = fill_dim - N_source (fills to edge).
    """
    if not is_transfer:
        return 1
    rows, cols = content.shape
    _, fdim, _ = _stack_and_fill_dims(side, rows, cols)

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
    """Apply progressive staircase fill to a content grid.

    Two modes:

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
        else:
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
# Spatial primitives
# ---------------------------------------------------------------------------

def erase_region(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    """Clear a bounding box to background color (in-place)."""
    r0, c0, r1, c1 = bbox
    grid[r0:r1 + 1, c0:c1 + 1] = bg


def stamp_bar_window(
    grid: np.ndarray,
    content: np.ndarray,
    side: WindowSide,
    bar_color: int,
    anchor_r: int,
    anchor_c: int,
) -> None:
    """Draw bar(1) + 0-border + content into *grid* at anchor position (in-place).

    Anchor is the top-left of the window.  For bar=top the bar is at
    row anchor_r; for bar=left the bar is at col anchor_c.
    """
    cr, cc = content.shape
    interior = np.zeros((cr + 2, cc + 2), dtype=grid.dtype)
    interior[1:-1, 1:-1] = content
    ih, iw = interior.shape

    H, W = grid.shape
    if side == "top":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + 1 + ih)
        c_end = min(W, c0 + iw)
        grid[r0, c0:c_end] = bar_color
        h_avail = r_end - (r0 + 1)
        w_avail = c_end - c0
        if h_avail > 0 and w_avail > 0:
            grid[r0 + 1:r0 + 1 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
    elif side == "bottom":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih + 1)
        c_end = min(W, c0 + iw)
        h_avail = min(ih, r_end - r0 - 1)
        w_avail = c_end - c0
        if h_avail > 0 and w_avail > 0:
            grid[r0:r0 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
        bar_r = r0 + h_avail
        if bar_r < H:
            grid[bar_r, c0:c_end] = bar_color
    elif side == "left":
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih)
        c_end = min(W, c0 + 1 + iw)
        h_avail = r_end - r0
        w_avail = c_end - (c0 + 1)
        grid[r0:r_end, c0] = bar_color
        if h_avail > 0 and w_avail > 0:
            grid[r0:r0 + h_avail, c0 + 1:c0 + 1 + w_avail] = interior[:h_avail, :w_avail]
    else:  # right
        r0 = max(0, anchor_r)
        c0 = max(0, anchor_c)
        r_end = min(H, r0 + ih)
        c_end = min(W, c0 + iw + 1)
        h_avail = r_end - r0
        w_avail = min(iw, c_end - c0 - 1)
        if h_avail > 0 and w_avail > 0:
            grid[r0:r0 + h_avail, c0:c0 + w_avail] = interior[:h_avail, :w_avail]
        bar_c = c0 + w_avail
        if bar_c < W:
            grid[r0:r_end, bar_c] = bar_color
