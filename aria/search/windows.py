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
