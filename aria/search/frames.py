"""Frame-item extraction helpers for G07-style output tasks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from aria.types import Grid


_STRUCTURE4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


@dataclass(frozen=True)
class RectFrame:
    color: int
    row: int
    col: int
    height: int
    width: int
    patch: np.ndarray

    @property
    def bottom(self) -> int:
        return self.row + self.height - 1

    @property
    def right(self) -> int:
        return self.col + self.width - 1

    @property
    def interior(self) -> np.ndarray:
        if self.height <= 2 or self.width <= 2:
            return np.zeros((0, 0), dtype=self.patch.dtype)
        return self.patch[1:-1, 1:-1]


@dataclass(frozen=True)
class RectItem:
    color: int
    row: int
    col: int
    height: int
    width: int
    patch: np.ndarray
    kind: str
    interior_bg: bool


def extract_rect_frames(
    grid: Grid,
    *,
    bg: int = 0,
    min_span: int = 4,
) -> list[RectFrame]:
    """Extract exact hollow rectangular frames as reusable frame items."""
    frames: list[RectFrame] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        labels, count = ndimage.label(grid == color, structure=_STRUCTURE4)
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            if len(coords) == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < min_span or w < min_span:
                continue
            mask = labels[r0:r1 + 1, c0:c1 + 1] == idx
            border = np.zeros_like(mask, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(mask[border]):
                continue
            if np.any(mask[1:-1, 1:-1]):
                continue
            patch = grid[r0:r1 + 1, c0:c1 + 1].copy()
            frames.append(
                RectFrame(
                    color=int(color),
                    row=int(r0),
                    col=int(c0),
                    height=h,
                    width=w,
                    patch=patch,
                )
            )
    frames.sort(key=lambda frame: (frame.row, frame.col, frame.color))
    return frames


def extract_rect_items(
    grid: Grid,
    *,
    bg: int = 0,
    min_span: int = 4,
    allow_frames: bool = True,
    allow_solids: bool = True,
) -> list[RectItem]:
    """Extract exact rectangular items, including hollow frames and solid blocks."""
    items: list[RectItem] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        labels, count = ndimage.label(grid == color, structure=_STRUCTURE4)
        for idx in range(1, count + 1):
            coords = np.argwhere(labels == idx)
            if len(coords) == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < min_span or w < min_span:
                continue
            mask = labels[r0:r1 + 1, c0:c1 + 1] == idx
            patch = grid[r0:r1 + 1, c0:c1 + 1].copy()
            if allow_solids and np.all(mask):
                items.append(
                    RectItem(
                        color=int(color),
                        row=int(r0),
                        col=int(c0),
                        height=h,
                        width=w,
                        patch=patch,
                        kind="solid",
                        interior_bg=False,
                    )
                )
                continue
            if not allow_frames:
                continue
            border = np.zeros_like(mask, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.all(mask[border]) and not np.any(mask[1:-1, 1:-1]):
                items.append(
                    RectItem(
                        color=int(color),
                        row=int(r0),
                        col=int(c0),
                        height=h,
                        width=w,
                        patch=patch,
                        kind="frame",
                        interior_bg=True,
                    )
                )
    items.sort(key=lambda item: (item.row, item.col, item.color, item.kind))
    return items


def group_frames_by_color(frames: list[RectFrame]) -> dict[int, list[RectFrame]]:
    """Stable family grouping for frame pack tasks."""
    grouped: dict[int, list[RectFrame]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame.color)].append(frame)
    return dict(sorted(grouped.items()))


def render_frame_family(
    frames: list[RectFrame],
    *,
    shape: tuple[int, int],
    bg: int,
    compact_rows: bool = False,
    compact_cols: bool = True,
) -> np.ndarray:
    """Render one color-family of frames into a canvas, with optional compaction."""
    h, w = shape
    canvas = np.full((h, w), bg, dtype=np.int64)
    for frame in frames:
        patch = frame.patch
        mask = patch != bg
        canvas[frame.row:frame.row + frame.height, frame.col:frame.col + frame.width][mask] = patch[mask]

    rows = list(range(h))
    cols = list(range(w))
    if compact_rows:
        rows = [r for r in rows if np.any(canvas[r] != bg)]
    if compact_cols:
        cols = [c for c in cols if np.any(canvas[:, c] != bg)]
    if not rows or not cols:
        return np.zeros((0, 0), dtype=canvas.dtype)
    return canvas[np.ix_(rows, cols)]
