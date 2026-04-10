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
    seen: set[tuple[int, int, int, int, int, str]] = set()
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
                key = (int(color), int(r0), int(c0), h, w, "solid")
                if key not in seen:
                    seen.add(key)
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
                key = (int(color), int(r0), int(c0), h, w, "frame")
                if key not in seen:
                    seen.add(key)
                    items.append(
                        RectItem(
                            color=int(color),
                            row=int(r0),
                            col=int(c0),
                            height=h,
                            width=w,
                            patch=patch,
                            kind="frame",
                            interior_bg=bool(np.all(patch[1:-1, 1:-1] == bg)),
                        )
                    )
        if allow_frames:
            rows, cols = grid.shape
            for r0 in range(rows - min_span + 1):
                for c0 in range(cols - min_span + 1):
                    for h in range(min_span, rows - r0 + 1):
                        for w in range(min_span, cols - c0 + 1):
                            patch = grid[r0:r0 + h, c0:c0 + w]
                            border = np.zeros((h, w), dtype=bool)
                            border[0, :] = True
                            border[-1, :] = True
                            border[:, 0] = True
                            border[:, -1] = True
                            if not np.all(patch[border] == color):
                                continue
                            if np.any(patch[1:-1, 1:-1] == color):
                                continue
                            key = (int(color), int(r0), int(c0), h, w, "frame")
                            if key in seen:
                                continue
                            seen.add(key)
                            items.append(
                                RectItem(
                                    color=int(color),
                                    row=int(r0),
                                    col=int(c0),
                                    height=h,
                                    width=w,
                                    patch=patch.copy(),
                                    kind="frame",
                                    interior_bg=bool(np.all(patch[1:-1, 1:-1] == bg)),
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


def max_rect_row_overlap(items: list[RectItem]) -> int:
    """Return the maximum number of row-overlapping rectangles in a family."""
    events: list[tuple[int, int]] = []
    for item in items:
        events.append((item.row, 1))
        events.append((item.row + item.height, -1))
    depth = best = 0
    for _, delta in sorted(events):
        depth += delta
        best = max(best, depth)
    return best


def assign_rect_family_lanes(
    items: list[RectItem],
    lane_count: int | None = None,
    *,
    default_lane: int = 0,
) -> dict[int, int]:
    """Assign each rectangle to a non-overlapping lane with compact column clusters.

    The objective is to keep items with similar source columns in the same lane while
    ensuring row-overlapping rectangles are never assigned to the same lane.
    """
    if not items:
        return {}
    if lane_count is None:
        lane_count = max_rect_row_overlap(items)
    lane_count = max(1, int(lane_count))
    overlaps: list[list[int]] = [[] for _ in items]
    for i, item_i in enumerate(items):
        for j in range(i + 1, len(items)):
            item_j = items[j]
            if item_i.row < item_j.row + item_j.height and item_j.row < item_i.row + item_i.height:
                overlaps[i].append(j)
                overlaps[j].append(i)

    components: list[list[int]] = []
    seen = [False] * len(items)
    for start in range(len(items)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            idx = stack.pop()
            comp.append(idx)
            for nxt in overlaps[idx]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(sorted(comp, key=lambda idx: (items[idx].row, items[idx].col, idx)))

    result: dict[int, int] = {}
    for comp in sorted(components, key=lambda comp: min(items[idx].row for idx in comp)):
        if len(comp) == 1:
            result[comp[0]] = max(0, min(lane_count - 1, default_lane))
            continue

        ordered = sorted(comp, key=lambda idx: (items[idx].row, items[idx].col, idx))
        lane_cols: list[list[int]] = [[] for _ in range(lane_count)]
        lane_busy_until = [-1] * lane_count
        assignment = [-1] * len(ordered)
        best_cost: float | None = None
        best_assignment: list[int] | None = None

        def rec(i: int, cost: float) -> None:
            nonlocal best_cost, best_assignment
            if best_cost is not None and cost >= best_cost:
                return
            if i >= len(ordered):
                best_cost = cost
                best_assignment = assignment.copy()
                return
            item = items[ordered[i]]
            options: list[tuple[float, int]] = []
            for lane in range(lane_count):
                if lane_busy_until[lane] > item.row:
                    continue
                cols = lane_cols[lane]
                if cols:
                    mean = float(sum(cols)) / float(len(cols))
                    penalty = float((item.col - mean) ** 2)
                else:
                    penalty = float((lane - min(i, lane_count - 1)) ** 2)
                options.append((penalty, lane))
            for penalty, lane in sorted(options):
                prev_busy = lane_busy_until[lane]
                lane_busy_until[lane] = item.row + item.height
                lane_cols[lane].append(item.col)
                assignment[i] = lane
                rec(i + 1, cost + penalty)
                assignment[i] = -1
                lane_cols[lane].pop()
                lane_busy_until[lane] = prev_busy

        rec(0, 0.0)
        if best_assignment is None:
            raise ValueError("Failed to assign family lanes")

        lane_to_cols: dict[int, list[int]] = defaultdict(list)
        for idx, lane in zip(ordered, best_assignment):
            lane_to_cols[lane].append(items[idx].col)
        used_lanes = sorted(set(best_assignment))
        lane_order = {
            lane: rank
            for rank, lane in enumerate(
                sorted(used_lanes, key=lambda lane: (float(sum(lane_to_cols[lane])) / float(len(lane_to_cols[lane])), lane))
            )
        }
        for idx, lane in zip(ordered, best_assignment):
            result[idx] = lane_order[lane]
    return result


def render_rect_family_side(
    items: list[RectItem],
    *,
    shape: tuple[int, int],
    bg: int,
    side: str,
) -> np.ndarray:
    """Render a rectangular-item family against one side with lane packing."""
    if not items:
        return np.full(shape, bg, dtype=np.int64)

    h, w = shape
    item_w = items[0].width
    lane_count = max_rect_row_overlap(items)
    default_lane = 0 if side == "left" else lane_count - 1
    lane_map = assign_rect_family_lanes(items, lane_count, default_lane=default_lane)
    start_col = 0 if side == "left" else w - lane_count * item_w
    lane_cols = [start_col + lane * item_w for lane in range(lane_count)]

    canvas = np.full((h, w), bg, dtype=np.int64)
    for idx, item in enumerate(items):
        col = lane_cols[lane_map[idx]]
        patch = item.patch
        mask = patch != bg
        canvas[item.row:item.row + item.height, col:col + item.width][mask] = patch[mask]
    return canvas
