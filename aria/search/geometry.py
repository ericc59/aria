"""Reusable geometric helpers for the canonical aria/search stack.

These helpers stay intentionally small: they expose the recurring spatial
operations that multiple search ops already need today, without introducing a
large scene-graph framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


Direction = Literal["up", "down", "left", "right"]
DiagDirection = Literal["up_left", "up_right", "down_left", "down_right"]
Bridge = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class BBox:
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1


def bbox(cells: Iterable[tuple[int, int]]) -> BBox:
    """Bounding box of a non-empty cell set."""
    pts = list(cells)
    if not pts:
        raise ValueError("bbox() requires at least one cell")
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return BBox(min(rs), max(rs), min(cs), max(cs))


def component_center(cells: Iterable[tuple[int, int]]) -> tuple[float, float]:
    """Arithmetic center of a cell set."""
    pts = list(cells)
    if not pts:
        raise ValueError("component_center() requires at least one cell")
    return (
        sum(r for r, _ in pts) / len(pts),
        sum(c for _, c in pts) / len(pts),
    )


def boundary_cells(cells: Iterable[tuple[int, int]], direction: Direction) -> list[tuple[int, int]]:
    """Cells on one boundary of a component bbox."""
    pts = list(cells)
    bounds = bbox(pts)
    if direction == "up":
        return [(r, c) for r, c in pts if r == bounds.top]
    if direction == "down":
        return [(r, c) for r, c in pts if r == bounds.bottom]
    if direction == "left":
        return [(r, c) for r, c in pts if c == bounds.left]
    return [(r, c) for r, c in pts if c == bounds.right]


def closest_boundary_anchor(
    cells: Iterable[tuple[int, int]],
    direction: Direction,
    *,
    target_row: float,
    target_col: float,
) -> int | None:
    """Boundary-aligned row/col anchor closest to a target point."""
    edge = boundary_cells(cells, direction)
    if not edge:
        return None
    if direction in ("up", "down"):
        edge = sorted(edge, key=lambda rc: abs(rc[1] - target_col))
        return edge[0][1]
    edge = sorted(edge, key=lambda rc: abs(rc[0] - target_row))
    return edge[0][0]


def axis_ray(
    start: tuple[int, int],
    direction: Direction,
    grid_shape: tuple[int, int],
    *,
    include_start: bool = False,
) -> list[tuple[int, int]]:
    """Project an orthogonal ray from a start cell until grid bounds."""
    h, w = grid_shape
    dr, dc = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]
    r, c = start
    result: list[tuple[int, int]] = []
    if include_start and 0 <= r < h and 0 <= c < w:
        result.append((r, c))
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return result
        result.append((r, c))


def diagonal_ray(
    start: tuple[int, int],
    direction: DiagDirection,
    grid_shape: tuple[int, int],
    *,
    include_start: bool = False,
) -> list[tuple[int, int]]:
    """Project a diagonal ray from a start cell until grid bounds."""
    h, w = grid_shape
    dr, dc = {
        "up_left": (-1, -1),
        "up_right": (-1, 1),
        "down_left": (1, -1),
        "down_right": (1, 1),
    }[direction]
    r, c = start
    result: list[tuple[int, int]] = []
    if include_start and 0 <= r < h and 0 <= c < w:
        result.append((r, c))
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return result
        result.append((r, c))


def orthogonal_path(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    order: Literal["hv", "vh"] = "hv",
    include_start: bool = True,
) -> list[tuple[int, int]]:
    """Axis-aligned L path between two cells."""
    sr, sc = start
    er, ec = end
    cells: list[tuple[int, int]] = [(sr, sc)] if include_start else []

    def _append_line(r0: int, c0: int, r1: int, c1: int) -> None:
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            cells.append((r, c))

    if order == "hv":
        _append_line(sr, sc, sr, ec)
        _append_line(sr, ec, er, ec)
    else:
        _append_line(sr, sc, er, sc)
        _append_line(er, sc, er, ec)
    return cells


def orthogonal_bridge(
    cells_a: Iterable[tuple[int, int]],
    cells_b: Iterable[tuple[int, int]],
) -> Bridge | None:
    """Minimal axis-aligned rectangle connecting aligned bbox gaps."""
    a = bbox(cells_a)
    b = bbox(cells_b)

    row0 = max(a.top, b.top)
    row1 = min(a.bottom, b.bottom)
    if row0 <= row1:
        if a.right < b.left:
            return ("h", row0, row1, a.right + 1, b.left - 1)
        if b.right < a.left:
            return ("h", row0, row1, b.right + 1, a.left - 1)

    col0 = max(a.left, b.left)
    col1 = min(a.right, b.right)
    if col0 <= col1:
        if a.bottom < b.top:
            return ("v", a.bottom + 1, b.top - 1, col0, col1)
        if b.bottom < a.top:
            return ("v", b.bottom + 1, a.top - 1, col0, col1)

    return None


def bridge_area(bridge: Bridge | None) -> int:
    """Area of an orthogonal bridge rectangle."""
    if bridge is None:
        return 0
    _, r0, r1, c0, c1 = bridge
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def paint_bridge(arr, bridge: Bridge | None, color: int) -> None:
    """Paint an orthogonal bridge rectangle in-place."""
    if bridge is None:
        return
    _, r0, r1, c0, c1 = bridge
    if r0 > r1 or c0 > c1:
        return
    arr[r0:r1 + 1, c0:c1 + 1] = color
