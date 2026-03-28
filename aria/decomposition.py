"""Decomposition layer — structural views of a grid before rule inference.

Provides reusable decompositions that sketches and observation can consume:
- raw objects (connected components with role metadata)
- framed regions (interior grids within frame borders)
- composite motifs (singleton center + enclosing frame CCs)
- marker neighborhoods (spatial context around singleton markers)

Each decomposition is role-oriented: it exposes structural relationships
(enclosure, adjacency, singleton-ness) rather than just pixel blobs.
Color values are recorded but structure is expressed in color-invariant terms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from aria.types import Grid


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def detect_bg(grid: Grid) -> int:
    """Detect the background color as the most common value."""
    if grid.size == 0:
        return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# Raw objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawObject:
    """A connected component in a grid."""

    color: int
    row: int          # top-left row of bounding box
    col: int          # top-left col of bounding box
    size: int         # pixel count
    mask: np.ndarray  # bool mask within bbox
    # Derived
    bbox_h: int = 0
    bbox_w: int = 0
    is_singleton: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RawObject):
            return NotImplemented
        return (
            self.color == other.color
            and self.row == other.row
            and self.col == other.col
            and self.size == other.size
            and np.array_equal(self.mask, other.mask)
        )

    def __hash__(self) -> int:
        return hash((self.color, self.row, self.col, self.size))

    @property
    def center_row(self) -> int:
        return self.row + self.bbox_h // 2

    @property
    def center_col(self) -> int:
        return self.col + self.bbox_w // 2

    def to_dict(self) -> dict:
        """Legacy dict format for observe.py compatibility."""
        return {
            "color": self.color,
            "row": self.row,
            "col": self.col,
            "size": self.size,
            "mask": self.mask,
            "mask_bytes": self.mask.tobytes(),
            "mask_shape": self.mask.shape,
        }


def extract_objects(grid: Grid, bg: int | None = None) -> list[RawObject]:
    """Extract connected components from a grid, excluding background.

    Args:
        grid: input grid
        bg: background color. If None, detected as most common color.
    """
    if grid.size == 0:
        return []
    if bg is None:
        bg = detect_bg(grid)

    objects: list[RawObject] = []
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labeled == label_id)
            min_r, max_r = int(ys.min()), int(ys.max())
            min_c, max_c = int(xs.min()), int(xs.max())
            mask = labeled[min_r:max_r + 1, min_c:max_c + 1] == label_id
            size = int(mask.sum())
            objects.append(RawObject(
                color=color,
                row=min_r,
                col=min_c,
                size=size,
                mask=mask,
                bbox_h=max_r - min_r + 1,
                bbox_w=max_c - min_c + 1,
                is_singleton=(size == 1),
            ))
    return objects


@dataclass(frozen=True)
class ObjectDecomposition:
    """All connected components in a grid with role annotations."""

    bg_color: int
    objects: tuple[RawObject, ...]
    singletons: tuple[RawObject, ...]      # size == 1
    non_singletons: tuple[RawObject, ...]  # size > 1
    color_counts: dict[int, int]           # color → number of CCs of that color


def decompose_objects(grid: Grid, bg: int | None = None) -> ObjectDecomposition:
    """Decompose a grid into annotated connected components."""
    if bg is None:
        bg = detect_bg(grid)
    objs = extract_objects(grid, bg)
    singletons = tuple(o for o in objs if o.is_singleton)
    non_singletons = tuple(o for o in objs if not o.is_singleton)
    color_counts = dict(Counter(o.color for o in objs))
    return ObjectDecomposition(
        bg_color=bg,
        objects=tuple(objs),
        singletons=singletons,
        non_singletons=non_singletons,
        color_counts=color_counts,
    )


# ---------------------------------------------------------------------------
# Framed regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramedRegion:
    """A rectangular interior region enclosed by a frame color."""

    frame_color: int
    interior: Grid             # the sub-grid inside the frame
    row: int                   # top-left row of interior in the original grid
    col: int                   # top-left col of interior
    height: int
    width: int
    # Interior analysis
    interior_colors: tuple[int, ...]   # unique colors in interior
    interior_bg: int                    # most common color in interior


def detect_framed_regions(grid: Grid, bg: int | None = None) -> list[FramedRegion]:
    """Detect rectangular regions enclosed by a uniform frame color.

    Looks for a color that forms all four borders of a rectangular region.
    Handles both full-grid frames and sub-region frames.
    """
    if bg is None:
        bg = detect_bg(grid)
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return []

    regions: list[FramedRegion] = []

    # Check full-grid frame
    _check_frame_at(grid, bg, 0, 0, rows, cols, regions)

    # Check sub-regions bounded by non-bg color lines
    # Look for horizontal and vertical separators
    for color in range(10):
        if color == bg:
            continue
        # Find rectangular sub-regions bounded by this color
        _find_color_bounded_regions(grid, color, bg, regions)

    return regions


def _check_frame_at(
    grid: Grid, bg: int,
    r0: int, c0: int, h: int, w: int,
    results: list[FramedRegion],
) -> None:
    """Check if (r0,c0,h,w) has a uniform single-color frame border."""
    if h < 3 or w < 3:
        return

    # Top row
    top_colors = set(int(grid[r0, c0 + c]) for c in range(w))
    if len(top_colors) != 1:
        return
    frame_color = top_colors.pop()
    if frame_color == bg:
        # bg-colored frame is still a frame if interior differs
        pass

    # Bottom row
    bottom_colors = set(int(grid[r0 + h - 1, c0 + c]) for c in range(w))
    if bottom_colors != {frame_color}:
        return

    # Left col
    left_colors = set(int(grid[r0 + r, c0]) for r in range(h))
    if left_colors != {frame_color}:
        return

    # Right col
    right_colors = set(int(grid[r0 + r, c0 + w - 1]) for r in range(h))
    if right_colors != {frame_color}:
        return

    # Extract interior
    interior = grid[r0 + 1:r0 + h - 1, c0 + 1:c0 + w - 1].copy()
    if interior.size == 0:
        return

    int_colors = tuple(sorted(set(int(v) for v in np.unique(interior))))
    int_bg = detect_bg(interior)

    results.append(FramedRegion(
        frame_color=frame_color,
        interior=interior,
        row=r0 + 1,
        col=c0 + 1,
        height=h - 2,
        width=w - 2,
        interior_colors=int_colors,
        interior_bg=int_bg,
    ))


def _find_color_bounded_regions(
    grid: Grid, color: int, bg: int,
    results: list[FramedRegion],
) -> None:
    """Find rectangular sub-regions where `color` forms all four borders.

    Scans for rows/cols that are entirely `color` and checks the rectangles
    they form.
    """
    rows, cols = grid.shape

    # Find full rows of this color
    full_rows = []
    for r in range(rows):
        if all(int(grid[r, c]) == color for c in range(cols)):
            full_rows.append(r)

    # Find full cols of this color
    full_cols = []
    for c in range(cols):
        if all(int(grid[r, c]) == color for r in range(rows)):
            full_cols.append(c)

    # Check rectangles formed by consecutive row/col pairs
    for i in range(len(full_rows) - 1):
        for j in range(len(full_cols) - 1):
            r0 = full_rows[i]
            r1 = full_rows[i + 1]
            c0 = full_cols[j]
            c1 = full_cols[j + 1]
            if r1 - r0 >= 3 and c1 - c0 >= 3:
                _check_frame_at(grid, bg, r0, c0, r1 - r0 + 1, c1 - c0 + 1, results)


# ---------------------------------------------------------------------------
# Composite motifs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeMotif:
    """A multi-color structural unit: singleton center + adjacent frame CCs."""

    center: RawObject
    frames: tuple[RawObject, ...]
    center_row: int
    center_col: int

    @property
    def structural_signature(self) -> tuple:
        """Color-invariant structural fingerprint."""
        all_rows = [self.center_row]
        all_cols = [self.center_col]
        for f in self.frames:
            for dr in range(f.bbox_h):
                for dc in range(f.bbox_w):
                    if f.mask[dr, dc]:
                        all_rows.append(f.row + dr)
                        all_cols.append(f.col + dc)

        min_r, max_r = min(all_rows), max(all_rows)
        min_c, max_c = min(all_cols), max(all_cols)
        bbox_h = max_r - min_r + 1
        bbox_w = max_c - min_c + 1
        center_rel_r = self.center_row - min_r
        center_rel_c = self.center_col - min_c
        frame_sizes = tuple(sorted(f.size for f in self.frames))
        n_components = 1 + len(self.frames)
        return (n_components, frame_sizes, bbox_h, bbox_w, center_rel_r, center_rel_c)


@dataclass(frozen=True)
class CompositeDecomposition:
    """Composite motifs and isolated singletons in a grid."""

    bg_color: int
    composites: tuple[CompositeMotif, ...]
    isolated: tuple[RawObject, ...]   # singletons with no adjacent frame
    center_color: int | None          # dominant center color, if clear
    frame_color: int | None           # dominant frame color, if clear
    anchor: RawObject | None          # isolated singleton of center color


def decompose_composites(grid: Grid, bg: int | None = None) -> CompositeDecomposition:
    """Identify composite motifs (center+frame) and isolated anchors."""
    if bg is None:
        bg = detect_bg(grid)
    objs = extract_objects(grid, bg)

    rows, cols = grid.shape
    singletons = [o for o in objs if o.is_singleton]
    non_singletons = [o for o in objs if not o.is_singleton]

    # Build spatial lookup for non-singletons
    cell_to_obj: dict[tuple[int, int], RawObject] = {}
    for obj in non_singletons:
        for dr in range(obj.bbox_h):
            for dc in range(obj.bbox_w):
                if obj.mask[dr, dc]:
                    cell_to_obj[(obj.row + dr, obj.col + dc)] = obj

    composites: list[CompositeMotif] = []
    isolated: list[RawObject] = []

    for s in singletons:
        adj_frames: dict[int, RawObject] = {}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = s.row + dr, s.col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_color = int(grid[nr, nc])
            if cell_color == bg or cell_color == s.color:
                continue
            neighbor = cell_to_obj.get((nr, nc))
            if neighbor is not None and neighbor.color != s.color:
                adj_frames[id(neighbor)] = neighbor
            for other in singletons:
                if other is s or other.color == s.color:
                    continue
                if other.row == nr and other.col == nc:
                    adj_frames[id(other)] = other

        if adj_frames:
            composites.append(CompositeMotif(
                center=s,
                frames=tuple(adj_frames.values()),
                center_row=s.row,
                center_col=s.col,
            ))
        else:
            isolated.append(s)

    # Identify dominant center/frame colors
    center_color = None
    frame_color = None
    anchor = None
    if composites:
        pair_counts: Counter = Counter()
        for comp in composites:
            for f in comp.frames:
                if f.color != comp.center.color:
                    pair_counts[(comp.center.color, f.color)] += 1
        if pair_counts:
            (cc, fc), _ = pair_counts.most_common(1)[0]
            center_color = cc
            frame_color = fc
            # Anchor: isolated singleton of center_color
            anchors = [s for s in isolated if s.color == cc]
            if len(anchors) == 1:
                anchor = anchors[0]

    return CompositeDecomposition(
        bg_color=bg,
        composites=tuple(composites),
        isolated=tuple(isolated),
        center_color=center_color,
        frame_color=frame_color,
        anchor=anchor,
    )


# ---------------------------------------------------------------------------
# Marker neighborhoods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerNeighborhood:
    """Spatial context around a singleton marker object."""

    marker: RawObject
    marker_color: int
    marker_row: int
    marker_col: int
    # Nearby objects within a radius
    neighbors: tuple[RawObject, ...]
    # Which objects are above/below/left/right
    above: tuple[RawObject, ...]
    below: tuple[RawObject, ...]
    left: tuple[RawObject, ...]
    right: tuple[RawObject, ...]
    # Distance to nearest object of each color
    nearest_by_color: dict[int, float]


def decompose_marker_neighborhoods(
    grid: Grid,
    bg: int | None = None,
    radius: int = 0,
) -> list[MarkerNeighborhood]:
    """Find singleton markers and describe their spatial neighborhoods.

    A marker is a singleton (size-1) object whose color appears exactly
    once in the grid. If radius == 0, all objects are considered neighbors.
    """
    if bg is None:
        bg = detect_bg(grid)
    objs = extract_objects(grid, bg)
    if not objs:
        return []

    # Find colors that appear as exactly one singleton
    color_singletons: dict[int, list[RawObject]] = {}
    for o in objs:
        if o.is_singleton:
            color_singletons.setdefault(o.color, []).append(o)

    markers: list[RawObject] = []
    for color, slist in color_singletons.items():
        # A marker's color should appear in few CCs total
        total_of_color = sum(1 for o in objs if o.color == color)
        if len(slist) == 1 and total_of_color <= 2:
            markers.append(slist[0])

    neighborhoods: list[MarkerNeighborhood] = []
    for marker in markers:
        mr, mc = marker.center_row, marker.center_col
        non_marker = [o for o in objs if o is not marker]

        # Filter by radius if set
        if radius > 0:
            neighbors = [
                o for o in non_marker
                if abs(o.center_row - mr) + abs(o.center_col - mc) <= radius
            ]
        else:
            neighbors = non_marker

        above = tuple(o for o in neighbors if o.center_row < mr)
        below = tuple(o for o in neighbors if o.center_row > mr)
        left = tuple(o for o in neighbors if o.center_col < mc)
        right = tuple(o for o in neighbors if o.center_col > mc)

        nearest: dict[int, float] = {}
        for o in non_marker:
            d = abs(o.center_row - mr) + abs(o.center_col - mc)
            if o.color not in nearest or d < nearest[o.color]:
                nearest[o.color] = float(d)

        neighborhoods.append(MarkerNeighborhood(
            marker=marker,
            marker_color=marker.color,
            marker_row=mr,
            marker_col=mc,
            neighbors=tuple(neighbors),
            above=above,
            below=below,
            left=left,
            right=right,
            nearest_by_color=nearest,
        ))

    return neighborhoods


# ---------------------------------------------------------------------------
# Full grid decomposition summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridDecomposition:
    """All decomposition views for one grid."""

    bg_color: int
    objects: ObjectDecomposition
    framed_regions: tuple[FramedRegion, ...]
    composites: CompositeDecomposition
    marker_neighborhoods: tuple[MarkerNeighborhood, ...]


def decompose_grid(grid: Grid, bg: int | None = None) -> GridDecomposition:
    """Compute all decomposition views for a grid."""
    if bg is None:
        bg = detect_bg(grid)
    return GridDecomposition(
        bg_color=bg,
        objects=decompose_objects(grid, bg),
        framed_regions=tuple(detect_framed_regions(grid, bg)),
        composites=decompose_composites(grid, bg),
        marker_neighborhoods=tuple(decompose_marker_neighborhoods(grid, bg)),
    )
