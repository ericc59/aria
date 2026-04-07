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


def extract_objects(
    grid: Grid,
    bg: int | None = None,
    *,
    connectivity: int = 4,
) -> list[RawObject]:
    """Extract connected components from a grid, excluding background.

    Args:
        grid: input grid
        bg: background color. If None, detected as most common color.
    """
    if grid.size == 0:
        return []
    if bg is None:
        bg = detect_bg(grid)

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    elif connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unsupported connectivity: {connectivity}")

    objects: list[RawObject] = []
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary, structure=structure)
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


def decompose_objects(
    grid: Grid,
    bg: int | None = None,
    *,
    connectivity: int = 4,
) -> ObjectDecomposition:
    """Decompose a grid into annotated connected components."""
    if bg is None:
        bg = detect_bg(grid)
    objs = extract_objects(grid, bg, connectivity=connectivity)
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


def detect_boxed_regions(grid: Grid, bg: int | None = None) -> list[FramedRegion]:
    """Detect rectangular boxed regions, including sparse internal frames.

    This extends `detect_framed_regions` with a more permissive scan for
    axis-aligned rectangles whose border is a single color, even when the
    rectangle is not induced by full-grid separator rows/cols.
    """
    if bg is None:
        bg = detect_bg(grid)
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return []

    regions = detect_framed_regions(grid, bg)
    seen = {
        (region.frame_color, region.row, region.col, region.height, region.width)
        for region in regions
    }

    palette = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    for color in palette:
        _find_sparse_rectangular_frames(grid, color, seen, regions)

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


def _find_sparse_rectangular_frames(
    grid: Grid,
    color: int,
    seen: set[tuple[int, int, int, int, int]],
    results: list[FramedRegion],
) -> None:
    """Find internal rectangular frames of a single color.

    Candidate rectangles come from matching horizontal color runs. Vertical
    sides are then verified with prefix-sum queries, which is much cheaper than
    scanning every possible rectangle in the grid.
    """
    rows, cols = grid.shape
    mask = grid == color
    if int(mask.sum()) < 8:
        return

    prefix = _binary_prefix_sum(mask)
    row_runs: dict[tuple[int, int], list[int]] = {}

    for r in range(rows):
        c = 0
        while c < cols:
            if not mask[r, c]:
                c += 1
                continue
            start = c
            while c + 1 < cols and mask[r, c + 1]:
                c += 1
            end = c
            if end - start + 1 >= 3:
                row_runs.setdefault((start, end), []).append(r)
            c += 1

    for (c0, c1), run_rows in row_runs.items():
        if len(run_rows) < 2:
            continue
        for i, r0 in enumerate(run_rows[:-1]):
            for r1 in run_rows[i + 1:]:
                if r1 - r0 < 2:
                    continue
                if _sum_rect(prefix, r0, c0, r1, c0) != r1 - r0 + 1:
                    continue
                if _sum_rect(prefix, r0, c1, r1, c1) != r1 - r0 + 1:
                    continue
                interior = grid[r0 + 1:r1, c0 + 1:c1].copy()
                if interior.size == 0:
                    continue
                key = (color, r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    FramedRegion(
                        frame_color=color,
                        interior=interior,
                        row=r0 + 1,
                        col=c0 + 1,
                        height=r1 - r0 - 1,
                        width=c1 - c0 - 1,
                        interior_colors=tuple(sorted(set(int(v) for v in np.unique(interior)))),
                        interior_bg=detect_bg(interior),
                    )
                )


def _binary_prefix_sum(mask: np.ndarray) -> np.ndarray:
    """2D prefix sum over a boolean mask, padded by one row/col."""
    rows, cols = mask.shape
    prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    prefix[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), axis=0), axis=1)
    return prefix


def _sum_rect(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    """Inclusive rectangle sum using a 1-padded prefix sum."""
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


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
# Panel decomposition (column-only or row-only separators)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One panel in a panel decomposition."""

    index: int
    grid: Grid                # the sub-grid content
    row: int                  # top-left row in parent grid
    col: int                  # top-left col in parent grid
    height: int
    width: int
    palette: frozenset[int]
    bg_color: int


@dataclass(frozen=True)
class PanelDecomposition:
    """Grid split into panels by separator rows or columns.

    Unlike a full partition (which requires both axes), panels are split
    along a single axis. This captures tasks where columns (or rows) of
    a separator color divide the grid into independent sub-grids.
    """

    axis: str                         # "row" or "col"
    separator_color: int
    separator_positions: tuple[int, ...]
    panels: tuple[Panel, ...]
    n_panels: int
    uniform_panel_size: bool


def detect_panels(grid: Grid, bg: int | None = None) -> PanelDecomposition | None:
    """Detect panel structure: full-span separators on one axis.

    Returns the best panel decomposition or None. Tries column separators
    first (more common in ARC), then row separators.
    """
    if bg is None:
        bg = detect_bg(grid)
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return None

    # Try column separators (each column entirely one non-bg color)
    col_sep = _detect_axis_separators(grid, axis="col", bg=bg)
    if col_sep is not None:
        return col_sep

    # Try row separators
    row_sep = _detect_axis_separators(grid, axis="row", bg=bg)
    if row_sep is not None:
        return row_sep

    return None


def _detect_axis_separators(
    grid: Grid, axis: str, bg: int,
) -> PanelDecomposition | None:
    """Detect separators along one axis (column or row)."""
    rows, cols = grid.shape

    # Collect uniform full-span lines per color
    candidates: dict[int, list[int]] = {}
    if axis == "col":
        for c in range(cols):
            vals = set(int(grid[r, c]) for r in range(rows))
            if len(vals) == 1:
                candidates.setdefault(next(iter(vals)), []).append(c)
    else:
        for r in range(rows):
            vals = set(int(grid[r, c]) for c in range(cols))
            if len(vals) == 1:
                candidates.setdefault(next(iter(vals)), []).append(r)

    best: PanelDecomposition | None = None

    for color, positions in candidates.items():
        # Need at least 1 inner separator (not just edge)
        span = cols if axis == "col" else rows
        inner = [p for p in positions if 0 < p < span - 1]
        if not inner:
            continue

        # Build intervals between separators
        all_seps = sorted(positions)
        intervals: list[tuple[int, int]] = []
        # Before first separator
        if all_seps[0] > 0:
            intervals.append((0, all_seps[0] - 1))
        # Between consecutive separators
        for i in range(len(all_seps) - 1):
            start = all_seps[i] + 1
            end = all_seps[i + 1] - 1
            if start <= end:
                intervals.append((start, end))
        # After last separator
        if all_seps[-1] < span - 1:
            intervals.append((all_seps[-1] + 1, span - 1))

        if len(intervals) < 2:
            continue

        # Extract panels
        panels: list[Panel] = []
        for idx, (start, end) in enumerate(intervals):
            if axis == "col":
                sub = grid[:, start:end + 1].copy()
                panel = Panel(
                    index=idx, grid=sub,
                    row=0, col=start,
                    height=rows, width=end - start + 1,
                    palette=frozenset(int(v) for v in np.unique(sub)),
                    bg_color=detect_bg(sub),
                )
            else:
                sub = grid[start:end + 1, :].copy()
                panel = Panel(
                    index=idx, grid=sub,
                    row=start, col=0,
                    height=end - start + 1, width=cols,
                    palette=frozenset(int(v) for v in np.unique(sub)),
                    bg_color=detect_bg(sub),
                )
            panels.append(panel)

        sizes = [(p.height, p.width) for p in panels]
        uniform = len(set(sizes)) == 1

        # Reject single-cell-wide panels from bg separator (noise)
        min_size = min(p.width if axis == "col" else p.height for p in panels)
        if color == bg and min_size <= 1 and len(panels) > 4:
            continue

        # Score: strongly prefer non-bg separator, prefer uniform, prefer
        # moderate panel count (2-6), penalize too many tiny panels
        is_non_bg = color != bg
        score = (
            100 * int(is_non_bg)
            + 50 * int(uniform)
            + 30 * int(2 <= len(panels) <= 6)
            - 5 * max(0, len(panels) - 6)  # penalize excessive panels
            + min_size  # prefer larger panels
        )
        decomp = PanelDecomposition(
            axis=axis,
            separator_color=color,
            separator_positions=tuple(all_seps),
            panels=tuple(panels),
            n_panels=len(panels),
            uniform_panel_size=uniform,
        )
        best_score = -1
        if best is not None:
            best_min_size = min(
                p.width if best.axis == "col" else p.height for p in best.panels
            )
            best_score = (
                100 * int(best.separator_color != bg)
                + 50 * int(best.uniform_panel_size)
                + 30 * int(2 <= best.n_panels <= 6)
                - 5 * max(0, best.n_panels - 6)
                + best_min_size
            )
        if best is None or score > best_score:
            best = decomp

    return best


# ---------------------------------------------------------------------------
# Partition decomposition (separator grid on both axes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionDecomposition:
    """Grid split into a cell grid by separators on both axes.

    Wraps the existing PartitionScene from graph.partition with
    additional metadata for the decomposition layer.
    """

    separator_color: int
    n_rows: int
    n_cols: int
    n_cells: int
    cell_grids: tuple[Grid, ...]        # one sub-grid per cell, row-major
    cell_bboxes: tuple[tuple[int, int, int, int], ...]  # (r0, c0, r1, c1)
    uniform: bool
    # Pass through the full PartitionScene for downstream use
    scene: object  # PartitionScene


def detect_partition_decomp(grid: Grid, bg: int | None = None) -> PartitionDecomposition | None:
    """Detect partition structure: separators on both axes.

    Extends graph.partition.detect_partition to also handle bg-colored
    separators (common in ARC-2).
    """
    from aria.graph.partition import detect_partition

    if bg is None:
        bg = detect_bg(grid)

    # Standard detection (non-bg separators)
    scene = detect_partition(grid, background=bg)

    # Fallback: try bg-colored separators
    if scene is None:
        scene = _detect_bg_partition(grid, bg)

    if scene is None:
        return None

    # Extract cell sub-grids
    cell_grids: list[Grid] = []
    cell_bboxes: list[tuple[int, int, int, int]] = []
    for cell in scene.cells:
        r0, c0, r1, c1 = cell.bbox
        cell_grids.append(grid[r0:r1 + 1, c0:c1 + 1].copy())
        cell_bboxes.append(cell.bbox)

    return PartitionDecomposition(
        separator_color=scene.separator_color,
        n_rows=scene.n_rows,
        n_cols=scene.n_cols,
        n_cells=len(scene.cells),
        cell_grids=tuple(cell_grids),
        cell_bboxes=tuple(cell_bboxes),
        uniform=scene.is_uniform_partition,
        scene=scene,
    )


def _detect_bg_partition(grid: Grid, bg: int) -> object | None:
    """Detect partition where separators are the background color.

    Many ARC tasks use the most common color as both background and
    separator. We look for full rows AND full columns of bg that
    create a grid of non-trivial cells.
    """
    rows, cols = grid.shape
    if rows < 5 or cols < 5:
        return None

    from aria.types import PartitionCell, PartitionScene

    full_rows = [r for r in range(rows) if np.all(grid[r, :] == bg)]
    full_cols = [c for c in range(cols) if np.all(grid[:, c] == bg)]

    if not full_rows or not full_cols:
        return None

    # Need at least 1 inner separator on each axis
    inner_rows = [r for r in full_rows if 0 < r < rows - 1]
    inner_cols = [c for c in full_cols if 0 < c < cols - 1]
    if not inner_rows or not inner_cols:
        return None

    # Compute intervals
    from aria.graph.partition import _compute_intervals
    row_intervals = _compute_intervals(full_rows, rows)
    col_intervals = _compute_intervals(full_cols, cols)

    if not row_intervals or not col_intervals:
        return None
    if len(row_intervals) * len(col_intervals) < 2:
        return None

    # Build cells — reject if cells are too trivial
    cells: list[PartitionCell] = []
    cell_shapes: list[tuple[int, int]] = []
    nonempty_count = 0
    for row_idx, (r0, r1) in enumerate(row_intervals):
        for col_idx, (c0, c1) in enumerate(col_intervals):
            cell_grid = grid[r0:r1 + 1, c0:c1 + 1]
            if cell_grid.size == 0:
                continue
            has_content = np.any(cell_grid != bg)
            if has_content:
                nonempty_count += 1
            cell_shapes.append((int(cell_grid.shape[0]), int(cell_grid.shape[1])))
            cells.append(PartitionCell(
                row_idx=row_idx, col_idx=col_idx,
                bbox=(r0, c0, r1, c1),
                dims=(int(cell_grid.shape[0]), int(cell_grid.shape[1])),
                background=bg,
                palette=frozenset(int(v) for v in np.unique(cell_grid)),
                obj_count=1 if has_content else 0,
            ))

    # Reject if too few cells have content (spurious bg partition)
    if nonempty_count < 2:
        return None

    return PartitionScene(
        separator_color=bg,
        separator_rows=tuple(sorted(full_rows)),
        separator_cols=tuple(sorted(full_cols)),
        cells=tuple(cells),
        n_rows=len(row_intervals),
        n_cols=len(col_intervals),
        cell_shapes=tuple(cell_shapes),
        is_uniform_partition=len(set(cell_shapes)) == 1,
    )


# ---------------------------------------------------------------------------
# Region decomposition (bordered rectangular sub-grids)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionDecomposition:
    """Rectangular bordered regions detected in a grid.

    Unifies framed_regions and boxed_regions into a single view.
    """

    regions: tuple[FramedRegion, ...]
    n_regions: int
    uniform_size: bool
    frame_colors: frozenset[int]


def detect_region_decomp(grid: Grid, bg: int | None = None) -> RegionDecomposition | None:
    """Detect bordered rectangular regions.

    Returns None if no regions found.
    """
    if bg is None:
        bg = detect_bg(grid)
    regions = detect_boxed_regions(grid, bg)
    if not regions:
        return None

    sizes = [(r.height, r.width) for r in regions]
    return RegionDecomposition(
        regions=tuple(regions),
        n_regions=len(regions),
        uniform_size=len(set(sizes)) == 1,
        frame_colors=frozenset(r.frame_color for r in regions),
    )


# ---------------------------------------------------------------------------
# Host-slot decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostSlotDecomposition:
    """Large host structures with smaller enclosed/adjacent slot entities.

    A host-slot decomposition exists when the grid contains composite
    motifs (center + frame structures) with an anchor singleton. This
    is more specific than plain composite decomposition: it identifies
    the spatial relationship between hosts (large structures), slots
    (positions to fill), and markers (indicators).
    """

    hosts: tuple[RawObject, ...]
    slots: tuple[RawObject, ...]      # small entities inside/adjacent to hosts
    markers: tuple[RawObject, ...]     # singletons that indicate fill rules
    anchor: RawObject | None
    n_hosts: int
    n_slots: int


def detect_host_slot_decomp(grid: Grid, bg: int | None = None) -> HostSlotDecomposition | None:
    """Detect host-slot structure.

    Requires:
    - Composite motifs (centers with adjacent frame CCs)
    - At least 2 composites with consistent structure
    - An anchor singleton

    Returns None if the structure isn't clearly host-slot.
    """
    if bg is None:
        bg = detect_bg(grid)
    comp = decompose_composites(grid, bg)

    if len(comp.composites) < 2:
        return None
    if comp.anchor is None:
        return None

    # Hosts are the frame CCs of composites (the larger structures)
    hosts: list[RawObject] = []
    slots: list[RawObject] = []  # centers are the "slot indicators"
    seen_host_ids: set[int] = set()

    for motif in comp.composites:
        for frame in motif.frames:
            fid = id(frame)
            if fid not in seen_host_ids:
                seen_host_ids.add(fid)
                hosts.append(frame)
        slots.append(motif.center)

    # Markers: isolated singletons that aren't part of composites
    markers = list(comp.isolated)

    if not hosts:
        return None

    return HostSlotDecomposition(
        hosts=tuple(hosts),
        slots=tuple(slots),
        markers=tuple(markers),
        anchor=comp.anchor,
        n_hosts=len(hosts),
        n_slots=len(slots),
    )


# ---------------------------------------------------------------------------
# Multi-view decomposition hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompositionHypothesis:
    """A named decomposition view with confidence and metadata."""

    label: str            # "object", "panel", "partition", "region", "host_slot", "frame"
    confidence: float     # 0.0-1.0
    evidence: tuple[str, ...]
    data: object | None = None  # the decomposition-specific data structure


def propose_decompositions(grid: Grid, bg: int | None = None) -> list[DecompositionHypothesis]:
    """Propose all applicable decomposition views for a grid.

    Returns a ranked list of hypotheses. Multiple views can apply
    simultaneously — downstream code picks the most useful one.
    """
    if bg is None:
        bg = detect_bg(grid)

    hypotheses: list[DecompositionHypothesis] = []

    # 1. Object decomposition (always)
    obj_decomp = decompose_objects(grid, bg)
    if obj_decomp.objects:
        hypotheses.append(DecompositionHypothesis(
            label="object",
            confidence=1.0,
            evidence=("connected_components",),
            data=obj_decomp,
        ))

    # 2. Partition (both axes, non-bg separators first, then bg)
    part = detect_partition_decomp(grid, bg)
    if part is not None:
        conf = 0.95 if part.separator_color != bg else 0.75
        hypotheses.append(DecompositionHypothesis(
            label="partition",
            confidence=conf,
            evidence=(
                f"sep_color={part.separator_color}",
                f"{part.n_rows}x{part.n_cols}_cells",
                "uniform" if part.uniform else "non_uniform",
            ),
            data=part,
        ))

    # 3. Panel (single-axis separators)
    panel = detect_panels(grid, bg)
    if panel is not None:
        # Don't emit panel if a stronger partition already found
        is_redundant = part is not None and part.separator_color == panel.separator_color
        if not is_redundant:
            hypotheses.append(DecompositionHypothesis(
                label="panel",
                confidence=0.85,
                evidence=(
                    f"axis={panel.axis}",
                    f"sep_color={panel.separator_color}",
                    f"{panel.n_panels}_panels",
                ),
                data=panel,
            ))

    # 4. Region (bordered rectangles)
    region = detect_region_decomp(grid, bg)
    if region is not None and region.n_regions >= 1:
        hypotheses.append(DecompositionHypothesis(
            label="region",
            confidence=0.80 if region.n_regions >= 2 else 0.60,
            evidence=(
                f"{region.n_regions}_regions",
                "uniform" if region.uniform_size else "varied",
            ),
            data=region,
        ))

    # 5. Frame (legacy framed regions — distinct from boxed regions)
    framed = detect_framed_regions(grid, bg)
    if framed:
        hypotheses.append(DecompositionHypothesis(
            label="frame",
            confidence=0.85,
            evidence=(f"{len(framed)}_framed_regions",),
            data=framed,
        ))

    # 6. Host-slot (composites with anchor)
    hs = detect_host_slot_decomp(grid, bg)
    if hs is not None:
        hypotheses.append(DecompositionHypothesis(
            label="host_slot",
            confidence=0.80,
            evidence=(
                f"{hs.n_hosts}_hosts",
                f"{hs.n_slots}_slots",
                "has_anchor",
            ),
            data=hs,
        ))

    # Sort by confidence, highest first
    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses


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
    # New multi-view hypotheses
    hypotheses: tuple[DecompositionHypothesis, ...] = ()


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
        hypotheses=tuple(propose_decompositions(grid, bg)),
    )
