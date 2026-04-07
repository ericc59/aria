"""Relation layer — correspondences, slot grids, and legend mappings.

This module bridges perception (which detects entities) and reasoning
(which needs to know how entities relate to each other).

It exposes reusable relation structures that stage-1, scene programs,
and the solver can consume:

- SlotGrid: a regular array of positions (detected from object layout,
  NOT from separator lines — that's PartitionScene)
- LegendMapping: parsed legend entries ready for color/shape lookup
- ZoneMapping: correspondence between zones and a target structure
- Correspondence: generic entity-to-entity mapping with an indexing key
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.decomposition import RawObject
from aria.types import DemoPair, Grid, LegendEntry, LegendInfo, Zone


# ---------------------------------------------------------------------------
# SlotGrid — regular array of positions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotAnchor:
    """One position in a slot grid."""

    row_idx: int
    col_idx: int
    row: int  # pixel row of slot top-left
    col: int  # pixel col of slot top-left
    height: int
    width: int


@dataclass(frozen=True)
class SlotGrid:
    """A regular array of equally-spaced positions.

    Unlike PartitionScene (which requires explicit separator lines),
    a SlotGrid is inferred from the spatial regularity of objects
    or repeated sub-patterns.
    """

    n_rows: int
    n_cols: int
    slot_height: int
    slot_width: int
    row_stride: int  # pixels between slot row starts
    col_stride: int  # pixels between slot col starts
    origin_row: int  # pixel row of the (0,0) slot
    origin_col: int  # pixel col of the (0,0) slot
    slots: tuple[SlotAnchor, ...]

    def slot_at(self, row_idx: int, col_idx: int) -> SlotAnchor | None:
        for s in self.slots:
            if s.row_idx == row_idx and s.col_idx == col_idx:
                return s
        return None

    def extract_slot(self, grid: Grid, row_idx: int, col_idx: int) -> Grid | None:
        s = self.slot_at(row_idx, col_idx)
        if s is None:
            return None
        return grid[s.row : s.row + s.height, s.col : s.col + s.width].copy()


def detect_slot_grid(state: GridPerceptionState) -> SlotGrid | None:
    """Detect a regular array from object positions.

    Looks for objects whose top-left corners form a grid pattern
    with consistent row/col strides.
    """
    # Try non-singletons first (more interesting), then all objects
    candidate_lists = [
        state.objects.non_singletons,
        state.objects8.non_singletons,
        list(state.objects.objects),
        list(state.objects8.objects),
    ]
    for objs in candidate_lists:
        if len(objs) < 4:
            continue
        result = _detect_slot_grid_from_objects(objs, state)
        if result is not None:
            return result
    return None


def _detect_slot_grid_from_objects(
    objects: Sequence[RawObject],
    state: GridPerceptionState,
) -> SlotGrid | None:
    if len(objects) < 4:
        return None

    # Collect unique row and col positions
    rows_set: set[int] = set()
    cols_set: set[int] = set()
    for obj in objects:
        rows_set.add(obj.row)
        cols_set.add(obj.col)

    rows_sorted = sorted(rows_set)
    cols_sorted = sorted(cols_set)

    if len(rows_sorted) < 2 or len(cols_sorted) < 2:
        return None

    # Check for consistent row stride
    row_diffs = [rows_sorted[i + 1] - rows_sorted[i] for i in range(len(rows_sorted) - 1)]
    col_diffs = [cols_sorted[i + 1] - cols_sorted[i] for i in range(len(cols_sorted) - 1)]

    if not row_diffs or not col_diffs:
        return None

    row_stride = row_diffs[0]
    col_stride = col_diffs[0]

    if row_stride < 1 or col_stride < 1:
        return None

    # Allow small tolerance: all diffs must equal the stride
    if not all(d == row_stride for d in row_diffs):
        return None
    if not all(d == col_stride for d in col_diffs):
        return None

    n_rows = len(rows_sorted)
    n_cols = len(cols_sorted)

    # Verify most grid positions are occupied
    position_set = {(obj.row, obj.col) for obj in objects}
    expected = {(r, c) for r in rows_sorted for c in cols_sorted}
    occupied = expected & position_set
    if len(occupied) < len(expected) * 0.5:
        return None

    # Determine slot size from object bboxes at occupied positions
    heights: list[int] = []
    widths: list[int] = []
    for obj in objects:
        if (obj.row, obj.col) in expected:
            heights.append(obj.bbox_h)
            widths.append(obj.bbox_w)

    if not heights or not widths:
        return None

    # Use the most common bbox size as slot size
    slot_h = max(set(heights), key=heights.count)
    slot_w = max(set(widths), key=widths.count)

    # Build slot anchors
    row_to_idx = {r: i for i, r in enumerate(rows_sorted)}
    col_to_idx = {c: i for i, c in enumerate(cols_sorted)}
    slots: list[SlotAnchor] = []
    for ri, r in enumerate(rows_sorted):
        for ci, c in enumerate(cols_sorted):
            slots.append(SlotAnchor(
                row_idx=ri,
                col_idx=ci,
                row=r,
                col=c,
                height=slot_h,
                width=slot_w,
            ))

    return SlotGrid(
        n_rows=n_rows,
        n_cols=n_cols,
        slot_height=slot_h,
        slot_width=slot_w,
        row_stride=row_stride,
        col_stride=col_stride,
        origin_row=rows_sorted[0],
        origin_col=cols_sorted[0],
        slots=tuple(slots),
    )


# ---------------------------------------------------------------------------
# LegendMapping — parsed legend for lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegendMapping:
    """A parsed legend ready for color lookup.

    Wraps LegendInfo with bidirectional lookup and region exclusion.
    """

    info: LegendInfo
    key_to_value: Mapping[int, int]
    value_to_key: Mapping[int, int]
    legend_bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1)
    edge: str

    def lookup(self, key_color: int) -> int | None:
        return self.key_to_value.get(key_color)

    def reverse_lookup(self, value_color: int) -> int | None:
        return self.value_to_key.get(value_color)

    def is_in_legend_region(self, row: int, col: int) -> bool:
        r0, c0, r1, c1 = self.legend_bbox
        return r0 <= row <= r1 and c0 <= col <= c1


def build_legend_mapping(state: GridPerceptionState) -> LegendMapping | None:
    """Build a LegendMapping from perception state."""
    if state.legend is None:
        return None
    info = state.legend
    k2v: dict[int, int] = {}
    v2k: dict[int, int] = {}
    for entry in info.entries:
        k2v[entry.key_color] = entry.value_color
        v2k[entry.value_color] = entry.key_color
    return LegendMapping(
        info=info,
        key_to_value=k2v,
        value_to_key=v2k,
        legend_bbox=info.region_bbox,
        edge=info.edge,
    )


# ---------------------------------------------------------------------------
# ZoneMapping — correspondence between zones
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneMapping:
    """Maps zones to an output structure.

    zone_order: indices into state.zones in the order they map to output.
    mapping_kind: how zones relate to output ('summary_grid', 'select_one',
                  'boolean_combine', 'overlay').
    """

    zone_order: tuple[int, ...]
    mapping_kind: str
    params: Mapping[str, object] = field(default_factory=dict)


def detect_zone_summary_grid(
    state: GridPerceptionState,
    output: Grid,
) -> ZoneMapping | None:
    """Check if output is a summary grid where each cell derives from a zone.

    The output has shape (zone_rows, zone_cols) where zone_rows * zone_cols
    equals the number of content zones, and each output cell is a single
    color derived from the corresponding zone.

    Tries multiple zone property extractors until one verifies.
    """
    zones = state.zones
    if len(zones) < 2:
        return None

    oH, oW = output.shape

    # Filter zones that are actual content (not the whole grid)
    content_zones = [z for z in zones if z.h * z.w < state.dims[0] * state.dims[1]]
    if len(content_zones) < 2:
        return None

    n = len(content_zones)

    # Check if output shape matches zone count in some layout
    layouts: list[tuple[int, int]] = []
    for r in range(1, n + 1):
        if n % r == 0:
            c = n // r
            layouts.append((r, c))

    for lr, lc in layouts:
        if (oH, oW) != (lr, lc):
            continue
        zone_order = _match_zone_order_to_grid_layout(content_zones, lr, lc)
        if zone_order is None:
            continue
        ordered_zones = [content_zones[zi] for zi in zone_order]

        # Try each property extractor
        for prop_name, extractor in _ZONE_PROPERTY_EXTRACTORS:
            predicted = np.zeros((lr, lc), dtype=output.dtype)
            valid = True
            for idx, z in enumerate(ordered_zones):
                r_idx, c_idx = divmod(idx, lc)
                val = extractor(z.grid, state.bg_color)
                if val is None:
                    valid = False
                    break
                predicted[r_idx, c_idx] = val
            if valid and np.array_equal(predicted, output):
                return ZoneMapping(
                    zone_order=tuple(zone_order),
                    mapping_kind="summary_grid",
                    params={"property": prop_name, "rows": lr, "cols": lc},
                )
    return None


def _zone_prop_dominant_non_bg(grid: Grid, bg: int) -> int | None:
    """Most common non-bg color, or bg if zone is all background."""
    return _dominant_non_bg_color(grid, bg) or bg


def _zone_prop_unique_non_bg(grid: Grid, bg: int) -> int | None:
    """The unique non-bg color if exactly one exists, else None."""
    non_bg = set(int(v) for v in grid.ravel() if int(v) != bg)
    if len(non_bg) == 1:
        return non_bg.pop()
    return None


def _zone_prop_minority_color(grid: Grid, bg: int) -> int | None:
    """The least common non-bg color (the 'odd one out')."""
    flat = grid.ravel()
    non_bg = flat[flat != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[int(np.argmin(counts))])


def _zone_prop_non_bg_pixel_count(grid: Grid, bg: int) -> int | None:
    """Count of non-bg pixels."""
    return int(np.sum(grid != bg))


def _zone_prop_unique_color_count(grid: Grid, bg: int) -> int | None:
    """Count of distinct non-bg colors."""
    non_bg = set(int(v) for v in grid.ravel() if int(v) != bg)
    return len(non_bg)


def _zone_prop_has_non_bg(grid: Grid, bg: int) -> int | None:
    """1 if zone has any non-bg pixel, 0 otherwise."""
    return 1 if np.any(grid != bg) else 0


def _zone_prop_has_multiple_non_bg(grid: Grid, bg: int) -> int | None:
    """1 if zone has 2+ non-bg pixels, 0 otherwise."""
    return 1 if int(np.sum(grid != bg)) >= 2 else 0


def _zone_prop_is_fully_non_bg(grid: Grid, bg: int) -> int | None:
    """1 if every pixel is non-bg, 0 otherwise."""
    return 1 if not np.any(grid == bg) else 0


def _zone_prop_majority_is_non_bg(grid: Grid, bg: int) -> int | None:
    """1 if more than half the pixels are non-bg, 0 otherwise."""
    total = grid.size
    non_bg = int(np.sum(grid != bg))
    return 1 if non_bg > total // 2 else 0


_ZONE_PROPERTY_EXTRACTORS: list[tuple[str, object]] = [
    ("dominant_non_bg_color", _zone_prop_dominant_non_bg),
    ("unique_non_bg_color", _zone_prop_unique_non_bg),
    ("minority_color", _zone_prop_minority_color),
    ("non_bg_pixel_count", _zone_prop_non_bg_pixel_count),
    ("unique_color_count", _zone_prop_unique_color_count),
    ("has_non_bg", _zone_prop_has_non_bg),
    ("has_multiple_non_bg", _zone_prop_has_multiple_non_bg),
    ("is_fully_non_bg", _zone_prop_is_fully_non_bg),
    ("majority_is_non_bg", _zone_prop_majority_is_non_bg),
]


def _match_zone_order_to_grid_layout(
    zones: list[Zone],
    n_rows: int,
    n_cols: int,
) -> list[int] | None:
    """Order zones by their spatial position to match a row-major grid."""
    if len(zones) != n_rows * n_cols:
        return None
    indexed = list(enumerate(zones))
    indexed.sort(key=lambda item: (item[1].y, item[1].x))
    return [i for i, _ in indexed]


def _dominant_non_bg_color(grid: Grid, bg: int) -> int | None:
    """Find the most common non-background color in a grid."""
    flat = grid.ravel()
    non_bg = flat[flat != bg]
    if len(non_bg) == 0:
        return None
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# Correspondence — generic entity mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrespondenceEntry:
    """One mapping in a correspondence."""

    source_id: str  # e.g. "zone:0", "object:3", "slot:1,2"
    target_id: str  # e.g. "legend:blue", "zone:1", "output_cell:0,1"
    key: object = None  # indexing key (color, position, etc.)


@dataclass(frozen=True)
class Correspondence:
    """A set of entity-to-entity mappings with a named indexing scheme.

    Examples:
    - slot_to_legend: slot grid positions map to legend entries by color
    - zone_to_zone: input zones map to output zones by position
    - object_to_slot: objects map to slot grid positions
    """

    kind: str  # 'slot_to_legend', 'zone_to_zone', 'object_to_slot', etc.
    entries: tuple[CorrespondenceEntry, ...]
    index_key: str  # what's used for matching: 'color', 'position', 'order'
    params: Mapping[str, object] = field(default_factory=dict)


def build_zone_to_zone_correspondence(
    input_state: GridPerceptionState,
    output_state: GridPerceptionState,
) -> Correspondence | None:
    """Build positional correspondence between input and output zones.

    Matches zones by spatial order when both grids have the same
    zone count.
    """
    in_zones = input_state.zones
    out_zones = output_state.zones
    if len(in_zones) < 2 or len(in_zones) != len(out_zones):
        return None

    # Sort by position
    in_sorted = sorted(enumerate(in_zones), key=lambda item: (item[1].y, item[1].x))
    out_sorted = sorted(enumerate(out_zones), key=lambda item: (item[1].y, item[1].x))

    entries: list[CorrespondenceEntry] = []
    for (ii, _), (oi, _) in zip(in_sorted, out_sorted):
        entries.append(CorrespondenceEntry(
            source_id=f"zone:{ii}",
            target_id=f"zone:{oi}",
            key=(ii, oi),
        ))

    return Correspondence(
        kind="zone_to_zone",
        entries=tuple(entries),
        index_key="position",
    )


def build_object_to_slot_correspondence(
    state: GridPerceptionState,
    slot_grid: SlotGrid,
) -> Correspondence | None:
    """Map objects to their nearest slot grid position."""
    objects = list(state.objects.objects)
    if not objects or not slot_grid.slots:
        return None

    entries: list[CorrespondenceEntry] = []
    for obj_idx, obj in enumerate(objects):
        best_slot = None
        best_dist = float("inf")
        for slot in slot_grid.slots:
            dist = abs(obj.row - slot.row) + abs(obj.col - slot.col)
            if dist < best_dist:
                best_dist = dist
                best_slot = slot
        if best_slot is not None and best_dist <= max(slot_grid.row_stride, slot_grid.col_stride):
            entries.append(CorrespondenceEntry(
                source_id=f"object:{obj_idx}",
                target_id=f"slot:{best_slot.row_idx},{best_slot.col_idx}",
                key=(best_slot.row_idx, best_slot.col_idx),
            ))

    if not entries:
        return None

    return Correspondence(
        kind="object_to_slot",
        entries=tuple(entries),
        index_key="position",
    )


# ---------------------------------------------------------------------------
# Multi-demo verification helpers
# ---------------------------------------------------------------------------


def detect_partition_summary_grid(
    state: GridPerceptionState,
    output: Grid,
) -> ZoneMapping | None:
    """Check if output is a summary grid over partition cells.

    Each output cell (r, c) derives from the partition cell at (r, c)
    via a scalar property (dominant color, count, etc.).
    """
    if state.partition is None:
        return None

    p = state.partition
    oH, oW = output.shape
    if (oH, oW) != (p.n_rows, p.n_cols):
        return None

    # Build cell grid indexed by (row_idx, col_idx)
    cell_map: dict[tuple[int, int], object] = {}
    for cell in p.cells:
        cell_map[(cell.row_idx, cell.col_idx)] = cell

    if len(cell_map) != p.n_rows * p.n_cols:
        return None

    # Try each property extractor
    for prop_name, extractor in _ZONE_PROPERTY_EXTRACTORS:
        predicted = np.zeros((oH, oW), dtype=output.dtype)
        valid = True
        for ri in range(p.n_rows):
            for ci in range(p.n_cols):
                cell = cell_map.get((ri, ci))
                if cell is None:
                    valid = False
                    break
                r0, c0, r1, c1 = cell.bbox
                cell_grid = state.grid[r0 : r1 + 1, c0 : c1 + 1]
                val = extractor(cell_grid, state.bg_color)
                if val is None:
                    valid = False
                    break
                predicted[ri, ci] = val
            if not valid:
                break
        if valid and np.array_equal(predicted, output):
            return ZoneMapping(
                zone_order=(),
                mapping_kind="partition_summary_grid",
                params={"property": prop_name, "source": "partition"},
            )

    return None


def verify_zone_summary_grid(
    demos: tuple[DemoPair, ...],
) -> ZoneMapping | None:
    """Check if zone/partition summary grid relation holds across all demos.

    Tries partition-based summary first (more reliable), then zone-based.
    """
    if not demos:
        return None

    # Try partition-based summary
    first_state = perceive_grid(demos[0].input)
    mapping = detect_partition_summary_grid(first_state, demos[0].output)
    if mapping is not None:
        # Verify on remaining demos
        all_match = True
        for demo in demos[1:]:
            state = perceive_grid(demo.input)
            m2 = detect_partition_summary_grid(state, demo.output)
            if m2 is None or m2.params.get("property") != mapping.params.get("property"):
                all_match = False
                break
        if all_match:
            return mapping

    # Try zone-based summary
    mapping = detect_zone_summary_grid(first_state, demos[0].output)
    if mapping is None:
        return None
    for demo in demos[1:]:
        state = perceive_grid(demo.input)
        m2 = detect_zone_summary_grid(state, demo.output)
        if m2 is None or m2.mapping_kind != mapping.mapping_kind:
            return None
        if m2.params.get("property") != mapping.params.get("property"):
            return None
    return mapping


def verify_slot_grid_consistent(
    demos: tuple[DemoPair, ...],
) -> SlotGrid | None:
    """Check if all demos have a consistent slot grid structure."""
    if not demos:
        return None
    grids: list[SlotGrid] = []
    for demo in demos:
        state = perceive_grid(demo.input)
        sg = detect_slot_grid(state)
        if sg is None:
            return None
        grids.append(sg)
    # Check consistency: same n_rows, n_cols
    first = grids[0]
    if all(g.n_rows == first.n_rows and g.n_cols == first.n_cols for g in grids):
        return first
    return None
