"""Shared grid perception state for stage-1 reasoning.

This module is intentionally structural, not task-specific. It exposes
reusable facts about a single grid so stage-1 reasoners like output-size
and output-background inference do not each re-implement their own scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from aria.decomposition import (
    FramedRegion,
    ObjectDecomposition,
    decompose_objects,
    detect_bg,
    detect_boxed_regions,
    detect_framed_regions,
    detect_partition_decomp,
    detect_panels,
)
from aria.graph.partition import detect_partition
from aria.graph.zones import find_zones
from aria.types import Grid, LegendInfo, PartitionScene, RoleBinding, Zone


Interval = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class GridPerceptionState:
    grid: Grid
    dims: tuple[int, int]
    bg_color: int
    palette: frozenset[int]
    non_bg_colors: frozenset[int]
    color_pixel_counts: Mapping[int, int]
    color_bboxes: Mapping[int, BBox]
    objects: ObjectDecomposition
    objects8: ObjectDecomposition
    partition: PartitionScene | None
    framed_regions: tuple[FramedRegion, ...]
    boxed_regions: tuple[FramedRegion, ...]
    uniform_rows_by_color: Mapping[int, tuple[int, ...]]
    uniform_cols_by_color: Mapping[int, tuple[int, ...]]
    bg_separator_rows: tuple[int, ...]
    bg_separator_cols: tuple[int, ...]
    row_intervals: tuple[Interval, ...]
    col_intervals: tuple[Interval, ...]
    tight_non_bg_bbox: BBox | None
    zones: tuple[Zone, ...]
    legend: LegendInfo | None
    roles: tuple[RoleBinding, ...]


def _detect_partition_extended(grid: Grid, bg: int) -> PartitionScene | None:
    """Extended partition detection: standard → bg-colored → panel-derived.

    Uses the new decomposition detectors that handle bg-colored separators
    and column/row-only panels, falling back to the standard detector.
    """
    # 1. Standard non-bg partition (fastest, most reliable)
    part = detect_partition(grid, background=bg)
    if part is not None:
        return part

    # 2. Extended: bg-colored separators + panel-derived
    decomp = detect_partition_decomp(grid, bg)
    if decomp is not None:
        return decomp.scene  # type: ignore[return-value]

    # 3. Panel-derived: build a synthetic PartitionScene from panels
    panel = detect_panels(grid, bg)
    if panel is not None and panel.n_panels >= 2:
        return _panels_to_partition_scene(panel, grid, bg)

    return None


def _panels_to_partition_scene(panel, grid: Grid, bg: int) -> PartitionScene:
    """Convert a PanelDecomposition into a PartitionScene.

    This allows downstream code that expects PartitionScene (scene_solve
    families 3-5, 7, 9-13) to consume single-axis panel decompositions.
    """
    from aria.types import PartitionCell

    cells = []
    cell_shapes = []
    for p in panel.panels:
        r0, c0 = p.row, p.col
        r1, c1 = r0 + p.height - 1, c0 + p.width - 1
        cell_shapes.append((p.height, p.width))
        cells.append(PartitionCell(
            row_idx=0 if panel.axis == "col" else p.index,
            col_idx=p.index if panel.axis == "col" else 0,
            bbox=(r0, c0, r1, c1),
            dims=(p.height, p.width),
            background=bg,
            palette=p.palette,
            obj_count=1,  # rough — at least non-empty
        ))

    n_rows = 1 if panel.axis == "col" else panel.n_panels
    n_cols = panel.n_panels if panel.axis == "col" else 1

    return PartitionScene(
        separator_color=panel.separator_color,
        separator_rows=() if panel.axis == "col" else panel.separator_positions,
        separator_cols=panel.separator_positions if panel.axis == "col" else (),
        cells=tuple(cells),
        n_rows=n_rows,
        n_cols=n_cols,
        cell_shapes=tuple(cell_shapes),
        is_uniform_partition=panel.uniform_panel_size,
    )


def perceive_grid(grid: Grid) -> GridPerceptionState:
    """Build a reusable structural view of a single grid."""
    bg = detect_bg(grid)
    palette = frozenset(int(value) for value in np.unique(grid))
    non_bg_colors = frozenset(color for color in palette if color != bg)
    color_pixel_counts = _color_pixel_counts(grid)
    color_bboxes = _color_bboxes(grid)
    objects = decompose_objects(grid, bg, connectivity=4)
    objects8 = decompose_objects(grid, bg, connectivity=8)
    partition = _detect_partition_extended(grid, bg)
    framed_regions = tuple(detect_framed_regions(grid, bg=bg))
    boxed_regions = tuple(detect_boxed_regions(grid, bg=bg))
    uniform_rows = _uniform_rows_by_color(grid)
    uniform_cols = _uniform_cols_by_color(grid)
    sep_rows = _background_separator_rows(grid, bg)
    sep_cols = _background_separator_cols(grid, bg)
    zones = tuple(find_zones(grid))
    legend = _detect_legend_safe(grid, bg, partition)
    roles = _infer_roles_safe(objects, partition, legend)
    return GridPerceptionState(
        grid=grid,
        dims=(int(grid.shape[0]), int(grid.shape[1])),
        bg_color=bg,
        palette=palette,
        non_bg_colors=non_bg_colors,
        color_pixel_counts=color_pixel_counts,
        color_bboxes=color_bboxes,
        objects=objects,
        objects8=objects8,
        partition=partition,
        framed_regions=framed_regions,
        boxed_regions=boxed_regions,
        uniform_rows_by_color=uniform_rows,
        uniform_cols_by_color=uniform_cols,
        bg_separator_rows=tuple(sep_rows),
        bg_separator_cols=tuple(sep_cols),
        row_intervals=tuple(_compute_intervals(sep_rows, int(grid.shape[0]))),
        col_intervals=tuple(_compute_intervals(sep_cols, int(grid.shape[1]))),
        tight_non_bg_bbox=_tight_non_bg_bbox(grid, bg),
        zones=zones,
        legend=legend,
        roles=roles,
    )


def _background_separator_rows(grid: Grid, bg: int) -> list[int]:
    rows = int(grid.shape[0])
    return [
        row
        for row in range(rows)
        if {int(value) for value in grid[row, :]} == {bg}
    ]


def _background_separator_cols(grid: Grid, bg: int) -> list[int]:
    cols = int(grid.shape[1])
    return [
        col
        for col in range(cols)
        if {int(value) for value in grid[:, col]} == {bg}
    ]


def _uniform_rows_by_color(grid: Grid) -> dict[int, tuple[int, ...]]:
    rows = int(grid.shape[0])
    by_color: dict[int, list[int]] = {}
    for row in range(rows):
        values = {int(value) for value in grid[row, :]}
        if len(values) != 1:
            continue
        color = next(iter(values))
        by_color.setdefault(color, []).append(row)
    return {color: tuple(indices) for color, indices in by_color.items()}


def _uniform_cols_by_color(grid: Grid) -> dict[int, tuple[int, ...]]:
    cols = int(grid.shape[1])
    by_color: dict[int, list[int]] = {}
    for col in range(cols):
        values = {int(value) for value in grid[:, col]}
        if len(values) != 1:
            continue
        color = next(iter(values))
        by_color.setdefault(color, []).append(col)
    return {color: tuple(indices) for color, indices in by_color.items()}


def _compute_intervals(separators: list[int], size: int) -> list[Interval]:
    seps = sorted(set(separators))
    if not seps:
        return [(0, size - 1)]

    intervals: list[Interval] = []
    if seps[0] > 0:
        intervals.append((0, seps[0] - 1))

    for idx in range(len(seps) - 1):
        start = seps[idx] + 1
        end = seps[idx + 1] - 1
        if start <= end:
            intervals.append((start, end))

    if seps[-1] < size - 1:
        intervals.append((seps[-1] + 1, size - 1))

    return intervals


def _tight_non_bg_bbox(grid: Grid, bg: int) -> BBox | None:
    rows, cols = grid.shape
    min_r = rows
    min_c = cols
    max_r = -1
    max_c = -1
    for r in range(rows):
        for c in range(cols):
            if int(grid[r, c]) == bg:
                continue
            if r < min_r:
                min_r = r
            if c < min_c:
                min_c = c
            if r > max_r:
                max_r = r
            if c > max_c:
                max_c = c
    if max_r < 0:
        return None
    return (int(min_r), int(min_c), int(max_r), int(max_c))


def _color_pixel_counts(grid: Grid) -> dict[int, int]:
    counts: dict[int, int] = {}
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            counts[color] = counts.get(color, 0) + 1
    return counts


def _color_bboxes(grid: Grid) -> dict[int, BBox]:
    rows, cols = grid.shape
    mins: dict[int, tuple[int, int]] = {}
    maxs: dict[int, tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if color not in mins:
                mins[color] = (r, c)
                maxs[color] = (r, c)
                continue
            min_r, min_c = mins[color]
            max_r, max_c = maxs[color]
            mins[color] = (min(min_r, r), min(min_c, c))
            maxs[color] = (max(max_r, r), max(max_c, c))
    return {
        color: (min_r, min_c, maxs[color][0], maxs[color][1])
        for color, (min_r, min_c) in mins.items()
    }


def _detect_legend_safe(
    grid: Grid,
    bg: int,
    partition: PartitionScene | None,
) -> LegendInfo | None:
    try:
        from aria.graph.legend import detect_legend
        return detect_legend(grid, background=bg, partition=partition)
    except Exception:
        return None


def _infer_roles_safe(
    objects: ObjectDecomposition,
    partition: PartitionScene | None,
    legend: LegendInfo | None,
) -> tuple[RoleBinding, ...]:
    try:
        from aria.graph.cc_label import label_4conn
        from aria.graph.roles import infer_roles
        from aria.graph.symmetry import detect_obj_symmetry
        from aria.types import ObjectNode, Shape

        nodes: list[ObjectNode] = []
        for idx, raw in enumerate(objects.objects):
            nodes.append(ObjectNode(
                id=idx,
                color=int(raw.color),
                mask=raw.mask,
                bbox=(int(raw.col), int(raw.row), int(raw.bbox_w), int(raw.bbox_h)),
                shape=Shape.IRREGULAR,
                symmetry=frozenset(),
                size=int(raw.size),
            ))
        return infer_roles(tuple(nodes), partition, legend)
    except Exception:
        return ()
