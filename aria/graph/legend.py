"""Legend/key-region detection for ARC grids."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from aria.graph.cc_label import label_4conn
from aria.types import Grid, LegendEntry, LegendInfo, ObjectNode, PartitionScene


def detect_legend(
    grid: Grid,
    *,
    background: int,
    partition: PartitionScene | None = None,
) -> LegendInfo | None:
    """Detect compact edge/separator key regions that define color mappings."""
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return None

    separator_rows = list(partition.separator_rows) if partition else _find_separator_rows(grid, background)
    separator_cols = list(partition.separator_cols) if partition else _find_separator_cols(grid, background)

    if separator_rows:
        for sep_row in separator_rows:
            top_region = grid[:sep_row, :]
            bottom_region = grid[sep_row + 1 :, :]
            if top_region.size and bottom_region.size:
                regions = [
                    (top_region, 0, 0, "top"),
                    (bottom_region, sep_row + 1, 0, "bottom"),
                ]
                regions.sort(key=lambda item: item[0].size)
                for region, offset_r, offset_c, edge in regions:
                    legend = _check_region(
                        region,
                        background,
                        offset_r,
                        offset_c,
                        edge,
                        full_shape=grid.shape,
                    )
                    if legend is not None:
                        return legend

    if separator_cols:
        for sep_col in separator_cols:
            left_region = grid[:, :sep_col]
            right_region = grid[:, sep_col + 1 :]
            if left_region.size and right_region.size:
                regions = [
                    (left_region, 0, 0, "left"),
                    (right_region, 0, sep_col + 1, "right"),
                ]
                regions.sort(key=lambda item: item[0].size)
                for region, offset_r, offset_c, edge in regions:
                    legend = _check_region(
                        region,
                        background,
                        offset_r,
                        offset_c,
                        edge,
                        full_shape=grid.shape,
                    )
                    if legend is not None:
                        return legend

    strip_h = min(3, rows // 3) if rows >= 6 else 0
    if strip_h > 0:
        for region, offset_r, edge in (
            (grid[:strip_h, :], 0, "top"),
            (grid[rows - strip_h :, :], rows - strip_h, "bottom"),
        ):
            legend = _check_region(
                region,
                background,
                offset_r,
                0,
                edge,
                full_shape=grid.shape,
            )
            if legend is not None:
                return legend

    strip_w = min(3, cols // 3) if cols >= 6 else 0
    if strip_w > 0:
        for region, offset_c, edge in (
            (grid[:, :strip_w], 0, "left"),
            (grid[:, cols - strip_w :], cols - strip_w, "right"),
        ):
            legend = _check_region(
                region,
                background,
                0,
                offset_c,
                edge,
                full_shape=grid.shape,
            )
            if legend is not None:
                return legend

    return None


def _find_separator_rows(grid: Grid, background: int) -> list[int]:
    rows, _cols = grid.shape
    return [
        row for row in range(rows)
        if len(set(int(value) for value in grid[row, :])) == 1
        and int(grid[row, 0]) != background
    ]


def _find_separator_cols(grid: Grid, background: int) -> list[int]:
    _rows, cols = grid.shape
    return [
        col for col in range(cols)
        if len(set(int(value) for value in grid[:, col])) == 1
        and int(grid[0, col]) != background
    ]


def _check_region(
    region: Grid,
    background: int,
    offset_r: int,
    offset_c: int,
    edge: str,
    *,
    full_shape: tuple[int, int],
) -> LegendInfo | None:
    if region.size == 0:
        return None

    full_area = full_shape[0] * full_shape[1]
    region_area = int(region.shape[0] * region.shape[1])
    if region_area > full_area / 2:
        return None

    objects = label_4conn(region, ignore_color=background)
    if len(objects) < 2:
        return None

    entries = _find_row_pairs(objects)
    if len(entries) < 2:
        entries = _find_col_pairs(objects)
    if len(entries) < 2:
        return None
    if 2 * len(entries) < max(2, int(np.ceil(len(objects) * 0.75))):
        return None

    rows, cols = region.shape
    return LegendInfo(
        region_bbox=(offset_r, offset_c, offset_r + rows - 1, offset_c + cols - 1),
        entries=tuple(entries),
        edge=edge,
    )


def _find_row_pairs(objects: list[ObjectNode]) -> list[LegendEntry]:
    grouped: dict[int, list[ObjectNode]] = defaultdict(list)
    for obj in objects:
        centroid_r = obj.bbox[1] + obj.bbox[3] / 2.0
        grouped[round(centroid_r)].append(obj)

    entries: list[LegendEntry] = []
    for _row, group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        group.sort(key=lambda obj: obj.bbox[0])
        left, right = group
        if left.color != right.color:
            entries.append(LegendEntry(
                key_color=int(left.color),
                value_color=int(right.color),
            ))
    return entries


def _find_col_pairs(objects: list[ObjectNode]) -> list[LegendEntry]:
    grouped: dict[int, list[ObjectNode]] = defaultdict(list)
    for obj in objects:
        centroid_c = obj.bbox[0] + obj.bbox[2] / 2.0
        grouped[round(centroid_c)].append(obj)

    entries: list[LegendEntry] = []
    for _col, group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        group.sort(key=lambda obj: obj.bbox[1])
        top, bottom = group
        if top.color != bottom.color:
            entries.append(LegendEntry(
                key_color=int(top.color),
                value_color=int(bottom.color),
            ))
    return entries
