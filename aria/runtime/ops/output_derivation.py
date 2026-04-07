"""Stage-1 direct-derivation runtime op.

This op executes a verified OutputDerivationSpec as a normal runtime step.
It is intentionally narrow: exact clone/interior/border extraction from a
selected input candidate.
"""

from __future__ import annotations

import numpy as np

from aria.core.output_derivation import (
    CODE_TO_SELECTOR,
    decode_output_derivation_spec,
    predict_output_derivation,
)
from aria.core.grid_perception import perceive_grid
from aria.core.output_size import (
    _object_decomposition_for_connectivity,
    _salient_solid_rectangles,
    _selected_object_for_params,
)
from aria.runtime.ops import OpSignature, register
from aria.types import Grid, Type


def _derive_output_from_input(
    grid: Grid,
    kind_code: int,
    relation_code: int,
    selector_code: int,
    arg0: int,
    arg1: int,
    arg2: int,
) -> Grid:
    spec = decode_output_derivation_spec(
        kind_code,
        relation_code,
        selector_code,
        arg0,
        arg1,
        arg2,
    )
    if spec is None:
        return grid.copy()
    result = predict_output_derivation(spec, grid)
    return grid.copy() if result is None else result


def _render_scaled_selected_object(
    grid: Grid,
    connectivity: int,
    selector_code: int,
    rank: int,
    row_scale: int,
    col_scale: int,
    selector_arg: int,
) -> Grid:
    selector = CODE_TO_SELECTOR.get(selector_code)
    if selector is None or row_scale < 1 or col_scale < 1:
        return grid.copy()
    params: dict[str, object] = {"connectivity": connectivity, "rank": rank}
    if selector == "color_bbox_area_desc":
        params["color"] = selector_arg
    state = perceive_grid(grid)
    selected = _selected_object_for_params(state, selector, params)
    if selected is None:
        return grid.copy()
    template = np.full((int(selected.bbox_h), int(selected.bbox_w)), state.bg_color, dtype=grid.dtype)
    template[selected.mask] = int(selected.color)
    return np.repeat(np.repeat(template, row_scale, axis=0), col_scale, axis=1)


def _render_marker_stacked_selected_object(
    grid: Grid,
    connectivity: int,
    selector_code: int,
    rank: int,
    selector_arg: int,
) -> Grid:
    selector = CODE_TO_SELECTOR.get(selector_code)
    if selector is None:
        return grid.copy()
    params: dict[str, object] = {"connectivity": connectivity, "rank": rank}
    if selector == "color_bbox_area_desc":
        params["color"] = selector_arg
    state = perceive_grid(grid)
    selected = _selected_object_for_params(state, selector, params)
    if selected is None:
        return grid.copy()

    decomp = _object_decomposition_for_connectivity(state, params)
    markers = [
        obj for obj in decomp.singletons
        if not (
            obj.color == selected.color
            and obj.row == selected.row
            and obj.col == selected.col
            and obj.size == selected.size
        )
    ]
    if not markers:
        return grid.copy()

    rows = {int(obj.row) for obj in markers}
    cols = {int(obj.col) for obj in markers}
    template = np.full((int(selected.bbox_h), int(selected.bbox_w)), state.bg_color, dtype=grid.dtype)
    template[selected.mask] = int(selected.color)

    if len(cols) == 1 and len(rows) == len(markers):
        ordered = sorted(markers, key=lambda obj: (int(obj.row), int(obj.col), int(obj.color)))
        out = np.full((int(selected.bbox_h) * len(ordered), int(selected.bbox_w)), state.bg_color, dtype=grid.dtype)
        for idx, marker in enumerate(ordered):
            block = np.full_like(template, state.bg_color)
            block[selected.mask] = int(marker.color)
            r0 = idx * int(selected.bbox_h)
            out[r0:r0 + int(selected.bbox_h), :] = block
        return out

    if len(rows) == 1 and len(cols) == len(markers):
        ordered = sorted(markers, key=lambda obj: (int(obj.col), int(obj.row), int(obj.color)))
        out = np.full((int(selected.bbox_h), int(selected.bbox_w) * len(ordered)), state.bg_color, dtype=grid.dtype)
        for idx, marker in enumerate(ordered):
            block = np.full_like(template, state.bg_color)
            block[selected.mask] = int(marker.color)
            c0 = idx * int(selected.bbox_w)
            out[:, c0:c0 + int(selected.bbox_w)] = block
        return out

    return grid.copy()


def _render_solid_rectangle_layout(grid: Grid) -> Grid:
    state = perceive_grid(grid)
    rectangles = _salient_solid_rectangles(state)
    if not rectangles:
        return grid.copy()

    row_groups = _interval_groups(
        [(int(obj.row), int(obj.row + obj.bbox_h - 1)) for obj in rectangles]
    )
    col_groups = _interval_groups(
        [(int(obj.col), int(obj.col + obj.bbox_w - 1)) for obj in rectangles]
    )
    if not row_groups or not col_groups:
        return grid.copy()

    out = np.full((len(row_groups), len(col_groups)), state.bg_color, dtype=grid.dtype)
    for obj in rectangles:
        row_idx = _find_group_index((int(obj.row), int(obj.row + obj.bbox_h - 1)), row_groups)
        col_idx = _find_group_index((int(obj.col), int(obj.col + obj.bbox_w - 1)), col_groups)
        if row_idx is None or col_idx is None:
            return grid.copy()
        out[row_idx, col_idx] = int(obj.color)
    return out


def _apply_tile_transform(grid: Grid, transform_code: int) -> Grid:
    if transform_code == 0:
        return grid
    if transform_code == 1:
        return np.fliplr(grid).astype(np.uint8).copy()
    if transform_code == 2:
        return np.flipud(grid).astype(np.uint8).copy()
    if transform_code == 3:
        return np.rot90(grid, k=2).astype(np.uint8).copy()
    return grid


def _render_tiled_input_pattern(
    grid: Grid,
    row_repeat: int,
    col_repeat: int,
    odd_row_transform: int,
    odd_col_transform: int,
) -> Grid:
    if row_repeat < 1 or col_repeat < 1:
        return grid.copy()

    row_blocks: list[Grid] = []
    for row_idx in range(row_repeat):
        col_blocks: list[Grid] = []
        for col_idx in range(col_repeat):
            tile = grid
            if row_idx % 2 == 1:
                tile = _apply_tile_transform(tile, odd_row_transform)
            if col_idx % 2 == 1:
                tile = _apply_tile_transform(tile, odd_col_transform)
            col_blocks.append(tile)
        row_blocks.append(np.hstack(col_blocks).astype(np.uint8))
    return np.vstack(row_blocks).astype(np.uint8)


def _interval_groups(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans)
    if not ordered:
        return []
    groups: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return [(start, end) for start, end in groups]


def _find_group_index(span: tuple[int, int], groups: list[tuple[int, int]]) -> int | None:
    start, end = span
    for idx, (group_start, group_end) in enumerate(groups):
        if start >= group_start and end <= group_end:
            return idx
    return None


register(
    "derive_output_from_input",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("kind_code", Type.INT),
            ("relation_code", Type.INT),
            ("selector_code", Type.INT),
            ("arg0", Type.INT),
            ("arg1", Type.INT),
            ("arg2", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _derive_output_from_input,
)

register(
    "render_scaled_selected_object",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("connectivity", Type.INT),
            ("selector_code", Type.INT),
            ("rank", Type.INT),
            ("row_scale", Type.INT),
            ("col_scale", Type.INT),
            ("selector_arg", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _render_scaled_selected_object,
)

register(
    "render_marker_stacked_selected_object",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("connectivity", Type.INT),
            ("selector_code", Type.INT),
            ("rank", Type.INT),
            ("selector_arg", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _render_marker_stacked_selected_object,
)

register(
    "render_solid_rectangle_layout",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _render_solid_rectangle_layout,
)

register(
    "render_tiled_input_pattern",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("row_repeat", Type.INT),
            ("col_repeat", Type.INT),
            ("odd_row_transform", Type.INT),
            ("odd_col_transform", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _render_tiled_input_pattern,
)
