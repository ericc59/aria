"""Stage-1 output-size inference.

This module is intentionally narrow:

- infer output size only
- verify the hypothesis on every train demo for the task
- fail closed if no size rule verifies

No background inference. No object actions. No rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.types import DemoPair, Grid


Size = tuple[int, int]


MODE_SAME_AS_INPUT = "same_as_input"
MODE_TRANSPOSE_INPUT = "transpose_input"
MODE_SCALE_INPUT = "scale_input"
MODE_SCALE_INPUT_BY_PALETTE_SIZE = "scale_input_by_palette_size"
MODE_ADDITIVE_INPUT = "additive_input"
MODE_SQUARE_CANVAS_SCALED = "square_canvas_scaled"
MODE_ROW_STRIP_SQUARE_FROM_COLS = "row_strip_square_from_cols"
MODE_ROW_STRIP_HALF_COLS_BY_COLS = "row_strip_half_cols_by_cols"
MODE_ROW_STRIP_SQUARE_BY_NON_BG_PIXEL_COUNT = "row_strip_square_by_non_bg_pixel_count"
MODE_PRESERVE_INPUT_ROWS_EXPAND_COLS_BY_ROWS_MINUS_ONE = "preserve_input_rows_expand_cols_by_rows_minus_one"
MODE_EXPAND_ROWS_TO_FOUR_TIMES_MINUS_THREE_PRESERVE_COLS = "expand_rows_to_four_times_minus_three_preserve_cols"
MODE_SEPARATOR_CELL_SIZE = "separator_cell_size"
MODE_SEPARATOR_PANEL_SIZE = "separator_panel_size"
MODE_SELECTED_PARTITION_CELL_SIZE = "selected_partition_cell_size"
MODE_DOMINANT_SHAPE_TOP_ROW_SPAN = "dominant_shape_top_row_span"
MODE_PARTITION_GRID_SHAPE = "partition_grid_shape"
MODE_OBJECT_POSITION_GRID_SHAPE = "object_position_grid_shape"
MODE_FRAME_INTERIOR_SIZE = "frame_interior_size"
MODE_SELECTED_BOXED_REGION_INTERIOR = "selected_boxed_region_interior"
MODE_SELECTED_STRIP_BLOCK_SIZE = "selected_strip_block_size"
MODE_SOLID_RECTANGLE_LAYOUT_SHAPE = "solid_rectangle_layout_shape"
MODE_SCALED_BBOX_OF_SELECTED_OBJECT = "scaled_bbox_of_selected_object"
MODE_MARKER_STACKED_SELECTED_OBJECT = "marker_stacked_selected_object"
MODE_TIGHT_BBOX_OF_NON_BG = "tight_bbox_of_non_bg"
MODE_BBOX_OF_SELECTED_OBJECT = "bbox_of_selected_object"
MODE_BBOX_OF_SELECTED_COLOR = "bbox_of_selected_color"
MODE_NON_BG_COLOR_COUNT_STRIP = "non_bg_color_count_strip"
MODE_PRESERVE_INPUT_ROWS_NON_BG_COLS = "preserve_input_rows_non_bg_cols"
MODE_CONSTANT_ROWS_PRESERVE_INPUT_COLS = "constant_rows_preserve_input_cols"
MODE_FIXED_OUTPUT_DIMS = "fixed_output_dims"


@dataclass(frozen=True)
class OutputSizeSpec:
    mode: str
    params: Mapping[str, object] = field(default_factory=dict)
    rationale: str = ""


def predict_output_size(spec: OutputSizeSpec, grid: Grid) -> Size | None:
    return predict_output_size_from_state(spec, perceive_grid(grid))


def predict_output_size_from_state(
    spec: OutputSizeSpec,
    state: GridPerceptionState,
) -> Size | None:
    rows, cols = state.dims

    if spec.mode == MODE_SAME_AS_INPUT:
        return (rows, cols)

    if spec.mode == MODE_TRANSPOSE_INPUT:
        return (cols, rows)

    if spec.mode == MODE_SCALE_INPUT:
        row_ratio = spec.params.get("row_ratio")
        col_ratio = spec.params.get("col_ratio")
        if not isinstance(row_ratio, Fraction) or not isinstance(col_ratio, Fraction):
            return None
        if (rows * row_ratio.numerator) % row_ratio.denominator != 0:
            return None
        if (cols * col_ratio.numerator) % col_ratio.denominator != 0:
            return None
        return (
            (rows * row_ratio.numerator) // row_ratio.denominator,
            (cols * col_ratio.numerator) // col_ratio.denominator,
        )

    if spec.mode == MODE_SCALE_INPUT_BY_PALETTE_SIZE:
        factor = len(state.palette)
        return (rows * factor, cols * factor)

    if spec.mode == MODE_ADDITIVE_INPUT:
        row_delta = spec.params.get("row_delta")
        col_delta = spec.params.get("col_delta")
        if not isinstance(row_delta, int) or not isinstance(col_delta, int):
            return None
        return (max(1, rows + row_delta), max(1, cols + col_delta))

    if spec.mode == MODE_SQUARE_CANVAS_SCALED:
        source = spec.params.get("factor_source")
        if not isinstance(source, str):
            return None
        return _square_canvas_scaled_size(state, source)

    if spec.mode == MODE_ROW_STRIP_SQUARE_FROM_COLS:
        return _row_strip_square_from_cols_size(state)

    if spec.mode == MODE_ROW_STRIP_HALF_COLS_BY_COLS:
        return _row_strip_half_cols_by_cols_size(state)

    if spec.mode == MODE_ROW_STRIP_SQUARE_BY_NON_BG_PIXEL_COUNT:
        return _row_strip_square_by_non_bg_pixel_count_size(state)

    if spec.mode == MODE_PRESERVE_INPUT_ROWS_EXPAND_COLS_BY_ROWS_MINUS_ONE:
        return _preserve_input_rows_expand_cols_by_rows_minus_one_size(state)

    if spec.mode == MODE_EXPAND_ROWS_TO_FOUR_TIMES_MINUS_THREE_PRESERVE_COLS:
        return _expand_rows_to_four_times_minus_three_preserve_cols_size(state)

    if spec.mode == MODE_SEPARATOR_CELL_SIZE:
        return _separator_cell_shape(state)

    if spec.mode == MODE_SEPARATOR_PANEL_SIZE:
        return _separator_panel_shape(state)

    if spec.mode == MODE_SELECTED_PARTITION_CELL_SIZE:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _selected_partition_cell_size(state, selector, spec.params)

    if spec.mode == MODE_DOMINANT_SHAPE_TOP_ROW_SPAN:
        return _dominant_shape_top_row_span_size(state, spec.params)

    if spec.mode == MODE_PARTITION_GRID_SHAPE:
        if state.partition is None:
            return None
        return (int(state.partition.n_rows), int(state.partition.n_cols))

    if spec.mode == MODE_OBJECT_POSITION_GRID_SHAPE:
        anchor = spec.params.get("anchor")
        if not isinstance(anchor, str):
            return None
        return _object_position_grid_shape(state, anchor)

    if spec.mode == MODE_FRAME_INTERIOR_SIZE:
        if not state.framed_regions:
            return None
        selector = spec.params.get("selector")
        if isinstance(selector, str):
            return _selected_frame_interior_size(state, selector, spec.params)
        select = spec.params.get("select", "largest")
        if select == "largest":
            region = max(state.framed_regions, key=lambda r: r.height * r.width)
            return (int(region.height), int(region.width))
        return None

    if spec.mode == MODE_SELECTED_BOXED_REGION_INTERIOR:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _selected_boxed_region_interior_size(state, selector, spec.params)

    if spec.mode == MODE_SELECTED_STRIP_BLOCK_SIZE:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _selected_strip_block_size(state, selector, spec.params)

    if spec.mode == MODE_SOLID_RECTANGLE_LAYOUT_SHAPE:
        return _solid_rectangle_layout_shape_size(state)

    if spec.mode == MODE_SCALED_BBOX_OF_SELECTED_OBJECT:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _scaled_selected_object_bbox_size(state, selector, spec.params)

    if spec.mode == MODE_MARKER_STACKED_SELECTED_OBJECT:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _marker_stacked_selected_object_size(state, selector, spec.params)

    if spec.mode == MODE_TIGHT_BBOX_OF_NON_BG:
        return _tight_bbox_size(state)

    if spec.mode == MODE_BBOX_OF_SELECTED_OBJECT:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _selected_object_bbox_size(state, selector, spec.params)

    if spec.mode == MODE_BBOX_OF_SELECTED_COLOR:
        selector = spec.params.get("selector")
        if not isinstance(selector, str):
            return None
        return _selected_color_bbox_size(state, selector, spec.params)

    if spec.mode == MODE_NON_BG_COLOR_COUNT_STRIP:
        orientation = spec.params.get("orientation")
        if orientation == "horizontal":
            return _non_bg_color_count_strip_size(state, horizontal=True)
        if orientation == "vertical":
            return _non_bg_color_count_strip_size(state, horizontal=False)
        return None

    if spec.mode == MODE_PRESERVE_INPUT_ROWS_NON_BG_COLS:
        return _preserve_input_rows_non_bg_cols_size(state)

    if spec.mode == MODE_CONSTANT_ROWS_PRESERVE_INPUT_COLS:
        rows_constant = spec.params.get("rows")
        if not isinstance(rows_constant, int):
            return None
        return (rows_constant, cols)

    if spec.mode == MODE_FIXED_OUTPUT_DIMS:
        dims = spec.params.get("dims")
        if (
            isinstance(dims, tuple)
            and len(dims) == 2
            and isinstance(dims[0], int)
            and isinstance(dims[1], int)
        ):
            return (dims[0], dims[1])
        return None

    return None


def verify_output_size_spec(spec: OutputSizeSpec, demos: tuple[DemoPair, ...]) -> bool:
    if not demos:
        return False
    return all(
        predict_output_size(spec, demo.input) == _grid_size(demo.output)
        for demo in demos
    )


def infer_verified_output_size_specs(demos: tuple[DemoPair, ...]) -> tuple[OutputSizeSpec, ...]:
    candidates = (
        _candidate_same_as_input(demos),
        _candidate_transpose_input(demos),
        _candidate_scale_input_by_palette_size(demos),
        _candidate_square_canvas_scaled(demos),
        _candidate_row_strip_square_from_cols(demos),
        _candidate_row_strip_half_cols_by_cols(demos),
        _candidate_row_strip_square_by_non_bg_pixel_count(demos),
        _candidate_preserve_input_rows_expand_cols_by_rows_minus_one(demos),
        _candidate_expand_rows_to_four_times_minus_three_preserve_cols(demos),
        _candidate_selected_partition_cell_size(demos),
        _candidate_separator_cell_size(demos),
        _candidate_separator_panel_size(demos),
        _candidate_partition_grid_shape(demos),
        _candidate_frame_interior_size(demos),
        _candidate_selected_boxed_region_interior(demos),
        _candidate_selected_strip_block_size(demos),
        _candidate_tight_bbox_of_non_bg(demos),
        _candidate_bbox_of_selected_object(demos),
        _candidate_bbox_of_selected_color(demos),
        _candidate_object_position_grid_shape(demos),
        _candidate_solid_rectangle_layout_shape(demos),
        _candidate_dominant_shape_top_row_span(demos),
        _candidate_non_bg_color_count_strip(demos),
        _candidate_scale_input(demos),
        _candidate_additive_input(demos),
        _candidate_preserve_input_rows_non_bg_cols(demos),
        _candidate_constant_rows_preserve_input_cols(demos),
        _candidate_fixed_output_dims(demos),
        _candidate_scaled_bbox_of_selected_object(demos),
        _candidate_marker_stacked_selected_object(demos),
    )
    verified: list[OutputSizeSpec] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if verify_output_size_spec(candidate, demos):
            verified.append(candidate)
    return tuple(verified)


def infer_output_size_spec(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    specs = infer_verified_output_size_specs(demos)
    return specs[0] if specs else None


def _candidate_same_as_input(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if demos and all(_grid_size(d.input) == _grid_size(d.output) for d in demos):
        return OutputSizeSpec(
            mode=MODE_SAME_AS_INPUT,
            rationale="output dims equal input dims on every demo",
        )
    return None


def _candidate_transpose_input(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if demos and all(_grid_size(d.output) == (d.input.shape[1], d.input.shape[0]) for d in demos):
        return OutputSizeSpec(
            mode=MODE_TRANSPOSE_INPUT,
            rationale="output dims are transposed input dims on every demo",
        )
    return None


def _candidate_scale_input(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    row_ratios: set[Fraction] = set()
    col_ratios: set[Fraction] = set()
    for demo in demos:
        in_rows, in_cols = _grid_size(demo.input)
        out_rows, out_cols = _grid_size(demo.output)
        if in_rows == 0 or in_cols == 0:
            return None
        row_ratios.add(Fraction(out_rows, in_rows))
        col_ratios.add(Fraction(out_cols, in_cols))
    if len(row_ratios) == 1 and len(col_ratios) == 1:
        row_ratio = next(iter(row_ratios))
        col_ratio = next(iter(col_ratios))
        if row_ratio == 1 and col_ratio == 1:
            return None
        if row_ratio.denominator != 1 or col_ratio.denominator != 1:
            return None
        return OutputSizeSpec(
            mode=MODE_SCALE_INPUT,
            params={"row_ratio": row_ratio, "col_ratio": col_ratio},
            rationale=f"output dims scale input by ({row_ratio}, {col_ratio})",
        )
    return None


def _candidate_scale_input_by_palette_size(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    if all(
        _scale_input_by_palette_size_size(perceive_grid(demo.input)) == _grid_size(demo.output)
        for demo in demos
    ):
        return OutputSizeSpec(
            mode=MODE_SCALE_INPUT_BY_PALETTE_SIZE,
            rationale="output dims equal input dims scaled by palette size on every demo",
        )
    return None


def _candidate_additive_input(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    row_deltas = {_grid_size(d.output)[0] - _grid_size(d.input)[0] for d in demos}
    col_deltas = {_grid_size(d.output)[1] - _grid_size(d.input)[1] for d in demos}
    if len(row_deltas) == 1 and len(col_deltas) == 1:
        row_delta = next(iter(row_deltas))
        col_delta = next(iter(col_deltas))
        if row_delta == 0 and col_delta == 0:
            return None
        return OutputSizeSpec(
            mode=MODE_ADDITIVE_INPUT,
            params={"row_delta": row_delta, "col_delta": col_delta},
            rationale=f"output dims add ({row_delta}, {col_delta}) to input dims",
        )
    return None


def _candidate_square_canvas_scaled(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    for source in ("input_side", "non_bg_color_count", "bg_pixel_count", "non_bg_pixel_count"):
        if all(
            _square_canvas_scaled_size(perceive_grid(demo.input), source) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_SQUARE_CANVAS_SCALED,
                params={"factor_source": source},
                rationale=f"output dims equal square canvas scaled by {source} on every demo",
            )
    return None


def _candidate_separator_cell_size(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        shape = _separator_cell_shape(perceive_grid(demo.input))
        if shape is None or shape != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_SEPARATOR_CELL_SIZE,
        rationale="output dims equal uniform separator-cell dims on every demo",
    )


def _candidate_separator_panel_size(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        shape = _separator_panel_shape(perceive_grid(demo.input))
        if shape is None or shape != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_SEPARATOR_PANEL_SIZE,
        rationale="output dims equal separator panel (cell + border) dims on every demo",
    )


def _candidate_selected_partition_cell_size(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _partition_cell_size_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_partition_cell_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_SELECTED_PARTITION_CELL_SIZE,
                params=params,
                rationale=f"output dims equal selected partition cell ({selector}) on every demo",
            )
    return None


def _candidate_dominant_shape_top_row_span(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    # Cheap gate: only relevant when output dims differ from input.
    if all(_grid_size(d.input) == _grid_size(d.output) for d in demos):
        return None
    for axis in (0, 1):
        for rank in range(3):
            params: dict[str, object] = {"axis": axis, "rank": rank}
            if all(
                _dominant_shape_top_row_span_size(perceive_grid(demo.input), params) == _grid_size(demo.output)
                for demo in demos
            ):
                axis_label = "row" if axis == 0 else "col"
                return OutputSizeSpec(
                    mode=MODE_DOMINANT_SHAPE_TOP_ROW_SPAN,
                    params=params,
                    rationale=f"output dims equal the {axis_label} rank-{rank} span of the dominant repeated object shape on every demo",
                )
    return None


def _candidate_row_strip_square_from_cols(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _row_strip_square_from_cols_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_ROW_STRIP_SQUARE_FROM_COLS,
        rationale="output dims form a square whose side equals input cols for every 1-row demo",
    )


def _candidate_row_strip_half_cols_by_cols(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _row_strip_half_cols_by_cols_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_ROW_STRIP_HALF_COLS_BY_COLS,
        rationale="output dims equal (input cols / 2, input cols) for every even-width 1-row demo",
    )


def _candidate_row_strip_square_by_non_bg_pixel_count(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _row_strip_square_by_non_bg_pixel_count_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_ROW_STRIP_SQUARE_BY_NON_BG_PIXEL_COUNT,
        rationale="output dims form a square whose side equals input cols times non-background pixel count for every 1-row demo",
    )


def _candidate_preserve_input_rows_expand_cols_by_rows_minus_one(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _preserve_input_rows_expand_cols_by_rows_minus_one_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_PRESERVE_INPUT_ROWS_EXPAND_COLS_BY_ROWS_MINUS_ONE,
        rationale="output dims preserve input rows and expand cols by (input rows - 1) on every demo",
    )


def _candidate_expand_rows_to_four_times_minus_three_preserve_cols(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _expand_rows_to_four_times_minus_three_preserve_cols_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_EXPAND_ROWS_TO_FOUR_TIMES_MINUS_THREE_PRESERVE_COLS,
        rationale="output dims expand rows to (4 * input rows - 3) while preserving cols on every demo",
    )


def _candidate_partition_grid_shape(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        state = perceive_grid(demo.input)
        if state.partition is None:
            return None
        if (int(state.partition.n_rows), int(state.partition.n_cols)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_PARTITION_GRID_SHAPE,
        rationale="output dims equal partition grid shape on every demo",
    )


def _candidate_object_position_grid_shape(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    # Cheap gate: output must be smaller than input (summary-grid pattern)
    in_h, in_w = _grid_size(demos[0].input)
    out_h, out_w = _grid_size(demos[0].output)
    if out_h >= in_h and out_w >= in_w:
        return None
    anchors = (
        "top_left",
        "count_square",
        "count_by_input_cols",
        "input_rows_by_count",
        "singleton_count_square",
        "max_height_by_count",
        "count_by_max_width",
        "max_height_by_max_width",
    )
    for anchor in anchors:
        if all(
            _object_position_grid_shape(perceive_grid(demo.input), anchor) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_OBJECT_POSITION_GRID_SHAPE,
                params={"anchor": anchor},
                rationale=f"output dims equal object-position grid shape ({anchor}) on every demo",
            )
    return None


def _candidate_frame_interior_size(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _frame_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_frame_interior_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_FRAME_INTERIOR_SIZE,
                params=params,
                rationale=f"output dims equal interior of selected frame ({selector}) on every demo",
            )
    return None


def _candidate_selected_boxed_region_interior(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _boxed_region_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_boxed_region_interior_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_SELECTED_BOXED_REGION_INTERIOR,
                params=params,
                rationale=f"output dims equal interior of selected boxed region ({selector}) on every demo",
            )
    return None


def _candidate_selected_strip_block_size(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _strip_block_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_strip_block_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_SELECTED_STRIP_BLOCK_SIZE,
                params=params,
                rationale=f"output dims equal selected strip block ({selector}) on every demo",
            )
    return None


def _candidate_solid_rectangle_layout_shape(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    if all(
        _solid_rectangle_layout_shape_size(perceive_grid(demo.input)) == _grid_size(demo.output)
        for demo in demos
    ):
        return OutputSizeSpec(
            mode=MODE_SOLID_RECTANGLE_LAYOUT_SHAPE,
            rationale="output dims equal row/col layout of salient solid rectangles on every demo",
        )
    return None


def _candidate_scaled_bbox_of_selected_object(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    # Cheap gate: skip when output == input dims (same_as_input tasks).
    # Scaled bbox is only relevant when the output is a different size.
    if all(_grid_size(d.input) == _grid_size(d.output) for d in demos):
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _scaled_object_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _scaled_selected_object_bbox_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            connectivity = params["connectivity"]
            row_scale = params["row_scale"]
            col_scale = params["col_scale"]
            return OutputSizeSpec(
                mode=MODE_SCALED_BBOX_OF_SELECTED_OBJECT,
                params=params,
                rationale=(
                    f"output dims equal bbox of selected object ({selector}) with "
                    f"{connectivity}-connected scale ({row_scale}, {col_scale}) on every demo"
                ),
            )
    return None


def _candidate_marker_stacked_selected_object(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    # Cheap gate: skip same-dims tasks (stacking changes output size)
    if all(_grid_size(d.input) == _grid_size(d.output) for d in demos):
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _marker_stacked_object_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _marker_stacked_selected_object_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            connectivity = params["connectivity"]
            return OutputSizeSpec(
                mode=MODE_MARKER_STACKED_SELECTED_OBJECT,
                params=params,
                rationale=(
                    f"output dims equal bbox of selected object ({selector}) tiled by singleton markers "
                    f"under {connectivity}-connected marker alignment on every demo"
                ),
            )
    return None


def _candidate_tight_bbox_of_non_bg(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _tight_bbox_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_TIGHT_BBOX_OF_NON_BG,
        rationale="output dims equal tight bbox of non-background input content",
    )


def _candidate_fixed_output_dims(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    dims = {_grid_size(d.output) for d in demos}
    if len(dims) == 1:
        fixed = next(iter(dims))
        return OutputSizeSpec(
            mode=MODE_FIXED_OUTPUT_DIMS,
            params={"dims": fixed},
            rationale=f"all demos share constant output dims {fixed}",
        )
    return None


def _candidate_preserve_input_rows_non_bg_cols(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    for demo in demos:
        if _preserve_input_rows_non_bg_cols_size(perceive_grid(demo.input)) != _grid_size(demo.output):
            return None
    return OutputSizeSpec(
        mode=MODE_PRESERVE_INPUT_ROWS_NON_BG_COLS,
        rationale="output dims preserve input rows and collapse width to non-background columns on every demo",
    )


def _candidate_constant_rows_preserve_input_cols(
    demos: tuple[DemoPair, ...],
) -> OutputSizeSpec | None:
    if not demos:
        return None
    output_rows = {_grid_size(demo.output)[0] for demo in demos}
    if len(output_rows) != 1:
        return None
    rows_constant = next(iter(output_rows))
    if all((rows_constant, demo.input.shape[1]) == _grid_size(demo.output) for demo in demos):
        return OutputSizeSpec(
            mode=MODE_CONSTANT_ROWS_PRESERVE_INPUT_COLS,
            params={"rows": rows_constant},
            rationale=f"output dims preserve input cols with constant row count {rows_constant} on every demo",
        )
    return None


def _candidate_bbox_of_selected_object(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _object_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_object_bbox_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_BBOX_OF_SELECTED_OBJECT,
                params=params,
                rationale=f"output dims equal bbox of selected object ({selector}) on every demo",
            )
    return None


def _candidate_non_bg_color_count_strip(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    horizontal_ok = all(
        _non_bg_color_count_strip_size(perceive_grid(demo.input), horizontal=True) == _grid_size(demo.output)
        for demo in demos
    )
    if horizontal_ok:
        return OutputSizeSpec(
            mode=MODE_NON_BG_COLOR_COUNT_STRIP,
            params={"orientation": "horizontal"},
            rationale="output dims equal horizontal strip of non-background color count on every demo",
        )
    vertical_ok = all(
        _non_bg_color_count_strip_size(perceive_grid(demo.input), horizontal=False) == _grid_size(demo.output)
        for demo in demos
    )
    if vertical_ok:
        return OutputSizeSpec(
            mode=MODE_NON_BG_COLOR_COUNT_STRIP,
            params={"orientation": "vertical"},
            rationale="output dims equal vertical strip of non-background color count on every demo",
        )
    return None


def _candidate_bbox_of_selected_color(demos: tuple[DemoPair, ...]) -> OutputSizeSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = _grid_size(demos[0].output)
    candidates = _color_selector_candidates(first_state, target)
    for params in candidates:
        selector = params["selector"]
        if all(
            _selected_color_bbox_size(perceive_grid(demo.input), selector, params) == _grid_size(demo.output)
            for demo in demos
        ):
            return OutputSizeSpec(
                mode=MODE_BBOX_OF_SELECTED_COLOR,
                params=params,
                rationale=f"output dims equal bbox of selected color ({selector}) on every demo",
            )
    return None


def _grid_size(grid: Grid) -> Size:
    return (int(grid.shape[0]), int(grid.shape[1]))


def _separator_cell_shape(state: GridPerceptionState) -> Size | None:
    if state.partition is not None and state.partition.is_uniform_partition and state.partition.cell_shapes:
        if len(set(state.partition.cell_shapes)) == 1:
            shape = state.partition.cell_shapes[0]
            return (int(shape[0]), int(shape[1]))

    rows, cols = state.dims
    for color in sorted(set(state.uniform_rows_by_color) | set(state.uniform_cols_by_color)):
        row_intervals = _compute_intervals(list(state.uniform_rows_by_color.get(color, ())), rows)
        col_intervals = _compute_intervals(list(state.uniform_cols_by_color.get(color, ())), cols)
        if len(row_intervals) * len(col_intervals) < 2:
            continue
        heights = {r1 - r0 + 1 for r0, r1 in row_intervals}
        widths = {c1 - c0 + 1 for c0, c1 in col_intervals}
        if len(heights) == 1 and len(widths) == 1:
            return (next(iter(heights)), next(iter(widths)))
    return None


def _separator_panel_shape(state: GridPerceptionState) -> Size | None:
    """Panel size = cell + one separator border on each side."""
    cell = _separator_cell_shape(state)
    if cell is None:
        return None
    cell_h, cell_w = cell
    rows, cols = state.dims

    has_row_seps = False
    has_col_seps = False
    for color in sorted(set(state.uniform_rows_by_color) | set(state.uniform_cols_by_color)):
        row_seps = sorted(state.uniform_rows_by_color.get(color, ()))
        col_seps = sorted(state.uniform_cols_by_color.get(color, ()))
        if len(row_seps) >= 2:
            stride = row_seps[1] - row_seps[0]
            if all(row_seps[j + 1] - row_seps[j] == stride for j in range(len(row_seps) - 1)):
                if stride - 1 == cell_h or stride == cell_h + 1:
                    has_row_seps = True
        if len(col_seps) >= 2:
            stride = col_seps[1] - col_seps[0]
            if all(col_seps[j + 1] - col_seps[j] == stride for j in range(len(col_seps) - 1)):
                if stride - 1 == cell_w or stride == cell_w + 1:
                    has_col_seps = True

    panel_h = cell_h + 2 if has_row_seps else rows
    panel_w = cell_w + 2 if has_col_seps else cols
    if (panel_h, panel_w) == (rows, cols):
        return None
    return (panel_h, panel_w)


def _square_canvas_scaled_size(state: GridPerceptionState, source: str) -> Size | None:
    rows, cols = state.dims
    if rows != cols:
        return None
    side = rows
    if source == "input_side":
        factor = side
    elif source == "non_bg_color_count":
        factor = len(state.non_bg_colors)
    elif source == "bg_pixel_count":
        total = rows * cols
        non_bg = sum(v for c, v in state.color_pixel_counts.items() if c != state.bg_color)
        factor = total - non_bg
    elif source == "non_bg_pixel_count":
        factor = sum(v for c, v in state.color_pixel_counts.items() if c != state.bg_color)
    else:
        return None
    if factor <= 0:
        return None
    return (side * factor, side * factor)


def _scale_input_by_palette_size_size(state: GridPerceptionState) -> Size:
    rows, cols = state.dims
    factor = len(state.palette)
    return (rows * factor, cols * factor)


def _row_strip_square_from_cols_size(state: GridPerceptionState) -> Size | None:
    rows, cols = state.dims
    if rows != 1:
        return None
    return (cols, cols)


def _row_strip_half_cols_by_cols_size(state: GridPerceptionState) -> Size | None:
    rows, cols = state.dims
    if rows != 1 or cols % 2 != 0:
        return None
    return (cols // 2, cols)


def _row_strip_square_by_non_bg_pixel_count_size(state: GridPerceptionState) -> Size | None:
    rows, cols = state.dims
    if rows != 1:
        return None
    non_bg_pixels = sum(
        count
        for color, count in state.color_pixel_counts.items()
        if color != state.bg_color
    )
    return (cols * non_bg_pixels, cols * non_bg_pixels)


def _preserve_input_rows_expand_cols_by_rows_minus_one_size(
    state: GridPerceptionState,
) -> Size:
    rows, cols = state.dims
    return (rows, cols + rows - 1)


def _expand_rows_to_four_times_minus_three_preserve_cols_size(
    state: GridPerceptionState,
) -> Size:
    rows, cols = state.dims
    return (4 * rows - 3, cols)


def _tight_bbox_size(state: GridPerceptionState) -> Size | None:
    if state.tight_non_bg_bbox is None:
        return None
    min_r, min_c, max_r, max_c = state.tight_non_bg_bbox
    return (max_r - min_r + 1, max_c - min_c + 1)


def _scaled_selected_object_bbox_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object] | None = None,
) -> Size | None:
    if params is None:
        params = {}
    row_scale = params.get("row_scale")
    col_scale = params.get("col_scale")
    if not isinstance(row_scale, int) or not isinstance(col_scale, int):
        return None
    if row_scale < 1 or col_scale < 1:
        return None
    bbox = _selected_object_bbox_size(state, selector, params)
    if bbox is None:
        return None
    return (bbox[0] * row_scale, bbox[1] * col_scale)


def _marker_stacked_selected_object_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object] | None = None,
) -> Size | None:
    if params is None:
        params = {}
    bbox = _selected_object_bbox_size(state, selector, params)
    if bbox is None:
        return None
    decomp = _object_decomposition_for_connectivity(state, params)
    selected = _selected_object_for_params(state, selector, params)
    if selected is None:
        return None

    markers = [
        obj for obj in decomp.singletons
        if not (obj.color == selected.color and obj.row == selected.row and obj.col == selected.col and obj.size == selected.size)
    ]
    if not markers:
        return None
    rows = {int(obj.row) for obj in markers}
    cols = {int(obj.col) for obj in markers}
    count = len(markers)
    if len(cols) == 1 and len(rows) == count:
        return (bbox[0] * count, bbox[1])
    if len(rows) == 1 and len(cols) == count:
        return (bbox[0], bbox[1] * count)
    return None


def _selected_object_bbox_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object] | None = None,
) -> Size | None:
    selected = _selected_object_for_params(state, selector, params)
    if selected is None:
        return None
    return (int(selected.bbox_h), int(selected.bbox_w))


def _object_position_grid_shape(state: GridPerceptionState, anchor: str) -> Size | None:
    objects = [obj for obj in state.objects.objects if obj.size > 1]
    if not objects:
        objects = list(state.objects.objects)
    if not objects:
        return None

    if anchor == "top_left":
        rows = {int(obj.row) for obj in objects}
        cols = {int(obj.col) for obj in objects}
        return (len(rows), len(cols))

    n = len(objects)
    if anchor == "count_square":
        return (n, n)
    if anchor == "count_by_input_cols":
        return (n, state.dims[1])
    if anchor == "input_rows_by_count":
        return (state.dims[0], n)

    n_sing = len(state.objects.singletons)
    if anchor == "singleton_count_square" and n_sing > 0:
        return (n_sing, n_sing)

    if objects:
        max_h = max(int(o.bbox_h) for o in objects)
        max_w = max(int(o.bbox_w) for o in objects)
        if anchor == "max_height_by_count":
            return (max_h, n)
        if anchor == "count_by_max_width":
            return (n, max_w)
        if anchor == "max_height_by_max_width":
            return (max_h, max_w)

    return None


def _partition_cell_size_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    cells = _partition_cells(state)
    if not cells:
        return ()

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for selector, ordered in (
        ("row_major", cells),
        ("col_major", sorted(cells, key=lambda cell: (cell[2], cell[0], cell[3] - cell[2], cell[1] - cell[0]))),
    ):
        for index, cell in enumerate(ordered):
            r0, r1, c0, c1 = cell
            dims = (r1 - r0 + 1, c1 - c0 + 1)
            if dims != target:
                continue
            key = (selector, index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": selector, "index": index})
    return tuple(candidates)


def _selected_partition_cell_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> Size | None:
    cell = _select_partition_cell_bounds(state, selector, params)
    if cell is None:
        return None
    r0, r1, c0, c1 = cell
    return (r1 - r0 + 1, c1 - c0 + 1)


def _selected_color_bbox_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object] | None = None,
) -> Size | None:
    colors = [color for color in state.non_bg_colors if color in state.color_bboxes]
    if not colors:
        return None

    if params is None:
        params = {}

    def bbox_size(color: int) -> Size:
        min_r, min_c, max_r, max_c = state.color_bboxes[color]
        return (max_r - min_r + 1, max_c - min_c + 1)

    # Legacy selectors (backward compat)
    if selector == "min_pixel_count":
        selected = min(
            colors,
            key=lambda color: (
                state.color_pixel_counts.get(color, 0),
                bbox_size(color)[0] * bbox_size(color)[1],
                color,
            ),
        )
        return bbox_size(selected)

    if selector == "min_bbox_area":
        selected = min(
            colors,
            key=lambda color: (
                bbox_size(color)[0] * bbox_size(color)[1],
                bbox_size(color)[0],
                bbox_size(color)[1],
                state.color_pixel_counts.get(color, 0),
                color,
            ),
        )
        return bbox_size(selected)

    if selector == "max_pixel_count":
        selected = max(
            colors,
            key=lambda color: (
                state.color_pixel_counts.get(color, 0),
                -(bbox_size(color)[0] * bbox_size(color)[1]),
                -color,
            ),
        )
        return bbox_size(selected)

    if selector == "max_bbox_area":
        selected = max(
            colors,
            key=lambda color: (
                bbox_size(color)[0] * bbox_size(color)[1],
                state.color_pixel_counts.get(color, 0),
                -color,
            ),
        )
        return bbox_size(selected)

    # Rank-based selectors
    rank = params.get("rank")
    if not isinstance(rank, int):
        return None

    if selector == "pixel_count_desc":
        ordered = sorted(
            colors,
            key=lambda c: (state.color_pixel_counts.get(c, 0), bbox_size(c)[0] * bbox_size(c)[1], -c),
            reverse=True,
        )
    elif selector == "pixel_count_asc":
        ordered = sorted(
            colors,
            key=lambda c: (state.color_pixel_counts.get(c, 0), bbox_size(c)[0] * bbox_size(c)[1], c),
        )
    elif selector == "bbox_area_desc":
        ordered = sorted(
            colors,
            key=lambda c: (bbox_size(c)[0] * bbox_size(c)[1], state.color_pixel_counts.get(c, 0), -c),
            reverse=True,
        )
    elif selector == "bbox_area_asc":
        ordered = sorted(
            colors,
            key=lambda c: (bbox_size(c)[0] * bbox_size(c)[1], state.color_pixel_counts.get(c, 0), c),
        )
    else:
        return None

    if not (0 <= rank < len(ordered)):
        return None
    return bbox_size(ordered[rank])


def _frame_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    if not state.framed_regions:
        return ()

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    by_area_desc = sorted(
        state.framed_regions,
        key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
        reverse=True,
    )
    by_area_asc = sorted(
        state.framed_regions,
        key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
    )

    for selector, ordered in (("area_desc", by_area_desc), ("area_asc", by_area_asc)):
        for rank, region in enumerate(ordered):
            if (region.height, region.width) != target:
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": selector, "rank": rank})

    for frame_color in sorted({region.frame_color for region in state.framed_regions}):
        pool = [r for r in by_area_desc if r.frame_color == frame_color]
        if not pool:
            continue
        region = pool[0]
        if (region.height, region.width) != target:
            continue
        key = ("frame_color_largest", frame_color)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"selector": "frame_color_largest", "frame_color": frame_color})

    return tuple(candidates)


def _selected_frame_interior_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> Size | None:
    regions = list(state.framed_regions)
    if not regions:
        return None

    if selector == "area_desc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
            reverse=True,
        )
        if not (0 <= rank < len(ordered)):
            return None
        region = ordered[rank]
        return (int(region.height), int(region.width))

    if selector == "area_asc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
        )
        if not (0 <= rank < len(ordered)):
            return None
        region = ordered[rank]
        return (int(region.height), int(region.width))

    if selector == "frame_color_largest":
        frame_color = params.get("frame_color")
        if not isinstance(frame_color, int):
            return None
        pool = [region for region in regions if region.frame_color == frame_color]
        if not pool:
            return None
        region = max(
            pool,
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col),
        )
        return (int(region.height), int(region.width))

    return None


def _object_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    objects = list(state.objects.objects)
    non_singletons = [obj for obj in objects if obj.size > 1]
    pool = non_singletons if non_singletons else objects
    if not pool:
        return ()

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for selector, key_fn, reverse in (
        ("bbox_area_desc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col), True),
        ("bbox_area_asc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col), False),
        ("pixel_count_desc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col), True),
        ("pixel_count_asc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col), False),
    ):
        ordered = sorted(pool, key=key_fn, reverse=reverse)
        for rank, obj in enumerate(ordered):
            if (int(obj.bbox_h), int(obj.bbox_w)) != target:
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": selector, "rank": rank})

    for color in sorted({int(obj.color) for obj in pool}):
        ordered = sorted(
            [obj for obj in pool if int(obj.color) == color],
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
            reverse=True,
        )
        for rank, obj in enumerate(ordered):
            if (int(obj.bbox_h), int(obj.bbox_w)) != target:
                continue
            key = ("color_bbox_area_desc", color, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": "color_bbox_area_desc", "color": color, "rank": rank})

    candidates.sort(key=lambda c: c.get("rank", 0))
    return tuple(candidates)


def _scaled_object_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    target_h, target_w = target
    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for connectivity in (4, 8):
        decomp = _object_decomposition_for_connectivity(state, {"connectivity": connectivity})
        objects = list(decomp.objects)
        non_singletons = [obj for obj in objects if obj.size > 1]
        pool = non_singletons if non_singletons else objects
        if not pool:
            continue

        for selector, key_fn, reverse in (
            ("bbox_area_desc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col), True),
            ("bbox_area_asc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col), False),
            ("pixel_count_desc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col), True),
            ("pixel_count_asc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col), False),
        ):
            ordered = sorted(pool, key=key_fn, reverse=reverse)
            for rank, obj in enumerate(ordered):
                if target_h % int(obj.bbox_h) != 0 or target_w % int(obj.bbox_w) != 0:
                    continue
                row_scale = target_h // int(obj.bbox_h)
                col_scale = target_w // int(obj.bbox_w)
                if row_scale == 1 and col_scale == 1:
                    continue
                key = (connectivity, selector, rank, row_scale, col_scale)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "connectivity": connectivity,
                        "selector": selector,
                        "rank": rank,
                        "row_scale": row_scale,
                        "col_scale": col_scale,
                    }
                )

        for color in sorted({int(obj.color) for obj in pool}):
            ordered = sorted(
                [obj for obj in pool if int(obj.color) == color],
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
                reverse=True,
            )
            for rank, obj in enumerate(ordered):
                if target_h % int(obj.bbox_h) != 0 or target_w % int(obj.bbox_w) != 0:
                    continue
                row_scale = target_h // int(obj.bbox_h)
                col_scale = target_w // int(obj.bbox_w)
                if row_scale == 1 and col_scale == 1:
                    continue
                key = (connectivity, "color_bbox_area_desc", color, rank, row_scale, col_scale)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "connectivity": connectivity,
                        "selector": "color_bbox_area_desc",
                        "color": color,
                        "rank": rank,
                        "row_scale": row_scale,
                        "col_scale": col_scale,
                    }
                )

    candidates.sort(key=lambda c: (int(c["row_scale"]), int(c["col_scale"]), int(c.get("rank", 0))))
    return tuple(candidates)


def _marker_stacked_object_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for connectivity in (4, 8):
        decomp = _object_decomposition_for_connectivity(state, {"connectivity": connectivity})
        objects = list(decomp.objects)
        non_singletons = [obj for obj in objects if obj.size > 1]
        pool = non_singletons if non_singletons else objects
        singleton_count = len(decomp.singletons)
        if not pool or singleton_count == 0:
            continue
        for selector, key_fn, reverse in (
            ("bbox_area_desc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col), True),
            ("bbox_area_asc", lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col), False),
            ("pixel_count_desc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col), True),
            ("pixel_count_asc", lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col), False),
        ):
            ordered = sorted(pool, key=key_fn, reverse=reverse)
            for rank, obj in enumerate(ordered):
                if singleton_count <= 1:
                    continue
                rows = {int(marker.row) for marker in decomp.singletons if marker != obj}
                cols = {int(marker.col) for marker in decomp.singletons if marker != obj}
                count = len([marker for marker in decomp.singletons if marker != obj])
                if count <= 0:
                    continue
                if len(cols) == 1 and len(rows) == count:
                    predicted = (int(obj.bbox_h) * count, int(obj.bbox_w))
                elif len(rows) == 1 and len(cols) == count:
                    predicted = (int(obj.bbox_h), int(obj.bbox_w) * count)
                else:
                    continue
                if predicted != target:
                    continue
                key = (connectivity, selector, rank)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "connectivity": connectivity,
                        "selector": selector,
                        "rank": rank,
                    }
                )

        for color in sorted({int(obj.color) for obj in pool}):
            ordered = sorted(
                [obj for obj in pool if int(obj.color) == color],
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
                reverse=True,
            )
            for rank, obj in enumerate(ordered):
                rows = {int(marker.row) for marker in decomp.singletons if marker != obj}
                cols = {int(marker.col) for marker in decomp.singletons if marker != obj}
                count = len([marker for marker in decomp.singletons if marker != obj])
                if count <= 0:
                    continue
                if len(cols) == 1 and len(rows) == count:
                    predicted = (int(obj.bbox_h) * count, int(obj.bbox_w))
                elif len(rows) == 1 and len(cols) == count:
                    predicted = (int(obj.bbox_h), int(obj.bbox_w) * count)
                else:
                    continue
                if predicted != target:
                    continue
                key = (connectivity, "color_bbox_area_desc", color, rank)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "connectivity": connectivity,
                        "selector": "color_bbox_area_desc",
                        "color": color,
                        "rank": rank,
                    }
                )
    candidates.sort(key=lambda c: (int(c.get("rank", 0)), int(c["connectivity"])))
    return tuple(candidates)


def _object_decomposition_for_connectivity(
    state: GridPerceptionState,
    params: Mapping[str, object] | None = None,
):
    connectivity = 4 if params is None else params.get("connectivity", 4)
    if connectivity == 8:
        return state.objects8
    return state.objects


def _selected_object_for_params(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object] | None = None,
):
    decomp = _object_decomposition_for_connectivity(state, params)
    objects = list(decomp.objects)
    non_singletons = [obj for obj in objects if obj.size > 1]
    pool = non_singletons if non_singletons else objects
    if not pool:
        return None

    if params is None:
        params = {}

    if selector == "largest_non_single_by_bbox_area":
        if not non_singletons:
            return None
        return max(
            non_singletons,
            key=lambda obj: (obj.bbox_h * obj.bbox_w, -obj.row, -obj.col),
        )

    rank = params.get("rank")
    if not isinstance(rank, int):
        return None

    if selector == "bbox_area_desc":
        ordered = sorted(
            pool,
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
            reverse=True,
        )
    elif selector == "bbox_area_asc":
        ordered = sorted(
            pool,
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col),
        )
    elif selector in {"pixel_count_desc", "pixel_size_desc"}:
        ordered = sorted(
            pool,
            key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col),
            reverse=True,
        )
    elif selector in {"pixel_count_asc", "pixel_size_asc"}:
        ordered = sorted(
            pool,
            key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col),
        )
    elif selector == "color_bbox_area_desc":
        color = params.get("color")
        if not isinstance(color, int):
            return None
        ordered = sorted(
            [obj for obj in pool if int(obj.color) == color],
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
            reverse=True,
        )
    else:
        return None

    if not (0 <= rank < len(ordered)):
        return None
    return ordered[rank]


def _color_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    colors = [color for color in state.non_bg_colors if color in state.color_bboxes]
    if not colors:
        return ()

    def bbox_size(color: int) -> Size:
        min_r, min_c, max_r, max_c = state.color_bboxes[color]
        return (max_r - min_r + 1, max_c - min_c + 1)

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for selector, key_fn, reverse in (
        ("pixel_count_desc", lambda c: (state.color_pixel_counts.get(c, 0), bbox_size(c)[0] * bbox_size(c)[1], -c), True),
        ("pixel_count_asc", lambda c: (state.color_pixel_counts.get(c, 0), bbox_size(c)[0] * bbox_size(c)[1], c), False),
        ("bbox_area_desc", lambda c: (bbox_size(c)[0] * bbox_size(c)[1], state.color_pixel_counts.get(c, 0), -c), True),
        ("bbox_area_asc", lambda c: (bbox_size(c)[0] * bbox_size(c)[1], state.color_pixel_counts.get(c, 0), c), False),
    ):
        ordered = sorted(colors, key=key_fn, reverse=reverse)
        for rank, color in enumerate(ordered):
            if bbox_size(color) != target:
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": selector, "rank": rank})

    candidates.sort(key=lambda c: c.get("rank", 0))
    return tuple(candidates)


def _boxed_region_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    if not state.boxed_regions:
        return ()

    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    by_area_desc = sorted(
        state.boxed_regions,
        key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
        reverse=True,
    )
    by_area_asc = sorted(
        state.boxed_regions,
        key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
    )

    for selector, ordered in (("area_desc", by_area_desc), ("area_asc", by_area_asc)):
        for rank, region in enumerate(ordered):
            if (region.height, region.width) != target:
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"selector": selector, "rank": rank})

    for frame_color in sorted({region.frame_color for region in state.boxed_regions}):
        pool = [
            region
            for region in by_area_desc
            if region.frame_color == frame_color
        ]
        if not pool:
            continue
        region = pool[0]
        if (region.height, region.width) != target:
            continue
        key = ("frame_color_largest", frame_color)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"selector": "frame_color_largest", "frame_color": frame_color})

    return tuple(candidates)


def _selected_boxed_region_interior_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> Size | None:
    regions = list(state.boxed_regions)
    if not regions:
        return None

    if selector == "area_desc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
            reverse=True,
        )
        if not (0 <= rank < len(ordered)):
            return None
        region = ordered[rank]
        return (int(region.height), int(region.width))

    if selector == "area_asc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
        )
        if not (0 <= rank < len(ordered)):
            return None
        region = ordered[rank]
        return (int(region.height), int(region.width))

    if selector == "frame_color_largest":
        frame_color = params.get("frame_color")
        if not isinstance(frame_color, int):
            return None
        pool = [region for region in regions if region.frame_color == frame_color]
        if not pool:
            return None
        region = max(
            pool,
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col),
        )
        return (int(region.height), int(region.width))

    return None


def _partition_cells(state: GridPerceptionState) -> list[tuple[int, int, int, int]]:
    return [
        (rr[0], rr[1], cc[0], cc[1])
        for rr in state.row_intervals
        for cc in state.col_intervals
        if rr[0] <= rr[1] and cc[0] <= cc[1]
    ]


def _select_partition_cell_bounds(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    cells = _partition_cells(state)
    if not cells:
        return None
    index = params.get("index")
    if not isinstance(index, int):
        return None
    if selector == "row_major":
        ordered = cells
    elif selector == "col_major":
        ordered = sorted(cells, key=lambda cell: (cell[2], cell[0], cell[3] - cell[2], cell[1] - cell[0]))
    else:
        return None
    return ordered[index] if 0 <= index < len(ordered) else None


def _strip_block_selector_candidates(
    state: GridPerceptionState,
    target: Size,
) -> tuple[dict[str, object], ...]:
    candidates: list[dict[str, object]] = []
    target_rows, target_cols = target
    rows, cols = state.dims

    count_candidates: set[int] = set()
    if rows == target_rows and target_cols > 0 and cols % target_cols == 0 and cols > target_cols:
        count_candidates.add(cols // target_cols)
    if cols == target_cols and target_rows > 0 and rows % target_rows == 0 and rows > target_rows:
        count_candidates.add(rows // target_rows)

    for block_count in sorted(count_candidates):
        for selector in (
            "row_panels",
            "col_panels",
            "leading_tiled_axis",
            "trailing_tiled_axis",
            "leading_long_axis_by_count",
            "trailing_long_axis_by_count",
        ):
            params = {"selector": selector, "block_count": block_count}
            if _selected_strip_block_size(state, selector, params) == target:
                candidates.append(params)
    return tuple(candidates)


def _selected_strip_block_size(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> Size | None:
    block = _select_strip_block_bounds(state, selector, params)
    if block is None:
        return None
    r0, r1, c0, c1 = block
    return (r1 - r0 + 1, c1 - c0 + 1)


def _select_strip_block_bounds(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    rows, cols = state.dims
    block_count = params.get("block_count")
    if not isinstance(block_count, int) or block_count <= 1:
        return None

    if selector == "row_panels":
        if rows % block_count != 0:
            return None
        bh = rows // block_count
        if bh >= rows:
            return None
        return (0, bh - 1, 0, cols - 1)

    if selector == "col_panels":
        if cols % block_count != 0:
            return None
        bw = cols // block_count
        if bw >= cols:
            return None
        return (0, rows - 1, 0, bw - 1)

    col_block = None
    row_block = None
    if cols % block_count == 0:
        block_width = cols // block_count
        if block_width < cols:
            if selector == "leading_tiled_axis":
                return (0, rows - 1, 0, block_width - 1)
            if selector == "trailing_tiled_axis":
                return (0, rows - 1, cols - block_width, cols - 1)
            col_block = {
                "leading_long_axis_by_count": (0, rows - 1, 0, block_width - 1),
                "trailing_long_axis_by_count": (0, rows - 1, cols - block_width, cols - 1),
            }

    if rows % block_count == 0:
        block_height = rows // block_count
        if block_height < rows:
            if selector == "leading_tiled_axis":
                return (0, block_height - 1, 0, cols - 1)
            if selector == "trailing_tiled_axis":
                return (rows - block_height, rows - 1, 0, cols - 1)
            row_block = {
                "leading_long_axis_by_count": (0, block_height - 1, 0, cols - 1),
                "trailing_long_axis_by_count": (rows - block_height, rows - 1, 0, cols - 1),
            }

    if selector in {"leading_long_axis_by_count", "trailing_long_axis_by_count"}:
        if col_block is None and row_block is None:
            return None
        if col_block is None:
            return row_block[selector]
        if row_block is None:
            return col_block[selector]
        if cols > rows:
            return col_block[selector]
        if rows > cols:
            return row_block[selector]
        return None

    return None


def _solid_rectangle_layout_shape_size(state: GridPerceptionState) -> Size | None:
    rectangles = _salient_solid_rectangles(state)
    if not rectangles:
        return None
    row_groups = _count_interval_clusters(
        (int(obj.row), int(obj.row + obj.bbox_h - 1))
        for obj in rectangles
    )
    col_groups = _count_interval_clusters(
        (int(obj.col), int(obj.col + obj.bbox_w - 1))
        for obj in rectangles
    )
    if row_groups <= 0 or col_groups <= 0:
        return None
    return (row_groups, col_groups)


def _salient_solid_rectangles(state: GridPerceptionState) -> tuple[object, ...]:
    rectangles = [
        obj
        for obj in state.objects.objects
        if obj.size == obj.bbox_h * obj.bbox_w and obj.bbox_h >= 3 and obj.bbox_w >= 3
    ]
    return tuple(rectangles)


def _count_interval_clusters(intervals: object) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    clusters = 1
    current_end = ordered[0][1]
    for start, end in ordered[1:]:
        if start <= current_end:
            if end > current_end:
                current_end = end
            continue
        clusters += 1
        current_end = end
    return clusters


def _non_bg_color_count_strip_size(state: GridPerceptionState, *, horizontal: bool) -> Size:
    count = len(state.non_bg_colors)
    if horizontal:
        return (1, count)
    return (count, 1)


def _preserve_input_rows_non_bg_cols_size(state: GridPerceptionState) -> Size:
    rows, cols = state.dims
    non_bg_cols = sum(
        any(int(value) != state.bg_color for value in state.grid[:, col])
        for col in range(cols)
    )
    return (rows, non_bg_cols)


def _dominant_shape_top_row_span_size(
    state: GridPerceptionState,
    params: Mapping[str, object] | None = None,
) -> Size | None:
    objects = [obj for obj in state.objects.objects if obj.size > 1]
    if not objects:
        return None

    if params is None:
        params = {}
    axis = params.get("axis", 0)
    rank = params.get("rank", 0)
    if not isinstance(axis, int) or not isinstance(rank, int):
        return None

    shape_counts: dict[tuple[int, int], int] = {}
    for obj in objects:
        shape = (int(obj.bbox_h), int(obj.bbox_w))
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    dominant_shape = max(
        shape_counts,
        key=lambda shape: (shape_counts[shape], shape[0] * shape[1], shape),
    )
    dominant_objects = [
        obj for obj in objects
        if (int(obj.bbox_h), int(obj.bbox_w)) == dominant_shape
    ]
    if not dominant_objects:
        return None

    if axis == 0:
        row_values = sorted({int(obj.row) for obj in dominant_objects})
        if rank >= len(row_values):
            return None
        target_pos = row_values[rank]
        subset = [obj for obj in dominant_objects if int(obj.row) == target_pos]
    elif axis == 1:
        col_values = sorted({int(obj.col) for obj in dominant_objects})
        if rank >= len(col_values):
            return None
        target_pos = col_values[rank]
        subset = [obj for obj in dominant_objects if int(obj.col) == target_pos]
    else:
        return None

    if not subset:
        return None

    min_row = min(int(obj.row) for obj in subset)
    min_col = min(int(obj.col) for obj in subset)
    max_row = max(int(obj.row) + int(obj.bbox_h) - 1 for obj in subset)
    max_col = max(int(obj.col) + int(obj.bbox_w) - 1 for obj in subset)
    return (max_row - min_row + 1, max_col - min_col + 1)


def _compute_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    seps = sorted(set(separators))
    if not seps:
        return [(0, size - 1)]

    intervals: list[tuple[int, int]] = []
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
