from fractions import Fraction

from aria.core.output_size import (
    MODE_ADDITIVE_INPUT,
    MODE_BBOX_OF_SELECTED_COLOR,
    MODE_BBOX_OF_SELECTED_OBJECT,
    MODE_CONSTANT_ROWS_PRESERVE_INPUT_COLS,
    MODE_DOMINANT_SHAPE_TOP_ROW_SPAN,
    MODE_SCALE_INPUT_BY_PALETTE_SIZE,
    MODE_FRAME_INTERIOR_SIZE,
    MODE_NON_BG_COLOR_COUNT_STRIP,
    MODE_OBJECT_POSITION_GRID_SHAPE,
    MODE_PARTITION_GRID_SHAPE,
    MODE_SELECTED_PARTITION_CELL_SIZE,
    MODE_PRESERVE_INPUT_ROWS_NON_BG_COLS,
    MODE_PRESERVE_INPUT_ROWS_EXPAND_COLS_BY_ROWS_MINUS_ONE,
    MODE_EXPAND_ROWS_TO_FOUR_TIMES_MINUS_THREE_PRESERVE_COLS,
    MODE_ROW_STRIP_HALF_COLS_BY_COLS,
    MODE_SELECTED_BOXED_REGION_INTERIOR,
    MODE_ROW_STRIP_SQUARE_BY_NON_BG_PIXEL_COUNT,
    MODE_ROW_STRIP_SQUARE_FROM_COLS,
    MODE_MARKER_STACKED_SELECTED_OBJECT,
    MODE_SCALE_INPUT,
    MODE_SEPARATOR_CELL_SIZE,
    MODE_SELECTED_STRIP_BLOCK_SIZE,
    MODE_SAME_AS_INPUT,
    MODE_SOLID_RECTANGLE_LAYOUT_SHAPE,
    MODE_SCALED_BBOX_OF_SELECTED_OBJECT,
    MODE_SQUARE_CANVAS_SCALED,
    MODE_TIGHT_BBOX_OF_NON_BG,
    infer_output_size_spec,
    infer_verified_output_size_specs,
    predict_output_size,
    verify_output_size_spec,
)
from aria.datasets import get_dataset, load_arc_task
from aria.types import DemoPair, grid_from_list


def test_output_size_spec_same_as_input_on_reference_transfer_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "009d5c81")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SAME_AS_INPUT
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_scale_input_on_tile_transform_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "00576224")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SCALE_INPUT
    assert spec.params["row_ratio"] == Fraction(3, 1)
    assert spec.params["col_ratio"] == Fraction(3, 1)
    assert predict_output_size(spec, task.test[0].input) == (6, 6)


def test_output_size_spec_scale_input_by_palette_size_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "a59b95c0")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SCALE_INPUT_BY_PALETTE_SIZE
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_additive_input_on_vertical_extension_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "017c7c7b")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_ADDITIVE_INPUT
    assert spec.params == {"row_delta": 3, "col_delta": 0}
    assert predict_output_size(spec, task.test[0].input) == (9, 3)


def test_output_size_spec_square_canvas_scaled_by_input_side_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "973e499e")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SQUARE_CANVAS_SCALED
    assert spec.params == {"factor_source": "input_side"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_square_canvas_scaled_by_non_bg_color_count_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "469497ad")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SQUARE_CANVAS_SCALED
    assert spec.params == {"factor_source": "non_bg_color_count"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_separator_cell_size_on_panel_boolean_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0520fde7")
    specs = infer_verified_output_size_specs(task.train)
    modes = [spec.mode for spec in specs]
    assert MODE_SEPARATOR_CELL_SIZE in modes


def test_output_size_spec_partition_grid_shape_on_partition_summary_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1190e5a7")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_PARTITION_GRID_SHAPE
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_object_position_grid_shape_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "458e3a53")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_OBJECT_POSITION_GRID_SHAPE
    assert spec.params == {"anchor": "top_left"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_tight_bbox_of_non_bg_on_synthetic_crop():
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [2, 2],
                [2, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [3, 3],
                [3, 0],
            ]),
        ),
    )
    specs = infer_verified_output_size_specs(demos)
    modes = [spec.mode for spec in specs]
    assert MODE_TIGHT_BBOX_OF_NON_BG in modes


def test_output_size_spec_selected_object_bbox_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "25e02866")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_BBOX_OF_SELECTED_OBJECT
    assert spec.params["selector"] == "bbox_area_desc"
    assert spec.params["rank"] == 0
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_object_bbox_color_rank_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "8e1813be")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_BBOX_OF_SELECTED_OBJECT
    assert spec.params == {"selector": "color_bbox_area_desc", "color": 5, "rank": 0}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_color_bbox_min_pixel_count_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "846bdb03")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_BBOX_OF_SELECTED_COLOR
    assert spec.params["selector"] == "pixel_count_asc"
    assert spec.params["rank"] == 0
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_color_bbox_min_bbox_area_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "de493100")
    specs = infer_verified_output_size_specs(task.train)
    matching = [
        spec for spec in specs
        if spec.mode == MODE_BBOX_OF_SELECTED_COLOR
        and spec.params.get("selector") == "bbox_area_asc"
        and spec.params.get("rank") == 0
    ]
    assert matching
    assert verify_output_size_spec(matching[0], task.train)


def test_output_size_spec_selected_color_bbox_max_bbox_area_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "b0f4d537")
    specs = infer_verified_output_size_specs(task.train)
    matching = [
        spec for spec in specs
        if spec.mode == MODE_BBOX_OF_SELECTED_COLOR
        and spec.params.get("selector") == "bbox_area_desc"
        and spec.params.get("rank") == 0
    ]
    assert matching
    assert verify_output_size_spec(matching[0], task.train)


def test_output_size_spec_selected_color_bbox_max_pixel_count_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "c64f1187")
    specs = infer_verified_output_size_specs(task.train)
    matching = [
        spec for spec in specs
        if spec.mode == MODE_BBOX_OF_SELECTED_COLOR
        and spec.params.get("selector") == "pixel_count_desc"
        and spec.params.get("rank") == 0
    ]
    assert matching
    assert verify_output_size_spec(matching[0], task.train)


def test_output_size_spec_selected_partition_cell_size_on_synthetic_task():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 2, 0, 3, 4],
                [5, 6, 0, 7, 8],
                [0, 0, 0, 0, 0],
                [9, 1, 0, 2, 3],
                [4, 5, 0, 6, 7],
            ]),
            output=grid_from_list([
                [1, 2],
                [5, 6],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [8, 7, 0, 6, 5],
                [4, 3, 0, 2, 1],
                [0, 0, 0, 0, 0],
                [9, 9, 0, 1, 1],
                [8, 8, 0, 2, 2],
            ]),
            output=grid_from_list([
                [8, 7],
                [4, 3],
            ]),
        ),
    )
    spec = infer_output_size_spec(demos)
    assert spec is not None
    assert spec.mode == MODE_SELECTED_PARTITION_CELL_SIZE
    assert spec.params == {"selector": "row_major", "index": 0}
    assert verify_output_size_spec(spec, demos)


def test_output_size_spec_non_bg_color_count_horizontal_strip_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "22425bda")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_NON_BG_COLOR_COUNT_STRIP
    assert spec.params == {"orientation": "horizontal"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_non_bg_color_count_vertical_strip_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "68bc2e87")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_NON_BG_COLOR_COUNT_STRIP
    assert spec.params == {"orientation": "vertical"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_preserve_input_rows_non_bg_cols_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "d017b73f")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_PRESERVE_INPUT_ROWS_NON_BG_COLS
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_constant_rows_preserve_input_cols_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "7e4d4f7c")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_CONSTANT_ROWS_PRESERVE_INPUT_COLS
    assert spec.params == {"rows": 3}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_row_strip_square_from_cols_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "c1990cce")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_ROW_STRIP_SQUARE_FROM_COLS
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_row_strip_half_cols_by_cols_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "bbc9ae5d")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_ROW_STRIP_HALF_COLS_BY_COLS
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_row_strip_square_by_non_bg_pixel_count_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "feca6190")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_ROW_STRIP_SQUARE_BY_NON_BG_PIXEL_COUNT
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_preserve_input_rows_expand_cols_by_rows_minus_one_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "3cd86f4f")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_PRESERVE_INPUT_ROWS_EXPAND_COLS_BY_ROWS_MINUS_ONE
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_expand_rows_to_four_times_minus_three_preserve_cols_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "eb281b96")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_EXPAND_ROWS_TO_FOUR_TIMES_MINUS_THREE_PRESERVE_COLS
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_dominant_shape_top_row_span_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "15660dd6")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_DOMINANT_SHAPE_TOP_ROW_SPAN
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_boxed_region_interior_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SELECTED_BOXED_REGION_INTERIOR
    assert spec.params == {"selector": "area_desc", "rank": 0}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_strip_block_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "2dee498d")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SELECTED_STRIP_BLOCK_SIZE
    assert spec.params["block_count"] == 3
    assert spec.params["selector"] in ("col_panels", "leading_tiled_axis")
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_strip_block_long_axis_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "7b7f7511")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SELECTED_STRIP_BLOCK_SIZE
    assert spec.params == {"selector": "leading_long_axis_by_count", "block_count": 2}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_solid_rectangle_layout_shape_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0a1d4ef5")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SOLID_RECTANGLE_LAYOUT_SHAPE
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_scaled_bbox_of_selected_object_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SCALED_BBOX_OF_SELECTED_OBJECT
    assert spec.params == {
        "connectivity": 8,
        "selector": "bbox_area_desc",
        "rank": 0,
        "row_scale": 2,
        "col_scale": 2,
    }
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_marker_stacked_selected_object_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "12997ef3")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_MARKER_STACKED_SELECTED_OBJECT
    assert spec.params == {
        "connectivity": 8,
        "selector": "bbox_area_desc",
        "rank": 0,
    }
    assert verify_output_size_spec(spec, task.train)


def test_output_size_spec_selected_smallest_boxed_region_interior_on_synthetic_task():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 1, 1, 1, 1, 0, 2, 2, 2],
                [1, 0, 0, 0, 1, 0, 2, 0, 2],
                [1, 0, 0, 0, 1, 0, 2, 2, 2],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]),
            output=grid_from_list([[0]]),
        ),
        DemoPair(
            input=grid_from_list([
                [3, 3, 3, 3, 0, 4, 4, 4],
                [3, 0, 0, 3, 0, 4, 0, 4],
                [3, 0, 0, 3, 0, 4, 4, 4],
                [3, 3, 3, 3, 0, 0, 0, 0],
            ]),
            output=grid_from_list([[0]]),
        ),
    )
    spec = infer_output_size_spec(demos)
    assert spec is not None
    assert spec.mode == MODE_SELECTED_BOXED_REGION_INTERIOR
    assert spec.params in (
        {"selector": "area_asc", "rank": 0},
        {"selector": "area_desc", "rank": 1},
    )
    assert verify_output_size_spec(spec, demos)


def test_output_size_spec_frame_interior_size_on_synthetic_frame():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [2, 2, 2, 2],
                [2, 0, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ]),
            output=grid_from_list([
                [0, 0],
                [0, 0],
            ]),
        ),
    )
    specs = infer_verified_output_size_specs(demos)
    assert any(spec.mode == MODE_FRAME_INTERIOR_SIZE for spec in specs)


def test_output_size_spec_fails_closed_when_no_rule_verifies():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1], [0]]),
        ),
    )
    assert infer_output_size_spec(demos) is None


def test_frame_interior_selector_uses_new_format():
    """Frame interior selector uses area_desc + rank format."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [2, 2, 2, 2],
                [2, 0, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ]),
            output=grid_from_list([
                [0, 0],
                [0, 0],
            ]),
        ),
    )
    specs = infer_verified_output_size_specs(demos)
    frame_specs = [s for s in specs if s.mode == MODE_FRAME_INTERIOR_SIZE]
    assert frame_specs
    spec = frame_specs[0]
    assert spec.params["selector"] == "area_desc"
    assert spec.params["rank"] == 0
    assert verify_output_size_spec(spec, demos)


def test_frame_interior_legacy_select_largest_still_works():
    """Legacy select=largest param still predicts correctly."""
    from aria.core.output_size import OutputSizeSpec, predict_output_size
    spec = OutputSizeSpec(
        mode=MODE_FRAME_INTERIOR_SIZE,
        params={"select": "largest"},
    )
    grid = grid_from_list([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    result = predict_output_size(spec, grid)
    assert result == (2, 3)


def test_object_bbox_smallest_object_selection():
    """Select smallest non-singleton object bbox."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([[2, 2]]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0, 0],
                [0, 3, 3, 3, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 4, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([[4, 4]]),
        ),
    )
    spec = infer_output_size_spec(demos)
    assert spec is not None
    assert spec.mode == MODE_BBOX_OF_SELECTED_OBJECT
    assert spec.params["selector"] == "bbox_area_asc"
    assert spec.params["rank"] == 0
    assert verify_output_size_spec(spec, demos)


def test_color_bbox_second_rank_selector():
    """Rank > 0 color selector: second largest by pixel count."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 3, 3],
                [0, 0, 0, 3, 3],
            ]),
            output=grid_from_list([
                [2, 2],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [4, 4, 4, 0, 0],
                [4, 4, 0, 0, 0],
                [0, 0, 5, 5, 0],
                [0, 0, 0, 6, 6],
                [0, 0, 0, 6, 6],
            ]),
            output=grid_from_list([
                [5, 5],
            ]),
        ),
    )
    specs = infer_verified_output_size_specs(demos)
    color_specs = [s for s in specs if s.mode == MODE_BBOX_OF_SELECTED_COLOR]
    assert color_specs
    spec = color_specs[0]
    assert spec.params.get("rank", -1) >= 0
    assert verify_output_size_spec(spec, demos)


def test_dominant_shape_span_with_axis_and_rank_params():
    """Dominant shape span uses new axis + rank params."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "15660dd6")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_DOMINANT_SHAPE_TOP_ROW_SPAN
    assert "axis" in spec.params
    assert "rank" in spec.params
    assert verify_output_size_spec(spec, task.train)


def test_output_size_count_square_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "d0f5fe59")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_OBJECT_POSITION_GRID_SHAPE
    assert spec.params == {"anchor": "count_square"}
    assert verify_output_size_spec(spec, task.train)


def test_output_size_separator_panel_on_real_task():
    from aria.core.output_size import MODE_SEPARATOR_PANEL_SIZE
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "ba1aa698")
    spec = infer_output_size_spec(task.train)
    assert spec is not None
    assert spec.mode == MODE_SEPARATOR_PANEL_SIZE
    assert verify_output_size_spec(spec, task.train)


def test_no_task_id_in_output_size():
    import inspect
    from aria.core.output_size import (
        _frame_selector_candidates,
        _selected_frame_interior_size,
        _object_selector_candidates,
        _color_selector_candidates,
        _separator_panel_shape,
    )
    for fn in [
        _frame_selector_candidates,
        _selected_frame_interior_size,
        _object_selector_candidates,
        _color_selector_candidates,
        _separator_panel_shape,
    ]:
        assert "1b59e163" not in inspect.getsource(fn)
