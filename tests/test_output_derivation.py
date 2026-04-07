from aria.core.output_derivation import (
    KIND_BOXED_REGION,
    KIND_FRAME_REGION,
    KIND_OBJECT,
    KIND_PARTITION_CELL,
    KIND_STRIP_BLOCK,
    RELATION_BORDER,
    RELATION_CLONE,
    RELATION_INTERIOR,
    infer_output_derivation_spec,
    infer_verified_output_derivation_specs,
    predict_output_derivation,
    verify_output_derivation_spec,
)
from aria.datasets import get_dataset, load_arc_task
from aria.types import DemoPair, grid_from_list


def test_output_derivation_spec_selected_boxed_region_interior_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    spec = infer_output_derivation_spec(task.train)
    assert spec is not None
    assert spec.candidate_kind == KIND_BOXED_REGION
    assert spec.relation == RELATION_INTERIOR
    assert spec.selector == "area_desc"
    assert spec.params == {"rank": 0}
    assert verify_output_derivation_spec(spec, task.train)


def test_output_derivation_spec_selected_object_clone_on_synthetic_task():
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0, 4],
                [0, 3, 0, 3, 0, 4],
                [0, 3, 3, 3, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [3, 3, 3],
                [3, 0, 3],
                [3, 3, 3],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0],
                [5, 0, 0, 2, 2, 2, 0],
                [5, 0, 0, 2, 0, 2, 0],
                [0, 0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [2, 2, 2],
                [2, 0, 2],
                [2, 2, 2],
            ]),
        ),
    )
    spec = infer_output_derivation_spec(demos)
    assert spec is not None
    assert spec.candidate_kind == KIND_OBJECT
    assert spec.relation == RELATION_CLONE
    assert spec.selector == "bbox_area_desc"
    assert spec.params == {"rank": 0}
    assert verify_output_derivation_spec(spec, demos)


def test_output_derivation_spec_selected_object_interior_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1c786137")
    spec = infer_output_derivation_spec(task.train)
    assert spec is not None
    assert spec.candidate_kind == KIND_OBJECT
    assert spec.relation == RELATION_INTERIOR
    assert spec.selector == "bbox_area_desc"
    assert spec.params == {"rank": 0}
    assert verify_output_derivation_spec(spec, task.train)


def test_output_derivation_spec_selected_boxed_border_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "c909285e")
    spec = infer_output_derivation_spec(task.train)
    assert spec is not None
    assert spec.candidate_kind == KIND_BOXED_REGION
    assert spec.relation == RELATION_BORDER
    assert spec.selector == "bottom_right_desc"
    assert spec.params == {"rank": 0}
    assert verify_output_derivation_spec(spec, task.train)


def test_output_derivation_spec_selected_frame_interior_on_synthetic_task():
    demos = (
        DemoPair(
            input=grid_from_list([
                [7, 7, 7, 7],
                [7, 1, 2, 7],
                [7, 3, 4, 7],
                [7, 7, 7, 7],
            ]),
            output=grid_from_list([
                [1, 2],
                [3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [5, 5, 5, 5, 5],
                [5, 9, 8, 7, 5],
                [5, 6, 5, 4, 5],
                [5, 3, 2, 1, 5],
                [5, 5, 5, 5, 5],
            ]),
            output=grid_from_list([
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1],
            ]),
        ),
    )
    spec = infer_output_derivation_spec(demos)
    assert spec is not None
    assert spec.candidate_kind == KIND_FRAME_REGION
    assert spec.relation == RELATION_INTERIOR
    assert spec.selector == "area_desc"
    assert spec.params == {"rank": 0}
    assert verify_output_derivation_spec(spec, demos)


def test_output_derivation_spec_selected_boxed_border_on_synthetic_task():
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 6, 6, 6, 6, 6, 0],
                [0, 6, 1, 2, 3, 6, 0],
                [0, 6, 4, 0, 5, 6, 0],
                [0, 6, 6, 6, 6, 6, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [6, 6, 6, 6, 6],
                [6, 1, 2, 3, 6],
                [6, 4, 0, 5, 6],
                [6, 6, 6, 6, 6],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 8, 0],
                [0, 8, 9, 1, 8, 0],
                [0, 8, 2, 3, 8, 0],
                [0, 8, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [8, 8, 8, 8],
                [8, 9, 1, 8],
                [8, 2, 3, 8],
                [8, 8, 8, 8],
            ]),
        ),
    )
    specs = infer_verified_output_derivation_specs(demos)
    matching = [
        spec for spec in specs
        if spec.candidate_kind == KIND_BOXED_REGION
        and spec.relation == RELATION_BORDER
        and spec.selector == "area_desc"
        and spec.params == {"rank": 1}
    ]
    assert matching
    assert verify_output_derivation_spec(matching[0], demos)


def test_output_derivation_spec_selected_partition_cell_clone_on_synthetic_task():
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
    spec = infer_output_derivation_spec(demos)
    assert spec is not None
    assert spec.candidate_kind == KIND_PARTITION_CELL
    assert spec.relation == RELATION_CLONE
    assert spec.selector == "row_major"
    assert spec.params == {"index": 0}
    assert verify_output_derivation_spec(spec, demos)
    predicted = predict_output_derivation(spec, demos[1].input)
    assert predicted is not None
    assert predicted.shape == demos[1].output.shape


def test_output_derivation_spec_selected_strip_block_clone_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "2dee498d")
    spec = infer_output_derivation_spec(task.train)
    assert spec is not None
    assert spec.candidate_kind == KIND_STRIP_BLOCK
    assert spec.relation == RELATION_CLONE
    assert spec.selector == "leading_tiled_axis"
    assert spec.params == {"block_count": 3}
    assert verify_output_derivation_spec(spec, task.train)


def test_output_derivation_spec_selected_strip_block_clone_axis_agnostic_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "7b7f7511")
    spec = infer_output_derivation_spec(task.train)
    assert spec is not None
    assert spec.candidate_kind == KIND_STRIP_BLOCK
    assert spec.relation == RELATION_CLONE
    assert spec.selector == "leading_long_axis_by_count"
    assert spec.params == {"block_count": 2}
    assert verify_output_derivation_spec(spec, task.train)
