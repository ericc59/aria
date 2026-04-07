from aria.core.output_stage1 import (
    compile_stage1_derivation_program,
    compile_stage1_program,
    infer_output_stage1_spec,
)
from aria.datasets import get_dataset, load_arc_task
from aria.types import DemoPair, grid_from_list
from aria.verify.verifier import verify


def test_output_stage1_spec_for_boxed_region_interior_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec.mode == "selected_boxed_region_interior"
    assert spec.derivation_spec is not None
    assert spec.derivation_spec.candidate_kind == "boxed_region"
    assert spec.derivation_spec.relation == "interior"


def test_output_stage1_spec_for_object_clone_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1cf80156")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec is not None
    assert spec.derivation_spec is not None
    assert spec.derivation_spec.candidate_kind == "object"
    assert spec.derivation_spec.relation == "clone"


def test_output_stage1_spec_for_strip_block_clone_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "2dee498d")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec.mode == "selected_strip_block_size"
    assert spec.derivation_spec is not None
    assert spec.derivation_spec.candidate_kind == "strip_block"
    assert spec.derivation_spec.relation == "clone"


def test_output_stage1_spec_for_scaled_object_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec.mode == "scaled_bbox_of_selected_object"
    assert spec.derivation_spec is None


def test_output_stage1_spec_for_marker_stacked_object_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "12997ef3")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec.mode == "marker_stacked_selected_object"
    assert spec.derivation_spec is None


def test_output_stage1_spec_for_tiled_input_pattern_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "00576224")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.size_spec.mode == "scale_input"
    assert spec.derivation_spec is None
    assert spec.render_spec is not None
    assert spec.render_spec["kind"] == "tiled_input_pattern"
    assert spec.render_spec["row_repeat"] == 3
    assert spec.render_spec["col_repeat"] == 3
    assert spec.render_spec["odd_row_transform"] == "flip_lr"
    assert spec.render_spec["odd_col_transform"] == "identity"


def test_output_stage1_derivation_compiles_to_verified_program():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    program = compile_stage1_derivation_program(spec)
    assert program is not None
    assert verify(program, task.train).passed is True


def test_output_stage1_scaled_object_compiles_to_verified_program():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    program = compile_stage1_program(spec)
    assert program is not None
    assert verify(program, task.train).passed is True


def test_output_stage1_marker_stacked_object_compiles_to_verified_program():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "12997ef3")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    program = compile_stage1_program(spec)
    assert program is not None
    assert verify(program, task.train).passed is True


def test_output_stage1_tiled_input_pattern_compiles_to_verified_program():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "00576224")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    program = compile_stage1_program(spec)
    assert program is not None
    assert verify(program, task.train).passed is True


def test_output_stage1_solid_rectangle_layout_compiles_to_verified_program():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0a1d4ef5")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    program = compile_stage1_program(spec)
    assert program is not None
    assert verify(program, task.train).passed is True


def test_output_stage1_partition_cell_clone_compiles_to_verified_program():
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
    spec = infer_output_stage1_spec(demos)
    assert spec is not None
    assert spec.size_spec.mode == "selected_partition_cell_size"
    assert spec.derivation_spec is not None
    assert spec.derivation_spec.candidate_kind == "partition_cell"
    assert spec.derivation_spec.relation == "clone"
    program = compile_stage1_program(spec)
    assert program is not None
    assert verify(program, demos).passed is True
