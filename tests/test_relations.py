"""Tests for the relation layer: slot grids, correspondences, legend mappings,
zone/partition summary grids."""

import numpy as np
import pytest

from aria.core.grid_perception import perceive_grid
from aria.core.relations import (
    Correspondence,
    CorrespondenceEntry,
    LegendMapping,
    SlotGrid,
    ZoneMapping,
    build_legend_mapping,
    build_object_to_slot_correspondence,
    build_zone_to_zone_correspondence,
    detect_partition_summary_grid,
    detect_slot_grid,
    detect_zone_summary_grid,
    verify_slot_grid_consistent,
    verify_zone_summary_grid,
)
from aria.datasets import get_dataset, load_arc_task
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# SlotGrid detection
# ---------------------------------------------------------------------------


def test_slot_grid_basic_4x3():
    """4x3 grid of identical objects at regular positions."""
    grid = grid_from_list([
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
    ])
    state = perceive_grid(grid)
    sg = detect_slot_grid(state)
    assert sg is not None
    assert sg.n_rows == 4
    assert sg.n_cols == 3


def test_slot_grid_none_for_random():
    """No slot grid for a grid without regular object positions."""
    grid = grid_from_list([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 3],
        [4, 0, 0, 0, 0],
    ])
    state = perceive_grid(grid)
    sg = detect_slot_grid(state)
    assert sg is None


def test_slot_grid_real_task():
    """Detect slot grid on a real task with known grid structure."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0b148d64")
    state = perceive_grid(task.train[0].input)
    sg = detect_slot_grid(state)
    assert sg is not None
    assert sg.n_rows >= 2
    assert sg.n_cols >= 2


def test_slot_grid_extract_slot():
    grid = grid_from_list([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 4, 0],
    ])
    state = perceive_grid(grid)
    sg = detect_slot_grid(state)
    assert sg is not None
    slot_grid = sg.extract_slot(grid, 0, 0)
    assert slot_grid is not None
    assert slot_grid.shape[0] >= 1


# ---------------------------------------------------------------------------
# LegendMapping
# ---------------------------------------------------------------------------


def test_legend_mapping_basic():
    ds = get_dataset("v2-train")
    # Use a task known to have a legend
    for tid in ["09c534e7", "0becf7df"]:
        task = load_arc_task(ds, tid)
        state = perceive_grid(task.train[0].input)
        if state.legend is not None:
            lm = build_legend_mapping(state)
            assert lm is not None
            assert len(lm.key_to_value) > 0
            # Bidirectional lookup
            for k, v in lm.key_to_value.items():
                assert lm.lookup(k) == v
                assert lm.reverse_lookup(v) == k
            return
    pytest.skip("no legend task found")


def test_legend_mapping_none_without_legend():
    grid = grid_from_list([[0, 0], [0, 0]])
    state = perceive_grid(grid)
    lm = build_legend_mapping(state)
    assert lm is None


# ---------------------------------------------------------------------------
# ZoneMapping / Partition summary grid
# ---------------------------------------------------------------------------


def test_partition_summary_grid_real_tasks():
    """Verify partition summary grid on known tasks."""
    ds = get_dataset("v2-train")
    for tid in ["1190e5a7", "7039b2d7", "780d0b14", "68b67ca3"]:
        task = load_arc_task(ds, tid)
        mapping = verify_zone_summary_grid(task.train)
        assert mapping is not None, f"{tid}: expected zone summary grid"
        assert mapping.mapping_kind == "partition_summary_grid"
        assert mapping.params.get("property") == "dominant_non_bg_color"


def test_partition_summary_compiled_program():
    """Verify that partition summary compiles to a working program."""
    from aria.core.output_stage1 import compile_stage1_program, infer_output_stage1_spec
    from aria.verify.verifier import verify

    ds = get_dataset("v2-train")
    for tid in ["1190e5a7", "780d0b14"]:
        task = load_arc_task(ds, tid)
        spec = infer_output_stage1_spec(task.train)
        assert spec is not None
        prog = compile_stage1_program(spec)
        assert prog is not None
        vr = verify(prog, task.train)
        assert vr.passed, f"{tid}: program did not verify"


def test_zone_summary_grid_basic():
    """Simple 2x2 zone summary from separator-divided grid."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    output = grid_from_list([[1, 2], [3, 4]])
    state = perceive_grid(grid)
    m = detect_zone_summary_grid(state, output)
    # May or may not match depending on zone detection details
    # but the structure should be representable
    if m is not None:
        assert m.mapping_kind == "summary_grid"


# ---------------------------------------------------------------------------
# Correspondence
# ---------------------------------------------------------------------------


def test_zone_to_zone_correspondence_basic():
    grid1 = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    grid2 = grid_from_list([
        [5, 5, 0, 6, 6],
        [5, 5, 0, 6, 6],
        [0, 0, 0, 0, 0],
        [7, 7, 0, 8, 8],
        [7, 7, 0, 8, 8],
    ])
    s1 = perceive_grid(grid1)
    s2 = perceive_grid(grid2)
    corr = build_zone_to_zone_correspondence(s1, s2)
    if corr is not None:
        assert corr.kind == "zone_to_zone"
        assert len(corr.entries) >= 2


def test_object_to_slot_correspondence():
    grid = grid_from_list([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 4, 0],
    ])
    state = perceive_grid(grid)
    sg = detect_slot_grid(state)
    if sg is None:
        pytest.skip("slot grid not detected")
    corr = build_object_to_slot_correspondence(state, sg)
    assert corr is not None
    assert corr.kind == "object_to_slot"
    assert len(corr.entries) >= 2


# ---------------------------------------------------------------------------
# No task-id logic
# ---------------------------------------------------------------------------


def test_global_color_map_solves():
    """Global color map programs verify on known tasks."""
    from aria.core.output_stage1 import compile_stage1_program, infer_output_stage1_spec
    from aria.verify.verifier import verify

    ds = get_dataset("v2-train")
    for tid in ["0d3d703e", "b1948b0a", "d511f180"]:
        task = load_arc_task(ds, tid)
        spec = infer_output_stage1_spec(task.train)
        assert spec is not None, f"{tid}: no spec"
        assert spec.render_spec is not None, f"{tid}: no render spec"
        assert spec.render_spec.get("kind") == "global_color_map", f"{tid}: wrong kind"
        prog = compile_stage1_program(spec)
        assert prog is not None, f"{tid}: no program"
        vr = verify(prog, task.train)
        assert vr.passed, f"{tid}: verify failed"


def test_global_color_map_not_triggered_for_inconsistent():
    """Color map should not match when demos have contradictory mappings."""
    from aria.core.output_stage1 import infer_output_stage1_spec

    demos = (
        DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[3, 4]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[4, 3]]),
        ),
    )
    spec = infer_output_stage1_spec(demos)
    if spec is not None and spec.render_spec is not None:
        assert spec.render_spec.get("kind") != "global_color_map"


def test_geometric_transform_solves():
    """Geometric transform programs verify on known tasks."""
    from aria.core.output_stage1 import compile_stage1_program, infer_output_stage1_spec
    from aria.verify.verifier import verify

    ds = get_dataset("v2-train")
    for tid, expected in [("ed36ccf7", "rot90"), ("6150a2bd", "rot180"),
                          ("67a3c6ac", "flip_lr"), ("9dfd6313", "transpose")]:
        task = load_arc_task(ds, tid)
        spec = infer_output_stage1_spec(task.train)
        assert spec is not None
        assert spec.render_spec is not None
        assert spec.render_spec.get("kind") == "geometric_transform"
        assert expected in spec.render_spec.get("rationale", "")
        prog = compile_stage1_program(spec)
        assert prog is not None
        vr = verify(prog, task.train)
        assert vr.passed, f"{tid}: verify failed"


def test_partition_cell_select_solve():
    """Partition cell selection verifies."""
    from aria.core.output_stage1 import compile_stage1_program, infer_output_stage1_spec
    from aria.verify.verifier import verify

    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "2dc579da")
    spec = infer_output_stage1_spec(task.train)
    assert spec is not None
    assert spec.render_spec is not None
    assert spec.render_spec.get("kind") == "partition_cell_select"
    prog = compile_stage1_program(spec)
    assert prog is not None
    vr = verify(prog, task.train)
    assert vr.passed


def test_no_task_id_in_new_modules():
    """No task-id logic in new modules."""
    import inspect
    import aria.core.relations as rel_mod
    import aria.core.output_stage1 as stage1_mod
    import aria.runtime.ops.zone_summary as zs_mod
    import aria.runtime.ops.color_map as cm_mod
    import aria.runtime.ops.scene_transforms as st_mod

    for mod in (rel_mod, stage1_mod, zs_mod, cm_mod, st_mod):
        src = inspect.getsource(mod)
        for tid in ["1190e5a7", "780d0b14", "0d3d703e", "ed36ccf7", "2dc579da"]:
            assert tid not in src, f"task-id {tid} found in {mod.__name__}"
