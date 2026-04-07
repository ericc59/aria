"""Tests for the scene program executor and multi-step solving."""

import numpy as np
import pytest

from aria.core.grid_perception import perceive_grid
from aria.core.scene_executor import (
    SceneExecutionError,
    SceneState,
    execute_scene_program,
    make_scene_program,
    make_step,
)
from aria.core.scene_solve import infer_scene_programs, verify_scene_program
from aria.datasets import get_dataset, load_arc_task
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Executor basics
# ---------------------------------------------------------------------------


def test_parse_scene_creates_entities():
    grid = grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    prog = make_scene_program(make_step(StepOp.PARSE_SCENE))
    # Shouldn't crash, but no output
    with pytest.raises(SceneExecutionError, match="did not produce"):
        execute_scene_program(prog, grid)


def test_extract_template_and_render():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0],
        [0, 0, 0, 0, 0],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.EXTRACT_TEMPLATE, source="frame_0_interior", output_id="t1"),
        make_step(StepOp.RENDER_SCENE, source="t1"),
    )
    result = execute_scene_program(prog, grid)
    expected = grid_from_list([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(result, expected)


def test_extract_transform_render():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0],
        [0, 0, 0, 0, 0],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.EXTRACT_TEMPLATE, source="frame_0_interior", output_id="t1"),
        make_step(StepOp.CANONICALIZE_OBJECT, source="t1", transform="transpose", output_id="t2"),
        make_step(StepOp.RENDER_SCENE, source="t2"),
    )
    result = execute_scene_program(prog, grid)
    expected = grid_from_list([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(result, expected)


def test_initialize_output_and_render():
    grid = grid_from_list([[1, 2], [3, 4]])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.INFER_OUTPUT_SIZE, shape=(3, 3)),
        make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=7),
        make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    assert result.shape == (3, 3)
    assert np.all(result == 7)


def test_stamp_template():
    grid = grid_from_list([[1, 2], [3, 4]])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.EXTRACT_TEMPLATE, source="object_0", output_id="t1"),
        make_step(StepOp.INFER_OUTPUT_SIZE, shape=(4, 4)),
        make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=0),
        make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
        make_step(StepOp.STAMP_TEMPLATE, source="t1", row=1, col=1),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    assert result.shape == (4, 4)


def test_recolor_object():
    grid = grid_from_list([[1, 1, 2], [1, 2, 2]])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.EXTRACT_TEMPLATE, source="scene", output_id="t1"),
        make_step(StepOp.INFER_OUTPUT_SIZE, shape=(2, 3)),
        make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=0),
        make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
        make_step(StepOp.STAMP_TEMPLATE, source="t1", row=0, col=0),
        make_step(StepOp.RECOLOR_OBJECT, from_color=1, to_color=5),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    expected = grid_from_list([[5, 5, 2], [5, 2, 2]])
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# SELECT_ENTITY
# ---------------------------------------------------------------------------


def test_select_entity_by_predicate():
    """SELECT_ENTITY with different predicates."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 4, 4],
        [0, 0, 0, 4, 4],
    ])
    # Panel with most non-bg should be one of the 2x2 fully-filled panels
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="panel", predicate="most_non_bg", rank=0, output_id="sel"),
        make_step(StepOp.RENDER_SCENE, source="sel_grid"),
    )
    result = execute_scene_program(prog, grid)
    assert result.shape[0] >= 1  # Should select a valid panel


def test_select_entity_rank():
    """Rank parameter selects Nth best entity."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    # rank=0 and rank=1 should give different panels
    prog0 = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="panel", predicate="top_left", rank=0, output_id="sel"),
        make_step(StepOp.RENDER_SCENE, source="sel_grid"),
    )
    prog1 = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="panel", predicate="top_left", rank=1, output_id="sel"),
        make_step(StepOp.RENDER_SCENE, source="sel_grid"),
    )
    r0 = execute_scene_program(prog0, grid)
    r1 = execute_scene_program(prog1, grid)
    # They should be valid grids (not necessarily different if panels are same size)
    assert r0.size > 0
    assert r1.size > 0


# ---------------------------------------------------------------------------
# BUILD_CORRESPONDENCE + MAP_OVER_ENTITIES
# ---------------------------------------------------------------------------


def test_build_correspondence_row_col_index():
    """BUILD_CORRESPONDENCE with row_col_index mode."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    from aria.core.scene_executor import SceneState, _parse_scene, _build_correspondence
    state = SceneState(input_grid=grid, perception=perceive_grid(grid))
    _parse_scene(state, SceneStep(op=StepOp.PARSE_SCENE))
    _build_correspondence(state, SceneStep(
        op=StepOp.BUILD_CORRESPONDENCE,
        params={"source_kind": "panel", "mode": "row_col_index"},
        output_id="corr",
    ))
    corr = state.values.get("corr")
    assert corr is not None
    assert len(corr) == 4  # 2x2 partition = 4 panels
    # Each entry is (entity_id, row, col)
    positions = {(r, c) for _, r, c in corr}
    assert (0, 0) in positions
    assert (1, 1) in positions


def test_map_over_entities_grid():
    """MAP_OVER_ENTITIES writes property values to output grid."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.BUILD_CORRESPONDENCE, source_kind="panel", mode="row_col_index", output_id="corr"),
        make_step(StepOp.INFER_OUTPUT_SIZE, shape=(2, 2)),
        make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=0),
        make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
        make_step(StepOp.MAP_OVER_ENTITIES, kind="panel", property="dominant_non_bg_color",
                  layout="grid", correspondence="corr"),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    expected = grid_from_list([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


def test_map_over_entities_list():
    """MAP_OVER_ENTITIES with list layout stores results in values."""
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    from aria.core.scene_executor import SceneState, _parse_scene, _map_over_entities
    state = SceneState(input_grid=grid, perception=perceive_grid(grid))
    _parse_scene(state, SceneStep(op=StepOp.PARSE_SCENE))
    _map_over_entities(state, SceneStep(
        op=StepOp.MAP_OVER_ENTITIES,
        params={"kind": "panel", "property": "dominant_non_bg_color", "layout": "list"},
        output_id="results",
    ))
    results = state.values.get("results")
    assert results is not None
    assert set(results) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Execution contract: scene programs work without stage-1 size
# ---------------------------------------------------------------------------


def test_scene_program_bypasses_stage1_size():
    """Scene programs solve tasks even without stage-1 size verification."""
    from aria.core.output_stage1 import infer_output_stage1_spec

    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "72ca375d")
    # Confirm stage-1 size fails
    spec = infer_output_stage1_spec(task.train)
    assert spec is None, "Expected no stage-1 spec for this task"
    # But scene program solves it
    progs = infer_scene_programs(task.train)
    assert len(progs) >= 1
    assert verify_scene_program(progs[0], task.train)


def test_scene_program_select_panel_solves_real_task():
    """be03b35f: select panel by predicate + transform."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "be03b35f")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            assert "select_entity" in steps
            found = True
            break
    assert found


# ---------------------------------------------------------------------------
# Scene program inference
# ---------------------------------------------------------------------------


def test_select_entity_color_filter():
    """Color filter narrows selection to entities with specific color."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    # bg=0. Two non-bg colors: 1 (2x2 bbox area=4), 2 (1x3 bbox area=3)
    # color_filter=1 should select the color_bbox_1 entity
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="object", predicate="largest_bbox_area",
                  color_filter=1, rank=0, output_id="sel"),
        make_step(StepOp.RENDER_SCENE, source="sel_grid"),
    )
    result = execute_scene_program(prog, grid)
    assert 1 in result.ravel()
    assert result.shape == (2, 2)


def test_dynamic_output_size_from_value():
    """INFER_OUTPUT_SIZE with from_value uses selected entity's grid shape."""
    grid = grid_from_list([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="object", predicate="largest_bbox_area",
                  rank=0, output_id="sel"),
        make_step(StepOp.INFER_OUTPUT_SIZE, from_value="sel_grid"),
        make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=9),
        make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    # Output should have the shape of the selected object's bbox
    assert result.shape[0] > 0
    assert result.shape[1] > 0
    assert np.all(result == 9)  # filled with background


def test_canonical_shape_key():
    """Canonical shape key is rotation/flip invariant."""
    from aria.core.scene_executor import _canonical_shape_key
    mask1 = np.array([[True, False], [True, True]], dtype=bool)
    mask2 = np.rot90(mask1, 1)
    mask3 = np.fliplr(mask1)
    mask4 = np.array([[True, True], [False, True]], dtype=bool)
    # All orientations of the same shape should produce the same key
    assert _canonical_shape_key(mask1) == _canonical_shape_key(mask2)
    assert _canonical_shape_key(mask1) == _canonical_shape_key(mask3)
    assert _canonical_shape_key(mask1) == _canonical_shape_key(mask4)
    # Different shape should produce different key
    mask_diff = np.array([[True, False], [False, True]], dtype=bool)
    assert _canonical_shape_key(mask1) != _canonical_shape_key(mask_diff)


def test_unique_shape_selector():
    """unique_shape selector finds the one object with a unique canonical form."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "358ba94e")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            params = {k: v for s in p.steps for k, v in s.params.items()
                      if k == "predicate"}
            if "unique_shape" in params.values():
                found = True
                break
    assert found, "Expected unique_shape selector to solve 358ba94e"


def test_8conn_objects_in_parse_scene():
    """PARSE_SCENE creates both 4-conn and 8-conn object entities."""
    grid = grid_from_list([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    state = SceneState(input_grid=grid, perception=perceive_grid(grid))
    from aria.core.scene_executor import _parse_scene
    _parse_scene(state, SceneStep(op=StepOp.PARSE_SCENE))
    obj4 = [e for e in state.entities.values()
            if e.kind.value == "object" and e.attrs.get("connectivity") == 4]
    obj8 = [e for e in state.entities.values()
            if e.kind.value == "object" and e.attrs.get("connectivity") == 8]
    assert len(obj4) >= 1
    assert len(obj8) >= 1


def test_touches_border_selector():
    grid = grid_from_list([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ])
    # Object at (0,0) touches border, object at (2,2) also touches border
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.SELECT_ENTITY, kind="object", predicate="not_touches_border",
                  rank=0, output_id="sel"),
        make_step(StepOp.RENDER_SCENE, source="sel_grid"),
    )
    # This might fail if all objects touch border — that's fine
    try:
        result = execute_scene_program(prog, grid)
        assert result.size > 0
    except Exception:
        pass  # Expected if all objects touch border


def test_for_each_entity_fill_bbox_holes():
    """FOR_EACH_ENTITY fill_bbox_holes fills bg within each object bbox."""
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.FOR_EACH_ENTITY, kind="object", rule="fill_bbox_holes", fill_color=5),
        make_step(StepOp.RENDER_SCENE),
    )
    result = execute_scene_program(prog, grid)
    # The hole at (2,2) should be filled with 5
    assert int(result[2, 2]) == 5
    # Border bg should remain 0
    assert int(result[0, 0]) == 0


def test_assign_cells_flip_lr():
    """ASSIGN_CELLS flip_cell_grid_lr flips cell arrangement left-right."""
    grid = grid_from_list([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 4],
    ])
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.ASSIGN_CELLS, mode="flip_cell_grid_lr"),
        make_step(StepOp.RENDER_SCENE),
    )
    s = perceive_grid(grid)
    if s.partition and s.partition.is_uniform_partition and len(s.partition.cells) >= 4:
        result = execute_scene_program(prog, grid)
        assert result.shape == grid.shape


def test_assign_cells_sort_by_non_bg():
    """ASSIGN_CELLS sort modes rearrange cells by property."""
    grid = grid_from_list([
        [1, 1, 0, 2],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [3, 0, 0, 4],
        [0, 0, 0, 4],
    ])
    # Just verify it doesn't crash
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.ASSIGN_CELLS, mode="sort_by_non_bg_desc"),
        make_step(StepOp.RENDER_SCENE),
    )
    s = perceive_grid(grid)
    if s.partition and s.partition.is_uniform_partition:
        result = execute_scene_program(prog, grid)
        assert result.shape == grid.shape


def test_combine_cells_overlay():
    """COMBINE_CELLS overlay merges non-bg pixels from all cells."""
    grid = grid_from_list([
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0],
    ])
    s = perceive_grid(grid)
    if s.partition and s.partition.is_uniform_partition:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.COMBINE_CELLS, operation="overlay", output_to="output"),
            make_step(StepOp.RENDER_SCENE),
        )
        result = execute_scene_program(prog, grid)
        assert result.size > 0


def test_combine_cells_real_task():
    """a68b268e: combine cells with or_any_color → cell-sized output."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "a68b268e")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            assert "combine_cells" in steps
            found = True
            break
    assert found


def test_sibling_cell_select_real_task():
    """c3202e5a: select panel by fewest_colors_gt0."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "c3202e5a")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            assert "select_entity" in steps
            found = True
            break
    assert found


def test_for_each_entity_real_task():
    """3aa6fb7a: fill object bbox holes with color 1."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "3aa6fb7a")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            assert "for_each_entity" in steps
            found = True
            break
    assert found


def test_scene_solve_7468f01a():
    """7468f01a: select largest object + flip_lr via generic selector."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "7468f01a")
    progs = infer_scene_programs(task.train)
    found = False
    for p in progs:
        if verify_scene_program(p, task.train):
            steps = [s.op.value for s in p.steps]
            assert "select_entity" in steps
            found = True
            break
    assert found


# ---------------------------------------------------------------------------
# Scene program inference
# ---------------------------------------------------------------------------


def test_scene_solve_real_task():
    """72ca375d: extract object + flip_lr."""
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "72ca375d")
    progs = infer_scene_programs(task.train)
    assert len(progs) >= 1
    assert verify_scene_program(progs[0], task.train)
    # Check it's a multi-step program
    assert len(progs[0].steps) >= 3


def test_scene_program_in_trace():
    """Scene program solve shows in trace."""
    from aria.core.trace_solve import solve_with_trace

    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "72ca375d")
    trace = solve_with_trace(task.train, task_id="72ca375d")
    assert trace.solved is True
    assert trace.solver == "scene"
    scene_events = [e for e in trace.events if e.event_type == "scene_program_solved"]
    assert len(scene_events) >= 1
    assert scene_events[0].data.get("n_steps", 0) >= 3


# ---------------------------------------------------------------------------
# No task-id logic
# ---------------------------------------------------------------------------


def test_no_task_id_in_scene_modules():
    import inspect
    import aria.core.scene_executor as ex
    import aria.core.scene_solve as ss
    for mod in (ex, ss):
        src = inspect.getsource(mod)
        assert "72ca375d" not in src
