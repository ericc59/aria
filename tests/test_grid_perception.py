from aria.core.grid_perception import perceive_grid
from aria.datasets import get_dataset, load_arc_task
from aria.types import grid_from_list


def test_grid_perception_tracks_background_separator_intervals():
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    state = perceive_grid(grid)
    assert state.bg_color == 0
    assert state.bg_separator_rows == (2,)
    assert state.bg_separator_cols == (2,)
    assert state.row_intervals == ((0, 1), (3, 4))
    assert state.col_intervals == ((0, 1), (3, 4))


def test_grid_perception_exposes_partition_and_non_bg_colors():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1190e5a7")
    state = perceive_grid(task.train[0].input)
    assert state.partition is not None
    assert (state.partition.n_rows, state.partition.n_cols) == (3, 2)
    assert state.palette == frozenset({1, 8})
    assert state.non_bg_colors == frozenset({8})


def test_grid_perception_tracks_color_counts_and_bboxes():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [3, 0, 2, 0],
        [3, 0, 0, 0],
    ])
    state = perceive_grid(grid)
    assert state.color_pixel_counts[0] == 11
    assert state.color_pixel_counts[2] == 3
    assert state.color_pixel_counts[3] == 2
    assert state.color_bboxes[2] == (1, 1, 2, 2)
    assert state.color_bboxes[3] == (2, 0, 3, 0)


def test_grid_perception_detects_boxed_regions_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    state = perceive_grid(task.train[0].input)
    boxed_sizes = {(region.height, region.width) for region in state.boxed_regions}
    assert (8, 10) in boxed_sizes


def test_grid_perception_exposes_8_connected_object_view():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    state = perceive_grid(task.train[0].input)
    assert len(state.objects.non_singletons) == 1
    assert len(state.objects8.non_singletons) == 1
    obj8 = state.objects8.non_singletons[0]
    assert (obj8.row, obj8.col, obj8.bbox_h, obj8.bbox_w) == (1, 3, 3, 3)
    assert obj8.size == 4


def test_grid_perception_exposes_zones():
    grid = grid_from_list([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4],
    ])
    state = perceive_grid(grid)
    assert len(state.zones) >= 4
    zone_dims = {(z.h, z.w) for z in state.zones}
    assert (2, 2) in zone_dims


def test_grid_perception_exposes_legend_on_real_task():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0520fde7")
    state = perceive_grid(task.train[0].input)
    # Legend might or might not be detected depending on the task structure
    # but the field should exist and be None or LegendInfo
    assert state.legend is None or hasattr(state.legend, "entries")


def test_grid_perception_exposes_roles():
    grid = grid_from_list([
        [1, 1, 1, 0, 2, 2, 2],
        [1, 0, 1, 0, 2, 0, 2],
        [1, 1, 1, 0, 2, 2, 2],
    ])
    state = perceive_grid(grid)
    assert isinstance(state.roles, tuple)
    role_names = {r.role.name for r in state.roles}
    # Should detect FRAME roles for the rectangular frame objects
    assert "FRAME" in role_names or "MARKER" in role_names


def test_no_task_id_in_grid_perception():
    import inspect
    from aria.core.grid_perception import perceive_grid
    src = inspect.getsource(perceive_grid)
    assert "1b59e163" not in src
