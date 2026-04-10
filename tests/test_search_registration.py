from aria.datasets import get_dataset, load_arc_task
from aria.search.registration import cluster_movable_modules, extract_anchored_shapes


def test_extract_anchored_shapes_recovers_20270e3b_components():
    task = load_arc_task(get_dataset("v2-eval"), "20270e3b")
    ex = task.train[0]
    shapes = extract_anchored_shapes(ex.input, shape_color=4, anchor_color=7)
    assert [(shape.row, shape.col, shape.height, shape.width) for shape in shapes] == [
        (0, 0, 7, 8),
        (1, 9, 3, 4),
        (4, 10, 1, 2),
    ]
    assert shapes[0].anchors_global == ((1, 3), (1, 4), (1, 5))
    assert shapes[1].anchors_global == ((0, 9), (0, 10), (0, 11))
    assert shapes[2].anchors_global == ()


def test_cluster_movable_modules_attaches_small_neighbor_to_anchored_shape():
    task = load_arc_task(get_dataset("v2-eval"), "20270e3b")
    ex = task.train[0]
    shapes = extract_anchored_shapes(ex.input, shape_color=4, anchor_color=7)
    base_idx, modules = cluster_movable_modules(shapes)
    assert base_idx == 0
    assert [module.component_indices for module in modules] == [(1, 2)]
    assert [module.anchored for module in modules] == [True]


def test_cluster_movable_modules_keeps_20270e3b_train3_source_separate_from_base():
    task = load_arc_task(get_dataset("v2-eval"), "20270e3b")
    ex = task.train[3]
    shapes = extract_anchored_shapes(ex.input, shape_color=4, anchor_color=7)
    base_idx, modules = cluster_movable_modules(shapes)
    assert base_idx == 0
    assert [module.component_indices for module in modules] == [(1,)]
    assert modules[0].anchored is True
