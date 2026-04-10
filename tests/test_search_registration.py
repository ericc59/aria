from aria.datasets import get_dataset, load_arc_task
import numpy as np

from aria.search.registration import (
    base_registration_patch,
    cluster_movable_modules,
    extract_anchored_shapes,
    module_anchor_patch,
    overlay_registration_candidates,
)


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


def test_overlay_registration_candidates_include_exact_train0_solution():
    task = load_arc_task(get_dataset("v2-eval"), "20270e3b")
    ex = task.train[0]
    shapes = extract_anchored_shapes(ex.input, shape_color=4, anchor_color=7)
    base_idx, modules = cluster_movable_modules(shapes)
    base_patch, target_sites = base_registration_patch(shapes[base_idx], shape_color=4)
    module_patch, module_mask, source_anchors = module_anchor_patch(
        ex.input,
        shapes,
        modules[0],
        shape_color=4,
        anchor_color=7,
    )
    candidates = overlay_registration_candidates(
        base_patch,
        target_sites,
        module_patch,
        module_mask,
        source_anchors,
        bg_color=1,
        shape_color=4,
        anchor_color=7,
    )
    assert any(np.array_equal(candidate.canvas, ex.output) for candidate in candidates)


def test_overlay_registration_candidates_include_train3_registered_shift():
    task = load_arc_task(get_dataset("v2-eval"), "20270e3b")
    ex = task.train[3]
    shapes = extract_anchored_shapes(ex.input, shape_color=4, anchor_color=7)
    base_idx, modules = cluster_movable_modules(shapes)
    base_patch, target_sites = base_registration_patch(shapes[base_idx], shape_color=4)
    module_patch, module_mask, source_anchors = module_anchor_patch(
        ex.input,
        shapes,
        modules[0],
        shape_color=4,
        anchor_color=7,
    )
    candidates = overlay_registration_candidates(
        base_patch,
        target_sites,
        module_patch,
        module_mask,
        source_anchors,
        bg_color=1,
        shape_color=4,
        anchor_color=7,
    )
    assert any(
        candidate.shift_row == 2
        and candidate.shift_col == 5
        and candidate.target_site == (3, 5)
        and candidate.source_anchor == (1, 0)
        and candidate.canvas.shape == ex.output.shape
        for candidate in candidates
    )
