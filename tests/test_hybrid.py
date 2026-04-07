"""Tests for the DEPRECATED hybrid neural-symbolic solver.

These tests exercise the experimental stepper-op ranking path, which is
NOT part of the canonical architecture. They are preserved to prevent
regressions in the experimental code, not as examples of the right approach.

The canonical architecture is: aria.core.graph + aria.core.protocol
(fit -> specialize -> compile -> verify).
"""

from __future__ import annotations

import numpy as np

from aria.core.experimental.hybrid import HybridSolver
from aria.core.experimental.neural import OpPredictor, TaskFeatures, TrainingExample, extract_features
from aria.core.world import build_world_model
from aria.types import DemoPair, grid_from_list
from aria.verify.verifier import verify


def _rotate_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _fill_task():
    return (
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [8, 0, 0, 8], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [8, 3, 3, 8], [0, 0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 8, 0], [0, 0, 0], [0, 8, 0]]),
            output=grid_from_list([[0, 8, 0], [0, 3, 0], [0, 8, 0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_feature_extraction():
    demos = _rotate_task()
    world = build_world_model(demos, task_id="test")
    features = extract_features(world)
    assert isinstance(features, TaskFeatures)
    assert features.same_dims is True
    arr = features.to_array()
    assert arr.shape == (26,)
    assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# Op predictor
# ---------------------------------------------------------------------------


def test_predictor_untrained_returns_empty():
    pred = OpPredictor()
    features = TaskFeatures(
        rows=3, cols=3, same_dims=True, change_fraction=0.5,
        pixels_added=0, pixels_removed=0, pixels_modified=4,
        n_input_colors=2, n_output_colors=2, n_added_colors=0, n_removed_colors=0,
        n_input_objects=1, n_output_objects=1, objects_added=0, objects_removed=0,
        same_object_count=True, has_frames=False, n_frames=0,
        has_symmetry_h=False, has_symmetry_v=False,
        is_color_map=True, is_additive=False, is_subtractive=False,
        changes_inside_frames=False, n_demos=2, bg_color=0,
    )
    scores = pred.predict_op_scores(features)
    assert scores == {}


def test_predictor_trains_from_examples():
    pred = OpPredictor()

    # Add some examples
    for i in range(5):
        features = TaskFeatures(
            rows=3+i, cols=3+i, same_dims=True, change_fraction=0.3,
            pixels_added=0, pixels_removed=0, pixels_modified=3,
            n_input_colors=2, n_output_colors=2, n_added_colors=0, n_removed_colors=0,
            n_input_objects=1, n_output_objects=1, objects_added=0, objects_removed=0,
            same_object_count=True, has_frames=False, n_frames=0,
            has_symmetry_h=False, has_symmetry_v=False,
            is_color_map=True, is_additive=False, is_subtractive=False,
            changes_inside_frames=False, n_demos=2, bg_color=0,
        )
        pred.add_example(TrainingExample(
            task_id=f"task_{i}",
            features=features,
            winning_ops=("apply_color_map",),
        ))

    pred.train()
    assert pred._trained

    scores = pred.predict_op_scores(features)
    assert "apply_color_map" in scores
    assert scores["apply_color_map"] > 0


# ---------------------------------------------------------------------------
# Hybrid solver
# ---------------------------------------------------------------------------


def test_hybrid_bootstrap_solves():
    solver = HybridSolver(beam_width=3, max_steps=4)
    tasks = [
        ("rotate", _rotate_task()),
        ("fill", _fill_task()),
    ]
    n = solver.bootstrap(tasks)
    assert n >= 1
    assert len(solver._solved) >= 1


def test_hybrid_train_after_bootstrap():
    solver = HybridSolver(beam_width=3, max_steps=4)
    tasks = [("rotate", _rotate_task()), ("fill", _fill_task())]
    solver.bootstrap(tasks)
    solver.train()
    assert solver.predictor.n_examples >= 1


def test_hybrid_programs_verify():
    solver = HybridSolver(beam_width=3, max_steps=4)
    tasks = [("rotate", _rotate_task()), ("fill", _fill_task())]
    solver.bootstrap(tasks)

    for tid, prog in solver._solved.items():
        demos = dict(tasks)[tid]
        vr = verify(prog, demos)
        assert vr.passed, f"{tid} program should verify"
