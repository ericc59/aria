"""DEPRECATED — Learned operation predictor for hybrid solving.

This module is superseded by the canonical graph pipeline.
The stepper-op ranking approach was an experiment; the future
learned path is the per-task recurrent graph editor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskFeatures:
    """Fixed-size feature vector extracted from task demos."""
    # Shape
    rows: int
    cols: int
    same_dims: bool
    # Pixel changes
    change_fraction: float
    pixels_added: int
    pixels_removed: int
    pixels_modified: int
    # Colors
    n_input_colors: int
    n_output_colors: int
    n_added_colors: int
    n_removed_colors: int
    # Objects
    n_input_objects: int
    n_output_objects: int
    objects_added: int
    objects_removed: int
    same_object_count: bool
    # Structure
    has_frames: bool
    n_frames: int
    has_symmetry_h: bool
    has_symmetry_v: bool
    # Roles
    is_color_map: bool
    is_additive: bool
    is_subtractive: bool
    changes_inside_frames: bool
    # Derived
    n_demos: int
    bg_color: int

    def to_array(self) -> np.ndarray:
        """Convert to a fixed-size float array for sklearn."""
        return np.array([
            self.rows, self.cols, float(self.same_dims),
            self.change_fraction,
            self.pixels_added, self.pixels_removed, self.pixels_modified,
            self.n_input_colors, self.n_output_colors,
            self.n_added_colors, self.n_removed_colors,
            self.n_input_objects, self.n_output_objects,
            self.objects_added, self.objects_removed,
            float(self.same_object_count),
            float(self.has_frames), self.n_frames,
            float(self.has_symmetry_h), float(self.has_symmetry_v),
            float(self.is_color_map), float(self.is_additive),
            float(self.is_subtractive), float(self.changes_inside_frames),
            self.n_demos, self.bg_color,
        ], dtype=np.float32)


def extract_features(world: Any) -> TaskFeatures:
    """Extract a TaskFeatures vector from a WorldModel."""
    return TaskFeatures(
        rows=world.pixels.dims[0],
        cols=world.pixels.dims[1],
        same_dims=world.pixels.same_dims,
        change_fraction=world.pixels.change_fraction,
        pixels_added=world.roles.pixels_added,
        pixels_removed=world.roles.pixels_removed,
        pixels_modified=world.roles.pixels_modified,
        n_input_colors=len(world.pixels.input_palette),
        n_output_colors=len(world.pixels.output_palette),
        n_added_colors=len(world.pixels.added_colors),
        n_removed_colors=len(world.pixels.removed_colors),
        n_input_objects=len(world.objects.input_objects),
        n_output_objects=len(world.objects.output_objects),
        objects_added=world.objects.objects_added,
        objects_removed=world.objects.objects_removed,
        same_object_count=world.objects.same_count,
        has_frames=world.structure.has_frames,
        n_frames=len(world.structure.framed_regions),
        has_symmetry_h=world.structure.has_symmetry_h,
        has_symmetry_v=world.structure.has_symmetry_v,
        is_color_map=world.roles.is_color_map,
        is_additive=world.roles.is_additive,
        is_subtractive=world.roles.is_subtractive,
        changes_inside_frames=world.roles.changes_inside_frames,
        n_demos=world.n_demos,
        bg_color=world.pixels.bg_color,
    )


# ---------------------------------------------------------------------------
# Training data: (features, winning_ops) from solved tasks
# ---------------------------------------------------------------------------


@dataclass
class TrainingExample:
    """One solved task: features + which ops were in the solution."""
    task_id: str
    features: TaskFeatures
    winning_ops: tuple[str, ...]  # op names from the solution steps


class OpPredictor:
    """Lightweight learned model: task features → op probabilities.

    Uses a multi-label classifier (one per known op) to predict which
    ops are likely useful for a task, given its structural features.
    Falls back to uniform scores when untrained or for unknown ops.
    """

    def __init__(self) -> None:
        self._examples: list[TrainingExample] = []
        self._known_ops: set[str] = set()
        self._trained = False
        self._models: dict[str, Any] = {}  # op_name -> classifier
        self._op_prior: dict[str, float] = {}  # op_name -> base frequency

    def add_example(self, example: TrainingExample) -> None:
        """Add a solved task to the training set."""
        self._examples.append(example)
        self._known_ops.update(example.winning_ops)
        self._trained = False  # invalidate

    @property
    def n_examples(self) -> int:
        return len(self._examples)

    def train(self) -> None:
        """Train the model from accumulated examples.

        Uses a simple decision tree per op — lightweight, interpretable,
        works with as few as 5 examples.
        """
        from sklearn.tree import DecisionTreeClassifier

        if len(self._examples) < 3:
            self._trained = False
            return

        X = np.array([e.features.to_array() for e in self._examples])

        for op in self._known_ops:
            y = np.array([
                1 if op in e.winning_ops else 0
                for e in self._examples
            ])
            # Only train if we have positive examples
            if y.sum() == 0:
                continue

            self._op_prior[op] = float(y.mean())

            # Simple decision tree — handles small datasets well
            clf = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=max(1, len(self._examples) // 5),
            )
            try:
                clf.fit(X, y)
                self._models[op] = clf
            except Exception:
                pass

        self._trained = True

    def predict_op_scores(self, features: TaskFeatures) -> dict[str, float]:
        """Predict likelihood of each known op for a task.

        Returns {op_name: score} where score is 0-1 probability.
        Higher = more likely to help.
        """
        if not self._trained or not self._models:
            return {}

        x = features.to_array().reshape(1, -1)
        scores: dict[str, float] = {}

        for op, clf in self._models.items():
            try:
                proba = clf.predict_proba(x)
                # probability of class 1 (op is useful)
                if proba.shape[1] >= 2:
                    scores[op] = float(proba[0, 1])
                else:
                    scores[op] = self._op_prior.get(op, 0.0)
            except Exception:
                scores[op] = self._op_prior.get(op, 0.0)

        return scores

    def rank_candidates(
        self,
        candidates: list,
        features: TaskFeatures,
    ) -> list:
        """Reorder candidates by predicted likelihood. Best first."""
        scores = self.predict_op_scores(features)
        if not scores:
            return candidates  # no model, no reordering

        def _score(cand: Any) -> float:
            op = cand.op
            # Normalize compound op names
            if op.startswith("__compound_"):
                op = op.replace("__compound_", "").rstrip("_")
            # Check if any scored op is a substring (handles op variants)
            best = 0.0
            for scored_op, s in scores.items():
                if scored_op in op or op in scored_op:
                    best = max(best, s)
            # Also check description
            for scored_op, s in scores.items():
                if scored_op in cand.description:
                    best = max(best, s)
            return best

        return sorted(candidates, key=_score, reverse=True)
