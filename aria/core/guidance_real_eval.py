"""Real-data evaluation of learned guidance models.

Task-aware splits, real ARC dataset generation, and comparison of
learned models against current symbolic heuristic ordering.

No task-id logic. No benchmark hacks. Exact verification is truth.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from aria.core.guidance_datasets import (
    FEATURE_NAMES,
    GuidanceDataset,
    GuidanceExample,
    extract_perception_features,
    features_to_array,
)
from aria.core.guidance_models import (
    DecisionStump,
    GuidanceModel,
    LinearSoftmax,
    MajorityBaseline,
    ModelEvalResult,
    NearestCentroid,
    evaluate_model,
)
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Task-aware splits
# ---------------------------------------------------------------------------


def task_aware_split(
    dataset: GuidanceDataset,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[GuidanceDataset, GuidanceDataset]]:
    """Split dataset into folds where all examples from the same task
    stay in the same fold. Returns list of (train, test) pairs.

    This prevents task-structure leakage across train/test.
    """
    # Group examples by task_id
    task_groups: dict[str, list[int]] = {}
    for i, ex in enumerate(dataset.examples):
        task_groups.setdefault(ex.task_id, []).append(i)

    # Shuffle task ids
    task_ids = list(task_groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)

    # Assign tasks to folds
    fold_assignments: dict[str, int] = {}
    for i, tid in enumerate(task_ids):
        fold_assignments[tid] = i % n_folds

    folds = []
    for fold_idx in range(n_folds):
        test_indices = []
        train_indices = []
        for tid, indices in task_groups.items():
            if fold_assignments[tid] == fold_idx:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)

        if not test_indices or not train_indices:
            continue

        train_ds = GuidanceDataset(
            name=dataset.name,
            label_names=dataset.label_names,
            examples=[dataset.examples[i] for i in train_indices],
        )
        test_ds = GuidanceDataset(
            name=dataset.name,
            label_names=dataset.label_names,
            examples=[dataset.examples[i] for i in test_indices],
        )
        folds.append((train_ds, test_ds))

    return folds


# ---------------------------------------------------------------------------
# Heuristic baseline: current symbolic default ordering
# ---------------------------------------------------------------------------


class SymbolicDefaultModel(GuidanceModel):
    """Emulates the current symbolic system's default candidate ordering.

    For output-size: same_dims > scale > partition_derived > frame_derived > ...
    For periodic axis: row first (the current default)
    For periodic period: 2 first (the current default)
    For transform: rotate_90 first (tried first by fitter)
    For lane: order from mechanism_evidence ranking
    """
    name = "symbolic_default"

    def __init__(self, dataset_name: str, label_names: tuple[str, ...]) -> None:
        self._dataset_name = dataset_name
        self._label_names = label_names
        self._n_classes = len(label_names)
        self._default_scores = self._compute_defaults()

    def _compute_defaults(self) -> np.ndarray:
        """Return a static score vector reflecting current default ordering."""
        scores = np.ones(self._n_classes) * 0.1

        if self._dataset_name == "output_size_mode":
            priority = ["same_dims", "scale", "partition_derived", "frame_derived",
                        "bbox_derived", "structure_derived", "transpose", "additive", "fixed", "other"]
            for i, name in enumerate(self._label_names):
                if name in priority:
                    scores[i] = 1.0 - 0.05 * priority.index(name)

        elif self._dataset_name == "periodic_axis":
            for i, name in enumerate(self._label_names):
                if name == "row":
                    scores[i] = 0.8
                elif name == "col":
                    scores[i] = 0.5

        elif self._dataset_name == "periodic_period":
            for i, name in enumerate(self._label_names):
                try:
                    p = int(name)
                    scores[i] = 1.0 - (p - 2) * 0.15
                except ValueError:
                    pass

        elif self._dataset_name == "transform_choice":
            priority = ["rotate_90", "rotate_180", "rotate_270",
                        "reflect_row", "reflect_col", "transpose"]
            for i, name in enumerate(self._label_names):
                if name in priority:
                    scores[i] = 1.0 - 0.1 * priority.index(name)

        elif self._dataset_name == "lane_ranking":
            priority = ["periodic_repair", "replication", "relocation", "grid_transform"]
            for i, name in enumerate(self._label_names):
                if name in priority:
                    scores[i] = 1.0 - 0.15 * priority.index(name)

        return scores

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass  # static, no learning

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.tile(self._default_scores, (n, 1))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Cross-validated evaluation result for one model."""
    model_name: str
    n_folds: int
    mean_top1: float
    std_top1: float
    mean_top3: float
    mean_avg_rank: float
    fold_results: list[ModelEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "mean_top1": round(self.mean_top1, 4),
            "std_top1": round(self.std_top1, 4),
            "mean_top3": round(self.mean_top3, 4),
            "mean_avg_rank": round(self.mean_avg_rank, 2),
        }


@dataclass
class SubproblemRealReport:
    """Real-data evaluation for one subproblem."""
    dataset_name: str
    n_tasks: int
    n_examples: int
    n_classes: int
    label_distribution: dict[str, int]
    sufficient_data: bool
    reason: str = ""
    cv_results: list[CVResult] = field(default_factory=list)
    integration_ready: bool = False
    integration_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_tasks": self.n_tasks,
            "n_examples": self.n_examples,
            "n_classes": self.n_classes,
            "label_distribution": self.label_distribution,
            "sufficient_data": self.sufficient_data,
            "reason": self.reason,
            "cv_results": [r.to_dict() for r in self.cv_results],
            "integration_ready": self.integration_ready,
            "integration_reason": self.integration_reason,
        }


def evaluate_on_real_data(
    dataset: GuidanceDataset,
    n_folds: int = 5,
    seed: int = 42,
    min_examples: int = 10,
    min_classes_with_support: int = 2,
) -> SubproblemRealReport:
    """Full real-data evaluation with task-aware CV."""

    # Count unique tasks
    task_ids = set(ex.task_id for ex in dataset.examples)
    label_dist = dataset.label_distribution()

    report = SubproblemRealReport(
        dataset_name=dataset.name,
        n_tasks=len(task_ids),
        n_examples=dataset.n_examples,
        n_classes=dataset.n_classes,
        label_distribution=label_dist,
        sufficient_data=True,
    )

    # Check sufficiency
    classes_with_support = sum(1 for v in label_dist.values() if v >= 3)
    if dataset.n_examples < min_examples:
        report.sufficient_data = False
        report.reason = f"too few examples ({dataset.n_examples} < {min_examples})"
        return report
    if classes_with_support < min_classes_with_support:
        report.sufficient_data = False
        report.reason = f"too few classes with >=3 examples ({classes_with_support})"
        return report

    # Task-aware CV
    folds = task_aware_split(dataset, n_folds=n_folds, seed=seed)
    if len(folds) < 2:
        report.sufficient_data = False
        report.reason = "insufficient folds after task-aware split"
        return report

    # Models to evaluate
    model_factories: list[tuple[str, Callable]] = [
        ("symbolic_default", lambda: SymbolicDefaultModel(dataset.name, dataset.label_names)),
        ("majority_baseline", MajorityBaseline),
        ("nearest_centroid", NearestCentroid),
        ("decision_stump", DecisionStump),
        ("linear_softmax", lambda: LinearSoftmax(lr=0.05, epochs=200)),
    ]

    for model_name, factory in model_factories:
        fold_results = []
        for train_ds, test_ds in folds:
            X_train, y_train = train_ds.to_arrays()
            model = factory()
            model.fit(X_train, y_train)
            result = evaluate_model(model, test_ds)
            fold_results.append(result)

        top1s = [r.top1_accuracy for r in fold_results]
        top3s = [r.top3_recall for r in fold_results]
        ranks = [r.avg_rank_of_winner for r in fold_results]

        report.cv_results.append(CVResult(
            model_name=model_name,
            n_folds=len(folds),
            mean_top1=float(np.mean(top1s)),
            std_top1=float(np.std(top1s)),
            mean_top3=float(np.mean(top3s)),
            mean_avg_rank=float(np.mean(ranks)),
            fold_results=fold_results,
        ))

    # Integration readiness: learned model must beat symbolic default
    _assess_integration_readiness(report)

    return report


def _assess_integration_readiness(report: SubproblemRealReport) -> None:
    """Determine if any learned model is ready for integration."""
    if not report.sufficient_data:
        report.integration_ready = False
        report.integration_reason = report.reason
        return

    # Find symbolic default and best learned
    default_result = None
    best_learned = None
    best_learned_name = ""

    for cv in report.cv_results:
        if cv.model_name == "symbolic_default":
            default_result = cv
        elif cv.model_name not in ("majority_baseline",):
            if best_learned is None or cv.mean_top1 > best_learned.mean_top1:
                best_learned = cv
                best_learned_name = cv.model_name

    if default_result is None or best_learned is None:
        report.integration_ready = False
        report.integration_reason = "missing baseline comparison"
        return

    gain = best_learned.mean_top1 - default_result.mean_top1
    if gain > 0.05 and best_learned.mean_top1 > 0.5:
        report.integration_ready = True
        report.integration_reason = (
            f"{best_learned_name} beats symbolic_default by {gain:.1%} "
            f"({best_learned.mean_top1:.1%} vs {default_result.mean_top1:.1%})"
        )
    elif gain > 0:
        report.integration_ready = False
        report.integration_reason = (
            f"{best_learned_name} slightly better ({gain:.1%}) but not enough "
            f"for confident integration"
        )
    else:
        report.integration_ready = False
        report.integration_reason = (
            f"symbolic_default ({default_result.mean_top1:.1%}) matches or beats "
            f"best learned ({best_learned.mean_top1:.1%})"
        )


# ---------------------------------------------------------------------------
# Shadow-mode trace fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowComparison:
    """Shadow comparison of model vs default ordering for one candidate decision."""
    subproblem: str
    default_rank: int           # rank of winner under default ordering
    model_rank: int             # rank of winner under model ordering
    model_name: str
    winner_label: str
    rank_improvement: int       # default_rank - model_rank (positive = model is better)
    would_save_attempts: int    # how many candidates skipped if model ordering was used

    def to_dict(self) -> dict:
        return {
            "subproblem": self.subproblem,
            "default_rank": self.default_rank,
            "model_rank": self.model_rank,
            "model_name": self.model_name,
            "winner_label": self.winner_label,
            "rank_improvement": self.rank_improvement,
            "would_save_attempts": self.would_save_attempts,
        }

    @staticmethod
    def from_dict(d: dict) -> ShadowComparison:
        return ShadowComparison(**d)


def compute_shadow_comparisons(
    dataset: GuidanceDataset,
    model: GuidanceModel,
    default_model: SymbolicDefaultModel,
) -> list[ShadowComparison]:
    """Compare model ordering vs default ordering for each example."""
    X, y = dataset.to_arrays()
    if len(X) == 0:
        return []

    model_probs = model.predict_proba(X)
    default_probs = default_model.predict_proba(X)

    comparisons = []
    for i in range(len(X)):
        true_label = y[i]
        label_name = dataset.label_names[true_label]

        model_order = np.argsort(-model_probs[i])
        default_order = np.argsort(-default_probs[i])

        model_rank = int(np.where(model_order == true_label)[0][0])
        default_rank = int(np.where(default_order == true_label)[0][0])

        comparisons.append(ShadowComparison(
            subproblem=dataset.name,
            default_rank=default_rank,
            model_rank=model_rank,
            model_name=model.name,
            winner_label=label_name,
            rank_improvement=default_rank - model_rank,
            would_save_attempts=max(0, default_rank - model_rank),
        ))

    return comparisons


def summarize_shadow(comparisons: list[ShadowComparison]) -> dict[str, Any]:
    """Summarize shadow comparison results."""
    if not comparisons:
        return {"n": 0}

    improvements = [c.rank_improvement for c in comparisons]
    saves = [c.would_save_attempts for c in comparisons]

    return {
        "n": len(comparisons),
        "mean_rank_improvement": round(float(np.mean(improvements)), 2),
        "median_rank_improvement": round(float(np.median(improvements)), 2),
        "pct_improved": round(sum(1 for x in improvements if x > 0) / len(improvements), 4),
        "pct_same": round(sum(1 for x in improvements if x == 0) / len(improvements), 4),
        "pct_worse": round(sum(1 for x in improvements if x < 0) / len(improvements), 4),
        "mean_attempts_saved": round(float(np.mean(saves)), 2),
        "total_attempts_saved": sum(saves),
    }


# ---------------------------------------------------------------------------
# Full pipeline: generate + evaluate + report
# ---------------------------------------------------------------------------


def run_real_evaluation(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
    subproblems: list[str] | None = None,
) -> dict[str, SubproblemRealReport]:
    """Run full real-data evaluation pipeline."""
    from aria.core.guidance_datasets import (
        build_output_size_dataset,
        build_periodic_axis_dataset,
        build_periodic_period_dataset,
        build_transform_dataset,
        build_lane_ranking_dataset,
    )

    builders = {
        "output_size_mode": build_output_size_dataset,
        "periodic_axis": build_periodic_axis_dataset,
        "periodic_period": build_periodic_period_dataset,
        "transform_choice": build_transform_dataset,
        "lane_ranking": build_lane_ranking_dataset,
    }

    if subproblems is None:
        subproblems = list(builders.keys())

    reports = {}
    for name in subproblems:
        if name not in builders:
            continue
        ds = builders[name](task_demo_pairs)
        report = evaluate_on_real_data(ds)
        reports[name] = report

    return reports
