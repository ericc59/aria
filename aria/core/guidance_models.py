"""Small learned guidance models for bounded symbolic decisions.

Numpy-only implementations — no sklearn, no torch.
Models act as proposal rankers, not end-to-end solvers.
Exact verification remains the final arbiter.

Supported models:
- NearestCentroid: per-class mean, classify by distance
- DecisionStump: single best feature threshold
- LinearSoftmax: softmax regression trained with SGD
- MajorityBaseline: always predict most common class

No task-id logic. No solver replacement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aria.core.guidance_datasets import GuidanceDataset, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class ModelEvalResult:
    """Evaluation result for a guidance model."""
    model_name: str
    dataset_name: str
    n_examples: int
    n_classes: int
    top1_accuracy: float
    top3_recall: float
    avg_rank_of_winner: float
    per_class_accuracy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "n_examples": self.n_examples,
            "n_classes": self.n_classes,
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_recall": round(self.top3_recall, 4),
            "avg_rank_of_winner": round(self.avg_rank_of_winner, 2),
            "per_class_accuracy": {k: round(v, 4) for k, v in self.per_class_accuracy.items()},
        }


def evaluate_model(
    model: GuidanceModel,
    dataset: GuidanceDataset,
) -> ModelEvalResult:
    """Evaluate a trained model on a dataset."""
    X, y = dataset.to_arrays()
    if len(X) == 0:
        return ModelEvalResult(
            model_name=model.name, dataset_name=dataset.name,
            n_examples=0, n_classes=dataset.n_classes,
            top1_accuracy=0.0, top3_recall=0.0, avg_rank_of_winner=0.0,
        )

    probs = model.predict_proba(X)  # (n, n_classes)
    preds = np.argmax(probs, axis=1)

    # Top-1 accuracy
    top1 = float(np.mean(preds == y))

    # Top-3 recall
    k = min(3, dataset.n_classes)
    top_k = np.argsort(-probs, axis=1)[:, :k]
    top3_hits = sum(1 for i in range(len(y)) if y[i] in top_k[i])
    top3 = top3_hits / max(len(y), 1)

    # Average rank of true label
    ranks = np.argsort(-probs, axis=1)
    avg_rank = 0.0
    for i in range(len(y)):
        matches = np.where(ranks[i] == y[i])[0]
        rank_of_true = int(matches[0]) if len(matches) > 0 else probs.shape[1] - 1
        avg_rank += rank_of_true
    avg_rank /= max(len(y), 1)

    # Per-class accuracy
    per_class: dict[str, float] = {}
    for c in range(dataset.n_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        class_acc = float(np.mean(preds[mask] == c))
        per_class[dataset.label_names[c]] = class_acc

    return ModelEvalResult(
        model_name=model.name, dataset_name=dataset.name,
        n_examples=len(X), n_classes=dataset.n_classes,
        top1_accuracy=top1, top3_recall=top3,
        avg_rank_of_winner=avg_rank, per_class_accuracy=per_class,
    )


# ---------------------------------------------------------------------------
# Model base
# ---------------------------------------------------------------------------


class GuidanceModel:
    """Base interface for guidance models."""
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) probability matrix."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------


class MajorityBaseline(GuidanceModel):
    """Always predict the most common training class."""
    name = "majority_baseline"

    def __init__(self) -> None:
        self._n_classes = 0
        self._majority = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_classes = int(y.max()) + 1 if len(y) > 0 else 1
        if len(y) > 0:
            counts = np.bincount(y, minlength=self._n_classes)
            self._majority = int(np.argmax(counts))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.full((n, self._n_classes), 1e-6)
        probs[:, self._majority] = 1.0
        return probs / probs.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Nearest centroid
# ---------------------------------------------------------------------------


class NearestCentroid(GuidanceModel):
    """Classify by distance to per-class mean."""
    name = "nearest_centroid"

    def __init__(self) -> None:
        self._centroids: np.ndarray | None = None
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_classes = int(y.max()) + 1 if len(y) > 0 else 1
        self._centroids = np.zeros((self._n_classes, X.shape[1] if len(X) > 0 else 1))
        for c in range(self._n_classes):
            mask = y == c
            if mask.sum() > 0:
                self._centroids[c] = X[mask].mean(axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            return np.ones((len(X), 1))
        # Negative squared distance → softmax-like
        dists = np.zeros((len(X), self._n_classes))
        for c in range(self._n_classes):
            diff = X - self._centroids[c]
            dists[:, c] = -np.sum(diff ** 2, axis=1)
        # Stabilize and softmax
        dists -= dists.max(axis=1, keepdims=True)
        exp_d = np.exp(dists)
        return exp_d / exp_d.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Decision stump
# ---------------------------------------------------------------------------


class DecisionStump(GuidanceModel):
    """Single best feature threshold classifier."""
    name = "decision_stump"

    def __init__(self) -> None:
        self._feature_idx = 0
        self._threshold = 0.0
        self._left_probs: np.ndarray | None = None
        self._right_probs: np.ndarray | None = None
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0:
            return
        self._n_classes = int(y.max()) + 1
        n_features = X.shape[1]

        best_acc = -1.0
        for f in range(n_features):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            # Try midpoints between unique values
            for i in range(len(vals) - 1):
                thresh = (vals[i] + vals[i + 1]) / 2
                left = y[X[:, f] <= thresh]
                right = y[X[:, f] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                left_pred = np.bincount(left, minlength=self._n_classes).argmax()
                right_pred = np.bincount(right, minlength=self._n_classes).argmax()
                acc = (np.sum(left == left_pred) + np.sum(right == right_pred)) / len(y)
                if acc > best_acc:
                    best_acc = acc
                    self._feature_idx = f
                    self._threshold = thresh
                    self._left_probs = np.bincount(left, minlength=self._n_classes).astype(float)
                    self._left_probs /= self._left_probs.sum()
                    self._right_probs = np.bincount(right, minlength=self._n_classes).astype(float)
                    self._right_probs /= self._right_probs.sum()

        if self._left_probs is None:
            # Fallback to uniform
            self._left_probs = np.ones(self._n_classes) / self._n_classes
            self._right_probs = np.ones(self._n_classes) / self._n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._left_probs is None:
            return np.ones((len(X), max(self._n_classes, 1))) / max(self._n_classes, 1)
        result = np.zeros((len(X), self._n_classes))
        for i in range(len(X)):
            if X[i, self._feature_idx] <= self._threshold:
                result[i] = self._left_probs
            else:
                result[i] = self._right_probs
        return result


# ---------------------------------------------------------------------------
# Linear softmax (SGD)
# ---------------------------------------------------------------------------


class LinearSoftmax(GuidanceModel):
    """Softmax regression trained with mini-batch SGD."""
    name = "linear_softmax"

    def __init__(self, lr: float = 0.01, epochs: int = 100, reg: float = 1e-4) -> None:
        self._lr = lr
        self._epochs = epochs
        self._reg = reg
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0:
            return
        n, d = X.shape
        self._n_classes = int(y.max()) + 1
        k = self._n_classes

        # Normalize features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        Xn = (X - self._mean) / self._std

        self._W = np.zeros((d, k))
        self._b = np.zeros(k)

        for epoch in range(self._epochs):
            # Forward
            logits = Xn @ self._W + self._b
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            # One-hot targets
            targets = np.zeros((n, k))
            targets[np.arange(n), y] = 1.0

            # Gradient
            grad = probs - targets  # (n, k)
            dW = (Xn.T @ grad) / n + self._reg * self._W
            db = grad.mean(axis=0)

            self._W -= self._lr * dW
            self._b -= self._lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            return np.ones((len(X), max(self._n_classes, 1))) / max(self._n_classes, 1)
        Xn = (X - self._mean) / self._std
        logits = Xn @ self._W + self._b
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Train + evaluate pipeline
# ---------------------------------------------------------------------------


@dataclass
class SubproblemReport:
    """Report comparing models on one subproblem."""
    dataset_name: str
    n_train: int
    n_eval: int
    n_classes: int
    label_distribution: dict[str, int]
    results: list[ModelEvalResult]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_classes": self.n_classes,
            "label_distribution": self.label_distribution,
            "results": [r.to_dict() for r in self.results],
        }


def train_and_evaluate(
    dataset: GuidanceDataset,
    train_frac: float = 0.8,
    seed: int = 42,
) -> SubproblemReport:
    """Train all models on a dataset and evaluate with a train/eval split.

    Uses a simple random split (not task-aware).
    """
    X, y = dataset.to_arrays()
    n = len(X)

    if n < 4:
        # Too few examples for meaningful train/eval
        return SubproblemReport(
            dataset_name=dataset.name,
            n_train=n, n_eval=0,
            n_classes=dataset.n_classes,
            label_distribution=dataset.label_distribution(),
            results=[],
        )

    # Random split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * train_frac)
    train_idx = indices[:split]
    eval_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_eval, y_eval = X[eval_idx], y[eval_idx]

    # Build eval dataset
    eval_ds = GuidanceDataset(
        name=dataset.name,
        label_names=dataset.label_names,
        examples=[dataset.examples[i] for i in eval_idx],
    )

    models: list[GuidanceModel] = [
        MajorityBaseline(),
        NearestCentroid(),
        DecisionStump(),
        LinearSoftmax(lr=0.05, epochs=200),
    ]

    results = []
    for model in models:
        model.fit(X_train, y_train)
        result = evaluate_model(model, eval_ds)
        results.append(result)

    return SubproblemReport(
        dataset_name=dataset.name,
        n_train=len(train_idx),
        n_eval=len(eval_idx),
        n_classes=dataset.n_classes,
        label_distribution=dataset.label_distribution(),
        results=results,
    )
