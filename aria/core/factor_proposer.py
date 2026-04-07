"""Factorized multi-head proposer.

Predicts 6 independent factor distributions from cross-demo features.
Joint probability = product of per-factor marginals.
Uses existing LinearSoftmax / NearestCentroid from guidance_models.

No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as cart_product
from typing import Any, Sequence

import numpy as np

from aria.core.guidance_proposer import extract_cross_demo_features
from aria.factors import (
    Correspondence,
    Decomposition,
    Depth,
    FactorSet,
    Op,
    Scope,
    Selector,
    FACTOR_ENUMS,
    FACTOR_NAMES,
    is_compatible,
)
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Per-factor head
# ---------------------------------------------------------------------------


@dataclass
class FactorHead:
    """One classification head for a single factor dimension."""
    factor_name: str
    n_classes: int
    class_names: tuple[str, ...]
    weights: np.ndarray | None = None  # (n_features, n_classes) or None
    bias: np.ndarray | None = None     # (n_classes,)
    trained: bool = False

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for one sample (1D feature vector).

        Returns array of shape (n_classes,).
        """
        if not self.trained or self.weights is None:
            # Uniform prior
            return np.ones(self.n_classes) / self.n_classes

        logits = x @ self.weights + self.bias
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def fit(self, X: np.ndarray, y: np.ndarray, *, lr: float = 0.01, epochs: int = 200) -> None:
        """Train linear softmax via SGD."""
        if len(X) == 0:
            return
        n_features = X.shape[1]
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            # One-hot targets
            targets = np.zeros_like(probs)
            targets[np.arange(len(y)), y] = 1.0

            # Gradient
            grad = probs - targets  # (n, n_classes)
            self.weights -= lr * (X.T @ grad) / len(X)
            self.bias -= lr * grad.mean(axis=0)

        self.trained = True


# ---------------------------------------------------------------------------
# Multi-head proposer
# ---------------------------------------------------------------------------


@dataclass
class FactorProposer:
    """6-head independent factor proposer."""
    heads: dict[str, FactorHead] = field(default_factory=dict)

    def __post_init__(self):
        if not self.heads:
            for name in FACTOR_NAMES:
                enum_cls = FACTOR_ENUMS[name]
                members = list(enum_cls)
                self.heads[name] = FactorHead(
                    factor_name=name,
                    n_classes=len(members),
                    class_names=tuple(m.value if hasattr(m, 'value') else str(m) for m in members),
                )

    @property
    def trained(self) -> bool:
        return all(h.trained for h in self.heads.values())

    def predict_factor_probs(
        self, features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict per-factor probability distributions.

        Args:
            features: 1D feature vector (from extract_cross_demo_features)

        Returns:
            dict mapping factor name → probability array
        """
        return {
            name: head.predict_proba(features)
            for name, head in self.heads.items()
        }

    def top_k_factor_sets(
        self,
        features: np.ndarray,
        k: int = 50,
        *,
        per_factor_k: int = 3,
    ) -> list[tuple[FactorSet, float]]:
        """Return top-k compatible factor combinations by joint probability.

        Strategy: take top per_factor_k values per factor, form all combos,
        filter by compatibility, rank by joint probability, return top k.
        """
        probs = self.predict_factor_probs(features)

        # Get top-k indices per factor
        top_indices: dict[str, list[int]] = {}
        for name in FACTOR_NAMES:
            p = probs[name]
            n = min(per_factor_k, len(p))
            top_indices[name] = list(np.argsort(-p)[:n])

        # Enumerate combos from top indices
        enum_lists = {name: list(FACTOR_ENUMS[name]) for name in FACTOR_NAMES}

        candidates: list[tuple[FactorSet, float]] = []
        for d_i, sel_i, sc_i, op_i, corr_i, dep_i in cart_product(
            top_indices["decomposition"],
            top_indices["selector"],
            top_indices["scope"],
            top_indices["op"],
            top_indices["correspondence"],
            top_indices["depth"],
        ):
            fs = FactorSet(
                decomposition=enum_lists["decomposition"][d_i],
                selector=enum_lists["selector"][sel_i],
                scope=enum_lists["scope"][sc_i],
                op=enum_lists["op"][op_i],
                correspondence=enum_lists["correspondence"][corr_i],
                depth=enum_lists["depth"][dep_i],
            )
            if not is_compatible(fs):
                continue

            # Joint probability = product of marginals
            joint_prob = (
                probs["decomposition"][d_i]
                * probs["selector"][sel_i]
                * probs["scope"][sc_i]
                * probs["op"][op_i]
                * probs["correspondence"][corr_i]
                * probs["depth"][dep_i]
            )
            candidates.append((fs, float(joint_prob)))

        # Sort by joint probability descending
        candidates.sort(key=lambda x: -x[1])

        # Deduplicate (same FactorSet could appear from different index combos)
        seen: set[tuple] = set()
        unique: list[tuple[FactorSet, float]] = []
        for fs, prob in candidates:
            key = fs.as_tuple()
            if key not in seen:
                seen.add(key)
                unique.append((fs, prob))
            if len(unique) >= k:
                break

        return unique

    def fit_from_labels(
        self,
        features_list: list[np.ndarray],
        factor_labels: list[FactorSet],
        **kwargs: Any,
    ) -> None:
        """Train all heads from paired (features, FactorSet) examples."""
        if not features_list or not factor_labels:
            return

        X = np.stack(features_list)
        enum_lists = {name: list(FACTOR_ENUMS[name]) for name in FACTOR_NAMES}
        enum_to_idx = {
            name: {v: i for i, v in enumerate(vals)}
            for name, vals in enum_lists.items()
        }

        for name in FACTOR_NAMES:
            y = np.array([
                enum_to_idx[name][getattr(fs, name)]
                for fs in factor_labels
            ], dtype=int)
            self.heads[name].fit(X, y, **kwargs)


# ---------------------------------------------------------------------------
# Uniform fallback (no trained model)
# ---------------------------------------------------------------------------


def uniform_factor_ranking(
    *,
    per_factor_k: int = 0,
    max_combos: int = 100,
) -> list[FactorSet]:
    """Enumerate compatible combos with uniform prior.

    Used when no trained proposer is available.
    Returns combos in a fixed deterministic order.
    """
    from aria.factors import enumerate_compatible

    all_combos = enumerate_compatible()

    # Prioritize depth=1 over depth=2 over depth=3
    all_combos.sort(key=lambda fs: fs.depth.value)

    return all_combos[:max_combos]
