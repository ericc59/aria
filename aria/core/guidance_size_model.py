"""Output-size mode guidance model — serializable artifact and shadow reranker.

Provides:
1. Model artifact: train, serialize, load the linear softmax output-size predictor
2. Shadow reranker: reorder output-size candidates without changing solver behavior
3. Trace/export: record default vs model ordering per task

No task-id logic. Exact verification unchanged. Model only reorders existing candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aria.core.guidance_datasets import (
    FEATURE_NAMES,
    SIZE_FAMILY_NAMES,
    SIZE_MODE_FAMILIES,
    extract_perception_features,
    features_to_array,
)
from aria.types import DemoPair


MODEL_VERSION = 1
MODEL_NAME = "output_size_linear_softmax_v1"


# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------


@dataclass
class SizeModelArtifact:
    """Serializable output-size mode guidance model."""
    version: int
    name: str
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    weights: np.ndarray         # (n_features, n_classes)
    bias: np.ndarray            # (n_classes,)
    mean: np.ndarray            # (n_features,) for normalization
    std: np.ndarray             # (n_features,) for normalization

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities from a feature vector.

        Args:
            features: shape (n_features,) or (n_samples, n_features)
        Returns:
            shape (n_classes,) or (n_samples, n_classes) probabilities
        """
        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        Xn = (features - self.mean) / self.std
        logits = Xn @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        return probs[0] if single else probs

    def rank_families(self, features: np.ndarray) -> list[str]:
        """Return family names ranked by predicted probability (best first)."""
        probs = self.predict_proba(features)
        order = np.argsort(-probs)
        return [self.label_names[i] for i in order]

    def save(self, path: str | Path) -> None:
        """Save model artifact to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p.with_suffix(".npz"),
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
        )
        meta = {
            "version": self.version,
            "name": self.name,
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
        }
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> SizeModelArtifact:
        """Load model artifact from disk."""
        p = Path(path)
        with open(p.with_suffix(".json")) as f:
            meta = json.load(f)
        arrays = np.load(p.with_suffix(".npz"))
        return SizeModelArtifact(
            version=meta["version"],
            name=meta["name"],
            feature_names=tuple(meta["feature_names"]),
            label_names=tuple(meta["label_names"]),
            weights=arrays["weights"],
            bias=arrays["bias"],
            mean=arrays["mean"],
            std=arrays["std"],
        )


def train_size_model(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
    lr: float = 0.05,
    epochs: int = 200,
    reg: float = 1e-4,
) -> SizeModelArtifact:
    """Train the output-size mode model from task demos.

    Returns a serializable artifact ready for shadow integration.
    """
    from aria.core.guidance_datasets import build_output_size_dataset

    ds = build_output_size_dataset(task_demo_pairs)
    X, y = ds.to_arrays()

    if len(X) == 0:
        # Empty model
        d = len(FEATURE_NAMES)
        k = len(SIZE_FAMILY_NAMES)
        return SizeModelArtifact(
            version=MODEL_VERSION, name=MODEL_NAME,
            feature_names=tuple(FEATURE_NAMES),
            label_names=SIZE_FAMILY_NAMES,
            weights=np.zeros((d, k)),
            bias=np.zeros(k),
            mean=np.zeros(d),
            std=np.ones(d),
        )

    n, d = X.shape
    k = len(SIZE_FAMILY_NAMES)

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    # SGD
    W = np.zeros((d, k))
    b = np.zeros(k)

    for epoch in range(epochs):
        logits = Xn @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        targets = np.zeros((n, k))
        targets[np.arange(n), y] = 1.0

        grad = probs - targets
        dW = (Xn.T @ grad) / n + reg * W
        db = grad.mean(axis=0)

        W -= lr * dW
        b -= lr * db

    return SizeModelArtifact(
        version=MODEL_VERSION, name=MODEL_NAME,
        feature_names=tuple(FEATURE_NAMES),
        label_names=SIZE_FAMILY_NAMES,
        weights=W, bias=b, mean=mean, std=std,
    )


# ---------------------------------------------------------------------------
# Shadow-mode reranker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowRankResult:
    """Shadow comparison for one task's output-size candidates."""
    task_id: str
    n_candidates: int
    default_winner_mode: str
    default_winner_family: str
    default_winner_rank: int        # rank in default (symbolic) ordering
    model_winner_rank: int          # rank the model would have assigned
    model_top_family: str           # model's top-1 prediction
    rank_improvement: int           # default - model (positive = model better)
    would_save_attempts: int        # candidates skipped if model ordering used
    model_ranking: tuple[str, ...]  # full model-ranked family ordering

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_candidates": self.n_candidates,
            "default_winner_mode": self.default_winner_mode,
            "default_winner_family": self.default_winner_family,
            "default_winner_rank": self.default_winner_rank,
            "model_winner_rank": self.model_winner_rank,
            "model_top_family": self.model_top_family,
            "rank_improvement": self.rank_improvement,
            "would_save_attempts": self.would_save_attempts,
            "model_ranking": list(self.model_ranking),
        }

    @staticmethod
    def from_dict(d: dict) -> ShadowRankResult:
        return ShadowRankResult(**{**d, "model_ranking": tuple(d.get("model_ranking", []))})


def shadow_rerank_task(
    task_id: str,
    demos: tuple[DemoPair, ...],
    model: SizeModelArtifact,
) -> ShadowRankResult | None:
    """Compute shadow reranking for one task's output-size candidates.

    Returns None if no verified output-size spec exists.
    """
    from aria.core.output_size import infer_verified_output_size_specs

    verified = infer_verified_output_size_specs(demos)
    if not verified:
        return None

    # Default winner = first verified (rank 0 in default ordering)
    default_winner = verified[0]
    default_winner_mode = default_winner.mode
    default_winner_family = SIZE_MODE_FAMILIES.get(default_winner_mode, "other")

    # Build family list for all candidates in default order
    candidate_families = []
    for spec in verified:
        fam = SIZE_MODE_FAMILIES.get(spec.mode, "other")
        candidate_families.append((spec.mode, fam))

    # Model prediction
    features = extract_perception_features(demos)
    feat_arr = features_to_array(features)
    model_ranking = model.rank_families(feat_arr)

    # Find where the default winner family appears in model ranking
    model_winner_rank = _find_family_rank(default_winner_family, model_ranking, candidate_families)

    # Find default rank of the winner (always 0 since it's the first verified)
    default_winner_rank = 0

    return ShadowRankResult(
        task_id=task_id,
        n_candidates=len(verified),
        default_winner_mode=default_winner_mode,
        default_winner_family=default_winner_family,
        default_winner_rank=default_winner_rank,
        model_winner_rank=model_winner_rank,
        model_top_family=model_ranking[0] if model_ranking else "",
        rank_improvement=default_winner_rank - model_winner_rank,
        would_save_attempts=max(0, default_winner_rank - model_winner_rank),
        model_ranking=tuple(model_ranking),
    )


def _find_family_rank(
    target_family: str,
    model_ranking: list[str],
    candidate_families: list[tuple[str, str]],
) -> int:
    """Find the rank of target_family in model_ranking, considering only
    families that actually appear in the candidate list."""
    present_families = {fam for _, fam in candidate_families}
    rank = 0
    for fam in model_ranking:
        if fam == target_family:
            return rank
        if fam in present_families:
            rank += 1
    return rank


# ---------------------------------------------------------------------------
# Batch shadow evaluation
# ---------------------------------------------------------------------------


@dataclass
class ShadowEvalReport:
    """Aggregate shadow-mode evaluation results."""
    n_tasks: int
    n_evaluated: int
    avg_candidates: float
    avg_default_rank: float
    avg_model_rank: float
    avg_rank_improvement: float
    pct_improved: float         # model rank < default rank
    pct_same: float
    pct_worse: float
    total_attempts_saved: int
    avg_attempts_saved: float
    # Per-family breakdown
    family_results: dict[str, dict] = field(default_factory=dict)
    # Distribution of improvements
    improvement_distribution: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "n_evaluated": self.n_evaluated,
            "avg_candidates": round(self.avg_candidates, 1),
            "avg_default_rank": round(self.avg_default_rank, 2),
            "avg_model_rank": round(self.avg_model_rank, 2),
            "avg_rank_improvement": round(self.avg_rank_improvement, 2),
            "pct_improved": round(self.pct_improved, 4),
            "pct_same": round(self.pct_same, 4),
            "pct_worse": round(self.pct_worse, 4),
            "total_attempts_saved": self.total_attempts_saved,
            "avg_attempts_saved": round(self.avg_attempts_saved, 2),
            "family_results": self.family_results,
            "improvement_distribution": self.improvement_distribution,
        }


def shadow_evaluate_batch(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
    model: SizeModelArtifact,
) -> tuple[ShadowEvalReport, list[ShadowRankResult]]:
    """Run shadow evaluation on a batch of tasks."""
    results: list[ShadowRankResult] = []
    for task_id, demos in task_demo_pairs:
        r = shadow_rerank_task(task_id, demos, model)
        if r is not None:
            results.append(r)

    if not results:
        return ShadowEvalReport(
            n_tasks=len(task_demo_pairs), n_evaluated=0,
            avg_candidates=0, avg_default_rank=0, avg_model_rank=0,
            avg_rank_improvement=0, pct_improved=0, pct_same=0, pct_worse=0,
            total_attempts_saved=0, avg_attempts_saved=0,
        ), []

    n = len(results)
    improvements = [r.rank_improvement for r in results]
    saves = [r.would_save_attempts for r in results]
    candidates = [r.n_candidates for r in results]

    # Per-family stats
    family_data: dict[str, list[int]] = {}
    for r in results:
        family_data.setdefault(r.default_winner_family, []).append(r.rank_improvement)

    family_results = {}
    for fam, imps in family_data.items():
        family_results[fam] = {
            "count": len(imps),
            "avg_improvement": round(float(np.mean(imps)), 2),
            "pct_improved": round(sum(1 for x in imps if x > 0) / len(imps), 4),
            "pct_same": round(sum(1 for x in imps if x == 0) / len(imps), 4),
        }

    # Improvement distribution
    imp_dist = {}
    for imp in improvements:
        imp_dist[imp] = imp_dist.get(imp, 0) + 1

    report = ShadowEvalReport(
        n_tasks=len(task_demo_pairs),
        n_evaluated=n,
        avg_candidates=float(np.mean(candidates)),
        avg_default_rank=float(np.mean([r.default_winner_rank for r in results])),
        avg_model_rank=float(np.mean([r.model_winner_rank for r in results])),
        avg_rank_improvement=float(np.mean(improvements)),
        pct_improved=sum(1 for x in improvements if x > 0) / n,
        pct_same=sum(1 for x in improvements if x == 0) / n,
        pct_worse=sum(1 for x in improvements if x < 0) / n,
        total_attempts_saved=sum(saves),
        avg_attempts_saved=float(np.mean(saves)),
        family_results=family_results,
        improvement_distribution=imp_dist,
    )

    return report, results
