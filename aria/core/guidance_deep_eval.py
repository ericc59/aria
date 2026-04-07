"""Evaluation metrics and lightweight baselines for deep inner-loop traces.

Measures:
- alternative-param winner-in-top-k
- retrieval hit rate / adapted winner rank
- residual-localization coverage
- family-level gains from alternative enumeration
- lightweight reranking baselines

No solver changes. No task-id logic. No heavy ML.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from aria.core.guidance_deep_traces import (
    DeepTrace,
    LibraryRetrievalEpisode,
    ParamAlternative,
    ParamAlternativeEpisode,
    StructuredResidual,
)


# ---------------------------------------------------------------------------
# Param alternative metrics
# ---------------------------------------------------------------------------


@dataclass
class ParamAltMetrics:
    """Aggregate metrics for within-lane parameter alternatives."""
    n_tasks: int = 0
    avg_alternatives: float = 0.0
    n_families: dict[str, int] = field(default_factory=dict)
    n_verified_per_family: dict[str, int] = field(default_factory=dict)

    # Winner-in-top-k
    winner_at_1: int = 0
    winner_at_3: int = 0
    winner_at_5: int = 0
    n_tasks_with_winner: int = 0
    avg_winner_rank: float = 0.0

    # Best-failed stats
    n_tasks_with_best_failed: int = 0
    avg_best_failed_residual: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "avg_alternatives": round(self.avg_alternatives, 1),
            "n_families": self.n_families,
            "n_verified_per_family": self.n_verified_per_family,
            "winner_at_1": self.winner_at_1,
            "winner_at_3": self.winner_at_3,
            "winner_at_5": self.winner_at_5,
            "n_tasks_with_winner": self.n_tasks_with_winner,
            "avg_winner_rank": round(self.avg_winner_rank, 2),
            "n_tasks_with_best_failed": self.n_tasks_with_best_failed,
            "avg_best_failed_residual": round(self.avg_best_failed_residual, 4),
        }


@dataclass
class LibraryMetrics:
    """Aggregate metrics for library retrieval/adaptation."""
    n_tasks: int = 0
    avg_templates_available: float = 0.0
    avg_retrieved: float = 0.0
    avg_adapted: float = 0.0
    avg_compiled: float = 0.0
    avg_verified: float = 0.0
    n_tasks_with_winner: int = 0
    strategy_counts: dict[str, int] = field(default_factory=dict)
    strategy_verified: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "avg_templates_available": round(self.avg_templates_available, 1),
            "avg_retrieved": round(self.avg_retrieved, 1),
            "avg_adapted": round(self.avg_adapted, 1),
            "avg_compiled": round(self.avg_compiled, 1),
            "avg_verified": round(self.avg_verified, 1),
            "n_tasks_with_winner": self.n_tasks_with_winner,
            "strategy_counts": self.strategy_counts,
            "strategy_verified": self.strategy_verified,
        }


@dataclass
class ResidualMetrics:
    """Aggregate metrics for structured residuals."""
    n_tasks: int = 0
    n_tasks_with_residuals: int = 0
    avg_residuals_per_task: float = 0.0
    n_localized: int = 0
    n_global: int = 0
    avg_regions_per_residual: float = 0.0
    n_with_blame: int = 0
    n_with_hints: int = 0
    avg_hint_count: float = 0.0
    dominant_region_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "n_tasks_with_residuals": self.n_tasks_with_residuals,
            "avg_residuals_per_task": round(self.avg_residuals_per_task, 1),
            "n_localized": self.n_localized,
            "n_global": self.n_global,
            "avg_regions_per_residual": round(self.avg_regions_per_residual, 1),
            "n_with_blame": self.n_with_blame,
            "n_with_hints": self.n_with_hints,
            "avg_hint_count": round(self.avg_hint_count, 1),
            "dominant_region_counts": self.dominant_region_counts,
        }


@dataclass
class DeepEvalReport:
    """Complete deep-trace evaluation report."""
    param_alt_metrics: ParamAltMetrics = field(default_factory=ParamAltMetrics)
    library_metrics: LibraryMetrics = field(default_factory=LibraryMetrics)
    residual_metrics: ResidualMetrics = field(default_factory=ResidualMetrics)

    def to_dict(self) -> dict:
        return {
            "param_alternatives": self.param_alt_metrics.to_dict(),
            "library_retrieval": self.library_metrics.to_dict(),
            "structured_residuals": self.residual_metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_deep_traces(traces: list[DeepTrace]) -> DeepEvalReport:
    """Evaluate deep traces across tasks."""
    report = DeepEvalReport()

    param_eps = [t.param_alternatives for t in traces if t.param_alternatives]
    lib_eps = [t.library_retrieval for t in traces if t.library_retrieval]
    all_residuals = [r for t in traces for r in t.structured_residuals]

    if param_eps:
        report.param_alt_metrics = _eval_param_alts(param_eps)
    if lib_eps:
        report.library_metrics = _eval_library(lib_eps)
    if traces:
        report.residual_metrics = _eval_residuals(traces, all_residuals)

    return report


def _eval_param_alts(episodes: list[ParamAlternativeEpisode]) -> ParamAltMetrics:
    m = ParamAltMetrics(n_tasks=len(episodes))

    total_alts = 0
    family_counts: Counter = Counter()
    family_verified: Counter = Counter()
    winner_rank_sum = 0.0

    for ep in episodes:
        total_alts += ep.n_total
        for a in ep.alternatives:
            family_counts[a.family] += 1
            if a.verified:
                family_verified[a.family] += 1

        if ep.winner_alt_id:
            m.n_tasks_with_winner += 1
            for a in ep.alternatives:
                if a.alt_id == ep.winner_alt_id:
                    winner_rank_sum += a.rank
                    if a.rank == 0:
                        m.winner_at_1 += 1
                    if a.rank < 3:
                        m.winner_at_3 += 1
                    if a.rank < 5:
                        m.winner_at_5 += 1
                    break

        if ep.best_failed_residual is not None:
            m.n_tasks_with_best_failed += 1
            m.avg_best_failed_residual += ep.best_failed_residual

    n = max(len(episodes), 1)
    m.avg_alternatives = total_alts / n
    m.n_families = dict(family_counts)
    m.n_verified_per_family = dict(family_verified)
    m.avg_winner_rank = winner_rank_sum / max(m.n_tasks_with_winner, 1)
    m.avg_best_failed_residual = (
        m.avg_best_failed_residual / max(m.n_tasks_with_best_failed, 1)
    )

    return m


def _eval_library(episodes: list[LibraryRetrievalEpisode]) -> LibraryMetrics:
    m = LibraryMetrics(n_tasks=len(episodes))

    strategy_counts: Counter = Counter()
    strategy_verified: Counter = Counter()

    for ep in episodes:
        m.avg_templates_available += ep.n_templates_available
        m.avg_retrieved += ep.n_retrieved
        m.avg_adapted += ep.n_adapted
        m.avg_compiled += ep.n_compiled
        m.avg_verified += ep.n_verified
        if ep.winner_record_id:
            m.n_tasks_with_winner += 1
        for r in ep.records:
            strategy_counts[r.strategy] += 1
            if r.verified:
                strategy_verified[r.strategy] += 1

    n = max(len(episodes), 1)
    m.avg_templates_available /= n
    m.avg_retrieved /= n
    m.avg_adapted /= n
    m.avg_compiled /= n
    m.avg_verified /= n
    m.strategy_counts = dict(strategy_counts)
    m.strategy_verified = dict(strategy_verified)

    return m


def _eval_residuals(traces: list[DeepTrace], residuals: list[StructuredResidual]) -> ResidualMetrics:
    m = ResidualMetrics(n_tasks=len(traces))
    m.n_tasks_with_residuals = sum(1 for t in traces if t.structured_residuals)

    total_regions = 0
    total_hints = 0
    dominant_counts: Counter = Counter()

    for r in residuals:
        total_regions += len(r.regions)
        total_hints += r.repair_hint_count
        if r.is_localized:
            m.n_localized += 1
        else:
            m.n_global += 1
        if r.dominant_region:
            dominant_counts[r.dominant_region] += 1
        if r.blamed_node_ids:
            m.n_with_blame += 1
        if r.repair_hint_count > 0:
            m.n_with_hints += 1

    n_res = max(len(residuals), 1)
    m.avg_residuals_per_task = len(residuals) / max(len(traces), 1)
    m.avg_regions_per_residual = total_regions / n_res
    m.avg_hint_count = total_hints / n_res
    m.dominant_region_counts = dict(dominant_counts)

    return m


# ---------------------------------------------------------------------------
# Lightweight baselines
# ---------------------------------------------------------------------------


@dataclass
class ParamFrequencyReranker:
    """Rerank parameter alternatives by family-specific param-set win frequency."""
    family_param_wins: dict[str, Counter] = field(default_factory=dict)

    def fit(self, episodes: list[ParamAlternativeEpisode]) -> None:
        for ep in episodes:
            for a in ep.alternatives:
                if a.family not in self.family_param_wins:
                    self.family_param_wins[a.family] = Counter()
                key = _param_key(a.param_set)
                if a.verified:
                    self.family_param_wins[a.family][key] += 1

    def rerank(self, alternatives: list[ParamAlternative]) -> list[ParamAlternative]:
        def score(a: ParamAlternative) -> float:
            wins = self.family_param_wins.get(a.family, Counter())
            return wins.get(_param_key(a.param_set), 0)
        return sorted(alternatives, key=lambda a: (-score(a), a.rank))


@dataclass
class ResidualGuidedReranker:
    """Rerank alternatives using residual features to predict which family helps."""
    # residual_pattern -> family -> win count
    pattern_wins: dict[str, Counter] = field(default_factory=dict)

    def fit(self, traces: list[DeepTrace]) -> None:
        for t in traces:
            if not t.param_alternatives or not t.structured_residuals:
                continue
            pattern = _residual_pattern(t.structured_residuals)
            if pattern not in self.pattern_wins:
                self.pattern_wins[pattern] = Counter()
            if t.param_alternatives.winner_family:
                self.pattern_wins[pattern][t.param_alternatives.winner_family] += 1

    def predict_family(self, residuals: tuple[StructuredResidual, ...]) -> str | None:
        pattern = _residual_pattern(residuals)
        wins = self.pattern_wins.get(pattern, Counter())
        if wins:
            return wins.most_common(1)[0][0]
        return None


def _param_key(param_set: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(param_set.items()))


def _residual_pattern(residuals: tuple[StructuredResidual, ...]) -> str:
    """Hash residual features into a discrete pattern for conditioning."""
    if not residuals:
        return "none"
    r = residuals[0]
    parts = []
    if r.is_localized:
        parts.append("localized")
    else:
        parts.append("global")
    if r.blamed_node_ids:
        parts.append(f"blamed:{len(r.blamed_node_ids)}")
    if r.repair_hint_count > 0:
        parts.append(f"hints:{r.repair_hint_count}")
    # Coarse diff bucket
    if r.diff_fraction < 0.1:
        parts.append("low_diff")
    elif r.diff_fraction < 0.5:
        parts.append("mid_diff")
    else:
        parts.append("high_diff")
    return "|".join(parts)


def evaluate_param_reranker(
    reranker: ParamFrequencyReranker,
    episodes: list[ParamAlternativeEpisode],
) -> dict[str, Any]:
    """Evaluate a param reranker: does it place winners higher?"""
    original_ranks = []
    reranked_ranks = []

    for ep in episodes:
        if not ep.winner_alt_id:
            continue
        winner_label = None
        for i, a in enumerate(ep.alternatives):
            if a.alt_id == ep.winner_alt_id:
                original_ranks.append(a.rank)
                winner_label = a.alt_id
                break

        if winner_label is None:
            continue

        reranked = reranker.rerank(list(ep.alternatives))
        for i, a in enumerate(reranked):
            if a.alt_id == winner_label:
                reranked_ranks.append(i)
                break

    n = max(len(original_ranks), 1)
    return {
        "n_evaluated": len(original_ranks),
        "original_avg_rank": round(sum(original_ranks) / n, 2) if original_ranks else 0,
        "reranked_avg_rank": round(sum(reranked_ranks) / n, 2) if reranked_ranks else 0,
    }
