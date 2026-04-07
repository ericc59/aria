"""Specialization-alternative evaluation metrics and lightweight baselines.

Measures which binding types benefit most from alternative exploration,
and provides simple reranking baselines.

No solver changes. No task-id logic. No heavy ML.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from aria.core.guidance_spec_traces import (
    SpecializationAlternative,
    SpecializationDecision,
    SpecializationEpisode,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BindingTypeStats:
    """Per-binding-type aggregated stats."""
    binding_type: str
    n_decisions: int = 0
    n_with_alternatives: int = 0
    avg_alternatives: float = 0.0
    n_default_verified: int = 0
    n_alt_verified: int = 0
    n_default_wrong: int = 0       # default didn't verify but an alt did
    default_win_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "binding_type": self.binding_type,
            "n_decisions": self.n_decisions,
            "n_with_alternatives": self.n_with_alternatives,
            "avg_alternatives": round(self.avg_alternatives, 1),
            "n_default_verified": self.n_default_verified,
            "n_alt_verified": self.n_alt_verified,
            "n_default_wrong": self.n_default_wrong,
            "default_win_rate": round(self.default_win_rate, 4),
        }


@dataclass
class SourceStats:
    """Per-evidence-source win rates."""
    source: str
    n_chosen: int = 0
    n_verified: int = 0
    win_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_chosen": self.n_chosen,
            "n_verified": self.n_verified,
            "win_rate": round(self.win_rate, 4),
        }


@dataclass
class SpecEvalReport:
    """Specialization-alternative evaluation report."""
    n_tasks: int = 0
    n_tasks_default_verified: int = 0
    n_total_decisions: int = 0
    n_decisions_with_alts: int = 0
    avg_alts_per_decision: float = 0.0

    # Winner-in-top-k
    winner_at_1: int = 0
    winner_at_3: int = 0
    n_decisions_evaluated: int = 0
    avg_winner_rank: float = 0.0

    # Default vs best gap
    n_default_wrong: int = 0
    default_correct_rate: float = 0.0

    # Per-binding-type
    binding_type_stats: dict[str, BindingTypeStats] = field(default_factory=dict)

    # Per-evidence-source
    source_stats: dict[str, SourceStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "n_tasks_default_verified": self.n_tasks_default_verified,
            "n_total_decisions": self.n_total_decisions,
            "n_decisions_with_alts": self.n_decisions_with_alts,
            "avg_alts_per_decision": round(self.avg_alts_per_decision, 1),
            "winner_at_1": self.winner_at_1,
            "winner_at_3": self.winner_at_3,
            "n_decisions_evaluated": self.n_decisions_evaluated,
            "avg_winner_rank": round(self.avg_winner_rank, 2),
            "n_default_wrong": self.n_default_wrong,
            "default_correct_rate": round(self.default_correct_rate, 4),
            "binding_type_stats": {k: v.to_dict() for k, v in self.binding_type_stats.items()},
            "source_stats": {k: v.to_dict() for k, v in self.source_stats.items()},
        }


def evaluate_spec_traces(episodes: list[SpecializationEpisode]) -> SpecEvalReport:
    """Evaluate specialization-alternative quality."""
    report = SpecEvalReport(n_tasks=len(episodes))

    total_alts = 0
    winner_rank_sum = 0.0
    n_evaluated = 0
    bt_data: dict[str, list] = {}
    source_chosen: Counter = Counter()
    source_verified: Counter = Counter()

    for ep in episodes:
        if ep.default_verified:
            report.n_tasks_default_verified += 1

        report.n_default_wrong += ep.n_default_wrong

        for d in ep.decisions:
            report.n_total_decisions += 1
            n_alts = len(d.alternatives)
            total_alts += n_alts

            if n_alts > 1:
                report.n_decisions_with_alts += 1

            # Track binding type stats
            bt = d.binding_type
            if bt not in bt_data:
                bt_data[bt] = []
            bt_data[bt].append(d)

            # Track source stats
            source_chosen[d.chosen_source] += 1
            if ep.default_verified:
                source_verified[d.chosen_source] += 1

            # Winner-in-top-k: the "winner" is rank 0 if default verified,
            # or best_alternative_rank if an alternative verified
            if ep.default_verified:
                report.winner_at_1 += 1
                report.winner_at_3 += 1
                winner_rank_sum += 0
                n_evaluated += 1
            elif d.best_alternative_rank is not None:
                rank = d.best_alternative_rank
                if rank <= 0:
                    report.winner_at_1 += 1
                if rank < 3:
                    report.winner_at_3 += 1
                winner_rank_sum += rank
                n_evaluated += 1

    report.n_decisions_evaluated = n_evaluated
    report.avg_winner_rank = winner_rank_sum / max(n_evaluated, 1)
    report.avg_alts_per_decision = total_alts / max(report.n_total_decisions, 1)
    report.default_correct_rate = report.n_tasks_default_verified / max(len(episodes), 1)

    # Binding type stats
    for bt, decs in bt_data.items():
        bts = BindingTypeStats(binding_type=bt, n_decisions=len(decs))
        total_bt_alts = 0
        for d in decs:
            if len(d.alternatives) > 1:
                bts.n_with_alternatives += 1
            total_bt_alts += len(d.alternatives)
            if d.verified_count > 0:
                bts.n_alt_verified += 1
        bts.avg_alternatives = total_bt_alts / max(len(decs), 1)
        # Default win rate approximation
        bts.n_default_verified = sum(1 for ep in episodes if ep.default_verified)
        bts.default_win_rate = bts.n_default_verified / max(len(decs), 1)
        report.binding_type_stats[bt] = bts

    # Source stats
    for source in source_chosen:
        ss = SourceStats(
            source=source,
            n_chosen=source_chosen[source],
            n_verified=source_verified.get(source, 0),
        )
        ss.win_rate = ss.n_verified / max(ss.n_chosen, 1)
        report.source_stats[source] = ss

    return report


# ---------------------------------------------------------------------------
# Lightweight baselines
# ---------------------------------------------------------------------------


@dataclass
class SourceWinReranker:
    """Rerank alternatives by evidence-source win frequency."""
    source_wins: Counter = field(default_factory=Counter)
    source_totals: Counter = field(default_factory=Counter)

    def fit(self, episodes: list[SpecializationEpisode]) -> None:
        for ep in episodes:
            for d in ep.decisions:
                for a in d.alternatives:
                    self.source_totals[a.source] += 1
                    if a.chosen and ep.default_verified:
                        self.source_wins[a.source] += 1

    def score(self, alt: SpecializationAlternative) -> float:
        total = self.source_totals.get(alt.source, 0)
        if total == 0:
            return 0.0
        return self.source_wins.get(alt.source, 0) / total

    def rerank(self, alternatives: list[SpecializationAlternative]) -> list[SpecializationAlternative]:
        return sorted(alternatives, key=lambda a: -self.score(a))


@dataclass
class BindingTypeReranker:
    """Rerank alternatives by binding-type-specific value win frequency."""
    # binding_type -> value_str -> win count
    type_value_wins: dict[str, Counter] = field(default_factory=dict)
    type_value_totals: dict[str, Counter] = field(default_factory=dict)

    def fit(self, episodes: list[SpecializationEpisode]) -> None:
        for ep in episodes:
            for d in ep.decisions:
                bt = d.binding_type
                if bt not in self.type_value_wins:
                    self.type_value_wins[bt] = Counter()
                    self.type_value_totals[bt] = Counter()
                for a in d.alternatives:
                    key = str(a.value)
                    self.type_value_totals[bt][key] += 1
                    if a.chosen and ep.default_verified:
                        self.type_value_wins[bt][key] += 1

    def rerank(
        self, binding_type: str, alternatives: list[SpecializationAlternative],
    ) -> list[SpecializationAlternative]:
        wins = self.type_value_wins.get(binding_type, Counter())
        totals = self.type_value_totals.get(binding_type, Counter())

        def score(a: SpecializationAlternative) -> float:
            key = str(a.value)
            t = totals.get(key, 0)
            return wins.get(key, 0) / t if t > 0 else 0.0

        return sorted(alternatives, key=lambda a: -score(a))


def evaluate_source_reranker(
    reranker: SourceWinReranker,
    episodes: list[SpecializationEpisode],
) -> dict[str, Any]:
    """Compare default ordering vs. source-reranked ordering."""
    original_at_1 = 0
    reranked_at_1 = 0
    n_evaluated = 0

    for ep in episodes:
        if not ep.default_verified:
            continue
        for d in ep.decisions:
            if len(d.alternatives) <= 1:
                continue
            n_evaluated += 1

            # Default: chosen alt is at rank 0
            chosen_alts = [a for a in d.alternatives if a.chosen]
            if chosen_alts and chosen_alts[0].rank == 0:
                original_at_1 += 1

            # Reranked
            reranked = reranker.rerank(list(d.alternatives))
            if reranked and reranked[0].chosen:
                reranked_at_1 += 1

    return {
        "n_evaluated": n_evaluated,
        "original_at_1": original_at_1,
        "reranked_at_1": reranked_at_1,
    }
