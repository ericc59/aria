"""Inner-loop evaluation metrics and lightweight baselines.

Measures guidance quality at the graph-edit and parameter-trial level.
Also provides simple reranking baselines over inner-loop traces.

No solver changes. No task-id logic. No heavy ML.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from aria.core.guidance_inner_traces import (
    FailureCategory,
    GraphEditCandidate,
    GraphEditEpisode,
    InnerLoopTrace,
    ParamTrialCandidate,
    ParamTrialEpisode,
)


# ---------------------------------------------------------------------------
# Graph-edit metrics
# ---------------------------------------------------------------------------


@dataclass
class EditSearchMetrics:
    """Aggregate metrics for graph-edit search across tasks."""
    n_tasks: int = 0
    n_solved: int = 0

    avg_edits_generated: float = 0.0
    avg_edits_attempted: float = 0.0
    avg_edits_compiled: float = 0.0
    avg_unique_states: float = 0.0
    avg_max_depth: float = 0.0

    # Winner stats
    winner_at_depth_0: int = 0
    winner_at_depth_1: int = 0
    avg_winner_depth: float = 0.0

    # Branch factor
    avg_branch_factor: float = 0.0

    # Failure category breakdown
    failure_counts: dict[str, int] = field(default_factory=dict)

    # Residual reduction
    n_residual_reduced: int = 0
    avg_residual_reduction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "n_solved": self.n_solved,
            "avg_edits_generated": round(self.avg_edits_generated, 1),
            "avg_edits_attempted": round(self.avg_edits_attempted, 1),
            "avg_edits_compiled": round(self.avg_edits_compiled, 1),
            "avg_unique_states": round(self.avg_unique_states, 1),
            "avg_max_depth": round(self.avg_max_depth, 1),
            "winner_at_depth_0": self.winner_at_depth_0,
            "winner_at_depth_1": self.winner_at_depth_1,
            "avg_winner_depth": round(self.avg_winner_depth, 2),
            "avg_branch_factor": round(self.avg_branch_factor, 1),
            "failure_counts": self.failure_counts,
            "n_residual_reduced": self.n_residual_reduced,
            "avg_residual_reduction": round(self.avg_residual_reduction, 4),
        }


@dataclass
class ParamTrialMetrics:
    """Aggregate metrics for parameter/specialization trials."""
    n_tasks: int = 0
    avg_trials_per_task: float = 0.0
    avg_families_tried: float = 0.0
    avg_gate_passed: float = 0.0
    avg_compiled: float = 0.0
    avg_verified: float = 0.0

    # Family-specific stats
    family_stats: dict[str, dict] = field(default_factory=dict)

    # Failure category breakdown
    failure_counts: dict[str, int] = field(default_factory=dict)

    # Winner stats
    n_tasks_with_winner: int = 0
    avg_winner_rank: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "avg_trials_per_task": round(self.avg_trials_per_task, 1),
            "avg_families_tried": round(self.avg_families_tried, 1),
            "avg_gate_passed": round(self.avg_gate_passed, 1),
            "avg_compiled": round(self.avg_compiled, 1),
            "avg_verified": round(self.avg_verified, 1),
            "family_stats": self.family_stats,
            "failure_counts": self.failure_counts,
            "n_tasks_with_winner": self.n_tasks_with_winner,
            "avg_winner_rank": round(self.avg_winner_rank, 2),
        }


@dataclass
class InnerLoopReport:
    """Complete inner-loop evaluation report."""
    edit_metrics: EditSearchMetrics = field(default_factory=EditSearchMetrics)
    param_metrics: ParamTrialMetrics = field(default_factory=ParamTrialMetrics)

    def to_dict(self) -> dict:
        return {
            "edit_search": self.edit_metrics.to_dict(),
            "param_trials": self.param_metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_inner_traces(traces: list[InnerLoopTrace]) -> InnerLoopReport:
    """Evaluate inner-loop trace quality."""
    report = InnerLoopReport()

    edit_episodes = [t.edit_episode for t in traces if t.edit_episode is not None]
    param_episodes = [t.param_episode for t in traces if t.param_episode is not None]

    if edit_episodes:
        report.edit_metrics = _evaluate_edit_episodes(edit_episodes)
    if param_episodes:
        report.param_metrics = _evaluate_param_episodes(param_episodes)

    return report


def _evaluate_edit_episodes(episodes: list[GraphEditEpisode]) -> EditSearchMetrics:
    m = EditSearchMetrics(n_tasks=len(episodes))

    total_gen = 0
    total_att = 0
    total_comp = 0
    total_unique = 0
    total_depth = 0
    winner_depth_sum = 0.0
    n_winners = 0
    failure_counts: Counter = Counter()
    residual_reductions: list[float] = []

    for ep in episodes:
        total_gen += ep.n_edits_generated
        total_att += ep.n_edits_attempted
        total_comp += ep.n_edits_compiled
        total_unique += ep.n_unique_states
        total_depth += ep.max_depth

        if ep.solved:
            m.n_solved += 1
            if ep.winner_depth is not None:
                winner_depth_sum += ep.winner_depth
                n_winners += 1
                if ep.winner_depth == 0:
                    m.winner_at_depth_0 += 1
                elif ep.winner_depth == 1:
                    m.winner_at_depth_1 += 1

        for edit in ep.edits:
            failure_counts[edit.failure_category] += 1
            if (edit.residual_fraction is not None
                    and edit.failure_category == FailureCategory.RESIDUAL_REDUCED):
                residual_reductions.append(1.0 - edit.residual_fraction)

    n = max(len(episodes), 1)
    m.avg_edits_generated = total_gen / n
    m.avg_edits_attempted = total_att / n
    m.avg_edits_compiled = total_comp / n
    m.avg_unique_states = total_unique / n
    m.avg_max_depth = total_depth / n
    m.avg_winner_depth = winner_depth_sum / max(n_winners, 1)
    m.failure_counts = dict(failure_counts)

    # Branch factor: generated / unique states
    m.avg_branch_factor = total_gen / max(total_unique, 1)

    # Residual reduction
    m.n_residual_reduced = len(residual_reductions)
    m.avg_residual_reduction = (
        sum(residual_reductions) / len(residual_reductions)
        if residual_reductions else 0.0
    )

    return m


def _evaluate_param_episodes(episodes: list[ParamTrialEpisode]) -> ParamTrialMetrics:
    m = ParamTrialMetrics(n_tasks=len(episodes))

    total_trials = 0
    total_families = 0
    total_gate = 0
    total_comp = 0
    total_ver = 0
    winner_rank_sum = 0.0
    n_winners = 0
    failure_counts: Counter = Counter()
    family_data: dict[str, dict[str, int]] = {}

    for ep in episodes:
        total_trials += len(ep.trials)
        total_families += ep.n_families_tried
        total_gate += ep.n_gate_passed
        total_comp += ep.n_compiled
        total_ver += ep.n_verified

        if ep.winner_trial_id:
            m.n_tasks_with_winner += 1
            for t in ep.trials:
                if t.trial_id == ep.winner_trial_id:
                    winner_rank_sum += t.rank
                    break

        for t in ep.trials:
            failure_counts[t.failure_category] += 1
            if t.family not in family_data:
                family_data[t.family] = {"proposed": 0, "gate_passed": 0, "compiled": 0, "verified": 0}
            family_data[t.family]["proposed"] += 1
            if t.gate_passed:
                family_data[t.family]["gate_passed"] += 1
            if t.compile_succeeded:
                family_data[t.family]["compiled"] += 1
            if t.verified:
                family_data[t.family]["verified"] += 1

    n = max(len(episodes), 1)
    m.avg_trials_per_task = total_trials / n
    m.avg_families_tried = total_families / n
    m.avg_gate_passed = total_gate / n
    m.avg_compiled = total_comp / n
    m.avg_verified = total_ver / n
    m.avg_winner_rank = winner_rank_sum / max(m.n_tasks_with_winner, 1)
    m.failure_counts = dict(failure_counts)
    m.family_stats = family_data

    return m


# ---------------------------------------------------------------------------
# Lightweight baselines for inner-loop reranking
# ---------------------------------------------------------------------------


@dataclass
class EditTypeReranker:
    """Rerank graph-edit candidates by action-type win frequency."""
    type_wins: Counter = field(default_factory=Counter)
    type_total: Counter = field(default_factory=Counter)

    def fit(self, episodes: list[GraphEditEpisode]) -> None:
        for ep in episodes:
            for edit in ep.edits:
                self.type_total[edit.action_type] += 1
                if edit.verified:
                    self.type_wins[edit.action_type] += 1

    def score(self, edit: GraphEditCandidate) -> float:
        total = self.type_total.get(edit.action_type, 0)
        if total == 0:
            return 0.0
        return self.type_wins.get(edit.action_type, 0) / total

    def rerank(self, edits: list[GraphEditCandidate]) -> list[GraphEditCandidate]:
        return sorted(edits, key=lambda e: -self.score(e))


@dataclass
class FamilyParamReranker:
    """Rerank parameter trials by family-specific win rates."""
    family_wins: dict[str, Counter] = field(default_factory=dict)
    family_total: dict[str, Counter] = field(default_factory=dict)

    def fit(self, episodes: list[ParamTrialEpisode]) -> None:
        for ep in episodes:
            for t in ep.trials:
                if t.family not in self.family_wins:
                    self.family_wins[t.family] = Counter()
                    self.family_total[t.family] = Counter()
                self.family_total[t.family][t.param_name] += 1
                if t.verified:
                    self.family_wins[t.family][t.param_name] += 1

    def score(self, trial: ParamTrialCandidate) -> float:
        total = self.family_total.get(trial.family, Counter()).get(trial.param_name, 0)
        if total == 0:
            return 0.0
        return self.family_wins.get(trial.family, Counter()).get(trial.param_name, 0) / total

    def rerank(self, trials: list[ParamTrialCandidate]) -> list[ParamTrialCandidate]:
        return sorted(trials, key=lambda t: -self.score(t))


def evaluate_edit_reranker(
    reranker: EditTypeReranker,
    episodes: list[GraphEditEpisode],
) -> dict[str, Any]:
    """Evaluate an edit reranker: does it place winners higher?"""
    original_ranks = []
    reranked_ranks = []

    for ep in episodes:
        if not ep.solved or not ep.winner_edit_id:
            continue

        edits = list(ep.edits)
        winner_label = None
        for i, e in enumerate(edits):
            if e.edit_id == ep.winner_edit_id:
                original_ranks.append(i)
                winner_label = e.action_type
                break

        if winner_label is None:
            continue

        reranked = reranker.rerank(edits)
        for i, e in enumerate(reranked):
            if e.edit_id == ep.winner_edit_id:
                reranked_ranks.append(i)
                break

    n = max(len(original_ranks), 1)
    return {
        "n_evaluated": len(original_ranks),
        "original_avg_rank": sum(original_ranks) / n if original_ranks else 0,
        "reranked_avg_rank": sum(reranked_ranks) / n if reranked_ranks else 0,
    }
