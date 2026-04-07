"""Decision-level evaluation metrics for search-decision traces.

Measures guidance quality at the candidate/decision level:
- winner-in-top-k rates
- average rank of winner
- best-failed residual by family
- calibration of ranking vs verification
- precision/recall of "attempt-worthy" candidates
- family-level overproposal stats

No solver changes. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from aria.core.guidance_traces import CandidateRecord, DecisionStep, SearchEpisode


# ---------------------------------------------------------------------------
# Per-stage metrics
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """Aggregate metrics for one stage across tasks."""
    stage: str
    n_tasks_with_candidates: int = 0
    total_candidates: int = 0
    total_attempted: int = 0
    total_verified: int = 0
    avg_candidates: float = 0.0
    avg_attempted: float = 0.0

    # Winner-in-top-k
    winner_at_1: int = 0
    winner_at_3: int = 0
    winner_at_5: int = 0
    n_tasks_with_winner: int = 0

    # Average rank of winner (0-indexed)
    winner_rank_sum: float = 0.0
    avg_winner_rank: float = 0.0

    # Attempt-worthy precision: fraction of attempted candidates that verified
    attempt_worthy_precision: float = 0.0

    # Overproposal: candidates proposed but never verifying
    overproposal_rate: float = 0.0

    # Best-failed stats
    n_tasks_with_best_failed: int = 0
    avg_best_failed_diff_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_tasks_with_candidates": self.n_tasks_with_candidates,
            "total_candidates": self.total_candidates,
            "total_attempted": self.total_attempted,
            "total_verified": self.total_verified,
            "avg_candidates": round(self.avg_candidates, 2),
            "avg_attempted": round(self.avg_attempted, 2),
            "winner_at_1": self.winner_at_1,
            "winner_at_3": self.winner_at_3,
            "winner_at_5": self.winner_at_5,
            "n_tasks_with_winner": self.n_tasks_with_winner,
            "avg_winner_rank": round(self.avg_winner_rank, 2),
            "attempt_worthy_precision": round(self.attempt_worthy_precision, 4),
            "overproposal_rate": round(self.overproposal_rate, 4),
            "n_tasks_with_best_failed": self.n_tasks_with_best_failed,
            "avg_best_failed_diff_fraction": round(self.avg_best_failed_diff_fraction, 4),
        }


@dataclass
class FamilyConfusion:
    """Per-family overproposal and hit rates."""
    family: str
    proposed: int = 0
    attempted: int = 0
    verified: int = 0
    best_failed_count: int = 0

    @property
    def hit_rate(self) -> float:
        return self.verified / max(self.proposed, 1)

    @property
    def attempt_rate(self) -> float:
        return self.attempted / max(self.proposed, 1)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "proposed": self.proposed,
            "attempted": self.attempted,
            "verified": self.verified,
            "best_failed_count": self.best_failed_count,
            "hit_rate": round(self.hit_rate, 4),
            "attempt_rate": round(self.attempt_rate, 4),
        }


@dataclass
class CalibrationBucket:
    """Score calibration: does a higher score predict verification?"""
    score_range: str   # e.g. "[0.0, 0.2)"
    n_candidates: int = 0
    n_verified: int = 0

    @property
    def verification_rate(self) -> float:
        return self.n_verified / max(self.n_candidates, 1)

    def to_dict(self) -> dict:
        return {
            "score_range": self.score_range,
            "n_candidates": self.n_candidates,
            "n_verified": self.n_verified,
            "verification_rate": round(self.verification_rate, 4),
        }


# ---------------------------------------------------------------------------
# Full trace evaluation report
# ---------------------------------------------------------------------------


@dataclass
class TraceEvalReport:
    """Complete decision-trace evaluation report."""
    n_episodes: int = 0
    n_solved: int = 0
    solve_rate: float = 0.0

    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    family_confusion: dict[str, FamilyConfusion] = field(default_factory=dict)
    calibration: list[CalibrationBucket] = field(default_factory=list)

    # Per-task details
    per_task: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "n_solved": self.n_solved,
            "solve_rate": round(self.solve_rate, 4),
            "stage_metrics": {k: v.to_dict() for k, v in self.stage_metrics.items()},
            "family_confusion": {k: v.to_dict() for k, v in self.family_confusion.items()},
            "calibration": [b.to_dict() for b in self.calibration],
        }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_search_traces(episodes: list[SearchEpisode]) -> TraceEvalReport:
    """Evaluate decision-trace quality across episodes."""
    report = TraceEvalReport(n_episodes=len(episodes))
    report.n_solved = sum(1 for e in episodes if e.solved)
    report.solve_rate = report.n_solved / max(len(episodes), 1)

    # Collect per-stage data
    stage_data: dict[str, list[DecisionStep]] = {}
    all_scored_candidates: list[tuple[float, bool]] = []

    for episode in episodes:
        task_detail: dict[str, Any] = {
            "task_id": episode.task_id,
            "solved": episode.solved,
            "total_candidates": episode.total_candidates,
            "final_winner_stage": episode.final_winner_stage,
        }

        for step in episode.steps:
            stage_data.setdefault(step.stage, []).append(step)

            # Collect scored candidates for calibration
            for c in step.candidates:
                if c.score is not None:
                    all_scored_candidates.append((c.score, c.verified))

        report.per_task.append(task_detail)

    # Compute per-stage metrics
    for stage, steps in stage_data.items():
        report.stage_metrics[stage] = _compute_stage_metrics(stage, steps)

    # Compute family confusion (across lane + graph stages)
    report.family_confusion = _compute_family_confusion(stage_data)

    # Compute score calibration
    report.calibration = _compute_calibration(all_scored_candidates)

    return report


def _compute_stage_metrics(stage: str, steps: list[DecisionStep]) -> StageMetrics:
    """Compute aggregate metrics for one stage."""
    m = StageMetrics(stage=stage)

    total_cands = 0
    total_attempted = 0
    total_verified = 0
    tasks_with_cands = 0
    tasks_with_winner = 0
    winner_rank_sum = 0.0
    best_failed_diff_sum = 0.0
    tasks_with_bf = 0

    for step in steps:
        if step.n_candidates > 0:
            tasks_with_cands += 1
            total_cands += step.n_candidates
            total_attempted += step.n_attempted
            total_verified += step.n_verified

            if step.winner_id:
                tasks_with_winner += 1
                # Find winner rank
                for c in step.candidates:
                    if c.candidate_id == step.winner_id:
                        winner_rank_sum += c.rank
                        if c.rank == 0:
                            m.winner_at_1 += 1
                        if c.rank < 3:
                            m.winner_at_3 += 1
                        if c.rank < 5:
                            m.winner_at_5 += 1
                        break

            if step.best_failed_id:
                for c in step.candidates:
                    if c.candidate_id == step.best_failed_id and c.residual:
                        diff_frac = c.residual.get("diff_fraction", 0)
                        best_failed_diff_sum += diff_frac
                        tasks_with_bf += 1
                        break

    m.n_tasks_with_candidates = tasks_with_cands
    m.total_candidates = total_cands
    m.total_attempted = total_attempted
    m.total_verified = total_verified
    m.avg_candidates = total_cands / max(tasks_with_cands, 1)
    m.avg_attempted = total_attempted / max(tasks_with_cands, 1)
    m.n_tasks_with_winner = tasks_with_winner
    m.avg_winner_rank = winner_rank_sum / max(tasks_with_winner, 1)
    m.attempt_worthy_precision = total_verified / max(total_attempted, 1)
    m.overproposal_rate = (total_cands - total_verified) / max(total_cands, 1)
    m.n_tasks_with_best_failed = tasks_with_bf
    m.avg_best_failed_diff_fraction = best_failed_diff_sum / max(tasks_with_bf, 1)

    return m


def _compute_family_confusion(
    stage_data: dict[str, list[DecisionStep]],
) -> dict[str, FamilyConfusion]:
    """Compute per-family proposal/hit stats from lane and graph stages."""
    families: dict[str, FamilyConfusion] = {}

    for stage in ("lane", "graph"):
        for step in stage_data.get(stage, []):
            for c in step.candidates:
                label = c.label
                if label not in families:
                    families[label] = FamilyConfusion(family=label)
                fc = families[label]
                fc.proposed += 1
                if c.attempted:
                    fc.attempted += 1
                if c.verified:
                    fc.verified += 1
                if c.candidate_id == step.best_failed_id:
                    fc.best_failed_count += 1

    return families


def _compute_calibration(
    scored: list[tuple[float, bool]],
    n_buckets: int = 5,
) -> list[CalibrationBucket]:
    """Compute score calibration buckets."""
    if not scored:
        return []

    boundaries = [i / n_buckets for i in range(n_buckets + 1)]
    buckets = []

    for i in range(n_buckets):
        lo, hi = boundaries[i], boundaries[i + 1]
        label = f"[{lo:.1f}, {hi:.1f})"
        bucket = CalibrationBucket(score_range=label)

        for score, verified in scored:
            if lo <= score < hi or (i == n_buckets - 1 and score == hi):
                bucket.n_candidates += 1
                if verified:
                    bucket.n_verified += 1

        buckets.append(bucket)

    return buckets


# ---------------------------------------------------------------------------
# Convenience: winner-in-top-k for a specific stage
# ---------------------------------------------------------------------------


def winner_in_top_k(episode: SearchEpisode, stage: str, k: int) -> bool | None:
    """Check if the winning candidate for a stage appears in top-k.

    Returns None if the stage has no candidates or no winner.
    """
    step = episode.step_for_stage(stage)
    if step is None or not step.candidates or not step.winner_id:
        return None

    for c in step.candidates:
        if c.candidate_id == step.winner_id:
            return c.rank < k
    return None


def winner_rank(episode: SearchEpisode, stage: str) -> int | None:
    """Return the rank of the winning candidate for a stage."""
    step = episode.step_for_stage(stage)
    if step is None or not step.winner_id:
        return None

    for c in step.candidates:
        if c.candidate_id == step.winner_id:
            return c.rank
    return None
