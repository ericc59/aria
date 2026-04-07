"""Lightweight guidance baselines over search-decision traces.

Validates that trace data is useful by testing simple reranking strategies:
1. Frequency-based reranking of candidates
2. Retrieval-based reranking using perception signatures
3. Feature-conditioned heuristic reranking

No heavy ML. No solver changes. No task-id logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from aria.core.guidance_baselines import PerceptionSignature
from aria.core.guidance_traces import CandidateRecord, DecisionStep, SearchEpisode


# ---------------------------------------------------------------------------
# Frequency-based reranker
# ---------------------------------------------------------------------------


@dataclass
class FrequencyReranker:
    """Rerank candidates by historical win frequency.

    Fits from solved episodes: for each stage, count how often each
    candidate label won. At inference, boost candidates with higher
    historical win rates.
    """
    # stage -> label -> win count
    win_counts: dict[str, Counter] = field(default_factory=dict)
    # stage -> label -> total proposal count
    proposal_counts: dict[str, Counter] = field(default_factory=dict)

    def fit(self, episodes: list[SearchEpisode]) -> None:
        """Learn win frequencies from episodes."""
        for ep in episodes:
            for step in ep.steps:
                stage = step.stage
                if stage not in self.win_counts:
                    self.win_counts[stage] = Counter()
                    self.proposal_counts[stage] = Counter()

                for c in step.candidates:
                    self.proposal_counts[stage][c.label] += 1
                    if c.verified:
                        self.win_counts[stage][c.label] += 1

    def rerank(self, step: DecisionStep) -> list[CandidateRecord]:
        """Rerank candidates in a decision step by win frequency."""
        wins = self.win_counts.get(step.stage, Counter())
        proposals = self.proposal_counts.get(step.stage, Counter())

        def score(c: CandidateRecord) -> float:
            total = proposals.get(c.label, 0)
            if total == 0:
                return 0.0
            return wins.get(c.label, 0) / total

        ranked = sorted(step.candidates, key=lambda c: -score(c))
        return list(ranked)

    def predict_top_k(self, step: DecisionStep, k: int = 3) -> list[str]:
        """Return top-k candidate labels by win frequency."""
        reranked = self.rerank(step)
        return [c.label for c in reranked[:k]]


# ---------------------------------------------------------------------------
# Retrieval-based reranker
# ---------------------------------------------------------------------------


@dataclass
class RetrievalReranker:
    """Rerank candidates using perception-signature similarity to solved tasks.

    For each stage, retrieves k-nearest solved episodes and boosts candidates
    that won in those neighbors.
    """
    # (signature, stage -> winning label) pairs
    entries: list[tuple[PerceptionSignature, dict[str, str]]] = field(default_factory=list)

    def fit(self, episodes: list[SearchEpisode], records: list[dict]) -> None:
        """Learn from paired (episode, record) data."""
        record_by_id = {r.get("task_id", ""): r for r in records}

        for ep in episodes:
            rec = record_by_id.get(ep.task_id)
            if rec is None:
                continue

            sig = PerceptionSignature.from_record(rec)
            stage_winners: dict[str, str] = {}
            for step in ep.steps:
                if step.winner_id:
                    for c in step.candidates:
                        if c.candidate_id == step.winner_id:
                            stage_winners[step.stage] = c.label
                            break

            if stage_winners:
                self.entries.append((sig, stage_winners))

    def rerank(self, step: DecisionStep, query_record: dict, k: int = 5) -> list[CandidateRecord]:
        """Rerank candidates using k-nearest neighbor votes."""
        query_sig = PerceptionSignature.from_record(query_record)

        # Find k-nearest entries
        scored = [(sig.overlap(query_sig), winners) for sig, winners in self.entries]
        scored.sort(key=lambda x: -x[0])
        neighbors = scored[:k]

        # Count votes for each label at this stage
        votes: Counter = Counter()
        for _, winners in neighbors:
            winner_label = winners.get(step.stage)
            if winner_label:
                votes[winner_label] += 1

        def score(c: CandidateRecord) -> float:
            return votes.get(c.label, 0)

        ranked = sorted(step.candidates, key=lambda c: (-score(c), c.rank))
        return list(ranked)


# ---------------------------------------------------------------------------
# Feature-conditioned heuristic reranker
# ---------------------------------------------------------------------------


@dataclass
class FeatureConditionedReranker:
    """Rerank using simple feature -> winner associations.

    Learns rules like: "if same_dims=True and has_frame=True, then
    periodic_repair wins in the lane stage 80% of the time."
    """
    # (feature_key, stage) -> label -> win count
    conditional_wins: dict[tuple[str, str], Counter] = field(default_factory=dict)
    conditional_totals: dict[tuple[str, str], Counter] = field(default_factory=dict)

    def fit(self, episodes: list[SearchEpisode], records: list[dict]) -> None:
        """Learn feature-conditioned win rates."""
        record_by_id = {r.get("task_id", ""): r for r in records}

        for ep in episodes:
            rec = record_by_id.get(ep.task_id)
            if rec is None:
                continue

            sig = PerceptionSignature.from_record(rec)
            feature_keys = [
                f"same_dims={sig.same_dims}",
                f"has_frame={sig.has_frame}",
                f"has_partition={sig.has_partition}",
                f"has_legend={sig.has_legend}",
                f"n_obj={sig.n_objects_bucket}",
            ]

            for step in ep.steps:
                winner_label = None
                if step.winner_id:
                    for c in step.candidates:
                        if c.candidate_id == step.winner_id:
                            winner_label = c.label
                            break

                for fk in feature_keys:
                    key = (fk, step.stage)
                    if key not in self.conditional_wins:
                        self.conditional_wins[key] = Counter()
                        self.conditional_totals[key] = Counter()

                    for c in step.candidates:
                        self.conditional_totals[key][c.label] += 1
                    if winner_label:
                        self.conditional_wins[key][winner_label] += 1

    def rerank(self, step: DecisionStep, query_record: dict) -> list[CandidateRecord]:
        """Rerank using feature-conditioned win rates."""
        sig = PerceptionSignature.from_record(query_record)
        feature_keys = [
            f"same_dims={sig.same_dims}",
            f"has_frame={sig.has_frame}",
            f"has_partition={sig.has_partition}",
            f"has_legend={sig.has_legend}",
            f"n_obj={sig.n_objects_bucket}",
        ]

        scores: dict[str, float] = {}
        for c in step.candidates:
            score = 0.0
            for fk in feature_keys:
                key = (fk, step.stage)
                wins = self.conditional_wins.get(key, Counter())
                totals = self.conditional_totals.get(key, Counter())
                total = totals.get(c.label, 0)
                if total > 0:
                    score += wins.get(c.label, 0) / total
            scores[c.candidate_id] = score

        ranked = sorted(step.candidates, key=lambda c: (-scores.get(c.candidate_id, 0), c.rank))
        return list(ranked)


# ---------------------------------------------------------------------------
# Reranking evaluation
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    """Result of evaluating a reranker on a set of episodes."""
    reranker_name: str
    stage: str
    n_evaluated: int = 0
    original_winner_at_1: int = 0
    reranked_winner_at_1: int = 0
    original_avg_rank: float = 0.0
    reranked_avg_rank: float = 0.0

    def to_dict(self) -> dict:
        return {
            "reranker_name": self.reranker_name,
            "stage": self.stage,
            "n_evaluated": self.n_evaluated,
            "original_winner_at_1": self.original_winner_at_1,
            "reranked_winner_at_1": self.reranked_winner_at_1,
            "original_avg_rank": round(self.original_avg_rank, 2),
            "reranked_avg_rank": round(self.reranked_avg_rank, 2),
            "winner_at_1_lift": (
                self.reranked_winner_at_1 - self.original_winner_at_1
                if self.n_evaluated > 0 else 0
            ),
        }


def evaluate_reranker(
    reranker_name: str,
    rerank_fn,
    episodes: list[SearchEpisode],
    stage: str,
) -> RerankResult:
    """Evaluate a reranking function on episodes for a specific stage.

    rerank_fn: (DecisionStep) -> list[CandidateRecord]
    """
    result = RerankResult(reranker_name=reranker_name, stage=stage)

    original_rank_sum = 0.0
    reranked_rank_sum = 0.0

    for ep in episodes:
        step = ep.step_for_stage(stage)
        if step is None or not step.winner_id or not step.candidates:
            continue

        result.n_evaluated += 1

        # Find winner label
        winner_label = None
        for c in step.candidates:
            if c.candidate_id == step.winner_id:
                winner_label = c.label
                original_rank_sum += c.rank
                if c.rank == 0:
                    result.original_winner_at_1 += 1
                break

        if winner_label is None:
            continue

        # Apply reranking
        reranked = rerank_fn(step)
        for i, c in enumerate(reranked):
            if c.label == winner_label:
                reranked_rank_sum += i
                if i == 0:
                    result.reranked_winner_at_1 += 1
                break

    n = max(result.n_evaluated, 1)
    result.original_avg_rank = original_rank_sum / n
    result.reranked_avg_rank = reranked_rank_sum / n

    return result
