"""Offline evaluator for local refinement policies.

Reads JSONL evaluation data, runs a ``LocalPolicy`` on each example,
and scores accuracy for NEXT_FOCUS prediction and action-ranking tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from aria.local_policy import (
    ActionRanking,
    FocusPrediction,
    LocalPolicy,
    PolicyInput,
)


# ---------------------------------------------------------------------------
# Eval-data schema  (one JSON object per line in the JSONL file)
# ---------------------------------------------------------------------------
#
# Required fields:
#   task_signatures : list[str]
#   round_index     : int
#   prior_focuses   : list[str]
#   prior_error_types : list[str | null]
#
# For NEXT_FOCUS scoring:
#   gold_focus      : str
#
# For ranking scoring:
#   gold_ranking    : list[str]   (best-first)
#   candidate_ops   : list[str]   (unordered candidates)
#


@dataclass(frozen=True)
class EvalExample:
    """Single evaluation example parsed from JSONL."""
    policy_input: PolicyInput
    gold_focus: str | None = None
    gold_ranking: tuple[str, ...] | None = None
    candidate_ops: tuple[str, ...] = ()


def load_eval_examples(path: str | Path) -> list[EvalExample]:
    """Load evaluation examples from a JSONL file."""
    examples: list[EvalExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp = PolicyInput(
                task_signatures=tuple(obj["task_signatures"]),
                round_index=obj["round_index"],
                prior_focuses=tuple(obj.get("prior_focuses", ())),
                prior_error_types=tuple(obj.get("prior_error_types", ())),
                candidate_ops=tuple(obj.get("candidate_ops", ())),
            )
            examples.append(EvalExample(
                policy_input=inp,
                gold_focus=obj.get("gold_focus"),
                gold_ranking=tuple(obj["gold_ranking"]) if "gold_ranking" in obj else None,
                candidate_ops=tuple(obj.get("candidate_ops", ())),
            ))
    return examples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def focus_accuracy(
    policy: LocalPolicy,
    examples: Sequence[EvalExample],
) -> float:
    """Fraction of examples where predicted focus == gold_focus."""
    scored = [ex for ex in examples if ex.gold_focus is not None]
    if not scored:
        return 0.0
    correct = 0
    for ex in scored:
        pred = policy.predict_next_focus(ex.policy_input)
        if pred.focus == ex.gold_focus:
            correct += 1
    return correct / len(scored)


def _reciprocal_rank(predicted: Sequence[str], gold_first: str) -> float:
    """1 / (rank of gold_first in predicted), or 0 if absent."""
    for i, item in enumerate(predicted):
        if item == gold_first:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    policy: LocalPolicy,
    examples: Sequence[EvalExample],
) -> float:
    """MRR over ranking examples (gold_ranking[0] = best action)."""
    scored = [ex for ex in examples if ex.gold_ranking and ex.candidate_ops]
    if not scored:
        return 0.0
    total = 0.0
    for ex in scored:
        ranking = policy.rank_actions(ex.policy_input, ex.candidate_ops)
        total += _reciprocal_rank(ranking.ranked_actions, ex.gold_ranking[0])
    return total / len(scored)


def top_k_accuracy(
    policy: LocalPolicy,
    examples: Sequence[EvalExample],
    k: int = 3,
) -> float:
    """Fraction of examples where gold_ranking[0] appears in the top-k predicted."""
    scored = [ex for ex in examples if ex.gold_ranking and ex.candidate_ops]
    if not scored:
        return 0.0
    hits = 0
    for ex in scored:
        ranking = policy.rank_actions(ex.policy_input, ex.candidate_ops)
        if ex.gold_ranking[0] in ranking.ranked_actions[:k]:
            hits += 1
    return hits / len(scored)


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    n_examples: int = 0
    n_focus_examples: int = 0
    n_ranking_examples: int = 0
    focus_accuracy: float = 0.0
    mrr: float = 0.0
    top3_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "n_focus_examples": self.n_focus_examples,
            "n_ranking_examples": self.n_ranking_examples,
            "focus_accuracy": round(self.focus_accuracy, 4),
            "mrr": round(self.mrr, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
        }

    def summary(self) -> str:
        lines = [
            f"examples        : {self.n_examples}",
            f"  focus         : {self.n_focus_examples}",
            f"  ranking       : {self.n_ranking_examples}",
            f"focus_accuracy  : {self.focus_accuracy:.4f}",
            f"mrr             : {self.mrr:.4f}",
            f"top3_accuracy   : {self.top3_accuracy:.4f}",
        ]
        return "\n".join(lines)


def evaluate(
    policy: LocalPolicy,
    examples: Sequence[EvalExample],
) -> EvalReport:
    """Run the full evaluation suite and return a compact report."""
    focus_exs = [ex for ex in examples if ex.gold_focus is not None]
    rank_exs = [ex for ex in examples if ex.gold_ranking and ex.candidate_ops]

    return EvalReport(
        n_examples=len(examples),
        n_focus_examples=len(focus_exs),
        n_ranking_examples=len(rank_exs),
        focus_accuracy=focus_accuracy(policy, examples),
        mrr=mean_reciprocal_rank(policy, examples),
        top3_accuracy=top_k_accuracy(policy, examples),
    )
