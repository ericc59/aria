"""Offline evaluator for local refinement policies.

Reads JSONL evaluation data, runs a ``LocalPolicy`` on each example,
and scores accuracy for NEXT_FOCUS prediction and action-ranking tasks.

Metrics:
  - focus_accuracy: exact-match on predicted focus label
  - focus_top_k_accuracy: gold focus in top-k predicted labels
  - mrr: mean reciprocal rank for action ranking
  - top_k_accuracy: gold best-action in top-k predicted
  - ndcg: normalized discounted cumulative gain for full ranking
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
# Optional (for edit recovery scoring):
#   gold_edit_before : str
#   gold_edit_after  : str
#


@dataclass(frozen=True)
class EvalExample:
    """Single evaluation example parsed from JSONL."""
    policy_input: PolicyInput
    gold_focus: str | None = None
    gold_ranking: tuple[str, ...] | None = None
    candidate_ops: tuple[str, ...] = ()
    gold_edit_before: str | None = None
    gold_edit_after: str | None = None


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
                gold_edit_before=obj.get("gold_edit_before"),
                gold_edit_after=obj.get("gold_edit_after"),
            ))
    return examples


# ---------------------------------------------------------------------------
# Focus metrics
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


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

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


def _dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(
    policy: LocalPolicy,
    examples: Sequence[EvalExample],
) -> float:
    """NDCG over ranking examples using positional relevance from gold ranking."""
    scored = [ex for ex in examples if ex.gold_ranking and ex.candidate_ops]
    if not scored:
        return 0.0
    total = 0.0
    for ex in scored:
        ranking = policy.rank_actions(ex.policy_input, ex.candidate_ops)
        n = len(ex.gold_ranking)
        # Relevance: position-based, best item gets highest relevance.
        gold_rel = {item: n - i for i, item in enumerate(ex.gold_ranking)}

        predicted_rels = [float(gold_rel.get(a, 0)) for a in ranking.ranked_actions]
        ideal_rels = sorted(predicted_rels, reverse=True)

        ideal = _dcg(ideal_rels)
        if ideal == 0:
            continue
        total += _dcg(predicted_rels) / ideal
    count = sum(1 for ex in scored
                if _dcg([float(len(ex.gold_ranking) - i) for i in range(len(ex.gold_ranking))]) > 0)
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Edit recovery metrics
# ---------------------------------------------------------------------------

def _normalized_edit_distance(a: str, b: str) -> float:
    """Line-level normalized edit distance (0 = identical, 1 = completely different)."""
    lines_a = a.strip().splitlines()
    lines_b = b.strip().splitlines()
    m, n = len(lines_a), len(lines_b)
    if m == 0 and n == 0:
        return 0.0

    # DP for line-level Levenshtein
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if lines_a[i - 1] == lines_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max(m, n)


def edit_recovery_exact_match(
    predicted_edits: Sequence[str],
    gold_edits: Sequence[str],
) -> float:
    """Fraction of predicted edits that exactly match the gold edit."""
    if not gold_edits:
        return 0.0
    matches = sum(1 for p, g in zip(predicted_edits, gold_edits)
                  if p.strip() == g.strip())
    return matches / len(gold_edits)


def edit_recovery_similarity(
    predicted_edits: Sequence[str],
    gold_edits: Sequence[str],
) -> float:
    """Mean (1 - normalized_edit_distance) over predicted vs gold edits."""
    if not gold_edits:
        return 0.0
    total = sum(1.0 - _normalized_edit_distance(p, g)
                for p, g in zip(predicted_edits, gold_edits))
    return total / len(gold_edits)


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
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    ndcg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_focus_examples": self.n_focus_examples,
            "n_ranking_examples": self.n_ranking_examples,
            "focus_accuracy": round(self.focus_accuracy, 4),
            "mrr": round(self.mrr, 4),
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "ndcg": round(self.ndcg, 4),
        }

    def summary(self) -> str:
        lines = [
            f"examples        : {self.n_examples}",
            f"  focus         : {self.n_focus_examples}",
            f"  ranking       : {self.n_ranking_examples}",
            f"focus_accuracy  : {self.focus_accuracy:.4f}",
            f"mrr             : {self.mrr:.4f}",
            f"top1_accuracy   : {self.top1_accuracy:.4f}",
            f"top3_accuracy   : {self.top3_accuracy:.4f}",
            f"ndcg            : {self.ndcg:.4f}",
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
        top1_accuracy=top_k_accuracy(policy, examples, k=1),
        top3_accuracy=top_k_accuracy(policy, examples),
        ndcg=ndcg(policy, examples),
    )


# ---------------------------------------------------------------------------
# Scorecard: compact, comparable across runs
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    """Aggregates policy eval + dataset stats into one comparable object."""
    run_label: str = ""
    policy_name: str = ""
    eval_report: dict[str, Any] = field(default_factory=dict)
    dataset_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_label": self.run_label,
            "policy_name": self.policy_name,
            "eval": self.eval_report,
            "dataset": self.dataset_stats,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [f"=== Scorecard: {self.run_label} (policy={self.policy_name}) ==="]

        ev = self.eval_report
        if ev:
            lines.append(f"  focus_accuracy : {ev.get('focus_accuracy', 'N/A')}")
            lines.append(f"  mrr            : {ev.get('mrr', 'N/A')}")
            lines.append(f"  top1_accuracy  : {ev.get('top1_accuracy', 'N/A')}")
            lines.append(f"  top3_accuracy  : {ev.get('top3_accuracy', 'N/A')}")
            lines.append(f"  ndcg           : {ev.get('ndcg', 'N/A')}")

        ds = self.dataset_stats
        if ds:
            lines.append(f"  dataset_total  : {ds.get('total', 'N/A')}")
            lines.append(f"  task_types     : {ds.get('by_task_type', 'N/A')}")
            lines.append(f"  unique_tasks   : {ds.get('unique_task_ids', 'N/A')}")

        return "\n".join(lines)


def compare_scorecards(cards: Sequence[Scorecard]) -> str:
    """Format multiple scorecards side-by-side for comparison."""
    if not cards:
        return "(no scorecards)"

    metrics = ["focus_accuracy", "mrr", "top1_accuracy", "top3_accuracy", "ndcg"]
    labels = [c.run_label or c.policy_name or f"run_{i}" for i, c in enumerate(cards)]

    header = f"{'metric':<18}" + "".join(f"{l:>14}" for l in labels)
    lines = [header, "-" * len(header)]

    for m in metrics:
        row = f"{m:<18}"
        for c in cards:
            val = c.eval_report.get(m)
            row += f"{val:>14.4f}" if isinstance(val, (int, float)) else f"{'N/A':>14}"
        lines.append(row)

    # Dataset row
    row = f"{'dataset_total':<18}"
    for c in cards:
        val = c.dataset_stats.get("total", "N/A")
        row += f"{val:>14}" if isinstance(val, int) else f"{str(val):>14}"
    lines.append(row)

    return "\n".join(lines)
