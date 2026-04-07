"""Proposal-evaluation harness — measure guidance quality before wiring into search.

Metrics measure whether the symbolic system's proposals contain the correct
answer, independently of whether the solver actually reaches it. This tells
us whether guidance is good before it is wired into search.

No solver changes. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from aria.core.guidance_labels import (
    CorrespondenceLabel,
    DerivationLabel,
    LegendLabel,
    OutputSizeLabel,
    ProgramFamilyLabel,
    SlotGridLabel,
    TaskGuidanceLabels,
)


# ---------------------------------------------------------------------------
# Recall@K metrics
# ---------------------------------------------------------------------------


def recall_at_k(proposals: Sequence[str], gold: str, k: int) -> bool:
    """Check if gold appears in the top-k proposals."""
    return gold in proposals[:k]


def recall_at_k_output_size(
    proposal_modes: Sequence[str],
    gold: OutputSizeLabel,
    k: int,
) -> bool:
    """Does the gold output-size mode appear in top-k proposed modes?"""
    return gold.mode in proposal_modes[:k]


def recall_at_k_derivation(
    proposal_specs: Sequence[dict],
    gold: DerivationLabel,
    k: int,
) -> bool:
    """Does the gold derivation spec appear in top-k proposals?

    Matches on (candidate_kind, relation, selector) triple.
    """
    gold_key = (gold.candidate_kind, gold.relation, gold.selector)
    for spec in proposal_specs[:k]:
        prop_key = (spec.get("candidate_kind"), spec.get("relation"), spec.get("selector"))
        if prop_key == gold_key:
            return True
    return False


def recall_at_k_program_family(
    proposal_families: Sequence[str],
    gold: ProgramFamilyLabel,
    k: int,
) -> bool:
    """Does the gold program family appear in top-k proposals?"""
    return gold.top_family in proposal_families[:k]


# ---------------------------------------------------------------------------
# Detection precision/recall/F1
# ---------------------------------------------------------------------------


@dataclass
class PRFResult:
    """Precision, recall, F1 for a binary detection task."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def legend_detection_prf(
    predictions: Sequence[LegendLabel],
    golds: Sequence[LegendLabel],
) -> PRFResult:
    """Precision/recall/F1 for legend detection (presence + edge match)."""
    result = PRFResult()
    for pred, gold in zip(predictions, golds):
        if gold.present and pred.present and pred.edge == gold.edge:
            result.tp += 1
        elif gold.present and not pred.present:
            result.fn += 1
        elif not gold.present and pred.present:
            result.fp += 1
        else:
            result.tn += 1
    return result


def slot_grid_detection_prf(
    predictions: Sequence[SlotGridLabel],
    golds: Sequence[SlotGridLabel],
) -> PRFResult:
    """Precision/recall for slot-grid detection (presence + dims match)."""
    result = PRFResult()
    for pred, gold in zip(predictions, golds):
        if gold.present and pred.present:
            if pred.n_rows == gold.n_rows and pred.n_cols == gold.n_cols:
                result.tp += 1
            else:
                # Detected but wrong dims: count as FP + FN
                result.fp += 1
                result.fn += 1
        elif gold.present and not pred.present:
            result.fn += 1
        elif not gold.present and pred.present:
            result.fp += 1
        else:
            result.tn += 1
    return result


def correspondence_recall(
    predicted_pairs: int,
    gold: CorrespondenceLabel,
) -> float:
    """Recall of correspondence pairs (simple count-based)."""
    if gold.n_pairs == 0:
        return 1.0 if predicted_pairs == 0 else 0.0
    return min(predicted_pairs, gold.n_pairs) / gold.n_pairs


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregate evaluation of proposal quality."""
    n_tasks: int = 0

    # Recall@k for output size
    size_recall_at_1: float = 0.0
    size_recall_at_3: float = 0.0
    size_recall_at_5: float = 0.0
    size_n_evaluated: int = 0

    # Recall@k for derivation
    deriv_recall_at_1: float = 0.0
    deriv_recall_at_3: float = 0.0
    deriv_n_evaluated: int = 0

    # Legend detection
    legend_prf: PRFResult = field(default_factory=PRFResult)

    # Slot grid detection
    slot_grid_prf: PRFResult = field(default_factory=PRFResult)

    # Program family recall@k
    family_recall_at_1: float = 0.0
    family_recall_at_3: float = 0.0
    family_n_evaluated: int = 0

    # Per-task details
    per_task: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "output_size": {
                "recall_at_1": round(self.size_recall_at_1, 4),
                "recall_at_3": round(self.size_recall_at_3, 4),
                "recall_at_5": round(self.size_recall_at_5, 4),
                "n_evaluated": self.size_n_evaluated,
            },
            "derivation": {
                "recall_at_1": round(self.deriv_recall_at_1, 4),
                "recall_at_3": round(self.deriv_recall_at_3, 4),
                "n_evaluated": self.deriv_n_evaluated,
            },
            "legend": self.legend_prf.to_dict(),
            "slot_grid": self.slot_grid_prf.to_dict(),
            "program_family": {
                "recall_at_1": round(self.family_recall_at_1, 4),
                "recall_at_3": round(self.family_recall_at_3, 4),
                "n_evaluated": self.family_n_evaluated,
            },
        }


def evaluate_proposals(
    records: list[dict],
    labels: list[TaskGuidanceLabels],
) -> EvalReport:
    """Evaluate proposal quality from exported records against gold labels.

    Records come from guidance_export.export_task().
    Labels come from guidance_labels.extract_labels().

    The "proposals" are what the current symbolic system produces.
    The "gold" labels are verified ground truth from the same system.
    This measures internal consistency and coverage — how well the system's
    own proposals rank the correct answer.
    """
    label_by_id = {l.task_id: l for l in labels}
    report = EvalReport(n_tasks=len(records))

    size_hits_1, size_hits_3, size_hits_5, size_total = 0, 0, 0, 0
    deriv_hits_1, deriv_hits_3, deriv_total = 0, 0, 0
    family_hits_1, family_hits_3, family_total = 0, 0, 0

    legend_preds: list[LegendLabel] = []
    legend_golds: list[LegendLabel] = []
    sg_preds: list[SlotGridLabel] = []
    sg_golds: list[SlotGridLabel] = []

    for record in records:
        tid = record.get("task_id", "")
        gold = label_by_id.get(tid)
        if gold is None:
            continue

        task_detail: dict[str, Any] = {"task_id": tid}

        # Output size: proposals = lane ranking order of size modes
        if gold.output_size is not None:
            lane_modes = [l["name"] for l in record.get("lane_ranking", [])]
            # Also include the actual size_spec mode as the top proposal
            proposals = []
            ss = record.get("size_spec")
            if ss and "mode" in ss:
                proposals.append(ss["mode"])
            proposals.extend(m for m in lane_modes if m not in proposals)

            h1 = recall_at_k_output_size(proposals, gold.output_size, 1)
            h3 = recall_at_k_output_size(proposals, gold.output_size, 3)
            h5 = recall_at_k_output_size(proposals, gold.output_size, 5)
            size_hits_1 += h1
            size_hits_3 += h3
            size_hits_5 += h5
            size_total += 1
            task_detail["size_hit_at_1"] = h1

        # Derivation
        if gold.derivation is not None:
            ds = record.get("derivation_spec")
            proposals = [ds] if ds else []
            h1 = recall_at_k_derivation(proposals, gold.derivation, 1)
            h3 = recall_at_k_derivation(proposals, gold.derivation, 3)
            deriv_hits_1 += h1
            deriv_hits_3 += h3
            deriv_total += 1
            task_detail["deriv_hit_at_1"] = h1

        # Legend detection
        rec_legend = record.get("legend")
        pred_legend = LegendLabel(
            present=rec_legend is not None,
            edge=rec_legend.get("edge") if rec_legend else None,
            n_entries=rec_legend.get("n_entries", 0) if rec_legend else 0,
        )
        legend_preds.append(pred_legend)
        legend_golds.append(gold.legend)

        # Slot grid detection
        rec_sg = record.get("slot_grid")
        pred_sg = SlotGridLabel(
            present=rec_sg is not None,
            n_rows=rec_sg.get("n_rows") if rec_sg else None,
            n_cols=rec_sg.get("n_cols") if rec_sg else None,
        )
        sg_preds.append(pred_sg)
        sg_golds.append(gold.slot_grid)

        # Program family
        if gold.program_family.top_family:
            lane_names = [l["name"] for l in record.get("lane_ranking", [])]
            h1 = recall_at_k_program_family(lane_names, gold.program_family, 1)
            h3 = recall_at_k_program_family(lane_names, gold.program_family, 3)
            family_hits_1 += h1
            family_hits_3 += h3
            family_total += 1
            task_detail["family_hit_at_1"] = h1

        report.per_task.append(task_detail)

    # Aggregate
    report.size_recall_at_1 = size_hits_1 / max(size_total, 1)
    report.size_recall_at_3 = size_hits_3 / max(size_total, 1)
    report.size_recall_at_5 = size_hits_5 / max(size_total, 1)
    report.size_n_evaluated = size_total

    report.deriv_recall_at_1 = deriv_hits_1 / max(deriv_total, 1)
    report.deriv_recall_at_3 = deriv_hits_3 / max(deriv_total, 1)
    report.deriv_n_evaluated = deriv_total

    report.legend_prf = legend_detection_prf(legend_preds, legend_golds)
    report.slot_grid_prf = slot_grid_detection_prf(sg_preds, sg_golds)

    report.family_recall_at_1 = family_hits_1 / max(family_total, 1)
    report.family_recall_at_3 = family_hits_3 / max(family_total, 1)
    report.family_n_evaluated = family_total

    return report
