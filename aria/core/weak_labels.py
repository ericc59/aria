"""Weak supervision labels from existing audits.

Produces per-task labels for mechanism class, executor sufficiency,
and composition plausibility. Confidence derived from evidence/audit
signals, not from learned models.

Part of the canonical architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class WeakLabel:
    """Weak supervision label for one task."""
    task_id: str
    # Mechanism class
    top_class: str              # best structural hypothesis
    top_class_score: float      # 0-1
    alt_classes: tuple[str, ...]  # plausible alternatives
    # Executor sufficiency
    exec_sufficient: bool       # current op can realize this class
    exec_confidence: str        # "high", "medium", "low", "none"
    # Composition
    composition_plausible: bool
    composition_sequences: tuple[str, ...] = ()
    # Overall
    solved: bool = False
    confidence: str = "low"     # "high", "medium", "low"


def label_task(
    task_id: str,
    demos: Sequence[Any],
) -> WeakLabel:
    """Produce weak labels for one task from existing evidence/audit."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve

    ev, ranking = compute_evidence_and_rank(demos)

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()
    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)

    top = ranking.lanes[0] if ranking.lanes else None
    top_class = top.name if top else ""
    top_score = top.final_score if top else 0.0

    # Alternatives: lanes with gate_pass and score > 0
    alts = tuple(c.name for c in ranking.lanes[1:]
                 if c.gate_pass and c.final_score > 0.1)

    # Executor sufficiency
    exec_suf = result.solved
    if result.solved:
        exec_conf = "high"
    elif top and top.exec_hint >= 0.5:
        exec_conf = "medium"
    elif top and top.exec_hint > 0:
        exec_conf = "low"
    else:
        exec_conf = "none"

    # Composition plausibility
    gated = [c for c in ranking.lanes if c.gate_pass]
    comp_plausible = len(gated) >= 2 or (ev.has_framed_region and ev.n_input_singles > 0)
    comp_seqs: list[str] = []
    if ev.has_framed_region and ev.n_input_singles > 0:
        comp_seqs.append("periodic->relocation")
    if ev.has_framed_region and ev.output_grows_shapes:
        comp_seqs.append("periodic->replication")
    if not ev.same_dims:
        comp_seqs.append("canvas->relocation")

    # Overall confidence
    if result.solved:
        confidence = "high"
    elif top_score >= 0.5 and exec_conf in ("medium", "high"):
        confidence = "medium"
    else:
        confidence = "low"

    return WeakLabel(
        task_id=task_id,
        top_class=top_class,
        top_class_score=round(top_score, 3),
        alt_classes=alts,
        exec_sufficient=exec_suf,
        exec_confidence=exec_conf,
        composition_plausible=comp_plausible,
        composition_sequences=tuple(comp_seqs),
        solved=result.solved,
        confidence=confidence,
    )


def label_batch(
    task_ids: list[str],
    demos_fn,
) -> list[WeakLabel]:
    """Label a batch of tasks."""
    labels = []
    for tid in task_ids:
        try:
            demos = demos_fn(tid)
            labels.append(label_task(tid, demos))
        except Exception:
            pass
    return labels


def format_label_summary(labels: list[WeakLabel]) -> str:
    """Format label distribution summary."""
    from collections import Counter
    lines = [f"=== Weak Labels ({len(labels)} tasks) ==="]

    # Class distribution
    classes = Counter(l.top_class for l in labels)
    lines.append("\nTop class distribution:")
    for cls, count in classes.most_common():
        lines.append(f"  {cls}: {count}")

    # Confidence
    confs = Counter(l.confidence for l in labels)
    lines.append("\nOverall confidence:")
    for conf, count in confs.most_common():
        lines.append(f"  {conf}: {count}")

    # Exec confidence
    exec_confs = Counter(l.exec_confidence for l in labels)
    lines.append("\nExecutor confidence:")
    for conf, count in exec_confs.most_common():
        lines.append(f"  {conf}: {count}")

    # Composition
    comp = sum(1 for l in labels if l.composition_plausible)
    lines.append(f"\nComposition plausible: {comp}/{len(labels)}")

    # Solved
    solved = sum(1 for l in labels if l.solved)
    lines.append(f"Solved: {solved}/{len(labels)}")

    return "\n".join(lines)


def save_labels(labels: list[WeakLabel], path: str) -> None:
    """Save labels as JSONL."""
    from dataclasses import asdict
    with open(path, "w") as f:
        for l in labels:
            f.write(json.dumps(asdict(l), default=str) + "\n")
