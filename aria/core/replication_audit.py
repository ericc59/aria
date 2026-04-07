"""Replication-specific failure audit.

For tasks with high replication class-fit, diagnoses why the current
replicate_templates executor fails and labels the dominant failure mode.

Failure labels:
  verified              — executor succeeds
  wrong_exemplar        — no anchored exemplar found in the task
  wrong_anchor_key      — anchor found but output clones don't appear at predicted positions
  wrong_target_set      — anchors found but no targets with matching key color
  wrong_placement       — some clones match but not all (placement offset wrong)
  wrong_source_policy   — clones match but source retention/erasure is wrong
  not_replication       — output shape count doesn't grow (task is not replication)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.core.mechanism_evidence import compute_evidence_and_rank


LABEL_VERIFIED = "verified"
LABEL_WRONG_EXEMPLAR = "wrong_exemplar"
LABEL_WRONG_ANCHOR_KEY = "wrong_anchor_key"
LABEL_WRONG_TARGET_SET = "wrong_target_set"
LABEL_WRONG_PLACEMENT = "wrong_placement"
LABEL_WRONG_SOURCE_POLICY = "wrong_source_policy"
LABEL_NOT_REPLICATION = "not_replication"


@dataclass
class ReplicationAuditRecord:
    task_id: str
    class_score: float
    best_diff: int
    total_pixels: int
    diff_fraction: float
    failure_label: str
    n_anchored: int = 0
    n_targets: int = 0
    n_output_matches: int = 0
    output_grows: bool = False


@dataclass
class ReplicationAuditReport:
    n_tasks: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    records: list[ReplicationAuditRecord] = field(default_factory=list)
    near_misses: list[ReplicationAuditRecord] = field(default_factory=list)


def audit_replication(
    task_ids: list[str],
    demos_fn,
    *,
    class_threshold: float = 0.7,
) -> ReplicationAuditReport:
    """Audit replication executor failures on high-fit tasks."""
    from aria.runtime.ops.replicate import (
        _replicate_templates, _find_anchor, ALL_KEY_RULES, ALL_SOURCE_POLICIES,
        ALL_PLACE_RULES, KEY_ADJACENT_DIFF_COLOR,
    )
    from aria.runtime.ops.relate_paint import _get_shapes_and_markers
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.executor import execute
    from aria.decomposition import detect_bg
    from aria.types import Program, Bind, Call, Literal, Ref, Type

    report = ReplicationAuditReport()

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        ev, ranking = compute_evidence_and_rank(demos)
        repl = next((c for c in ranking.lanes if c.name == "replication"), None)
        if repl is None or repl.class_score < class_threshold:
            continue

        report.n_tasks += 1
        total_pixels = sum(d.output.size for d in demos)

        # Try all replication params
        best_diff = float("inf")
        for kr in ALL_KEY_RULES:
            for sp in ALL_SOURCE_POLICIES:
                for pr in ALL_PLACE_RULES:
                    prog = Program(
                        steps=(Bind("v0", Type.GRID, Call("replicate_templates", (
                            Ref("input"), Literal(kr, Type.INT),
                            Literal(sp, Type.INT), Literal(pr, Type.INT),
                        ))),),
                        output="v0",
                    )
                    try:
                        diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
                        best_diff = min(best_diff, diff)
                    except Exception:
                        pass

        if best_diff == float("inf"):
            best_diff = total_pixels

        # Diagnose on first demo
        demo = demos[0]
        bg = detect_bg(demo.input)
        shapes, singles, _ = _get_shapes_and_markers(demo.input)
        out_shapes = [o for o in _find_objects(demo.output) if o.color != bg and o.size > 1]

        n_anchored = 0
        n_targets = 0
        n_output_matches = 0
        for s in shapes:
            anchor, offset = _find_anchor(s, singles, KEY_ADJACENT_DIFF_COLOR)
            if anchor is None:
                continue
            key = anchor.color
            tgts = [m for m in singles if m.color == key and m is not anchor]
            if not tgts:
                continue
            n_anchored += 1
            n_targets += len(tgts)
            for t in tgts:
                tx, ty = t.bbox[0], t.bbox[1]
                clone_r = ty - offset[0]
                clone_c = tx - offset[1]
                if any(so.color == s.color and so.size == s.size
                       and so.bbox[0] == clone_c and so.bbox[1] == clone_r
                       for so in out_shapes):
                    n_output_matches += 1

        # Output shape growth
        total_in = sum(len([o for o in _find_objects(d.input)
                            if o.color != detect_bg(d.input) and o.size > 1]) for d in demos)
        total_out = sum(len([o for o in _find_objects(d.output)
                             if o.color != detect_bg(d.output) and o.size > 1]) for d in demos)
        output_grows = total_out > total_in

        # Classify failure
        frac = best_diff / max(total_pixels, 1)
        if best_diff == 0:
            label = LABEL_VERIFIED
        elif not output_grows and total_out <= total_in:
            label = LABEL_NOT_REPLICATION
        elif n_anchored == 0:
            label = LABEL_WRONG_EXEMPLAR
        elif n_targets == 0:
            label = LABEL_WRONG_TARGET_SET
        elif n_output_matches == 0:
            label = LABEL_WRONG_ANCHOR_KEY
        elif n_output_matches < n_targets and frac >= 0.10:
            label = LABEL_WRONG_PLACEMENT
        else:
            label = LABEL_WRONG_SOURCE_POLICY

        rec = ReplicationAuditRecord(
            task_id=tid,
            class_score=repl.class_score,
            best_diff=int(best_diff),
            total_pixels=total_pixels,
            diff_fraction=frac,
            failure_label=label,
            n_anchored=n_anchored,
            n_targets=n_targets,
            n_output_matches=n_output_matches,
            output_grows=output_grows,
        )
        report.records.append(rec)
        report.label_distribution[label] = report.label_distribution.get(label, 0) + 1

    # Near-misses: lowest non-zero diff
    report.near_misses = sorted(
        [r for r in report.records if r.best_diff > 0],
        key=lambda r: r.diff_fraction,
    )[:10]

    return report


def format_replication_audit(report: ReplicationAuditReport) -> str:
    lines = [
        f"=== Replication Failure Audit ({report.n_tasks} high-fit tasks) ===",
        "",
        "Failure label distribution:",
    ]
    for label, count in sorted(report.label_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")

    if report.near_misses:
        lines.append("")
        lines.append("Top near-misses:")
        for r in report.near_misses[:5]:
            lines.append(f"  {r.task_id}: diff={r.best_diff}/{r.total_pixels} ({r.diff_fraction:.1%}) "
                        f"[{r.failure_label}] anchored={r.n_anchored} targets={r.n_targets} "
                        f"matches={r.n_output_matches} grows={r.output_grows}")
    return "\n".join(lines)
