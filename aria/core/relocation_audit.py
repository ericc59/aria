"""Relocation-specific failure audit.

For tasks with high relocation class-fit, diagnoses why the current
relocate_objects executor fails.

Failure labels:
  verified             — executor succeeds
  wrong_pairing        — objects exist in both in/out but pairing is wrong
  wrong_alignment      — pairing seems right but placement offset is wrong
  wrong_extraction     — wrong objects selected as shapes or markers
  wrong_count          — shape count changes between in/out (not 1:1 relocation)
  wrong_retention      — shapes at right positions but source erasure/retention wrong
  not_relocation       — task doesn't structurally match relocation (e.g. dims change)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.core.mechanism_evidence import compute_evidence_and_rank


LABEL_VERIFIED = "verified"
LABEL_WRONG_PAIRING = "wrong_pairing"
LABEL_WRONG_ALIGNMENT = "wrong_alignment"
LABEL_WRONG_EXTRACTION = "wrong_extraction"
LABEL_WRONG_COUNT = "wrong_count"
LABEL_WRONG_RETENTION = "wrong_retention"
LABEL_NOT_RELOCATION = "not_relocation"


@dataclass
class RelocationAuditRecord:
    task_id: str
    class_score: float
    best_diff: int
    total_pixels: int
    diff_fraction: float
    failure_label: str
    n_in_shapes: int = 0
    n_out_shapes: int = 0
    n_in_singles: int = 0
    same_shape_set: bool = False


@dataclass
class RelocationAuditReport:
    n_tasks: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    records: list[RelocationAuditRecord] = field(default_factory=list)
    near_misses: list[RelocationAuditRecord] = field(default_factory=list)


def audit_relocation(
    task_ids: list[str],
    demos_fn,
    *,
    class_threshold: float = 0.3,
) -> RelocationAuditReport:
    """Audit relocation executor failures on class-fit tasks."""
    from aria.runtime.ops.relate_paint import (
        _relocate_objects, ALL_MATCH_RULES, ALL_ALIGNS,
    )
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.executor import execute
    from aria.decomposition import detect_bg
    from aria.types import Program, Bind, Call, Literal, Ref, Type

    report = RelocationAuditReport()

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        ev, ranking = compute_evidence_and_rank(demos)
        reloc = next((c for c in ranking.lanes if c.name == "relocation"), None)
        if reloc is None or reloc.class_score < class_threshold:
            continue

        report.n_tasks += 1
        total_pixels = sum(d.output.size for d in demos)

        # Try all relocation params
        best_diff = float("inf")
        for mr in ALL_MATCH_RULES:
            for al in ALL_ALIGNS:
                prog = Program(
                    steps=(Bind("v0", Type.GRID, Call("relocate_objects", (
                        Ref("input"), Literal(mr, Type.INT), Literal(al, Type.INT),
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

        # Structural analysis on demos
        demo = demos[0]
        bg = detect_bg(demo.input)
        in_shapes = [o for o in _find_objects(demo.input) if o.color != bg and o.size > 1]
        out_shapes = [o for o in _find_objects(demo.output) if o.color != bg and o.size > 1]
        in_singles = [o for o in _find_objects(demo.input) if o.color != bg and o.size == 1]

        n_in = len(in_shapes)
        n_out = len(out_shapes)

        # Same shape set?
        in_sig = sorted((o.color, o.size) for o in in_shapes)
        out_sig = sorted((o.color, o.size) for o in out_shapes)
        same_set = in_sig == out_sig

        # Classify failure
        frac = best_diff / max(total_pixels, 1)
        if best_diff == 0:
            label = LABEL_VERIFIED
        elif not ev.same_dims:
            label = LABEL_NOT_RELOCATION
        elif n_in == 0 or len(in_singles) == 0:
            label = LABEL_WRONG_EXTRACTION
        elif abs(n_in - n_out) > max(n_in, n_out) // 2:
            label = LABEL_WRONG_COUNT
        elif same_set and frac < 0.15:
            label = LABEL_WRONG_ALIGNMENT
        elif same_set:
            label = LABEL_WRONG_PAIRING
        elif frac < 0.10:
            label = LABEL_WRONG_RETENTION
        else:
            label = LABEL_WRONG_PAIRING

        rec = RelocationAuditRecord(
            task_id=tid,
            class_score=reloc.class_score,
            best_diff=int(best_diff),
            total_pixels=total_pixels,
            diff_fraction=frac,
            failure_label=label,
            n_in_shapes=n_in,
            n_out_shapes=n_out,
            n_in_singles=len(in_singles),
            same_shape_set=same_set,
        )
        report.records.append(rec)
        report.label_distribution[label] = report.label_distribution.get(label, 0) + 1

    report.near_misses = sorted(
        [r for r in report.records if r.best_diff > 0],
        key=lambda r: r.diff_fraction,
    )[:10]

    return report


def format_relocation_audit(report: RelocationAuditReport) -> str:
    lines = [
        f"=== Relocation Failure Audit ({report.n_tasks} class-fit tasks) ===",
        "",
        "Failure label distribution:",
    ]
    for label, count in sorted(report.label_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")

    if report.near_misses:
        lines.append("")
        lines.append("Top near-misses:")
        for r in report.near_misses[:5]:
            lines.append(
                f"  {r.task_id}: diff={r.best_diff}/{r.total_pixels} ({r.diff_fraction:.1%}) "
                f"[{r.failure_label}] shapes={r.n_in_shapes}->{r.n_out_shapes} "
                f"singles={r.n_in_singles} same_set={r.same_shape_set}"
            )
    return "\n".join(lines)
