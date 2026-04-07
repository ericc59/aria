"""Lane composition audit — detect multi-stage task structure.

Inspects unsolved tasks to determine whether they plausibly require
a composition of existing lanes rather than a missing new lane.

Composition labels:
  single_lane       — one existing lane should suffice (executor gap)
  two_stage         — two existing lanes in sequence could work
  cross_lane        — ambiguous between multiple single-lane interpretations
  no_lane_match     — no current lane matches the task structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


LABEL_SINGLE_LANE = "single_lane"
LABEL_TWO_STAGE = "two_stage"
LABEL_CROSS_LANE = "cross_lane"
LABEL_NO_MATCH = "no_lane_match"


@dataclass
class CompositionRecord:
    task_id: str
    label: str
    rationale: str
    plausible_sequences: list[str] = field(default_factory=list)
    # Structural cues
    has_frame: bool = False
    has_objects: bool = False
    has_markers: bool = False
    dims_change: bool = False
    output_grows: bool = False
    n_distinct_regions: int = 0


@dataclass
class CompositionReport:
    n_tasks: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    sequence_counts: dict[str, int] = field(default_factory=dict)
    records: list[CompositionRecord] = field(default_factory=list)


def audit_composition(
    task_ids: list[str],
    demos_fn,
) -> CompositionReport:
    """Audit unsolved tasks for composition potential."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.decomposition import detect_bg, detect_framed_regions
    from aria.runtime.ops.selection import _find_objects

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    report = CompositionReport()
    for label in (LABEL_SINGLE_LANE, LABEL_TWO_STAGE, LABEL_CROSS_LANE, LABEL_NO_MATCH):
        report.label_counts[label] = 0

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        # Skip solved tasks
        result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=tid)
        if result.solved:
            continue

        report.n_tasks += 1
        ev, ranking = compute_evidence_and_rank(demos)
        demo = demos[0]

        # Structural cues
        bg = detect_bg(demo.input)
        has_frame = False
        try:
            regions = detect_framed_regions(demo.input)
            has_frame = len(regions) > 0
        except Exception:
            pass

        all_objs = list(_find_objects(demo.input))
        shapes = [o for o in all_objs if o.color != bg and o.size > 1]
        markers = [o for o in all_objs if o.color != bg and o.size == 1]
        dims_change = demo.input.shape != demo.output.shape

        out_objs = list(_find_objects(demo.output))
        out_shapes = [o for o in out_objs if o.color != detect_bg(demo.output) and o.size > 1]
        output_grows = len(out_shapes) > len(shapes)

        # Count gated lanes
        gated = [c for c in ranking.lanes if c.gate_pass]
        n_gated = len(gated)

        # Classify
        label, rationale, sequences = _classify_composition(
            ev, ranking, has_frame, shapes, markers,
            dims_change, output_grows, n_gated,
        )

        rec = CompositionRecord(
            task_id=tid,
            label=label,
            rationale=rationale,
            plausible_sequences=sequences,
            has_frame=has_frame,
            has_objects=len(shapes) > 0,
            has_markers=len(markers) > 0,
            dims_change=dims_change,
            output_grows=output_grows,
        )
        report.records.append(rec)
        report.label_counts[label] += 1
        for seq in sequences:
            report.sequence_counts[seq] = report.sequence_counts.get(seq, 0) + 1

    return report


def _classify_composition(
    ev, ranking, has_frame, shapes, markers,
    dims_change, output_grows, n_gated,
) -> tuple[str, str, list[str]]:
    """Classify an unsolved task's composition potential."""
    sequences: list[str] = []

    # No structural match at all
    if n_gated == 0:
        return LABEL_NO_MATCH, "no lane gates pass", []

    # Strong single-lane evidence with no composition cues
    top = ranking.lanes[0]
    if top.final_score > 0.5 and n_gated == 1:
        return LABEL_SINGLE_LANE, f"strong {top.name} fit, sole gated lane", []

    # Composition cues: frame + objects/markers suggests periodic + relocation/replication
    if has_frame and (len(shapes) > 0 or len(markers) > 0):
        if output_grows:
            sequences.append("periodic->replication")
        if len(markers) > 0:
            sequences.append("periodic->relocation")
        sequences.append("periodic->transform")

    # Dims change + objects suggests canvas + relocation
    if dims_change and len(shapes) > 0:
        sequences.append("canvas->relocation")
        sequences.append("canvas->replication")

    # Objects + markers + frame suggests extract->repair->paint
    if has_frame and len(markers) > 0 and len(shapes) > 0:
        sequences.append("extract->repair->paint")

    # Multiple gated lanes with moderate scores
    if n_gated >= 2 and not sequences:
        lane_names = [c.name for c in ranking.lanes if c.gate_pass]
        return LABEL_CROSS_LANE, f"ambiguous among {lane_names}", []

    if sequences:
        return LABEL_TWO_STAGE, f"multi-stage: {sequences[0]}", sequences

    # Single gated lane, low score
    if n_gated == 1:
        return LABEL_SINGLE_LANE, f"weak {top.name} fit", []

    return LABEL_CROSS_LANE, "multiple lanes gated, no clear composition", []


def format_composition_audit(report: CompositionReport) -> str:
    lines = [
        f"=== Lane Composition Audit ({report.n_tasks} unsolved tasks) ===",
        "",
        "Label distribution:",
    ]
    for label, count in sorted(report.label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")

    if report.sequence_counts:
        lines.append("")
        lines.append("Plausible lane sequences:")
        for seq, count in sorted(report.sequence_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {seq}: {count}")

    return "\n".join(lines)
