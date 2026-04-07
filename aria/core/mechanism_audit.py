"""Mechanism evidence audit — calibration and discrimination analysis.

Runs over a task slice and records per-task:
  - mechanism_class_fit: structural evidence that a mechanism class applies
  - executable_fit: whether the current implementation actually verifies

These are explicitly separated to prevent conflating "the task looks like
replication" with "our replicate_templates op actually solves it."

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.core.mechanism_evidence import MechanismEvidence, LaneRanking, compute_evidence_and_rank


@dataclass
class LaneFit:
    """Per-lane fit assessment for one task."""
    name: str
    class_score: float        # mechanism_class_fit: structural evidence (0-1)
    class_rationale: str
    gate_pass: bool           # whether structural gate is met
    verified: bool = False    # executable_fit: did the lane verify exactly?
    residual_diff: int = -1   # best residual diff if compiled but not verified (-1 = not tried)
    compile_attempted: bool = False


@dataclass
class TaskAuditRecord:
    """Full audit record for one task."""
    task_id: str
    n_demos: int
    solved: bool
    solver_lane: str = ""     # which lane actually solved it
    lanes: list[LaneFit] = field(default_factory=list)
    evidence: MechanismEvidence | None = None
    top_class_lane: str = ""  # best mechanism_class_fit
    top_exec_lane: str = ""   # lane that verified (if any)
    class_exec_match: bool = False  # did top class == verifying lane?


@dataclass
class AuditReport:
    """Aggregate calibration report."""
    n_tasks: int = 0
    n_with_graphs: int = 0
    n_solved: int = 0
    top_lane_verifies: int = 0
    top2_contains_verifier: int = 0
    # Per-lane stats
    replication_top_count: int = 0
    replication_top_verifies: int = 0
    relocation_top_count: int = 0
    relocation_top_verifies: int = 0
    periodic_top_count: int = 0
    periodic_top_verifies: int = 0
    transform_top_count: int = 0
    transform_top_verifies: int = 0
    # False positives: top-ranked but didn't verify
    false_positives: list[tuple[str, str, float]] = field(default_factory=list)  # (task_id, lane, score)
    records: list[TaskAuditRecord] = field(default_factory=list)


def audit_task(
    task_id: str,
    demos: tuple,
    *,
    fitter: Any = None,
    specializer: Any = None,
    compiler: Any = None,
    verifier: Any = None,
) -> TaskAuditRecord:
    """Audit one task: compute evidence, try each lane, record results."""
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.core.graph import CompileSuccess, CompileFailure
    from aria.runtime.ops.replicate import (
        _replicate_templates, ALL_KEY_RULES, ALL_SOURCE_POLICIES, ALL_PLACE_RULES,
    )
    from aria.runtime.ops.relate_paint import (
        _relocate_objects, ALL_MATCH_RULES, ALL_ALIGNS,
    )
    from aria.runtime.executor import execute
    from aria.verify.verifier import verify
    from aria.types import Program, Bind, Call, Literal, Ref, Type

    if fitter is None:
        fitter = ARCFitter()
    if specializer is None:
        specializer = ARCSpecializer()
    if compiler is None:
        compiler = ARCCompiler()
    if verifier is None:
        verifier = ARCVerifier()

    ev, ranking = compute_evidence_and_rank(demos)

    # Static pipeline solve
    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)

    record = TaskAuditRecord(
        task_id=task_id,
        n_demos=len(demos),
        solved=result.solved,
        evidence=ev,
        top_class_lane=ranking.lanes[0].name if ranking.lanes else "",
    )

    # Record each lane's class fit + exec hint
    for candidate in ranking.lanes:
        lf = LaneFit(
            name=candidate.name,
            class_score=candidate.class_score,
            class_rationale=candidate.rationale,
            gate_pass=candidate.gate_pass,
        )
        record.lanes.append(lf)

    # Check executable fit for each lane independently
    # Replication
    repl_lf = next((lf for lf in record.lanes if lf.name == "replication"), None)
    if repl_lf and repl_lf.gate_pass:
        repl_lf.compile_attempted = True
        best_diff = _try_lane_replication(demos)
        repl_lf.residual_diff = best_diff
        repl_lf.verified = best_diff == 0

    # Relocation
    reloc_lf = next((lf for lf in record.lanes if lf.name == "relocation"), None)
    if reloc_lf and reloc_lf.gate_pass:
        reloc_lf.compile_attempted = True
        best_diff = _try_lane_relocation(demos)
        reloc_lf.residual_diff = best_diff
        reloc_lf.verified = best_diff == 0

    # Periodic repair — use the static pipeline result
    periodic_lf = next((lf for lf in record.lanes if lf.name == "periodic_repair"), None)
    if periodic_lf and periodic_lf.gate_pass:
        periodic_lf.compile_attempted = True
        # The fitter already proposed a periodic graph; check if it verified
        for attempt in result.attempts:
            if isinstance(attempt.compile_result, CompileSuccess):
                periodic_lf.verified = attempt.verified
                periodic_lf.residual_diff = 0 if attempt.verified else -1

    # Overall
    verified_lanes = [lf for lf in record.lanes if lf.verified]
    if verified_lanes:
        record.top_exec_lane = verified_lanes[0].name
        record.solver_lane = verified_lanes[0].name
    if result.solved:
        record.solved = True
    record.class_exec_match = record.top_class_lane == record.top_exec_lane and record.top_exec_lane != ""

    return record


def _try_lane_replication(demos: tuple) -> int:
    """Try all replication parameter combinations, return best diff."""
    from aria.runtime.ops.replicate import (
        _replicate_templates, ALL_KEY_RULES, ALL_SOURCE_POLICIES, ALL_PLACE_RULES,
    )
    from aria.runtime.executor import execute
    from aria.types import Program, Bind, Call, Literal, Ref, Type
    from aria.verify.verifier import verify

    best = float("inf")
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
                    vr = verify(prog, demos)
                    if vr.passed:
                        return 0
                    diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
                    best = min(best, diff)
                except Exception:
                    pass
    return int(best) if best < float("inf") else -1


def _try_lane_relocation(demos: tuple) -> int:
    """Try all relocation parameter combinations, return best diff."""
    from aria.runtime.ops.relate_paint import _relocate_objects, ALL_MATCH_RULES, ALL_ALIGNS
    from aria.runtime.executor import execute
    from aria.types import Program, Bind, Call, Literal, Ref, Type
    from aria.verify.verifier import verify

    best = float("inf")
    for mr in ALL_MATCH_RULES:
        for al in ALL_ALIGNS:
            prog = Program(
                steps=(Bind("v0", Type.GRID, Call("relocate_objects", (
                    Ref("input"), Literal(mr, Type.INT), Literal(al, Type.INT),
                ))),),
                output="v0",
            )
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    return 0
                diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
                best = min(best, diff)
            except Exception:
                pass
    return int(best) if best < float("inf") else -1


def run_audit(
    task_ids: list[str],
    demos_fn,  # callable: task_id -> demos
) -> AuditReport:
    """Run the full audit over a task slice."""
    report = AuditReport()

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        record = audit_task(tid, demos)
        report.records.append(record)
        report.n_tasks += 1

        if any(lf.compile_attempted for lf in record.lanes):
            report.n_with_graphs += 1
        if record.solved:
            report.n_solved += 1

        # Top-lane stats
        if record.lanes:
            top = record.lanes[0]
            top_name = top.name

            if top_name == "replication":
                report.replication_top_count += 1
                if top.verified:
                    report.replication_top_verifies += 1
            elif top_name == "relocation":
                report.relocation_top_count += 1
                if top.verified:
                    report.relocation_top_verifies += 1
            elif top_name == "periodic_repair":
                report.periodic_top_count += 1
                if top.verified:
                    report.periodic_top_verifies += 1
            elif top_name == "grid_transform":
                report.transform_top_count += 1
                if top.verified:
                    report.transform_top_verifies += 1

            if top.verified:
                report.top_lane_verifies += 1
            elif top.class_score > 0.5:
                report.false_positives.append((tid, top_name, top.class_score))

            # Top-2 contains verifier?
            verified_names = {lf.name for lf in record.lanes if lf.verified}
            top2_names = {record.lanes[i].name for i in range(min(2, len(record.lanes)))}
            if verified_names & top2_names:
                report.top2_contains_verifier += 1

    return report


def format_report(report: AuditReport) -> str:
    """Format the audit report as readable text."""
    lines = [
        f"=== Mechanism Evidence Audit ===",
        f"Tasks: {report.n_tasks}",
        f"Tasks with compilable lanes: {report.n_with_graphs}",
        f"Solved: {report.n_solved}",
        f"",
        f"--- Lane Selection Calibration ---",
        f"Top-ranked lane verifies: {report.top_lane_verifies}/{report.n_with_graphs}",
        f"Top-2 contains verifier:  {report.top2_contains_verifier}/{report.n_with_graphs}",
        f"",
        f"--- Per-Lane Top Ranking ---",
        f"Replication:    top={report.replication_top_count} verifies={report.replication_top_verifies}",
        f"Relocation:     top={report.relocation_top_count} verifies={report.relocation_top_verifies}",
        f"Periodic:       top={report.periodic_top_count} verifies={report.periodic_top_verifies}",
        f"Grid transform: top={report.transform_top_count} verifies={report.transform_top_verifies}",
    ]

    if report.false_positives:
        lines.append(f"")
        lines.append(f"--- False Positives (top-ranked, score>0.5, didn't verify) ---")
        for tid, lane, score in report.false_positives[:10]:
            lines.append(f"  {tid}: {lane} score={score:.2f}")

    return "\n".join(lines)
