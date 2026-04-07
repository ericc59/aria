"""Benchmark discipline — repeatable change evaluation and freeze criteria.

Records before/after metrics for any architecture change so we can
detect drift, false improvements, and determine what to freeze.

Part of the canonical architecture.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkSnapshot:
    """One measurement of the system on a task slice."""
    timestamp: str = ""
    label: str = ""            # e.g., "before-replication-gate-fix"
    n_tasks: int = 0
    n_solved: int = 0
    solved_ids: list[str] = field(default_factory=list)
    near_miss_count: int = 0   # tasks with best diff < 10% of pixels
    near_miss_ids: list[str] = field(default_factory=list)
    # Per-lane stats
    lane_top_counts: dict[str, int] = field(default_factory=dict)
    lane_verify_counts: dict[str, int] = field(default_factory=dict)
    lane_false_positives: dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkComparison:
    """Before/after comparison of two snapshots."""
    before: BenchmarkSnapshot
    after: BenchmarkSnapshot
    solves_gained: list[str] = field(default_factory=list)
    solves_lost: list[str] = field(default_factory=list)
    near_misses_gained: list[str] = field(default_factory=list)
    near_misses_lost: list[str] = field(default_factory=list)
    fp_change: dict[str, int] = field(default_factory=dict)  # lane -> delta


# ---------------------------------------------------------------------------
# Freeze criteria
# ---------------------------------------------------------------------------

@dataclass
class FreezeCriteria:
    """Explicit criteria for when a component should be frozen."""
    component: str                # "selector", "periodic_executor", etc.
    freeze: bool = False
    reason: str = ""


def evaluate_freeze(before: BenchmarkSnapshot, after: BenchmarkSnapshot) -> list[FreezeCriteria]:
    """Evaluate freeze criteria based on a before/after comparison."""
    criteria = []

    # Selector: freeze if false positives didn't drop and solves didn't increase
    total_fp_before = sum(before.lane_false_positives.values())
    total_fp_after = sum(after.lane_false_positives.values())
    solve_gain = after.n_solved - before.n_solved

    criteria.append(FreezeCriteria(
        component="selector",
        freeze=solve_gain <= 0 and total_fp_after >= total_fp_before,
        reason=(f"solves {'gained' if solve_gain > 0 else 'unchanged'} ({solve_gain:+d}), "
                f"FPs {'reduced' if total_fp_after < total_fp_before else 'unchanged'} "
                f"({total_fp_after - total_fp_before:+d})"),
    ))

    # Per-lane executor freeze: freeze if no new verifications
    for lane in ("replication", "relocation", "periodic_repair"):
        v_before = before.lane_verify_counts.get(lane, 0)
        v_after = after.lane_verify_counts.get(lane, 0)
        criteria.append(FreezeCriteria(
            component=f"{lane}_executor",
            freeze=v_after <= v_before,
            reason=f"verifications {'gained' if v_after > v_before else 'unchanged'} ({v_after - v_before:+d})",
        ))

    return criteria


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------


def capture_snapshot(
    task_ids: list[str],
    demos_fn,
    *,
    label: str = "",
) -> BenchmarkSnapshot:
    """Capture a benchmark snapshot on the given task slice."""
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.mechanism_audit import audit_task
    from aria.core.protocol import solve as core_solve

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    snap = BenchmarkSnapshot(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        label=label,
        n_tasks=len(task_ids),
    )

    for lane in ("replication", "relocation", "periodic_repair", "grid_transform"):
        snap.lane_top_counts[lane] = 0
        snap.lane_verify_counts[lane] = 0
        snap.lane_false_positives[lane] = 0

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=tid)
        if result.solved:
            snap.n_solved += 1
            snap.solved_ids.append(tid)

        record = audit_task(tid, demos)
        if record.lanes:
            top = record.lanes[0]
            if top.name in snap.lane_top_counts:
                snap.lane_top_counts[top.name] += 1
            if top.class_score > 0.5 and not top.verified:
                if top.name in snap.lane_false_positives:
                    snap.lane_false_positives[top.name] += 1

        for lf in record.lanes:
            if lf.verified and lf.name in snap.lane_verify_counts:
                snap.lane_verify_counts[lf.name] += 1

        # Near-miss: any lane with diff < 10% of pixels
        total_px = sum(d.output.size for d in demos)
        for lf in record.lanes:
            if lf.compile_attempted and 0 < lf.residual_diff <= total_px * 0.10:
                snap.near_miss_count += 1
                snap.near_miss_ids.append(tid)
                break

    return snap


def compare_snapshots(before: BenchmarkSnapshot, after: BenchmarkSnapshot) -> BenchmarkComparison:
    """Compare two snapshots."""
    before_solved = set(before.solved_ids)
    after_solved = set(after.solved_ids)
    before_nm = set(before.near_miss_ids)
    after_nm = set(after.near_miss_ids)

    fp_change = {}
    for lane in set(list(before.lane_false_positives.keys()) + list(after.lane_false_positives.keys())):
        fp_change[lane] = after.lane_false_positives.get(lane, 0) - before.lane_false_positives.get(lane, 0)

    return BenchmarkComparison(
        before=before,
        after=after,
        solves_gained=sorted(after_solved - before_solved),
        solves_lost=sorted(before_solved - after_solved),
        near_misses_gained=sorted(after_nm - before_nm),
        near_misses_lost=sorted(before_nm - after_nm),
        fp_change=fp_change,
    )


def format_comparison(comp: BenchmarkComparison) -> str:
    """Format a comparison as readable text."""
    lines = [
        f"=== Benchmark Comparison ===",
        f"Before: {comp.before.label} ({comp.before.timestamp})",
        f"After:  {comp.after.label} ({comp.after.timestamp})",
        f"",
        f"Solves: {comp.before.n_solved} -> {comp.after.n_solved} "
        f"(gained={comp.solves_gained or 'none'}, lost={comp.solves_lost or 'none'})",
        f"Near-misses: {comp.before.near_miss_count} -> {comp.after.near_miss_count}",
        f"",
        f"False positive changes:",
    ]
    for lane, delta in sorted(comp.fp_change.items()):
        lines.append(f"  {lane}: {delta:+d}")

    return "\n".join(lines)


def format_freeze(criteria: list[FreezeCriteria]) -> str:
    """Format freeze recommendations."""
    lines = ["=== Freeze Recommendations ==="]
    for c in criteria:
        status = "FREEZE" if c.freeze else "ITERATE"
        lines.append(f"  {c.component:30s} [{status}] {c.reason}")
    return "\n".join(lines)


def save_snapshot(snap: BenchmarkSnapshot, path: str | Path) -> None:
    """Save a snapshot to JSON."""
    from dataclasses import asdict
    Path(path).write_text(json.dumps(asdict(snap), indent=2, default=str))
