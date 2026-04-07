"""Lane-coverage audit — measure where each lane's executable coverage fails.

For each lane, separates:
  class-fit count → compile-attempt → compile-success → verify-success
and clusters non-verifying residuals into a small inspectable taxonomy.

Part of the canonical architecture. No new lanes, no parameter expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.core.mechanism_evidence import compute_evidence_and_rank


# ---------------------------------------------------------------------------
# Residual taxonomy — a small fixed set of failure categories
# ---------------------------------------------------------------------------

RESIDUAL_NEAR_PERFECT = "near_perfect"          # diff <= 5% of pixels
RESIDUAL_PARTIAL_MATCH = "partial_match"        # diff 5-25%
RESIDUAL_WRONG_PLACEMENT = "wrong_placement"    # >50% of diff is positional (shapes exist but wrong location)
RESIDUAL_WRONG_CONTENT = "wrong_content"        # objects at right positions but wrong color/pattern
RESIDUAL_MISSING_OBJECTS = "missing_objects"     # output has more objects than predicted
RESIDUAL_EXTRA_OBJECTS = "extra_objects"         # predicted has more objects than output
RESIDUAL_LARGE_MISMATCH = "large_mismatch"      # diff > 25% — wrong mechanism
RESIDUAL_NO_COMPILE = "no_compile"              # lane couldn't compile at all

ALL_RESIDUAL_CATEGORIES = (
    RESIDUAL_NEAR_PERFECT, RESIDUAL_PARTIAL_MATCH,
    RESIDUAL_WRONG_PLACEMENT, RESIDUAL_WRONG_CONTENT,
    RESIDUAL_MISSING_OBJECTS, RESIDUAL_EXTRA_OBJECTS,
    RESIDUAL_LARGE_MISMATCH, RESIDUAL_NO_COMPILE,
)


def classify_residual(
    diff_pixels: int,
    total_pixels: int,
    predicted_grids: list[Any] | None = None,
    target_grids: list[Any] | None = None,
) -> str:
    """Classify a non-verifying residual into a taxonomy category."""
    if diff_pixels < 0:
        return RESIDUAL_NO_COMPILE
    if total_pixels == 0:
        return RESIDUAL_NO_COMPILE

    frac = diff_pixels / total_pixels

    if frac <= 0.05:
        return RESIDUAL_NEAR_PERFECT
    if frac <= 0.25:
        # Try to distinguish placement vs content errors
        if predicted_grids and target_grids:
            placement_err, content_err = _classify_error_type(predicted_grids, target_grids)
            if placement_err > content_err:
                return RESIDUAL_WRONG_PLACEMENT
            if content_err > placement_err:
                return RESIDUAL_WRONG_CONTENT
        return RESIDUAL_PARTIAL_MATCH

    return RESIDUAL_LARGE_MISMATCH


def _classify_error_type(
    predicted_grids: list[Any],
    target_grids: list[Any],
) -> tuple[float, float]:
    """Estimate whether errors are primarily positional or content-based.

    Returns (placement_score, content_score) — both 0-1.
    """
    from aria.runtime.ops.selection import _find_objects
    from aria.decomposition import detect_bg

    placement_errors = 0
    content_errors = 0
    total_objects = 0

    for pred, target in zip(predicted_grids, target_grids):
        if pred is None:
            continue
        bg_p = detect_bg(pred)
        bg_t = detect_bg(target)
        pred_objs = [o for o in _find_objects(pred) if o.color != bg_p and o.size > 1]
        tgt_objs = [o for o in _find_objects(target) if o.color != bg_t and o.size > 1]

        total_objects += max(len(pred_objs), len(tgt_objs))

        # Count predicted objects that exist in target at different positions (placement)
        # vs predicted objects with no matching target at all (content)
        for po in pred_objs:
            found_at_pos = any(
                to.bbox == po.bbox and to.color == po.color and to.size == po.size
                for to in tgt_objs
            )
            found_anywhere = any(
                to.color == po.color and to.size == po.size
                for to in tgt_objs
            )
            if found_at_pos:
                pass  # correct
            elif found_anywhere:
                placement_errors += 1
            else:
                content_errors += 1

    total = max(total_objects, 1)
    return placement_errors / total, content_errors / total


# ---------------------------------------------------------------------------
# Per-lane audit record
# ---------------------------------------------------------------------------


@dataclass
class LaneCoverageRecord:
    """Coverage audit for one lane on one task."""
    task_id: str
    lane: str
    class_fit: float          # class_score from evidence
    exec_hint: float          # executable sufficiency hint
    compile_attempted: bool = False
    compile_succeeded: bool = False   # program was built (may not verify)
    verified: bool = False
    residual_diff: int = -1
    total_pixels: int = 0
    residual_category: str = RESIDUAL_NO_COMPILE
    anti_evidence: str = ""


@dataclass
class LaneCoverageReport:
    """Aggregate coverage report for one lane across a task slice."""
    lane: str
    class_fit_count: int = 0      # tasks with class_score > 0
    compile_attempt_count: int = 0
    compile_success_count: int = 0  # compiled (program built, even if not verified)
    verify_success_count: int = 0
    residual_distribution: dict[str, int] = field(default_factory=dict)
    near_misses: list[tuple[str, int, str]] = field(default_factory=list)  # (task_id, diff, category)
    top_false_positives: list[tuple[str, float]] = field(default_factory=list)  # (task_id, class_score)
    records: list[LaneCoverageRecord] = field(default_factory=list)


@dataclass
class FullCoverageReport:
    """Coverage audit across all lanes."""
    n_tasks: int = 0
    lane_reports: dict[str, LaneCoverageReport] = field(default_factory=dict)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------


def run_lane_coverage(
    task_ids: list[str],
    demos_fn,  # callable: task_id -> demos
    *,
    max_near_misses: int = 10,
    max_false_positives: int = 10,
) -> FullCoverageReport:
    """Run per-lane coverage audit on a task slice."""
    from aria.core.mechanism_audit import audit_task

    report = FullCoverageReport(n_tasks=len(task_ids))

    for lane_name in ("replication", "relocation", "periodic_repair"):
        report.lane_reports[lane_name] = LaneCoverageReport(lane=lane_name)
        for cat in ALL_RESIDUAL_CATEGORIES:
            report.lane_reports[lane_name].residual_distribution[cat] = 0

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        record = audit_task(tid, demos)

        for lf in record.lanes:
            if lf.name not in report.lane_reports:
                continue
            lr = report.lane_reports[lf.name]

            if lf.class_score > 0:
                lr.class_fit_count += 1

            total_pixels = sum(d.output.size for d in demos)

            cr = LaneCoverageRecord(
                task_id=tid,
                lane=lf.name,
                class_fit=lf.class_score,
                exec_hint=0.0,
                compile_attempted=lf.compile_attempted,
                verified=lf.verified,
                residual_diff=lf.residual_diff,
                total_pixels=total_pixels,
            )

            if lf.compile_attempted:
                lr.compile_attempt_count += 1
                if lf.residual_diff >= 0:
                    lr.compile_success_count += 1
                    cr.compile_succeeded = True
                    cr.residual_category = classify_residual(
                        lf.residual_diff, total_pixels,
                    )
                else:
                    cr.residual_category = RESIDUAL_NO_COMPILE

            if lf.verified:
                lr.verify_success_count += 1
                cr.residual_category = RESIDUAL_NEAR_PERFECT

            lr.residual_distribution[cr.residual_category] = (
                lr.residual_distribution.get(cr.residual_category, 0) + 1
            )
            lr.records.append(cr)

    # Post-process: near-misses and false positives
    for lane_name, lr in report.lane_reports.items():
        # Near misses: lowest non-zero residual diff
        compiled = [(r.task_id, r.residual_diff, r.residual_category)
                    for r in lr.records
                    if r.compile_succeeded and not r.verified and r.residual_diff >= 0]
        compiled.sort(key=lambda x: x[1])
        lr.near_misses = compiled[:max_near_misses]

        # False positives: highest class_fit that didn't verify
        non_verified = [(r.task_id, r.class_fit)
                       for r in lr.records
                       if r.class_fit > 0.5 and not r.verified]
        non_verified.sort(key=lambda x: -x[1])
        lr.top_false_positives = non_verified[:max_false_positives]

    # Recommendation
    best_lane = max(report.lane_reports.values(),
                    key=lambda lr: lr.compile_success_count - lr.verify_success_count)
    report.recommendation = (
        f"Lane '{best_lane.lane}' has the most compilable-but-unverified tasks "
        f"({best_lane.compile_success_count} compiled, {best_lane.verify_success_count} verified). "
        f"Improving this executor would have the highest coverage impact."
    )

    return report


def format_lane_coverage(report: FullCoverageReport) -> str:
    """Format a lane coverage report as readable text."""
    lines = [
        f"=== Lane Coverage Audit ({report.n_tasks} tasks) ===",
        "",
    ]

    for lane_name in ("replication", "relocation", "periodic_repair"):
        lr = report.lane_reports.get(lane_name)
        if lr is None:
            continue

        lines.append(f"--- {lane_name} ---")
        lines.append(f"  Funnel: class-fit={lr.class_fit_count} → compiled={lr.compile_attempt_count}"
                     f" → built={lr.compile_success_count} → verified={lr.verify_success_count}")
        lines.append(f"  Residual distribution:")
        for cat, count in sorted(lr.residual_distribution.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"    {cat}: {count}")

        if lr.near_misses:
            lines.append(f"  Top near-misses:")
            for tid, diff, cat in lr.near_misses[:5]:
                lines.append(f"    {tid}: diff={diff} [{cat}]")

        lines.append("")

    lines.append(f"Recommendation: {report.recommendation}")
    return "\n".join(lines)
