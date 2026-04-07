"""Regression guardrails — compact metrics and threshold rules.

Every major architecture change should produce a guardrail report.
Regressions are flagged when metrics cross explicit thresholds.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailMetrics:
    """Compact set of health metrics."""
    solved_count: int = 0
    near_miss_count: int = 0         # diff < 10%
    near_miss_median_diff: float = 0.0
    top_rank_verify_rate: float = 0.0  # fraction of top-ranked lanes that verify
    avg_compile_attempts: float = 0.0
    per_lane_fp: dict[str, int] = field(default_factory=dict)  # false positives per lane


@dataclass(frozen=True)
class GuardrailCheck:
    """One threshold check result."""
    metric: str
    value: float
    threshold: float
    passed: bool
    severity: str    # "error", "warning"
    message: str


def compute_guardrails(
    task_ids: list[str],
    demos_fn,
) -> GuardrailMetrics:
    """Compute guardrail metrics on a task slice."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.mechanism_audit import audit_task
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    metrics = GuardrailMetrics()
    top_verified = 0
    top_total = 0
    total_compiles = 0
    near_miss_diffs: list[float] = []

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=tid)
        if result.solved:
            metrics.solved_count += 1
        total_compiles += result.graphs_compiled

        record = audit_task(tid, demos)
        if record.lanes:
            top = record.lanes[0]
            top_total += 1
            if top.verified:
                top_verified += 1
            if top.class_score > 0.5 and not top.verified:
                lane = top.name
                metrics.per_lane_fp[lane] = metrics.per_lane_fp.get(lane, 0) + 1

        # Near-miss check
        total_px = sum(d.output.size for d in demos)
        for lf in record.lanes:
            if lf.compile_attempted and 0 < lf.residual_diff <= total_px * 0.10:
                metrics.near_miss_count += 1
                near_miss_diffs.append(lf.residual_diff / total_px)
                break

    metrics.top_rank_verify_rate = top_verified / max(top_total, 1)
    metrics.avg_compile_attempts = total_compiles / max(len(task_ids), 1)
    if near_miss_diffs:
        near_miss_diffs.sort()
        metrics.near_miss_median_diff = near_miss_diffs[len(near_miss_diffs) // 2]

    return metrics


def check_regression(
    before: GuardrailMetrics,
    after: GuardrailMetrics,
) -> list[GuardrailCheck]:
    """Check for regressions between two metric snapshots."""
    checks = []

    # Solved count must not decrease
    checks.append(GuardrailCheck(
        metric="solved_count",
        value=after.solved_count,
        threshold=before.solved_count,
        passed=after.solved_count >= before.solved_count,
        severity="error",
        message=(f"solved: {before.solved_count} -> {after.solved_count}"
                 + (" REGRESSION" if after.solved_count < before.solved_count else " ok")),
    ))

    # Near-miss count should not drop significantly
    checks.append(GuardrailCheck(
        metric="near_miss_count",
        value=after.near_miss_count,
        threshold=before.near_miss_count * 0.8,
        passed=after.near_miss_count >= before.near_miss_count * 0.8,
        severity="warning",
        message=f"near-misses: {before.near_miss_count} -> {after.near_miss_count}",
    ))

    # Top-rank verify rate should not decrease
    checks.append(GuardrailCheck(
        metric="top_rank_verify_rate",
        value=after.top_rank_verify_rate,
        threshold=before.top_rank_verify_rate,
        passed=after.top_rank_verify_rate >= before.top_rank_verify_rate - 0.01,
        severity="warning",
        message=f"top-rank verify: {before.top_rank_verify_rate:.3f} -> {after.top_rank_verify_rate:.3f}",
    ))

    # Total FPs should not increase significantly
    before_fp = sum(before.per_lane_fp.values())
    after_fp = sum(after.per_lane_fp.values())
    checks.append(GuardrailCheck(
        metric="total_false_positives",
        value=after_fp,
        threshold=before_fp * 1.2,
        passed=after_fp <= before_fp * 1.2,
        severity="warning",
        message=f"FPs: {before_fp} -> {after_fp}",
    ))

    return checks


def format_guardrails(metrics: GuardrailMetrics, checks: list[GuardrailCheck] | None = None) -> str:
    lines = ["=== Guardrail Report ==="]
    lines.append(f"  solved:              {metrics.solved_count}")
    lines.append(f"  near-misses (<10%):  {metrics.near_miss_count}")
    lines.append(f"  near-miss median:    {metrics.near_miss_median_diff:.3f}")
    lines.append(f"  top-rank verify:     {metrics.top_rank_verify_rate:.3f}")
    lines.append(f"  avg compiles/task:   {metrics.avg_compile_attempts:.1f}")
    lines.append(f"  FPs by lane:         {dict(sorted(metrics.per_lane_fp.items()))}")

    if checks:
        lines.append("")
        all_pass = all(c.passed for c in checks)
        lines.append(f"Regression checks: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
        for c in checks:
            status = "PASS" if c.passed else f"FAIL [{c.severity}]"
            lines.append(f"  [{status}] {c.message}")

    return "\n".join(lines)
