"""Tests for regression guardrails."""

from __future__ import annotations

from aria.core.guardrails import (
    GuardrailCheck, GuardrailMetrics,
    check_regression, compute_guardrails, format_guardrails,
)
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_compute_returns_metrics():
    m = compute_guardrails(["t1"], lambda tid: _simple())
    assert isinstance(m, GuardrailMetrics)
    assert m.solved_count >= 0


def test_check_no_regression():
    before = GuardrailMetrics(solved_count=3, near_miss_count=10, top_rank_verify_rate=0.02)
    after = GuardrailMetrics(solved_count=3, near_miss_count=10, top_rank_verify_rate=0.02)
    checks = check_regression(before, after)
    assert all(c.passed for c in checks)


def test_check_detects_solve_regression():
    before = GuardrailMetrics(solved_count=3)
    after = GuardrailMetrics(solved_count=2)
    checks = check_regression(before, after)
    solve_check = next(c for c in checks if c.metric == "solved_count")
    assert not solve_check.passed
    assert solve_check.severity == "error"


def test_check_detects_fp_increase():
    before = GuardrailMetrics(per_lane_fp={"replication": 10})
    after = GuardrailMetrics(per_lane_fp={"replication": 15})
    checks = check_regression(before, after)
    fp_check = next(c for c in checks if c.metric == "total_false_positives")
    assert not fp_check.passed


def test_format_guardrails():
    m = GuardrailMetrics(solved_count=2, near_miss_count=5)
    text = format_guardrails(m)
    assert "Guardrail Report" in text


def test_format_with_checks():
    before = GuardrailMetrics(solved_count=2)
    after = GuardrailMetrics(solved_count=2)
    checks = check_regression(before, after)
    text = format_guardrails(after, checks)
    assert "ALL PASS" in text


def test_no_task_id():
    import inspect
    src = inspect.getsource(compute_guardrails)
    assert "1b59e163" not in src
