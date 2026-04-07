"""Tests for lane-coverage audit."""

from __future__ import annotations

from aria.core.lane_coverage import (
    ALL_RESIDUAL_CATEGORIES,
    RESIDUAL_LARGE_MISMATCH,
    RESIDUAL_NEAR_PERFECT,
    RESIDUAL_NO_COMPILE,
    RESIDUAL_PARTIAL_MATCH,
    FullCoverageReport,
    LaneCoverageRecord,
    LaneCoverageReport,
    classify_residual,
    format_lane_coverage,
    run_lane_coverage,
)
from aria.types import DemoPair, grid_from_list


def _simple_task():
    return (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )


# ---------------------------------------------------------------------------
# Residual taxonomy
# ---------------------------------------------------------------------------


def test_classify_residual_near_perfect():
    assert classify_residual(2, 100) == RESIDUAL_NEAR_PERFECT


def test_classify_residual_partial():
    assert classify_residual(15, 100) == RESIDUAL_PARTIAL_MATCH


def test_classify_residual_large():
    assert classify_residual(50, 100) == RESIDUAL_LARGE_MISMATCH


def test_classify_residual_no_compile():
    assert classify_residual(-1, 100) == RESIDUAL_NO_COMPILE


def test_residual_categories_are_strings():
    for cat in ALL_RESIDUAL_CATEGORIES:
        assert isinstance(cat, str)


# ---------------------------------------------------------------------------
# Audit output structure
# ---------------------------------------------------------------------------


def test_run_audit_returns_report():
    def demos_fn(tid):
        return _simple_task()
    report = run_lane_coverage(["t1", "t2"], demos_fn)
    assert isinstance(report, FullCoverageReport)
    assert report.n_tasks == 2
    assert "replication" in report.lane_reports
    assert "relocation" in report.lane_reports
    assert "periodic_repair" in report.lane_reports


def test_lane_report_has_funnel():
    def demos_fn(tid):
        return _simple_task()
    report = run_lane_coverage(["t1"], demos_fn)
    for lr in report.lane_reports.values():
        assert isinstance(lr, LaneCoverageReport)
        assert isinstance(lr.class_fit_count, int)
        assert isinstance(lr.compile_attempt_count, int)
        assert isinstance(lr.compile_success_count, int)
        assert isinstance(lr.verify_success_count, int)


def test_lane_report_has_residual_distribution():
    def demos_fn(tid):
        return _simple_task()
    report = run_lane_coverage(["t1"], demos_fn)
    for lr in report.lane_reports.values():
        assert isinstance(lr.residual_distribution, dict)
        total = sum(lr.residual_distribution.values())
        # Each task should be counted in exactly one category per lane
        assert total <= 1  # may be 0 if not compiled


def test_report_has_recommendation():
    def demos_fn(tid):
        return _simple_task()
    report = run_lane_coverage(["t1"], demos_fn)
    assert report.recommendation != ""


def test_format_produces_string():
    def demos_fn(tid):
        return _simple_task()
    report = run_lane_coverage(["t1"], demos_fn)
    text = format_lane_coverage(report)
    assert isinstance(text, str)
    assert "Lane Coverage Audit" in text


# ---------------------------------------------------------------------------
# No task-id logic
# ---------------------------------------------------------------------------


def test_no_task_id_in_coverage():
    import inspect
    from aria.core.lane_coverage import run_lane_coverage, classify_residual
    src = inspect.getsource(run_lane_coverage) + inspect.getsource(classify_residual)
    assert "1b59e163" not in src
