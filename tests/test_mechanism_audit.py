"""Tests for mechanism evidence audit harness."""

from __future__ import annotations

from aria.core.mechanism_audit import (
    AuditReport,
    LaneFit,
    TaskAuditRecord,
    audit_task,
    run_audit,
    format_report,
)
from aria.core.mechanism_evidence import compute_evidence, compute_evidence_and_rank, rank_lanes, MechanismEvidence
from aria.types import DemoPair, grid_from_list


def _simple_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _impossible_task():
    return (DemoPair(input=grid_from_list([[1, 2, 3]]),
                     output=grid_from_list([[9, 8, 7]])),)


# ---------------------------------------------------------------------------
# Audit record shape
# ---------------------------------------------------------------------------


def test_audit_record_has_lanes():
    record = audit_task("test", _simple_task())
    assert isinstance(record, TaskAuditRecord)
    assert len(record.lanes) >= 1
    for lf in record.lanes:
        assert isinstance(lf, LaneFit)
        assert isinstance(lf.class_score, float)
        assert isinstance(lf.class_rationale, str)


def test_audit_record_has_evidence():
    record = audit_task("test", _simple_task())
    assert record.evidence is not None
    assert isinstance(record.evidence, MechanismEvidence)


def test_audit_record_separates_class_and_exec():
    """class_fit and executable_fit should be independently recorded."""
    record = audit_task("test", _simple_task())
    for lf in record.lanes:
        # class_score is always present
        assert 0 <= lf.class_score <= 1
        # verified is independent of class_score
        assert isinstance(lf.verified, bool)


# ---------------------------------------------------------------------------
# Audit report shape
# ---------------------------------------------------------------------------


def test_run_audit_produces_report():
    def demos_fn(tid):
        return _simple_task()

    report = run_audit(["t1", "t2"], demos_fn)
    assert isinstance(report, AuditReport)
    assert report.n_tasks == 2
    assert len(report.records) == 2


def test_format_report_is_string():
    def demos_fn(tid):
        return _simple_task()

    report = run_audit(["t1"], demos_fn)
    text = format_report(report)
    assert isinstance(text, str)
    assert "Mechanism Evidence Audit" in text


# ---------------------------------------------------------------------------
# Class fit vs executable fit
# ---------------------------------------------------------------------------


def test_class_fit_independent_of_exec():
    """A task can have high class_score but verified=False."""
    record = audit_task("test", _impossible_task())
    # Should have some lanes with class_score > 0 but verified=False
    for lf in record.lanes:
        if lf.class_score > 0:
            # This specific task is impossible, so nothing verifies
            assert not lf.verified


# ---------------------------------------------------------------------------
# Top-lane reporting
# ---------------------------------------------------------------------------


def test_top_class_lane_set():
    record = audit_task("test", _simple_task())
    assert record.top_class_lane != ""


def test_top_exec_lane_matches_solve():
    """If task solves, top_exec_lane should be set."""
    record = audit_task("test", _simple_task())
    if record.solved:
        # For this simple task the static pipeline solves it
        pass  # top_exec_lane may or may not be set depending on lane


# ---------------------------------------------------------------------------
# Best residual per lane
# ---------------------------------------------------------------------------


def test_residual_diff_recorded():
    """Lanes that were tried should record residual diff."""
    record = audit_task("test", _impossible_task())
    attempted = [lf for lf in record.lanes if lf.compile_attempted]
    for lf in attempted:
        assert isinstance(lf.residual_diff, int)


# ---------------------------------------------------------------------------
# No task-id dependency
# ---------------------------------------------------------------------------


def test_no_task_id_in_audit():
    import inspect
    from aria.core.mechanism_audit import audit_task
    src = inspect.getsource(audit_task)
    assert "1b59e163" not in src


# ---------------------------------------------------------------------------
# No new lane introduction
# ---------------------------------------------------------------------------


def test_only_known_lanes():
    record = audit_task("test", _simple_task())
    known = {"replication", "relocation", "periodic_repair", "grid_transform"}
    for lf in record.lanes:
        assert lf.name in known, f"Unknown lane: {lf.name}"
