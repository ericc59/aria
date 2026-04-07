"""Tests for replication-specific failure audit."""

from __future__ import annotations

from aria.core.replication_audit import (
    LABEL_NOT_REPLICATION, LABEL_VERIFIED, LABEL_WRONG_ANCHOR_KEY,
    LABEL_WRONG_EXEMPLAR, ReplicationAuditRecord, ReplicationAuditReport,
    audit_replication, format_replication_audit,
)
from aria.types import DemoPair, grid_from_list


def _simple_task():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_audit_returns_report():
    report = audit_replication(["t1"], lambda tid: _simple_task())
    assert isinstance(report, ReplicationAuditReport)


def test_audit_labels_are_strings():
    for label in [LABEL_NOT_REPLICATION, LABEL_VERIFIED, LABEL_WRONG_ANCHOR_KEY, LABEL_WRONG_EXEMPLAR]:
        assert isinstance(label, str)


def test_format_produces_string():
    report = audit_replication(["t1"], lambda tid: _simple_task())
    text = format_replication_audit(report)
    assert isinstance(text, str)


def test_no_task_id_in_audit():
    import inspect
    from aria.core.replication_audit import audit_replication
    src = inspect.getsource(audit_replication)
    assert "1b59e163" not in src
