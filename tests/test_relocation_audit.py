"""Tests for relocation-specific failure audit."""

from __future__ import annotations

from aria.core.relocation_audit import (
    LABEL_NOT_RELOCATION, LABEL_VERIFIED, LABEL_WRONG_PAIRING,
    RelocationAuditRecord, RelocationAuditReport,
    audit_relocation, format_relocation_audit,
)
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_audit_returns_report():
    report = audit_relocation(["t1"], lambda tid: _simple())
    assert isinstance(report, RelocationAuditReport)


def test_labels_are_strings():
    for l in [LABEL_NOT_RELOCATION, LABEL_VERIFIED, LABEL_WRONG_PAIRING]:
        assert isinstance(l, str)


def test_format_produces_string():
    report = audit_relocation(["t1"], lambda tid: _simple())
    assert isinstance(format_relocation_audit(report), str)


def test_no_task_id():
    import inspect
    src = inspect.getsource(audit_relocation)
    assert "1b59e163" not in src
