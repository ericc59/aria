"""Tests for stage-1 gap audit."""

from __future__ import annotations

from aria.core.stage1_audit import (
    LABEL_CROP_SUBGRID, LABEL_ISOLATE_REGION, LABEL_NO_CLEAR_GAP,
    Stage1Report, audit_stage1_gaps, format_stage1_audit,
)
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[9, 8]])),)


def test_audit_returns_report():
    report = audit_stage1_gaps(["t1"], lambda tid: _simple())
    assert isinstance(report, Stage1Report)


def test_format_produces_string():
    report = audit_stage1_gaps(["t1"], lambda tid: _simple())
    assert "Stage-1 Gap Audit" in format_stage1_audit(report)


def test_labels_are_strings():
    for l in [LABEL_CROP_SUBGRID, LABEL_ISOLATE_REGION, LABEL_NO_CLEAR_GAP]:
        assert isinstance(l, str)


def test_no_task_id():
    import inspect
    src = inspect.getsource(audit_stage1_gaps)
    assert "1b59e163" not in src
