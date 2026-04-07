"""Tests for lane composition audit."""

from __future__ import annotations

from aria.core.composition_audit import (
    LABEL_SINGLE_LANE, LABEL_TWO_STAGE, LABEL_CROSS_LANE, LABEL_NO_MATCH,
    CompositionRecord, CompositionReport,
    audit_composition, format_composition_audit,
)
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[9, 8]])),)


def test_audit_returns_report():
    report = audit_composition(["t1"], lambda tid: _simple())
    assert isinstance(report, CompositionReport)


def test_labels_are_strings():
    for l in [LABEL_SINGLE_LANE, LABEL_TWO_STAGE, LABEL_CROSS_LANE, LABEL_NO_MATCH]:
        assert isinstance(l, str)


def test_format_produces_string():
    report = audit_composition(["t1"], lambda tid: _simple())
    assert isinstance(format_composition_audit(report), str)
    assert "Composition Audit" in format_composition_audit(report)


def test_no_task_id():
    import inspect
    src = inspect.getsource(audit_composition)
    assert "1b59e163" not in src
