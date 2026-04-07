"""Tests for weak supervision labels."""

from __future__ import annotations

from aria.core.weak_labels import WeakLabel, label_task, label_batch, format_label_summary
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_label_task_returns_label():
    label = label_task("t1", _simple())
    assert isinstance(label, WeakLabel)
    assert label.task_id == "t1"
    assert label.top_class != ""
    assert label.confidence in ("high", "medium", "low")


def test_label_confidence_levels():
    label = label_task("t1", _simple())
    assert label.exec_confidence in ("high", "medium", "low", "none")


def test_label_batch():
    labels = label_batch(["t1", "t2"], lambda tid: _simple())
    assert len(labels) == 2


def test_format_summary():
    labels = label_batch(["t1"], lambda tid: _simple())
    text = format_label_summary(labels)
    assert "Weak Labels" in text


def test_serializable():
    from dataclasses import asdict
    import json
    label = label_task("t1", _simple())
    d = asdict(label)
    s = json.dumps(d, default=str)
    assert "t1" in s


def test_no_task_id():
    import inspect
    src = inspect.getsource(label_task)
    assert "1b59e163" not in src
