"""Tests for benchmark slice curriculum."""

from __future__ import annotations

from aria.core.slices import SLICES, SliceResult, build_slices, format_slices
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_slices_defined():
    assert len(SLICES) >= 5
    for s in SLICES:
        assert s.name
        assert s.rationale
        assert callable(s.entry_rule)


def test_build_slices_returns_dict():
    result = build_slices(["t1"], lambda tid: _simple())
    assert isinstance(result, dict)
    for name, sr in result.items():
        assert isinstance(sr, SliceResult)
        assert isinstance(sr.task_ids, list)


def test_format_slices():
    result = build_slices(["t1"], lambda tid: _simple())
    text = format_slices(result)
    assert "Benchmark Slices" in text


def test_no_task_id_in_rules():
    import inspect
    for s in SLICES:
        src = inspect.getsource(s.entry_rule)
        assert "1b59e163" not in src
