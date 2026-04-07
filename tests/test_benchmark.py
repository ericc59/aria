"""Tests for benchmark discipline harness."""

from __future__ import annotations

from aria.core.benchmark import (
    BenchmarkComparison, BenchmarkSnapshot, FreezeCriteria,
    capture_snapshot, compare_snapshots, evaluate_freeze,
    format_comparison, format_freeze, save_snapshot,
)
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_capture_snapshot_shape():
    snap = capture_snapshot(["t1", "t2"], lambda tid: _simple(), label="test")
    assert isinstance(snap, BenchmarkSnapshot)
    assert snap.n_tasks == 2
    assert snap.label == "test"
    assert isinstance(snap.lane_top_counts, dict)
    assert isinstance(snap.lane_false_positives, dict)


def test_compare_snapshots():
    before = BenchmarkSnapshot(n_solved=2, solved_ids=["a", "b"], near_miss_count=5, near_miss_ids=["c"])
    after = BenchmarkSnapshot(n_solved=3, solved_ids=["a", "b", "d"], near_miss_count=4, near_miss_ids=["c", "e"])
    comp = compare_snapshots(before, after)
    assert comp.solves_gained == ["d"]
    assert comp.solves_lost == []


def test_format_comparison():
    before = BenchmarkSnapshot(label="before", n_solved=1, lane_false_positives={"replication": 5})
    after = BenchmarkSnapshot(label="after", n_solved=2, lane_false_positives={"replication": 3})
    comp = compare_snapshots(before, after)
    text = format_comparison(comp)
    assert "Benchmark Comparison" in text


def test_evaluate_freeze_no_improvement():
    before = BenchmarkSnapshot(n_solved=2, lane_false_positives={"replication": 10},
                               lane_verify_counts={"replication": 1, "relocation": 0, "periodic_repair": 1})
    after = BenchmarkSnapshot(n_solved=2, lane_false_positives={"replication": 10},
                              lane_verify_counts={"replication": 1, "relocation": 0, "periodic_repair": 1})
    criteria = evaluate_freeze(before, after)
    # Selector should freeze (no change)
    selector = next(c for c in criteria if c.component == "selector")
    assert selector.freeze


def test_evaluate_freeze_improvement():
    before = BenchmarkSnapshot(n_solved=2, lane_false_positives={"replication": 10},
                               lane_verify_counts={"replication": 1, "relocation": 0, "periodic_repair": 1})
    after = BenchmarkSnapshot(n_solved=3, lane_false_positives={"replication": 5},
                              lane_verify_counts={"replication": 2, "relocation": 0, "periodic_repair": 1})
    criteria = evaluate_freeze(before, after)
    selector = next(c for c in criteria if c.component == "selector")
    assert not selector.freeze  # improved
    repl = next(c for c in criteria if c.component == "replication_executor")
    assert not repl.freeze  # gained verification


def test_format_freeze():
    criteria = [FreezeCriteria("selector", True, "no improvement")]
    text = format_freeze(criteria)
    assert "FREEZE" in text


def test_no_task_id():
    import inspect
    src = inspect.getsource(capture_snapshot)
    assert "1b59e163" not in src
