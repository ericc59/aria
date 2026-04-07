"""Tests for consensus-controlled compositional search."""

from __future__ import annotations

import numpy as np
import pytest

from aria.consensus_search import (
    CompositionBranch,
    ScopeResult,
    check_correspondence_consistency,
    check_scope_consistency,
    consensus_compose_search,
    probe_scope,
)
from aria.consensus_trace import ConsensusTrace
from aria.core.grid_perception import perceive_grid
from aria.types import DemoPair, Grid


def _grid(rows: list[list[int]]) -> Grid:
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# Scope consistency checks
# ---------------------------------------------------------------------------


class TestScopeConsistency:
    def test_objects_all_present(self):
        results = (
            ScopeResult(found=True, scope_kind="objects", object_count=3),
            ScopeResult(found=True, scope_kind="objects", object_count=2),
        )
        passed, score, detail = check_scope_consistency(results)
        assert passed
        assert score == 1.0

    def test_objects_one_missing(self):
        results = (
            ScopeResult(found=True, scope_kind="objects", object_count=3),
            ScopeResult(found=False, scope_kind="objects", object_count=0),
        )
        passed, score, detail = check_scope_consistency(results)
        assert not passed
        assert score == 0.0

    def test_partition_consistent(self):
        results = (
            ScopeResult(found=True, scope_kind="partition", n_entities=4),
            ScopeResult(found=True, scope_kind="partition", n_entities=4),
        )
        passed, score, detail = check_scope_consistency(results)
        assert passed

    def test_partition_inconsistent(self):
        results = (
            ScopeResult(found=True, scope_kind="partition", n_entities=4),
            ScopeResult(found=True, scope_kind="partition", n_entities=9),
        )
        passed, score, detail = check_scope_consistency(results)
        assert not passed

    def test_enclosed_present(self):
        results = (
            ScopeResult(found=True, scope_kind="enclosed", region_count=2),
            ScopeResult(found=True, scope_kind="enclosed", region_count=1),
        )
        passed, score, detail = check_scope_consistency(results)
        assert passed

    def test_enclosed_missing(self):
        results = (
            ScopeResult(found=True, scope_kind="enclosed", region_count=2),
            ScopeResult(found=False, scope_kind="enclosed", region_count=0),
        )
        passed, score, detail = check_scope_consistency(results)
        assert not passed

    def test_empty(self):
        passed, score, detail = check_scope_consistency(())
        assert passed


# ---------------------------------------------------------------------------
# Scope probing
# ---------------------------------------------------------------------------


class TestProbeScope:
    def test_probe_objects(self):
        g = _grid([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 0],
        ])
        p = perceive_grid(g)
        results = probe_scope((p,), "objects")
        assert len(results) == 1
        assert results[0].found
        assert results[0].object_count >= 1

    def test_probe_objects_empty(self):
        g = _grid([[0, 0], [0, 0]])
        p = perceive_grid(g)
        results = probe_scope((p,), "objects")
        assert not results[0].found

    def test_probe_partition(self):
        g = _grid([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        p = perceive_grid(g)
        results = probe_scope((p,), "partition")
        assert results[0].found
        assert results[0].n_entities > 0

    def test_probe_frame(self):
        g = _grid([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ])
        p = perceive_grid(g)
        results = probe_scope((p,), "frame")
        assert results[0].found


# ---------------------------------------------------------------------------
# Correspondence consistency
# ---------------------------------------------------------------------------


class TestCorrespondenceConsistency:
    def test_both_have_objects(self):
        g1 = _grid([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ])
        g2 = _grid([
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        p1 = perceive_grid(g1)
        p2 = perceive_grid(g2)
        passed, score, detail = check_correspondence_consistency(
            (p1, p2), "object", "object",
        )
        assert passed

    def test_one_has_no_objects(self):
        g1 = _grid([[0, 1], [0, 0]])
        g2 = _grid([[0, 0], [0, 0]])
        p1 = perceive_grid(g1)
        p2 = perceive_grid(g2)
        passed, score, detail = check_correspondence_consistency(
            (p1, p2), "object", "object",
        )
        assert not passed


# ---------------------------------------------------------------------------
# Compositional search integration
# ---------------------------------------------------------------------------


class TestConsensusComposeSearch:
    def test_empty_demos(self):
        results = consensus_compose_search(())
        assert results == []

    def test_diff_dims_skipped(self):
        demos = (
            DemoPair(input=_grid([[1, 2]]), output=_grid([[1], [2]])),
        )
        results = consensus_compose_search(demos)
        assert results == []

    def test_same_dims_runs(self):
        """Smoke test: same-dims task runs without error."""
        g_in = _grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        g_out = _grid([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        demos = (DemoPair(input=g_in, output=g_out),)
        # Should run without error; may or may not find a program
        results = consensus_compose_search(demos)
        # No assertion on results — just checking no crash

    def test_with_trace(self):
        """Consensus trace records branches during composition search."""
        g_in = _grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        g_out = _grid([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        demos = (DemoPair(input=g_in, output=g_out),)
        trace = ConsensusTrace()
        consensus_compose_search(demos, trace=trace)
        # Should have some trace entries from scope probing
        assert trace.branches_created > 0


# ---------------------------------------------------------------------------
# Scope consistency across real-looking demos
# ---------------------------------------------------------------------------


class TestCrossDemoConsistency:
    def test_consistent_object_scopes(self):
        """Two demos with similar object structure should not be pruned."""
        g1 = _grid([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        g2 = _grid([
            [0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 3, 3, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        p1 = perceive_grid(g1)
        p2 = perceive_grid(g2)
        results = probe_scope((p1, p2), "objects")
        passed, score, _ = check_scope_consistency(results)
        assert passed

    def test_inconsistent_partition(self):
        """Demos with different partition structures should be pruned."""
        # 2x2 partition
        g1 = _grid([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        # No clear partition
        g2 = _grid([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        p1 = perceive_grid(g1)
        p2 = perceive_grid(g2)
        results = probe_scope((p1, p2), "partition")
        passed, score, _ = check_scope_consistency(results)
        # p2 likely has no partition, so this should fail
        if not results[1].found:
            assert not passed
