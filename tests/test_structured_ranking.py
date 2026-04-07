"""Tests for structured lane ranking experiment assessment."""

from __future__ import annotations

from aria.core.mechanism_evidence import compute_evidence_and_rank, MechanismEvidence
from aria.types import DemoPair, grid_from_list


def test_evidence_fields_are_structured():
    """Evidence fields should be named, typed, and inspectable — no black box."""
    demos = (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)
    ev, ranking = compute_evidence_and_rank(demos)
    # All fields are explicit attributes, not opaque arrays
    assert hasattr(ev, 'n_input_shapes')
    assert hasattr(ev, 'has_anchored_exemplars')
    assert hasattr(ev, 'replication_score')
    assert isinstance(ev.n_input_shapes, int)


def test_ranking_is_inspectable():
    """Lane ranking should expose rationale, not just scores."""
    demos = (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)
    ev, ranking = compute_evidence_and_rank(demos)
    for c in ranking.lanes:
        assert hasattr(c, 'rationale')
        assert hasattr(c, 'anti_evidence')
        assert hasattr(c, 'exec_hint')
        assert isinstance(c.rationale, str)


def test_current_ranking_handles_solved_tasks():
    """Solved tasks should have the correct lane ranked with gate_pass=True."""
    from aria.datasets import get_dataset, load_arc_task
    ds = get_dataset('v2-train')

    # 00d62c1b is solved by periodic repair
    task = load_arc_task(ds, '00d62c1b')
    ev, ranking = compute_evidence_and_rank(task.train)
    periodic = next(c for c in ranking.lanes if c.name == 'periodic_repair')
    assert periodic.gate_pass

    # 1b59e163 is solved by replication
    task = load_arc_task(ds, '1b59e163')
    ev, ranking = compute_evidence_and_rank(task.train)
    repl = next(c for c in ranking.lanes if c.name == 'replication')
    assert repl.gate_pass
    assert repl.final_score > 0
