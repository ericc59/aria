"""Tests for mechanism evidence and lane selection."""

from __future__ import annotations

import numpy as np

from aria.core.mechanism_evidence import (
    MechanismEvidence,
    LaneCandidate,
    LaneRanking,
    compute_evidence,
    compute_evidence_and_rank,
    rank_lanes,
)
from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _replication_task():
    """Task with replication cues: anchored exemplars + targets + output grows."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0, 5, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 5, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
        ),
    )


def _relocation_task():
    """Task with relocation cues: shapes shift, 1:1 count."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
            ]),
        ),
    )


def _periodic_task():
    """Task with periodic cues: framed region, same dims."""
    return (
        DemoPair(
            input=grid_from_list([
                [4, 4, 4, 4, 4],
                [4, 1, 2, 1, 4],
                [4, 3, 0, 3, 4],
                [4, 1, 2, 1, 4],
                [4, 4, 4, 4, 4],
            ]),
            output=grid_from_list([
                [4, 4, 4, 4, 4],
                [4, 1, 2, 1, 4],
                [4, 3, 2, 3, 4],
                [4, 1, 2, 1, 4],
                [4, 4, 4, 4, 4],
            ]),
        ),
    )


def test_evidence_replication_cues():
    ev = compute_evidence(_replication_task())
    assert ev.has_anchored_exemplars
    assert ev.n_anchored_exemplars >= 1
    assert ev.replication_target_count >= 1
    assert ev.replication_score > 0


def test_evidence_relocation_cues():
    ev = compute_evidence(_relocation_task())
    assert ev.shapes_shift_position
    assert ev.n_input_shapes > 0
    assert ev.relocation_score > 0


def test_evidence_periodic_cues():
    ev = compute_evidence(_periodic_task())
    assert ev.has_framed_region
    assert ev.same_dims
    assert ev.periodic_repair_score > 0


def test_evidence_no_task_id():
    """Evidence must not depend on task IDs."""
    import inspect
    from aria.core.mechanism_evidence import compute_evidence
    src = inspect.getsource(compute_evidence)
    assert "1b59e163" not in src
    assert "task_id" not in src


# ---------------------------------------------------------------------------
# Lane ranking
# ---------------------------------------------------------------------------


def test_ranking_replication_favored():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    repl = next(c for c in ranking.lanes if c.name == "replication")
    assert repl.gate_pass
    assert repl.class_score > 0
    # With output clone matches, replication should rank high
    assert repl.final_score > 0


def test_ranking_relocation_favored():
    ev, ranking = compute_evidence_and_rank(_relocation_task())
    reloc = next(c for c in ranking.lanes if c.name == "relocation")
    assert reloc.gate_pass
    assert reloc.final_score > 0


def test_ranking_periodic_favored():
    ev, ranking = compute_evidence_and_rank(_periodic_task())
    periodic = next(c for c in ranking.lanes if c.name == "periodic_repair")
    assert periodic.gate_pass
    assert periodic.final_score > 0


def test_ranking_has_rationale():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    for c in ranking.lanes:
        assert isinstance(c.rationale, str)
        assert isinstance(c.final_score, float)


def test_ranking_description():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    assert "hypothesis=" in ranking.description


# ---------------------------------------------------------------------------
# Ambiguous cases
# ---------------------------------------------------------------------------


def test_evidence_ambiguous():
    """A task with mixed signals should produce non-zero scores for multiple lanes."""
    # Both replication and relocation cues present
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 5, 0, 0, 5],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 5, 0, 0, 5],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    ev, ranking = compute_evidence_and_rank(demos)
    scores = {c.name: c.score for c in ranking.lanes}
    # Both should have non-zero scores
    assert scores["replication"] > 0
    assert scores["relocation"] > 0


# ---------------------------------------------------------------------------
# Compile dispatch uses evidence
# ---------------------------------------------------------------------------


def test_compile_dispatch_uses_evidence_for_1b59e163():
    """1b59e163 should be solved via replication, ranked by evidence."""
    from aria.datasets import get_dataset, load_arc_task
    ds = get_dataset('v2-train')
    task = load_arc_task(ds, '1b59e163')
    demos = task.train

    ev, ranking = compute_evidence_and_rank(demos)
    assert ranking.lanes[0].name == "replication"

    result = solve_arc_task(demos, task_id='1b59e163', use_editor_search=True)
    assert result.solved


# ---------------------------------------------------------------------------
# Trace includes evidence
# ---------------------------------------------------------------------------


def test_trace_includes_mechanism_evidence():
    from aria.core.trace_solve import solve_with_trace
    from aria.datasets import get_dataset, load_arc_task
    # Use a real task that passes stage-1 size but isn't directly solvable,
    # so it reaches the static phase where mechanism evidence runs.
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "0520fde7")
    trace = solve_with_trace(task.train, task_id="f25fbde4")
    # Evidence event should be present
    ev_events = [e for e in trace.events if e.event_type == "mechanism_class_fit"]
    assert len(ev_events) >= 1
    ev_data = ev_events[0].data
    assert "evidence" in ev_data
    assert "ranking" in ev_data
    assert "top_hypothesis" in ev_data
    assert "note" in ev_data
    # Ranking should contain exec_hint and anti_evidence
    for lane in ev_data["ranking"]:
        assert "exec_hint" in lane
        assert "anti_evidence" in lane
        assert "final_score" in lane


# ---------------------------------------------------------------------------
# Anti-evidence demotion
# ---------------------------------------------------------------------------


def test_anti_evidence_demotes_replication_without_output_matches():
    """Replication should be demoted when output doesn't show clones at predicted positions."""
    # Task where anchored exemplars exist but output doesn't match cloning pattern
    demos = (DemoPair(
        input=grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 5, 0, 0, 5],
            [0, 0, 0, 0, 0],
        ]),
        output=grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 9, 9, 0],  # completely different — NOT a clone
        ]),
    ),)
    ev, ranking = compute_evidence_and_rank(demos)
    repl = next(c for c in ranking.lanes if c.name == "replication")
    # Anti-evidence should be present
    assert repl.anti_evidence != ""
    # final_score should be lower than class_score
    assert repl.final_score < repl.class_score * 0.5


def test_anti_evidence_demotes_relocation_when_output_grows():
    """Relocation should be demoted when output has more shapes than input."""
    demos = (DemoPair(
        input=grid_from_list([[0, 1, 1, 0, 0], [0, 0, 0, 0, 5]]),
        output=grid_from_list([[1, 1, 1, 1, 0], [0, 0, 0, 0, 5]]),  # more shapes
    ),)
    ev, ranking = compute_evidence_and_rank(demos)
    reloc = next(c for c in ranking.lanes if c.name == "relocation")
    if reloc.class_score > 0:
        assert "more shapes" in reloc.anti_evidence or reloc.final_score < reloc.class_score


def test_lane_candidate_has_exec_hint():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    for c in ranking.lanes:
        assert hasattr(c, "exec_hint")
        assert isinstance(c.exec_hint, float)


def test_lane_candidate_has_anti_evidence():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    for c in ranking.lanes:
        assert hasattr(c, "anti_evidence")
        assert isinstance(c.anti_evidence, str)


def test_wording_uses_hypothesis():
    ev, ranking = compute_evidence_and_rank(_replication_task())
    assert "hypothesis" in ranking.description


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_solved_tasks_unaffected():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved


def test_1b59e163_still_solves():
    from aria.datasets import get_dataset, load_arc_task
    ds = get_dataset('v2-train')
    task = load_arc_task(ds, '1b59e163')
    result = solve_arc_task(task.train, task_id='1b59e163', use_editor_search=True)
    assert result.solved
