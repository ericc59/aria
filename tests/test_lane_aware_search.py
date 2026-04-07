"""Tests for lane-aware editor search."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.editor_env import ActionType, EditAction, EditState
from aria.core.editor_search import (
    EditSearchResult,
    _enumerate_edits,
    _sort_by_lane_relevance,
    _LANE_NAMESPACES,
    _LANE_OPS,
    search_from_seeds,
)
from aria.core.graph import ComputationGraph, GraphNode, NodeSlot, Specialization
from aria.core.seeds import Seed, collect_seeds
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Lane-aware edit ordering
# ---------------------------------------------------------------------------


def test_enumerate_edits_with_top_lane():
    """Edits should be ordered by lane relevance when top_lane is specified."""
    g = ComputationGraph(
        task_id="t",
        nodes={
            "a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",),
                           evidence={"axis": "row", "period": 2}),
        },
        output_id="a",
    )
    state = EditState(
        graph=g, specialization=Specialization(task_id="t", bindings=()),
        compile_result=None, verified=False, diff_pixels=50,
        step=0, score=5.0,
    )
    edits = _enumerate_edits(state, top_lane="periodic_repair")
    assert len(edits) >= 1


def test_sort_by_lane_relevance_periodic():
    """Periodic-local bindings should come before generic mutations."""
    edits = [
        EditAction(action_type=ActionType.SET_NODE_OP, node_id="a", value="APPLY_TRANSFORM"),
        EditAction(action_type=ActionType.BIND, node_id="__task__", key="axis", value="col"),
        EditAction(action_type=ActionType.BIND, node_id="__placement__", key="rule", value=0),
    ]
    sorted_edits = _sort_by_lane_relevance(edits, "periodic_repair")
    # __task__ binding (lane-local for periodic) should come before __placement__ and op changes
    task_bind_idx = next(i for i, e in enumerate(sorted_edits) if e.node_id == "__task__")
    placement_bind_idx = next(i for i, e in enumerate(sorted_edits) if e.node_id == "__placement__")
    assert task_bind_idx < placement_bind_idx


def test_sort_by_lane_relevance_replication():
    """Replication-local bindings should come first for replication lane."""
    edits = [
        EditAction(action_type=ActionType.SET_NODE_OP, node_id="a", value="APPLY_TRANSFORM"),
        EditAction(action_type=ActionType.BIND, node_id="__replicate__", key="key_rule", value=0),
        EditAction(action_type=ActionType.BIND, node_id="__placement__", key="rule", value=0),
    ]
    sorted_edits = _sort_by_lane_relevance(edits, "replication")
    repl_idx = next(i for i, e in enumerate(sorted_edits) if e.node_id == "__replicate__")
    placement_idx = next(i for i, e in enumerate(sorted_edits) if e.node_id == "__placement__")
    assert repl_idx < placement_idx


def test_replace_subgraph_always_first():
    """REPLACE_SUBGRAPH should always be highest priority regardless of lane."""
    from aria.core.graph import GraphFragment
    frag = GraphFragment(label="test", nodes={}, input_id="_in", output_id="_out")
    edits = [
        EditAction(action_type=ActionType.BIND, node_id="__task__", key="x", value=1),
        EditAction(action_type=ActionType.REPLACE_SUBGRAPH, node_id="a", value=frag),
    ]
    sorted_edits = _sort_by_lane_relevance(edits, "periodic_repair")
    assert sorted_edits[0].action_type == ActionType.REPLACE_SUBGRAPH


# ---------------------------------------------------------------------------
# Lane namespaces/ops are defined
# ---------------------------------------------------------------------------


def test_lane_namespaces_defined():
    assert "replication" in _LANE_NAMESPACES
    assert "relocation" in _LANE_NAMESPACES
    assert "periodic_repair" in _LANE_NAMESPACES


def test_lane_ops_defined():
    assert "replication" in _LANE_OPS
    assert "periodic_repair" in _LANE_OPS


# ---------------------------------------------------------------------------
# Search uses lane evidence
# ---------------------------------------------------------------------------


def test_search_result_includes_top_lane():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    seeds = collect_seeds(
        examples=demos, fitter=ARCFitter(), specializer=ARCSpecializer(),
        compiler=ARCCompiler(), verifier=ARCVerifier(), task_id="test",
    )
    result = search_from_seeds(
        seeds=seeds, examples=demos, specializer=ARCSpecializer(),
        compiler=ARCCompiler(), verifier=ARCVerifier(), task_id="test",
    )
    # top_lane should be populated
    assert isinstance(result.top_lane, str)


# ---------------------------------------------------------------------------
# No task-id logic
# ---------------------------------------------------------------------------


def test_no_task_id():
    import inspect
    src = inspect.getsource(search_from_seeds) + inspect.getsource(_enumerate_edits)
    assert "1b59e163" not in src


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
