"""Tests for the single-task trace capture and HTML viewer."""

from __future__ import annotations

import json

from aria.datasets import get_dataset, load_arc_task
from aria.core.graph import (
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RoleBinding,
    Specialization,
    ResolvedBinding,
)
from aria.core.editor_env import ActionType, EditAction, EditState
from aria.core.trace import TaskTrace, _grid_to_list, _graph_to_dict, _spec_to_dict
from aria.core.trace_solve import solve_with_trace
from aria.core.trace_viewer import generate_html
from aria.types import DemoPair, grid_from_list


def _rotate_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _impossible_task():
    # Two demos with contradictory mappings so no color map or derivation works
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[7, 9, 8]]),
        ),
    )


def _no_stage1_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1], [0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Trace serialization
# ---------------------------------------------------------------------------


def test_trace_to_dict_is_json_safe():
    trace = TaskTrace(task_id="test")
    trace.set_demos(_rotate_task())
    trace.add_event("static", "phase_start")
    d = trace.to_dict()
    # Must be JSON-serializable
    s = json.dumps(d, default=str)
    assert '"task_id": "test"' in s


def test_trace_to_json():
    trace = TaskTrace(task_id="test")
    trace.set_demos(_rotate_task())
    j = trace.to_json()
    parsed = json.loads(j)
    assert parsed["task_id"] == "test"
    assert len(parsed["demos"]) == 2


def test_trace_set_demos():
    trace = TaskTrace(task_id="test")
    trace.set_demos(_rotate_task())
    assert len(trace.demos) == 2
    assert trace.demos[0]["input"] == [[1, 2], [3, 4]]
    assert trace.demos[0]["output"] == [[4, 3], [2, 1]]


def test_trace_add_seed():
    trace = TaskTrace(task_id="test")
    g = ComputationGraph(
        task_id="test",
        nodes={"a": GraphNode(id="a", op="X", inputs=("input",))},
        output_id="a",
    )
    trace.add_seed("fitter", g, None, False)
    assert len(trace.seeds) == 1
    assert trace.seeds[0]["provenance"] == "fitter"
    # Must be JSON-serializable
    json.dumps(trace.seeds[0], default=str)


def test_trace_add_event():
    trace = TaskTrace(task_id="test")
    trace.add_event("static", "compile", status="failure", reason="no matching lane")
    assert len(trace.events) == 1
    assert trace.events[0].phase == "static"
    assert trace.events[0].data["reason"] == "no matching lane"


def test_trace_add_state_snapshot():
    g = ComputationGraph(task_id="t", nodes={}, output_id="")
    spec = Specialization(task_id="t", bindings=())
    state = EditState(
        graph=g, specialization=spec, compile_result=None,
        verified=False, diff_pixels=10, step=0, score=5.0,
    )
    trace = TaskTrace(task_id="t")
    trace.add_state_snapshot("deterministic", state, label="initial")
    assert len(trace.events) == 1
    d = trace.events[0].data
    assert d["diff_pixels"] == 10
    assert d["label"] == "initial"


def test_trace_add_action():
    trace = TaskTrace(task_id="t")
    action = EditAction(action_type=ActionType.SET_SLOT, node_id="a", key="x", value=42)
    trace.add_action("deterministic", action, depth=1)
    assert len(trace.events) == 1
    assert trace.events[0].data["action"]["type"] == "SET_SLOT"


# ---------------------------------------------------------------------------
# Graph/spec serialization helpers
# ---------------------------------------------------------------------------


def test_graph_to_dict():
    g = ComputationGraph(
        task_id="test",
        nodes={
            "a": GraphNode(id="a", op="BIND_ROLE", inputs=("input",),
                           roles=(RoleBinding(name="bg", kind="BG"),)),
            "b": GraphNode(id="b", op="APPLY_TRANSFORM", inputs=("a",),
                           slots=(NodeSlot(name="t", typ="TRANSFORM", evidence="rotate 90"),)),
        },
        output_id="b",
    )
    d = _graph_to_dict(g)
    assert d["output_id"] == "b"
    assert "a" in d["nodes"]
    assert d["nodes"]["b"]["slots"][0]["evidence"] == "rotate 90"
    # Must be JSON-safe
    json.dumps(d, default=str)


def test_spec_to_dict():
    spec = Specialization(
        task_id="t",
        bindings=(ResolvedBinding(node_id="a", name="x", value=42, source="evidence"),),
    )
    d = _spec_to_dict(spec)
    assert d["bindings"][0]["value"] == 42
    json.dumps(d, default=str)


def test_spec_to_dict_none():
    assert _spec_to_dict(None) is None


# ---------------------------------------------------------------------------
# Full instrumented solve
# ---------------------------------------------------------------------------


def test_solve_with_trace_solved():
    demos = _rotate_task()
    trace = solve_with_trace(demos, task_id="test_rotate")
    assert trace.solved is True
    assert trace.solver in ("static", "stage1")
    assert len(trace.demos) == 2
    assert len(trace.events) >= 2  # at least phase_start + phase_end


def test_solve_with_trace_unsolved():
    demos = _impossible_task()
    trace = solve_with_trace(demos, task_id="test_impossible")
    assert trace.solved is False
    assert len(trace.seeds) >= 1
    assert len(trace.events) >= 4  # static start/end + more


def test_solve_with_trace_has_seeds():
    demos = _impossible_task()
    trace = solve_with_trace(demos, task_id="test")
    assert len(trace.seeds) >= 1
    for seed in trace.seeds:
        assert "provenance" in seed
        assert "graph" in seed


def test_solve_with_trace_json_roundtrip():
    demos = _impossible_task()
    trace = solve_with_trace(demos, task_id="test")
    j = trace.to_json()
    parsed = json.loads(j)
    assert parsed["task_id"] == "test"
    assert isinstance(parsed["events"], list)


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------


def test_generate_html_produces_output():
    demos = _impossible_task()
    trace = solve_with_trace(demos, task_id="test")
    html = generate_html(trace)
    assert "<!DOCTYPE html>" in html
    assert "test" in html
    assert "UNSOLVED" in html


def test_generate_html_solved_task():
    demos = _rotate_task()
    trace = solve_with_trace(demos, task_id="test_rotate")
    html = generate_html(trace)
    assert "SOLVED" in html
    assert "stage1" in html or "static" in html


def test_generate_html_has_demo_grids():
    demos = _rotate_task()
    trace = solve_with_trace(demos, task_id="test")
    html = generate_html(trace)
    # Should contain colored grid cells
    assert "background:" in html
    assert "Demo 0" in html


def test_generate_html_has_seeds_section():
    demos = _impossible_task()
    trace = solve_with_trace(demos, task_id="test")
    html = generate_html(trace)
    assert "Seeds" in html


def test_solve_with_trace_records_stage1_gate_failure():
    trace = solve_with_trace(_no_stage1_task(), task_id="test_stage1_fail")
    assert trace.solved is False
    assert trace.seeds == []
    stage1_events = [e for e in trace.events if e.phase == "stage1"]
    assert stage1_events
    output_spec = [e for e in trace.events if e.phase == "stage1" and e.event_type == "output_spec"]
    assert output_spec and output_spec[0].data["verified"] is False
    skipped = [e for e in trace.events if e.phase == "static" and e.event_type == "skipped"]
    assert skipped and skipped[0].data["reason"] == "stage1_gate_failed"


def test_solve_with_trace_can_succeed_via_stage1_derivation():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    trace = solve_with_trace(task.train, task_id="1a6449f1")
    assert trace.solved is True
    assert trace.solver == "stage1"
    direct = [e for e in trace.events if e.phase == "stage1" and e.event_type == "direct_stage1_program"]
    assert direct and direct[0].data["verified"] is True


def test_solve_with_trace_can_succeed_via_stage1_scaled_render():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    trace = solve_with_trace(task.train, task_id="f25fbde4")
    assert trace.solved is True
    assert trace.solver == "stage1"
    direct = [e for e in trace.events if e.phase == "stage1" and e.event_type == "direct_stage1_program"]
    assert direct and direct[0].data["verified"] is True


def test_solve_with_trace_can_succeed_via_stage1_tiled_input_render():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "00576224")
    trace = solve_with_trace(task.train, task_id="00576224")
    assert trace.solved is True
    assert trace.solver == "stage1"
    direct = [e for e in trace.events if e.phase == "stage1" and e.event_type == "direct_stage1_program"]
    assert direct and direct[0].data["verified"] is True


# ---------------------------------------------------------------------------
# Tracing does not break canonical solve when disabled
# ---------------------------------------------------------------------------


def test_canonical_solve_still_works():
    """The canonical path works independently of tracing."""
    from aria.core.arc import solve_arc_task
    demos = _rotate_task()
    result = solve_arc_task(demos, task_id="test", use_editor_search=False)
    assert result.solved is True
