"""Tests for the per-task graph editing environment scaffold."""

from __future__ import annotations

from aria.core.editor_env import (
    ActionType,
    EditAction,
    EditState,
    GraphEditEnv,
    score_graph,
)
from aria.core.graph import (
    CompileFailure,
    CompileSuccess,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    ResolvedBinding,
    RoleBinding,
    Specialization,
)
from aria.core.arc import ARCCompiler, ARCSpecializer, ARCVerifier
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# score_graph
# ---------------------------------------------------------------------------


def test_score_empty_graph():
    g = ComputationGraph(task_id="t", nodes={}, output_id="")
    s = Specialization(task_id="t", bindings=())
    assert score_graph(g, s) == 0.0


def test_score_increases_with_nodes():
    g1 = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="x")},
        output_id="a",
    )
    g2 = ComputationGraph(
        task_id="t",
        nodes={
            "a": GraphNode(id="a", op="x"),
            "b": GraphNode(id="b", op="y", inputs=("a",)),
        },
        output_id="b",
    )
    s = Specialization(task_id="t", bindings=())
    assert score_graph(g2, s) > score_graph(g1, s)


def test_score_penalizes_unresolved_slots():
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(
            id="a", op="x",
            slots=(NodeSlot(name="p", typ="INT"),),
        )},
        output_id="a",
    )
    s_unresolved = Specialization(task_id="t", bindings=())
    s_resolved = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="a", name="p", value=42),
    ))
    # Unresolved costs more than resolved
    assert score_graph(g, s_unresolved) > score_graph(g, s_resolved)


# ---------------------------------------------------------------------------
# GraphEditEnv — basic lifecycle
# ---------------------------------------------------------------------------


def test_env_reset_returns_state():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
    )
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    state = env.reset()
    assert isinstance(state, EditState)
    assert state.step == 0
    assert state.verified is False
    assert len(state.graph.nodes) == 0


def test_env_reset_with_seed_graph():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
    )
    seed = ComputationGraph(
        task_id="test",
        nodes={"a": GraphNode(id="a", op="APPLY_TRANSFORM", inputs=("input",))},
        output_id="a",
    )
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    state = env.reset(initial_graph=seed)
    assert len(state.graph.nodes) == 1


# ---------------------------------------------------------------------------
# GraphEditEnv — action primitives
# ---------------------------------------------------------------------------


def test_env_add_node():
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    state = env.reset()
    state = env.step(EditAction(
        action_type=ActionType.ADD_NODE,
        node_id="n1",
        value="APPLY_TRANSFORM",
    ))
    assert "n1" in state.graph.nodes
    assert state.graph.nodes["n1"].op == "APPLY_TRANSFORM"
    assert state.step == 1


def test_env_remove_node():
    seed = ComputationGraph(
        task_id="t",
        nodes={
            "a": GraphNode(id="a", op="x"),
            "b": GraphNode(id="b", op="y", inputs=("a",)),
        },
        output_id="b",
    )
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=seed)
    state = env.step(EditAction(action_type=ActionType.REMOVE_NODE, node_id="a"))
    assert "a" not in state.graph.nodes
    # Edge from b->a should be cleaned up
    assert "a" not in state.graph.nodes.get("b", GraphNode(id="b", op="y")).inputs


def test_env_set_node_op():
    seed = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="old_op")},
        output_id="a",
    )
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=seed)
    state = env.step(EditAction(
        action_type=ActionType.SET_NODE_OP, node_id="a", value="new_op",
    ))
    assert state.graph.nodes["a"].op == "new_op"


def test_env_add_and_remove_edge():
    seed = ComputationGraph(
        task_id="t",
        nodes={
            "a": GraphNode(id="a", op="x"),
            "b": GraphNode(id="b", op="y"),
        },
        output_id="b",
    )
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=seed)

    # Add edge b->a
    state = env.step(EditAction(
        action_type=ActionType.ADD_EDGE, node_id="b", target_id="a",
    ))
    assert "a" in state.graph.nodes["b"].inputs

    # Remove edge b->a
    state = env.step(EditAction(
        action_type=ActionType.REMOVE_EDGE, node_id="b", target_id="a",
    ))
    assert "a" not in state.graph.nodes["b"].inputs


def test_env_set_slot():
    seed = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(
            id="a", op="x",
            slots=(NodeSlot(name="p", typ="INT"),),
        )},
        output_id="a",
    )
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=seed)
    state = env.step(EditAction(
        action_type=ActionType.SET_SLOT, node_id="a", key="p", value=42,
    ))
    assert state.graph.nodes["a"].slots[0].evidence == 42


def test_env_add_role():
    seed = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="x")},
        output_id="a",
    )
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=seed)
    state = env.step(EditAction(
        action_type=ActionType.ADD_ROLE, node_id="a", key="bg", value="BACKGROUND",
    ))
    assert len(state.graph.nodes["a"].roles) == 1
    assert state.graph.nodes["a"].roles[0].kind == "BACKGROUND"


def test_env_bind_and_unbind():
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset()

    # Bind
    state = env.step(EditAction(
        action_type=ActionType.BIND, node_id="a", key="x", value=42,
    ))
    assert state.specialization.get("a", "x") == 42

    # Unbind
    state = env.step(EditAction(
        action_type=ActionType.UNBIND, node_id="a", key="x",
    ))
    assert state.specialization.get("a", "x") is None


def test_env_compile_action():
    """Compile action triggers the full compile+verify pipeline."""
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset()
    # With an empty graph, compile should fail gracefully
    state = env.step(EditAction(action_type=ActionType.COMPILE))
    assert state.verified is False
    assert state.compile_result is not None


def test_env_history_tracks_actions():
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset()
    state = env.step(EditAction(action_type=ActionType.ADD_NODE, node_id="a", value="x"))
    state = env.step(EditAction(action_type=ActionType.SET_NODE_OP, node_id="a", value="y"))
    assert len(state.history) == 2
    assert state.history[0].action_type == ActionType.ADD_NODE
    assert state.history[1].action_type == ActionType.SET_NODE_OP


# ---------------------------------------------------------------------------
# Verify the canonical path still works end-to-end
# ---------------------------------------------------------------------------


def test_canonical_path_arc_tile():
    """The canonical graph pipeline solves a tile task."""
    from aria.core.arc import solve_arc_task
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 6, 5, 6], [7, 8, 7, 8],
                [5, 6, 5, 6], [7, 8, 7, 8],
            ]),
        ),
    )
    result = solve_arc_task(demos, task_id="tile")
    assert result.solved is True


def test_canonical_path_arc_rotate():
    """The canonical graph pipeline solves a rotation task."""
    from aria.core.arc import solve_arc_task
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )
    result = solve_arc_task(demos, task_id="rotate")
    assert result.solved is True
