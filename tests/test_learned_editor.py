"""Tests for the per-task learned graph editor (V0)."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.editor_env import ActionType, EditAction, EditState, GraphEditEnv
from aria.core.editor_policy import (
    ACTION_ENCODE_DIM,
    STATE_DIM,
    EditPolicy,
    build_action_table,
    encode_action,
    encode_state,
)
from aria.core.editor_train import (
    LearnedEditResult,
    Trajectory,
    _rollout,
    _score_trajectory,
    train_and_solve,
)
from aria.core.graph import (
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RoleBinding,
    Specialization,
)
from aria.core.seeds import Seed, collect_seeds
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


def _tile_task():
    return (
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


def _impossible_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
    )


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def test_encode_state_shape():
    demos = _rotate_task()
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    state = env.reset()
    vec = encode_state(state)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec))


def test_encode_state_changes_after_edit():
    demos = _rotate_task()
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    seed = ComputationGraph(
        task_id="test",
        nodes={"a": GraphNode(id="a", op="APPLY_TRANSFORM", inputs=("input",))},
        output_id="a",
    )
    state1 = env.reset(initial_graph=seed)
    vec1 = encode_state(state1)

    state2 = env.step(EditAction(
        action_type=ActionType.SET_SLOT,
        node_id="a", key="transform", value="rotate 90 degrees",
    ))
    vec2 = encode_state(state2)

    # Vectors should differ (step count changed, slot count changed)
    assert not np.allclose(vec1, vec2)


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


def test_encode_action_shape():
    action = EditAction(action_type=ActionType.COMPILE)
    vec = encode_action(action)
    assert vec.shape == (ACTION_ENCODE_DIM,)
    assert vec.dtype == np.float32


def test_encode_action_compile_flag():
    compile_vec = encode_action(EditAction(action_type=ActionType.COMPILE))
    stop_vec = encode_action(EditAction(action_type=ActionType.STOP))
    assert compile_vec[9] == 1.0  # is_compile
    assert stop_vec[10] == 1.0   # is_stop


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_init_and_params():
    policy = EditPolicy(hidden=16, seed=0)
    assert policy.n_params > 0
    params = policy.get_params()
    assert params.shape == (policy.n_params,)

    # Round-trip
    policy.set_params(params)
    params2 = policy.get_params()
    np.testing.assert_allclose(params, params2)


def test_policy_select_action():
    policy = EditPolicy(hidden=16, seed=0)
    demos = _rotate_task()
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    seed = ComputationGraph(
        task_id="test",
        nodes={
            "roles": GraphNode(id="roles", op="BIND_ROLE", inputs=("input",),
                               roles=(RoleBinding(name="bg", kind="BG"),)),
            "t": GraphNode(id="t", op="APPLY_TRANSFORM", inputs=("roles",),
                           slots=(NodeSlot(name="transform", typ="TRANSFORM"),)),
        },
        output_id="t",
    )
    state = env.reset(initial_graph=seed)
    actions = build_action_table(state)
    assert len(actions) > 2  # at least some edits + COMPILE + STOP

    rng = np.random.RandomState(42)
    idx = policy.select_action(state, actions, temperature=1.0, rng=rng)
    assert 0 <= idx < len(actions)


def test_action_table_includes_compile_and_stop():
    demos = _rotate_task()
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    state = env.reset()
    actions = build_action_table(state)
    types = {a.action_type for a in actions}
    assert ActionType.COMPILE in types
    assert ActionType.STOP in types


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def test_rollout_runs():
    demos = _rotate_task()
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    seed = ComputationGraph(
        task_id="test",
        nodes={
            "roles": GraphNode(id="roles", op="BIND_ROLE", inputs=("input",),
                               roles=(RoleBinding(name="bg", kind="BG"),)),
            "t": GraphNode(id="t", op="APPLY_TRANSFORM", inputs=("roles",),
                           slots=(NodeSlot(name="transform", typ="TRANSFORM"),)),
        },
        output_id="t",
    )
    state = env.reset(initial_graph=seed)
    policy = EditPolicy(hidden=16, seed=0)
    rng = np.random.RandomState(42)

    traj = _rollout(policy, env, state, max_steps=4, temperature=1.0,
                     rng=rng, max_compiles=1)
    assert isinstance(traj, Trajectory)
    assert traj.depth >= 1
    assert traj.compile_count <= 1
    assert isinstance(traj.score, float)


# ---------------------------------------------------------------------------
# Training loop end-to-end
# ---------------------------------------------------------------------------


def test_train_and_solve_runs():
    """Training loop runs end-to-end on a known task."""
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = train_and_solve(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        n_rounds=2,
        n_trajectories=4,
    )
    assert isinstance(result, LearnedEditResult)
    # Should solve via seed fast-path
    assert result.solved is True
    assert result.program is not None


def test_train_unsolved_for_impossible():
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = train_and_solve(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        n_rounds=2,
        n_trajectories=4,
        max_steps_per_traj=3,
    )
    assert result.solved is False
    assert result.trajectories_sampled >= 1


def test_train_reports_diagnostics():
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = train_and_solve(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        n_rounds=2,
        n_trajectories=4,
    )
    assert result.rounds_run >= 1
    assert result.trajectories_sampled >= 1
    assert result.solver == "learned_editor"


# ---------------------------------------------------------------------------
# Canonical solve path integration
# ---------------------------------------------------------------------------


def test_solve_arc_with_learned_editor():
    """solve_arc_task with learned editor solves known tasks."""
    demos = _rotate_task()
    result = solve_arc_task(demos, task_id="test",
                            use_editor_search=True, use_learned_editor=True)
    assert result.solved is True


def test_solve_arc_learned_editor_only():
    """Learned editor can solve without deterministic editor."""
    demos = _tile_task()
    result = solve_arc_task(demos, task_id="test",
                            use_editor_search=False, use_learned_editor=True)
    assert result.solved is True  # static pipeline should solve this


# ---------------------------------------------------------------------------
# No deprecated dependency
# ---------------------------------------------------------------------------


def test_no_deprecated_imports():
    import aria.core.editor_policy
    import aria.core.editor_train

    for mod in [aria.core.editor_policy, aria.core.editor_train]:
        source = open(mod.__file__).read()
        assert "aria.core.hybrid" not in source
        assert "aria.core.neural" not in source
        assert "aria.core.experimental" not in source
