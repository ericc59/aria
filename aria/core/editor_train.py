"""Per-task CEM training for the learned graph editor.

Cross-entropy method (CEM) loop:
  1. Sample N trajectories from the policy
  2. Score each trajectory via compile/verify + MDL
  3. Select top-K (elite) trajectories
  4. Fit policy parameters to the elite distribution
  5. Repeat for R rounds

No backprop through compile/verify. No pretrained weights.
Trains from scratch on each task's demos at solve time.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.core.editor_env import (
    ActionType,
    EditAction,
    EditState,
    GraphEditEnv,
    score_graph,
)
from aria.core.editor_policy import (
    EditPolicy,
    build_action_table,
    encode_state,
    encode_action,
    STATE_DIM,
    ACTION_ENCODE_DIM,
)
from aria.core.graph import CompileSuccess, ComputationGraph, Specialization
from aria.core.protocol import Compiler, Specializer, Verifier
from aria.core.seeds import Seed


# ---------------------------------------------------------------------------
# Trajectory and scoring
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One sampled edit trajectory."""
    states: list[EditState]
    actions: list[EditAction]
    final_state: EditState
    score: float          # combined score (lower is better)
    verified: bool
    compile_count: int
    depth: int


def _score_trajectory(traj: Trajectory) -> float:
    """Score a trajectory. Lower is better. Verified wins enormously."""
    s = traj.final_state
    if s.verified:
        return -1e9 + s.score  # verified: prefer simpler graphs

    compile_bonus = 0.0
    if s.compile_result is not None and isinstance(s.compile_result, CompileSuccess):
        compile_bonus = -1000.0

    return s.diff_pixels * 10.0 + s.score + compile_bonus + traj.depth * 0.5


def _rollout(
    policy: EditPolicy,
    env: GraphEditEnv,
    seed_state: EditState,
    *,
    max_steps: int = 8,
    temperature: float = 1.0,
    rng: np.random.RandomState,
    max_compiles: int = 2,
) -> Trajectory:
    """Sample one trajectory from the policy starting at seed_state."""
    env._state = seed_state
    state = seed_state
    states = [state]
    actions_taken: list[EditAction] = []
    compiles = 0

    for step in range(max_steps):
        action_table = build_action_table(state)
        if not action_table:
            break

        # If we've exhausted compile budget, remove COMPILE from options
        if compiles >= max_compiles:
            action_table = [a for a in action_table
                           if a.action_type != ActionType.COMPILE]
            if not action_table:
                break

        idx = policy.select_action(state, action_table, temperature=temperature, rng=rng)
        action = action_table[idx]

        if action.action_type == ActionType.STOP:
            actions_taken.append(action)
            break

        if action.action_type == ActionType.COMPILE:
            compiles += 1

        env._state = state
        state = env.step(action)
        states.append(state)
        actions_taken.append(action)

        if state.verified:
            break

    return Trajectory(
        states=states,
        actions=actions_taken,
        final_state=state,
        score=_score_trajectory(
            Trajectory(states, actions_taken, state, 0, state.verified, compiles, len(actions_taken))
        ),
        verified=state.verified,
        compile_count=compiles,
        depth=len(actions_taken),
    )


# ---------------------------------------------------------------------------
# CEM training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedEditResult:
    """Outcome of the per-task learned editor."""
    solved: bool
    program: Any | None = None
    best_score: float = float("inf")
    seed_provenance: str = ""
    description: str = ""
    # Diagnostics
    rounds_run: int = 0
    trajectories_sampled: int = 0
    elites_kept: int = 0
    compiles_attempted: int = 0
    unique_verified: int = 0
    max_depth_reached: int = 0
    winning_edit_depth: int = 0
    winning_round: int = 0
    solver: str = ""  # "learned_editor"


def train_and_solve(
    seeds: list[Seed],
    examples: Sequence[Any],
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    task_id: str = "",
    n_rounds: int = 5,
    n_trajectories: int = 16,
    n_elite: int = 4,
    max_steps_per_traj: int = 6,
    max_compiles_per_traj: int = 2,
    temperature_start: float = 2.0,
    temperature_end: float = 0.5,
    noise_std: float = 0.3,
    seed: int = 42,
) -> LearnedEditResult:
    """Train a per-task editor from scratch and return the best program.

    CEM loop:
    1. For each seed, initialize a policy
    2. Sample N trajectories per round
    3. Score and rank trajectories
    4. Fit policy parameters toward elite distribution
    5. Decay temperature
    6. Stop on verification or budget
    """
    if not seeds:
        return LearnedEditResult(solved=False)

    rng = np.random.RandomState(seed)
    env = GraphEditEnv(
        examples=examples,
        specializer=specializer,
        compiler=compiler,
        verifier=verifier,
        task_id=task_id,
    )

    best_result = LearnedEditResult(solved=False)
    total_trajectories = 0
    total_compiles = 0
    total_elites = 0
    max_depth = 0

    for seed_obj in seeds:
        # Fast path: already verified seed
        if seed_obj.already_verified:
            state = env.reset(initial_graph=seed_obj.graph)
            if seed_obj.specialization is not None:
                for b in seed_obj.specialization.bindings:
                    state = env.step(EditAction(
                        action_type=ActionType.BIND,
                        node_id=b.node_id, key=b.name, value=b.value,
                    ))
            env._state = state
            state = env.step(EditAction(action_type=ActionType.COMPILE))
            total_compiles += 1
            if state.verified:
                prog = _extract_program(state)
                if prog is not None:
                    return LearnedEditResult(
                        solved=True,
                        program=prog,
                        best_score=state.score,
                        seed_provenance=seed_obj.provenance,
                        description=f"seed verified ({seed_obj.provenance})",
                        compiles_attempted=total_compiles,
                        solver="learned_editor",
                    )

        # Initialize policy for this seed
        policy = EditPolicy(hidden=32, seed=rng.randint(0, 2**31))

        # Prepare seed state
        seed_state = env.reset(initial_graph=seed_obj.graph)
        if seed_obj.specialization is not None:
            for b in seed_obj.specialization.bindings:
                env._state = seed_state
                seed_state = env.step(EditAction(
                    action_type=ActionType.BIND,
                    node_id=b.node_id, key=b.name, value=b.value,
                ))

        for round_idx in range(n_rounds):
            temperature = temperature_start + (temperature_end - temperature_start) * (round_idx / max(n_rounds - 1, 1))

            # Sample trajectories
            trajectories: list[Trajectory] = []
            for _ in range(n_trajectories):
                traj = _rollout(
                    policy, env, seed_state,
                    max_steps=max_steps_per_traj,
                    temperature=temperature,
                    rng=rng,
                    max_compiles=max_compiles_per_traj,
                )
                trajectories.append(traj)
                total_trajectories += 1
                total_compiles += traj.compile_count
                max_depth = max(max_depth, traj.depth)

                if traj.verified:
                    prog = _extract_program(traj.final_state)
                    if prog is not None:
                        return LearnedEditResult(
                            solved=True,
                            program=prog,
                            best_score=traj.score,
                            seed_provenance=seed_obj.provenance,
                            description=f"learned editor solved at round {round_idx}",
                            rounds_run=round_idx + 1,
                            trajectories_sampled=total_trajectories,
                            elites_kept=total_elites,
                            compiles_attempted=total_compiles,
                            unique_verified=1,
                            max_depth_reached=max_depth,
                            winning_edit_depth=traj.depth,
                            winning_round=round_idx,
                            solver="learned_editor",
                        )

            # Rank and select elites
            trajectories.sort(key=lambda t: t.score)
            elites = trajectories[:n_elite]
            total_elites += len(elites)

            # Fit policy to elites via parameter perturbation
            _update_policy_from_elites(policy, elites, seed_state, noise_std, rng)

    return LearnedEditResult(
        solved=False,
        rounds_run=n_rounds,
        trajectories_sampled=total_trajectories,
        elites_kept=total_elites,
        compiles_attempted=total_compiles,
        max_depth_reached=max_depth,
        description="no verified program found",
        solver="learned_editor",
    )


def _update_policy_from_elites(
    policy: EditPolicy,
    elites: list[Trajectory],
    seed_state: EditState,
    noise_std: float,
    rng: np.random.RandomState,
) -> None:
    """Update policy parameters toward the elite trajectory distribution.

    Simple approach: compute the mean action-selection target from elites,
    then do a small gradient step on the policy weights toward producing
    higher scores for elite (state, action) pairs.

    This is a CEM-style update: we're fitting the policy's weight
    distribution to make elite actions more likely.
    """
    if not elites:
        return

    current_params = policy.get_params()

    # Try several perturbations, keep the best
    best_params = current_params
    best_elite_score = _evaluate_on_elites(policy, elites)

    for _ in range(5):
        # Perturb parameters
        noise = rng.randn(len(current_params)).astype(np.float32) * noise_std
        candidate_params = current_params + noise
        policy.set_params(candidate_params)
        score = _evaluate_on_elites(policy, elites)

        if score > best_elite_score:
            best_elite_score = score
            best_params = candidate_params.copy()

    policy.set_params(best_params)


def _evaluate_on_elites(policy: EditPolicy, elites: list[Trajectory]) -> float:
    """How well does the current policy explain elite trajectories?

    Returns the sum of log-probabilities of elite actions under the policy.
    Higher is better.
    """
    total = 0.0
    for traj in elites:
        for state, action in zip(traj.states, traj.actions):
            state_vec = encode_state(state)
            action_vec = encode_action(action)
            score = policy.score_action(state_vec, action_vec)
            total += score  # higher score = more preferred
    return total


def _extract_program(state: EditState) -> Any | None:
    """Extract program from a verified state."""
    if state.compile_result is not None and isinstance(state.compile_result, CompileSuccess):
        return state.compile_result.program
    return None
