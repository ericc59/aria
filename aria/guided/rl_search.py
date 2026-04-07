"""RL-trained multi-step search for ARC.

The policy network learns to select program construction steps
directly from ARC task verification rewards. No synthetic labels needed.

Training loop:
1. For each ARC task, run multiple search episodes
2. Each episode: policy proposes extensions, executor verifies
3. Reward = 1 if program verifies on all train demos, 0 otherwise
4. Update policy via REINFORCE (policy gradient)

The policy is an MLP that scores (workspace_features, extension) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from collections import defaultdict

from aria.guided.grammar import (
    Program, Action, Act, Target, Rewrite, execute_program,
)
from aria.guided.search import (
    SearchResult, _verify, _enumerate_extensions, predict_test,
)
from aria.guided.training_data import TARGET_TO_IDX, REWRITE_TO_IDX, _featurize_workspace
from aria.guided.workspace import build_workspace, _detect_bg
from aria.types import Grid


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class PolicyNet:
    """MLP policy: scores (workspace, extension) → scalar."""

    def __init__(self, input_dim: int = 30, n_targets: int = 7, n_rewrites: int = 6, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Encode: workspace features (input_dim) + one-hot target (n_targets) + one-hot rewrite (n_rewrites) + is_last (1)
        self.ws_dim = input_dim
        self.n_targets = n_targets
        self.n_rewrites = n_rewrites
        total_in = input_dim + n_targets + n_rewrites + 1
        h = 32

        self.W1 = rng.randn(total_in, h).astype(np.float32) * np.sqrt(2.0 / total_in)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = rng.randn(h, 1).astype(np.float32) * 0.1
        self.b2 = np.zeros(1, dtype=np.float32)

        self.total_in = total_in

    def score(self, ws_features: np.ndarray, target_idx: int, rewrite_idx: int, is_last: bool) -> float:
        x = self._encode(ws_features, target_idx, rewrite_idx, is_last)
        h = np.maximum(0, x @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2)[0])

    def score_batch(self, ws_features: np.ndarray, extensions: list[dict]) -> list[float]:
        return [self.score(ws_features, e["target_idx"], e["rewrite_idx"], e["is_last"])
                for e in extensions]

    def _encode(self, ws_features, target_idx, rewrite_idx, is_last):
        x = np.zeros(self.total_in, dtype=np.float32)
        n = min(len(ws_features), self.ws_dim)
        x[:n] = ws_features[:n]
        x[self.ws_dim + target_idx] = 1.0
        x[self.ws_dim + self.n_targets + rewrite_idx] = 1.0
        x[-1] = 1.0 if is_last else 0.0
        return x

    def gradient_step(self, ws_features, target_idx, rewrite_idx, is_last, advantage, lr=0.001):
        """One REINFORCE gradient step."""
        x = self._encode(ws_features, target_idx, rewrite_idx, is_last)
        # Forward
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        score = float((h1 @ self.W2 + self.b2)[0])

        # We want to increase score when advantage > 0, decrease when < 0
        # d(score)/d(params) * advantage
        d_out = np.array([advantage], dtype=np.float32)
        dW2 = np.outer(h1, d_out)
        db2 = d_out
        dh1 = d_out @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2


# ---------------------------------------------------------------------------
# RL search: policy-guided with epsilon-greedy exploration
# ---------------------------------------------------------------------------

def rl_search(
    train: list[tuple[Grid, Grid]],
    policy: PolicyNet,
    max_candidates: int = 500,
    max_steps: int = 3,
    top_k: int = 10,
    epsilon: float = 0.0,
) -> SearchResult:
    """Search guided by RL policy."""
    if not train:
        return SearchResult(False, None, 0, 0)

    bgs = [_detect_bg(inp) for inp, _ in train]
    ws = build_workspace(train[0][0], train[0][1])
    features = _featurize_workspace(ws)
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)
    candidates_tried = 0

    # BFS with policy-ranked extensions
    from collections import deque
    queue = deque()
    queue.append(Program())

    rng = np.random.RandomState()

    while queue and candidates_tried < max_candidates:
        partial = queue.popleft()
        n_steps = sum(1 for a in partial.actions if a.act in (Act.NEXT, Act.STOP))
        if n_steps >= max_steps:
            continue

        extensions = _enumerate_extensions(partial, train, bgs)
        if not extensions:
            continue

        # Score with policy
        ext_infos = []
        for ext in extensions:
            target_idx = 0
            rewrite_idx = 0
            is_last = ext[-1].act == Act.STOP
            for a in ext:
                if a.act == Act.SELECT_TARGET and a.choice is not None:
                    target_idx = TARGET_TO_IDX.get(a.choice, 0)
                if a.act == Act.REWRITE and a.choice is not None:
                    rewrite_idx = REWRITE_TO_IDX.get(a.choice, 0)
            score = policy.score(features, target_idx, rewrite_idx, is_last)
            ext_infos.append((score, ext, target_idx, rewrite_idx, is_last))

        # Epsilon-greedy: with probability epsilon, shuffle; otherwise sort by score
        if rng.rand() < epsilon:
            rng.shuffle(ext_infos)
        else:
            ext_infos.sort(key=lambda x: -x[0])

        for score, ext, ti, ri, is_last in ext_infos[:top_k]:
            candidates_tried += 1
            candidate = partial.copy()
            for action in ext:
                candidate.append(action)

            if is_last:
                ok, diff = _verify(candidate, train, bgs)
                if ok:
                    return SearchResult(True, candidate, candidates_tried, 0)
                if diff < best_diff:
                    best_diff = diff
            else:
                queue.append(candidate)

            if candidates_tried >= max_candidates:
                break

    return SearchResult(False, None, candidates_tried, best_diff)


# ---------------------------------------------------------------------------
# RL training loop
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One search episode's trace."""
    task_id: str
    steps: list[dict]  # each: {ws_features, target_idx, rewrite_idx, is_last}
    reward: float      # 1.0 if solved, 0.0 if not


def train_rl(
    policy: PolicyNet,
    arc_tasks: list[tuple[str, list[tuple[Grid, Grid]]]],
    n_epochs: int = 50,
    episodes_per_task: int = 5,
    max_candidates: int = 200,
    lr: float = 0.001,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
) -> dict[str, list]:
    """Train policy via REINFORCE on ARC tasks."""
    history = {"epoch": [], "n_solved": [], "avg_reward": []}
    rng = np.random.RandomState(42)

    for epoch in range(n_epochs):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epoch / max(1, n_epochs - 1)
        episodes: list[Episode] = []

        for task_id, train in arc_tasks:
            bgs = [_detect_bg(inp) for inp, _ in train]
            ws = build_workspace(train[0][0], train[0][1])
            features = _featurize_workspace(ws)

            for _ in range(episodes_per_task):
                # Run one episode
                trace = []
                partial = Program()
                solved = False
                cands = 0

                for step in range(3):
                    extensions = _enumerate_extensions(partial, train, bgs)
                    if not extensions:
                        break

                    # Score and sample
                    scored = []
                    for ext in extensions:
                        ti, ri, il = 0, 0, ext[-1].act == Act.STOP
                        for a in ext:
                            if a.act == Act.SELECT_TARGET and a.choice:
                                ti = TARGET_TO_IDX.get(a.choice, 0)
                            if a.act == Act.REWRITE and a.choice:
                                ri = REWRITE_TO_IDX.get(a.choice, 0)
                        s = policy.score(features, ti, ri, il)
                        scored.append((s, ext, ti, ri, il))

                    # Epsilon-greedy selection
                    if rng.rand() < epsilon:
                        idx = rng.randint(0, len(scored))
                    else:
                        probs = _softmax_scores([s for s, _, _, _, _ in scored])
                        idx = rng.choice(len(scored), p=probs)

                    _, ext, ti, ri, il = scored[idx]
                    trace.append({"ws_features": features, "target_idx": ti,
                                  "rewrite_idx": ri, "is_last": il})

                    for action in ext:
                        partial.append(action)

                    cands += 1

                    if il:  # STOP
                        ok, diff = _verify(partial, train, bgs)
                        if ok:
                            solved = True
                        break

                episodes.append(Episode(task_id, trace, 1.0 if solved else 0.0))

        # REINFORCE update
        baseline = np.mean([e.reward for e in episodes])
        n_solved = sum(1 for e in episodes if e.reward > 0)

        for ep in episodes:
            advantage = ep.reward - baseline
            if abs(advantage) < 1e-6:
                continue
            for step in ep.steps:
                policy.gradient_step(
                    step["ws_features"], step["target_idx"],
                    step["rewrite_idx"], step["is_last"],
                    advantage, lr=lr,
                )

        avg_r = np.mean([e.reward for e in episodes])
        history["epoch"].append(epoch)
        history["n_solved"].append(n_solved)
        history["avg_reward"].append(avg_r)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:>3d}: solved={n_solved}/{len(episodes)} "
                  f"avg_reward={avg_r:.3f} epsilon={epsilon:.2f}")

    return history


def _softmax_scores(scores):
    scores = np.array(scores, dtype=np.float64)
    scores -= scores.max()
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum()
