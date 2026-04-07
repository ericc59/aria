"""Step-level selector: predicts (target, rewrite, is_last) for each step.

Trained on per-step supervision from latent programs.
Used during search to ORDER candidate extensions by likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.guided.training_data import StepExample, TARGETS, REWRITES


class StepSelector:
    """MLP predicting target type, rewrite op, and stop/next for each step."""

    def __init__(self, input_dim: int = 30, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        h1, h2 = 48, 24
        self.n_targets = len(TARGETS)
        self.n_rewrites = len(REWRITES)

        self.W1 = rng.randn(input_dim, h1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W_target = rng.randn(h2, self.n_targets).astype(np.float32) * 0.1
        self.b_target = np.zeros(self.n_targets, dtype=np.float32)
        self.W_rewrite = rng.randn(h2, self.n_rewrites).astype(np.float32) * 0.1
        self.b_rewrite = np.zeros(self.n_rewrites, dtype=np.float32)
        self.W_stop = rng.randn(h2, 2).astype(np.float32) * 0.1
        self.b_stop = np.zeros(2, dtype=np.float32)

    def forward(self, x: np.ndarray):
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        x = x[:self.input_dim]
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W_target + self.b_target,
                h2 @ self.W_rewrite + self.b_rewrite,
                h2 @ self.W_stop + self.b_stop)

    def score_step(self, features: np.ndarray, target_idx: int, rewrite_idx: int, is_last: bool) -> float:
        """Score a candidate step. Higher = more likely."""
        t_logits, r_logits, s_logits = self.forward(features)
        t_score = t_logits[target_idx] - np.max(t_logits)
        r_score = r_logits[rewrite_idx] - np.max(r_logits)
        s_score = s_logits[1 if is_last else 0] - np.max(s_logits)
        return float(t_score + r_score + s_score)

    def rank_steps(self, features: np.ndarray, candidates: list[dict]) -> list[tuple[float, dict]]:
        """Rank candidate steps by score."""
        scored = []
        for c in candidates:
            score = self.score_step(features, c["target_idx"], c["rewrite_idx"], c["is_last"])
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return scored


def train_selector(model: StepSelector, examples: list[StepExample],
                   epochs: int = 60, lr: float = 0.01) -> dict:
    """Train via SGD on per-step cross-entropy."""
    rng = np.random.RandomState(123)
    history = {"loss": [], "target_acc": [], "rewrite_acc": []}

    for epoch in range(epochs):
        indices = rng.permutation(len(examples))
        total_loss = 0.0
        t_correct = r_correct = n = 0

        for idx in indices:
            ex = examples[idx]
            x = ex.features
            if len(x) < model.input_dim:
                x = np.pad(x, (0, model.input_dim - len(x)))
            x = x[:model.input_dim]

            # Forward
            z1 = x @ model.W1 + model.b1
            h1 = np.maximum(0, z1)
            z2 = h1 @ model.W2 + model.b2
            h2 = np.maximum(0, z2)
            t_logits = h2 @ model.W_target + model.b_target
            r_logits = h2 @ model.W_rewrite + model.b_rewrite

            t_probs = _softmax(t_logits)
            r_probs = _softmax(r_logits)
            loss = -np.log(t_probs[ex.target_idx] + 1e-10) - np.log(r_probs[ex.rewrite_idx] + 1e-10)
            total_loss += loss
            t_correct += (np.argmax(t_logits) == ex.target_idx)
            r_correct += (np.argmax(r_logits) == ex.rewrite_idx)
            n += 1

            # Backward (target head)
            dt = t_probs.copy(); dt[ex.target_idx] -= 1
            dr = r_probs.copy(); dr[ex.rewrite_idx] -= 1
            dh2 = dt @ model.W_target.T + dr @ model.W_rewrite.T
            dz2 = dh2 * (z2 > 0)
            dh1 = dz2 @ model.W2.T
            dz1 = dh1 * (z1 > 0)

            model.W_target -= lr * np.outer(h2, dt)
            model.b_target -= lr * dt
            model.W_rewrite -= lr * np.outer(h2, dr)
            model.b_rewrite -= lr * dr
            model.W2 -= lr * np.outer(h1, dz2)
            model.b2 -= lr * dz2
            model.W1 -= lr * np.outer(x, dz1)
            model.b1 -= lr * dz1

        history["loss"].append(total_loss / max(n, 1))
        history["target_acc"].append(t_correct / max(n, 1))
        history["rewrite_acc"].append(r_correct / max(n, 1))

    return history


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
