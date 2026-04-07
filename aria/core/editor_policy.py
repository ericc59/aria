"""Per-task learned graph editor — V0 policy and state encoding.

A small neural policy that maps structured editor state to a
distribution over graph/spec edit actions. Trained from scratch
on each task's demos at test time using CEM (cross-entropy method).

Operates on ComputationGraph + Specialization through GraphEditEnv.
No pretrained weights. No remote calls. No pixel prediction.

Part of the canonical architecture.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from aria.core.editor_env import (
    ActionType,
    EditAction,
    EditState,
    GraphEditEnv,
    score_graph,
)
from aria.core.editor_search import (
    _ARC_OPS,
    _SLOT_EVIDENCE_OPTIONS,
    _TRANSFORM_VALUES,
    _enumerate_edits,
)
from aria.core.graph import (
    CompileSuccess,
    ComputationGraph,
    Specialization,
)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

# Feature vector layout (fixed size = 40):
#   [0]     n_nodes
#   [1]     n_edges
#   [2]     n_slots_total
#   [3]     n_resolved_bindings
#   [4]     n_unresolved_slots (slots - bindings, clamped >= 0)
#   [5]     mdl_score
#   [6]     compiled (0/1)
#   [7]     verified (0/1)
#   [8]     diff_pixels (normalized)
#   [9]     edit_step
#   [10-15] op_counts: count of each ARC op in the graph (6 ops)
#   [16]    has_output_node (0/1)
#   [17]    graph_valid (0/1)
#   [18-27] slot_type_counts (10 slots for known types)
#   [28-37] binding_namespace_counts (10 buckets)
#   [38]    n_role_bindings
#   [39]    last_action_type_id (0 if first step)
#   [40]    has_diagnostic (0/1)
#   [41]    diagnostic_diff_fraction (0-1)
#   [42]    n_repair_hints
#   [43]    max_hint_confidence (0-1)
#   [44]    n_blamed_bindings
#   [45]    n_subgraph_blames
#   [46]    n_replacement_fragments
#   [47]    max_blame_confidence (0-1)
#   [48]    interior_diff_fraction (0-1, from region residuals)
#   [49]    frame_diff_fraction (0-1, from region residuals)

STATE_DIM = 50

_OP_INDEX = {op: i for i, op in enumerate(_ARC_OPS)}


def encode_state(state: EditState) -> np.ndarray:
    """Encode EditState into a fixed-size float vector."""
    v = np.zeros(STATE_DIM, dtype=np.float32)
    graph = state.graph
    spec = state.specialization

    v[0] = len(graph.nodes)
    v[1] = sum(len(n.inputs) for n in graph.nodes.values())
    v[2] = sum(len(n.slots) for n in graph.nodes.values())
    v[3] = len(spec.bindings)
    v[4] = max(v[2] - v[3], 0)
    v[5] = state.score / 20.0  # normalize
    v[6] = float(state.compile_result is not None
                 and isinstance(state.compile_result, CompileSuccess))
    v[7] = float(state.verified)
    v[8] = min(state.diff_pixels / 100.0, 10.0)  # cap
    v[9] = state.step / 20.0

    # Op counts
    for node in graph.nodes.values():
        idx = _OP_INDEX.get(node.op)
        if idx is not None and 10 + idx < 16:
            v[10 + idx] += 1

    v[16] = float(graph.output_id in graph.nodes or graph.output_id == "input")
    v[17] = float(len(graph.validate()) == 0)

    # Slot name distribution (hash into 10 buckets)
    for node in graph.nodes.values():
        for slot in node.slots:
            bucket = hash(slot.name) % 10
            v[18 + bucket] += 1

    # Binding namespace distribution (hash into 10 buckets)
    for b in spec.bindings:
        bucket = hash(b.node_id) % 10
        v[28 + bucket] += 1

    v[38] = sum(len(n.roles) for n in graph.nodes.values())

    if state.history:
        v[39] = float(state.history[-1].action_type.value) / 11.0

    # Diagnostic features
    diag = state.diagnostic if hasattr(state, 'diagnostic') else None
    if diag is not None:
        v[40] = 1.0
        v[41] = min(diag.diff_fraction, 1.0)
        v[42] = len(diag.repair_hints) / 10.0
        v[43] = max((h.confidence for h in diag.repair_hints), default=0.0)
        v[44] = len(diag.blamed_bindings) / 5.0
        # Structural diagnostic features
        blames = getattr(diag, 'subgraph_blames', ())
        frags = getattr(diag, 'replacement_fragments', ())
        v[45] = len(blames) / 3.0
        v[46] = len(frags) / 5.0
        v[47] = max((b.confidence for b in blames), default=0.0)
        regions = getattr(diag, 'region_residuals', ())
        for r in regions:
            if r.region_label == "interior":
                v[48] = min(r.diff_fraction, 1.0)
            elif r.region_label == "frame":
                v[49] = min(r.diff_fraction, 1.0)

    return v


# ---------------------------------------------------------------------------
# Action table — flat list of concrete EditActions per state
# ---------------------------------------------------------------------------


def build_action_table(state: EditState) -> list[EditAction]:
    """Build the flat action table for the current state.

    Includes all local graph/spec edits plus COMPILE and STOP.
    The policy selects an index into this table.
    """
    actions = _enumerate_edits(state)
    actions.append(EditAction(action_type=ActionType.COMPILE))
    actions.append(EditAction(action_type=ActionType.STOP))
    return actions


# ---------------------------------------------------------------------------
# Policy network — small MLP
# ---------------------------------------------------------------------------


class EditPolicy:
    """Small MLP policy: state vector -> action logits.

    Architecture: state_dim -> hidden -> hidden -> 1 (per-action score).
    The policy scores each candidate action by concatenating
    (state_encoding, action_encoding) and producing a scalar.

    Total params ~= state_dim*hidden + hidden*hidden + hidden*1 + biases
    With hidden=32: 40*32 + 32*32 + 32*1 + 32+32+1 = 1280+1024+32+65 ≈ 2.4K params
    """

    def __init__(self, hidden: int = 32, seed: int | None = None) -> None:
        self.hidden = hidden
        rng = np.random.RandomState(seed)

        input_dim = STATE_DIM + ACTION_ENCODE_DIM
        # Layer 1: input -> hidden
        self.w1 = rng.randn(input_dim, hidden).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        # Layer 2: hidden -> hidden
        self.w2 = rng.randn(hidden, hidden).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden, dtype=np.float32)
        # Layer 3: hidden -> 1
        self.w3 = rng.randn(hidden, 1).astype(np.float32) * 0.1
        self.b3 = np.zeros(1, dtype=np.float32)

    def score_action(self, state_vec: np.ndarray, action_vec: np.ndarray) -> float:
        """Score a single (state, action) pair."""
        x = np.concatenate([state_vec, action_vec])
        h = np.tanh(x @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        return float((h @ self.w3 + self.b3)[0])

    def select_action(
        self, state: EditState, actions: list[EditAction],
        temperature: float = 1.0, rng: np.random.RandomState | None = None,
    ) -> int:
        """Select an action index from the action table.

        Returns the index into `actions`.
        Uses softmax sampling with the given temperature.
        """
        if not actions:
            return 0
        if rng is None:
            rng = np.random.RandomState()

        state_vec = encode_state(state)
        logits = np.array([
            self.score_action(state_vec, encode_action(a))
            for a in actions
        ])

        # Softmax with temperature
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()  # stability
        probs = np.exp(logits)
        probs = probs / (probs.sum() + 1e-10)

        return int(rng.choice(len(actions), p=probs))

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            self.w1.ravel(), self.b1.ravel(),
            self.w2.ravel(), self.b2.ravel(),
            self.w3.ravel(), self.b3.ravel(),
        ])

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from a flat vector."""
        idx = 0
        for attr in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
            arr = getattr(self, attr)
            size = arr.size
            setattr(self, attr, params[idx:idx + size].reshape(arr.shape).astype(np.float32))
            idx += size

    @property
    def n_params(self) -> int:
        return sum(
            getattr(self, a).size
            for a in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')
        )


# ---------------------------------------------------------------------------
# Action encoding — fixed-size vector for each EditAction
# ---------------------------------------------------------------------------

# Action features (fixed size = 12):
#   [0]     action_type_id (normalized)
#   [1-6]   one-hot for target op (if SET_NODE_OP)
#   [7]     slot_name_hash (normalized)
#   [8]     value_hash (normalized)
#   [9]     is_compile (0/1)
#   [10]    is_stop (0/1)
#   [11]    is_bind (0/1)

ACTION_ENCODE_DIM = 12

_ACTION_TYPE_NORM = 1.0 / max(len(ActionType), 1)


def encode_action(action: EditAction) -> np.ndarray:
    """Encode an EditAction into a fixed-size float vector."""
    v = np.zeros(ACTION_ENCODE_DIM, dtype=np.float32)
    v[0] = action.action_type.value * _ACTION_TYPE_NORM

    # One-hot for target op
    if action.action_type == ActionType.SET_NODE_OP and isinstance(action.value, str):
        idx = _OP_INDEX.get(action.value)
        if idx is not None and 1 + idx < 7:
            v[1 + idx] = 1.0

    if action.key:
        v[7] = (hash(action.key) % 100) / 100.0
    if action.value is not None:
        v[8] = (hash(str(action.value)) % 100) / 100.0

    v[9] = float(action.action_type == ActionType.COMPILE)
    v[10] = float(action.action_type == ActionType.STOP)
    v[11] = float(action.action_type == ActionType.BIND)

    return v
