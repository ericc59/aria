"""Per-task graph editing environment — scaffold for the learned path.

This module defines the canonical environment interface for a per-task
recurrent graph editor. The editor operates on ComputationGraph +
Specialization, using the existing compile/verify pipeline as the
evaluation oracle.

Architecture:
    state  = (current_graph, current_specialization, compile_result, diff_signal)
    action = graph edit | specialization edit | compile | stop
    reward = exact verification (primary) + diff reduction (shaping) + MDL (regularization)

The neural editor itself is not implemented here. This module provides:
  1. EditState — the full state visible to the editor at each step
  2. EditAction — the discrete action space (graph rewrites)
  3. GraphEditEnv — the environment that applies actions and scores results
  4. score_graph — MDL-inspired scoring for graph complexity

Design principles:
  - Actions are typed graph rewrites, not raw pixel operations
  - The symbolic layer (compile + verify) is the evaluation oracle
  - Test-time training: the editor trains from scratch on each task's demos
  - MDL pressure rewards simpler graphs that explain all demos
  - No remote model calls, no pretrained weights in this scaffold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from aria.core.graph import (
    CompileResult,
    CompileSuccess,
    CompileFailure,
    ComputationGraph,
    GraphFragment,
    GraphNode,
    NodeSlot,
    ResolvedBinding,
    RoleBinding,
    Specialization,
    VerifyDiagnostic,
)
from aria.core.protocol import Compiler, Specializer, Verifier


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class ActionType(Enum):
    """Discrete action types the editor can take."""
    ADD_NODE = auto()       # add a new node to the graph
    REMOVE_NODE = auto()    # remove a node (and its edges)
    SET_NODE_OP = auto()    # change a node's abstract operation
    ADD_EDGE = auto()       # connect two nodes
    REMOVE_EDGE = auto()    # disconnect two nodes
    SET_SLOT = auto()       # set a slot value on a node
    ADD_ROLE = auto()       # add a role binding to a node
    BIND = auto()           # add a resolved binding to the specialization
    UNBIND = auto()         # remove a resolved binding
    REPLACE_SUBGRAPH = auto()  # replace blamed nodes with a typed fragment
    COMPILE = auto()        # trigger compile + verify (expensive)
    STOP = auto()           # declare done


@dataclass(frozen=True)
class EditAction:
    """One edit the agent proposes."""
    action_type: ActionType
    node_id: str = ""
    target_id: str = ""     # for edges: the other node
    key: str = ""           # slot name, role name, or binding name
    value: Any = None       # new value for SET_SLOT, BIND, etc.


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class EditState:
    """Full state visible to the editor at each step."""
    graph: ComputationGraph
    specialization: Specialization
    compile_result: CompileResult | None
    verified: bool
    diff_pixels: int            # total pixel diff across all demos (0 = solved)
    step: int                   # how many actions taken so far
    score: float                # current MDL + diff score (lower = better)
    history: list[EditAction] = field(default_factory=list)
    diagnostic: VerifyDiagnostic | None = None  # present when compiled but verify failed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_graph(graph: ComputationGraph, spec: Specialization) -> float:
    """MDL-inspired complexity score for a graph + specialization.

    Lower is better. Penalizes:
      - number of nodes (each node costs 1.0)
      - number of edges (each edge costs 0.5)
      - number of unresolved slots (each costs 2.0 — uncertainty penalty)
      - number of bindings (each costs 0.1 — slight preference for fewer params)

    This is a heuristic approximation of description length.
    A more principled version would compute actual bit-cost of encoding
    the graph + specialization.
    """
    n_nodes = len(graph.nodes)
    n_edges = sum(len(n.inputs) for n in graph.nodes.values())
    n_slots = sum(len(n.slots) for n in graph.nodes.values())
    n_resolved = len(spec.bindings)
    n_unresolved = n_slots - n_resolved

    return (
        1.0 * n_nodes
        + 0.5 * n_edges
        + 2.0 * max(n_unresolved, 0)
        + 0.1 * n_resolved
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class GraphEditEnv:
    """Environment for per-task graph editing.

    Wraps the compile/verify pipeline. The editor proposes actions;
    the environment applies them and returns the new state.

    Usage:
        env = GraphEditEnv(demos, specializer, compiler, verifier)
        state = env.reset()
        while not state.verified and state.step < max_steps:
            action = agent.propose(state)  # neural editor (not implemented here)
            state = env.step(action)
    """

    def __init__(
        self,
        examples: Sequence[Any],
        specializer: Specializer,
        compiler: Compiler,
        verifier: Verifier,
        *,
        task_id: str = "",
        available_ops: tuple[str, ...] = (),
    ) -> None:
        self.examples = list(examples)
        self.specializer = specializer
        self.compiler = compiler
        self.verifier = verifier
        self.task_id = task_id
        self.available_ops = available_ops
        self._state: EditState | None = None

    def reset(self, initial_graph: ComputationGraph | None = None) -> EditState:
        """Initialize the environment with an empty or seed graph."""
        if initial_graph is None:
            initial_graph = ComputationGraph(
                task_id=self.task_id,
                nodes={},
                output_id="",
            )
        spec = Specialization(task_id=self.task_id, bindings=())
        self._state = EditState(
            graph=initial_graph,
            specialization=spec,
            compile_result=None,
            verified=False,
            diff_pixels=self._compute_diff(None),
            step=0,
            score=score_graph(initial_graph, spec),
        )
        return self._state

    def step(self, action: EditAction) -> EditState:
        """Apply an action and return the new state."""
        assert self._state is not None, "call reset() first"
        state = self._state

        new_graph = state.graph
        new_spec = state.specialization
        compile_result = state.compile_result
        verified = state.verified

        if action.action_type == ActionType.ADD_NODE:
            new_graph = self._add_node(new_graph, action)
        elif action.action_type == ActionType.REMOVE_NODE:
            new_graph = self._remove_node(new_graph, action)
        elif action.action_type == ActionType.SET_NODE_OP:
            new_graph = self._set_node_op(new_graph, action)
        elif action.action_type == ActionType.ADD_EDGE:
            new_graph = self._add_edge(new_graph, action)
        elif action.action_type == ActionType.REMOVE_EDGE:
            new_graph = self._remove_edge(new_graph, action)
        elif action.action_type == ActionType.SET_SLOT:
            new_graph = self._set_slot(new_graph, action)
        elif action.action_type == ActionType.ADD_ROLE:
            new_graph = self._add_role(new_graph, action)
        elif action.action_type == ActionType.BIND:
            new_spec = self._add_binding(new_spec, action)
        elif action.action_type == ActionType.UNBIND:
            new_spec = self._remove_binding(new_spec, action)
        elif action.action_type == ActionType.REPLACE_SUBGRAPH:
            new_graph, new_spec = self._replace_subgraph(new_graph, new_spec, action)
        elif action.action_type == ActionType.COMPILE:
            compile_result, verified = self._compile_and_verify(new_graph, new_spec)
        elif action.action_type == ActionType.STOP:
            pass  # no-op, caller checks verified

        diff_pixels = self._compute_diff(compile_result)
        new_score = score_graph(new_graph, new_spec)

        # Extract diagnostic from compile failure if present
        diagnostic = None
        if isinstance(compile_result, CompileFailure) and compile_result.diagnostic is not None:
            diagnostic = compile_result.diagnostic

        self._state = EditState(
            graph=new_graph,
            specialization=new_spec,
            compile_result=compile_result,
            verified=verified,
            diff_pixels=diff_pixels,
            step=state.step + 1,
            score=new_score,
            history=state.history + [action],
            diagnostic=diagnostic,
        )
        return self._state

    # --- Graph edit primitives ---

    def _add_node(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if not action.node_id or action.node_id in graph.nodes:
            return graph
        op = action.value if isinstance(action.value, str) else "IDENTITY"
        new_node = GraphNode(id=action.node_id, op=op, inputs=())
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes,
            output_id=graph.output_id or action.node_id,
            metadata=graph.metadata,
        )

    def _remove_node(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        nodes = {k: v for k, v in graph.nodes.items() if k != action.node_id}
        # Remove edges pointing to the deleted node
        nodes = {
            k: GraphNode(
                id=v.id, op=v.op,
                inputs=tuple(i for i in v.inputs if i != action.node_id),
                output_type=v.output_type, roles=v.roles, slots=v.slots,
                description=v.description, evidence=v.evidence,
            )
            for k, v in nodes.items()
        }
        output_id = graph.output_id if graph.output_id != action.node_id else ""
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=output_id,
            metadata=graph.metadata,
        )

    def _set_node_op(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        old = graph.nodes[action.node_id]
        new_node = GraphNode(
            id=old.id, op=str(action.value), inputs=old.inputs,
            output_type=old.output_type, roles=old.roles, slots=old.slots,
            description=old.description, evidence=old.evidence,
        )
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=graph.output_id,
            metadata=graph.metadata,
        )

    def _add_edge(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        old = graph.nodes[action.node_id]
        if action.target_id in old.inputs:
            return graph  # already connected
        new_node = GraphNode(
            id=old.id, op=old.op,
            inputs=old.inputs + (action.target_id,),
            output_type=old.output_type, roles=old.roles, slots=old.slots,
            description=old.description, evidence=old.evidence,
        )
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=graph.output_id,
            metadata=graph.metadata,
        )

    def _remove_edge(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        old = graph.nodes[action.node_id]
        new_node = GraphNode(
            id=old.id, op=old.op,
            inputs=tuple(i for i in old.inputs if i != action.target_id),
            output_type=old.output_type, roles=old.roles, slots=old.slots,
            description=old.description, evidence=old.evidence,
        )
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=graph.output_id,
            metadata=graph.metadata,
        )

    def _set_slot(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        old = graph.nodes[action.node_id]
        new_slots = tuple(
            NodeSlot(name=s.name, typ=s.typ, constraint=s.constraint,
                     evidence=action.value if s.name == action.key else s.evidence)
            for s in old.slots
        )
        # If slot doesn't exist yet, add it
        if not any(s.name == action.key for s in old.slots):
            new_slots = new_slots + (
                NodeSlot(name=action.key, typ="ANY", evidence=action.value),
            )
        new_node = GraphNode(
            id=old.id, op=old.op, inputs=old.inputs,
            output_type=old.output_type, roles=old.roles, slots=new_slots,
            description=old.description, evidence=old.evidence,
        )
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=graph.output_id,
            metadata=graph.metadata,
        )

    def _add_role(self, graph: ComputationGraph, action: EditAction) -> ComputationGraph:
        if action.node_id not in graph.nodes:
            return graph
        old = graph.nodes[action.node_id]
        kind = str(action.value) if action.value else "UNKNOWN"
        new_role = RoleBinding(name=action.key, kind=kind)
        new_node = GraphNode(
            id=old.id, op=old.op, inputs=old.inputs,
            output_type=old.output_type,
            roles=old.roles + (new_role,),
            slots=old.slots,
            description=old.description, evidence=old.evidence,
        )
        nodes = dict(graph.nodes)
        nodes[action.node_id] = new_node
        return ComputationGraph(
            task_id=graph.task_id, nodes=nodes, output_id=graph.output_id,
            metadata=graph.metadata,
        )

    # --- Specialization edits ---

    def _add_binding(self, spec: Specialization, action: EditAction) -> Specialization:
        binding = ResolvedBinding(
            node_id=action.node_id, name=action.key,
            value=action.value, source="editor",
        )
        return Specialization(
            task_id=spec.task_id,
            bindings=spec.bindings + (binding,),
            metadata=spec.metadata,
        )

    def _remove_binding(self, spec: Specialization, action: EditAction) -> Specialization:
        new_bindings = tuple(
            b for b in spec.bindings
            if not (b.node_id == action.node_id and b.name == action.key)
        )
        return Specialization(
            task_id=spec.task_id,
            bindings=new_bindings,
            metadata=spec.metadata,
        )

    # --- Subgraph replacement ---

    def _replace_subgraph(
        self, graph: ComputationGraph, spec: Specialization, action: EditAction,
    ) -> tuple[ComputationGraph, Specialization]:
        """Replace blamed nodes with a typed fragment.

        action.value must be a GraphFragment.
        action.node_id is a comma-separated list of node ids to remove.
        The fragment is spliced in: its input_id connects to the
        predecessor of the first removed node, its output_id replaces
        references to the last removed node.
        """
        fragment = action.value
        if not isinstance(fragment, GraphFragment):
            return graph, spec

        remove_ids = set(action.node_id.split(",")) if action.node_id else set()
        if not remove_ids:
            return graph, spec

        # Find the predecessor and successor of the removed subgraph
        predecessor = None
        successor_refs: list[str] = []  # nodes that referenced removed nodes
        for nid in remove_ids:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            for inp in node.inputs:
                if inp not in remove_ids:
                    predecessor = inp
            for other_id, other in graph.nodes.items():
                if other_id not in remove_ids and nid in other.inputs:
                    successor_refs.append(other_id)

        # Also check if output_id points to a removed node
        old_output = graph.output_id
        new_output = old_output

        # Build new node dict: keep non-removed nodes, add fragment nodes
        nodes = {k: v for k, v in graph.nodes.items() if k not in remove_ids}

        # Remap fragment input connections
        for fid, fnode in fragment.nodes.items():
            inputs = tuple(
                (predecessor or "input") if inp == fragment.input_id else inp
                for inp in fnode.inputs
            )
            # If fragment.input_id is the node itself (self-referencing root),
            # wire the predecessor
            if fid == fragment.input_id:
                inputs = ((predecessor or "input"),)
            nodes[fid] = GraphNode(
                id=fid, op=fnode.op, inputs=inputs,
                output_type=fnode.output_type, roles=fnode.roles,
                slots=fnode.slots, description=fnode.description,
                evidence=dict(fnode.evidence),
            )

        # Remap successor references: replace removed node refs with fragment output
        for sid in successor_refs:
            if sid in nodes:
                old_node = nodes[sid]
                new_inputs = tuple(
                    fragment.output_id if inp in remove_ids else inp
                    for inp in old_node.inputs
                )
                nodes[sid] = GraphNode(
                    id=old_node.id, op=old_node.op, inputs=new_inputs,
                    output_type=old_node.output_type, roles=old_node.roles,
                    slots=old_node.slots, description=old_node.description,
                    evidence=dict(old_node.evidence),
                )

        # Remap output if it pointed to a removed node
        if old_output in remove_ids:
            new_output = fragment.output_id

        new_graph = ComputationGraph(
            task_id=graph.task_id,
            nodes=nodes,
            output_id=new_output,
            metadata=graph.metadata,
        )

        # Remove bindings for removed nodes
        new_bindings = tuple(
            b for b in spec.bindings if b.node_id not in remove_ids
        )
        new_spec = Specialization(
            task_id=spec.task_id,
            bindings=new_bindings,
            metadata=spec.metadata,
        )

        return new_graph, new_spec

    # --- Compile / verify ---

    def _compile_and_verify(
        self, graph: ComputationGraph, spec: Specialization,
    ) -> tuple[CompileResult, bool]:
        """Run the full compile + verify pipeline. This is the expensive step."""
        try:
            result = self.compiler.compile(graph, spec, self.examples)
        except Exception as e:
            return CompileFailure(task_id=self.task_id, reason=str(e)), False

        if not isinstance(result, CompileSuccess):
            return result, False

        try:
            vr = self.verifier.verify(result.program, self.examples)
            return result, vr.passed
        except Exception:
            return result, False

    def _compute_diff(self, compile_result: CompileResult | None) -> int:
        """Compute total pixel diff. Returns a large number if not compilable."""
        if compile_result is None or not isinstance(compile_result, CompileSuccess):
            # Estimate: total pixels across all demos
            total = 0
            for ex in self.examples:
                try:
                    total += ex.input.size
                except Exception:
                    total += 100
            return total

        # If we have a program, run it and measure diff
        try:
            from aria.runtime.executor import execute
            import numpy as np
            total_diff = 0
            for ex in self.examples:
                try:
                    predicted = execute(compile_result.program, ex.input)
                    total_diff += int(np.sum(predicted != ex.output))
                except Exception:
                    total_diff += ex.input.size
            return total_diff
        except Exception:
            return sum(ex.input.size for ex in self.examples)
