"""Multi-step graph-edit search over the canonical editor environment.

Bounded best-first search with explicit frontier, state deduplication,
and separated cheap expansion from expensive compile/verify.

Search algorithm:
  1. Seed the frontier with initial states from collected seeds
  2. Pop the best-scoring frontier state
  3. Enumerate legal edits (cheap structural mutations)
  4. Apply each edit, hash the resulting state for dedup
  5. Add unseen children to the frontier
  6. Periodically compile/verify the most promising frontier states
  7. Stop on verification success or budget exhaustion

State deduplication uses a canonical hash of (sorted graph nodes, sorted
specialization bindings) so structurally identical states reached via
different edit paths are only explored once.

Part of the canonical architecture.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Sequence

from aria.core.editor_env import (
    ActionType,
    EditAction,
    EditState,
    GraphEditEnv,
    score_graph,
)
from aria.core.graph import (
    CompileSuccess,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RoleBinding,
    Specialization,
)
from aria.core.protocol import Compiler, Specializer, Verifier
from aria.core.seeds import Seed


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditSearchResult:
    """Outcome of multi-step graph-edit search on one task."""
    solved: bool
    program: Any | None = None
    best_score: float = float("inf")
    seeds_tried: int = 0
    seed_provenance: str = ""
    description: str = ""
    # Search diagnostics
    frontier_expansions: int = 0
    compiles_attempted: int = 0
    unique_states_seen: int = 0
    max_depth_reached: int = 0
    winning_edit_depth: int = 0
    top_lane: str = ""  # lane evidence used for search ordering


# ---------------------------------------------------------------------------
# Known ARC ops and values for edit enumeration
# ---------------------------------------------------------------------------

_ARC_OPS = (
    "BIND_ROLE", "APPLY_TRANSFORM", "REPAIR_LINES", "REPAIR_2D_MOTIF",
    "CONSTRUCT_CANVAS", "APPLY_RELATION",
)

_TRANSFORM_VALUES = (
    "rotate 90 degrees", "rotate 180 degrees", "rotate 270 degrees",
    "reflect across row axis", "reflect across col axis",
    "transpose grid",
)

_SLOT_EVIDENCE_OPTIONS: dict[str, list[Any]] = {
    "transform": list(_TRANSFORM_VALUES),
    "degrees": [90, 180, 270],
    "axis": ["row", "col"],
    "fill_color": list(range(10)),
    "factor": [2, 3],
    "strategy": ["tile", "upscale"],
}


# ---------------------------------------------------------------------------
# State hashing for deduplication
# ---------------------------------------------------------------------------


def _state_hash(graph: ComputationGraph, spec: Specialization) -> int:
    """Canonical hash of (graph structure, specialization bindings).

    Two states with identical graph topology, node ops, slot evidence,
    and specialization bindings hash to the same value regardless of
    the edit path that produced them.
    """
    parts: list[str] = []

    # Graph: sorted by node id for canonical ordering
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        slots_str = "|".join(
            f"{s.name}={s.typ}={s.evidence}" for s in sorted(node.slots, key=lambda s: s.name)
        )
        roles_str = "|".join(
            f"{r.name}={r.kind}" for r in sorted(node.roles, key=lambda r: r.name)
        )
        parts.append(
            f"N:{nid}:{node.op}:{sorted(node.inputs)}:[{slots_str}]:[{roles_str}]"
        )

    parts.append(f"OUT:{graph.output_id}")

    # Specialization: sorted by (node_id, name)
    for b in sorted(spec.bindings, key=lambda b: (b.node_id, b.name)):
        parts.append(f"B:{b.node_id}:{b.name}={b.value}")

    return hash(tuple(parts))


# ---------------------------------------------------------------------------
# Edit enumeration
# ---------------------------------------------------------------------------


def _enumerate_edits(state: EditState, top_lane: str = "") -> list[EditAction]:
    """Generate candidate edits, prioritized by lane relevance.

    Order:
    1. Subgraph replacements from diagnostics (structural repair)
    2. Parameter repair hints from diagnostics
    3. Lane-local edits (bindings aligned with top lane)
    4. Cross-lane edits (generic mutations)
    """
    edits: list[EditAction] = []
    graph = state.graph

    # 0a. Subgraph replacement actions (highest priority — structural repair)
    if state.diagnostic is not None:
        for blame in getattr(state.diagnostic, 'subgraph_blames', ()):
            for fragment in getattr(state.diagnostic, 'replacement_fragments', ()):
                if fragment.label in blame.replacement_labels:
                    edits.append(EditAction(
                        action_type=ActionType.REPLACE_SUBGRAPH,
                        node_id=",".join(blame.node_ids),
                        value=fragment,
                    ))

    # 0b. Diagnostic-guided parameter repair actions
    if state.diagnostic is not None:
        for hint in state.diagnostic.repair_hints:
            for alt in hint.alternatives:
                edits.append(EditAction(
                    action_type=ActionType.BIND,
                    node_id=hint.node_id,
                    key=hint.binding_name,
                    value=alt,
                ))

    # 1. Change op on existing nodes (skip BIND_ROLE nodes)
    for nid, node in graph.nodes.items():
        if node.op == "BIND_ROLE":
            continue
        for op in _ARC_OPS:
            if op != node.op and op != "BIND_ROLE":
                edits.append(EditAction(
                    action_type=ActionType.SET_NODE_OP,
                    node_id=nid, value=op,
                ))

    # 2. Set slot evidence on existing nodes
    for nid, node in graph.nodes.items():
        for slot in node.slots:
            options = _SLOT_EVIDENCE_OPTIONS.get(slot.name, [])
            for val in options:
                if val != slot.evidence:
                    edits.append(EditAction(
                        action_type=ActionType.SET_SLOT,
                        node_id=nid, key=slot.name, value=val,
                    ))

    # 3. Add transform slot to APPLY_TRANSFORM nodes missing one
    for nid, node in graph.nodes.items():
        if node.op == "APPLY_TRANSFORM":
            has_transform = any(s.name == "transform" for s in node.slots)
            if not has_transform:
                for val in _TRANSFORM_VALUES:
                    edits.append(EditAction(
                        action_type=ActionType.SET_SLOT,
                        node_id=nid, key="transform", value=val,
                    ))

    # 4. Specialization bindings from node evidence
    for nid, node in graph.nodes.items():
        for key, val in node.evidence.items():
            existing = state.specialization.get(nid, key)
            if existing != val:
                edits.append(EditAction(
                    action_type=ActionType.BIND,
                    node_id=nid, key=key, value=val,
                ))

    # 5. Task-level bindings for movement/transform evidence
    for nid, node in graph.nodes.items():
        for key in ("strategy", "transform", "degrees", "axis", "direction",
                     "dr", "dc", "fill_color"):
            val = node.evidence.get(key)
            if val is not None:
                for ns in ("__movement__", "__grid_transform__"):
                    if state.specialization.get(ns, key) is None:
                        edits.append(EditAction(
                            action_type=ActionType.BIND,
                            node_id=ns, key=key, value=val,
                        ))

    # Sort: residual-prior + lane-local edits first, cross-lane edits last
    if top_lane:
        # Derive residual category from diagnostic if available
        residual_cat = ""
        if state.diagnostic is not None:
            from aria.core.lane_coverage import classify_residual
            residual_cat = classify_residual(
                state.diagnostic.total_diff,
                max(state.diff_pixels, 1),
            )
        edits = _sort_by_lane_relevance(edits, top_lane, residual_cat)

    return edits


# Lane-local binding namespaces for each lane
_LANE_NAMESPACES = {
    "replication": {"__replicate__", "__task__"},
    "relocation": {"__placement__", "__movement__"},
    "periodic_repair": {"__task__", "__periodic__"},
    "grid_transform": {"__grid_transform__", "__task__"},
}

# Ops characteristic of each lane
_LANE_OPS = {
    "replication": {"SELECT_SUBSET", "APPLY_RELATION", "PAINT"},
    "relocation": {"SELECT_SUBSET", "APPLY_RELATION", "PAINT", "APPLY_TRANSFORM"},
    "periodic_repair": {"REPAIR_LINES", "REPAIR_2D_MOTIF", "REPAIR_MISMATCH"},
    "grid_transform": {"APPLY_TRANSFORM"},
}


def _sort_by_lane_relevance(
    edits: list[EditAction],
    top_lane: str,
    residual_category: str = "",
) -> list[EditAction]:
    """Sort edits by lane relevance + residual prior.

    Priority (lower = tried first):
      0: diagnostic replacements
      1: residual-prior-preferred actions
      2: lane-local bindings
      3: lane-local op changes
      4: other bindings
      5: generic mutations
    """
    namespaces = _LANE_NAMESPACES.get(top_lane, set())
    ops = _LANE_OPS.get(top_lane, set())

    # Get residual prior if available
    prior_types: set[ActionType] = set()
    prior_ns: set[str] = set()
    if residual_category:
        from aria.core.residual_priors import get_edit_prior
        prior = get_edit_prior(residual_category, top_lane)
        prior_types = set(prior.action_types)
        prior_ns = set(prior.binding_namespaces)

    def _relevance(edit: EditAction) -> int:
        if edit.action_type == ActionType.REPLACE_SUBGRAPH:
            return 0
        # Residual-prior match
        if edit.action_type in prior_types:
            if edit.action_type == ActionType.BIND and prior_ns and edit.node_id in prior_ns:
                return 1  # residual-guided + namespace match
            if edit.action_type != ActionType.BIND:
                return 1  # residual-guided non-bind action
        # Lane-local
        if edit.action_type == ActionType.BIND and edit.node_id in namespaces:
            return 2
        if edit.action_type == ActionType.SET_NODE_OP and isinstance(edit.value, str) and edit.value in ops:
            return 3
        if edit.action_type == ActionType.BIND:
            return 4
        return 5

    return sorted(edits, key=_relevance)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _combined_score(state: EditState) -> float:
    """Priority score: lower is better.

    verified (-1e9 + MDL) > compiled (-1000 + diff + MDL) > diagnostic-guided (-500 + MDL) > uncompiled (diff + MDL)
    """
    if state.verified:
        return -1e9 + state.score

    compile_bonus = 0.0
    if state.compile_result is not None and isinstance(state.compile_result, CompileSuccess):
        compile_bonus = -1000.0

    # Boost states that resulted from diagnostic-guided actions
    diagnostic_bonus = 0.0
    if state.history:
        last = state.history[-1]
        if last.action_type == ActionType.REPLACE_SUBGRAPH:
            diagnostic_bonus = -500.0  # strongly prefer replacement states for compilation

    return state.diff_pixels * 10.0 + state.score + compile_bonus + diagnostic_bonus


# ---------------------------------------------------------------------------
# Frontier entry
# ---------------------------------------------------------------------------


_entry_counter = 0  # tiebreaker for heap ordering


@dataclass(order=True)
class _FrontierEntry:
    priority: float
    counter: int = field(compare=True)  # tiebreaker: FIFO among equal priority
    depth: int = field(compare=False)
    state: EditState = field(compare=False)
    provenance: str = field(compare=False)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_from_seeds(
    seeds: list[Seed],
    examples: Sequence[Any],
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    task_id: str = "",
    max_depth: int = 4,
    max_frontier: int = 200,
    max_total_compiles: int = 40,
    max_expansions: int = 300,
) -> EditSearchResult:
    """Lane-aware multi-step best-first search over graph edits.

    Uses mechanism evidence to:
    1. Prioritize seeds from the top-ranked lane
    2. Order edits so lane-local actions come before cross-lane mutations
    3. Allocate compile budget toward the most promising lane

    Deduplicates states across all seeds via canonical hashing.
    """
    global _entry_counter

    if not seeds:
        return EditSearchResult(solved=False)

    # Compute lane evidence once for the whole search
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    evidence, ranking = compute_evidence_and_rank(examples)
    top_lane = ranking.lanes[0].name if ranking.lanes else ""

    # Shared environment
    env = GraphEditEnv(
        examples=examples,
        specializer=specializer,
        compiler=compiler,
        verifier=verifier,
        task_id=task_id,
    )

    visited: set[int] = set()
    frontier: list[_FrontierEntry] = []
    total_compiles = 0
    total_expansions = 0
    max_depth_seen = 0
    seeds_tried = 0

    # --- Seed the frontier (lane-aware ordering) ---
    # Prioritize seeds whose graph ops align with the top lane
    lane_ops = _LANE_OPS.get(top_lane, set())
    def _seed_relevance(s: Seed) -> int:
        ops = s.graph.op_set if hasattr(s.graph, 'op_set') else set()
        overlap = len(ops & lane_ops)
        return -overlap  # more overlap = lower priority number = first
    sorted_seeds = sorted(seeds, key=_seed_relevance)

    for seed in sorted_seeds:
        seeds_tried += 1
        state = env.reset(initial_graph=seed.graph)

        # Apply seed specialization bindings (cheap)
        if seed.specialization is not None:
            for binding in seed.specialization.bindings:
                state = env.step(EditAction(
                    action_type=ActionType.BIND,
                    node_id=binding.node_id,
                    key=binding.name,
                    value=binding.value,
                ))

        h = _state_hash(state.graph, state.specialization)
        if h in visited:
            continue
        visited.add(h)

        # Fast path: seed already verified by collect_seeds
        if seed.already_verified:
            env._state = state
            state = env.step(EditAction(action_type=ActionType.COMPILE))
            total_compiles += 1
            if state.verified:
                prog = _extract_program(state)
                if prog is not None:
                    return EditSearchResult(
                        solved=True,
                        program=prog,
                        best_score=state.score,
                        seeds_tried=seeds_tried,
                        seed_provenance=seed.provenance,
                        description=f"seed verified ({seed.provenance})",
                        frontier_expansions=0,
                        compiles_attempted=total_compiles,
                        unique_states_seen=len(visited),
                        max_depth_reached=0,
                        winning_edit_depth=0,
                    )

        _entry_counter += 1
        heapq.heappush(frontier, _FrontierEntry(
            priority=_combined_score(state),
            counter=_entry_counter,
            depth=0,
            state=state,
            provenance=seed.provenance,
        ))

    # --- Main search loop ---
    while frontier and total_expansions < max_expansions:
        entry = heapq.heappop(frontier)
        state = entry.state
        depth = entry.depth

        max_depth_seen = max(max_depth_seen, depth)

        # --- Compile if not yet compiled (deferred from child creation) ---
        if state.compile_result is None and total_compiles < max_total_compiles:
            env._state = state
            state = env.step(EditAction(action_type=ActionType.COMPILE))
            total_compiles += 1

            if state.verified:
                prog = _extract_program(state)
                if prog is not None:
                    return EditSearchResult(
                        solved=True,
                        program=prog,
                        best_score=state.score,
                        seeds_tried=seeds_tried,
                        seed_provenance=entry.provenance,
                        description=f"verified at depth {depth}",
                        frontier_expansions=total_expansions,
                        compiles_attempted=total_compiles,
                        unique_states_seen=len(visited),
                        max_depth_reached=max_depth_seen,
                        winning_edit_depth=depth,
                    )

        # --- Expand: enumerate children if within depth budget ---
        if depth >= max_depth:
            continue

        total_expansions += 1
        child_edits = _enumerate_edits(state, top_lane=top_lane)

        for edit in child_edits:
            if len(frontier) >= max_frontier:
                break

            # Apply the edit (cheap — no compile)
            env._state = state
            child_state = env.step(edit)

            # Dedup
            child_hash = _state_hash(child_state.graph, child_state.specialization)
            if child_hash in visited:
                continue
            visited.add(child_hash)

            # Eagerly compile structural replacement children
            if (edit.action_type == ActionType.REPLACE_SUBGRAPH
                    and total_compiles < max_total_compiles):
                env._state = child_state
                child_state = env.step(EditAction(action_type=ActionType.COMPILE))
                total_compiles += 1
                if child_state.verified:
                    prog = _extract_program(child_state)
                    if prog is not None:
                        return EditSearchResult(
                            solved=True,
                            program=prog,
                            best_score=child_state.score,
                            seeds_tried=seeds_tried,
                            seed_provenance=entry.provenance,
                            description=f"structural replacement at depth {depth + 1}",
                            frontier_expansions=total_expansions,
                            compiles_attempted=total_compiles,
                            unique_states_seen=len(visited),
                            max_depth_reached=max_depth_seen,
                            winning_edit_depth=depth + 1,
                        )

            _entry_counter += 1
            heapq.heappush(frontier, _FrontierEntry(
                priority=_combined_score(child_state),
                counter=_entry_counter,
                depth=depth + 1,
                state=child_state,
                provenance=entry.provenance,
            ))

    return EditSearchResult(
        solved=False,
        best_score=float("inf"),
        seeds_tried=seeds_tried,
        description="no verified program found",
        frontier_expansions=total_expansions,
        compiles_attempted=total_compiles,
        unique_states_seen=len(visited),
        max_depth_reached=max_depth_seen,
        top_lane=top_lane,
    )


def _extract_program(state: EditState) -> Any | None:
    """Extract the program from a verified state."""
    if state.compile_result is not None and isinstance(state.compile_result, CompileSuccess):
        return state.compile_result.program
    return None
