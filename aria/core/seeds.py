"""Seed collection — gather initial graph candidates for editing.

Collects (ComputationGraph, Specialization | None, provenance) triples
from deterministic sources: direct fitters, library proposer, and
a minimal template seed. These serve as starting points for the
graph editor search.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from aria.core.graph import (
    CompileSuccess,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RoleBinding,
    Specialization,
)
from aria.core.library import GraphLibrary
from aria.core.proposer import propose_from_library
from aria.core.protocol import Compiler, Fitter, Specializer, Verifier


@dataclass(frozen=True)
class Seed:
    """One candidate starting point for graph editing."""
    graph: ComputationGraph
    specialization: Specialization | None
    provenance: str               # "fitter", "library", "template", etc.
    already_verified: bool = False # True if this seed already passes verification


def collect_seeds(
    examples: Sequence[Any],
    fitter: Fitter,
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    task_id: str = "",
    library: GraphLibrary | None = None,
    max_library_proposals: int = 20,
    include_templates: bool = True,
) -> list[Seed]:
    """Collect seeds from all deterministic sources.

    Order:
    1. Direct fitter proposals (specialized + compile-checked)
    2. Library proposals (if library provided)
    3. Minimal template seeds (common graph shapes)

    Seeds that already pass verification are marked as such.
    """
    seeds: list[Seed] = []
    seen_structures: set[tuple[str, ...]] = set()

    # --- Source 1: Direct fitters ---
    try:
        graphs = fitter.fit(examples, task_id=task_id)
    except Exception:
        graphs = []

    for graph in graphs:
        key = _structure_key(graph)
        if key in seen_structures:
            continue
        seen_structures.add(key)

        try:
            spec = specializer.specialize(graph, examples)
        except Exception:
            spec = Specialization(task_id=task_id, bindings=())

        # Check if it already compiles + verifies
        verified = False
        try:
            result = compiler.compile(graph, spec, examples)
            if isinstance(result, CompileSuccess) and result.scope == "task":
                vr = verifier.verify(result.program, examples)
                verified = vr.passed
        except Exception:
            pass

        seeds.append(Seed(
            graph=graph,
            specialization=spec,
            provenance="fitter",
            already_verified=verified,
        ))

    # --- Source 2: Library proposals ---
    if library is not None and library.size > 0:
        try:
            proposals = propose_from_library(
                library, examples, task_id=task_id,
                max_proposals=max_library_proposals,
            )
        except Exception:
            proposals = []

        for graph in proposals:
            key = _structure_key(graph)
            if key in seen_structures:
                continue
            seen_structures.add(key)

            try:
                spec = specializer.specialize(graph, examples)
            except Exception:
                spec = Specialization(task_id=task_id, bindings=())

            verified = False
            try:
                result = compiler.compile(graph, spec, examples)
                if isinstance(result, CompileSuccess) and result.scope == "task":
                    vr = verifier.verify(result.program, examples)
                    verified = vr.passed
            except Exception:
                pass

            seeds.append(Seed(
                graph=graph,
                specialization=spec,
                provenance="library",
                already_verified=verified,
            ))

    # --- Source 3: Minimal template seeds ---
    if include_templates:
        for tmpl_graph in _minimal_templates(task_id):
            key = _structure_key(tmpl_graph)
            if key in seen_structures:
                continue
            seen_structures.add(key)
            seeds.append(Seed(
                graph=tmpl_graph,
                specialization=None,
                provenance="template",
                already_verified=False,
            ))

    return seeds


def _minimal_templates(task_id: str) -> list[ComputationGraph]:
    """Generate a small set of common graph shape templates.

    These are empty-evidence graphs that the editor can populate.
    They provide structural starting points, not solutions.
    """
    templates = []

    # Single transform node
    templates.append(ComputationGraph(
        task_id=task_id,
        nodes={
            "roles": GraphNode(
                id="roles", op="BIND_ROLE", inputs=("input",),
                roles=(RoleBinding(name="bg", kind="BG"),),
            ),
            "transform": GraphNode(
                id="transform", op="APPLY_TRANSFORM", inputs=("roles",),
                slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            ),
        },
        output_id="transform",
        description="single transform template",
    ))

    # Two-step: transform then transform
    templates.append(ComputationGraph(
        task_id=task_id,
        nodes={
            "roles": GraphNode(
                id="roles", op="BIND_ROLE", inputs=("input",),
                roles=(RoleBinding(name="bg", kind="BG"),),
            ),
            "step1": GraphNode(
                id="step1", op="APPLY_TRANSFORM", inputs=("roles",),
                slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            ),
            "step2": GraphNode(
                id="step2", op="APPLY_TRANSFORM", inputs=("step1",),
                slots=(NodeSlot(name="transform", typ="TRANSFORM"),),
            ),
        },
        output_id="step2",
        description="two-step transform template",
    ))

    return templates


def _structure_key(graph: ComputationGraph) -> tuple[str, ...]:
    """Hashable key for deduplication."""
    parts = []
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        ev = tuple(sorted((k, str(v)) for k, v in node.evidence.items()))
        parts.append(f"{node.op}|{node.inputs}|{ev}")
    return tuple(parts)
