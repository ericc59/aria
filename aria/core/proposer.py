"""Compositional graph proposer — deterministic seed generation from library.

Constructs new ComputationGraph hypotheses from library fragments via
template reuse, parameterized variants, and sequential composition.
Part of the canonical architecture as a deterministic proposal source.

The proposer is NOT a learning system — it is a combinatorial enumerator
gated by exact verification. The future per-task recurrent editor
(aria.core.editor_env) will replace this as the primary hypothesis
generator for unsolved tasks.
"""

from __future__ import annotations

from typing import Any, Sequence

from aria.core.graph import (
    ComputationGraph,
    GraphNode,
    NodeSlot,
)
from aria.core.library import GraphLibrary, GraphTemplate


def propose_from_library(
    library: GraphLibrary,
    examples: Sequence[Any],
    task_id: str = "",
    *,
    max_proposals: int = 50,
) -> list[ComputationGraph]:
    """Propose computation graphs for a task using the template library.

    Proposal strategies (in order):
    1. Direct template reuse — re-specialize each template for new examples
    2. Parameterized variants — same structure, enumerated slot values
    3. Sequential composition — chain two single-node templates

    Returns a list of candidate graphs, NOT yet compiled or verified.
    The caller should specialize, compile, and verify each.
    """
    proposals: list[ComputationGraph] = []
    seen_structures: set[tuple[str, ...]] = set()

    # Strategy 1: Direct template reuse
    for template in library.templates:
        g = _adapt_template(template, task_id)
        key = _structure_key(g)
        if key not in seen_structures:
            seen_structures.add(key)
            proposals.append(g)
            if len(proposals) >= max_proposals:
                return proposals

    # Strategy 2: Parameterized variants of single-node templates
    for template in library.templates:
        for variant in _parameterized_variants(template, task_id, examples):
            key = _structure_key(variant)
            if key not in seen_structures:
                seen_structures.add(key)
                proposals.append(variant)
                if len(proposals) >= max_proposals:
                    return proposals

    # Strategy 3: Sequential composition of single-node templates
    single_node_templates = [
        t for t in library.templates if len(t.graph.nodes) <= 2
    ]
    for t1 in single_node_templates:
        for t2 in single_node_templates:
            composed = _compose_sequential(t1, t2, task_id)
            if composed is not None:
                key = _structure_key(composed)
                if key not in seen_structures:
                    seen_structures.add(key)
                    proposals.append(composed)
                    if len(proposals) >= max_proposals:
                        return proposals

    return proposals


def _adapt_template(template: GraphTemplate, task_id: str) -> ComputationGraph:
    """Create a fresh graph from a template, clearing task-specific evidence."""
    new_nodes = {}
    for nid, node in template.graph.nodes.items():
        # Keep structure (op, inputs, roles) but clear evidence
        new_nodes[nid] = GraphNode(
            id=nid,
            op=node.op,
            inputs=node.inputs,
            output_type=node.output_type,
            roles=node.roles,
            slots=tuple(
                NodeSlot(name=s.name, typ=s.typ, constraint=s.constraint, evidence=None)
                for s in node.slots
            ),
            description=node.description,
            evidence={},  # clear task-specific evidence
        )
    return ComputationGraph(
        task_id=task_id,
        nodes=new_nodes,
        output_id=template.graph.output_id,
        description=f"adapted from {template.source_task_id}",
        metadata={},
    )


def _parameterized_variants(
    template: GraphTemplate,
    task_id: str,
    examples: Sequence[Any],
) -> list[ComputationGraph]:
    """Generate parameterized variants of a template.

    For templates with slots (e.g., rotate by ?degrees), enumerate
    plausible values based on the domain.
    """
    variants = []

    for nid, node in template.graph.nodes.items():
        for slot in node.slots:
            values = _enumerate_slot_values(slot, examples)
            for val in values:
                # Create a variant with this specific slot value
                new_nodes = dict(template.graph.nodes)
                new_slots = tuple(
                    NodeSlot(
                        name=s.name, typ=s.typ,
                        constraint=s.constraint,
                        evidence=val if s.name == slot.name else s.evidence,
                    )
                    for s in node.slots
                )
                new_evidence = dict(node.evidence)
                # Propagate the slot value to evidence for specialization
                if slot.name == "degrees":
                    new_evidence["transform"] = "rotate"
                    new_evidence["degrees"] = val
                elif slot.name == "axis":
                    new_evidence["transform"] = "reflect"
                    new_evidence["axis"] = val
                elif slot.name == "fill_color":
                    new_evidence["transform"] = "fill_enclosed"
                    new_evidence["fill_color"] = val

                new_nodes[nid] = GraphNode(
                    id=nid, op=node.op, inputs=node.inputs,
                    output_type=node.output_type, roles=node.roles,
                    slots=new_slots, description=node.description,
                    evidence=new_evidence,
                )
                variants.append(ComputationGraph(
                    task_id=task_id,
                    nodes=new_nodes,
                    output_id=template.graph.output_id,
                    description=f"variant of {template.source_task_id}: {slot.name}={val}",
                    metadata={},
                ))

    return variants


def _enumerate_slot_values(slot: NodeSlot, examples: Sequence[Any]) -> list[Any]:
    """Enumerate plausible values for a slot based on its type."""
    if slot.typ == "INT":
        if slot.name == "degrees":
            return [90, 180, 270]
        if slot.name == "factor":
            return [2, 3, 4, 5]
        if slot.name == "fill_color":
            return list(range(10))
        return [1, 2, 3, 4, 5]
    if slot.typ == "AXIS":
        return ["row", "col"]
    if slot.typ == "COLOR":
        return list(range(10))
    if slot.typ == "TRANSFORM":
        return []  # too open-ended for enumeration
    return []


def _compose_sequential(
    t1: GraphTemplate,
    t2: GraphTemplate,
    task_id: str,
) -> ComputationGraph | None:
    """Compose two templates sequentially: output of t1 feeds into t2.

    Only works when both are single-output, grid→grid transformations.
    """
    # Find the "action" nodes (skip BIND_ROLE setup nodes)
    action1 = _find_action_node(t1.graph)
    action2 = _find_action_node(t2.graph)
    if action1 is None or action2 is None:
        return None
    if action1.op == action2.op and action1.evidence == action2.evidence:
        return None  # skip identity composition

    # Build composed graph: step1 takes input, step2 takes step1's output
    node1 = GraphNode(
        id="step1",
        op=action1.op,
        inputs=("input",),
        output_type=action1.output_type,
        roles=action1.roles,
        slots=action1.slots,
        description=action1.description,
        evidence=dict(action1.evidence),
    )
    node2 = GraphNode(
        id="step2",
        op=action2.op,
        inputs=("step1",),
        output_type=action2.output_type,
        roles=action2.roles,
        slots=action2.slots,
        description=action2.description,
        evidence=dict(action2.evidence),
    )

    return ComputationGraph(
        task_id=task_id,
        nodes={"step1": node1, "step2": node2},
        output_id="step2",
        description=(
            f"composition: {action1.op}({action1.evidence.get('transform','')}) → "
            f"{action2.op}({action2.evidence.get('transform','')})"
        ),
        metadata={},
    )


def _find_action_node(graph: ComputationGraph) -> GraphNode | None:
    """Find the primary action node in a graph (skip BIND_ROLE/setup)."""
    for nid in reversed(list(graph.topo_order())):
        node = graph.nodes[nid]
        if node.op not in ("BIND_ROLE", "IDENTIFY_ROLES"):
            return node
    return None


def _structure_key(graph: ComputationGraph) -> tuple[str, ...]:
    """A hashable key capturing the graph's structural identity."""
    parts = []
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        ev_key = tuple(sorted(
            (k, str(v)) for k, v in node.evidence.items()
        ))
        parts.append((node.op, node.inputs, ev_key))
    return tuple(str(p) for p in parts)
