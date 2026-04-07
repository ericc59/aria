"""Graph template library — stores verified computation graphs for reuse.

Part of the canonical architecture. Stores winning (graph, specialization)
pairs as templates for the compositional proposer. This is a deterministic
accumulator, not a learning system — it grows as fitters solve tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aria.core.graph import (
    ComputationGraph,
    CompileSuccess,
    GraphNode,
    NodeSlot,
    Specialization,
)


@dataclass(frozen=True)
class GraphTemplate:
    """A verified computation graph stored for reuse."""
    graph: ComputationGraph
    specialization: Specialization
    source_task_id: str
    # Structural features for matching
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeTemplate:
    """A single node extracted from a verified graph, for compositional reuse."""
    node: GraphNode
    source_task_id: str
    source_graph_op_set: frozenset[str]


class GraphLibrary:
    """Growing library of verified graph templates.

    The library is the system's memory.  It stores full graph templates
    and individual node templates.  The compositional proposer draws
    from both to construct new hypotheses.
    """

    def __init__(self) -> None:
        self._templates: list[GraphTemplate] = []
        self._node_templates: dict[str, list[NodeTemplate]] = {}  # op -> templates

    def add(
        self,
        graph: ComputationGraph,
        specialization: Specialization,
        source_task_id: str,
        features: dict[str, Any] | None = None,
    ) -> None:
        """Store a verified graph template."""
        self._templates.append(GraphTemplate(
            graph=graph,
            specialization=specialization,
            source_task_id=source_task_id,
            features=features or {},
        ))
        # Extract individual nodes for compositional reuse
        for node in graph.nodes.values():
            nt = NodeTemplate(
                node=node,
                source_task_id=source_task_id,
                source_graph_op_set=graph.op_set,
            )
            self._node_templates.setdefault(node.op, []).append(nt)

    @property
    def size(self) -> int:
        return len(self._templates)

    @property
    def templates(self) -> list[GraphTemplate]:
        return list(self._templates)

    @property
    def known_ops(self) -> frozenset[str]:
        """All abstract ops seen in verified templates."""
        ops: set[str] = set()
        for t in self._templates:
            ops.update(t.graph.op_set)
        return frozenset(ops)

    def templates_with_op(self, op: str) -> list[GraphTemplate]:
        """Find templates that contain a specific op."""
        return [t for t in self._templates if op in t.graph.op_set]

    def unique_graph_structures(self) -> list[frozenset[str]]:
        """Deduplicated list of op-set structures in the library."""
        seen: set[frozenset[str]] = set()
        result = []
        for t in self._templates:
            s = t.graph.op_set
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result
