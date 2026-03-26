"""Small inspectable graph over stored leaves and admitted abstractions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aria.library.store import Library
from aria.program_store import ProgramStore, StoredProgram
from aria.runtime.program import program_to_text
from aria.types import LibraryEntry, Program


@dataclass(frozen=True)
class LeafNode:
    id: str
    program_text: str
    task_ids: tuple[str, ...]
    signatures: tuple[str, ...]
    sources: tuple[str, ...]
    use_count: int
    step_count: int

    @property
    def is_transfer_backed(self) -> bool:
        return len(self.task_ids) >= 2

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": "leaf",
            "program_text": self.program_text,
            "task_ids": list(self.task_ids),
            "signatures": list(self.signatures),
            "sources": list(self.sources),
            "use_count": self.use_count,
            "step_count": self.step_count,
            "is_transfer_backed": self.is_transfer_backed,
        }


@dataclass(frozen=True)
class AbstractionNode:
    id: str
    name: str
    program_text: str
    params: tuple[tuple[str, str], ...]
    return_type: str
    use_count: int
    support_task_ids: tuple[str, ...]
    support_program_count: int
    mdl_gain: int
    step_count: int

    @property
    def strength(self) -> str:
        """Classify as strong (genuinely reusable) or weak (narrow evidence)."""
        if len(self.support_task_ids) >= 2 and self.mdl_gain > 0:
            return "strong"
        return "weak"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": "abstraction",
            "name": self.name,
            "program_text": self.program_text,
            "params": [[name, typ] for name, typ in self.params],
            "return_type": self.return_type,
            "use_count": self.use_count,
            "support_task_ids": list(self.support_task_ids),
            "support_program_count": self.support_program_count,
            "mdl_gain": self.mdl_gain,
            "step_count": self.step_count,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind}


@dataclass(frozen=True)
class AbstractionGraph:
    leaves: tuple[LeafNode, ...]
    abstractions: tuple[AbstractionNode, ...]
    edges: tuple[GraphEdge, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "leaves": [node.to_dict() for node in self.leaves],
            "abstractions": [node.to_dict() for node in self.abstractions],
            "edges": [edge.to_dict() for edge in self.edges],
            "summary": graph_summary(self),
        }


def graph_summary(graph: AbstractionGraph) -> dict[str, object]:
    """Compute transfer-quality statistics from the graph."""
    total_leaves = len(graph.leaves)
    transfer_backed_leaves = sum(1 for l in graph.leaves if l.is_transfer_backed)
    one_off_leaves = total_leaves - transfer_backed_leaves

    total_abstractions = len(graph.abstractions)
    strong = [a for a in graph.abstractions if a.strength == "strong"]
    weak = [a for a in graph.abstractions if a.strength == "weak"]

    mdl_gains = [a.mdl_gain for a in graph.abstractions if a.mdl_gain > 0]
    support_counts = [a.support_program_count for a in graph.abstractions]
    task_counts = [len(a.support_task_ids) for a in graph.abstractions]

    def _safe_mean(vals: list[int | float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total_leaves": total_leaves,
        "transfer_backed_leaves": transfer_backed_leaves,
        "one_off_leaves": one_off_leaves,
        "total_abstractions": total_abstractions,
        "strong_abstractions": len(strong),
        "weak_abstractions": len(weak),
        "avg_mdl_gain": _safe_mean(mdl_gains),
        "avg_support_program_count": _safe_mean(support_counts),
        "avg_support_task_count": _safe_mean(task_counts),
        "total_edges": len(graph.edges),
    }


def build_abstraction_graph(
    program_store: ProgramStore,
    library: Library,
) -> AbstractionGraph:
    leaf_nodes = tuple(
        _leaf_from_record(record)
        for record in sorted(program_store.all_records(), key=lambda r: r.program_text)
    )
    leaf_by_task: dict[str, set[str]] = {}
    for leaf in leaf_nodes:
        for task_id in leaf.task_ids:
            leaf_by_task.setdefault(task_id, set()).add(leaf.id)

    abstraction_nodes = tuple(
        _abstraction_from_entry(entry)
        for entry in sorted(library.all_entries(), key=lambda e: e.name)
    )
    edge_keys: set[tuple[str, str, str]] = set()
    edges: list[GraphEdge] = []
    for abstraction in abstraction_nodes:
        for task_id in abstraction.support_task_ids:
            for leaf_id in sorted(leaf_by_task.get(task_id, ())):
                edge_key = (abstraction.id, leaf_id, "supported_by")
                if edge_key in edge_keys:
                    continue
                edge_keys.add(edge_key)
                edges.append(GraphEdge(src=abstraction.id, dst=leaf_id, kind="supported_by"))

    return AbstractionGraph(
        leaves=leaf_nodes,
        abstractions=abstraction_nodes,
        edges=tuple(edges),
    )


def _leaf_from_record(record: StoredProgram) -> LeafNode:
    return LeafNode(
        id=f"leaf:{_text_digest(record.program_text)}",
        program_text=record.program_text,
        task_ids=record.task_ids,
        signatures=record.signatures,
        sources=record.sources,
        use_count=record.use_count,
        step_count=record.step_count,
    )


def _abstraction_from_entry(entry: LibraryEntry) -> AbstractionNode:
    return AbstractionNode(
        id=f"abstraction:{entry.name}",
        name=entry.name,
        program_text=program_to_text(Program(steps=entry.steps, output=entry.output)),
        params=tuple((name, typ.name) for name, typ in entry.params),
        return_type=entry.return_type.name,
        use_count=entry.use_count,
        support_task_ids=entry.support_task_ids,
        support_program_count=entry.support_program_count,
        mdl_gain=entry.mdl_gain,
        step_count=len(entry.steps),
    )


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
