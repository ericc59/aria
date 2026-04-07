"""Domain-general computation graph IR for few-shot program synthesis.

The key idea: a hypothesis about *how* to solve a task is represented as
a typed DAG of abstract operations, not as a concrete program.  The graph
captures the *shape* of the computation; a separate specialization pass
resolves task-specific parameters from example evidence.

This separation — graph structure vs. resolved bindings — is the core
abstraction.  It enables:
  - structural matching: "this task needs select → transform → paint"
  - partial specialization: resolve what you can, leave the rest as slots
  - compositional generalization: new graphs from known node types
  - honest failure: if specialization can't resolve a slot, say so

The types here are domain-independent.  A domain instantiation provides:
  - a set of node types (primitives / abstract operations)
  - a type system for edges and slots
  - concrete ops that nodes compile to
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Graph IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSlot:
    """A parameter slot on a graph node — to be resolved by specialization."""
    name: str
    typ: str              # domain-specific type name (e.g. "INT", "COLOR", "AXIS")
    constraint: str = ""  # human-readable constraint
    evidence: Any = None  # observed value(s) from examples, if available

    def __repr__(self) -> str:
        return f"?{self.name}:{self.typ}"


@dataclass(frozen=True)
class RoleBinding:
    """A symbolic role reference on a graph node."""
    name: str             # unique within the node
    kind: str             # structural role (e.g. "BG", "ANCHOR", "TARGET")
    description: str = ""

    def __repr__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class GraphNode:
    """One node in a computation graph hypothesis.

    Domain-independent: the 'op' field is a string naming an abstract
    operation.  The domain's compiler maps it to concrete runtime calls.
    """
    id: str                                   # unique within the graph
    op: str                                   # abstract operation name
    inputs: tuple[str, ...] = ()              # ids of predecessor nodes
    output_type: str = ""                     # what this node produces
    roles: tuple[RoleBinding, ...] = ()       # symbolic bindings
    slots: tuple[NodeSlot, ...] = ()          # parameters to resolve
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputationGraph:
    """A hypothesis expressed as a typed DAG of abstract operations.

    This is the central representation.  It is:
    - Small (2–8 nodes typically)
    - Inspectable (explicit edges, named slots, structural roles)
    - Compilable (a domain compiler maps it to executable programs)
    - Not executable (it has unresolved slots and abstract ops)
    """
    task_id: str
    nodes: dict[str, GraphNode]
    output_id: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def topo_order(self) -> tuple[str, ...]:
        """Topological sort of node ids (inputs before dependents)."""
        visited: set[str] = set()
        order: list[str] = []

        def _visit(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                return
            for dep in node.inputs:
                _visit(dep)
            order.append(nid)

        for nid in self.nodes:
            _visit(nid)
        return tuple(order)

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        """Direct input dependencies of a node."""
        node = self.nodes.get(node_id)
        return node.inputs if node else ()

    def successors(self, node_id: str) -> tuple[str, ...]:
        """Nodes that depend on *node_id*."""
        return tuple(
            nid for nid, n in self.nodes.items()
            if node_id in n.inputs
        )

    def validate(self) -> list[str]:
        """Structural validation — empty list means valid."""
        errors: list[str] = []
        ids = set(self.nodes.keys())
        for nid, node in self.nodes.items():
            for dep in node.inputs:
                if dep != "input" and dep not in ids:
                    errors.append(f"node {nid} depends on undefined {dep}")
        if self.output_id not in ids and self.output_id != "input":
            errors.append(f"output_id {self.output_id} not in graph")
        try:
            self.topo_order()
        except RecursionError:
            errors.append("graph contains a cycle")
        return errors

    @property
    def op_set(self) -> frozenset[str]:
        """Set of all abstract operation names in the graph."""
        return frozenset(n.op for n in self.nodes.values())

    @property
    def role_kinds(self) -> frozenset[str]:
        """Set of all role kinds across all nodes."""
        return frozenset(r.kind for n in self.nodes.values() for r in n.roles)


# ---------------------------------------------------------------------------
# Specialization — resolved static task structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBinding:
    """A single resolved binding: a slot or role pinned to a concrete value."""
    node_id: str          # which graph node (or "__task__" for global)
    name: str             # slot or role name
    value: Any            # resolved concrete value
    source: str = ""      # provenance: "evidence", "consensus", "inferred", etc.


@dataclass(frozen=True)
class Specialization:
    """Resolved static task structure extracted from example evidence.

    The symbolic analogue of "baking static structure into weights."
    Captures everything that is constant across examples so the compiler
    can use concrete values instead of searching.
    """
    task_id: str
    bindings: tuple[ResolvedBinding, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, node_id: str, name: str) -> Any | None:
        """Look up a resolved binding by node and name."""
        for b in self.bindings:
            if b.node_id == node_id and b.name == name:
                return b.value
        return None

    def bindings_for_node(self, node_id: str) -> tuple[ResolvedBinding, ...]:
        """All bindings for a specific graph node."""
        return tuple(b for b in self.bindings if b.node_id == node_id)

    @property
    def binding_names(self) -> frozenset[str]:
        """All bound names across all nodes."""
        return frozenset(b.name for b in self.bindings)


# ---------------------------------------------------------------------------
# Compilation protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileSuccess:
    """A graph that compiled into an executable program."""
    task_id: str
    program: Any          # domain-specific program type
    bindings_used: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    scope: str = "task"   # "task" (one program for all examples) or "per_example"


@dataclass(frozen=True)
class RepairHint:
    """One actionable repair suggestion from near-miss diagnostics.

    Tells the editor: "try changing this binding to one of these values."
    """
    node_id: str          # graph node or "__task__" for global bindings
    binding_name: str     # which binding to change
    current_value: Any    # what it is now
    alternatives: tuple   # bounded set of values to try
    confidence: float = 0.0  # 0-1, how likely this fixes the problem
    reason: str = ""      # why this was suggested


@dataclass(frozen=True)
class RegionResidual:
    """Localized residual for one spatial region of the grid."""
    region_label: str     # "frame", "interior", "cell_0", "full_grid", etc.
    diff_pixels: int
    total_pixels: int
    diff_fraction: float  # diff_pixels / total_pixels


@dataclass(frozen=True)
class ResidualPattern:
    """Characterization of the residual's spatial structure.

    Inferred from the diff mask. Used to condition dynamic fragment
    generation on what the residual actually looks like.
    """
    has_scattered_objects: bool = False  # diff clusters into multiple disjoint blobs
    n_diff_clusters: int = 0            # number of connected components in the diff
    has_isolated_pixels: bool = False    # single-pixel diffs (markers/attractors)
    n_isolated_pixels: int = 0          # count of single-pixel non-bg diffs
    has_shape_movement: bool = False     # diff pattern consistent with objects shifting
    n_colors_in_diff: int = 0           # number of distinct non-bg colors in diff region
    diff_is_local: bool = False         # errors concentrated in a small region
    diff_is_distributed: bool = False   # errors spread across the grid


@dataclass(frozen=True)
class SubgraphBlame:
    """Structural blame assignment: which node/subgraph is the wrong mechanism."""
    node_ids: tuple[str, ...]           # blamed node(s)
    ops: tuple[str, ...]                # their ops
    residual_overlap: float             # 0-1, how much of the residual this subgraph covers
    confidence: float                   # 0-1
    replacement_labels: tuple[str, ...] # structural fragment labels to try
    reason: str = ""


@dataclass(frozen=True)
class GraphFragment:
    """A small typed graph fragment usable as a subgraph replacement.

    Described structurally (not by family name). The fragment has
    designated input/output node ids so it can be spliced into a graph.
    """
    label: str                          # structural label (e.g. "unary_transform")
    nodes: dict[str, GraphNode]
    input_id: str                       # which node receives the predecessor's output
    output_id: str                      # which node produces this fragment's output
    description: str = ""


@dataclass(frozen=True)
class VerifyDiagnostic:
    """Structured residual from a compilable-but-wrong hypothesis.

    Produced when a graph compiles into a program that fails verification.
    Gives the editor actionable information about what went wrong.
    """
    per_demo_diff: tuple[int, ...] = ()   # mismatch pixel count per demo
    total_diff: int = 0                    # sum of per-demo diffs
    failed_demo: int = -1                  # first failing demo index
    diff_fraction: float = 0.0            # fraction of pixels wrong
    repair_hints: tuple[RepairHint, ...] = ()
    blamed_bindings: tuple[str, ...] = () # binding names suspected wrong
    region_residuals: tuple[RegionResidual, ...] = ()  # localized residuals
    subgraph_blames: tuple[SubgraphBlame, ...] = ()    # structural blame
    replacement_fragments: tuple[GraphFragment, ...] = ()  # available replacements
    description: str = ""


@dataclass(frozen=True)
class CompileFailure:
    """A graph that could not compile — with a structured reason."""
    task_id: str
    reason: str
    missing_ops: tuple[str, ...] = ()
    partial_evidence: dict[str, Any] = field(default_factory=dict)
    diagnostic: VerifyDiagnostic | None = None  # present when compiled but verify failed


CompileResult = CompileSuccess | CompileFailure
