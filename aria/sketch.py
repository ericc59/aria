"""Sketch IR — graph-shaped hypothesis representation.

Core types for the generalized compilation pipeline:

  SketchGraph    — DAG of SketchNodes (primitive ops with typed edges)
  Specialization — resolved static task structure (axis, period, roles, …)
  ResolvedBinding — one slot/role pinned to a concrete value

The linear Sketch/SketchStep types are retained for backward compatibility
and serialization; SketchGraph.from_sketch() converts between them.

A sketch is NOT executable.  It is compiled to a Program via:
  fit → SketchGraph → specialize → compile_sketch_graph → verify

Key properties:
- Role variables instead of literal colors (supports color rotation)
- Typed parameter slots resolved by the specialization pass
- Graph-shaped: nodes reference predecessors via explicit edges
- Small: a sketch is 2–6 nodes, not a search space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Role variables — symbolic names that bind to concrete values per demo
# ---------------------------------------------------------------------------


class RoleKind(Enum):
    """What structural role a color/object/region plays."""
    BG = auto()          # background / most-common color
    FRAME = auto()       # border / enclosure color
    ANCHOR = auto()      # stationary reference object
    CENTER = auto()      # center pixel of a composite motif
    CONTENT = auto()     # foreground content color
    MARKER = auto()      # singleton indicator
    FILL = auto()        # color used to fill / paint


@dataclass(frozen=True)
class RoleVar:
    """A symbolic reference to a color/value determined by structural role.

    Instead of Literal(3, Type.COLOR) the sketch says RoleVar("bg")
    and the compiler resolves it per demo.
    """
    name: str                    # unique within a sketch, e.g. "bg", "frame", "c1"
    kind: RoleKind               # structural constraint
    description: str = ""        # human-readable, e.g. "the most common color"

    def __repr__(self) -> str:
        return f"${self.name}"


# ---------------------------------------------------------------------------
# Parameter slots — values to be inferred or searched
# ---------------------------------------------------------------------------


class SlotType(Enum):
    """Types a parameter slot can take."""
    COLOR = auto()
    INT = auto()
    AXIS = auto()          # row or col
    DIR = auto()           # up/down/left/right
    PREDICATE = auto()     # a selection criterion
    TRANSFORM = auto()     # an object transform
    DIMS = auto()          # (rows, cols) pair
    REGION = auto()        # sub-grid bounding box


@dataclass(frozen=True)
class Slot:
    """An unresolved parameter in a sketch step.

    Slots are filled by the sketch compiler using demo evidence.
    """
    name: str
    typ: SlotType
    constraint: str = ""     # e.g. "positive", "from output palette", "< grid_width"
    evidence: Any = None     # observed value(s) from demos, if available

    def __repr__(self) -> str:
        return f"?{self.name}:{self.typ.name}"


# ---------------------------------------------------------------------------
# Sketch primitives — the vocabulary of hypothesis steps
# ---------------------------------------------------------------------------


class Primitive(Enum):
    """Sketch primitives — the atoms of graph-shaped hypotheses.

    Each primitive is a single, reusable operation type.  A sketch graph
    composes a small number of these into a DAG that describes the shape
    of the solution without committing to runtime ops or literal constants.
    """
    # Perception
    BIND_ROLE = auto()            # assign structural roles (bg/fg/frame/anchor)
    EXTRACT_VIEW = auto()         # extract a decomposition view (framed regions, composites, objects)
    SELECT_REGION = auto()        # select a specific region/sub-grid
    SELECT_SUBSET = auto()        # filter objects/regions by predicate
    PEEL_FRAME = auto()           # strip outermost frame border
    PARTITION_GRID = auto()       # split grid into sub-cells by separators

    # Analysis
    INFER_REGULARITY = auto()     # detect pattern/period/symmetry in content
    INFER_MOTIF = auto()          # infer repeating unit from 1D line
    APPLY_RELATION = auto()       # compute relation between objects/regions (alignment, distance)
    DETECT_MISMATCH = auto()      # identify cells violating a regularity

    # Transforms
    APPLY_TRANSFORM = auto()      # apply a transform to selected objects/grid
    REPAIR_MISMATCH = auto()      # fix cells that violate inferred regularity
    REPAIR_LINES = auto()         # infer motif + repair mismatches per line
    REPAIR_2D_MOTIF = auto()      # infer 2D tile motif + repair mismatches per cell

    # Construction
    CONSTRUCT_CANVAS = auto()     # create output grid with computed dimensions
    PAINT = auto()                # render objects/content onto a grid
    COMPOSE = auto()              # combine multiple sub-results sequentially

    # Iteration
    FOR_EACH = auto()             # apply sub-operation to each element


# Backward compatibility — legacy code may reference PrimitiveFamily
PrimitiveFamily = Primitive

# Legacy aliases: old lane-specific names → generic primitives.
# Retained so existing serialized sketches and tests keep working.
Primitive.IDENTIFY_ROLES = Primitive.BIND_ROLE
Primitive.EXTRACT_REGION = Primitive.SELECT_REGION
Primitive.FIND_OBJECTS = Primitive.EXTRACT_VIEW
Primitive.FIND_COMPOSITES = Primitive.EXTRACT_VIEW
Primitive.REGION_PERIODIC_REPAIR = Primitive.REPAIR_MISMATCH
Primitive.OBJECT_MOVE_BY_RELATION = Primitive.APPLY_TRANSFORM
Primitive.COMPOSITE_ROLE_ALIGNMENT = Primitive.APPLY_RELATION
Primitive.RECOLOR_BY_CONTEXT = Primitive.APPLY_TRANSFORM
Primitive.FILL_BY_RULE = Primitive.APPLY_TRANSFORM
Primitive.CANVAS_LAYOUT = Primitive.CONSTRUCT_CANVAS
Primitive.EXTRACT_AND_PACK = Primitive.SELECT_REGION
Primitive.TILE_EXPAND = Primitive.APPLY_TRANSFORM
Primitive.COMPOSE_SEQUENTIAL = Primitive.COMPOSE
Primitive.FOR_EACH_OBJECT = Primitive.FOR_EACH
Primitive.FOR_EACH_REGION = Primitive.FOR_EACH


# ---------------------------------------------------------------------------
# Sketch steps and full Sketch (linear form — see SketchGraph for DAG form)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SketchStep:
    """One node in the linear sketch representation.

    Describes *what* to do (primitive) and *with what parameters*
    (roles for symbolic bindings, slots for values to infer).
    Prefer SketchNode / SketchGraph for new code; SketchStep is
    retained for serialization and backward compatibility.
    """
    name: str                                # binding name, e.g. "s1"
    primitive: PrimitiveFamily
    roles: tuple[RoleVar, ...] = ()          # symbolic color/object bindings
    slots: tuple[Slot, ...] = ()             # parameters to fill
    input_refs: tuple[str, ...] = ()         # names of prior steps this depends on
    description: str = ""                    # task-specific human description
    confidence: float = 1.0                  # 0..1, how confident this step is
    evidence: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        roles_str = ", ".join(repr(r) for r in self.roles)
        slots_str = ", ".join(repr(s) for s in self.slots)
        parts = [self.primitive.name]
        if roles_str:
            parts.append(f"roles=[{roles_str}]")
        if slots_str:
            parts.append(f"slots=[{slots_str}]")
        if self.input_refs:
            parts.append(f"inputs={list(self.input_refs)}")
        return f"{self.name} = {', '.join(parts)}"


@dataclass(frozen=True)
class Sketch:
    """Linear sketch representation (ordered list of SketchSteps).

    This is the serialization-friendly form produced by fitters.
    For compilation, convert to SketchGraph via SketchGraph.from_sketch().
    """
    task_id: str
    steps: tuple[SketchStep, ...]
    output_ref: str                          # which step produces the final grid
    description: str = ""                    # task-specific summary
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primitive_families(self) -> tuple[Primitive, ...]:
        return tuple(s.primitive for s in self.steps)

    @property
    def primitive_pattern(self) -> tuple[str, ...]:
        """Structural pattern of primitives (for compiler dispatch)."""
        return tuple(s.primitive.name for s in self.steps)

    @property
    def role_vars(self) -> tuple[RoleVar, ...]:
        seen: dict[str, RoleVar] = {}
        for step in self.steps:
            for rv in step.roles:
                if rv.name not in seen:
                    seen[rv.name] = rv
        return tuple(seen.values())

    @property
    def open_slots(self) -> tuple[Slot, ...]:
        return tuple(s for step in self.steps for s in step.slots)

    def validate(self) -> list[str]:
        """Return a list of structural issues (empty = valid)."""
        errors: list[str] = []
        names = set()
        for step in self.steps:
            if step.name in names:
                errors.append(f"duplicate step name: {step.name}")
            names.add(step.name)
            for ref in step.input_refs:
                if ref != "input" and ref not in names:
                    errors.append(f"step {step.name} references undefined {ref}")
        if self.output_ref not in names and self.output_ref != "input":
            errors.append(f"output_ref {self.output_ref} not defined")
        return errors


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def sketch_to_dict(sketch: Sketch) -> dict[str, Any]:
    """Serialize a Sketch to a JSON-compatible dict."""
    return {
        "task_id": sketch.task_id,
        "description": sketch.description,
        "confidence": sketch.confidence,
        "output_ref": sketch.output_ref,
        "steps": [_step_to_dict(s) for s in sketch.steps],
        "metadata": sketch.metadata,
    }


def _step_to_dict(step: SketchStep) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": step.name,
        "primitive": step.primitive.name,
        "description": step.description,
    }
    if step.roles:
        d["roles"] = [
            {"name": r.name, "kind": r.kind.name, "description": r.description}
            for r in step.roles
        ]
    if step.slots:
        d["slots"] = [
            {"name": s.name, "type": s.typ.name, "constraint": s.constraint,
             "evidence": s.evidence}
            for s in step.slots
        ]
    if step.input_refs:
        d["input_refs"] = list(step.input_refs)
    if step.confidence < 1.0:
        d["confidence"] = step.confidence
    if step.evidence:
        d["evidence"] = step.evidence
    return d


def sketch_from_dict(d: dict[str, Any]) -> Sketch:
    """Deserialize a Sketch from a dict."""
    steps = tuple(_step_from_dict(sd) for sd in d["steps"])
    return Sketch(
        task_id=d["task_id"],
        steps=steps,
        output_ref=d["output_ref"],
        description=d.get("description", ""),
        confidence=d.get("confidence", 1.0),
        metadata=d.get("metadata", {}),
    )


def _step_from_dict(d: dict[str, Any]) -> SketchStep:
    roles = tuple(
        RoleVar(name=r["name"], kind=RoleKind[r["kind"]],
                description=r.get("description", ""))
        for r in d.get("roles", [])
    )
    slots = tuple(
        Slot(name=s["name"], typ=SlotType[s["type"]],
             constraint=s.get("constraint", ""),
             evidence=s.get("evidence"))
        for s in d.get("slots", [])
    )
    return SketchStep(
        name=d["name"],
        primitive=PrimitiveFamily[d["primitive"]],
        roles=roles,
        slots=slots,
        input_refs=tuple(d.get("input_refs", ())),
        description=d.get("description", ""),
        confidence=d.get("confidence", 1.0),
        evidence=d.get("evidence", {}),
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def sketch_to_text(sketch: Sketch) -> str:
    """Human-readable sketch representation."""
    lines: list[str] = []
    lines.append(f"sketch {sketch.task_id} {{")
    if sketch.description:
        lines.append(f"  // {sketch.description}")

    # Role variable declarations
    rvars = sketch.role_vars
    if rvars:
        lines.append(f"  roles {{")
        for rv in rvars:
            desc = f"  // {rv.description}" if rv.description else ""
            lines.append(f"    ${rv.name}: {rv.kind.name}{desc}")
        lines.append(f"  }}")

    # Steps
    for step in sketch.steps:
        inputs = ", ".join(step.input_refs) if step.input_refs else "input"
        roles_str = ", ".join(f"${r.name}" for r in step.roles)
        slots_str = ", ".join(f"?{s.name}" for s in step.slots)

        args = []
        if inputs != "input":
            args.append(inputs)
        if roles_str:
            args.append(roles_str)
        if slots_str:
            args.append(slots_str)

        arg_str = f"({', '.join(args)})" if args else "(input)"
        conf = f" [{step.confidence:.0%}]" if step.confidence < 1.0 else ""
        lines.append(f"  {step.name} = {step.primitive.name}{arg_str}{conf}")
        if step.description:
            lines.append(f"    // {step.description}")

    lines.append(f"  -> {sketch.output_ref}")
    lines.append(f"}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience builders for the first primitives
# ---------------------------------------------------------------------------


def make_identify_roles(
    bg_colors: list[int] | None = None,
) -> SketchStep:
    """Build an IDENTIFY_ROLES step."""
    roles = [RoleVar("bg", RoleKind.BG, "most common color per demo")]
    evidence = {}
    if bg_colors is not None:
        evidence["observed_bg_colors"] = bg_colors
    return SketchStep(
        name="roles",
        primitive=PrimitiveFamily.IDENTIFY_ROLES,
        roles=tuple(roles),
        description="assign structural color roles",
        evidence=evidence,
    )


def make_region_periodic_repair(
    *,
    region_ref: str = "input",
    frame_role: str = "frame",
    period_axis: str | None = None,
) -> SketchStep:
    """Build a REGION_PERIODIC_REPAIR step."""
    roles = [RoleVar(frame_role, RoleKind.FRAME, "frame/border color")]
    slots = []
    if period_axis:
        slots.append(Slot("axis", SlotType.AXIS, evidence=period_axis))
    else:
        slots.append(Slot("axis", SlotType.AXIS, constraint="infer from content"))
    slots.append(Slot("period", SlotType.INT, constraint="positive, infer from content"))

    return SketchStep(
        name="repair",
        primitive=PrimitiveFamily.REGION_PERIODIC_REPAIR,
        roles=tuple(roles),
        slots=tuple(slots),
        input_refs=(region_ref,),
        description="detect periodic pattern inside framed region, fix deviations",
    )


def make_object_move_by_relation(
    *,
    objects_ref: str = "objects",
    anchor_role: str = "anchor",
    predicate_desc: str = "",
) -> SketchStep:
    """Build an OBJECT_MOVE_BY_RELATION step."""
    roles = [RoleVar(anchor_role, RoleKind.ANCHOR, "stationary reference object")]
    slots = [
        Slot("selection", SlotType.PREDICATE,
             constraint=predicate_desc or "which objects to move"),
        Slot("delta_rule", SlotType.TRANSFORM,
             constraint="displacement derived from anchor relationship"),
    ]
    return SketchStep(
        name="moved",
        primitive=PrimitiveFamily.OBJECT_MOVE_BY_RELATION,
        roles=tuple(roles),
        slots=tuple(slots),
        input_refs=(objects_ref,),
        description="move selected objects by relation to anchor",
    )


def make_composite_role_alignment(
    *,
    composites_ref: str = "composites",
    anchor_role: str = "anchor",
    axis: str | None = None,
) -> SketchStep:
    """Build a COMPOSITE_ROLE_ALIGNMENT step."""
    roles = [
        RoleVar(anchor_role, RoleKind.ANCHOR, "stationary anchor singleton"),
        RoleVar("center", RoleKind.CENTER, "center pixel of composite motif"),
        RoleVar("frame", RoleKind.FRAME, "frame surrounding center"),
    ]
    slots = []
    if axis:
        slots.append(Slot("axis", SlotType.AXIS, evidence=axis))
    else:
        slots.append(Slot("axis", SlotType.AXIS, constraint="infer from anchor position"))

    return SketchStep(
        name="aligned",
        primitive=PrimitiveFamily.COMPOSITE_ROLE_ALIGNMENT,
        roles=tuple(roles),
        slots=tuple(slots),
        input_refs=(composites_ref,),
        description="align composite motifs so center matches anchor on axis",
    )


def make_canvas_layout(
    *,
    dims_evidence: tuple[int, int] | None = None,
    fixed_output: bool = False,
) -> SketchStep:
    """Build a CANVAS_LAYOUT step."""
    slots = [
        Slot("output_dims", SlotType.DIMS,
             constraint="fixed" if fixed_output else "infer from input structure",
             evidence=dims_evidence),
        Slot("layout_rule", SlotType.TRANSFORM,
             constraint="how to populate canvas from input content"),
    ]
    return SketchStep(
        name="canvas",
        primitive=PrimitiveFamily.CANVAS_LAYOUT,
        slots=tuple(slots),
        description="construct output canvas with inferred dimensions and layout",
    )


# ---------------------------------------------------------------------------
# Graph-shaped sketch IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SketchNode:
    """A node in a sketch dependency graph.

    Unlike SketchStep, a SketchNode has an explicit node id, typed
    input/output edges, and is designed for graph traversal rather than
    sequential reading.
    """
    id: str                                  # unique within the graph
    primitive: Primitive
    inputs: tuple[str, ...] = ()             # ids of nodes this depends on
    output_type: SlotType | None = None      # what this node produces
    roles: tuple[RoleVar, ...] = ()
    slots: tuple[Slot, ...] = ()
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SketchGraph:
    """A sketch represented as an explicit dependency graph (DAG).

    Nodes are primitive operations; edges are data dependencies.
    The graph makes dependencies inspectable and supports topological
    traversal, subgraph extraction, and specialization passes.
    """
    task_id: str
    nodes: dict[str, SketchNode]             # id -> node
    output_id: str                           # which node produces the final result
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
        """Structural validation — returns list of issues (empty = valid)."""
        errors: list[str] = []
        ids = set(self.nodes.keys())
        for nid, node in self.nodes.items():
            for dep in node.inputs:
                if dep != "input" and dep not in ids:
                    errors.append(f"node {nid} depends on undefined {dep}")
        if self.output_id not in ids and self.output_id != "input":
            errors.append(f"output_id {self.output_id} not in graph")
        # Check for cycles
        try:
            self.topo_order()
        except RecursionError:
            errors.append("graph contains a cycle")
        return errors

    @staticmethod
    def from_sketch(sketch: Sketch) -> SketchGraph:
        """Convert a linear Sketch into a SketchGraph."""
        nodes: dict[str, SketchNode] = {}
        for step in sketch.steps:
            nodes[step.name] = SketchNode(
                id=step.name,
                primitive=step.primitive,
                inputs=step.input_refs if step.input_refs else ("input",),
                roles=step.roles,
                slots=step.slots,
                description=step.description,
                evidence=dict(step.evidence),
            )
        return SketchGraph(
            task_id=sketch.task_id,
            nodes=nodes,
            output_id=sketch.output_ref,
            description=sketch.description,
            metadata=dict(sketch.metadata),
        )

    def to_sketch(self) -> Sketch:
        """Convert back to a linear Sketch (topo-ordered steps)."""
        ordered = self.topo_order()
        steps: list[SketchStep] = []
        for nid in ordered:
            node = self.nodes[nid]
            steps.append(SketchStep(
                name=node.id,
                primitive=node.primitive,
                roles=node.roles,
                slots=node.slots,
                input_refs=node.inputs,
                description=node.description,
                evidence=node.evidence,
            ))
        return Sketch(
            task_id=self.task_id,
            steps=tuple(steps),
            output_ref=self.output_id,
            description=self.description,
            metadata=dict(self.metadata),
        )


# ---------------------------------------------------------------------------
# Specialization — resolved static task structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBinding:
    """A single resolved static binding: a slot or role pinned to a value."""
    node_id: str          # which graph node this binding applies to
    name: str             # slot or role name
    value: Any            # resolved value (int, str, tuple, etc.)
    source: str = ""      # how it was resolved: "evidence", "consensus", "inferred"


@dataclass(frozen=True)
class Specialization:
    """Static task structure extracted from demo evidence.

    This is the symbolic analogue of "baking static structure into weights."
    It captures everything that is constant across demos so that
    compilation can use concrete values instead of re-inferring them.
    """
    task_id: str
    bindings: tuple[ResolvedBinding, ...]     # resolved slot/role values
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, node_id: str, name: str) -> Any | None:
        """Look up a resolved binding by node and name."""
        for b in self.bindings:
            if b.node_id == node_id and b.name == name:
                return b.value
        return None

    def bindings_for_node(self, node_id: str) -> tuple[ResolvedBinding, ...]:
        """All bindings that apply to a specific graph node."""
        return tuple(b for b in self.bindings if b.node_id == node_id)

    @property
    def binding_names(self) -> frozenset[str]:
        """All bound names across all nodes."""
        return frozenset(b.name for b in self.bindings)
