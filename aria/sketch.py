"""Sketch IR — task-local hypotheses before concrete programs.

A Sketch sits between raw observation and an executable Program.
It expresses *what kind of computation* would solve a task without
committing to specific ops or literal constants.

Key properties:
- Role variables instead of literal colors (supports color rotation)
- Typed parameter slots that a lightweight solver can fill
- Compositional: multiple SketchSteps interact via shared bindings
- Contextual: transforms gated by predicates over roles/relations
- Small: a Sketch is 2-6 steps, not a search space

A Sketch is NOT executable. It is compiled to a Program (or a small
set of candidate Programs) by a separate sketch compiler.
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


class PrimitiveFamily(Enum):
    """Families of sketch primitives.

    Each family is a *class* of computation, not a single op.
    """
    # Perception / decomposition
    IDENTIFY_ROLES = auto()           # assign bg/fg/frame/anchor roles
    EXTRACT_REGION = auto()           # crop a sub-grid by role or bbox
    FIND_OBJECTS = auto()             # connected component extraction
    FIND_COMPOSITES = auto()          # group CCs into composite motifs

    # Same-dims transforms
    REGION_PERIODIC_REPAIR = auto()   # detect period in region, fix deviations
    OBJECT_MOVE_BY_RELATION = auto()  # move objects based on relational predicate
    COMPOSITE_ROLE_ALIGNMENT = auto() # align composite motifs to anchor axis
    RECOLOR_BY_CONTEXT = auto()       # recolor objects based on spatial context
    FILL_BY_RULE = auto()             # fill region/boundary with rule-derived color

    # Dims-change construction
    CANVAS_LAYOUT = auto()            # determine output dims + populate canvas
    EXTRACT_AND_PACK = auto()         # select sub-regions, pack into output
    TILE_EXPAND = auto()              # expand input by tiling/fractal rule

    # Composition
    COMPOSE_SEQUENTIAL = auto()       # chain sub-sketches in order
    FOR_EACH_OBJECT = auto()          # apply sub-sketch per object
    FOR_EACH_REGION = auto()          # apply sub-sketch per region/zone


# ---------------------------------------------------------------------------
# Sketch steps and full Sketch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SketchStep:
    """One step in a sketch — a primitive applied with roles and slots.

    A step says *what* to do (primitive family) and *with what parameters*
    (roles for symbolic bindings, slots for values to infer), but not
    *how* to do it (that's the compiler's job).
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
    """A task-local hypothesis: a small sequence of typed sketch steps.

    A Sketch is the system's best guess at the *shape* of the solution
    before committing to concrete ops, literals, or search. It is:
    - 2-6 steps (compact)
    - role-normalized (no literal colors unless evidence is strong)
    - parameter-slotted (values inferred from demos, not hardcoded)
    - compositional (steps reference each other)
    """
    task_id: str
    steps: tuple[SketchStep, ...]
    output_ref: str                          # which step produces the final grid
    description: str = ""                    # task-specific summary
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primitive_families(self) -> tuple[PrimitiveFamily, ...]:
        return tuple(s.primitive for s in self.steps)

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
