"""Per-object observation: look at what happened to each object and find the rule.

The core reasoning step that's been missing. Instead of enumerating programs,
this module:
1. For each input object, observes what changed in the output at/around it
2. Expresses each observation as a typed relationship
3. Finds the common rule across objects and demos
4. Constructs a program from the rule

This is observation-driven synthesis at the object level.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.graph.extract import extract_with_delta
from aria.runtime.program import program_to_text
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Delta,
    Expr,
    Grid,
    Literal,
    ObjectNode,
    Program,
    Ref,
    StateGraph,
    Type,
)
from aria.verify.verifier import verify


@dataclass(frozen=True)
class ObjectObservation:
    """What happened to/around one input object."""

    obj_id: int
    color: int
    position: tuple[int, int]  # (row, col) of bbox top-left
    size: int

    # What appeared around this object
    added_offsets: tuple[tuple[int, int], ...]  # relative to object position
    added_color: int | None  # color of added pixels (None if inconsistent)

    # What changed
    color_changed_to: int | None  # if color changed
    moved_to: tuple[int, int] | None  # if position changed
    removed: bool


@dataclass(frozen=True)
class ObjectRule:
    """A generalized rule derived from multiple object observations."""

    kind: str  # "surround", "color_map", "move", "remove", "copy", ...
    input_color: int | None  # which input color this rule applies to (None = all)
    output_color: int | None  # what color the effect produces
    offsets: tuple[tuple[int, int], ...] | None  # spatial pattern
    details: dict[str, Any]  # additional rule-specific data


@dataclass(frozen=True)
class ObservationSynthesisResult:
    """Result of per-object observation and rule inference."""

    solved: bool
    winning_program: Program | None
    candidates_tested: int
    rules: tuple[ObjectRule, ...]
    observations_per_demo: tuple[tuple[ObjectObservation, ...], ...]


def observe_and_synthesize(
    demos: tuple[DemoPair, ...],
) -> ObservationSynthesisResult:
    """Observe per-object changes, find common rules, synthesize programs."""
    # Phase 1: observe what happened per object per demo
    all_observations: list[tuple[ObjectObservation, ...]] = []
    for demo in demos:
        obs = _observe_objects(demo)
        all_observations.append(obs)

    # Phase 2: find rules that are consistent across demos
    rules = _infer_rules(all_observations, demos)

    # Phase 3: for each rule, try to express it as a program and verify
    tested = 0
    for program, rule in _rules_to_programs(rules, demos):
        tested += 1
        result = verify(program, demos)
        if result.passed:
            return ObservationSynthesisResult(
                solved=True,
                winning_program=program,
                candidates_tested=tested,
                rules=tuple(rules),
                observations_per_demo=tuple(
                    tuple(obs) for obs in all_observations
                ),
            )

    return ObservationSynthesisResult(
        solved=False,
        winning_program=None,
        candidates_tested=tested,
        rules=tuple(rules),
        observations_per_demo=tuple(
            tuple(obs) for obs in all_observations
        ),
    )


# ---------------------------------------------------------------------------
# Phase 1: Per-object observation
# ---------------------------------------------------------------------------


def _observe_objects(demo: DemoPair) -> tuple[ObjectObservation, ...]:
    """For each input object, observe what happened in the output."""
    sg_in, sg_out, delta = extract_with_delta(demo.input, demo.output)
    inp, out = demo.input, demo.output

    observations: list[ObjectObservation] = []

    for obj in sg_in.objects:
        obj_col, obj_row = obj.bbox[0], obj.bbox[1]

        # Check if this object was removed
        removed = obj.id in delta.removed

        # Check if color changed
        color_changed_to = None
        for mod_id, field, old, new in delta.modified:
            if mod_id == obj.id and field == "color":
                color_changed_to = new

        # Check if position changed
        moved_to = None
        for mod_id, field, old, new in delta.modified:
            if mod_id == obj.id and field == "bbox":
                moved_to = (new[0], new[1])

        # Find added pixels near this object (attributed by closest-object)
        added_offsets, added_color = _find_added_near(
            obj, sg_in.objects, inp, out,
        )

        observations.append(ObjectObservation(
            obj_id=obj.id,
            color=obj.color,
            position=(obj_row, obj_col),
            size=obj.size,
            added_offsets=tuple(added_offsets),
            added_color=added_color,
            color_changed_to=color_changed_to,
            moved_to=moved_to,
            removed=removed,
        ))

    return tuple(observations)


def _find_added_near(
    obj: ObjectNode,
    all_input_objects: tuple[ObjectNode, ...],
    inp: Grid,
    out: Grid,
) -> tuple[list[tuple[int, int]], int | None]:
    """Find pixels added in the output that are closest to this input object.

    Each changed pixel is assigned to the nearest input object. Only
    pixels assigned to *this* object are returned.
    """
    if inp.shape != out.shape:
        return [], None

    obj_col, obj_row = obj.bbox[0], obj.bbox[1]
    rows, cols = inp.shape

    # Find all changed pixels (background in input, non-background in output)
    bg = 0  # could be detected, but 0 is background for nearly all ARC tasks
    changed_pixels: list[tuple[int, int, int]] = []  # (r, c, new_color)
    for r in range(rows):
        for c in range(cols):
            iv, ov = int(inp[r, c]), int(out[r, c])
            if iv == bg and ov != bg:
                changed_pixels.append((r, c, ov))

    if not changed_pixels:
        return [], None

    # For each changed pixel, find the closest input object
    offsets: list[tuple[int, int]] = []
    colors_added: list[int] = []

    for r, c, new_color in changed_pixels:
        # Find closest input object by Chebyshev distance
        closest_id = None
        closest_dist = float("inf")
        for other_obj in all_input_objects:
            oc, or_ = other_obj.bbox[0], other_obj.bbox[1]
            dist = max(abs(r - or_), abs(c - oc))
            if dist < closest_dist:
                closest_dist = dist
                closest_id = other_obj.id

        if closest_id == obj.id:
            offsets.append((r - obj_row, c - obj_col))
            colors_added.append(new_color)

    if not colors_added:
        return [], None

    color_counts = Counter(colors_added)
    most_common_color = color_counts.most_common(1)[0][0]
    return offsets, most_common_color


# ---------------------------------------------------------------------------
# Phase 2: Rule inference from cross-demo observations
# ---------------------------------------------------------------------------


def _infer_rules(
    all_observations: list[tuple[ObjectObservation, ...]],
    demos: tuple[DemoPair, ...],
) -> list[ObjectRule]:
    """Find rules consistent across demos."""
    rules: list[ObjectRule] = []

    # Strategy 1: surround pattern — same offsets added for objects of same color
    surround_rules = _infer_surround_rules(all_observations)
    rules.extend(surround_rules)

    # Strategy 2: global color map — every pixel of color X becomes color Y
    color_map_rules = _infer_color_map_rules(demos)
    rules.extend(color_map_rules)

    return rules


def _infer_surround_rules(
    all_observations: list[tuple[ObjectObservation, ...]],
) -> list[ObjectRule]:
    """Find surround patterns: input_color → (offsets, output_color)."""
    if not all_observations:
        return []

    # Group observations by input color across all demos
    by_color: dict[int, list[ObjectObservation]] = {}
    for demo_obs in all_observations:
        for obs in demo_obs:
            by_color.setdefault(obs.color, []).append(obs)

    rules: list[ObjectRule] = []

    for input_color, obs_list in by_color.items():
        # Check if all objects of this color have the same offset pattern
        offset_sets = []
        output_colors = []

        for obs in obs_list:
            if obs.added_offsets and obs.added_color is not None:
                offset_sets.append(frozenset(obs.added_offsets))
                output_colors.append(obs.added_color)

        if not offset_sets:
            continue

        # Check consistency: same offsets and same output color for all
        if len(set(offset_sets)) == 1 and len(set(output_colors)) == 1:
            offsets = tuple(sorted(offset_sets[0]))
            rules.append(ObjectRule(
                kind="surround",
                input_color=input_color,
                output_color=output_colors[0],
                offsets=offsets,
                details={"count": len(obs_list)},
            ))

    return rules


def _infer_color_map_rules(
    demos: tuple[DemoPair, ...],
) -> list[ObjectRule]:
    """Check if there's a consistent per-pixel color mapping across demos."""
    if not demos:
        return []

    # Already handled by synthesize.py — skip here to avoid duplication
    return []


# ---------------------------------------------------------------------------
# Phase 3: Rule → Program synthesis
# ---------------------------------------------------------------------------


def _rules_to_programs(
    rules: list[ObjectRule],
    demos: tuple[DemoPair, ...],
) -> list[tuple[Program, ObjectRule]]:
    """Convert inferred rules to candidate programs.

    Tries each rule individually, then tries composing all surround
    rules sequentially (for tasks with multiple color→pattern mappings).
    """
    programs: list[tuple[Program, ObjectRule]] = []

    # Individual rules
    surround_programs: list[Program] = []
    for rule in rules:
        if rule.kind == "surround":
            prog = _surround_rule_to_program(rule, demos)
            if prog is not None:
                programs.append((prog, rule))
                surround_programs.append(prog)

    # Compose all surround rules: apply each fill sequentially
    if len(surround_programs) >= 2:
        composed = _compose_surround_programs(surround_programs)
        if composed is not None:
            combined_rule = ObjectRule(
                kind="surround_composed",
                input_color=None,
                output_color=None,
                offsets=None,
                details={"rule_count": len(surround_programs)},
            )
            programs.append((composed, combined_rule))

    return programs


def _compose_surround_programs(programs: list[Program]) -> Program | None:
    """Chain multiple fill programs: output of each feeds into the next.

    Each program's internal bindings are renamed to avoid collisions,
    and the grid input of each subsequent program is wired to the
    output of the previous one.
    """
    if not programs:
        return None

    all_steps: list[Bind] = []
    prev_grid = "input"
    step_idx = 0

    for prog in programs:
        # Build rename map for this program's internal bindings
        rename: dict[str, str] = {"input": prev_grid}
        for step in prog.steps:
            if not isinstance(step, Bind):
                continue
            new_name = f"v{step_idx}"
            rename[step.name] = new_name
            step_idx += 1

        # Apply renames to each step
        for step in prog.steps:
            if not isinstance(step, Bind):
                continue
            new_name = rename[step.name]
            expr = _rewrite_refs(step.expr, rename)
            all_steps.append(Bind(
                name=new_name, typ=step.typ, expr=expr, declared=step.declared,
            ))

        prev_grid = all_steps[-1].name

    return Program(steps=tuple(all_steps), output=prev_grid)


def _rewrite_refs(expr: Expr, rename: dict[str, str]) -> Expr:
    """Replace references according to a rename map."""
    if isinstance(expr, Ref):
        return Ref(rename.get(expr.name, expr.name))
    if isinstance(expr, Call):
        return Call(
            op=expr.op,
            args=tuple(_rewrite_refs(a, rename) for a in expr.args),
        )
    return expr


def _surround_rule_to_program(
    rule: ObjectRule,
    demos: tuple[DemoPair, ...],
) -> Program | None:
    """Convert a surround rule to a DSL program.

    Surround rule: for each object of color C, place pixels of color D
    at specific offsets. The offset pattern determines which fill op to use:
    - cardinal offsets → fill_cardinal(by_color(C), radius, D, input)
    - diagonal offsets → fill_diagonal(by_color(C), radius, D, input)
    - all offsets → fill_around(by_color(C), radius, D, input)
    """
    if rule.input_color is None or rule.output_color is None or rule.offsets is None:
        return None

    from aria.runtime.ops import has_op

    offset_set = frozenset(rule.offsets)

    # Classify the offset pattern
    programs: list[Program] = []

    # Try fill_cardinal: only (±r, 0) and (0, ±r) offsets
    for radius in range(1, 4):
        cardinal_offsets = frozenset(
            (dr, dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (dr == 0) != (dc == 0) and abs(dr) + abs(dc) <= radius
        )
        if offset_set == cardinal_offsets and has_op("fill_cardinal"):
            programs.append(_make_fill_program(
                "fill_cardinal", rule.input_color, radius, rule.output_color,
            ))

    # Try fill_diagonal: only (±r, ±r) offsets
    for radius in range(1, 4):
        diagonal_offsets = frozenset(
            (dr, dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if abs(dr) == abs(dc) and dr != 0
        )
        if offset_set == diagonal_offsets and has_op("fill_diagonal"):
            programs.append(_make_fill_program(
                "fill_diagonal", rule.input_color, radius, rule.output_color,
            ))

    # Try fill_around: all offsets within Chebyshev distance
    for radius in range(1, 4):
        all_offsets = frozenset(
            (dr, dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (dr, dc) != (0, 0) and max(abs(dr), abs(dc)) <= radius
        )
        if offset_set == all_offsets and has_op("fill_around"):
            programs.append(_make_fill_program(
                "fill_around", rule.input_color, radius, rule.output_color,
            ))

    return programs[0] if programs else None


def _make_fill_program(
    fill_op: str,
    input_color: int,
    radius: int,
    output_color: int,
) -> Program:
    """Build: fill_op(by_color(C), radius, output_color, input)."""
    return Program(
        steps=(
            Bind("v0", Type.PREDICATE, Call("by_color", (
                Literal(input_color, Type.COLOR),
            ))),
            Bind("v1", Type.GRID, Call(fill_op, (
                Ref("v0"),
                Literal(radius, Type.INT),
                Literal(output_color, Type.COLOR),
                Ref("input"),
            ))),
        ),
        output="v1",
    )
