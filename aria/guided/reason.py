"""LEGACY — superseded by induce.py (clause-based induction).

This module's pattern-matching approach (find_consistent_rules) is the
brittle rule-menu that clause induction replaces. Its correspondence
logic is duplicated in induce.py._map_output_to_input.

Do not add new functionality here. Port useful patterns as predicates
or features for clause induction instead.

---
Reasoning layer: compare input and output perception across demos
to discover which input facts explain the output structure.

Works largest-objects-first. For each output object:
1. Find its corresponding input object (by shape, position, or color)
2. Describe what changed in terms of input facts
3. Check consistency across demos
4. Build a generalized rule

No hardcoded task types. The reasoning discovers patterns from facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import Counter

import numpy as np

from aria.guided.perceive import perceive, GridFacts, ObjFact, PairFact
from aria.types import Grid


# ---------------------------------------------------------------------------
# Output-to-input object mapping
# ---------------------------------------------------------------------------

@dataclass
class ObjectMapping:
    """How one output object relates to input objects."""
    out_obj: ObjFact
    in_obj: ObjFact | None          # best matching input object
    match_type: str                  # "same_pos_same_shape", "same_shape_diff_pos", etc.
    color_change: tuple[int, int]    # (from, to) or (-1, to) if new
    position_change: tuple[int, int] # (dr, dc) or (0,0) if same


@dataclass
class ColorExplanation:
    """How the output color is explained by input facts."""
    method: str       # "same", "from_object_N", "bg", "unknown"
    source_oid: int   # which input object provides the color (-1 if N/A)
    confidence: float


@dataclass
class DemoReasoning:
    """Reasoning results for one demo pair."""
    input_facts: GridFacts
    output_facts: GridFacts
    mappings: list[ObjectMapping]
    color_explanations: list[ColorExplanation]


# ---------------------------------------------------------------------------
# Main: reason about one task across all demos
# ---------------------------------------------------------------------------

def reason_about_task(
    demos: list[tuple[Grid, Grid]],
) -> list[DemoReasoning]:
    """For each demo, map output objects to input and explain."""
    results = []
    for inp, out in demos:
        in_facts = perceive(inp)
        out_facts = perceive(out)
        mappings = _map_output_to_input(out_facts, in_facts)
        explanations = _explain_colors(mappings, in_facts)
        results.append(DemoReasoning(in_facts, out_facts, mappings, explanations))
    return results


def find_consistent_rules(
    demo_reasonings: list[DemoReasoning],
) -> list[dict]:
    """Find rules that are consistent across all demos.

    A rule is: "output object with property X gets its color from
    input object with property Y."
    """
    rules = []

    # Rule type 1: all output objects get color from a specific input object role
    # e.g., "output color = smallest input object's color"
    color_source_rules = _find_color_source_rules(demo_reasonings)
    rules.extend(color_source_rules)

    # Rule type 2: output object shapes come from specific input objects
    shape_source_rules = _find_shape_source_rules(demo_reasonings)
    rules.extend(shape_source_rules)

    # Rule type 3: output positions follow a pattern derived from input
    position_rules = _find_position_rules(demo_reasonings)
    rules.extend(position_rules)

    return rules


# ---------------------------------------------------------------------------
# Output-to-input mapping
# ---------------------------------------------------------------------------

def _map_output_to_input(out_facts: GridFacts, in_facts: GridFacts) -> list[ObjectMapping]:
    """Map each output object to its best input match."""
    mappings = []
    used_in = set()

    # Process output objects largest first (already sorted)
    for out_obj in out_facts.objects:
        best_match, match_type = _find_best_input_match(out_obj, in_facts, used_in)

        if best_match:
            used_in.add(best_match.oid)
            dr = out_obj.row - best_match.row
            dc = out_obj.col - best_match.col
            color_change = (best_match.color, out_obj.color)
        else:
            dr, dc = 0, 0
            color_change = (-1, out_obj.color)

        mappings.append(ObjectMapping(
            out_obj=out_obj,
            in_obj=best_match,
            match_type=match_type,
            color_change=color_change,
            position_change=(dr, dc),
        ))

    return mappings


def _find_best_input_match(out_obj, in_facts, used):
    """Find the input object that best corresponds to this output object."""
    candidates = []

    for in_obj in in_facts.objects:
        if in_obj.oid in used:
            continue

        score = 0
        match_type = "none"

        # Same position AND same shape = strongest match
        same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
        same_shape = (in_obj.height == out_obj.height and in_obj.width == out_obj.width and
                      np.array_equal(in_obj.mask, out_obj.mask))
        same_color = (in_obj.color == out_obj.color)
        same_size = (in_obj.size == out_obj.size)

        if same_pos and same_shape and same_color:
            score = 100
            match_type = "identical"
        elif same_pos and same_shape:
            score = 90
            match_type = "same_pos_recolored"
        elif same_shape and same_color:
            score = 80
            match_type = "moved"
        elif same_shape:
            score = 70
            match_type = "moved_recolored"
        elif same_size and same_color:
            score = 40
            match_type = "reshaped"
        elif same_color:
            score = 30
            match_type = "color_match_only"
        elif same_size:
            score = 20
            match_type = "size_match_only"

        if score > 0:
            candidates.append((score, in_obj, match_type))

    if not candidates:
        return None, "new"

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1], candidates[0][2]


# ---------------------------------------------------------------------------
# Color explanation
# ---------------------------------------------------------------------------

def _explain_colors(mappings, in_facts):
    """For each mapped output object, explain where its color comes from."""
    explanations = []

    for m in mappings:
        out_c = m.out_obj.color

        # Same color as matched input object?
        if m.in_obj and m.in_obj.color == out_c:
            explanations.append(ColorExplanation("same_as_source", m.in_obj.oid, 1.0))
            continue

        # Color matches another input object?
        best_source = None
        best_score = 0
        for in_obj in in_facts.objects:
            if in_obj.color != out_c:
                continue
            # Prefer smallest (likely a marker/label)
            score = 1.0 / max(1, in_obj.size)
            # Boost if it's unique color
            if in_obj.n_same_color == 1:
                score *= 2
            if score > best_score:
                best_score = score
                best_source = in_obj

        if best_source:
            explanations.append(ColorExplanation(
                f"from_obj_{best_source.oid}",
                best_source.oid,
                min(1.0, best_score),
            ))
        else:
            explanations.append(ColorExplanation("unknown", -1, 0.0))

    return explanations


# ---------------------------------------------------------------------------
# Cross-demo rule discovery
# ---------------------------------------------------------------------------

def _find_color_source_rules(demo_reasonings):
    """Find consistent color source patterns across demos.

    Check patterns like:
    - "output color always comes from the smallest input object"
    - "output color always comes from the only singleton"
    - "output color always comes from the object at position (0,0)"
    """
    rules = []

    # Pattern: output color = smallest input object's color
    if _check_color_from_property(demo_reasonings, lambda objs: min(objs, key=lambda o: o.size)):
        rules.append({
            'type': 'color_source',
            'pattern': 'smallest_object_color',
            'description': 'output color comes from smallest input object',
        })

    # Pattern: output color = only-singleton's color (if exactly 1 singleton)
    if _check_color_from_property(demo_reasonings, _get_unique_singleton):
        rules.append({
            'type': 'color_source',
            'pattern': 'unique_singleton_color',
            'description': 'output color comes from the unique singleton',
        })

    # Pattern: output color = most-rare-color object
    if _check_color_from_property(demo_reasonings, _get_rarest_color_obj):
        rules.append({
            'type': 'color_source',
            'pattern': 'rarest_color',
            'description': 'output color comes from the rarest-colored object',
        })

    return rules


def _check_color_from_property(demo_reasonings, selector_fn):
    """Check if ALL non-preserved output objects get their color from
    the object selected by selector_fn in EVERY demo."""
    for dr in demo_reasonings:
        try:
            source_obj = selector_fn(dr.input_facts.objects)
        except (ValueError, StopIteration):
            return False
        if source_obj is None:
            return False

        for m, expl in zip(dr.mappings, dr.color_explanations):
            if m.match_type == "identical":
                continue  # preserved, skip
            if m.out_obj.color != source_obj.color:
                return False
    return True


def _get_unique_singleton(objs):
    singletons = [o for o in objs if o.size == 1]
    if len(singletons) == 1:
        return singletons[0]
    return None


def _get_rarest_color_obj(objs):
    if not objs:
        return None
    return min(objs, key=lambda o: o.n_same_color)


def _find_shape_source_rules(demo_reasonings):
    """Find consistent shape-source patterns."""
    rules = []

    # Pattern: output has same shapes as input largest object, just recolored
    all_match = True
    for dr in demo_reasonings:
        for m in dr.mappings:
            if m.match_type not in ("identical", "same_pos_recolored", "moved", "moved_recolored"):
                if m.match_type == "new":
                    all_match = False
                    break
        if not all_match:
            break

    if all_match:
        rules.append({
            'type': 'shape_source',
            'pattern': 'all_shapes_from_input',
            'description': 'all output shapes exist in input (possibly moved/recolored)',
        })

    return rules


def _find_position_rules(demo_reasonings):
    """Find consistent position patterns."""
    rules = []

    # Pattern: uniform offset (all objects move by same amount)
    offsets = set()
    for dr in demo_reasonings:
        for m in dr.mappings:
            if m.match_type in ("moved", "moved_recolored"):
                offsets.add(m.position_change)

    if len(offsets) == 1 and offsets != {(0, 0)}:
        dr, dc = next(iter(offsets))
        rules.append({
            'type': 'position',
            'pattern': 'uniform_offset',
            'offset': (dr, dc),
            'description': f'all objects move by ({dr},{dc})',
        })

    return rules
