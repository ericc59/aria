"""Object correspondence: map output objects to input sources.

This is THE correspondence engine for the active clause-based pipeline.
All other modules that need output→input object mapping should import from here.

For each output object, finds the best matching input object and classifies
the relationship (identical, recolored, moved, moved_recolored, new).
Uses ObjFact from perceive.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.guided.perceive import GridFacts, ObjFact


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ObjMapping:
    """How one output object relates to its input source."""
    out_obj: ObjFact
    in_obj: ObjFact | None
    match_type: str       # "identical", "recolored", "moved", "moved_recolored", "new"
    color_from: int       # input color (-1 if new)
    color_to: int         # output color


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def map_output_to_input(
    out_facts: GridFacts,
    in_facts: GridFacts,
) -> list[ObjMapping]:
    """Map each output object to its best input match.

    Processes output objects in order (largest first, matching perceive sort).
    Each input object is used at most once (greedy assignment).
    """
    mappings = []
    used = set()
    for out_obj in out_facts.objects:
        best, mtype = _best_match(out_obj, in_facts.objects, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj,
            in_obj=best,
            match_type=mtype,
            color_from=best.color if best else -1,
            color_to=out_obj.color,
        ))
    return mappings


def map_output_to_input_topk(
    out_facts: GridFacts,
    in_facts: GridFacts,
    k: int = 3,
) -> list[list[ObjMapping]]:
    """Generate K correspondence hypotheses using different strategies.

    Returns up to K distinct mapping lists. The first is always the default
    (largest-first greedy). Others use different tie-breaking or ordering.
    """
    hypotheses = []
    seen = set()

    # Strategy 1: default (largest first)
    m1 = map_output_to_input(out_facts, in_facts)
    key1 = _mapping_key(m1)
    hypotheses.append(m1)
    seen.add(key1)

    # Strategy 2: smallest output objects first
    if len(out_facts.objects) > 1:
        m2 = _map_with_order(out_facts, in_facts, reverse=True)
        key2 = _mapping_key(m2)
        if key2 not in seen:
            hypotheses.append(m2)
            seen.add(key2)

    # Strategy 3: prioritize same-color matches over same-shape
    if len(hypotheses) < k:
        m3 = _map_color_priority(out_facts, in_facts)
        key3 = _mapping_key(m3)
        if key3 not in seen:
            hypotheses.append(m3)
            seen.add(key3)

    return hypotheses[:k]


def _mapping_key(mappings):
    """Hashable key for a correspondence (for dedup)."""
    return tuple(
        (m.out_obj.oid, m.in_obj.oid if m.in_obj else -1, m.match_type)
        for m in mappings
    )


def _map_with_order(out_facts, in_facts, reverse=False):
    """Map with output objects in size order (reverse=True → smallest first)."""
    ordered = sorted(out_facts.objects, key=lambda o: o.size, reverse=not reverse)
    mappings = []
    used = set()
    for out_obj in ordered:
        best, mtype = _best_match(out_obj, in_facts.objects, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj, in_obj=best, match_type=mtype,
            color_from=best.color if best else -1, color_to=out_obj.color,
        ))
    return mappings


def _map_color_priority(out_facts, in_facts):
    """Map prioritizing color match over shape match."""
    mappings = []
    used = set()
    for out_obj in out_facts.objects:
        best, mtype = _best_match_color_first(out_obj, in_facts.objects, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj, in_obj=best, match_type=mtype,
            color_from=best.color if best else -1, color_to=out_obj.color,
        ))
    return mappings


def _best_match_color_first(out_obj, in_objs, used):
    """Like _best_match but prioritizes color match over position match."""
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
        same_shape = (in_obj.height == out_obj.height and
                      in_obj.width == out_obj.width and
                      np.array_equal(in_obj.mask, out_obj.mask))
        same_color = (in_obj.color == out_obj.color)
        if same_pos and same_shape and same_color:
            return in_obj, "identical"
        if same_shape and same_color:
            return in_obj, "moved"
        if same_pos and same_shape:
            return in_obj, "recolored"
        if same_shape:
            return in_obj, "moved_recolored"
    return None, "new"


def find_removed_objects(
    in_facts: GridFacts,
    mappings: list[ObjMapping],
) -> list[ObjFact]:
    """Find input objects that have no corresponding output object."""
    matched_oids = {m.in_obj.oid for m in mappings if m.in_obj is not None}
    return [o for o in in_facts.objects if o.oid not in matched_oids]


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _best_match(
    out_obj: ObjFact,
    in_objs: list[ObjFact],
    used: set[int],
) -> tuple[ObjFact | None, str]:
    """Find the best input match for one output object.

    Match priority: identical > recolored > moved > moved_recolored.
    Within each tier, first match wins (objects are sorted by size desc).
    """
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
        same_shape = (in_obj.height == out_obj.height and
                      in_obj.width == out_obj.width and
                      np.array_equal(in_obj.mask, out_obj.mask))
        same_color = (in_obj.color == out_obj.color)

        if same_pos and same_shape and same_color:
            return in_obj, "identical"
        if same_pos and same_shape:
            return in_obj, "recolored"
        if same_shape and same_color:
            return in_obj, "moved"
        if same_shape:
            return in_obj, "moved_recolored"

    return None, "new"
