"""Selector-relevant per-object facts for rule induction.

Extracts rich boolean attributes from perceived objects. Used by the
selector rule inducer to discover exact cross-demo selection rules.

NOT an execution layer. NOT part of the AST. This is derive-time
reasoning that enables richer object selection via bounded conjunctions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from aria.guided.perceive import GridFacts, ObjFact, PairFact


# ---------------------------------------------------------------------------
# Feature names (stable identifiers for rule induction)
# ---------------------------------------------------------------------------

STRUCTURAL_FEATURES = [
    # Shape
    'is_rectangular', 'is_square', 'is_line', 'is_singleton',
    # Size rank
    'is_largest', 'is_smallest', 'size_above_median', 'size_below_median',
    # Position
    'is_topmost', 'is_bottommost', 'is_leftmost', 'is_rightmost',
    'touches_top', 'touches_bottom', 'touches_left', 'touches_right',
    'touches_boundary', 'interior',
    # Color uniqueness
    'unique_color', 'has_color_partner',
    'majority_color', 'minority_color',
    # Shape/size uniqueness
    'unique_shape', 'unique_size', 'has_shape_partner',
    # Relational
    'enclosed_by_another', 'encloses_another',
    'adjacent_to_larger', 'adjacent_to_smaller',
]


# ---------------------------------------------------------------------------
# Full fact extraction (derive-time, from GridFacts)
# ---------------------------------------------------------------------------

def extract_object_facts(facts: GridFacts) -> list[dict[str, bool]]:
    """Extract boolean fact dictionary for each object.

    Returns one dict per object (same order as facts.objects).
    All values are bool. Feature names are stable identifiers
    suitable for rule induction via induce_boolean_dnf.
    """
    objs = facts.objects
    if not objs:
        return []

    ctx = _scene_context(objs, facts.pairs if hasattr(facts, 'pairs') else [])

    rows = []
    for obj in objs:
        row = _build_fact_row(obj, objs, ctx)
        # Per-color boolean features (dynamic, scene-dependent)
        for c in sorted(ctx['color_counts'].keys()):
            row[f'color_is_{c}'] = (obj.color == c)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Single-object fact extraction (execution-time, from obj + all_objs)
# ---------------------------------------------------------------------------

def check_rule_on_object(
    rule_dict: dict,
    obj: ObjFact,
    all_objs: list[ObjFact],
    pairs: list | None = None,
) -> bool:
    """Test whether a single object matches a DNF selection rule.

    Called at execution time from Predicate.test() for SELECTION_RULE.
    """
    from aria.search.rules import DNFRule

    rule = DNFRule.from_dict(rule_dict)
    ctx = _scene_context(all_objs, pairs or [])
    fact_row = _build_fact_row(obj, all_objs, ctx)
    # Add per-color features
    for c in sorted(ctx['color_counts'].keys()):
        fact_row[f'color_is_{c}'] = (obj.color == c)
    return rule.matches(fact_row)


def select_by_rule(rule_dict: dict, facts: GridFacts) -> list[ObjFact]:
    """Evaluate a DNF rule against all objects in a scene.

    Returns matching ObjFact list (derive-time, has full GridFacts).
    """
    from aria.search.rules import DNFRule

    rule = DNFRule.from_dict(rule_dict)
    fact_rows = extract_object_facts(facts)
    return [
        obj for obj, row in zip(facts.objects, fact_rows)
        if rule.matches(row)
    ]


# ---------------------------------------------------------------------------
# Shared: scene context and fact row builder
# ---------------------------------------------------------------------------

def _scene_context(
    objs: list[ObjFact],
    pairs: list,
) -> dict:
    """Precompute scene-level aggregates needed for feature extraction."""
    sizes = [o.size for o in objs]
    max_size = max(sizes)
    min_size = min(sizes)
    median_size = sorted(sizes)[len(sizes) // 2]
    min_row = min(o.row for o in objs)
    max_bottom = max(o.row + o.height for o in objs)
    min_col = min(o.col for o in objs)
    max_right = max(o.col + o.width for o in objs)

    color_counts = Counter(o.color for o in objs)
    max_color_count = max(color_counts.values())
    min_color_count = min(color_counts.values())

    # Containment (bbox-based)
    enclosed: set[int] = set()
    enclosing: set[int] = set()
    for a in objs:
        for b in objs:
            if a.oid == b.oid:
                continue
            if (b.row <= a.row and b.col <= a.col
                    and b.row + b.height >= a.row + a.height
                    and b.col + b.width >= a.col + a.width
                    and b.size > a.size):
                enclosed.add(a.oid)
                enclosing.add(b.oid)

    # Adjacency to larger/smaller
    adj_to_larger: set[int] = set()
    adj_to_smaller: set[int] = set()
    for pair in pairs:
        if not pair.adjacent:
            continue
        a = next((o for o in objs if o.oid == pair.oid_a), None)
        b = next((o for o in objs if o.oid == pair.oid_b), None)
        if a is None or b is None:
            continue
        if a.size < b.size:
            adj_to_larger.add(a.oid)
            adj_to_smaller.add(b.oid)
        elif a.size > b.size:
            adj_to_larger.add(b.oid)
            adj_to_smaller.add(a.oid)

    return {
        'max_size': max_size, 'min_size': min_size,
        'median_size': median_size,
        'min_row': min_row, 'max_bottom': max_bottom,
        'min_col': min_col, 'max_right': max_right,
        'color_counts': color_counts,
        'max_color_count': max_color_count,
        'min_color_count': min_color_count,
        'enclosed': enclosed, 'enclosing': enclosing,
        'adj_to_larger': adj_to_larger,
        'adj_to_smaller': adj_to_smaller,
    }


def _build_fact_row(
    obj: ObjFact,
    all_objs: list[ObjFact],
    ctx: dict,
) -> dict[str, bool]:
    """Build the boolean fact row for one object given precomputed context."""
    touches_boundary = (obj.touches_top or obj.touches_bottom
                        or obj.touches_left or obj.touches_right)

    return {
        # Shape
        'is_rectangular': obj.is_rectangular,
        'is_square': obj.is_square,
        'is_line': obj.is_line,
        'is_singleton': obj.size == 1,
        # Size rank
        'is_largest': obj.size == ctx['max_size'],
        'is_smallest': obj.size == ctx['min_size'],
        'size_above_median': obj.size > ctx['median_size'],
        'size_below_median': obj.size < ctx['median_size'],
        # Position
        'is_topmost': obj.row == ctx['min_row'],
        'is_bottommost': (obj.row + obj.height) == ctx['max_bottom'],
        'is_leftmost': obj.col == ctx['min_col'],
        'is_rightmost': (obj.col + obj.width) == ctx['max_right'],
        'touches_top': obj.touches_top,
        'touches_bottom': obj.touches_bottom,
        'touches_left': obj.touches_left,
        'touches_right': obj.touches_right,
        'touches_boundary': touches_boundary,
        'interior': not touches_boundary,
        # Color uniqueness
        'unique_color': obj.n_same_color == 1,
        'has_color_partner': obj.n_same_color > 1,
        'majority_color': ctx['color_counts'].get(obj.color, 0) == ctx['max_color_count'],
        'minority_color': ctx['color_counts'].get(obj.color, 0) == ctx['min_color_count'],
        # Shape/size uniqueness
        'unique_shape': obj.n_same_shape == 1,
        'unique_size': obj.n_same_size == 1,
        'has_shape_partner': obj.n_same_shape > 1,
        # Relational
        'enclosed_by_another': obj.oid in ctx['enclosed'],
        'encloses_another': obj.oid in ctx['enclosing'],
        'adjacent_to_larger': obj.oid in ctx['adj_to_larger'],
        'adjacent_to_smaller': obj.oid in ctx['adj_to_smaller'],
    }
