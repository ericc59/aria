"""Tests for selection_facts: per-object boolean fact extraction."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive
from aria.search.selection_facts import (
    extract_object_facts,
    select_by_rule,
    check_rule_on_object,
    STRUCTURAL_FEATURES,
)


def _make_grid_with_objects():
    """Build a small grid with diverse objects for fact verification.

    Layout (10x10, bg=0):
      - Large 3x3 red(2) rectangle at (0,0) — touches top+left, largest
      - Small 1x1 blue(1) singleton at (5,5) — interior, smallest, unique color
      - Medium 2x2 green(3) rectangle at (7,7) — touches bottom+right
      - 1x3 red(2) line at (4,0) — touches left, is_line
    """
    grid = np.zeros((10, 10), dtype=np.int8)
    # Large red block
    grid[0:3, 0:3] = 2
    # Blue singleton
    grid[5, 5] = 1
    # Green block bottom-right
    grid[8:10, 8:10] = 3
    # Red line
    grid[4, 0:3] = 2
    return grid


def test_extract_object_facts_shape():
    """Correct number of fact rows and correct structural features."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    assert len(rows) == len(facts.objects)
    for row in rows:
        # Every structural feature must be present
        for feat in STRUCTURAL_FEATURES:
            assert feat in row, f"missing feature: {feat}"
        # All values must be bool
        for feat in STRUCTURAL_FEATURES:
            assert isinstance(row[feat], bool), f"{feat} is {type(row[feat])}"


def test_extract_object_facts_singleton():
    """The 1x1 blue object should have is_singleton=True."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    # Find the singleton (size 1)
    singleton_rows = [r for r, o in zip(rows, facts.objects) if o.size == 1]
    assert len(singleton_rows) >= 1
    sr = singleton_rows[0]
    assert sr['is_singleton'] is True
    assert sr['interior'] is True
    assert sr['touches_boundary'] is False


def test_extract_object_facts_boundary():
    """Objects touching the grid boundary should have touches_boundary=True."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    for row, obj in zip(rows, facts.objects):
        expected = obj.touches_top or obj.touches_bottom or obj.touches_left or obj.touches_right
        assert row['touches_boundary'] == expected, (
            f"obj {obj.oid}: expected touches_boundary={expected}")
        assert row['interior'] == (not expected)


def test_extract_object_facts_largest_smallest():
    """is_largest and is_smallest should pick the correct objects."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    sizes = [o.size for o in facts.objects]
    max_size = max(sizes)
    min_size = min(sizes)

    largest_count = sum(1 for r in rows if r['is_largest'])
    smallest_count = sum(1 for r in rows if r['is_smallest'])
    assert largest_count >= 1
    assert smallest_count >= 1

    for row, obj in zip(rows, facts.objects):
        assert row['is_largest'] == (obj.size == max_size)
        assert row['is_smallest'] == (obj.size == min_size)


def test_extract_object_facts_color_features():
    """Per-color features should be present and correct."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    # Check that per-color features exist
    for row, obj in zip(rows, facts.objects):
        key = f'color_is_{obj.color}'
        assert key in row, f"missing per-color feature: {key}"
        assert row[key] is True


def test_extract_object_facts_uniqueness():
    """unique_color should be True iff the object is the only one with its color."""
    grid = _make_grid_with_objects()
    facts = perceive(grid)
    rows = extract_object_facts(facts)

    for row, obj in zip(rows, facts.objects):
        assert row['unique_color'] == (obj.n_same_color == 1)
        assert row['has_color_partner'] == (obj.n_same_color > 1)


def test_select_by_rule_trivial():
    """Rule selecting all singletons should work."""
    from aria.search.rules import DNFRule, ConjunctionRule, BoolAtom

    grid = _make_grid_with_objects()
    facts = perceive(grid)

    rule = DNFRule(clauses=(
        ConjunctionRule(atoms=(BoolAtom('is_singleton', True),)),
    ))

    selected = select_by_rule(rule.to_dict(), facts)
    assert all(o.size == 1 for o in selected)
    assert len(selected) >= 1


def test_select_by_rule_conjunction():
    """Rule with conjunction: interior AND is_singleton."""
    from aria.search.rules import DNFRule, ConjunctionRule, BoolAtom

    grid = _make_grid_with_objects()
    facts = perceive(grid)

    rule = DNFRule(clauses=(
        ConjunctionRule(atoms=(
            BoolAtom('interior', True),
            BoolAtom('is_singleton', True),
        )),
    ))

    selected = select_by_rule(rule.to_dict(), facts)
    for o in selected:
        assert o.size == 1
        touches = o.touches_top or o.touches_bottom or o.touches_left or o.touches_right
        assert not touches


def test_check_rule_on_object_consistency():
    """check_rule_on_object should agree with select_by_rule."""
    from aria.search.rules import DNFRule, ConjunctionRule, BoolAtom

    grid = _make_grid_with_objects()
    facts = perceive(grid)

    rule = DNFRule(clauses=(
        ConjunctionRule(atoms=(BoolAtom('is_rectangular', True),)),
    ))
    rule_dict = rule.to_dict()

    # Via select_by_rule
    selected_oids = set(o.oid for o in select_by_rule(rule_dict, facts))

    # Via check_rule_on_object (one at a time)
    for obj in facts.objects:
        result = check_rule_on_object(rule_dict, obj, facts.objects, facts.pairs)
        assert result == (obj.oid in selected_oids), (
            f"obj {obj.oid}: select_by_rule says {obj.oid in selected_oids}, "
            f"check_rule_on_object says {result}")
