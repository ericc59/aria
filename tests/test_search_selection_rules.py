"""Tests for selector rule induction: cross-demo exact selection."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive
from aria.search.selection_facts import extract_object_facts
from aria.search.rules import induce_boolean_dnf


def test_induce_single_predicate():
    """A single boolean feature should be found when it suffices."""
    rows = [
        {'is_rectangular': True, 'is_line': False, 'interior': True},
        {'is_rectangular': True, 'is_line': False, 'interior': False},
        {'is_rectangular': False, 'is_line': True, 'interior': True},
        {'is_rectangular': False, 'is_line': False, 'interior': False},
    ]
    labels = [True, True, False, False]

    rule = induce_boolean_dnf(
        rows, labels,
        candidate_fields=['is_rectangular', 'is_line', 'interior'],
        max_clause_size=3,
        max_clauses=2,
    )
    assert rule is not None
    # Should find is_rectangular=True as the simplest rule
    assert len(rule.clauses) == 1
    assert len(rule.clauses[0].atoms) == 1
    assert rule.clauses[0].atoms[0].field == 'is_rectangular'
    assert rule.clauses[0].atoms[0].value is True


def test_induce_conjunction():
    """A conjunction of two features should be found when single fails."""
    rows = [
        {'A': True, 'B': True, 'C': False},   # target
        {'A': True, 'B': False, 'C': True},    # not target
        {'A': False, 'B': True, 'C': True},    # not target
        {'A': False, 'B': False, 'C': False},  # not target
    ]
    labels = [True, False, False, False]

    rule = induce_boolean_dnf(
        rows, labels,
        candidate_fields=['A', 'B', 'C'],
        max_clause_size=3,
        max_clauses=2,
    )
    assert rule is not None
    # Should need A=True AND B=True (neither alone suffices)
    for row, label in zip(rows, labels):
        assert rule.matches(row) == label


def test_induce_negation():
    """Negated features (value=False) should work."""
    rows = [
        {'is_singleton': False, 'touches_boundary': True},   # target
        {'is_singleton': False, 'touches_boundary': False},   # target
        {'is_singleton': True, 'touches_boundary': True},     # not target
    ]
    labels = [True, True, False]

    rule = induce_boolean_dnf(
        rows, labels,
        candidate_fields=['is_singleton', 'touches_boundary'],
        max_clause_size=2,
        max_clauses=2,
    )
    assert rule is not None
    # Should find is_singleton=False
    for row, label in zip(rows, labels):
        assert rule.matches(row) == label


def test_induce_disjunction():
    """DNF with two clauses (disjunction of conjunctions)."""
    rows = [
        {'A': True, 'B': False},   # target (A=True)
        {'A': False, 'B': True},   # target (B=True)
        {'A': False, 'B': False},  # not target
    ]
    labels = [True, True, False]

    rule = induce_boolean_dnf(
        rows, labels,
        candidate_fields=['A', 'B'],
        max_clause_size=1,
        max_clauses=2,
    )
    assert rule is not None
    for row, label in zip(rows, labels):
        assert rule.matches(row) == label


def test_cross_demo_consistency():
    """Selector rule should work across multiple "demos" (pooled rows)."""
    # Simulating 2 demos with different object populations
    # Demo 0: rect objects are targets
    demo0_rows = [
        {'is_rectangular': True, 'interior': True, 'unique_color': False},
        {'is_rectangular': True, 'interior': False, 'unique_color': True},
        {'is_rectangular': False, 'interior': True, 'unique_color': False},
    ]
    demo0_labels = [True, True, False]

    # Demo 1: rect objects are also targets (consistent rule)
    demo1_rows = [
        {'is_rectangular': True, 'interior': False, 'unique_color': False},
        {'is_rectangular': False, 'interior': True, 'unique_color': True},
        {'is_rectangular': False, 'interior': False, 'unique_color': False},
    ]
    demo1_labels = [True, False, False]

    all_rows = demo0_rows + demo1_rows
    all_labels = demo0_labels + demo1_labels

    rule = induce_boolean_dnf(
        all_rows, all_labels,
        candidate_fields=['is_rectangular', 'interior', 'unique_color'],
        max_clause_size=3,
        max_clauses=2,
    )
    assert rule is not None
    for row, label in zip(all_rows, all_labels):
        assert rule.matches(row) == label


def test_find_selector_with_rule_induction():
    """_find_selector should fall back to rule induction for conjunctions."""
    from aria.search.derive import _find_selector

    # Build a grid where only rectangular interior objects should be selected
    grid = np.zeros((12, 12), dtype=np.int8)
    # Rectangular interior objects (targets)
    grid[3:5, 3:5] = 2
    grid[6:8, 6:8] = 3
    # Non-rectangular interior object
    grid[3, 7] = 4
    grid[4, 6] = 4
    grid[4, 8] = 4
    grid[5, 7] = 4
    # Rectangular border object (not target - touches border)
    grid[0, 4:6] = 5
    grid[1, 4:6] = 5

    facts = perceive(grid)

    # Get OIDs for rectangular interior objects
    target_oids = set()
    for obj in facts.objects:
        touches = (obj.touches_top or obj.touches_bottom
                   or obj.touches_left or obj.touches_right)
        if obj.is_rectangular and not touches:
            target_oids.add(obj.oid)

    assert len(target_oids) >= 2, "need at least 2 target objects"

    sel = _find_selector(target_oids, facts)
    assert sel is not None, "selector should be found via rule induction"


def test_find_selector_cross_demo():
    """Cross-demo selector rule induction should handle multi-demo pooling."""
    from aria.search.derive import _find_selector_cross_demo
    from aria.guided.synthesize import compute_transitions

    # Demo 0: move rectangular objects
    inp0 = np.zeros((8, 8), dtype=np.int8)
    inp0[1:3, 1:3] = 2   # rect, interior → will "move"
    inp0[1, 5] = 3        # singleton → stays
    out0 = np.zeros((8, 8), dtype=np.int8)
    out0[2:4, 1:3] = 2    # moved down 1
    out0[1, 5] = 3

    # Demo 1: same rule
    inp1 = np.zeros((8, 8), dtype=np.int8)
    inp1[2:4, 3:5] = 4   # rect, interior → will "move"
    inp1[6, 6] = 5        # singleton → stays
    out1 = np.zeros((8, 8), dtype=np.int8)
    out1[3:5, 3:5] = 4    # moved down 1
    out1[6, 6] = 5

    facts0 = perceive(inp0)
    facts1 = perceive(inp1)
    trans0 = compute_transitions(facts0, perceive(out0))
    trans1 = compute_transitions(facts1, perceive(out1))

    sel = _find_selector_cross_demo('moved', [trans0, trans1], [facts0, facts1])
    # Should find a rule that selects rectangular (or equivalent) objects
    assert sel is not None, "cross-demo selector should be found"


def test_by_rule_round_trip():
    """by_rule selector should survive to_dict / from_dict / to_predicates."""
    from aria.search.sketch import StepSelect
    from aria.guided.clause import Predicate, Pred

    rule_dict = {
        'kind': 'dnf',
        'clauses': [{'atoms': [
            {'field': 'is_rectangular', 'value': True},
            {'field': 'interior', 'value': True},
        ]}],
    }
    sel = StepSelect('by_rule', {'rule': rule_dict})

    # to_dict round trip
    d = sel.to_dict()
    sel2 = StepSelect.from_dict(d)
    assert sel2.role == 'by_rule'
    assert sel2.params['rule'] == rule_dict

    # to_predicates should return SELECTION_RULE predicate
    preds = sel2.to_predicates()
    assert len(preds) == 1
    assert preds[0].pred == Pred.SELECTION_RULE
