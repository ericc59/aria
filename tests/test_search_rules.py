from __future__ import annotations

from aria.search.rules import DNFRule, eval_rule, induce_boolean_dnf


def test_induce_boolean_dnf_for_workspace_highlight_rule():
    rows = [
        {"any_match": True, "full_match": False, "fewer_motifs": False, "color_disjoint": False},
        {"any_match": False, "full_match": False, "fewer_motifs": True, "color_disjoint": True},
        {"any_match": False, "full_match": False, "fewer_motifs": True, "color_disjoint": False},
        {"any_match": False, "full_match": False, "fewer_motifs": False, "color_disjoint": True},
        {"any_match": True, "full_match": True, "fewer_motifs": True, "color_disjoint": True},
    ]
    labels = [True, True, False, False, True]

    rule = induce_boolean_dnf(
        rows,
        labels,
        candidate_fields=["any_match", "full_match", "fewer_motifs", "color_disjoint"],
        max_clause_size=2,
        max_clauses=2,
    )

    assert rule is not None
    assert all(eval_rule(rule, row) == label for row, label in zip(rows, labels))
    assert eval_rule(rule, {"any_match": False, "full_match": False, "fewer_motifs": True, "color_disjoint": True})
    assert not eval_rule(rule, {"any_match": False, "full_match": False, "fewer_motifs": False, "color_disjoint": False})


def test_induce_boolean_dnf_for_p0_rule_and_roundtrip():
    rows = [
        {"any_full_match": True, "all_ws_highlight": False},
        {"any_full_match": False, "all_ws_highlight": True},
        {"any_full_match": False, "all_ws_highlight": False},
    ]
    labels = [True, True, False]

    rule = induce_boolean_dnf(
        rows,
        labels,
        candidate_fields=["any_full_match", "all_ws_highlight"],
        max_clause_size=2,
        max_clauses=2,
    )

    assert rule is not None
    encoded = rule.to_dict()
    roundtrip = DNFRule.from_dict(encoded)
    assert all(eval_rule(roundtrip, row) == label for row, label in zip(rows, labels))
    assert eval_rule(roundtrip, {"any_full_match": True, "all_ws_highlight": True})
    assert not eval_rule(roundtrip, {"any_full_match": False, "all_ws_highlight": False})


def test_induce_boolean_dnf_handles_constant_false_and_true():
    false_rule = induce_boolean_dnf(
        [{"x": False}, {"x": True}],
        [False, False],
        candidate_fields=["x"],
    )
    true_rule = induce_boolean_dnf(
        [{"x": False}, {"x": True}],
        [True, True],
        candidate_fields=["x"],
    )

    assert false_rule is not None
    assert true_rule is not None
    assert not eval_rule(false_rule, {"x": False})
    assert not eval_rule(false_rule, {"x": True})
    assert eval_rule(true_rule, {"x": False})
    assert eval_rule(true_rule, {"x": True})
