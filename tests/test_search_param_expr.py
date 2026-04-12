"""Tests for ParamExpr evaluation."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import ParamExpr
from aria.search.executor import eval_param_expr
from aria.guided.perceive import perceive


def _make_scene():
    """Create a simple scene with 3 objects for testing."""
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0:1, 0:1] = 1  # 1x1, color 1, size 1
    grid[2:4, 2:4] = 2  # 2x2, color 2, size 4
    grid[4:6, 0:3] = 3  # 2x3, color 3, size 6
    facts = perceive(grid)
    return facts


def test_const():
    facts = _make_scene()
    obj = facts.objects[0]
    assert eval_param_expr(ParamExpr('const', (42,)), obj, facts) == 42


def test_field_color():
    facts = _make_scene()
    for obj in facts.objects:
        result = eval_param_expr(ParamExpr('field', ('color',)), obj, facts)
        assert result == obj.color


def test_field_size():
    facts = _make_scene()
    for obj in facts.objects:
        result = eval_param_expr(ParamExpr('field', ('size',)), obj, facts)
        assert result == obj.size


def test_rank_by_size():
    facts = _make_scene()
    # Sort objects by size desc: 6, 4, 1
    objs_by_size = sorted(facts.objects, key=lambda o: -o.size)
    for obj in facts.objects:
        rank = eval_param_expr(ParamExpr('rank', ('size',)), obj, facts)
        expected = [o.size for o in objs_by_size].index(obj.size) + 1
        assert rank == expected


def test_mod():
    facts = _make_scene()
    for obj in facts.objects:
        result = eval_param_expr(ParamExpr('mod', ('size', 3)), obj, facts)
        assert result == obj.size % 3


def test_count_all():
    facts = _make_scene()
    obj = facts.objects[0]
    result = eval_param_expr(ParamExpr('count', ('all',)), obj, facts)
    assert result == len(facts.objects)


def test_count_rectangular():
    facts = _make_scene()
    obj = facts.objects[0]
    result = eval_param_expr(ParamExpr('count', ('is_rectangular',)), obj, facts)
    n_rect = sum(1 for o in facts.objects if o.is_rectangular)
    assert result == n_rect


def test_lookup():
    facts = _make_scene()
    table = {1: 9, 2: 8, 3: 7}
    for obj in facts.objects:
        result = eval_param_expr(
            ParamExpr('lookup', ('color', table)), obj, facts)
        assert result == table[obj.color]


def test_literal_passthrough():
    """Non-ParamExpr values pass through."""
    facts = _make_scene()
    obj = facts.objects[0]
    assert eval_param_expr(5, obj, facts) == 5


def test_serialization():
    expr = ParamExpr('lookup', ('color', {1: 9, 2: 8}))
    d = expr.to_dict()
    assert d['op'] == 'lookup'
    restored = ParamExpr.from_dict(d)
    assert restored.op == 'lookup'
