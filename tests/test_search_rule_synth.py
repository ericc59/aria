"""Tests for rule synthesis v1 (ParamExpr extensions)."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import ParamExpr
from aria.search.executor import eval_param_expr
from aria.guided.perceive import perceive


def test_neighbor_count_any():
    """neighbor_count('any') counts non-bg 4-connected neighbors."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 1] = 1  # center object
    grid[0, 1] = 2  # neighbor above
    grid[2, 1] = 3  # neighbor below
    # left and right are bg

    facts = perceive(grid)
    # Find the center object (color 1)
    obj = [o for o in facts.objects if o.color == 1][0]
    context = {'selected_objects': facts.objects, 'grid': grid}

    expr = ParamExpr('neighbor_count', ('any',))
    count = eval_param_expr(expr, obj, facts, context)
    assert count == 2  # two non-bg neighbors


def test_rank_asc():
    """rank with order='asc' ranks smallest first."""
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0, 0] = 1        # size 1
    grid[2:4, 2:4] = 2    # size 4

    facts = perceive(grid)
    small = [o for o in facts.objects if o.color == 1][0]
    large = [o for o in facts.objects if o.color == 2][0]
    context = {'selected_objects': facts.objects, 'grid': grid}

    expr = ParamExpr('rank', ('size', 'asc'))
    assert eval_param_expr(expr, small, facts, context) == 1  # smallest = rank 1
    assert eval_param_expr(expr, large, facts, context) == 2


def test_rank_by_row():
    """rank('row', 'desc') ranks topmost object last."""
    grid = np.zeros((6, 3), dtype=np.int8)
    grid[0:2, 0:2] = 1  # row 0
    grid[4:6, 0:2] = 2  # row 4

    facts = perceive(grid)
    top = [o for o in facts.objects if o.color == 1][0]
    bot = [o for o in facts.objects if o.color == 2][0]
    context = {'selected_objects': facts.objects, 'grid': grid}

    expr = ParamExpr('rank', ('row', 'desc'))
    assert eval_param_expr(expr, bot, facts, context) == 1  # row 4 is largest
    assert eval_param_expr(expr, top, facts, context) == 2


def test_recolor_by_neighbor_count_derive():
    """Derive recolor_by_neighbor_count on a synthetic demo."""
    from aria.search.derive import _derive_recolor_by_neighbor_count
    from aria.guided.synthesize import compute_transitions

    # Objects at separate positions, same shape, recolored by neighbor count.
    # A (color 1) at (0,0): isolated → 0 neighbors → color 5
    # B (color 2) at (2,0): has non-bg neighbor at (2,1)=fixed_color → 1 neighbor → color 6
    # C (color 3) at (4,2): has neighbors at (4,1) and (4,3) → 2 neighbors → color 7
    # The "neighbors" are fixed objects that don't change color.
    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0, 0] = 1  # A: isolated, 0 external neighbors
    inp[2, 0] = 2  # B
    inp[2, 1] = 8  # fixed neighbor of B (stays same color)
    inp[4, 2] = 3  # C
    inp[4, 1] = 8  # fixed neighbor of C
    inp[4, 3] = 8  # fixed neighbor of C

    out = inp.copy()
    out[0, 0] = 5  # A: 0 neighbors → 5
    out[2, 0] = 6  # B: 1 neighbor → 6
    out[4, 2] = 7  # C: 2 neighbors → 7

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    progs = _derive_recolor_by_neighbor_count(all_trans, all_facts, demos)
    assert len(progs) > 0
    p = progs[0]
    assert 'neighbor_count' in p.provenance
    assert np.array_equal(p.execute(inp), out)


def test_rank_recolor_by_row():
    """Rank recolor by row should work when size-based fails."""
    from aria.search.derive import _derive_rank_recolor_expr
    from aria.guided.synthesize import compute_transitions

    inp = np.zeros((6, 3), dtype=np.int8)
    inp[0:2, 0:2] = 1  # row 0, size 4 → color 7
    inp[4:6, 0:2] = 2  # row 4, size 4 → color 8

    out = np.zeros((6, 3), dtype=np.int8)
    out[0:2, 0:2] = 7
    out[4:6, 0:2] = 8

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    progs = _derive_rank_recolor_expr(all_trans, all_facts, demos)
    assert len(progs) > 0
    p = progs[0]
    assert np.array_equal(p.execute(inp), out)
    assert 'rank_recolor_expr' in p.provenance
