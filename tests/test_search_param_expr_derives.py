"""Tests for ParamExpr-based derive strategies."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep, ParamExpr


def test_rank_recolor_expr():
    """Rank-based recolor: 3 objects recolored by size rank via ParamExpr."""
    from aria.search.derive import _derive_rank_recolor_expr
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    inp = np.zeros((9, 3), dtype=np.int8)
    inp[0:1, 0:1] = 1   # size 1 → rank 3
    inp[2:4, 0:2] = 2   # size 4 → rank 2
    inp[5:8, 0:3] = 3   # size 9 → rank 1

    out = np.zeros((9, 3), dtype=np.int8)
    out[0:1, 0:1] = 7   # rank 3 → color 7
    out[2:4, 0:2] = 8   # rank 2 → color 8
    out[5:8, 0:3] = 9   # rank 1 → color 9

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    progs = _derive_rank_recolor_expr(all_trans, all_facts, demos)
    assert len(progs) > 0
    p = progs[0]
    assert p.provenance == 'derive:rank_recolor_expr'
    assert np.array_equal(p.execute(inp), out)

    # Check that it uses ParamExpr
    step = p.steps[0]
    assert isinstance(step.params.get('color'), ParamExpr)


def test_rank_recolor_expr_multi_demo():
    """Rank recolor expr must be consistent across multiple demos."""
    from aria.search.derive import _derive_rank_recolor_expr
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    # Demo 1: two objects
    inp1 = np.zeros((4, 4), dtype=np.int8)
    inp1[0:1, 0:1] = 1  # size 1 → rank 2
    inp1[2:4, 2:4] = 2  # size 4 → rank 1
    out1 = np.zeros((4, 4), dtype=np.int8)
    out1[0:1, 0:1] = 5  # rank 2 → 5
    out1[2:4, 2:4] = 6  # rank 1 → 6

    # Demo 2: same rank→color mapping
    inp2 = np.zeros((6, 6), dtype=np.int8)
    inp2[0:2, 0:2] = 3  # size 4 → rank 1
    inp2[4:5, 4:5] = 4  # size 1 → rank 2
    out2 = np.zeros((6, 6), dtype=np.int8)
    out2[0:2, 0:2] = 6  # rank 1 → 6
    out2[4:5, 4:5] = 5  # rank 2 → 5

    demos = [(inp1, out1), (inp2, out2)]
    all_facts = [perceive(inp1), perceive(inp2)]
    all_trans = [
        compute_transitions(perceive(inp1), perceive(out1)),
        compute_transitions(perceive(inp2), perceive(out2)),
    ]

    progs = _derive_rank_recolor_expr(all_trans, all_facts, demos)
    assert len(progs) > 0
    for p in progs:
        assert p.verify(demos)


def test_field_move_expr():
    """Per-color move: different colors move different amounts."""
    from aria.search.derive import _derive_field_move_expr
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    inp = np.zeros((8, 8), dtype=np.int8)
    inp[0:2, 0:2] = 1  # color 1 at (0,0)
    inp[0:2, 5:7] = 2  # color 2 at (0,5)

    out = np.zeros((8, 8), dtype=np.int8)
    out[3:5, 0:2] = 1  # color 1 moved by (3,0)
    out[5:7, 5:7] = 2  # color 2 moved by (5,0)

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    progs = _derive_field_move_expr(all_trans, all_facts, demos)
    assert len(progs) > 0
    p = progs[0]
    assert p.provenance == 'derive:field_move_expr'
    assert np.array_equal(p.execute(inp), out)

    # Check ParamExpr usage
    step = p.steps[0]
    assert isinstance(step.params.get('dr'), ParamExpr)
    assert isinstance(step.params.get('dc'), ParamExpr)
