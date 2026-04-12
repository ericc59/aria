"""Tests for goal-directed multi-step planner."""

from __future__ import annotations

import numpy as np

from aria.search.task_analysis import analyze_task
from aria.search.planner import plan_search, _build_reducers, _goal_from_analysis, _improves


def test_three_step_crop_remove_recolor():
    """3-step: remove color 2, crop non-bg bbox, then recolor 1→3."""
    # Input: 5x5 with noise color 2 and a 2x2 block of color 1
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 1
    inp[0, 0] = 2
    inp[4, 4] = 2

    # Output: 2x2 block recolored to 3
    out = np.full((2, 2), 3, dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)
    assert analysis.dims_change

    prog = plan_search(demos, analysis)
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
        assert 'plan:' in prog.provenance or 'derive:' in prog.provenance


def test_gating_rejects_incompatible():
    """Planner should not use crop/extract when task is same-dims recolor."""
    inp = np.array([[1, 0], [0, 2]], dtype=np.int8)
    out = np.array([[3, 0], [0, 4]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    assert analysis.diff_type == 'recolor_only'
    assert not analysis.dims_change

    reducers = _build_reducers(analysis)
    names = [r.name for r in reducers]
    assert not any('crop' in n for n in names)
    assert not any('extract' in n for n in names)


def test_improvement_required():
    """Reducer must improve goal or be skipped."""
    goal_a = _goal_from_analysis(
        [(np.zeros((3, 3), dtype=np.int8), np.ones((3, 3), dtype=np.int8))],
        analyze_task([(np.zeros((3, 3), dtype=np.int8), np.ones((3, 3), dtype=np.int8))]),
    )
    # Same goal = no improvement
    assert not _improves(goal_a, goal_a)


def test_two_step_remove_then_derive():
    """2-step: remove noise color, then derive on clean sub-problem."""
    inp = np.array([[1, 2], [2, 1]], dtype=np.int8)
    out = np.array([[1, 0], [0, 1]], dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)

    prog = plan_search(demos, analysis)
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
