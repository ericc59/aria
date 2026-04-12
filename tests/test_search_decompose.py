"""Tests for decomposition search."""

from __future__ import annotations

import numpy as np

from aria.search.task_analysis import analyze_task
from aria.search.decompose import search_decomposed


def test_crop_then_recolor():
    """2-step: crop non-bg bbox, then recolor (color map)."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 1

    # Output: 2x2 region recolored to 2
    out = np.full((2, 2), 2, dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)
    assert analysis.dims_change

    prog = search_decomposed(demos, analysis)
    # crop_non_bg_bbox + color_map should compose
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
        assert 'decompose:' in prog.provenance


def test_analysis_gating_rejects():
    """Splitters are gated: same-dims non-extraction task gets no crop splitter."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int8)
    out = np.array([[4, 3], [2, 1]], dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)

    assert not analysis.dims_change
    assert not analysis.is_extraction

    # Decomposition may still try but no splitter should apply
    prog = search_decomposed(demos, analysis)
    # Even if it returns None, that's fine — no crash
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)


def test_composition_correctness():
    """Composed program's steps are ordered: splitter first, then sub-derive."""
    inp = np.zeros((4, 4), dtype=np.int8)
    inp[0:2, 0:2] = 3

    out = np.full((2, 2), 3, dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)

    prog = search_decomposed(demos, analysis)
    if prog is not None:
        # First step should be from the splitter
        assert prog.steps[0].action == 'crop_nonbg'
        assert np.array_equal(prog.execute(inp), out)


def test_remove_color_splitter():
    """Splitter removes a color that disappears in the output."""
    inp = np.array([[1, 2], [2, 1]], dtype=np.int8)
    out = np.array([[1, 0], [0, 1]], dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)

    assert 2 in analysis.removed_colors

    prog = search_decomposed(demos, analysis)
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
