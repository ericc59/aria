"""Tests for panel-legend-map strategy."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_panel_legend_map_left():
    """Legend on left, color mapping applied to right panel."""
    # Layout: 3x1 legend | sep | 3x3 target
    inp = np.zeros((3, 5), dtype=np.int8)
    # Column separator at col 1
    inp[:, 1] = 5
    # Legend on left (col 0): three colors stacked
    inp[0, 0] = 1
    inp[1, 0] = 2
    inp[2, 0] = 3
    # Target on right (cols 2-4): has color 1 that should become 2
    inp[0, 2] = 1
    inp[1, 3] = 1

    out = inp.copy()
    out[0, 2] = 2  # 1 → 2
    out[1, 3] = 2

    prog = SearchProgram(
        steps=[SearchStep('panel_legend_map', {
            'axis': 'col', 'sep_idx': 1,
            'legend_side': 'left', 'mapping': {'1': 2},
        })],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out)


def test_panel_legend_map_top():
    """Legend on top, color mapping applied to bottom panel."""
    inp = np.zeros((5, 3), dtype=np.int8)
    inp[1, :] = 5  # row separator
    inp[0, 0] = 1
    inp[0, 1] = 2
    inp[0, 2] = 3
    # Target below
    inp[2, 0] = 1
    inp[3, 1] = 1
    inp[4, 2] = 1

    out = inp.copy()
    out[2, 0] = 3  # 1 → 3
    out[3, 1] = 3
    out[4, 2] = 3

    prog = SearchProgram(
        steps=[SearchStep('panel_legend_map', {
            'axis': 'row', 'sep_idx': 1,
            'legend_side': 'top', 'mapping': {'1': 3},
        })],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out)


def test_panel_legend_map_derive():
    """Derive should find panel_legend_map."""
    from aria.search.derive import _derive_panel_legend_map
    from aria.guided.perceive import perceive

    inp = np.zeros((3, 5), dtype=np.int8)
    inp[:, 1] = 5
    inp[0, 0] = 1
    inp[1, 0] = 2
    inp[0, 2] = 1
    inp[1, 3] = 1

    out = inp.copy()
    out[0, 2] = 2
    out[1, 3] = 2

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_panel_legend_map(facts, demos)
    assert any(p.provenance == 'derive:panel_legend_map' for p in progs)
