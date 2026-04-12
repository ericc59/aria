"""Tests for panel routing v2 (legend → panel mapping)."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_legend_left_two_pairs():
    """Legend on left with two color pairs: 1→2 and 3→4."""
    # Layout: 4x2 legend | sep col 2 | 4x3 target
    inp = np.zeros((4, 6), dtype=np.int8)
    inp[:, 2] = 5  # separator
    # Legend: rows with pair entries
    inp[0, 0] = 1; inp[0, 1] = 2  # pair 1→2
    inp[1, 0] = 3; inp[1, 1] = 4  # pair 3→4
    # Target: colors 1 and 3 present
    inp[0, 3] = 1
    inp[1, 4] = 3
    inp[2, 5] = 1
    inp[3, 3] = 3

    out = inp.copy()
    out[0, 3] = 2  # 1→2
    out[1, 4] = 4  # 3→4
    out[2, 5] = 2
    out[3, 3] = 4

    prog = SearchProgram(
        steps=[SearchStep('panel_legend_map', {
            'axis': 'col', 'sep_idx': 2,
            'legend_side': 'left', 'mapping': {'1': 2, '3': 4},
        })],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out)


def test_legend_top_maps_1_to_3():
    """Legend on top maps 1→3."""
    inp = np.zeros((5, 3), dtype=np.int8)
    inp[1, :] = 5  # row separator
    # Legend: row 0 has pair [1, 3, 0]
    inp[0, 0] = 1
    inp[0, 1] = 3
    # Target below
    inp[2, 0] = 1
    inp[3, 1] = 1

    out = inp.copy()
    out[2, 0] = 3
    out[3, 1] = 3

    prog = SearchProgram(
        steps=[SearchStep('panel_legend_map', {
            'axis': 'row', 'sep_idx': 1,
            'legend_side': 'top', 'mapping': {'1': 3},
        })],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out)


def test_legend_derive_two_pairs():
    """Derive finds panel_legend_map from legend pairs."""
    from aria.search.derive import _derive_panel_legend_map
    from aria.guided.perceive import perceive

    inp = np.zeros((4, 6), dtype=np.int8)
    inp[:, 2] = 5
    inp[0, 0] = 1; inp[0, 1] = 2
    inp[1, 0] = 3; inp[1, 1] = 4
    inp[0, 3] = 1
    inp[1, 4] = 3

    out = inp.copy()
    out[0, 3] = 2
    out[1, 4] = 4

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_panel_legend_map(facts, demos)
    assert any(p.provenance == 'derive:panel_legend_map' for p in progs)
    # Verify the derived program reproduces the output
    for p in progs:
        if p.provenance == 'derive:panel_legend_map':
            assert np.array_equal(p.execute(inp), out)


def test_legend_rejects_source_color_not_in_legend():
    """Mapping must be rejected if a source color doesn't appear in legend."""
    from aria.search.derive import _try_legend_mapping
    from aria.guided.perceive import perceive

    # Legend has colors 1,2 but target changes color 7→2
    inp = np.zeros((3, 5), dtype=np.int8)
    inp[:, 1] = 5
    inp[0, 0] = 1
    inp[1, 0] = 2
    inp[0, 2] = 7  # color 7 NOT in legend
    inp[1, 3] = 7

    out = inp.copy()
    out[0, 2] = 2  # 7→2 — but 7 not in legend
    out[1, 3] = 2

    facts = [perceive(inp)]
    result = _try_legend_mapping(facts, [(inp, out)], 'col', 1, 'left')
    assert result is None
