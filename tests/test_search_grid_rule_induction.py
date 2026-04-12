"""Tests for grid-conditional rule induction."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_parity_based_rule():
    """All empty cells filled from nearest in same row."""
    from aria.search.derive import _derive_grid_conditional_transfer
    from aria.guided.perceive import perceive

    # 2x3 grid: 2 cell-rows, 3 cell-cols, uniform 2x2 cells, sep=1
    inp = np.zeros((5, 8), dtype=np.int8)
    inp[2, :] = 5  # row sep
    inp[:, 2] = 5; inp[:, 5] = 5  # col seps
    # Row 0: left cell filled
    inp[0:2, 0:2] = 3
    # Row 1: left cell filled
    inp[3:5, 0:2] = 6

    out = inp.copy()
    # nearest_row fills: copy left into middle and right cells
    out[0:2, 3:5] = 3; out[0:2, 6:8] = 3
    out[3:5, 3:5] = 6; out[3:5, 6:8] = 6

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_conditional_transfer(facts, demos)
    assert len(progs) > 0
    for p in progs:
        assert np.array_equal(p.execute(inp), out)


def test_symmetry_mirror_hv():
    """Diagonal mirror (mirror_hv) across grid center."""
    # 2x2 grid, one cell filled
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = 5; grid[:, 2] = 5
    grid[0:2, 0:2] = 3  # top-left

    out = grid.copy()
    out[3:5, 3:5] = 3  # mirror_hv: bottom-right

    prog = SearchProgram(
        steps=[SearchStep('grid_conditional_transfer', {
            'rule': 'induced', 'cell_map': 'mirror_hv',
        })],
        provenance='test',
    )
    result = prog.execute(grid)
    assert np.array_equal(result, out)


def test_induced_rule_derive():
    """Derive finds mirror_hv via rule induction."""
    from aria.search.derive import _derive_grid_conditional_transfer
    from aria.guided.perceive import perceive

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[2, :] = 5; inp[:, 2] = 5
    inp[0:2, 0:2] = 3

    out = inp.copy()
    out[3:5, 3:5] = 3  # only mirror_hv produces this

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_conditional_transfer(facts, demos)
    assert len(progs) > 0
    for p in progs:
        assert np.array_equal(p.execute(inp), out)
