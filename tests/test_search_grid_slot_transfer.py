"""Tests for grid-slot transfer."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_grid_slot_transfer_separator():
    """Separator grid: move top row contents to bottom row."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = 5
    grid[:, 2] = 5
    # top-left cell filled with 3s, top-right filled with 4s
    grid[0:2, 0:2] = 3
    grid[0:2, 3:5] = 4

    prog = SearchProgram(
        steps=[SearchStep('grid_slot_transfer', {})],
        provenance='test',
    )
    result = prog.execute(grid)

    # contents moved to bottom row cells
    assert result[3:5, 0:2].sum() == 3 * 4
    assert result[3:5, 3:5].sum() == 4 * 4
    # source cells emptied
    assert result[0:2, 0:2].sum() == 0
    assert result[0:2, 3:5].sum() == 0


def test_grid_slot_transfer_implicit():
    """Implicit grid: move diagonal cells to the other diagonal."""
    grid = np.zeros((4, 6), dtype=np.int8)
    # cell size 2x3, positions (0,0),(0,3),(2,0),(2,3)
    grid[0:2, 0:3] = 3
    grid[2:4, 3:6] = 4

    prog = SearchProgram(
        steps=[SearchStep('grid_slot_transfer', {})],
        provenance='test',
    )
    result = prog.execute(grid)

    # contents should swap to the empty cells
    assert result[2:4, 0:3].sum() == 3 * 6
    assert result[0:2, 3:6].sum() == 4 * 6


def test_grid_slot_transfer_derive():
    """Derive should find grid_slot_transfer on a synthetic demo."""
    from aria.search.derive import _derive_grid_slot_transfer
    from aria.guided.perceive import perceive

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[2, :] = 5
    inp[:, 2] = 5
    inp[0:2, 0:2] = 3
    inp[0:2, 3:5] = 4

    out = np.zeros((5, 5), dtype=np.int8)
    out[2, :] = 5
    out[:, 2] = 5
    out[3:5, 0:2] = 3
    out[3:5, 3:5] = 4

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_slot_transfer(facts, demos)
    assert any(p.provenance == 'derive:grid_slot_transfer' for p in progs)
