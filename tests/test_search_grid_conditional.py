"""Tests for grid-conditional transfer."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_grid_conditional_nearest_row():
    """Fill empty cells from nearest non-empty in same row."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = 5  # row separator
    grid[:, 2] = 5  # col separator
    grid[0:2, 0:2] = 3  # top-left

    prog = SearchProgram(
        steps=[SearchStep('grid_conditional_transfer', {'rule': 'nearest_row'})],
        provenance='test',
    )
    result = prog.execute(grid)

    # top-right should be filled with 3 (nearest in same row)
    assert result[0:2, 3:5].sum() == 3 * 4


def test_grid_conditional_mirror_h():
    """Fill empty cells via horizontal mirror."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = 5
    grid[:, 2] = 5
    grid[0:2, 0:2] = 3
    grid[3:5, 0:2] = 4

    prog = SearchProgram(
        steps=[SearchStep('grid_conditional_transfer', {'rule': 'mirror_h'})],
        provenance='test',
    )
    result = prog.execute(grid)

    # right-side mirrors left-side
    assert result[0:2, 3:5].sum() == 3 * 4
    assert result[3:5, 3:5].sum() == 4 * 4


def test_grid_conditional_derive_nearest_row():
    """Derive should find nearest_row rule."""
    from aria.search.derive import _derive_grid_conditional_transfer
    from aria.guided.perceive import perceive

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[2, :] = 5
    inp[:, 2] = 5
    inp[0:2, 0:2] = 3

    out = inp.copy()
    out[0:2, 3:5] = 3  # right side filled from nearest in row

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_conditional_transfer(facts, demos)
    assert any(p.provenance == 'derive:grid_conditional_transfer' for p in progs)


def test_grid_conditional_derive_mirror_v():
    """Derive should find mirror_v rule."""
    from aria.search.derive import _derive_grid_conditional_transfer
    from aria.guided.perceive import perceive

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[2, :] = 5
    inp[:, 2] = 5
    inp[0:2, 0:2] = 3
    inp[0:2, 3:5] = 4

    out = inp.copy()
    out[3:5, 0:2] = 3  # bottom mirrors top
    out[3:5, 3:5] = 4

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_conditional_transfer(facts, demos)
    assert any(p.provenance == 'derive:grid_conditional_transfer' for p in progs)


def test_grid_conditional_rejects_filled_cell_change():
    """Derive must reject if a filled cell changes between input and output."""
    from aria.search.derive import _derive_grid_conditional_transfer
    from aria.guided.perceive import perceive

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[2, :] = 5
    inp[:, 2] = 5
    inp[0:2, 0:2] = 3

    out = inp.copy()
    out[0:2, 3:5] = 3  # fill empty cell (valid)
    out[0:2, 0:2] = 7  # mutate filled cell (invalid)

    demos = [(inp, out)]
    facts = [perceive(inp)]
    progs = _derive_grid_conditional_transfer(facts, demos)
    assert len(progs) == 0
