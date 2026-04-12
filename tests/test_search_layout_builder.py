"""Tests for layout builder (object_grid_pack extensions)."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_pack_col_major():
    """Pack objects by column-major order."""
    inp = np.zeros((6, 6), dtype=np.int8)
    # Object at (0,0) → column 0 first
    inp[0:2, 0:2] = 1
    # Object at (4,0) → still column 0, row 4
    inp[4:6, 0:2] = 2
    # Object at (0,4) → column 4
    inp[0:2, 4:6] = 3

    # col_major order: (0,0)=1, (4,0)=2, (0,4)=3
    # Pack into 1 row, 3 cols, cell 2x2, sep=0
    prog = SearchProgram(
        steps=[SearchStep('object_grid_pack', {
            'order': 'col_major', 'cell_h': 2, 'cell_w': 2,
            'out_rows': 1, 'out_cols': 3, 'sep': 0,
            'placement': 'top_left',
        })],
        provenance='test',
    )
    result = prog.execute(inp)

    assert result.shape == (2, 6)
    assert result[0:2, 0:2].sum() == 1 * 4  # first: (0,0)
    assert result[0:2, 2:4].sum() == 2 * 4  # second: (4,0)
    assert result[0:2, 4:6].sum() == 3 * 4  # third: (0,4)


def test_pack_size_desc_with_sep():
    """Pack objects by size descending with separator width 1."""
    inp = np.zeros((10, 10), dtype=np.int8)
    inp[0, 0] = 1           # 1x1
    inp[5:7, 5:7] = 2       # 2x2
    inp[7:10, 0:3] = 3      # 3x3

    prog = SearchProgram(
        steps=[SearchStep('object_grid_pack', {
            'order': 'size_desc', 'cell_h': 3, 'cell_w': 3,
            'out_rows': 1, 'out_cols': 3, 'sep': 1,
            'placement': 'top_left',
        })],
        provenance='test',
    )
    result = prog.execute(inp)

    # 1 row, 3 cols, cell 3x3, sep 1 → 3 x (3+1+3+1+3) = 3 x 11
    assert result.shape == (3, 11)
    # First cell: 3x3 of 3s (largest)
    assert result[0:3, 0:3].sum() == 3 * 9
    # Second cell: starts at col 4, 2x2 of 2s
    assert result[0:2, 4:6].sum() == 2 * 4
    # Third cell: starts at col 8, 1x1 of 1
    assert result[0, 8] == 1


def test_pack_centered_placement():
    """Pack with centered placement."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[0, 0] = 7  # 1x1 object

    prog = SearchProgram(
        steps=[SearchStep('object_grid_pack', {
            'order': 'row_major', 'cell_h': 3, 'cell_w': 3,
            'out_rows': 1, 'out_cols': 1, 'sep': 0,
            'placement': 'centered',
        })],
        provenance='test',
    )
    result = prog.execute(inp)

    assert result.shape == (3, 3)
    # Centered: (3-1)//2 = 1, so pixel at (1,1)
    assert result[1, 1] == 7
    assert result[0, 0] == 0


def test_pack_derive_col_major():
    """Derive finds col_major ordering."""
    from aria.search.derive import _derive_object_grid_pack_prescan

    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1  # col 0
    inp[4:6, 0:2] = 2  # col 0
    inp[0:2, 4:6] = 3  # col 4

    # col_major: 1, 2, 3 in a row
    out = np.zeros((2, 6), dtype=np.int8)
    out[0:2, 0:2] = 1
    out[0:2, 2:4] = 2
    out[0:2, 4:6] = 3

    demos = [(inp, out)]
    progs = _derive_object_grid_pack_prescan(demos)
    assert any(p.provenance == 'derive:object_grid_pack' for p in progs)
    for p in progs:
        if p.provenance == 'derive:object_grid_pack':
            assert np.array_equal(p.execute(inp), out)
