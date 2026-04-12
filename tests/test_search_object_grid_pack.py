"""Tests for object-grid-pack strategy."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_object_grid_pack_row_major():
    """Pack three 2x2 objects into a 2x6 row."""
    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1
    inp[0:2, 4:6] = 2
    inp[4:6, 0:2] = 3

    prog = SearchProgram(
        steps=[SearchStep('object_grid_pack', {
            'order': 'row_major', 'cell_h': 2, 'cell_w': 2,
            'out_rows': 1, 'out_cols': 3, 'sep': 0,
        })],
        provenance='test',
    )
    result = prog.execute(inp)

    assert result.shape == (2, 6)
    assert result[0:2, 0:2].sum() == 1 * 4
    assert result[0:2, 2:4].sum() == 2 * 4
    assert result[0:2, 4:6].sum() == 3 * 4


def test_object_grid_pack_size_desc():
    """Pack objects sorted by size descending."""
    inp = np.zeros((10, 10), dtype=np.int8)
    # Small 1x1
    inp[0, 0] = 1
    # Medium 2x2
    inp[5:7, 5:7] = 2
    # Large 3x3
    inp[7:10, 0:3] = 3

    prog = SearchProgram(
        steps=[SearchStep('object_grid_pack', {
            'order': 'size_desc', 'cell_h': 3, 'cell_w': 3,
            'out_rows': 1, 'out_cols': 3, 'sep': 0,
        })],
        provenance='test',
    )
    result = prog.execute(inp)

    assert result.shape == (3, 9)
    # First cell: largest (3x3 of 3s)
    assert result[0:3, 0:3].sum() == 3 * 9
    # Second cell: medium (2x2 of 2s)
    assert result[0:2, 3:5].sum() == 2 * 4
    # Third cell: small (1x1 of 1)
    assert result[0, 6] == 1


def test_object_grid_pack_derive():
    """Derive should find object_grid_pack on a synthetic demo."""
    from aria.search.derive import _derive_object_grid_pack_prescan

    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1
    inp[4:6, 4:6] = 2

    out = np.zeros((2, 4), dtype=np.int8)
    out[0:2, 0:2] = 1
    out[0:2, 2:4] = 2

    demos = [(inp, out)]
    progs = _derive_object_grid_pack_prescan(demos)
    assert any(p.provenance == 'derive:object_grid_pack' for p in progs)
