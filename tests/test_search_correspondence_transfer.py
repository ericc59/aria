"""Tests for correspondence-driven placement (correspondence_transfer)."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_swap_two_objects_position():
    """Two same-shape different-color objects swap positions."""
    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1  # object A at top-left
    inp[4:6, 4:6] = 2  # object B at bottom-right

    out = np.zeros((6, 6), dtype=np.int8)
    out[4:6, 4:6] = 1  # A moved to B's position
    out[0:2, 0:2] = 2  # B moved to A's position

    prog = SearchProgram(
        steps=[SearchStep('correspondence_transfer', {'mode': 'position_swap'})],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out)


def test_swap_color_rotation():
    """Three same-shape objects rotate colors."""
    inp = np.zeros((9, 3), dtype=np.int8)
    inp[0:2, 0:2] = 1
    inp[3:5, 0:2] = 2
    inp[6:8, 0:2] = 3

    prog = SearchProgram(
        steps=[SearchStep('correspondence_transfer', {'mode': 'color_permutation',
                                                     'mapping': {'1': 2, '2': 3, '3': 1}})],
        provenance='test',
    )
    result = prog.execute(inp)

    # Colors should be a non-identity permutation
    pos_colors = (int(result[0, 0]), int(result[3, 0]), int(result[6, 0]))
    assert set(pos_colors) == {1, 2, 3}
    assert pos_colors != (1, 2, 3)


def test_derive_color_swap():
    """Derive should find swap for color-permutation pattern."""
    from aria.search.derive import _derive_correspondence_transfer
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1
    inp[4:6, 4:6] = 2

    # Colors swap: 1→2, 2→1 (positions stay)
    out = np.zeros((6, 6), dtype=np.int8)
    out[0:2, 0:2] = 2
    out[4:6, 4:6] = 1

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    progs = _derive_correspondence_transfer(all_trans, all_facts, demos)
    assert any(p.provenance == 'derive:correspondence_transfer' for p in progs)
    for p in progs:
        if p.provenance == 'derive:correspondence_transfer':
            assert np.array_equal(p.execute(inp), out)


def test_derive_position_swap():
    """Derive should find swap for position-permutation pattern."""
    from aria.search.derive import _derive_correspondence_transfer
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    # Two same-color objects that exchange positions
    inp = np.zeros((8, 8), dtype=np.int8)
    inp[0:2, 0:3] = 1  # at (0,0)
    inp[5:7, 0:3] = 1  # at (5,0)

    out = np.zeros((8, 8), dtype=np.int8)
    out[5:7, 0:3] = 1  # first to (5,0) — second's old position
    out[0:2, 0:3] = 1  # second to (0,0) — first's old position

    demos = [(inp, out)]
    in_f = perceive(inp)
    out_f = perceive(out)
    all_facts = [in_f]
    all_trans = [compute_transitions(in_f, out_f)]

    # Same-color same-shape objects at same positions after swap → identity
    # Correspondence sees this as "identical", so no swap detected.
    # This is expected: same-color swaps are undetectable.
    progs = _derive_correspondence_transfer(all_trans, all_facts, demos)
    assert len(progs) == 0  # Can't detect same-color position swap
