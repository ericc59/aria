"""Tests for size-tolerant object correspondence."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive
from aria.guided.correspond import map_output_to_input, _match_cost, _classify_match


def test_exact_shape_movement_detected():
    """Objects that move without changing shape should be 'moved'."""
    inp = np.zeros((8, 8), dtype=np.int8)
    inp[1:3, 1:3] = 2

    out = np.zeros((8, 8), dtype=np.int8)
    out[4:6, 4:6] = 2

    in_facts = perceive(inp)
    out_facts = perceive(out)
    mappings = map_output_to_input(out_facts, in_facts)

    moved = [m for m in mappings if m.match_type == 'moved']
    assert len(moved) == 1
    assert moved[0].in_obj.color == 2


def test_near_shape_movement_classified_as_moved():
    """Objects that move AND grow slightly should still be 'moved' (not 'new')."""
    inp = np.zeros((10, 10), dtype=np.int8)
    inp[1:3, 1:4] = 3  # 2x3, size=6

    out = np.zeros((10, 10), dtype=np.int8)
    out[5:8, 5:9] = 3  # 3x4, size=12 (same color, bigger)

    in_facts = perceive(inp)
    out_facts = perceive(out)
    mappings = map_output_to_input(out_facts, in_facts)

    # Should be classified as 'moved' (same color, size ratio >= 0.5)
    moved = [m for m in mappings if m.match_type in ('moved', 'moved_recolored')]
    assert len(moved) == 1, f"expected moved, got: {[m.match_type for m in mappings]}"


def test_very_different_size_not_forced_moved():
    """Objects with vastly different sizes should NOT be matched as moved."""
    inp = np.zeros((10, 10), dtype=np.int8)
    inp[0, 0] = 5  # size=1

    out = np.zeros((10, 10), dtype=np.int8)
    out[2:8, 2:8] = 5  # size=36 (same color but ratio < 0.5)

    in_facts = perceive(inp)
    out_facts = perceive(out)
    mappings = map_output_to_input(out_facts, in_facts)

    # Size ratio 1/36 < 0.5 → should not be classified as 'moved'
    moved = [m for m in mappings if m.match_type == 'moved']
    # The 1-pixel object is too small for the near-shape tier (size < 2)
    assert len(moved) == 0


def test_match_cost_near_shape_better_than_poor():
    """Near-shape same-color match should have lower cost than poor match."""
    from aria.guided.perceive import ObjFact

    obj_in = ObjFact(oid=1, color=3, row=1, col=1, height=3, width=3, size=9,
                     mask=np.ones((3, 3), dtype=bool),
                     is_rectangular=True, is_square=True, is_line=False,
                     aspect_ratio=1.0, n_same_color=1, n_same_size=1, n_same_shape=1,
                     touches_top=False, touches_bottom=False,
                     touches_left=False, touches_right=False,
                     center_row=2.0, center_col=2.0)

    # Near-shape: same color, slightly bigger
    obj_out_near = ObjFact(oid=2, color=3, row=5, col=5, height=3, width=4, size=12,
                           mask=np.ones((3, 4), dtype=bool),
                           is_rectangular=True, is_square=False, is_line=False,
                           aspect_ratio=4/3, n_same_color=1, n_same_size=1, n_same_shape=1,
                           touches_top=False, touches_bottom=False,
                           touches_left=False, touches_right=False,
                           center_row=6.0, center_col=6.5)

    # Different color AND different shape (truly poor match)
    obj_out_poor = ObjFact(oid=3, color=7, row=5, col=5, height=5, width=5, size=25,
                           mask=np.ones((5, 5), dtype=bool),
                           is_rectangular=True, is_square=True, is_line=False,
                           aspect_ratio=1.0, n_same_color=1, n_same_size=1, n_same_shape=1,
                           touches_top=False, touches_bottom=False,
                           touches_left=False, touches_right=False,
                           center_row=7.0, center_col=7.0)

    cost_near = _match_cost(obj_out_near, obj_in)
    cost_poor = _match_cost(obj_out_poor, obj_in)

    # Near-shape same-color should be cheaper than different-color+different-shape
    assert cost_near < cost_poor, f"near={cost_near:.1f} should be < poor={cost_poor:.1f}"
    # And near-shape should be in the reasonable range (< 50)
    assert cost_near < 50, f"near-shape cost {cost_near:.1f} too high"
