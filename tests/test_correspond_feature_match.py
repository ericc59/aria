"""Tests for feature-based correspondence matching."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive, ObjFact
from aria.guided.correspond import map_output_to_input, _match_cost


def _make_obj(row, col, mask, color):
    """Helper to create ObjFact for testing."""
    h, w = mask.shape
    return ObjFact(
        oid=0, color=color, row=row, col=col,
        height=h, width=w,
        size=int(mask.sum()),
        mask=mask,
        is_rectangular=bool(mask.all()),
        is_square=(h == w),
        is_line=(h == 1 or w == 1),
        aspect_ratio=w / h if h > 0 else 1.0,
        n_same_color=1, n_same_size=1, n_same_shape=1,
        touches_top=(row == 0),
        touches_bottom=False,
        touches_left=(col == 0),
        touches_right=False,
        center_row=row + h / 2.0,
        center_col=col + w / 2.0,
    )


def test_same_color_similar_shape_moved():
    """Same-color object with similar shape moved should match as near-shape."""
    mask_a = np.ones((3, 3), dtype=bool)
    mask_b = np.ones((3, 3), dtype=bool)
    mask_b[0, 0] = False  # slightly different shape

    obj_in = _make_obj(0, 0, mask_a, color=2)
    obj_out = _make_obj(5, 5, mask_b, color=2)  # within distance guard

    cost = _match_cost(obj_out, obj_in)
    # Should be in near-shape range (< 50), not poor-match range
    assert cost < 50.0, f"Expected near-shape cost, got {cost}"


def test_different_color_similar_shape_feature_match():
    """Different-color but similar-shape objects should get feature-match cost."""
    mask = np.ones((3, 3), dtype=bool)
    obj_in = _make_obj(0, 0, mask, color=1)
    obj_out = _make_obj(0, 0, mask, color=5)  # same pos, same shape, diff color

    cost = _match_cost(obj_out, obj_in)
    # Same shape + same pos → recolored (cost 1.0)
    assert cost == 1.0

    # Now different position too
    obj_out2 = _make_obj(5, 5, mask, color=5)
    cost2 = _match_cost(obj_out2, obj_in)
    # Different color + different pos + same shape → moved_recolored (cost 3.0)
    assert cost2 == 3.0


def test_different_color_loses_to_same_color():
    """Same-color candidate should beat different-color for same shape."""
    mask = np.ones((3, 3), dtype=bool)
    obj_out = _make_obj(5, 5, mask, color=2)

    # Same color candidate at distance
    obj_same = _make_obj(0, 0, mask, color=2)
    # Different color candidate closer
    obj_diff = _make_obj(4, 4, mask, color=7)

    cost_same = _match_cost(obj_out, obj_same)
    cost_diff = _match_cost(obj_out, obj_diff)

    assert cost_same < cost_diff, \
        f"Same-color ({cost_same}) should beat diff-color ({cost_diff})"


def test_size_ratio_guard_prevents_tiny_huge():
    """Size ratio < 0.5 should prevent feature match."""
    mask_tiny = np.ones((1, 1), dtype=bool)
    mask_huge = np.ones((5, 5), dtype=bool)
    obj_tiny = _make_obj(0, 0, mask_tiny, color=3)
    obj_huge = _make_obj(0, 0, mask_huge, color=4)

    cost = _match_cost(obj_huge, obj_tiny)
    # Size ratio = 1/25 = 0.04, way below 0.5 → poor match
    assert cost >= 100.0, f"Expected poor match, got {cost}"


def test_feature_match_classified_as_moved_recolored():
    """Feature-matched pair should classify as moved_recolored."""
    from aria.guided.correspond import _classify_match

    mask_a = np.ones((3, 3), dtype=bool)
    mask_b = np.ones((3, 3), dtype=bool)
    # Same shape, different color, different position
    obj_in = _make_obj(0, 0, mask_a, color=1)
    obj_out = _make_obj(5, 5, mask_b, color=4)

    match_type, xform = _classify_match(obj_out, obj_in)
    assert match_type == "moved_recolored"
