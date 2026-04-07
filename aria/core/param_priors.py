"""Lane-local parameter priors — rank existing combinations by demo structure.

Simple rule-based ordering to reduce wasted compile attempts.
No parameter expansion. No learned model.
Part of the canonical architecture.
"""

from __future__ import annotations

from typing import Any, Sequence


def rank_relocation_params(
    demos: Sequence[Any],
) -> list[tuple[int, int]]:
    """Rank (match_rule, alignment) pairs by demo-structure priors.

    Returns pairs in priority order (best first).
    Based on audit data: shape_nearest and mutual_nearest are most
    often best, center and above_left are most often best alignment.
    """
    from aria.runtime.ops.relate_paint import (
        MATCH_SHAPE_NEAREST, MATCH_MARKER_NEAREST, MATCH_COLOR,
        MATCH_ORDERED_ROW, MATCH_ORDERED_COL, MATCH_SIZE, MATCH_MUTUAL_NEAREST,
        ALIGN_CENTER, ALIGN_AT_MARKER, ALIGN_ABOVE_LEFT, ALIGN_ABOVE_RIGHT,
        ALIGN_BELOW_LEFT, ALIGN_BELOW_RIGHT, ALIGN_MARKER_INTERIOR,
    )

    # Priority order derived from audit:
    # Match rules: shape_nearest > mutual_nearest > ordered_row > ordered_col > marker_nearest > size > color
    # Alignments: center > above_left > at_marker > below_left > below_right > above_right > marker_interior
    match_order = [
        MATCH_SHAPE_NEAREST, MATCH_MUTUAL_NEAREST,
        MATCH_ORDERED_ROW, MATCH_ORDERED_COL,
        MATCH_MARKER_NEAREST, MATCH_SIZE, MATCH_COLOR,
    ]
    align_order = [
        ALIGN_CENTER, ALIGN_ABOVE_LEFT, ALIGN_AT_MARKER,
        ALIGN_BELOW_LEFT, ALIGN_BELOW_RIGHT, ALIGN_ABOVE_RIGHT,
        ALIGN_MARKER_INTERIOR,
    ]

    return [(mr, al) for mr in match_order for al in align_order]


def rank_periodic_params(
    demos: Sequence[Any],
) -> list[tuple[int, int, int]]:
    """Rank (axis, period, mode) triples by demo-structure priors.

    Based on audit: axis=0/period=2 is usually best. Try the fitter's
    suggested axis/period first, then alternatives.
    """
    from aria.runtime.ops.periodic_repair import REPAIR_LINES_ONLY, REPAIR_MOTIF_2D, REPAIR_LINES_THEN_2D

    # Default priority order
    params = []
    for axis in (0, 1):
        for period in (2, 3, 4, 5):
            for mode in (REPAIR_LINES_ONLY, REPAIR_LINES_THEN_2D, REPAIR_MOTIF_2D):
                params.append((axis, period, mode))
    return params


def rank_replication_params(
    demos: Sequence[Any],
) -> list[tuple[int, int, int]]:
    """Rank (key_rule, source_policy, placement_rule) triples.

    Default order: adjacent_diff_color + erase + anchor_offset first
    (the most common successful configuration).
    """
    from aria.runtime.ops.replicate import (
        KEY_ADJACENT_DIFF_COLOR, KEY_ADJACENT_ANY,
        SOURCE_ERASE, SOURCE_KEEP,
        PLACE_ANCHOR_OFFSET, PLACE_CENTER,
    )

    return [
        (KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET),
        (KEY_ADJACENT_DIFF_COLOR, SOURCE_KEEP, PLACE_ANCHOR_OFFSET),
        (KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_CENTER),
        (KEY_ADJACENT_DIFF_COLOR, SOURCE_KEEP, PLACE_CENTER),
        (KEY_ADJACENT_ANY, SOURCE_ERASE, PLACE_ANCHOR_OFFSET),
        (KEY_ADJACENT_ANY, SOURCE_KEEP, PLACE_ANCHOR_OFFSET),
        (KEY_ADJACENT_ANY, SOURCE_ERASE, PLACE_CENTER),
        (KEY_ADJACENT_ANY, SOURCE_KEEP, PLACE_CENTER),
    ]
