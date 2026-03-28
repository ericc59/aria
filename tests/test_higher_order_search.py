"""Tests that higher-order predicates/transforms are reachable in search."""

from __future__ import annotations

from aria.library.store import Library
from aria.offline_search import (
    SearchResult,
    SearchTraceEntry,
    _is_searchable_sig,
    _SEARCHABLE_HIGHER_ORDER,
    search_program,
)
from aria.runtime.ops import all_ops
from aria.types import DemoPair, Type, grid_from_list


# ---------------------------------------------------------------------------
# Searchability filtering
# ---------------------------------------------------------------------------


def test_by_color_is_searchable():
    ops = all_ops()
    assert _is_searchable_sig("by_color", ops["by_color"])


def test_by_size_is_searchable():
    ops = all_ops()
    assert _is_searchable_sig("by_size", ops["by_size"])


def test_recolor_to_is_searchable():
    ops = all_ops()
    assert _is_searchable_sig("recolor_to", ops["recolor_to"])


def test_translate_by_is_searchable():
    ops = all_ops()
    assert _is_searchable_sig("translate_by", ops["translate_by"])


def test_non_whitelisted_predicate_ops_are_blocked():
    ops = all_ops()
    assert not _is_searchable_sig("by_shape", ops["by_shape"])
    assert not _is_searchable_sig("by_relative_pos", ops["by_relative_pos"])


def test_where_and_map_obj_are_searchable():
    ops = all_ops()
    assert _is_searchable_sig("where", ops["where"])
    assert _is_searchable_sig("map_obj", ops["map_obj"])


# ---------------------------------------------------------------------------
# Search reachability: predicate pipeline (find_objects → by_color → where → paint_objects)
# ---------------------------------------------------------------------------


def test_search_reaches_by_color_where_pipeline():
    """Search discovers a predicate-based pipeline to filter objects by color.

    Two demos so cell-level shortcuts cannot guess the pattern — the search
    must actually use the object pipeline.
    """
    inp1 = grid_from_list([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 1],
    ])
    out1 = grid_from_list([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ])
    inp2 = grid_from_list([
        [0, 1, 0],
        [2, 0, 3],
        [0, 1, 0],
    ])
    out2 = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    demos = (DemoPair(input=inp1, output=out1), DemoPair(input=inp2, output=out2))
    trace: list[SearchTraceEntry] = []

    result = search_program(
        demos,
        Library(),
        max_steps=4,
        max_candidates=5000,
        observer=trace.append,
    )

    assert result.solved, (
        f"Expected search to solve by_color pipeline, "
        f"tried {result.candidates_tried} candidates"
    )


def test_search_reaches_by_size_where_pipeline():
    """Search discovers a pipeline using by_size to filter objects."""
    # 2-pixel object (color 1) and 1-pixel object (color 1) — same color,
    # different sizes.  Output keeps only the 2-pixel object.
    inp = grid_from_list([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
    ])
    out = grid_from_list([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    demos = (DemoPair(input=inp, output=out),)

    result = search_program(
        demos,
        Library(),
        max_steps=4,
        max_candidates=3000,
    )

    assert result.solved, (
        f"Expected search to solve by_size pipeline, "
        f"tried {result.candidates_tried} candidates"
    )


# ---------------------------------------------------------------------------
# Search reachability: OBJ_TRANSFORM pipeline (recolor_to → map_obj)
# ---------------------------------------------------------------------------


def test_search_reaches_recolor_to_map_obj_pipeline():
    """Search discovers: find_objects → recolor_to(c) → map_obj → paint_objects.

    Grid has one object of color 1.  Output recolors it to color 2.
    """
    inp = grid_from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    out = grid_from_list([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ])

    demos = (DemoPair(input=inp, output=out),)

    result = search_program(
        demos,
        Library(),
        max_steps=4,
        max_candidates=3000,
    )

    assert result.solved, (
        f"Expected search to solve recolor_to pipeline, "
        f"tried {result.candidates_tried} candidates"
    )


# ---------------------------------------------------------------------------
# Search reachability: translate_by pipeline (find_objects → translate_by → map_obj → paint_objects)
# ---------------------------------------------------------------------------


def test_search_reaches_translate_by_map_obj_pipeline():
    """Search discovers: find_objects → translate_by(dr,dc) → map_obj → paint_objects.

    Two demos: each object gets a shadow/copy shifted right by 1 column.
    allowed_ops scopes the search to the object pipeline so we prove the
    4-step chain is type-correct, searchable, and produces the right output
    without being drowned out by hundreds of grid-level alternatives.
    """
    inp1 = grid_from_list([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    out1 = grid_from_list([
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    inp2 = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    out2 = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 0, 0],
    ])

    demos = (DemoPair(input=inp1, output=out1), DemoPair(input=inp2, output=out2))

    result = search_program(
        demos,
        Library(),
        max_steps=4,
        max_candidates=500,
        allowed_ops=frozenset({
            "find_objects", "translate_by", "map_obj", "paint_objects",
        }),
    )

    assert result.solved, (
        f"Expected search to solve translate_by pipeline, "
        f"tried {result.candidates_tried} candidates"
    )
    text = _program_text(result)
    assert "translate_by" in text, f"Expected translate_by in solution: {text}"


def test_translate_by_dr_dc_literals_are_clamped():
    """dr/dc literals stay in [-3, 3] so translate_by combinatorics are bounded."""
    from aria.offline_search import _build_literal_pool, _literals_for_param
    from aria.types import Literal

    inp = grid_from_list([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    demos = (DemoPair(input=inp, output=inp),)
    pool = _build_literal_pool(demos)

    dr_lits = _literals_for_param("translate_by", "dr", Type.INT, pool)
    dc_lits = _literals_for_param("translate_by", "dc", Type.INT, pool)

    for lit in dr_lits:
        assert -3 <= int(lit.value) <= 3, f"dr literal {lit.value} out of [-3,3]"
    for lit in dc_lits:
        assert -3 <= int(lit.value) <= 3, f"dc literal {lit.value} out of [-3,3]"
    # At most 7 values per param → max 49 combos
    assert len(dr_lits) <= 7
    assert len(dc_lits) <= 7


# ---------------------------------------------------------------------------
# Search space remains bounded
# ---------------------------------------------------------------------------


def test_search_space_does_not_blow_up():
    """Higher-order ops do not cause candidate explosion.

    Two contradictory demos make the task unsolvable.  Search should
    exhaust its budget without exceeding max_candidates.
    """
    inp1 = grid_from_list([[1, 2], [3, 4]])
    out1 = grid_from_list([[5, 6], [7, 8]])
    inp2 = grid_from_list([[1, 2], [3, 4]])
    out2 = grid_from_list([[8, 7], [6, 5]])

    demos = (DemoPair(input=inp1, output=out1), DemoPair(input=inp2, output=out2))
    max_cand = 500

    result = search_program(
        demos,
        Library(),
        max_steps=4,
        max_candidates=max_cand,
    )

    assert not result.solved
    assert result.candidates_tried <= max_cand


def test_whitelist_is_small_and_deterministic():
    """The higher-order whitelist stays small — a regression guard."""
    total = sum(len(v) for v in _SEARCHABLE_HIGHER_ORDER.values())
    assert total <= 8, f"Whitelist grew to {total} entries — review search-space impact"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _program_text(result: SearchResult) -> str:
    from aria.runtime.program import program_to_text
    assert result.winning_program is not None
    return program_to_text(result.winning_program)
