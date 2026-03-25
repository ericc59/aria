"""Offline search heuristics and pruning tests."""

from __future__ import annotations

from aria.library.store import Library
from aria.offline_search import (
    _depth_candidate_budget,
    _depth_frontier_budget,
    _literals_for_param,
    _ranked_op_signatures,
    search_program,
)
from aria.types import DemoPair, LibraryEntry, Literal, Type, grid_from_list


def test_ranked_op_signatures_prioritize_upscale_for_scaling_tasks():
    ranked = _ranked_op_signatures(
        Library(),
        program_store=None,
        include_core_ops=True,
        task_signatures=frozenset({"dims:different", "size:multiplicative", "size:scale_2x"}),
    )
    names = [name for name, _sig in ranked]

    assert names.index("upscale_grid") < names.index("reflect_grid")
    assert names.index("upscale_grid") < names.index("tile_grid")


def test_ranked_op_signatures_do_not_blindly_prefer_library_ops():
    library = Library()
    library.add(LibraryEntry(
        name="lib_identity_search_test",
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
        steps=(),
        output="grid",
        level=1,
        use_count=50,
    ))

    ranked = _ranked_op_signatures(
        library,
        program_store=None,
        include_core_ops=True,
        task_signatures=frozenset(),
    )
    names = [name for name, _sig in ranked]

    assert names.index("transpose_grid") < names.index("lib_identity_search_test")


def test_tile_grid_int_literals_are_positive_and_small():
    literal_pool = {
        Type.INT: tuple(Literal(value, Type.INT) for value in (-1, 0, 1, 2, 3, 4, 5, 30, 90, 180, 270)),
    }

    literals = _literals_for_param("tile_grid", "rows", Type.INT, literal_pool)
    values = [literal.value for literal in literals]

    assert values == [1, 2, 3, 4, 5, 30]


def test_rotate_grid_int_literals_are_degrees_only():
    literal_pool = {
        Type.INT: tuple(Literal(value, Type.INT) for value in (-1, 0, 1, 2, 3, 4, 5, 30)),
    }

    literals = _literals_for_param("rotate_grid", "degrees", Type.INT, literal_pool)
    values = [literal.value for literal in literals]

    assert values == [90, 180, 270]


def test_search_program_can_use_predicate_filter_chain():
    demos = (
        DemoPair(
            input=grid_from_list([[5, 0, 4]]),
            output=grid_from_list([[4]]),
        ),
    )

    result = search_program(
        demos,
        Library(),
        max_steps=5,
        max_candidates=2000,
    )

    assert result.solved
    assert result.winning_program is not None


def test_depth_candidate_budget_reserves_budget_for_later_depths():
    assert _depth_candidate_budget(
        depth=1,
        max_steps=5,
        max_candidates=200,
        candidates_tried=0,
    ) == 40
    assert _depth_candidate_budget(
        depth=4,
        max_steps=5,
        max_candidates=200,
        candidates_tried=80,
    ) == 60
    assert _depth_candidate_budget(
        depth=5,
        max_steps=5,
        max_candidates=200,
        candidates_tried=120,
    ) == 80


def test_depth_frontier_budget_scales_from_candidate_budget():
    assert _depth_frontier_budget(
        depth=1,
        max_steps=5,
        max_candidates=200,
        candidates_tried=0,
    ) == 320
    assert _depth_frontier_budget(
        depth=5,
        max_steps=5,
        max_candidates=200,
        candidates_tried=120,
    ) == 80
