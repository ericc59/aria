"""Offline solver and search regressions."""

from __future__ import annotations

from aria.library.store import Library
from aria.proposer.parser import parse_program
from aria.runtime.ops import reset_library_ops
from aria.solver import solve_task
from aria.types import DemoPair, LibraryEntry, Task, Type, grid_eq, grid_from_list


def test_solve_task_searches_core_ops_without_proposer():
    input_grid = grid_from_list([
        [1, 2, 0],
        [3, 4, 0],
    ])
    output_grid = grid_from_list([
        [1, 3],
        [2, 4],
        [0, 0],
    ])
    task = Task(
        train=(DemoPair(input=input_grid, output=output_grid),),
        test=(DemoPair(input=input_grid, output=output_grid),),
    )
    library = Library()

    result = solve_task(
        task,
        library=library,
        max_search_steps=1,
        max_search_candidates=200,
    )

    assert result.solved
    assert result.searched
    assert not result.retrieved
    assert result.search_candidates_tried >= 1
    assert result.abstractions_mined == 0
    assert library.all_entries() == []
    assert grid_eq(result.test_outputs[0], output_grid)


def test_solve_task_searches_library_entries_when_retrieval_misses():
    reset_library_ops()
    try:
        library = Library()
        entry_program = parse_program("""\
bind flipped = reflect_grid(HORIZONTAL, arg0)
yield flipped
""")
        library.add(LibraryEntry(
            name="flip_h",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=entry_program.steps,
            output=entry_program.output,
            level=1,
            use_count=5,
        ))

        input_grid = grid_from_list([
            [1, 2],
            [3, 4],
        ])
        output_grid = grid_from_list([
            [3, 4],
            [1, 2],
        ])
        task = Task(
            train=(DemoPair(input=input_grid, output=output_grid),),
            test=(DemoPair(input=input_grid, output=output_grid),),
        )

        result = solve_task(
            task,
            library=library,
            max_search_steps=1,
            max_search_candidates=50,
            include_core_ops=False,
        )

        assert result.solved
        assert result.searched
        assert not result.retrieved
        assert result.search_candidates_tried >= 1
        assert grid_eq(result.test_outputs[0], output_grid)
    finally:
        reset_library_ops()


def test_solve_task_forwards_beam_args_and_returns_beam_result():
    """beam_width>0 triggers beam refinement; result carries beam data."""
    inp = grid_from_list([[1, 0], [0, 0]])
    out = grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
    task = Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )

    result = solve_task(
        task,
        library=Library(),
        max_search_steps=1,
        max_search_candidates=10,
        max_refinement_rounds=1,
        beam_width=4,
        beam_rounds=1,
        beam_mutations_per_candidate=10,
    )

    # Task is unsolvable, but beam should have run
    assert not result.solved
    assert result.refinement_result is not None
    assert result.refinement_result.beam is not None
    assert result.refinement_result.beam.candidates_scored > 0


def test_solve_task_default_beam_width_zero_gives_no_beam():
    """Default beam_width=0 means no beam data."""
    inp = grid_from_list([[1, 0], [0, 0]])
    out = grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
    task = Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )

    result = solve_task(
        task,
        library=Library(),
        max_search_steps=1,
        max_search_candidates=5,
        max_refinement_rounds=1,
    )

    assert not result.solved
    assert result.refinement_result is not None
    assert result.refinement_result.beam is None
