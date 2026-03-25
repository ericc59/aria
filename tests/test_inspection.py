"""Inspection helper tests."""

from __future__ import annotations

from aria.inspection import inspect_task
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.types import DemoPair, Task, grid_from_list


def test_inspect_task_reports_retrieval_and_search_trace():
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

    store = ProgramStore()
    store.add_text(
        "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
        task_id="other-task",
        source="test",
        signatures=frozenset({"sym:input_reflective"}),
    )

    inspection = inspect_task(
        task.train,
        library=Library(),
        program_store=store,
        test_inputs=tuple(pair.input for pair in task.test),
        retrieval_limit=3,
        max_search_steps=1,
        max_search_candidates=20,
        search_trace_limit=5,
    )

    assert inspection["task_signatures"]
    assert inspection["output_size"]["classification"] == "fixed_output_dims"
    assert inspection["output_size"]["demos"][0]["full_context_prediction"] == (3, 2)
    assert inspection["output_size"]["demos"][0]["loo_prediction"] == (2, 3)
    assert inspection["output_size"]["tests"][0]["predicted_output_dims"] == (3, 2)
    assert len(inspection["demos"]) == 1
    assert inspection["retrieval"]
    assert inspection["retrieval"][0]["rank"] == 1
    assert inspection["search"]["trace"]
    assert inspection["search"]["trace"][0]["candidate_num"] == 1


def test_inspect_task_reports_correct_loo_output_size_for_scaling():
    task = Task(
        train=(
            DemoPair(
                input=grid_from_list([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]),
                output=grid_from_list([[0] * 6 for _ in range(6)]),
            ),
            DemoPair(
                input=grid_from_list([[1] * 4 for _ in range(4)]),
                output=grid_from_list([[0] * 8 for _ in range(8)]),
            ),
        ),
        test=(
            DemoPair(
                input=grid_from_list([[1] * 5 for _ in range(5)]),
                output=grid_from_list([[0] * 10 for _ in range(10)]),
            ),
        ),
    )

    inspection = inspect_task(
        task.train,
        library=Library(),
        program_store=ProgramStore(),
        test_inputs=tuple(pair.input for pair in task.test),
        retrieval_limit=0,
        max_search_steps=1,
        max_search_candidates=1,
        search_trace_limit=1,
    )

    assert inspection["output_size"]["classification"] == "multiplicative"
    assert inspection["output_size"]["demos"][0]["loo_prediction"] == (6, 6)
    assert inspection["output_size"]["demos"][1]["loo_prediction"] == (8, 8)
    assert inspection["output_size"]["tests"][0]["predicted_output_dims"] == (10, 10)
