"""Persistence and retrieval tests for the offline runtime path."""

from __future__ import annotations

import json

from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.proposer.parser import parse_program
from aria.runtime.ops import reset_library_ops
from aria.solver import solve_task
from aria.types import (
    DemoPair,
    LibraryEntry,
    Task,
    Type,
    grid_eq,
    grid_from_list,
)


def test_parse_program_accepts_serialized_let_syntax():
    program = parse_program("""\
let result: GRID = reflect_grid(HORIZONTAL, input)
-> result
""")
    assert program.output == "result"
    assert len(program.steps) == 1


def test_program_store_imports_reports_and_round_trips(tmp_path):
    solve_report = tmp_path / "solve_report.json"
    solve_report.write_text(json.dumps({
        "tasks": [
            {
                "task_id": "task-a",
                "program": "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
                "task_signatures": ["role:has_legend", "legend:present"],
            }
        ]
    }))

    corpus_report = tmp_path / "corpus_report.json"
    corpus_report.write_text(json.dumps({
        "tasks": {
            "task-b": {
                "programs": [
                    "bind result = transpose_grid(input)\nyield result",
                ]
            }
        }
    }))

    store = ProgramStore()
    assert store.import_path(solve_report) == 1
    assert store.import_path(corpus_report) == 1
    assert len(store) == 2

    output = tmp_path / "program_store.json"
    store.save_json(output)
    reloaded = ProgramStore.load_json(output)

    records = reloaded.ranked_records()
    assert len(records) == 2
    assert any("solve-report:solve_report.json" in record.sources for record in records)
    assert any("corpus-report:corpus_report.json" in record.sources for record in records)
    assert any("role:has_legend" in record.signatures for record in records)


def test_program_store_ranks_by_signature_overlap():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="legend-task",
        source="test",
        signatures=frozenset({"role:has_legend", "legend:present"}),
    )
    store.add_text(
        "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
        task_id="partition-task",
        source="test",
        signatures=frozenset({"partition:has_separator_grid"}),
    )

    ranked = store.ranked_records(frozenset({"role:has_legend"}))
    assert ranked[0].task_ids == ("legend-task",)


def test_library_round_trips_to_json(tmp_path):
    reset_library_ops()
    try:
        library = Library()
        program = parse_program("""\
bind flipped = reflect_grid(HORIZONTAL, input)
yield flipped
""")
        library.add(LibraryEntry(
            name="flip_h",
            params=(("input", Type.GRID),),
            return_type=Type.GRID,
            steps=program.steps,
            output=program.output,
            level=1,
            use_count=3,
        ))

        output = tmp_path / "library.json"
        library.save_json(output)
        reloaded = Library.load_json(output)

        assert reloaded.names() == ["flip_h"]
        entry = reloaded.get("flip_h")
        assert entry is not None
        assert entry.return_type == Type.GRID
        assert entry.use_count == 3
    finally:
        reset_library_ops()


def test_solve_task_retrieves_program_from_store():
    input_grid = grid_from_list([
        [0, 1],
        [0, 0],
    ])
    output_grid = grid_from_list([
        [0, 2],
        [0, 0],
    ])
    task = Task(
        train=(DemoPair(input=input_grid, output=output_grid),),
        test=(DemoPair(input=input_grid, output=output_grid),),
    )

    store = ProgramStore()
    store.add_text("""\
bind result = apply_color_map({1: 2}, input)
yield result
""", task_id="bootstrap", source="test")

    result = solve_task(
        task,
        library=Library(),
        program_store=store,
        task_id="new-task",
    )

    assert result.solved
    assert result.retrieved
    assert result.retrieval_candidates_tried == 1
    assert len(result.test_outputs) == 1
    assert grid_eq(result.test_outputs[0], output_grid)
