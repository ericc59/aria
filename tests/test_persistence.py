"""Persistence and retrieval tests for the offline runtime path."""

from __future__ import annotations

import json
from pathlib import Path

from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.proposer.parser import parse_program
from aria.runtime.ops import reset_library_ops
from aria.solver import solve_task
from aria.trace_store import RefinementTraceStore
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


def test_program_store_prefers_distinct_tasks_over_retrieval_echoes():
    store = ProgramStore()
    program_text = "let result: GRID = transpose_grid(input)\n-> result"

    # One-task program inflated by repeated sweeps should not outrank
    # a program that verified on multiple distinct tasks.
    for _ in range(3):
        store.add_text(
            program_text,
            task_id="echo-task",
            source="offline-retrieval",
            signatures=frozenset({"dims:same"}),
        )

    store.add_text(
        "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
        task_id="task-a",
        source="offline-search",
        signatures=frozenset({"dims:same"}),
    )
    store.add_text(
        "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
        task_id="task-b",
        source="offline-search",
        signatures=frozenset({"dims:same"}),
    )

    ranked = store.ranked_records(frozenset({"dims:same"}))
    assert ranked[0].task_ids == ("task-a", "task-b")
    assert ranked[0].distinct_task_count == 2


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
            support_task_ids=("task-a", "task-b"),
            support_program_count=2,
            mdl_gain=7,
        ))

        output = tmp_path / "library.json"
        library.save_json(output)
        reloaded = Library.load_json(output)

        assert reloaded.names() == ["flip_h"]
        entry = reloaded.get("flip_h")
        assert entry is not None
        assert entry.return_type == Type.GRID
        assert entry.use_count == 3
        assert entry.support_task_ids == ("task-a", "task-b")
        assert entry.support_program_count == 2
        assert entry.mdl_gain == 7
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


def test_solve_task_retrieval_prefers_transfer_backed_programs():
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
    # Wrong one-off leaf with high raw use_count from repeated retrieval.
    for _ in range(3):
        store.add_text(
            "bind result = reflect_grid(HORIZONTAL, input)\nyield result\n",
            task_id="echo-task",
            source="offline-retrieval",
            signatures=frozenset({"color:palette_subset"}),
        )
    # Correct program verified on two distinct tasks.
    store.add_text(
        "bind result = apply_color_map({1: 2}, input)\nyield result\n",
        task_id="task-a",
        source="offline-search",
        signatures=frozenset({"color:palette_subset"}),
    )
    store.add_text(
        "bind result = apply_color_map({1: 2}, input)\nyield result\n",
        task_id="task-b",
        source="offline-search",
        signatures=frozenset({"color:palette_subset"}),
    )

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


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------


def _transpose_task() -> Task:
    """A trivially solvable task (transpose) for trace tests."""
    input_grid = grid_from_list([
        [1, 2, 0],
        [3, 4, 0],
    ])
    output_grid = grid_from_list([
        [1, 3],
        [2, 4],
        [0, 0],
    ])
    return Task(
        train=(DemoPair(input=input_grid, output=output_grid),),
        test=(DemoPair(input=input_grid, output=output_grid),),
    )


def _unsolvable_task() -> Task:
    """A task that offline search cannot solve in tiny budgets."""
    inp = grid_from_list([[1, 0], [0, 0]])
    out = grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
    return Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )


def test_solve_task_exposes_refinement_result_for_solved_task():
    task = _transpose_task()
    result = solve_task(
        task,
        library=Library(),
        max_search_steps=1,
        max_search_candidates=200,
    )
    assert result.solved
    assert result.refinement_result is not None
    assert result.refinement_result.solved
    assert result.refinement_result is not None


def test_solve_task_exposes_refinement_result_for_failed_task():
    task = _unsolvable_task()
    result = solve_task(
        task,
        library=Library(),
        max_search_steps=1,
        max_search_candidates=5,
        max_refinement_rounds=1,
    )
    assert not result.solved
    assert result.refinement_result is not None
    assert not result.refinement_result.solved
    assert result.refinement_result is not None


def test_trace_store_persists_solved_task(tmp_path: Path):
    task = _transpose_task()
    result = solve_task(
        task,
        library=Library(),
        task_id="transpose-t",
        max_search_steps=1,
        max_search_candidates=200,
    )
    assert result.solved

    store = RefinementTraceStore()
    store.add_result(
        task_id="transpose-t",
        result=result.refinement_result,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    rec = loaded.all_records()[0]
    assert rec["task_id"] == "transpose-t"
    assert rec["solved"] is True
    assert rec["solve_source"] == "refinement"
    # Synthesis may solve before any rounds run
    assert rec["winning_program"] is not None
    assert len(rec["task_signatures"]) > 0


def test_trace_store_persists_unsolved_task(tmp_path: Path):
    task = _unsolvable_task()
    result = solve_task(
        task,
        library=Library(),
        task_id="unsolvable-t",
        max_search_steps=1,
        max_search_candidates=5,
        max_refinement_rounds=1,
    )
    assert not result.solved

    store = RefinementTraceStore()
    store.add_result(
        task_id="unsolvable-t",
        result=result.refinement_result,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    rec = loaded.all_records()[0]
    assert rec["task_id"] == "unsolvable-t"
    assert rec["solved"] is False
    assert rec["solve_source"] == "refinement"
    # Synthesis may solve before any rounds run
    assert rec["winning_program"] is None


def test_trace_store_persists_retrieval_task(tmp_path: Path):
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
    store.add_text(
        "bind result = apply_color_map({1: 2}, input)\nyield result\n",
        task_id="bootstrap",
        source="test",
        signatures=frozenset({"color:palette_subset"}),
    )

    result = solve_task(
        task,
        library=Library(),
        program_store=store,
        task_id="retrieval-t",
    )
    assert result.solved
    assert result.retrieved
    assert result.refinement_result is None
    assert result.winning_program is not None

    trace_store = RefinementTraceStore()
    trace_store.add_retrieval_result(
        task_id="retrieval-t",
        winning_program=result.winning_program,
        candidates_tried=result.retrieval_candidates_tried,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "traces.json"
    trace_store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    rec = loaded.all_records()[0]
    assert rec["task_id"] == "retrieval-t"
    assert rec["solve_source"] == "retrieval"
    assert rec["solved"] is True
    assert rec["rounds"] == []
    assert rec["retrieval"]["candidates_tried"] == 1
    assert rec["winning_program"] is not None


def test_trace_records_contain_scored_feedback(tmp_path: Path):
    task = _unsolvable_task()
    result = solve_task(
        task,
        library=Library(),
        task_id="fb-test",
        max_search_steps=1,
        max_search_candidates=20,
        max_refinement_rounds=2,
    )

    store = RefinementTraceStore()
    store.add_result(
        task_id="fb-test",
        result=result.refinement_result,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    rec = loaded.all_records()[0]
    for rnd in rec["rounds"]:
        fb = rnd["feedback"]
        assert "suggested_focus" in fb
        assert "dimension_mismatch_count" in fb
        assert "pixel_mismatch_count" in fb
        assert isinstance(rnd["candidates_tried"], int)


def test_trace_store_append_preserves_existing_records(tmp_path: Path):
    """Resume/append does not lose previously persisted traces."""
    path = tmp_path / "traces.json"

    # First session: write one record
    store1 = RefinementTraceStore()
    task1 = _transpose_task()
    r1 = solve_task(task1, library=Library(), task_id="t1",
                    max_search_steps=1, max_search_candidates=200)
    store1.add_result(task_id="t1", result=r1.refinement_result,
                      task_signatures=r1.task_signatures)
    store1.save_json(path)
    assert len(store1) == 1

    # Second session: reload, add another
    store2 = RefinementTraceStore.load_json(path)
    assert len(store2) == 1
    task2 = _unsolvable_task()
    r2 = solve_task(task2, library=Library(), task_id="t2",
                    max_search_steps=1, max_search_candidates=5,
                    max_refinement_rounds=1)
    store2.add_result(task_id="t2", result=r2.refinement_result,
                      task_signatures=r2.task_signatures)
    store2.save_json(path)

    # Verify both survive
    store3 = RefinementTraceStore.load_json(path)
    assert len(store3) == 2
    task_ids = [r["task_id"] for r in store3.all_records()]
    assert "t1" in task_ids
    assert "t2" in task_ids


def test_trace_store_load_reports_corruption(tmp_path: Path):
    path = tmp_path / "traces.json"
    path.write_text('{"version": 2, "records": [')

    try:
        RefinementTraceStore.load_json(path)
    except ValueError as exc:
        assert "Corrupt refinement trace store" in str(exc)
    else:
        raise AssertionError("Expected corrupt trace store to raise ValueError")


# ---------------------------------------------------------------------------
# Signature preservation
# ---------------------------------------------------------------------------


def test_offline_solve_saves_program_signatures():
    task = _transpose_task()
    store = ProgramStore()
    result = solve_task(
        task,
        library=Library(),
        program_store=store,
        task_id="sig-test",
        max_search_steps=1,
        max_search_candidates=200,
    )
    assert result.solved
    assert len(result.task_signatures) > 0

    records = store.all_records()
    assert len(records) >= 1
    # At least one record for sig-test should have signatures
    sig_records = [r for r in records if "sig-test" in r.task_ids]
    assert len(sig_records) >= 1
    assert len(sig_records[0].signatures) > 0


def test_backfill_signatures_updates_existing_records():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="old-task",
        source="legacy",
    )

    # Initially no signatures
    assert store.all_records()[0].signatures == ()

    updated = store.backfill_signatures(
        "old-task",
        frozenset({"dims:different", "change:additive"}),
    )
    assert updated == 1

    record = store.all_records()[0]
    assert "dims:different" in record.signatures
    assert "change:additive" in record.signatures


def test_backfill_signatures_merges_with_existing():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="t1",
        source="test",
        signatures=frozenset({"dims:different"}),
    )

    updated = store.backfill_signatures(
        "t1",
        frozenset({"dims:different", "change:additive"}),
    )
    assert updated == 1

    record = store.all_records()[0]
    assert "dims:different" in record.signatures
    assert "change:additive" in record.signatures


def test_backfill_signatures_skips_unrelated_tasks():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="t1",
        source="test",
    )

    updated = store.backfill_signatures("other-task", frozenset({"dims:same"}))
    assert updated == 0
    assert store.all_records()[0].signatures == ()


# ---------------------------------------------------------------------------
# Beam trace persistence
# ---------------------------------------------------------------------------


def test_trace_store_persists_beam_data_when_beam_enabled(tmp_path: Path):
    """solve_task with beam_width>0 produces a refinement_result with beam data
    that persists through RefinementTraceStore round-trip."""
    task = _unsolvable_task()
    result = solve_task(
        task,
        library=Library(),
        task_id="beam-persist",
        max_search_steps=1,
        max_search_candidates=10,
        max_refinement_rounds=1,
        beam_width=4,
        beam_rounds=1,
        beam_mutations_per_candidate=10,
    )
    assert not result.solved
    assert result.refinement_result is not None
    assert result.refinement_result.beam is not None

    store = RefinementTraceStore()
    store.add_result(
        task_id="beam-persist",
        result=result.refinement_result,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "beam_traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    rec = loaded.all_records()[0]
    assert rec["task_id"] == "beam-persist"
    assert "beam" in rec
    beam = rec["beam"]
    assert beam["candidates_scored"] > 0
    assert isinstance(beam["transitions"], list)
    assert isinstance(beam["round_summaries"], list)


def test_trace_store_no_beam_key_when_beam_disabled(tmp_path: Path):
    """Default beam_width=0 means the persisted record has no 'beam' key."""
    task = _unsolvable_task()
    result = solve_task(
        task,
        library=Library(),
        task_id="no-beam",
        max_search_steps=1,
        max_search_candidates=5,
        max_refinement_rounds=1,
    )

    store = RefinementTraceStore()
    store.add_result(
        task_id="no-beam",
        result=result.refinement_result,
        task_signatures=result.task_signatures,
    )

    path = tmp_path / "no_beam_traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    rec = loaded.all_records()[0]
    assert "beam" not in rec
