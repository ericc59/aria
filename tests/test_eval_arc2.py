"""Tests for aria.eval — evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from aria.datasets import DatasetInfo
from aria.eval import EvalConfig, evaluate_task, run_evaluation
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.trace_store import RefinementTraceStore
from aria.types import DemoPair, Task, grid_from_list


def _transpose_task() -> Task:
    inp = grid_from_list([[1, 2, 0], [3, 4, 0]])
    out = grid_from_list([[1, 3], [2, 4], [0, 0]])
    return Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )


def _unsolvable_task() -> Task:
    inp = grid_from_list([[1, 0], [0, 0]])
    out = grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
    return Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )


def _make_dataset(tmp_path: Path, tasks: dict[str, Task]) -> DatasetInfo:
    root = tmp_path / "ds"
    root.mkdir()
    for tid, task in tasks.items():
        data = {
            "train": [
                {"input": d.input.tolist(), "output": d.output.tolist()}
                for d in task.train
            ],
            "test": [
                {"input": d.input.tolist(), "output": d.output.tolist()}
                for d in task.test
            ],
        }
        (root / f"{tid}.json").write_text(json.dumps(data))
    return DatasetInfo(name="test-ds", version=99, split="testing", root=root)


# --- evaluate_task ---


def test_evaluate_task_solved():
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    outcome = evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
    )
    assert outcome["solved"]
    assert outcome["task_id"] == "t1"
    assert outcome["solve_source"] == "search"
    assert "program" in outcome
    assert outcome["test_results"][0]["correct"]


def test_evaluate_task_unsolved():
    config = EvalConfig(max_search_steps=1, max_search_candidates=5, max_refinement_rounds=1)
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    assert not outcome["solved"]
    assert outcome["solve_source"] == "unsolved"
    assert "program" not in outcome


def test_evaluate_task_with_trace_store():
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    store = RefinementTraceStore()
    evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
        trace_store=store,
    )
    assert len(store) == 1
    assert store.all_records()[0]["task_id"] == "t1"


def test_evaluate_task_with_beam():
    config = EvalConfig(
        max_search_steps=1, max_search_candidates=5,
        max_refinement_rounds=1, beam_width=4, beam_rounds=1,
        beam_mutations_per_candidate=10,
    )
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    assert not outcome["solved"]
    # beam ran but didn't solve — that's fine, just no crash


# --- run_evaluation ---


def test_run_evaluation_batch(tmp_path: Path):
    ds = _make_dataset(tmp_path, {
        "t1": _transpose_task(),
        "t2": _unsolvable_task(),
    })
    config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)

    progress: list[str] = []

    def on_done(tid: str, outcome: dict) -> None:
        progress.append(tid)

    report = run_evaluation(
        ds,
        library=Library(),
        config=config,
        on_task_done=on_done,
    )

    assert report["total"] == 2
    assert report["solved"] >= 1  # transpose should solve
    assert len(progress) == 2
    assert "config" in report
    assert report["config"]["dataset"] == "test-ds"
    assert report["config"]["dataset_version"] == 99


def test_run_evaluation_limit(tmp_path: Path):
    ds = _make_dataset(tmp_path, {
        "t1": _transpose_task(),
        "t2": _unsolvable_task(),
        "t3": _transpose_task(),
    })
    config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)

    report = run_evaluation(ds, library=Library(), config=config, limit=2)
    assert report["total"] == 2


def test_run_evaluation_with_trace_store(tmp_path: Path):
    ds = _make_dataset(tmp_path, {"t1": _transpose_task()})
    config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)
    store = RefinementTraceStore()

    run_evaluation(ds, library=Library(), config=config, trace_store=store)
    assert len(store) == 1


def test_run_evaluation_report_persists(tmp_path: Path):
    ds = _make_dataset(tmp_path, {"t1": _transpose_task()})
    config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)

    report = run_evaluation(ds, library=Library(), config=config)

    out = tmp_path / "report.json"
    with open(out, "w") as f:
        json.dump(report, f, default=str)

    with open(out) as f:
        loaded = json.load(f)

    assert loaded["total"] == 1
    assert loaded["solved"] == 1


# --- EvalConfig ---


def test_eval_config_defaults():
    c = EvalConfig()
    assert c.beam_width == 0
    assert c.max_search_steps == 3
    d = c.to_dict()
    assert d["beam_width"] == 0
    assert d["max_search_candidates"] == 5000


def test_eval_config_to_dict_complete():
    c = EvalConfig(beam_width=8, max_search_steps=4)
    d = c.to_dict()
    assert d["beam_width"] == 8
    assert d["max_search_steps"] == 4
    assert set(d.keys()) == {
        "retrieval_limit", "max_search_steps", "max_search_candidates",
        "max_refinement_rounds", "include_core_ops", "beam_width",
        "beam_rounds", "beam_mutations_per_candidate",
    }
