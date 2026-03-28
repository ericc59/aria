"""Tests for aria.eval — evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from aria.datasets import DatasetInfo
from aria.eval import (
    EvalConfig,
    classify_failure_cluster,
    compute_transfer_metrics,
    evaluate_task,
    failure_cluster_report,
    run_evaluation,
)
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
        "beam_rounds", "beam_mutations_per_candidate", "rerank_edits",
    }


# --- Observation provenance ---


def test_eval_outcome_has_solve_phase():
    """Each task outcome should have a solve_phase field."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    outcome = evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
    )
    assert "solve_phase" in outcome
    assert outcome["solve_phase"] in (
        "direct_synthesis", "dims_reconstruction", "observation",
        "skeleton", "structural_edit", "repair", "search",
        "retrieval", "unsolved",
    )


def test_eval_outcome_solve_phase_direct_synthesis():
    """Transpose task should be solved by direct synthesis."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    outcome = evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
    )
    assert outcome["solved"]
    assert outcome["solve_phase"] == "direct_synthesis"


def test_eval_outcome_unsolved_phase():
    config = EvalConfig(max_search_steps=1, max_search_candidates=5, max_refinement_rounds=1)
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    assert not outcome["solved"]
    assert outcome["solve_phase"] == "unsolved"


def test_eval_failure_bucket_unsolved():
    """Unsolved tasks should get a failure_bucket field."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=5, max_refinement_rounds=1)
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    assert not outcome["solved"]
    assert "failure_bucket" in outcome
    assert outcome["failure_bucket"] in (
        "no_candidate", "dims_change_unsupported",
        "dims_change_reconstruction_miss",
        "near_miss_wrong_output", "search_budget_exhausted",
    )


def test_eval_failure_bucket_not_on_solved():
    """Solved tasks should NOT have a failure_bucket."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    outcome = evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
    )
    assert outcome["solved"]
    assert "failure_bucket" not in outcome


def test_failure_bucket_in_transfer_metrics():
    outcomes = [
        {"task_id": "t1", "solved": True, "solve_phase": "direct_synthesis"},
        {"task_id": "t2", "solved": False, "failure_bucket": "search_budget_exhausted", "solve_phase": "unsolved"},
        {"task_id": "t3", "solved": False, "failure_bucket": "dims_change_unsupported", "solve_phase": "unsolved"},
    ]
    metrics = compute_transfer_metrics(outcomes)
    assert "failure_bucket_breakdown" in metrics
    assert metrics["failure_bucket_breakdown"]["search_budget_exhausted"] == 1
    assert metrics["failure_bucket_breakdown"]["dims_change_unsupported"] == 1


def test_transfer_metrics_include_phase_breakdown():
    outcomes = [
        {"task_id": "t1", "solved": True, "solve_source": "search", "solve_phase": "direct_synthesis"},
        {"task_id": "t2", "solved": False, "solve_source": "unsolved", "solve_phase": "unsolved"},
    ]
    metrics = compute_transfer_metrics(outcomes)
    assert "solve_phase_breakdown" in metrics
    assert metrics["solve_phase_breakdown"]["direct_synthesis"] == 1
    assert metrics["solve_phase_breakdown"]["unsolved"] == 1


# --- Failure clustering ---


def test_classify_cluster_same_dims_near_miss():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "color:palette_same"],
        "failure_bucket": "near_miss_wrong_output",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "same_dims_near_miss"


def test_classify_cluster_dims_change():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:different", "size:multiplicative"],
        "failure_bucket": "dims_change_unsupported",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "dims_change"


def test_classify_cluster_dims_change_from_bucket():
    """dims_change_unsupported bucket triggers dims_change even without signature."""
    outcome = {
        "solved": False,
        "task_signatures": [],
        "failure_bucket": "dims_change_unsupported",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "dims_change"


def test_classify_cluster_dims_change_reconstruction_miss():
    """dims_change_reconstruction_miss bucket routes to dims_change cluster."""
    outcome = {
        "solved": False,
        "task_signatures": ["dims:different", "size:grow"],
        "failure_bucket": "dims_change_reconstruction_miss",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "dims_change"


def test_classify_cluster_near_miss_without_dims_sig():
    """near_miss_wrong_output routes to same_dims_near_miss even without dims:same."""
    outcome = {
        "solved": False,
        "task_signatures": ["color:palette_same"],
        "failure_bucket": "near_miss_wrong_output",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "same_dims_near_miss"


def test_classify_cluster_selection():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "role:has_marker", "obj:few"],
        "failure_bucket": "search_budget_exhausted",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "selection"


def test_classify_cluster_composition():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "partition:has_separator_grid"],
        "failure_bucket": "search_budget_exhausted",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "composition"


def test_classify_cluster_transform():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "sym:input_reflective"],
        "failure_bucket": "search_budget_exhausted",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "transform"


def test_classify_cluster_same_dims_exhausted():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "color:few_colors"],
        "failure_bucket": "search_budget_exhausted",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "same_dims_exhausted"


def test_classify_cluster_no_signal():
    outcome = {
        "solved": False,
        "task_signatures": [],
        "failure_bucket": "no_candidate",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "no_signal"


def test_classify_cluster_secondary_hints():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:different", "size:multiplicative", "role:has_marker"],
        "failure_bucket": "dims_change_unsupported",
    }
    cluster = classify_failure_cluster(outcome)
    assert cluster["primary"] == "dims_change"
    assert "selection" in cluster["secondary"]
    assert "transform" in cluster["secondary"]


def test_classify_cluster_repair_hint():
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same"],
        "failure_bucket": "search_budget_exhausted",
        "repair_primary_error": "wrong_color",
    }
    cluster = classify_failure_cluster(outcome)
    assert "repair:wrong_color" in cluster["secondary"]


def test_classify_cluster_deterministic():
    """Same input always produces same output."""
    outcome = {
        "solved": False,
        "task_signatures": ["dims:same", "role:has_marker", "sym:input_reflective"],
        "failure_bucket": "search_budget_exhausted",
    }
    c1 = classify_failure_cluster(outcome)
    c2 = classify_failure_cluster(outcome)
    assert c1 == c2


def test_failure_cluster_report_counts():
    outcomes = [
        {"task_id": "t1", "solved": True},
        {"task_id": "t2", "solved": False, "failure_cluster": "dims_change"},
        {"task_id": "t3", "solved": False, "failure_cluster": "dims_change"},
        {"task_id": "t4", "solved": False, "failure_cluster": "selection"},
        {"task_id": "t5", "solved": False, "failure_cluster": "same_dims_near_miss"},
    ]
    report = failure_cluster_report(outcomes)
    assert report["total_unsolved"] == 4
    assert report["clusters"]["dims_change"]["count"] == 2
    assert report["clusters"]["selection"]["count"] == 1
    assert report["clusters"]["same_dims_near_miss"]["count"] == 1
    # Sorted by count descending
    names = list(report["clusters"].keys())
    assert names[0] == "dims_change"


def test_failure_cluster_report_empty():
    outcomes = [{"task_id": "t1", "solved": True}]
    report = failure_cluster_report(outcomes)
    assert report["total_unsolved"] == 0
    assert report["clusters"] == {}


def test_cluster_in_transfer_metrics():
    outcomes = [
        {"task_id": "t1", "solved": False, "failure_cluster": "dims_change", "solve_phase": "unsolved"},
        {"task_id": "t2", "solved": False, "failure_cluster": "selection", "solve_phase": "unsolved"},
        {"task_id": "t3", "solved": False, "failure_cluster": "dims_change", "solve_phase": "unsolved"},
    ]
    metrics = compute_transfer_metrics(outcomes)
    assert "failure_cluster_breakdown" in metrics
    assert metrics["failure_cluster_breakdown"]["dims_change"] == 2
    assert metrics["failure_cluster_breakdown"]["selection"] == 1


def test_evaluate_task_unsolved_has_cluster():
    """Unsolved tasks should get a failure_cluster field."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=5, max_refinement_rounds=1)
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    assert not outcome["solved"]
    assert "failure_cluster" in outcome
    assert outcome["failure_cluster"] in (
        "same_dims_near_miss", "dims_change", "selection",
        "composition", "transform", "same_dims_exhausted", "no_signal",
    )


def test_evaluate_task_solved_no_cluster():
    """Solved tasks should NOT have a failure_cluster."""
    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    outcome = evaluate_task(
        "t1", _transpose_task(),
        library=Library(),
        config=config,
    )
    assert outcome["solved"]
    assert "failure_cluster" not in outcome


def test_every_bucket_maps_to_a_valid_cluster():
    """Every known failure_bucket produces a cluster in FAILURE_CLUSTERS."""
    from aria.eval import FAILURE_CLUSTERS

    # (bucket, task_signatures) → expected cluster
    cases = [
        ("no_candidate", [], "no_signal"),
        ("no_candidate", ["dims:different"], "dims_change"),
        ("dims_change_unsupported", [], "dims_change"),
        ("dims_change_reconstruction_miss", ["dims:different"], "dims_change"),
        ("near_miss_wrong_output", ["dims:same"], "same_dims_near_miss"),
        ("near_miss_wrong_output", [], "same_dims_near_miss"),
        ("search_budget_exhausted", ["dims:same"], "same_dims_exhausted"),
        ("search_budget_exhausted", [], "no_signal"),
    ]
    for bucket, sigs, expected in cases:
        outcome = {"solved": False, "task_signatures": sigs, "failure_bucket": bucket}
        cluster = classify_failure_cluster(outcome)
        assert cluster["primary"] in FAILURE_CLUSTERS, (
            f"bucket={bucket!r} sigs={sigs!r} → {cluster['primary']!r} not in FAILURE_CLUSTERS"
        )
        assert cluster["primary"] == expected, (
            f"bucket={bucket!r} sigs={sigs!r}: expected {expected!r}, got {cluster['primary']!r}"
        )


def test_run_evaluation_has_failure_clusters(tmp_path: Path):
    ds = _make_dataset(tmp_path, {
        "t1": _transpose_task(),
        "t2": _unsolvable_task(),
    })
    config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)
    report = run_evaluation(ds, library=Library(), config=config)
    assert "failure_clusters" in report
    fc = report["failure_clusters"]
    assert fc["total_unsolved"] >= 0
    assert isinstance(fc["clusters"], dict)


# --- Reranking ablation ---


def test_eval_config_rerank_edits_default():
    c = EvalConfig()
    assert c.rerank_edits is True
    assert c.to_dict()["rerank_edits"] is True


def test_eval_config_rerank_edits_disabled():
    c = EvalConfig(rerank_edits=False)
    assert c.rerank_edits is False
    assert c.to_dict()["rerank_edits"] is False


def test_evaluate_task_reranking_disabled():
    """With rerank_edits=False, reranking fields should not appear."""
    config = EvalConfig(
        max_search_steps=1, max_search_candidates=5,
        max_refinement_rounds=1, rerank_edits=False,
    )
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    # When reranking is disabled, reranking_applied should not appear
    # (structural edit may or may not run depending on whether near-misses exist)
    if "structural_edit_tried" in outcome:
        assert "reranking_applied" not in outcome


def test_evaluate_task_reranking_enabled():
    """With rerank_edits=True (default), reranking fields appear when structural edit runs."""
    config = EvalConfig(
        max_search_steps=1, max_search_candidates=5,
        max_refinement_rounds=1, rerank_edits=True,
    )
    outcome = evaluate_task(
        "t2", _unsolvable_task(),
        library=Library(),
        config=config,
    )
    # If structural edit ran with a ranker, reranking fields appear
    if "structural_edit_tried" in outcome and outcome.get("reranking_applied"):
        assert "reranking_policy_name" in outcome
        assert "reranking_changed_order" in outcome
        assert "reranking_programs_ranked" in outcome


def test_reranking_summary_in_transfer_metrics():
    """Transfer metrics should include reranking summary."""
    outcomes = [
        {
            "task_id": "t1", "solved": True, "solve_phase": "structural_edit",
            "structural_edit_tried": 10, "structural_edit_solved": True,
            "reranking_applied": True, "reranking_changed_order": True,
            "structural_edit_winning_family": "semantic",
        },
        {
            "task_id": "t2", "solved": False, "solve_phase": "unsolved",
            "structural_edit_tried": 20, "structural_edit_solved": False,
            "reranking_applied": True, "reranking_changed_order": False,
        },
        {
            "task_id": "t3", "solved": True, "solve_phase": "search",
        },
    ]
    metrics = compute_transfer_metrics(outcomes)
    rr = metrics["reranking"]
    assert rr["structural_edit_tasks"] == 2
    assert rr["reranking_applied"] == 2
    assert rr["reranking_changed_order"] == 1
    assert rr["structural_edit_solved"] == 1
    assert rr["solved_with_reranking_applied"] == 1
    assert rr["solved_with_reranking_changed"] == 1
    assert rr["winning_family_breakdown"] == {"semantic": 1}


def test_reranking_summary_empty():
    """No structural edit tasks → empty reranking summary."""
    metrics = compute_transfer_metrics([
        {"task_id": "t1", "solved": True, "solve_phase": "search"},
    ])
    rr = metrics["reranking"]
    assert rr["structural_edit_tasks"] == 0
    assert rr["reranking_applied"] == 0


def test_reranking_summary_no_reranking():
    """Structural edit tasks without reranking → zeroes."""
    outcomes = [
        {
            "task_id": "t1", "solved": False, "solve_phase": "unsolved",
            "structural_edit_tried": 5, "structural_edit_solved": False,
        },
    ]
    metrics = compute_transfer_metrics(outcomes)
    rr = metrics["reranking"]
    assert rr["structural_edit_tasks"] == 1
    assert rr["reranking_applied"] == 0
    assert rr["reranking_changed_order"] == 0


def test_ablation_deterministic(tmp_path: Path):
    """Running with rerank on vs off should both complete without error."""
    ds = _make_dataset(tmp_path, {"t1": _transpose_task()})
    config_on = EvalConfig(
        max_search_steps=1, max_search_candidates=200,
        max_refinement_rounds=1, rerank_edits=True,
    )
    config_off = EvalConfig(
        max_search_steps=1, max_search_candidates=200,
        max_refinement_rounds=1, rerank_edits=False,
    )
    report_on = run_evaluation(ds, library=Library(), config=config_on)
    report_off = run_evaluation(ds, library=Library(), config=config_off)
    # Both should complete and have the same solve count (transpose is solved early)
    assert report_on["solved"] == report_off["solved"]
    # Both should have reranking metrics
    assert "reranking" in report_on["transfer_metrics"]
    assert "reranking" in report_off["transfer_metrics"]
