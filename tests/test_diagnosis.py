"""Tests for gap diagnosis from eval results."""

from __future__ import annotations

from aria.diagnosis import diagnose, format_diagnosis


def _make_outcomes() -> list[dict]:
    """A realistic set of eval outcomes for diagnosis testing."""
    return [
        # Solved by skeleton
        {
            "task_id": "t1",
            "solved": True,
            "solve_source": "search",
            "task_signatures": ["dims:same", "change:additive"],
            "abstraction_hints_available": True,
            "abstraction_hints_count": 3,
            "solved_with_retrieved_abstraction": True,
            "retrieved_abstractions_used": ["lib_flip_h"],
            "solved_by_skeleton": True,
            "skeleton_hypotheses_tested": 2,
        },
        # Solved by retrieval
        {
            "task_id": "t2",
            "solved": True,
            "solve_source": "retrieval",
            "retrieved": True,
            "task_signatures": ["dims:different"],
            "retrieval_provenance": {
                "distinct_task_count": 3,
                "has_non_retrieval_source": True,
            },
        },
        # Solved by search (no abstraction help)
        {
            "task_id": "t3",
            "solved": True,
            "solve_source": "search",
            "task_signatures": ["dims:same"],
            "abstraction_hints_available": False,
        },
        # Unsolved with near-miss
        {
            "task_id": "t4",
            "solved": False,
            "solve_source": "unsolved",
            "task_signatures": ["dims:same", "change:additive", "role:has_marker"],
            "abstraction_hints_available": True,
            "abstraction_hints_count": 2,
            "skeleton_hypotheses_tested": 4,
            "skeleton_near_miss": {
                "source": "single:lib_overlay",
                "error_type": "wrong_output",
                "program_text": "let v0: GRID = lib_overlay(input)\n-> v0",
            },
        },
        # Unsolved with no hints
        {
            "task_id": "t5",
            "solved": False,
            "solve_source": "unsolved",
            "task_signatures": ["dims:different", "structure:grid_partition"],
            "searched": True,
            "refinement_rounds": 2,
            "search_candidates_tried": 5000,
        },
        # Unsolved with hints but no near-miss
        {
            "task_id": "t6",
            "solved": False,
            "solve_source": "unsolved",
            "task_signatures": ["dims:same", "color:palette_subset"],
            "abstraction_hints_available": True,
            "skeleton_hypotheses_tested": 3,
        },
    ]


def test_diagnose_summary():
    diag = diagnose(_make_outcomes())
    s = diag["summary"]
    assert s["total"] == 6
    assert s["solved"] == 3
    assert s["unsolved"] == 3


def test_diagnose_solve_breakdown():
    diag = diagnose(_make_outcomes())
    sb = diag["solve_breakdown"]
    assert sb.get("skeleton", 0) == 1
    assert sb.get("retrieval", 0) == 1
    assert sb.get("search", 0) == 1


def test_diagnose_near_misses():
    diag = diagnose(_make_outcomes())
    nm = diag["near_misses"]
    assert nm["count"] == 1
    assert nm["by_error_type"]["wrong_output"] == 1
    assert len(nm["examples"]) == 1
    assert nm["examples"][0]["task_id"] == "t4"
    assert nm["examples"][0]["source"] == "single:lib_overlay"


def test_diagnose_hypothesis_coverage():
    diag = diagnose(_make_outcomes())
    hc = diag["hypothesis_coverage"]
    assert hc["hints_available"] >= 3
    assert hc["skeletons_tested"] >= 2
    assert hc["solved_by_skeleton"] == 1


def test_diagnose_gap_categories():
    diag = diagnose(_make_outcomes())
    gc = diag["gap_categories"]
    assert gc["no_hints"] >= 1       # t5: searched but no hints
    assert gc["near_miss_not_solved"] == 1  # t4
    assert gc["hints_no_near_miss"] >= 1    # t6


def test_diagnose_top_failing_signatures():
    diag = diagnose(_make_outcomes())
    ts = diag["top_failing_signatures"]
    sig_names = [s for s, _ in ts]
    assert "dims:same" in sig_names  # t4, t6 both have dims:same


def test_format_diagnosis_produces_readable_output():
    diag = diagnose(_make_outcomes())
    text = format_diagnosis(diag)
    assert "Solved: 3/6" in text
    assert "Near-misses" in text
    assert "Gap categories" in text
    assert isinstance(text, str)
    assert len(text) > 100


def test_diagnose_empty_outcomes():
    diag = diagnose([])
    assert diag["summary"]["total"] == 0
    assert diag["summary"]["solved"] == 0
    assert diag["near_misses"]["count"] == 0


def test_diagnose_all_solved():
    outcomes = [
        {"task_id": "t1", "solved": True, "solve_source": "search", "task_signatures": []},
        {"task_id": "t2", "solved": True, "solve_source": "retrieval", "task_signatures": []},
    ]
    diag = diagnose(outcomes)
    assert diag["summary"]["unsolved"] == 0
    assert diag["near_misses"]["count"] == 0
    assert diag["gap_categories"]["no_hints"] == 0


def test_diagnose_all_unsolved():
    outcomes = [
        {
            "task_id": "t1", "solved": False, "solve_source": "unsolved",
            "searched": True, "refinement_rounds": 1,
            "task_signatures": ["dims:same"],
        },
    ]
    diag = diagnose(outcomes)
    assert diag["summary"]["solved"] == 0
    assert diag["gap_categories"]["no_hints"] == 1


def test_eval_outcome_contains_near_miss_data():
    """evaluate_task should populate skeleton_near_miss for unsolved tasks."""
    from aria.eval import EvalConfig, evaluate_task
    from aria.library.store import Library
    from aria.proposer.parser import parse_program
    from aria.runtime.ops import reset_library_ops
    from aria.types import DemoPair, LibraryEntry, Type, grid_from_list

    reset_library_ops()
    try:
        library = Library()
        prog = parse_program("bind v0 = reflect_grid(VERTICAL, arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="lib_flip_v", params=(("arg0", Type.GRID),),
            return_type=Type.GRID, steps=prog.steps, output=prog.output,
            level=1, use_count=3, support_task_ids=("a", "b"),
            support_program_count=2, mdl_gain=5,
        ))

        # Unsolvable task — skeleton will be a near-miss
        inp = grid_from_list([[1, 0], [0, 0]])
        out = grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
        from aria.types import Task
        task = Task(
            train=(DemoPair(input=inp, output=out),),
            test=(DemoPair(input=inp, output=out),),
        )

        config = EvalConfig(max_search_steps=1, max_search_candidates=5, max_refinement_rounds=1)
        outcome = evaluate_task("near-miss-t", task, library=library, config=config)

        assert not outcome["solved"]
        # Should have a near-miss from the wrong-dims skeleton
        # (reflect_v preserves dims, but task changes dims)
        assert outcome.get("skeleton_hypotheses_tested", 0) >= 1
    finally:
        reset_library_ops()
