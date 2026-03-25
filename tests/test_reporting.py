"""Reporting summary tests."""

from __future__ import annotations

from aria.reporting import build_solve_report, extract_library_ops_used


def test_extract_library_ops_used_filters_to_library_names():
    program = """\
let x = reflect_grid(HORIZONTAL, input)
let y = flip_h(input)
let result = stack_v(y, x)
-> result
"""
    assert extract_library_ops_used(program, ["flip_h", "mirror_and_swap"]) == ["flip_h"]


def test_build_solve_report_tracks_sources_and_usage():
    tasks = [
        {
            "task_id": "a",
            "solved": True,
            "retrieved": True,
            "retrieval_sources": ["solve-report:v1-train.json"],
            "library_ops_used": ["flip_h"],
            "test_results": [{"correct": True}],
        },
        {
            "task_id": "b",
            "solved": True,
            "retrieved": False,
            "solve_source": "proposal",
            "library_ops_used": ["flip_h", "mirror_and_swap"],
            "test_results": [{"correct": False}],
        },
        {
            "task_id": "c",
            "solved": True,
            "retrieved": False,
            "solve_source": "search",
            "test_results": [{"correct": True}],
        },
        {
            "task_id": "d",
            "solved": False,
            "retrieved": False,
            "missing_ops": {"window_map": 2},
        },
    ]

    report = build_solve_report(tasks, {"provider": "claude"})

    assert report["solved"] == 3
    assert report["source_counts"] == {"retrieval": 1, "proposal": 1, "search": 1, "unsolved": 1}
    assert report["retrieval_hit_count"] == 1
    assert report["search_solve_count"] == 1
    assert report["retrieval_sources_global"] == {"solve-report:v1-train.json": 1}
    assert report["library_ops_global"] == {"flip_h": 2, "mirror_and_swap": 1}
    assert report["missing_ops_global"] == {"window_map": 2}
