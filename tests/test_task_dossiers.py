"""Tests for task dossier export helpers."""

from __future__ import annotations

from aria.eval.task_dossiers import render_task_markdown


def test_render_task_markdown_includes_core_sections():
    dossier = {
        "task_id": "abc12345",
        "gold": {
            "task_id": "abc12345",
            "decomposition": "object",
            "entities": [],
            "relations": [],
            "template": "match_recolor",
            "critical_slots": {},
        },
        "task": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]],
                }
            ],
            "test": [{"input": [[0, 1], [1, 0]], "output": [[0]]}],
        },
        "gates": {
            "first_failing_gate": "executor",
            "exact_solve": False,
            "gates": {
                "decomposition": {"passed": True, "score": 1.0},
                "executor": {"passed": False, "score": 0.0},
            },
        },
        "artifacts": {
            "decomposition_hypotheses": ["object"],
            "entities": [],
            "relations": [],
            "template_hypotheses": ["match_recolor"],
            "slot_candidates": {},
            "executor_attempted": True,
            "executor_ran": False,
            "executor_error": "missing_program",
        },
        "solver_summary": {
            "solved": False,
            "winning_program_text": None,
            "best_candidates": [],
        },
    }

    text = render_task_markdown(dossier)
    assert "# Task abc12345" in text
    assert "## Structural Gates" in text
    assert "## Gold Annotation" in text
    assert "## Train/Test Grids" in text
    assert "01\n10" in text
    assert "## Stage Artifacts" in text
    assert "## Solver Summary" in text

