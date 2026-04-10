from __future__ import annotations

import json
from pathlib import Path

from aria.search.proposal_corpus import examples_from_eval_reports, write_jsonl


def test_examples_from_eval_reports_filters_and_deduplicates(tmp_path: Path) -> None:
    report_a = {
        "tasks": [
            {
                "task_id": "a",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["dims:same", "color:few_colors"],
                "description": "search: derive:color_map [recolor_map]",
            },
            {
                "task_id": "b",
                "solved": False,
                "solve_source": "search",
                "task_signatures": ["dims:same"],
                "description": "search: derive:x [x]",
            },
        ],
    }
    report_b = {
        "tasks": [
            {
                "task_id": "a",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["color:few_colors", "dims:same"],
                "description": "search: derive:color_map [recolor_map]",
            },
            {
                "task_id": "c",
                "solved": True,
                "solve_source": "retrieval",
                "task_signatures": ["dims:different"],
                "description": "search: derive:tile [tile]",
            },
        ],
    }
    p1 = tmp_path / "r1.json"
    p2 = tmp_path / "r2.json"
    p1.write_text(json.dumps(report_a))
    p2.write_text(json.dumps(report_b))

    examples = examples_from_eval_reports([p1, p2])
    assert len(examples) == 1
    ex = examples[0]
    assert ex.task_id == "a"
    assert ex.family == "recolor_map"
    assert ex.task_signatures == ("color:few_colors", "dims:same")


def test_write_jsonl_emits_one_row_per_example(tmp_path: Path) -> None:
    report = {
        "tasks": [
            {
                "task_id": "a",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["dims:different"],
                "description": "search: derive:exact_tile [tile]",
            },
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    examples = examples_from_eval_reports([path])

    out = tmp_path / "corpus.jsonl"
    write_jsonl(out, examples)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["task_id"] == "a"
    assert row["family"] == "tile"
