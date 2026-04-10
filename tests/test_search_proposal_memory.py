from __future__ import annotations

import json
from pathlib import Path

from aria.search.proposal_memory import SearchProposalPrior
from aria.search.search import search_programs
from aria.types import grid_from_list


def test_proposal_prior_learns_from_eval_reports(tmp_path: Path) -> None:
    report = {
        "tasks": [
            {
                "task_id": "a",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["dims:different", "size:multiplicative"],
                "description": "search: derive:exact_tile [tile]",
            },
            {
                "task_id": "b",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["dims:different", "size:multiplicative"],
                "description": "search: derive:exact_tile [tile]",
            },
            {
                "task_id": "c",
                "solved": True,
                "solve_source": "search",
                "task_signatures": ["dims:same"],
                "description": "search: derive:color_map [recolor_map]",
            },
        ],
    }
    path = tmp_path / "eval_sample.json"
    path.write_text(json.dumps(report))

    prior = SearchProposalPrior.from_eval_reports([path])
    assert prior.global_counts["tile"] == 2
    assert prior.global_counts["recolor_map"] == 1
    assert prior.score_family("tile", frozenset({"size:multiplicative"})) > prior.score_family(
        "recolor_map",
        frozenset({"size:multiplicative"}),
    )


def test_search_programs_uses_signature_compatible_prior() -> None:
    inp = grid_from_list([[1, 2], [3, 4]])
    out = grid_from_list([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
    prog = search_programs([(inp, out)], time_budget=1.0)
    assert prog is not None
    assert "tile" in prog.description


def test_proposal_prior_round_trip(tmp_path: Path) -> None:
    prior = SearchProposalPrior(
        global_counts={"tile": 3, "recolor_map": 1},
        by_signature={
            "dims:different": {"tile": 3},
            "dims:same": {"recolor_map": 1},
        },
    )
    path = tmp_path / "prior.json"
    prior.save_json(path)
    loaded = SearchProposalPrior.load_json(path)
    assert loaded == prior
