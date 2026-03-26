"""Tests for aria.submission — ARC-AGI-2 submission builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aria.submission import (
    Hypothesis,
    build_submission,
    hypotheses_from_eval_outcomes,
    load_submission,
    save_submission,
    score_submission,
    select_attempts,
)
from aria.types import grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


def _h(rows: list[list[int]], rank: int = 0, prog: str = "") -> Hypothesis:
    return Hypothesis(grid=_grid(rows), rank=rank, program_text=prog)


# ---------------------------------------------------------------------------
# select_attempts
# ---------------------------------------------------------------------------


def test_select_single_hypothesis_duplicates():
    a1, a2 = select_attempts([_h([[1, 2]])])
    assert np.array_equal(a1, _grid([[1, 2]]))
    assert np.array_equal(a2, _grid([[1, 2]]))


def test_select_two_diverse_hypotheses():
    h1 = _h([[1, 2]], rank=0)
    h2 = _h([[3, 4]], rank=1)
    a1, a2 = select_attempts([h2, h1])  # out of rank order
    assert np.array_equal(a1, _grid([[1, 2]]))  # rank-0 first
    assert np.array_equal(a2, _grid([[3, 4]]))


def test_select_prefers_diverse_over_rank():
    h1 = _h([[1, 2]], rank=0)
    h2 = _h([[1, 2]], rank=1)  # identical to h1
    h3 = _h([[5, 6]], rank=2)  # diverse
    a1, a2 = select_attempts([h1, h2, h3])
    assert np.array_equal(a1, _grid([[1, 2]]))
    assert np.array_equal(a2, _grid([[5, 6]]))  # skipped identical h2


def test_select_shape_diversity():
    h1 = _h([[1, 2]], rank=0)
    h2 = _h([[1], [2]], rank=1)  # different shape
    a1, a2 = select_attempts([h1, h2])
    assert a1.shape == (1, 2)
    assert a2.shape == (2, 1)


# ---------------------------------------------------------------------------
# build_submission — shape
# ---------------------------------------------------------------------------


def test_submission_shape_correct():
    hyps = {
        "task_a": [[_h([[1, 2]])]],
        "task_b": [[_h([[3]]), _h([[4]])]],
    }
    sub = build_submission(hyps)

    assert set(sub.keys()) == {"task_a", "task_b"}
    for tid, outputs in sub.items():
        for entry in outputs:
            assert "attempt_1" in entry
            assert "attempt_2" in entry
            assert isinstance(entry["attempt_1"], list)
            assert isinstance(entry["attempt_2"], list)


def test_all_task_ids_present():
    task_ids = [f"t{i}" for i in range(10)]
    hyps = {tid: [[_h([[i]])]] for i, tid in enumerate(task_ids)}
    sub = build_submission(hyps)
    assert set(sub.keys()) == set(task_ids)


def test_both_attempts_always_present():
    hyps = {
        "empty": [[]],  # no hypotheses
        "one": [[_h([[1]])]],
        "two": [[_h([[1]]), _h([[2]])]],
    }
    sub = build_submission(hyps)
    for tid, outputs in sub.items():
        for entry in outputs:
            assert "attempt_1" in entry, f"Missing attempt_1 for {tid}"
            assert "attempt_2" in entry, f"Missing attempt_2 for {tid}"


def test_multi_test_output_ordering():
    out0 = _h([[1, 0]], rank=0)
    out1 = _h([[0, 1]], rank=0)
    hyps = {"t1": [[out0], [out1]]}
    sub = build_submission(hyps)
    assert sub["t1"][0]["attempt_1"] == [[1, 0]]
    assert sub["t1"][1]["attempt_1"] == [[0, 1]]


def test_single_hypothesis_duplication():
    hyps = {"t1": [[_h([[7, 8]])]]}
    sub = build_submission(hyps)
    assert sub["t1"][0]["attempt_1"] == [[7, 8]]
    assert sub["t1"][0]["attempt_2"] == [[7, 8]]


def test_empty_hypothesis_placeholder():
    hyps = {"t1": [[]]}
    sub = build_submission(hyps)
    assert sub["t1"][0]["attempt_1"] == [[0]]
    assert sub["t1"][0]["attempt_2"] == [[0]]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path):
    hyps = {"t1": [[_h([[1, 2]])]], "t2": [[_h([[3]])]]}
    sub = build_submission(hyps)
    path = tmp_path / "sub.json"
    save_submission(sub, path)
    loaded = load_submission(path)
    assert loaded == sub


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_single_correct():
    gt = {"t1": [_grid([[1, 2]])]}
    sub = {"t1": [{"attempt_1": [[1, 2]], "attempt_2": [[9, 9]]}]}
    scores = score_submission(sub, gt)
    assert scores["single_attempt_correct"] == 1
    assert scores["two_attempt_correct"] == 1


def test_score_two_attempt_only():
    gt = {"t1": [_grid([[1, 2]])]}
    sub = {"t1": [{"attempt_1": [[9, 9]], "attempt_2": [[1, 2]]}]}
    scores = score_submission(sub, gt)
    assert scores["single_attempt_correct"] == 0
    assert scores["two_attempt_correct"] == 1


def test_score_neither_correct():
    gt = {"t1": [_grid([[1, 2]])]}
    sub = {"t1": [{"attempt_1": [[9, 9]], "attempt_2": [[8, 8]]}]}
    scores = score_submission(sub, gt)
    assert scores["single_attempt_correct"] == 0
    assert scores["two_attempt_correct"] == 0


def test_two_attempt_beats_or_matches_single():
    """Two-attempt score >= single-attempt score on any fixture."""
    gt = {
        "t1": [_grid([[1, 2]])],
        "t2": [_grid([[3, 4]])],
        "t3": [_grid([[5, 6]])],
    }
    sub = {
        "t1": [{"attempt_1": [[1, 2]], "attempt_2": [[0, 0]]}],  # single wins
        "t2": [{"attempt_1": [[0, 0]], "attempt_2": [[3, 4]]}],  # two wins
        "t3": [{"attempt_1": [[0, 0]], "attempt_2": [[0, 0]]}],  # neither
    }
    scores = score_submission(sub, gt)
    assert scores["two_attempt_correct"] >= scores["single_attempt_correct"]
    assert scores["single_attempt_correct"] == 1
    assert scores["two_attempt_correct"] == 2


def test_score_multi_output_task():
    """A task is correct only if ALL test outputs match."""
    gt = {"t1": [_grid([[1]]), _grid([[2]])]}
    sub = {
        "t1": [
            {"attempt_1": [[1]], "attempt_2": [[9]]},
            {"attempt_1": [[9]], "attempt_2": [[2]]},
        ]
    }
    scores = score_submission(sub, gt)
    # Output 0: single=True, Output 1: single=False => task single=False
    # Output 0: two=True, Output 1: two=True => task two=True
    assert scores["single_attempt_correct"] == 0
    assert scores["two_attempt_correct"] == 1


# ---------------------------------------------------------------------------
# hypotheses_from_eval_outcomes
# ---------------------------------------------------------------------------


def test_hypotheses_from_solved_outcome():
    outcomes = [
        {
            "task_id": "t1",
            "solved": True,
            "program": "transpose",
            "test_outputs": [[[1, 2], [3, 4]]],
            "rank": 0,
        }
    ]
    hyps = hypotheses_from_eval_outcomes(outcomes)
    assert "t1" in hyps
    assert len(hyps["t1"]) == 1  # 1 test output
    assert len(hyps["t1"][0]) == 1  # 1 hypothesis
    assert np.array_equal(hyps["t1"][0][0].grid, _grid([[1, 2], [3, 4]]))


def test_hypotheses_from_unsolved_outcome():
    outcomes = [{"task_id": "t1", "solved": False}]
    hyps = hypotheses_from_eval_outcomes(outcomes)
    assert "t1" in hyps
    assert hyps["t1"] == [[]]


def test_hypotheses_merge_multiple_runs():
    outcomes = [
        {
            "task_id": "t1",
            "solved": True,
            "program": "p1",
            "test_outputs": [[[1]]],
            "rank": 0,
        },
        {
            "task_id": "t1",
            "solved": True,
            "program": "p2",
            "test_outputs": [[[2]]],
            "rank": 1,
        },
    ]
    hyps = hypotheses_from_eval_outcomes(outcomes)
    assert len(hyps["t1"][0]) == 2
    assert hyps["t1"][0][0].rank == 0
    assert hyps["t1"][0][1].rank == 1
