"""ARC-AGI-2 submission builder.

The competition allows exactly 2 predictions per task output
(``attempt_1``, ``attempt_2``).  A task output is correct if *either*
attempt matches the ground truth exactly.

This module:
- Selects up to 2 hypotheses per task output (diversity-aware).
- Builds the ``submission.json`` dict consumed by the ARC evaluator.
- Scores a submission against ground-truth tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aria.types import Grid, grid_eq


# ---------------------------------------------------------------------------
# Hypothesis representation
# ---------------------------------------------------------------------------

class Hypothesis:
    """One candidate output grid with provenance metadata."""

    __slots__ = ("grid", "rank", "program_text")

    def __init__(
        self,
        grid: Grid,
        rank: int = 0,
        program_text: str = "",
    ) -> None:
        self.grid = grid
        self.rank = rank
        self.program_text = program_text

    def __repr__(self) -> str:
        return f"Hypothesis(rank={self.rank}, shape={self.grid.shape})"


# ---------------------------------------------------------------------------
# Attempt selection policy
# ---------------------------------------------------------------------------

def _grids_structurally_diverse(a: Grid, b: Grid) -> bool:
    """True when two grids differ in shape or cell content."""
    if a.shape != b.shape:
        return True
    return not np.array_equal(a, b)


def select_attempts(hypotheses: list[Hypothesis]) -> tuple[Grid, Grid]:
    """Pick ``attempt_1`` and ``attempt_2`` from ranked hypotheses.

    Policy (deterministic):
    1. ``attempt_1`` = highest-ranked hypothesis (lowest rank value).
    2. ``attempt_2`` = first structurally-diverse hypothesis after rank-0,
       falling back to the second-ranked, falling back to duplicating
       attempt_1.

    Returns a (attempt_1, attempt_2) tuple of grids.
    """
    if not hypotheses:
        raise ValueError("select_attempts requires at least one hypothesis")

    ordered = sorted(hypotheses, key=lambda h: h.rank)
    best = ordered[0].grid

    if len(ordered) == 1:
        return best, best.copy()

    # Try to find a diverse second attempt.
    for h in ordered[1:]:
        if _grids_structurally_diverse(best, h.grid):
            return best, h.grid

    # All identical — use second-ranked anyway (still a copy).
    return best, ordered[1].grid


# ---------------------------------------------------------------------------
# Placeholder strategy
# ---------------------------------------------------------------------------

def _placeholder_grid() -> Grid:
    """Return a deterministic 1x1 zero grid as the fallback prediction.

    ARC requires every attempt field to contain a valid grid.  A single
    zero cell is the smallest legal grid and will never accidentally
    match a real answer (answers always have meaningful content).
    """
    return np.zeros((1, 1), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Submission builder
# ---------------------------------------------------------------------------

def _grid_to_list(g: Grid) -> list[list[int]]:
    return g.tolist()


def build_submission(
    task_hypotheses: dict[str, list[list[Hypothesis]]],
) -> dict[str, list[dict[str, Any]]]:
    """Build an ARC-AGI-2 ``submission.json`` payload.

    Parameters
    ----------
    task_hypotheses:
        Mapping from ``task_id`` to a list (one entry per test output)
        of hypothesis lists.  Each inner list is sorted by rank
        (best first) and may be empty.

    Returns
    -------
    dict mapping ``task_id`` to a list of dicts, each with keys
    ``attempt_1`` and ``attempt_2`` (nested ``list[list[int]]``).
    """
    submission: dict[str, list[dict[str, Any]]] = {}

    for task_id, per_output in task_hypotheses.items():
        attempts_list: list[dict[str, Any]] = []

        for hypotheses in per_output:
            if hypotheses:
                a1, a2 = select_attempts(hypotheses)
            else:
                ph = _placeholder_grid()
                a1, a2 = ph, ph.copy()

            attempts_list.append({
                "attempt_1": _grid_to_list(a1),
                "attempt_2": _grid_to_list(a2),
            })

        submission[task_id] = attempts_list

    return submission


def save_submission(
    submission: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    """Write a submission dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(submission, f, indent=2)


def load_submission(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load a submission dict from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_submission(
    submission: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, list[Grid]],
) -> dict[str, Any]:
    """Score a submission against ground truth.

    Returns a dict with:
    - ``total``: number of tasks
    - ``single_attempt_correct``: tasks correct with attempt_1 only
    - ``two_attempt_correct``: tasks correct with either attempt
    - ``single_attempt_rate``: fraction
    - ``two_attempt_rate``: fraction
    - ``per_task``: dict of task_id -> {single, two, per_output}
    """
    total = 0
    single_correct = 0
    two_correct = 0
    per_task: dict[str, dict[str, Any]] = {}

    for task_id, gt_outputs in ground_truth.items():
        total += 1
        attempts_list = submission.get(task_id, [])

        all_single = True
        all_two = True
        per_output_results: list[dict[str, bool]] = []

        for idx, gt_grid in enumerate(gt_outputs):
            if idx < len(attempts_list):
                entry = attempts_list[idx]
                a1 = np.array(entry["attempt_1"], dtype=np.uint8)
                a2 = np.array(entry["attempt_2"], dtype=np.uint8)
            else:
                a1 = _placeholder_grid()
                a2 = _placeholder_grid()

            s = grid_eq(a1, gt_grid)
            t = s or grid_eq(a2, gt_grid)

            per_output_results.append({"single": s, "two": t})
            if not s:
                all_single = False
            if not t:
                all_two = False

        if all_single:
            single_correct += 1
        if all_two:
            two_correct += 1

        per_task[task_id] = {
            "single": all_single,
            "two": all_two,
            "per_output": per_output_results,
        }

    return {
        "total": total,
        "single_attempt_correct": single_correct,
        "two_attempt_correct": two_correct,
        "single_attempt_rate": round(single_correct / total, 4) if total else 0.0,
        "two_attempt_rate": round(two_correct / total, 4) if total else 0.0,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Helpers for building hypotheses from eval outcomes
# ---------------------------------------------------------------------------

def hypotheses_from_eval_outcomes(
    outcomes: list[dict[str, Any]],
) -> dict[str, list[list[Hypothesis]]]:
    """Convert a list of eval outcome dicts into the hypothesis structure.

    Each outcome should have ``task_id``, and when solved:
    ``test_outputs`` (list of grid-as-list), ``program`` (text), and
    optionally ``rank``.

    Multiple outcomes for the same ``task_id`` are grouped and sorted by
    rank.  This supports collecting top-k hypotheses from multiple eval
    runs or from a single run that produces ranked candidates.
    """
    from collections import defaultdict

    # Group by task_id: collect (rank, program_text, test_outputs)
    by_task: dict[str, list[tuple[int, str, list[Any]]]] = defaultdict(list)

    for outcome in outcomes:
        tid = outcome["task_id"]
        if not outcome.get("solved"):
            # Ensure the task_id exists even if unsolved.
            if tid not in by_task:
                by_task[tid] = []
            continue

        rank = outcome.get("rank", 0)
        prog = outcome.get("program", "")
        test_outputs = outcome.get("test_outputs", [])
        by_task[tid].append((rank, prog, test_outputs))

    # Build per-output hypothesis lists.
    result: dict[str, list[list[Hypothesis]]] = {}

    for tid, entries in by_task.items():
        if not entries:
            # No solved outcome — determine number of test outputs from
            # the first outcome for this task, defaulting to 1.
            result[tid] = [[]]
            continue

        # Sort by rank.
        entries.sort(key=lambda e: e[0])

        # Determine number of test outputs from the first entry.
        n_outputs = len(entries[0][2]) if entries[0][2] else 1

        per_output: list[list[Hypothesis]] = [[] for _ in range(n_outputs)]

        for rank, prog, test_outputs in entries:
            for out_idx in range(min(len(test_outputs), n_outputs)):
                grid = test_outputs[out_idx]
                if not isinstance(grid, np.ndarray):
                    grid = np.array(grid, dtype=np.uint8)
                per_output[out_idx].append(
                    Hypothesis(grid=grid, rank=rank, program_text=prog)
                )

        result[tid] = per_output

    return result
