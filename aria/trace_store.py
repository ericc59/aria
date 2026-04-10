"""Persistence for verifier-driven refinement traces.

Stores both flat-search rounds and beam refinement transitions
in a format suitable for later training a local model:
  (task features, current program, verifier feedback) -> next edit
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from aria.refinement import BeamRefinementResult, RefinementResult
from aria.runtime.program import program_to_text
from aria.types import Program


class RefinementTraceStore:
    """Append-only JSON store for refinement trajectories."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def __len__(self) -> int:
        return len(self._records)

    def add_result(
        self,
        *,
        task_id: str | None,
        result: RefinementResult,
        task_signatures: tuple[str, ...] = (),
    ) -> None:
        record: dict = {
            "task_id": task_id,
            "solved": result.solved,
            "solve_source": "refinement",
            "candidates_tried": result.candidates_tried,
            "task_signatures": list(task_signatures),
            "winning_program": (
                program_to_text(result.winning_program)
                if result.winning_program is not None
                else None
            ),
            "rounds": [
                _serialize_round(round_result)
                for round_result in result.rounds
            ],
        }

        if result.beam is not None:
            record["beam"] = _serialize_beam(result.beam)

        self._records.append(record)

    def add_retrieval_result(
        self,
        *,
        task_id: str | None,
        winning_program: Program,
        candidates_tried: int,
        task_signatures: tuple[str, ...] = (),
    ) -> None:
        self._records.append({
            "task_id": task_id,
            "solved": True,
            "solve_source": "retrieval",
            "candidates_tried": candidates_tried,
            "task_signatures": list(task_signatures),
            "winning_program": program_to_text(winning_program),
            "rounds": [],
            "retrieval": {
                "candidates_tried": candidates_tried,
            },
        })

    def add_search_result(
        self,
        *,
        task_id: str | None,
        solved: bool,
        winning_program_text: str | None = None,
        task_signatures: tuple[str, ...] = (),
    ) -> None:
        """Record a minimal search-only solve attempt.

        The current canonical `aria/search` path does not emit refinement-round
        traces, but eval still uses the trace store as a per-task record sink.
        """
        self._records.append({
            "task_id": task_id,
            "solved": solved,
            "solve_source": "search" if solved else "unsolved",
            "candidates_tried": 0,
            "task_signatures": list(task_signatures),
            "winning_program": winning_program_text if solved else None,
            "rounds": [],
            "search": {"canonical": True},
        })

    def all_records(self) -> list[dict]:
        return list(self._records)

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2, "records": self._records}
        tmp_path = output.with_name(f".{output.name}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output)

    @classmethod
    def load_json(cls, path: str | Path) -> RefinementTraceStore:
        source = Path(path)
        store = cls()
        if not source.exists():
            return store
        try:
            with open(source) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt refinement trace store: {source}. "
                "This usually means a previous run was interrupted mid-write."
            ) from exc
        version = data.get("version")
        if version not in (1, 2) or not isinstance(data.get("records"), list):
            raise ValueError(f"Unsupported refinement trace store format: {source}")
        store._records = list(data["records"])
        return store


def _serialize_round(round_result) -> dict:  # type: ignore[no-untyped-def]
    return {
        "plan": {
            "name": round_result.plan.name,
            "max_steps": round_result.plan.max_steps,
            "max_candidates": round_result.plan.max_candidates,
            "allowed_ops": sorted(round_result.plan.allowed_ops or ()),
        },
        "solved": round_result.solved,
        "candidates_tried": round_result.candidates_tried,
        "feedback": asdict(round_result.feedback),
        "winning_program": (
            program_to_text(round_result.winning_program)
            if round_result.winning_program is not None
            else None
        ),
        "trace": [asdict(entry) for entry in round_result.trace],
    }


def _serialize_beam(beam: BeamRefinementResult) -> dict:
    return {
        "solved": beam.solved,
        "candidates_scored": beam.candidates_scored,
        "rounds_completed": beam.rounds_completed,
        "best_score": beam.best_score,
        "best_program_text": beam.best_program_text,
        "winning_program": (
            program_to_text(beam.winning_program)
            if beam.winning_program is not None
            else None
        ),
        "round_summaries": [asdict(rs) for rs in beam.round_summaries],
        "transitions": [asdict(t) for t in beam.transitions],
    }
