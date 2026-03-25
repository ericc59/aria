"""Program retrieval against persisted verified programs."""

from __future__ import annotations

from dataclasses import dataclass

from aria.graph.signatures import compute_task_signatures
from aria.program_store import ProgramStore, StoredProgram
from aria.proposer.parser import ParseError, parse_program
from aria.runtime.type_system import type_check
from aria.types import DemoPair, Program, Type, VerifyResult
from aria.verify.verifier import verify


@dataclass(frozen=True)
class RetrievalHit:
    record: StoredProgram
    program: Program
    verify_result: VerifyResult
    candidates_tried: int


def retrieve_program(
    demos: tuple[DemoPair, ...],
    program_store: ProgramStore,
    *,
    max_candidates: int = 0,
) -> RetrievalHit | None:
    """Replay persisted verified programs and return the first exact match."""
    task_signatures = compute_task_signatures(demos)

    for idx, record in enumerate(program_store.ranked_records(task_signatures), start=1):
        if max_candidates and idx > max_candidates:
            break

        try:
            program = parse_program(record.program_text)
        except ParseError:
            continue

        type_errors = type_check(
            program,
            initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
        )
        if type_errors:
            continue

        verify_result = verify(program, demos)
        if verify_result.passed:
            return RetrievalHit(
                record=record,
                program=program,
                verify_result=verify_result,
                candidates_tried=idx,
            )

    return None
