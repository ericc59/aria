"""Program retrieval against persisted verified programs.

Also provides abstraction retrieval: ranking library entries by
relevance to a task's signatures and provenance strength, so search
can use strong abstractions as preferred ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.program_store import ProgramStore, StoredProgram
from aria.proposer.parser import ParseError, parse_program
from aria.runtime.type_system import type_check
from aria.types import DemoPair, LibraryEntry, Program, Type, VerifyResult
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Whole-program retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Why this program ranked where it did and how transfer-worthy it is."""

    signature_overlap: int
    distinct_task_count: int
    has_non_retrieval_source: bool
    retrieval_echo_count: int
    retrieval_bucket: str  # "transfer" or "fallback"


@dataclass(frozen=True)
class RetrievalHit:
    record: StoredProgram
    program: Program
    verify_result: VerifyResult
    candidates_tried: int
    diagnostics: RetrievalDiagnostics | None = None


def retrieve_program(
    demos: tuple[DemoPair, ...],
    program_store: ProgramStore,
    *,
    max_candidates: int = 0,
) -> RetrievalHit | None:
    """Replay persisted verified programs and return the first exact match.

    Retrieval is staged: try programs with evidence of transfer first, then
    fall back to one-off leaves if nothing generalizable verifies.
    """
    task_signatures = compute_task_signatures(demos)
    transfer_first = program_store.ranked_records(task_signatures, transfer_only=True)
    transfer_set = set(id(r) for r in transfer_first)
    fallback = [
        record for record in program_store.ranked_records(task_signatures)
        if id(record) not in transfer_set
    ]

    idx = 0
    for bucket_name, bucket in (("transfer", transfer_first), ("fallback", fallback)):
        for record in bucket:
            idx += 1
            if max_candidates and idx > max_candidates:
                return None

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
                overlap = len(task_signatures & set(record.signatures))
                return RetrievalHit(
                    record=record,
                    program=program,
                    verify_result=verify_result,
                    candidates_tried=idx,
                    diagnostics=RetrievalDiagnostics(
                        signature_overlap=overlap,
                        distinct_task_count=record.distinct_task_count,
                        has_non_retrieval_source=record.has_non_retrieval_source,
                        retrieval_echo_count=record.retrieval_echo_count,
                        retrieval_bucket=bucket_name,
                    ),
                )

    return None


def retrieval_provenance(record: StoredProgram) -> dict[str, Any]:
    """Build a provenance dict from a StoredProgram for reporting."""
    return {
        "distinct_task_count": record.distinct_task_count,
        "has_non_retrieval_source": record.has_non_retrieval_source,
        "retrieval_echo_count": record.retrieval_echo_count,
        "task_ids": list(record.task_ids),
        "sources": list(record.sources),
    }


# ---------------------------------------------------------------------------
# Abstraction retrieval — library entries as search guidance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractionHint:
    """A library abstraction retrieved as search guidance for a task."""

    name: str
    score: float
    signature_overlap: int
    support_task_count: int
    support_program_count: int
    mdl_gain: int
    strength: str  # "strong" or "weak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "signature_overlap": self.signature_overlap,
            "support_task_count": self.support_task_count,
            "support_program_count": self.support_program_count,
            "mdl_gain": self.mdl_gain,
            "strength": self.strength,
        }


def retrieve_abstractions(
    demos: tuple[DemoPair, ...],
    library: Library,
    *,
    max_hints: int = 10,
) -> list[AbstractionHint]:
    """Rank library abstractions by relevance to a task.

    Returns a deterministic, score-ordered list of hints. Strong abstractions
    (multi-task support + positive MDL gain) rank above weak ones.
    """
    if not library.all_entries():
        return []

    task_signatures = compute_task_signatures(demos)
    hints: list[AbstractionHint] = []

    for entry in library.all_entries():
        overlap = len(task_signatures & set(entry.signatures))
        support_task_count = len(entry.support_task_ids)
        strength = (
            "strong"
            if support_task_count >= 2 and entry.mdl_gain > 0
            else "weak"
        )
        score = _abstraction_score(
            signature_overlap=overlap,
            support_task_count=support_task_count,
            support_program_count=entry.support_program_count,
            mdl_gain=entry.mdl_gain,
            step_count=len(entry.steps),
        )
        hints.append(AbstractionHint(
            name=entry.name,
            score=score,
            signature_overlap=overlap,
            support_task_count=support_task_count,
            support_program_count=entry.support_program_count,
            mdl_gain=entry.mdl_gain,
            strength=strength,
        ))

    hints.sort(key=lambda h: (-h.score, h.name))
    return hints[:max_hints]


def preferred_ops_from_hints(hints: list[AbstractionHint]) -> frozenset[str]:
    """Extract op names that search should try earlier."""
    return frozenset(h.name for h in hints if h.score > 0)


def _abstraction_score(
    *,
    signature_overlap: int,
    support_task_count: int,
    support_program_count: int,
    mdl_gain: int,
    step_count: int,
) -> float:
    """Deterministic score for ranking abstraction relevance.

    Higher = more relevant. The scoring is deliberately simple:
    signature overlap is the primary signal, then transfer evidence,
    then compression benefit, with a small simplicity bonus.
    """
    score = 0.0
    score += 20.0 * signature_overlap
    score += 8.0 * min(support_task_count, 5)
    score += 3.0 * min(support_program_count, 5)
    score += 2.0 * min(max(mdl_gain, 0), 10)
    score += max(0.0, 3.0 - step_count)  # prefer simpler
    return score


