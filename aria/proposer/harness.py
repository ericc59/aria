"""Proposer harness — manages the propose/verify/retry loop.

This is the integration point between the proposer model and the runtime.
The actual model call is abstracted behind a callable interface so we can
swap between local models, API calls, or mock proposers for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from aria.library.store import Library
from aria.proposer.parser import ParseError, parse_program
from aria.proposer.prompt import build_prompt
from aria.proposer.serializer import build_proposer_input
from aria.runtime.ops import all_ops
from aria.runtime.type_system import type_check
from aria.runtime.executor import ExecutionError, execute
from aria.types import (
    DemoPair,
    Delta,
    Grid,
    Program,
    StateGraph,
    StepTraceEntry,
    TaskContext,
    Type,
    VerifyResult,
    VerifyMode,
    grid_eq,
)
from aria.verify.verifier import verify
from aria.verify.mode import detect_mode


class ProposerModel(Protocol):
    """Interface for the proposer model."""

    def generate(self, prompt: str, n: int) -> list[str]:
        """Generate n candidate program texts given a prompt."""
        ...


@dataclass
class ProposalAttempt:
    program_text: str
    program: Program | None
    parse_error: str | None
    execution_error: str | None
    verify_result: VerifyResult | None


@dataclass
class ProposalResult:
    solved: bool
    winning_program: Program | None
    winning_result: VerifyResult | None
    rounds: int
    total_candidates: int
    all_attempts: list[list[ProposalAttempt]] = field(default_factory=list)


def _try_execute(program: Program, demos: tuple[DemoPair, ...]) -> str | None:
    """Try executing a program on all demos. Returns error string or None."""
    mode = detect_mode(program)
    for i, demo in enumerate(demos):
        if mode == VerifyMode.STATELESS:
            ctx = None
        else:
            ctx = TaskContext(demos=demos)
        try:
            execute(program, demo.input, ctx)
        except ExecutionError as e:
            return f"Demo {i}: {e}"
        except Exception as e:
            return f"Demo {i}: {type(e).__name__}: {e}"
    return None


def _format_error_feedback(attempts: list[ProposalAttempt]) -> list[dict[str, Any]]:
    """Build structured error feedback from failed attempts.

    Prioritizes: execution errors (most actionable) > wrong output > parse errors.
    Deduplicates by error message. Caps at 3 entries.
    """
    feedback: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Execution errors first — most actionable
    for att in attempts:
        if att.execution_error and att.execution_error not in seen:
            seen.add(att.execution_error)
            feedback.append({
                "attempt_num": len(feedback) + 1,
                "program_text": att.program_text,
                "error_type": "execution_error",
                "error_detail": att.execution_error,
            })

    # Wrong output — include actual vs expected grids
    for att in attempts:
        if att.verify_result and not att.verify_result.passed and att.program:
            key = f"wrong:{att.verify_result.failed_demo}:{att.program_text[:100]}"
            if key not in seen:
                seen.add(key)
                entry: dict[str, Any] = {
                    "attempt_num": len(feedback) + 1,
                    "program_text": att.program_text,
                    "error_type": att.verify_result.error_type or "wrong_output",
                    "failed_demo": att.verify_result.failed_demo,
                    "diff": att.verify_result.diff,
                }
                if att.verify_result.step_trace:
                    entry["step_trace"] = [
                        {
                            "step_name": e.step_name,
                            "value": e.value,
                            "ok": e.ok,
                            "suspect": e.suspect,
                        }
                        for e in att.verify_result.step_trace
                    ]
                feedback.append(entry)

    # Parse errors last
    for att in attempts:
        if att.parse_error:
            key = att.parse_error[:80]
            if key not in seen:
                seen.add(key)
                feedback.append({
                    "attempt_num": len(feedback) + 1,
                    "program_text": att.program_text[:200],
                    "error_type": "parse_error",
                    "error_detail": att.parse_error,
                })

    return feedback[:4]


def propose_and_verify(
    model: ProposerModel,
    demos: tuple[DemoPair, ...],
    state_graphs: list[StateGraph],
    deltas: list[Delta],
    library: Library,
    max_rounds: int = 4,
    k: int | None = None,
    task_id: str = "unknown",
) -> ProposalResult:
    """Run the full propose/verify/retry loop."""
    if k is None:
        k = 4
    core_ops = list(all_ops().keys())
    library_index = library.index_for_proposer()

    all_feedback: list[dict[str, Any]] = []
    all_round_attempts: list[list[ProposalAttempt]] = []
    total_candidates = 0

    for round_num in range(1, max_rounds + 1):
        proposer_input = build_proposer_input(
            state_graphs=state_graphs,
            deltas=deltas,
            core_ops=core_ops,
            library_ops=library_index,
            prior_attempts=all_feedback if round_num > 1 else None,
        )
        prompt = build_prompt(
            round_num, proposer_input,
            demos=demos,
            prior_attempts=all_feedback if round_num > 1 else None,
            task_id=task_id,
        )

        candidate_texts = model.generate(prompt, k)
        total_candidates += len(candidate_texts)

        round_attempts: list[ProposalAttempt] = []

        for text in candidate_texts:
            # Skip error markers from the proposer
            if text.startswith("--"):
                round_attempts.append(ProposalAttempt(
                    program_text=text,
                    program=None,
                    parse_error=text,
                    execution_error=None,
                    verify_result=None,
                ))
                continue

            # Parse
            try:
                program = parse_program(text)
            except ParseError as e:
                round_attempts.append(ProposalAttempt(
                    program_text=text,
                    program=None,
                    parse_error=str(e),
                    execution_error=None,
                    verify_result=None,
                ))
                continue

            # Static type check before execution
            type_errors = type_check(
                program,
                initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
            )
            if type_errors:
                round_attempts.append(ProposalAttempt(
                    program_text=text,
                    program=program,
                    parse_error=None,
                    execution_error="Type check failed: " + "; ".join(type_errors[:4]),
                    verify_result=None,
                ))
                continue

            # Try execution first to catch type errors
            exec_err = _try_execute(program, demos)
            if exec_err:
                round_attempts.append(ProposalAttempt(
                    program_text=text,
                    program=program,
                    parse_error=None,
                    execution_error=exec_err,
                    verify_result=None,
                ))
                continue

            # Verify
            result = verify(program, demos)
            attempt = ProposalAttempt(
                program_text=text,
                program=program,
                parse_error=None,
                execution_error=None,
                verify_result=result,
            )
            round_attempts.append(attempt)

            if result.passed:
                all_round_attempts.append(round_attempts)
                return ProposalResult(
                    solved=True,
                    winning_program=program,
                    winning_result=result,
                    rounds=round_num,
                    total_candidates=total_candidates,
                    all_attempts=all_round_attempts,
                )

        all_round_attempts.append(round_attempts)

        # Build feedback for next round
        new_feedback = _format_error_feedback(round_attempts)
        all_feedback.extend(new_feedback)
        # Keep only the most recent/relevant feedback
        all_feedback = all_feedback[-4:]

    return ProposalResult(
        solved=False,
        winning_program=None,
        winning_result=None,
        rounds=max_rounds,
        total_candidates=total_candidates,
        all_attempts=all_round_attempts,
    )
