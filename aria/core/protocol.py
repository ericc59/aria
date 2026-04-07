"""Domain protocol — interfaces a domain must implement.

A domain instantiation provides:
  1. Example type — what an input/output pair looks like
  2. Program type — what an executable program looks like
  3. Fitter — propose ComputationGraphs from examples
  4. Specializer — extract Specialization from graph + examples
  5. Compiler — compile graph + specialization into program
  6. Verifier — check program against examples (exact, binary)

The framework orchestrates: fit → specialize → compile → verify.
The domain provides the implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from aria.core.graph import (
    CompileResult,
    ComputationGraph,
    Specialization,
)


class Example(Protocol):
    """A single input/output example pair."""

    @property
    def input(self) -> Any: ...

    @property
    def output(self) -> Any: ...


class VerifyResult(Protocol):
    """Result of verifying a program against examples."""

    @property
    def passed(self) -> bool: ...


class Fitter(Protocol):
    """Propose computation graph hypotheses from examples."""

    def fit(
        self,
        examples: Sequence[Example],
        task_id: str = "",
    ) -> list[ComputationGraph]:
        """Return zero or more graph hypotheses."""
        ...


class Specializer(Protocol):
    """Extract resolved static structure from a graph and examples."""

    def specialize(
        self,
        graph: ComputationGraph,
        examples: Sequence[Example],
    ) -> Specialization:
        """Return a specialization bundle."""
        ...


class Compiler(Protocol):
    """Compile a graph + specialization into an executable program."""

    def compile(
        self,
        graph: ComputationGraph,
        specialization: Specialization,
        examples: Sequence[Example],
    ) -> CompileResult:
        """Return success with program, or structured failure."""
        ...


class Verifier(Protocol):
    """Check a program against examples. Exact, binary, no partial credit."""

    def verify(
        self,
        program: Any,
        examples: Sequence[Example],
    ) -> VerifyResult:
        """Return pass/fail."""
        ...


# ---------------------------------------------------------------------------
# Pipeline — the orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveAttempt:
    """Record of one attempt: graph → specialize → compile → verify."""
    graph: ComputationGraph
    specialization: Specialization
    compile_result: CompileResult
    verified: bool
    program: Any | None = None


@dataclass(frozen=True)
class SolveResult:
    """Outcome of running the full pipeline on a task."""
    task_id: str
    solved: bool
    winning_program: Any | None = None
    attempts: tuple[SolveAttempt, ...] = ()
    graphs_proposed: int = 0
    graphs_compiled: int = 0
    graphs_verified: int = 0


def solve(
    examples: Sequence[Example],
    fitter: Fitter,
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    task_id: str = "",
) -> SolveResult:
    """The core pipeline: fit → specialize → compile → verify.

    Domain-independent.  The domain provides the four components.
    """
    from aria.core.graph import CompileSuccess

    graphs = fitter.fit(examples, task_id=task_id)
    if not graphs:
        return SolveResult(task_id=task_id, solved=False)

    attempts: list[SolveAttempt] = []
    compiled = 0
    verified = 0

    for graph in graphs:
        spec = specializer.specialize(graph, examples)
        result = compiler.compile(graph, spec, examples)

        is_success = isinstance(result, CompileSuccess)
        if is_success:
            compiled += 1

        is_verified = False
        program = None
        if is_success:
            program = result.program
            vr = verifier.verify(program, examples)
            if vr.passed:
                is_verified = True
                verified += 1

        attempt = SolveAttempt(
            graph=graph,
            specialization=spec,
            compile_result=result,
            verified=is_verified,
            program=program if is_verified else None,
        )
        attempts.append(attempt)

        if is_verified:
            return SolveResult(
                task_id=task_id,
                solved=True,
                winning_program=program,
                attempts=tuple(attempts),
                graphs_proposed=len(graphs),
                graphs_compiled=compiled,
                graphs_verified=verified,
            )

    return SolveResult(
        task_id=task_id,
        solved=False,
        attempts=tuple(attempts),
        graphs_proposed=len(graphs),
        graphs_compiled=compiled,
        graphs_verified=verified,
    )
