"""Bootstrap-propose-verify loop — deterministic seed generation.

Part of the canonical architecture. Uses hand-written fitters to
bootstrap a GraphLibrary of verified templates, then uses the
compositional proposer to construct new hypotheses from library
fragments.

This is deterministic seed generation for the graph editor, not
the long-term learning system. The future per-task recurrent editor
(aria.core.editor_env) will replace the proposer as the primary
hypothesis generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from aria.core.graph import (
    CompileSuccess,
    ComputationGraph,
    Specialization,
)
from aria.core.library import GraphLibrary
from aria.core.proposer import propose_from_library
from aria.core.protocol import (
    Compiler,
    Fitter,
    SolveResult,
    Specializer,
    Verifier,
)


@dataclass(frozen=True)
class LearnResult:
    """Outcome of a learn-propose-verify pass over a task set."""
    phase1_solved: int          # solved by hand-written fitters
    phase2_solved: int          # additionally solved by library proposer
    total_solved: int
    total_tasks: int
    library_size: int           # templates in library after this pass
    solved_task_ids: tuple[str, ...]
    phase2_task_ids: tuple[str, ...]  # tasks solved ONLY by library proposer


def learn_and_propose(
    tasks: Sequence[tuple[str, Sequence[Any]]],
    fitter: Fitter,
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    library: GraphLibrary | None = None,
    max_proposals_per_task: int = 50,
) -> LearnResult:
    """Run the full learn-propose-verify loop.

    1. Bootstrap: try hand-written fitters on all tasks
    2. Propose: for unsolved tasks, propose from library
    3. Record what was solved by each phase
    """
    if library is None:
        library = GraphLibrary()

    solved_ids: list[str] = []
    phase2_ids: list[str] = []
    unsolved: list[tuple[str, Sequence[Any]]] = []

    # --- Phase 1: Bootstrap with fitters ---
    for task_id, examples in tasks:
        result = _try_fitters(
            task_id, examples, fitter, specializer, compiler, verifier,
        )
        if result is not None:
            graph, spec, program = result
            library.add(graph, spec, source_task_id=task_id)
            solved_ids.append(task_id)
        else:
            unsolved.append((task_id, examples))

    phase1_count = len(solved_ids)

    # --- Phase 2: Propose from library ---
    if library.size > 0 and unsolved:
        newly_solved = _propose_pass(
            unsolved, library, specializer, compiler, verifier,
            max_proposals=max_proposals_per_task,
        )
        for task_id, graph, spec, program in newly_solved:
            library.add(graph, spec, source_task_id=task_id)
            solved_ids.append(task_id)
            phase2_ids.append(task_id)

    return LearnResult(
        phase1_solved=phase1_count,
        phase2_solved=len(phase2_ids),
        total_solved=len(solved_ids),
        total_tasks=len(tasks),
        library_size=library.size,
        solved_task_ids=tuple(solved_ids),
        phase2_task_ids=tuple(phase2_ids),
    )


def _try_fitters(
    task_id: str,
    examples: Sequence[Any],
    fitter: Fitter,
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
) -> tuple[ComputationGraph, Specialization, Any] | None:
    """Try hand-written fitters, return (graph, spec, program) or None."""
    graphs = fitter.fit(examples, task_id=task_id)
    for graph in graphs:
        spec = specializer.specialize(graph, examples)
        result = compiler.compile(graph, spec, examples)
        if isinstance(result, CompileSuccess) and result.scope == "task":
            vr = verifier.verify(result.program, examples)
            if vr.passed:
                return graph, spec, result.program
    return None


def _propose_pass(
    unsolved: list[tuple[str, Sequence[Any]]],
    library: GraphLibrary,
    specializer: Specializer,
    compiler: Compiler,
    verifier: Verifier,
    *,
    max_proposals: int = 50,
) -> list[tuple[str, ComputationGraph, Specialization, Any]]:
    """Propose and verify graphs for unsolved tasks using the library."""
    newly_solved = []

    for task_id, examples in unsolved:
        proposals = propose_from_library(
            library, examples, task_id=task_id,
            max_proposals=max_proposals,
        )

        for graph in proposals:
            try:
                spec = specializer.specialize(graph, examples)
                result = compiler.compile(graph, spec, examples)
                if isinstance(result, CompileSuccess) and result.scope == "task":
                    vr = verifier.verify(result.program, examples)
                    if vr.passed:
                        newly_solved.append((task_id, graph, spec, result.program))
                        break  # solved, move to next task
            except Exception:
                continue

    return newly_solved
