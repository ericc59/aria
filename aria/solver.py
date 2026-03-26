"""Top-level offline solver: Task -> Answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.refinement import RefinementResult, run_refinement_loop
from aria.retrieval import retrieve_program
from aria.types import DemoPair, Grid, Program, Task, TaskContext, VerifyMode
from aria.verify.mode import detect_mode

if TYPE_CHECKING:
    from aria.local_policy import LocalPolicy


@dataclass
class SolveResult:
    task_id: str | None
    solved: bool
    test_outputs: list[Grid]
    winning_program: Program | None
    abstractions_mined: int
    retrieved: bool = False
    searched: bool = False
    refined: bool = False
    retrieval_candidates_tried: int = 0
    search_candidates_tried: int = 0
    refinement_rounds: int = 0
    task_signatures: tuple[str, ...] = ()
    refinement_result: RefinementResult | None = None


def solve_task(
    task: Task,
    library: Library,
    program_store: ProgramStore | None = None,
    *,
    task_id: str | None = None,
    retrieval_limit: int = 0,
    max_search_steps: int = 3,
    max_search_candidates: int = 5000,
    max_refinement_rounds: int = 2,
    include_core_ops: bool = True,
    beam_width: int = 0,
    beam_rounds: int = 3,
    beam_mutations_per_candidate: int = 30,
    local_policy: LocalPolicy | None = None,
) -> SolveResult:
    """Solve a single ARC task without any proposer model."""
    task_signatures = tuple(sorted(compute_task_signatures(task.train)))

    retrieval_hit = None
    if program_store is not None and len(program_store) > 0:
        retrieval_hit = retrieve_program(
            task.train,
            program_store,
            max_candidates=retrieval_limit,
        )

    if retrieval_hit is not None:
        program = retrieval_hit.program
        test_outputs = _execute_on_test(task, program)
        if program_store is not None:
            program_store.add_program(
                program,
                task_id=task_id,
                source="offline-retrieval",
                signatures=frozenset(task_signatures),
            )
        return SolveResult(
            task_id=task_id,
            solved=True,
            test_outputs=test_outputs,
            winning_program=program,
            abstractions_mined=0,
            retrieved=True,
            retrieval_candidates_tried=retrieval_hit.candidates_tried,
            task_signatures=task_signatures,
        )

    refinement_result = run_refinement_loop(
        task.train,
        library,
        program_store=program_store,
        max_steps=max_search_steps,
        max_candidates=max_search_candidates,
        max_rounds=max_refinement_rounds,
        include_core_ops=include_core_ops,
        local_policy=local_policy,
        beam_width=beam_width,
        beam_rounds=beam_rounds,
        beam_mutations_per_candidate=beam_mutations_per_candidate,
    )

    if not refinement_result.solved or refinement_result.winning_program is None:
        return SolveResult(
            task_id=task_id,
            solved=False,
            test_outputs=[],
            winning_program=None,
            abstractions_mined=0,
            searched=True,
            refined=max_refinement_rounds > 1,
            search_candidates_tried=refinement_result.candidates_tried,
            refinement_rounds=len(refinement_result.rounds),
            task_signatures=task_signatures,
            refinement_result=refinement_result,
        )

    program = refinement_result.winning_program
    test_outputs = _execute_on_test(task, program)
    if program_store is not None:
        program_store.add_program(
            program,
            task_id=task_id,
            source="offline-search",
            signatures=frozenset(task_signatures),
        )

    return SolveResult(
        task_id=task_id,
        solved=True,
        test_outputs=test_outputs,
        winning_program=program,
        abstractions_mined=0,
        searched=True,
        refined=max_refinement_rounds > 1,
        search_candidates_tried=refinement_result.candidates_tried,
        refinement_rounds=len(refinement_result.rounds),
        task_signatures=task_signatures,
        refinement_result=refinement_result,
    )


def _execute_on_test(task: Task, program: Program) -> list[Grid]:
    """Run a verified program on task test inputs."""
    from aria.runtime.executor import execute

    mode = detect_mode(program)
    test_outputs: list[Grid] = []

    for test_pair in task.test:
        if mode == VerifyMode.STATELESS:
            ctx = None
        else:
            ctx = TaskContext(demos=task.train)

        output = execute(program, test_pair.input, ctx)
        test_outputs.append(output)

    return test_outputs


def load_task(data: dict[str, Any]) -> Task:
    """Load a task from the standard ARC JSON format."""
    from aria.types import grid_from_list

    train = tuple(
        DemoPair(
            input=grid_from_list(pair["input"]),
            output=grid_from_list(pair["output"]),
        )
        for pair in data["train"]
    )
    test = tuple(
        DemoPair(
            input=grid_from_list(pair["input"]),
            output=grid_from_list(pair.get("output", [[0]])),
        )
        for pair in data["test"]
    )
    return Task(train=train, test=test)
