"""Top-level offline solver: Task -> Answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aria.graph.signatures import compute_task_signatures
from aria.library.admission import check_admission
from aria.library.mining import mine_abstractions
from aria.library.store import Library
from aria.offline_search import search_program
from aria.program_store import ProgramStore
from aria.retrieval import retrieve_program
from aria.types import DemoPair, Grid, Program, Task, TaskContext, VerifyMode
from aria.verify.mode import detect_mode


@dataclass
class SolveResult:
    task_id: str | None
    solved: bool
    test_outputs: list[Grid]
    winning_program: Program | None
    abstractions_mined: int
    retrieved: bool = False
    searched: bool = False
    retrieval_candidates_tried: int = 0
    search_candidates_tried: int = 0
    task_signatures: tuple[str, ...] = ()


def solve_task(
    task: Task,
    library: Library,
    program_store: ProgramStore | None = None,
    *,
    task_id: str | None = None,
    retrieval_limit: int = 0,
    max_search_steps: int = 3,
    max_search_candidates: int = 5000,
    include_core_ops: bool = True,
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

    search_result = search_program(
        task.train,
        library,
        program_store=program_store,
        max_steps=max_search_steps,
        max_candidates=max_search_candidates,
        include_core_ops=include_core_ops,
    )

    if not search_result.solved or search_result.winning_program is None:
        return SolveResult(
            task_id=task_id,
            solved=False,
            test_outputs=[],
            winning_program=None,
            abstractions_mined=0,
            searched=True,
            search_candidates_tried=search_result.candidates_tried,
            task_signatures=task_signatures,
        )

    program = search_result.winning_program
    test_outputs = _execute_on_test(task, program)
    if program_store is not None:
        program_store.add_program(
            program,
            task_id=task_id,
            source="offline-search",
            signatures=frozenset(task_signatures),
        )

    mined = mine_abstractions([program])
    abstractions_admitted = 0
    for candidate in mined:
        admitted, _reason = check_admission(
            candidate,
            [program],
            library,
            min_uses=1,
        )
        if admitted:
            library.add(candidate)
            abstractions_admitted += 1

    return SolveResult(
        task_id=task_id,
        solved=True,
        test_outputs=test_outputs,
        winning_program=program,
        abstractions_mined=abstractions_admitted,
        searched=True,
        search_candidates_tried=search_result.candidates_tried,
        task_signatures=task_signatures,
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
