"""Capture structured solve traces from search results.

Bridges the search engine output (SearchProgram/ASTProgram) to the
trace schema. Called after verification to record successful solves.
"""

from __future__ import annotations

from aria.search.trace_schema import SolveTrace
from aria.search.sketch import SearchProgram


def capture_solve_trace(
    *,
    task_id: str,
    task_signatures: tuple[str, ...],
    program: SearchProgram,
    n_demos: int = 0,
    test_correct: bool | None = None,
) -> SolveTrace:
    """Build a SolveTrace from a verified SearchProgram."""
    step_actions = tuple(s.action for s in program.steps)
    step_selectors = tuple(_summarize_selector(s.select) for s in program.steps)

    return SolveTrace(
        task_id=task_id,
        task_signatures=task_signatures,
        provenance=program.provenance,
        step_actions=step_actions,
        step_selectors=step_selectors,
        program_dict=program.to_dict(),
        n_demos=n_demos,
        n_steps=len(program.steps),
        test_correct=test_correct,
    )


def _summarize_selector(sel) -> str:
    """One-line summary of a StepSelect for trace storage."""
    if sel is None:
        return ''
    if sel.role == 'by_color':
        return f'color={sel.params.get("color", "?")}'
    if sel.role == 'by_rule':
        rule = sel.params.get('rule', {})
        clauses = rule.get('clauses', [])
        parts = []
        for clause in clauses:
            atoms = clause.get('atoms', [])
            conj = ' & '.join(
                f'{a["field"]}={"T" if a["value"] else "F"}' for a in atoms
            )
            parts.append(conj)
        return ' | '.join(parts) if parts else 'rule(empty)'
    if sel.role == 'by_predicate':
        return 'predicate'
    return sel.role
