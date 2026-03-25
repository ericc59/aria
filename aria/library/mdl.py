"""Minimum Description Length scoring for library admission.

Scores whether naming a sub-sequence makes the total description
of all verified programs shorter.
"""

from __future__ import annotations

from aria.types import (
    Assert,
    Bind,
    Call,
    Expr,
    Lambda,
    LibraryEntry,
    Literal,
    Program,
    Ref,
    Step,
)


def _expr_cost(expr: Expr) -> int:
    """Cost of an expression in description-length units (token count proxy)."""
    match expr:
        case Ref():
            return 1
        case Literal():
            return 1
        case Call(op=_, args=args):
            return 1 + sum(_expr_cost(a) for a in args)
        case Lambda(body=body):
            return 1 + _expr_cost(body)
        case _:
            return 1


def _step_cost(step: Step) -> int:
    """Cost of a single step."""
    match step:
        case Bind(expr=expr):
            return 1 + _expr_cost(expr)  # 1 for the binding itself
        case Assert(pred=pred):
            return 1 + _expr_cost(pred)
        case _:
            return 1


def program_cost(program: Program) -> int:
    """Total description length of a program."""
    return sum(_step_cost(s) for s in program.steps)


def corpus_cost(programs: list[Program]) -> int:
    """Total description length of a set of programs."""
    return sum(program_cost(p) for p in programs)


def _count_matches(
    program: Program,
    entry: LibraryEntry,
) -> int:
    """Count how many times a library entry's step pattern appears in a program.

    Simple heuristic: count contiguous ranges of steps that have the same
    number of steps as the entry and use the same operations in order.
    """
    entry_ops = []
    for s in entry.steps:
        if isinstance(s, Bind) and isinstance(s.expr, Call):
            entry_ops.append(s.expr.op)
        else:
            entry_ops.append(None)

    prog_ops = []
    for s in program.steps:
        if isinstance(s, Bind) and isinstance(s.expr, Call):
            prog_ops.append(s.expr.op)
        else:
            prog_ops.append(None)

    count = 0
    entry_len = len(entry_ops)
    for i in range(len(prog_ops) - entry_len + 1):
        if prog_ops[i:i + entry_len] == entry_ops:
            count += 1
    return count


def mdl_improvement(
    programs: list[Program],
    candidate: LibraryEntry,
) -> int:
    """Compute how much naming this candidate improves total corpus description length.

    Positive = improvement (shorter total description).
    Negative = naming this makes things worse.

    The cost of naming: 1 (for the library entry definition) + entry step costs.
    The savings: for each match in programs, replace N steps with 1 call.
    """
    # Cost to define the new abstraction
    definition_cost = 1 + sum(_step_cost(s) for s in candidate.steps)

    # Savings per replacement: (original steps cost) - (1 call cost)
    original_cost = sum(_step_cost(s) for s in candidate.steps)
    replacement_cost = 1 + len(candidate.params)  # call + args
    saving_per_match = original_cost - replacement_cost

    total_matches = sum(_count_matches(p, candidate) for p in programs)

    return (saving_per_match * total_matches) - definition_cost
