"""Subsequence extraction from verified programs.

Identifies self-contained "recipes" within verified programs that
can be parameterized and admitted as library entries.
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
    Type,
)


def _refs_in_expr(expr: Expr) -> set[str]:
    """Collect all name references in an expression."""
    match expr:
        case Ref(name=name):
            return {name}
        case Literal():
            return set()
        case Call(args=args):
            result: set[str] = set()
            for a in args:
                result |= _refs_in_expr(a)
            return result
        case Lambda(param=param, body=body):
            return _refs_in_expr(body) - {param}
        case _:
            return set()


def _names_bound(steps: tuple[Step, ...]) -> set[str]:
    """Names bound by a sequence of steps."""
    return {s.name for s in steps if isinstance(s, Bind)}


def _names_used(steps: tuple[Step, ...]) -> set[str]:
    """Names referenced by a sequence of steps."""
    refs: set[str] = set()
    for s in steps:
        match s:
            case Bind(expr=expr):
                refs |= _refs_in_expr(expr)
            case Assert(pred=pred):
                refs |= _refs_in_expr(pred)
    return refs


def extract_candidates(
    program: Program,
    min_length: int = 2,
    max_length: int = 6,
) -> list[tuple[int, int]]:
    """Find candidate subsequence ranges (start, end) in a program.

    A candidate is a contiguous range of steps where:
    - The inputs (names referenced but not bound within) come from before the range
    - At least one output (name bound within) is used after the range
    - Length is within [min_length, max_length]
    """
    steps = program.steps
    n = len(steps)
    candidates: list[tuple[int, int]] = []

    for start in range(n):
        for end in range(start + min_length, min(start + max_length + 1, n + 1)):
            subseq = steps[start:end]
            bound = _names_bound(subseq)
            used = _names_used(subseq)

            # Inputs: names used but not bound within the subsequence
            inputs = used - bound
            # All inputs must be bound before the subsequence
            pre_bound = _names_bound(steps[:start]) | {"input", "ctx"}
            if not inputs.issubset(pre_bound):
                continue

            # At least one output must be used after the subsequence
            post_used = _names_used(steps[end:])
            if program.output in bound:
                # The subsequence produces the final output
                candidates.append((start, end))
            elif bound & post_used:
                candidates.append((start, end))

    return candidates


def parameterize_candidate(
    program: Program,
    start: int,
    end: int,
) -> tuple[tuple[tuple[str, Type], ...], tuple[Step, ...], str] | None:
    """Attempt to parameterize a candidate subsequence.

    Returns (params, steps, output_name) or None if parameterization fails.
    Params are the external names the subsequence depends on.
    """
    subseq = program.steps[start:end]
    bound = _names_bound(subseq)
    used = _names_used(subseq)
    inputs = used - bound

    # Build param list from the types of the input bindings
    params: list[tuple[str, Type]] = []
    for name in sorted(inputs):
        # Find the type from earlier bindings
        for step in program.steps[:start]:
            if isinstance(step, Bind) and step.name == name:
                params.append((name, step.typ))
                break
        else:
            if name == "input":
                params.append(("input", Type.GRID))
            elif name == "ctx":
                params.append(("ctx", Type.TASK_CTX))
            else:
                return None  # can't determine type

    # Determine output: last bound name, or whichever is used after
    post_used = _names_used(program.steps[end:])
    outputs = bound & (post_used | {program.output})
    if not outputs:
        return None

    # Pick the last-bound output
    output_name = None
    for step in reversed(list(subseq)):
        if isinstance(step, Bind) and step.name in outputs:
            output_name = step.name
            break

    if output_name is None:
        return None

    return tuple(params), tuple(subseq), output_name


def mine_abstractions(
    programs: list[Program],
    min_length: int = 2,
    max_length: int = 6,
) -> list[LibraryEntry]:
    """Extract candidate abstractions from a set of verified programs.

    Returns candidates that appear in at least 1 program (admission gate
    enforces the 2+ threshold separately).
    """
    candidates: list[LibraryEntry] = []
    seen_sigs: set[tuple[tuple[str, Type], ...]] = set()

    for i, prog in enumerate(programs):
        ranges = extract_candidates(prog, min_length, max_length)
        for start, end in ranges:
            result = parameterize_candidate(prog, start, end)
            if result is None:
                continue

            params, steps, output_name = result
            # Determine return type from the output binding
            return_type = Type.GRID  # default
            for s in steps:
                if isinstance(s, Bind) and s.name == output_name:
                    return_type = s.typ
                    break

            # Deduplicate by parameter signature
            sig_key = params
            if sig_key in seen_sigs:
                continue
            seen_sigs.add(sig_key)

            name = f"lib_{i}_{start}_{end}"
            candidates.append(LibraryEntry(
                name=name,
                params=params,
                return_type=return_type,
                steps=steps,
                output=output_name,
                level=2,
                use_count=1,
            ))

    return candidates
