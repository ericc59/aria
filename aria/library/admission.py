"""Admission gate for library entries.

All four criteria must hold:
1. Sub-sequence appeared in 2+ independently verified programs (1 at test time)
2. MDL of total corpus improves
3. Abstraction type-checks as standalone operation
4. Not equivalent to an existing library entry
"""

from __future__ import annotations

from aria.library.mdl import mdl_improvement
from aria.library.store import Library
from aria.runtime.type_system import type_check
from aria.types import LibraryEntry, Program


def check_admission(
    candidate: LibraryEntry,
    programs: list[Program],
    library: Library,
    min_uses: int = 2,
) -> tuple[bool, str]:
    """Check whether a candidate should be admitted to the library.

    Returns (admitted, reason).
    """
    # Criterion 1: minimum use count
    if candidate.use_count < min_uses:
        return False, f"use_count {candidate.use_count} < {min_uses}"

    # Criterion 2: MDL improvement
    improvement = mdl_improvement(programs, candidate)
    if improvement <= 0:
        return False, f"MDL improvement {improvement} <= 0"

    # Criterion 3: type-checks as standalone
    if not candidate.steps:
        return False, "empty steps"
    if not candidate.output:
        return False, "no output name"
    type_errors = type_check(
        Program(steps=candidate.steps, output=candidate.output),
        initial_env={name: typ for name, typ in candidate.params},
    )
    if type_errors:
        return False, f"type check failed: {type_errors[0]}"

    # Criterion 4: not equivalent to existing entry
    for existing in library.all_entries():
        if _structurally_equivalent(candidate, existing):
            return False, f"equivalent to existing entry '{existing.name}'"

    return True, "admitted"


def _structurally_equivalent(a: LibraryEntry, b: LibraryEntry) -> bool:
    """Check if two library entries are structurally equivalent.

    Conservative check: same params types, same number of steps,
    same operation sequence.
    """
    if len(a.params) != len(b.params):
        return False
    if any(at != bt for (_, at), (_, bt) in zip(a.params, b.params)):
        return False
    if a.return_type != b.return_type:
        return False
    if len(a.steps) != len(b.steps):
        return False

    from aria.types import Bind, Call

    for sa, sb in zip(a.steps, b.steps):
        if type(sa) != type(sb):
            return False
        if isinstance(sa, Bind) and isinstance(sb, Bind):
            if isinstance(sa.expr, Call) and isinstance(sb.expr, Call):
                if sa.expr.op != sb.expr.op:
                    return False
    return True
