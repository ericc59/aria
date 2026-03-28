"""ARIA step-machine runtime.

Importing this package eagerly loads all op modules so they self-register
with the op registry. This guarantees the registry is populated before
any type-checking or execution occurs.
"""

# Eagerly import all op modules to trigger registration.
import aria.runtime.ops.selection  # noqa: F401
import aria.runtime.ops.spatial  # noqa: F401
import aria.runtime.ops.grid  # noqa: F401
import aria.runtime.ops.dims  # noqa: F401
import aria.runtime.ops.analysis  # noqa: F401
import aria.runtime.ops.composition  # noqa: F401
import aria.runtime.ops.topo  # noqa: F401
import aria.runtime.ops.context  # noqa: F401
import aria.runtime.ops.arithmetic  # noqa: F401
import aria.runtime.ops.cell  # noqa: F401
import aria.runtime.ops.objects  # noqa: F401

from aria.runtime.executor import execute, eval_expr, ExecutionError
from aria.runtime.type_system import type_check
from aria.runtime.program import (
    make_program,
    program_to_text,
    ref,
    lit,
    call,
    bind,
    lam,
    assert_step,
)

__all__ = [
    "execute",
    "eval_expr",
    "ExecutionError",
    "type_check",
    "make_program",
    "program_to_text",
    "ref",
    "lit",
    "call",
    "bind",
    "lam",
    "assert_step",
]
