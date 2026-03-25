"""Verification mode detection via static analysis of program AST.

Determines whether a program reads cross-demo context, and if so,
whether it uses sequential operations (Mode C) or general context (Mode B).
"""

from __future__ import annotations

from aria.types import (
    Assert,
    Bind,
    Call,
    Expr,
    Lambda,
    Literal,
    Program,
    Ref,
    Step,
    VerifyMode,
)

# Operations that read TaskContext
_CTX_OPS = frozenset({
    "demo_count",
    "demo_at",
    "infer_map",
    "disambiguate",
    "predict_dims",
})

# Subset of context ops that imply sequential/progression reasoning
_SEQUENTIAL_OPS = frozenset({
    "infer_step",
    "infer_iteration",
})


def _collect_ops(expr: Expr) -> set[str]:
    """Recursively collect all operation names used in an expression."""
    match expr:
        case Ref():
            return set()
        case Literal():
            return set()
        case Call(op=op, args=args):
            result = {op}
            for arg in args:
                result |= _collect_ops(arg)
            return result
        case Lambda(body=body):
            return _collect_ops(body)
        case _:
            return set()


def _collect_all_ops(program: Program) -> set[str]:
    """Collect all operation names used anywhere in a program."""
    ops: set[str] = set()
    for step in program.steps:
        match step:
            case Bind(expr=expr):
                ops |= _collect_ops(expr)
            case Assert(pred=pred):
                ops |= _collect_ops(pred)
    return ops


def detect_mode(program: Program) -> VerifyMode:
    """Determine the verification mode for a program.

    - Mode A (STATELESS): no context-reading operations
    - Mode B (LEAVE_ONE_OUT): reads context, but not sequential ops
    - Mode C (SEQUENTIAL): uses sequential context ops (infer_step, infer_iteration)
    """
    ops = _collect_all_ops(program)

    has_sequential = bool(ops & _SEQUENTIAL_OPS)
    has_ctx = bool(ops & (_CTX_OPS | _SEQUENTIAL_OPS))

    if has_sequential:
        return VerifyMode.SEQUENTIAL
    if has_ctx:
        return VerifyMode.LEAVE_ONE_OUT
    return VerifyMode.STATELESS
