"""Program construction helpers and serialization.

Convenience builders for constructing Program/Step/Expr AST nodes,
plus a text serializer for human-readable program output.
"""

from __future__ import annotations

from typing import Any

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
    Type,
)


# ---------------------------------------------------------------------------
# AST construction helpers
# ---------------------------------------------------------------------------


def ref(name: str) -> Ref:
    """Create a variable reference."""
    return Ref(name=name)


def lit(value: Any, typ: Type) -> Literal:
    """Create a typed literal."""
    return Literal(value=value, typ=typ)


def call(op: str, *args: Expr) -> Call:
    """Create an op invocation."""
    return Call(op=op, args=tuple(args))


def lam(param: str, param_type: Type, body: Expr) -> Lambda:
    """Create a single-parameter lambda node."""
    return Lambda(param=param, param_type=param_type, body=body)


def bind(name: str, typ: Type, expr: Expr, declared: bool = True) -> Bind:
    """Create a let-binding step."""
    return Bind(name=name, typ=typ, expr=expr, declared=declared)


def assert_step(pred: Expr) -> Assert:
    """Create an assertion step."""
    return Assert(pred=pred)


def make_program(steps: list[Step], output: str) -> Program:
    """Build a Program from a list of steps and an output binding name."""
    return Program(steps=tuple(steps), output=output)


# ---------------------------------------------------------------------------
# Text serialization
# ---------------------------------------------------------------------------


def _expr_to_text(expr: Expr) -> str:
    """Serialize an expression to readable text."""
    if isinstance(expr, Ref):
        return expr.name
    if isinstance(expr, Literal):
        return _literal_to_text(expr)
    if isinstance(expr, Call):
        args_text = ", ".join(_expr_to_text(a) for a in expr.args)
        return f"{expr.op}({args_text})"
    if isinstance(expr, Lambda):
        body_text = _expr_to_text(expr.body)
        return f"|{expr.param}: {expr.param_type.name}| {body_text}"
    raise TypeError(f"Unknown expr type: {type(expr)}")


def _literal_to_text(lit_node: Literal) -> str:
    """Serialize a literal value."""
    val = lit_node.value
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if hasattr(val, "name"):
        # Enum variant
        return val.name
    return repr(val)


def _step_to_text(step: Step) -> str:
    """Serialize a single step."""
    if isinstance(step, Bind):
        if not step.declared:
            return f"let {step.name} = {_expr_to_text(step.expr)}"
        return f"let {step.name}: {step.typ.name} = {_expr_to_text(step.expr)}"
    if isinstance(step, Assert):
        return f"assert {_expr_to_text(step.pred)}"
    raise TypeError(f"Unknown step type: {type(step)}")


def program_to_text(program: Program) -> str:
    """Serialize a Program to a human-readable string."""
    lines: list[str] = []
    for step in program.steps:
        lines.append(_step_to_text(step))
    lines.append(f"-> {program.output}")
    return "\n".join(lines)
