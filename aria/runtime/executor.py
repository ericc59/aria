"""Program execution engine for ARIA.

Walks steps in order, evaluates expressions, builds up an environment,
and yields the output grid.
"""

from __future__ import annotations

from typing import Any, Callable

from aria.types import (
    Assert,
    Bind,
    Call,
    Expr,
    Grid,
    Lambda,
    Literal,
    Program,
    Ref,
    TaskContext,
)
from aria.runtime.ops import get_op


class ExecutionError(Exception):
    """Raised when program execution fails."""


def eval_expr(expr: Expr, env: dict[str, Any]) -> Any:
    """Recursively evaluate an expression in the given environment.

    Parameters
    ----------
    expr : Expr
        The AST node to evaluate.
    env : dict[str, Any]
        The current variable bindings.

    Returns
    -------
    Any
        The computed value.
    """
    if isinstance(expr, Ref):
        if expr.name not in env:
            raise ExecutionError(f"Unbound variable: '{expr.name}'")
        return env[expr.name]

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, Lambda):
        return _make_closure(expr, env)

    if isinstance(expr, Call):
        return _eval_call(expr, env)

    raise ExecutionError(f"Unknown expression type: {type(expr).__name__}")


def _make_closure(lam: Lambda, env: dict[str, Any]) -> Callable[..., Any]:
    """Create a closure that captures the current environment."""
    captured = dict(env)

    def closure(*args: Any) -> Any:
        if not args:
            return closure

        inner_env = {**captured, lam.param: args[0]}
        result = eval_expr(lam.body, inner_env)

        if len(args) == 1:
            return result
        return _apply_curried_callable(result, args[1:])

    setattr(closure, "_aria_curried", True)

    return closure


def _apply_curried_callable(fn: Any, args: list[Any] | tuple[Any, ...]) -> Any:
    """Apply curried callables one argument at a time."""
    result = fn
    for arg in args:
        if not callable(result):
            raise ExecutionError(
                "Callable returned a non-callable before consuming all arguments"
            )
        result = result(arg)
    return result


def _eval_call(call_expr: Call, env: dict[str, Any]) -> Any:
    """Evaluate an op invocation.

    Resolution order:
    1. If the name is a bound variable holding a callable, call it directly.
       This handles patterns like: bind step = infer_step(ctx) / bind result = step(input)
    2. Otherwise look up in the op registry.

    Supports partial application: if fewer args are provided than the op
    expects, return a closure that captures the provided args and waits
    for the rest.
    """
    args = [eval_expr(a, env) for a in call_expr.args]

    # Check if the name is a bound callable variable first
    if call_expr.op in env and callable(env[call_expr.op]):
        fn = env[call_expr.op]
        try:
            if getattr(fn, "_aria_curried", False):
                return _apply_curried_callable(fn, args)
            return fn(*args)
        except Exception as exc:
            raise ExecutionError(
                f"Calling '{call_expr.op}' failed: {exc}"
            ) from exc

    # Fall back to op registry
    from aria.runtime.ops import has_op
    if not has_op(call_expr.op):
        raise ExecutionError(f"Unknown operation: '{call_expr.op}'")

    sig, impl = get_op(call_expr.op)
    n_params = len(sig.params)
    n_args = len(args)

    if n_args < n_params:
        def curried(*remaining: Any) -> Any:
            all_args = args + list(remaining)
            return impl(*all_args)
        return curried

    try:
        return impl(*args)
    except Exception as exc:
        raise ExecutionError(
            f"Op '{call_expr.op}' failed: {exc}"
        ) from exc


def execute(
    program: Program,
    input_grid: Grid,
    ctx: TaskContext | None = None,
) -> Grid:
    """Execute a program against an input grid, returning the output grid.

    Parameters
    ----------
    program : Program
        The program AST to execute.
    input_grid : Grid
        The input grid bound as ``input`` in the environment.
    ctx : TaskContext or None
        Optional task context bound as ``ctx``.

    Returns
    -------
    Grid
        The output grid (the value of the program's output binding).
    """
    env: dict[str, Any] = {"input": input_grid}
    if ctx is not None:
        env["ctx"] = ctx

    for idx, step in enumerate(program.steps):
        if isinstance(step, Bind):
            try:
                value = eval_expr(step.expr, env)
            except ExecutionError:
                raise
            except Exception as exc:
                raise ExecutionError(
                    f"Step {idx} ({step.name}): evaluation failed: {exc}"
                ) from exc
            env[step.name] = value

        elif isinstance(step, Assert):
            try:
                result = eval_expr(step.pred, env)
            except ExecutionError:
                raise
            except Exception as exc:
                raise ExecutionError(
                    f"Step {idx} (assert): evaluation failed: {exc}"
                ) from exc
            if not result:
                raise ExecutionError(f"Step {idx}: assertion failed")

        else:
            raise ExecutionError(
                f"Step {idx}: unknown step kind {type(step).__name__}"
            )

    if program.output not in env:
        raise ExecutionError(f"Output binding '{program.output}' not found")

    return env[program.output]
