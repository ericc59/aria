"""Static type checker for ARIA programs.

Walks the AST, resolves op signatures from the registry, and verifies
that argument types match the declared parameter types. Builds up a
type environment step by step. Returns a list of errors (empty = passes).
"""

from __future__ import annotations

from aria.types import (
    Assert,
    Bind,
    Call,
    Expr,
    ForEach,
    Lambda,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.runtime.ops import get_op, has_op


# Types that are compatible with CALLABLE in higher-order positions.
_CALLABLE_TYPES = frozenset({
    Type.PREDICATE,
    Type.OBJ_TRANSFORM,
    Type.GRID_TRANSFORM,
    Type.CALLABLE,
})


def _types_compatible(expected: Type, actual: Type) -> bool:
    """Check whether `actual` can satisfy `expected`."""
    if expected == actual:
        return True
    # Any callable sub-type satisfies CALLABLE.
    if expected == Type.CALLABLE and actual in _CALLABLE_TYPES:
        return True
    if actual == Type.CALLABLE and expected in _CALLABLE_TYPES:
        return True
    # INT and COLOR are interchangeable in the step language.
    if {expected, actual} == {Type.INT, Type.COLOR}:
        return True
    return False


def _callable_type_from_signature(
    remaining_params: tuple[tuple[str, Type], ...],
    return_type: Type,
) -> Type:
    """Infer the type produced by partial application."""
    if len(remaining_params) == 1:
        _, param_type = remaining_params[0]
        if param_type == Type.OBJECT and return_type == Type.BOOL:
            return Type.PREDICATE
        if param_type == Type.OBJECT and return_type == Type.OBJECT:
            return Type.OBJ_TRANSFORM
        if param_type == Type.GRID and return_type == Type.GRID:
            return Type.GRID_TRANSFORM
    return Type.CALLABLE


def _infer_bound_callable_call(
    op_name: str,
    args: tuple[Expr, ...],
    env: dict[str, Type],
    errors: list[str],
    location: str,
) -> Type | None:
    """Infer calling a callable bound in the environment."""
    fn_type = env[op_name]

    if fn_type == Type.PREDICATE:
        if len(args) != 1:
            errors.append(f"{location}: callable '{op_name}' expects 1 arg, got {len(args)}")
            return Type.BOOL
        arg_type = _infer_expr(args[0], env, errors, location)
        if arg_type is not None and not _types_compatible(Type.OBJECT, arg_type):
            errors.append(
                f"{location}: callable '{op_name}' expects OBJECT, got {arg_type.name}"
            )
        return Type.BOOL

    if fn_type == Type.OBJ_TRANSFORM:
        if len(args) != 1:
            errors.append(f"{location}: callable '{op_name}' expects 1 arg, got {len(args)}")
            return Type.OBJECT
        arg_type = _infer_expr(args[0], env, errors, location)
        if arg_type is not None and not _types_compatible(Type.OBJECT, arg_type):
            errors.append(
                f"{location}: callable '{op_name}' expects OBJECT, got {arg_type.name}"
            )
        return Type.OBJECT

    if fn_type == Type.GRID_TRANSFORM:
        if len(args) != 1:
            errors.append(f"{location}: callable '{op_name}' expects 1 arg, got {len(args)}")
            return Type.GRID
        arg_type = _infer_expr(args[0], env, errors, location)
        if arg_type is not None and not _types_compatible(Type.GRID, arg_type):
            errors.append(
                f"{location}: callable '{op_name}' expects GRID, got {arg_type.name}"
            )
        return Type.GRID

    for arg_expr in args:
        _infer_expr(arg_expr, env, errors, location)
    return Type.CALLABLE


def _infer_callable_result_type(
    expr: Expr,
    env: dict[str, Type],
    errors: list[str],
    location: str,
) -> Type | None:
    """Best-effort inference of the result type produced by a callable."""
    if isinstance(expr, Lambda):
        inner_env = {**env, expr.param: expr.param_type}
        return _infer_expr(expr.body, inner_env, errors, location)

    if isinstance(expr, Call) and has_op(expr.op):
        sig, _ = get_op(expr.op)
        if len(expr.args) < len(sig.params):
            return sig.return_type

    if isinstance(expr, Ref) and expr.name in env:
        fn_type = env[expr.name]
        if fn_type == Type.PREDICATE:
            return Type.BOOL
        if fn_type == Type.OBJ_TRANSFORM:
            return Type.OBJECT
        if fn_type == Type.GRID_TRANSFORM:
            return Type.GRID

    return None


def _infer_expr(
    expr: Expr,
    env: dict[str, Type],
    errors: list[str],
    location: str,
) -> Type | None:
    """Infer the type of an expression, appending errors as found."""
    if isinstance(expr, Ref):
        if expr.name not in env:
            errors.append(f"{location}: unbound name '{expr.name}'")
            return None
        return env[expr.name]

    if isinstance(expr, Literal):
        return expr.typ

    if isinstance(expr, Lambda):
        # Build inner env with the lambda parameter.
        inner_env = {**env, expr.param: expr.param_type}
        body_type = _infer_expr(expr.body, inner_env, errors, location)
        # Determine the function type from body.
        if body_type == Type.BOOL:
            return Type.PREDICATE
        if body_type == Type.OBJECT:
            return Type.OBJ_TRANSFORM
        if body_type == Type.GRID:
            return Type.GRID_TRANSFORM
        return Type.CALLABLE

    if isinstance(expr, Call):
        if expr.op in env and env[expr.op] in _CALLABLE_TYPES:
            return _infer_bound_callable_call(expr.op, expr.args, env, errors, location)

        if not has_op(expr.op):
            errors.append(f"{location}: unknown op '{expr.op}'")
            return None

        sig, _ = get_op(expr.op)
        param_types = sig.params

        if len(expr.args) > len(param_types):
            errors.append(
                f"{location}: op '{expr.op}' expects {len(param_types)} args, "
                f"got {len(expr.args)}"
            )
            return sig.return_type

        for i, (arg_expr, (param_name, param_type)) in enumerate(
            zip(expr.args, param_types)
        ):
            arg_type = _infer_expr(arg_expr, env, errors, location)
            if arg_type is not None and not _types_compatible(param_type, arg_type):
                errors.append(
                    f"{location}: arg '{param_name}' (#{i}) of '{expr.op}' "
                    f"expects {param_type.name}, got {arg_type.name}"
                )

        if len(expr.args) < len(param_types):
            return _callable_type_from_signature(param_types[len(expr.args):], sig.return_type)

        if expr.op == "map_list" and len(expr.args) == 2:
            item_type = _infer_callable_result_type(expr.args[0], env, errors, location)
            if item_type in (Type.INT, Type.COLOR):
                return Type.INT_LIST
            if item_type == Type.OBJECT:
                return Type.OBJECT_LIST

        return sig.return_type

    errors.append(f"{location}: unknown expression kind {type(expr).__name__}")
    return None


def type_check(
    program: Program,
    initial_env: dict[str, Type] | None = None,
) -> list[str]:
    """Type-check a program, returning a list of error messages.

    An empty list means the program passes type checking.

    Parameters
    ----------
    program : Program
        The AST to check.
    initial_env : dict[str, Type] or None
        Pre-bound names (e.g. ``{"input": Type.GRID, "ctx": Type.TASK_CTX}``).
    """
    env: dict[str, Type] = dict(initial_env) if initial_env else {}
    errors: list[str] = []

    for idx, step in enumerate(program.steps):
        loc = f"step {idx}"

        if isinstance(step, Bind):
            loc = f"step {idx} ({step.name})"
            inferred = _infer_expr(step.expr, env, errors, loc)
            if step.declared and inferred is not None and not _types_compatible(step.typ, inferred):
                errors.append(
                    f"{loc}: declared type {step.typ.name} "
                    f"but expression yields {inferred.name}"
                )
            if step.name in env:
                errors.append(f"{loc}: shadowing existing binding '{step.name}'")
            env[step.name] = inferred if inferred is not None else step.typ

        elif isinstance(step, Assert):
            loc = f"step {idx} (assert)"
            inferred = _infer_expr(step.pred, env, errors, loc)
            if inferred is not None and inferred != Type.BOOL:
                errors.append(f"{loc}: assert expects BOOL, got {inferred.name}")

        elif isinstance(step, ForEach):
            loc = f"step {idx} (for_each)"
            # Check source type
            src_type = _infer_expr(step.source, env, errors, loc)
            if src_type is not None and src_type not in (Type.OBJECT_SET, Type.OBJECT_LIST):
                errors.append(f"{loc}: source must be OBJECT_SET/OBJECT_LIST, got {src_type.name}")
            # Check accumulator exists and is GRID
            if step.accumulator not in env:
                errors.append(f"{loc}: accumulator '{step.accumulator}' not bound")
            elif env[step.accumulator] != Type.GRID:
                errors.append(f"{loc}: accumulator must be GRID")
            # Type-check body in inner env
            inner_env = dict(env)
            inner_env[step.iter_name] = Type.OBJECT
            for bi, body_step in enumerate(step.body):
                bloc = f"{loc} body[{bi}]"
                if isinstance(body_step, Bind):
                    inferred = _infer_expr(body_step.expr, inner_env, errors, bloc)
                    inner_env[body_step.name] = inferred if inferred is not None else body_step.typ
                elif isinstance(body_step, Assert):
                    inferred = _infer_expr(body_step.pred, inner_env, errors, bloc)
                    if inferred is not None and inferred != Type.BOOL:
                        errors.append(f"{bloc}: assert expects BOOL, got {inferred.name}")
            # Output binding
            env[step.output_name] = Type.GRID

        else:
            errors.append(f"step {idx}: unknown step kind {type(step).__name__}")

    # Verify output binding exists.
    if program.output not in env:
        errors.append(f"output: unbound name '{program.output}'")

    return errors
