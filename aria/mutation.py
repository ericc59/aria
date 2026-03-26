"""Type-safe AST mutation operators for program refinement.

Generates nearby candidate programs via generic edits: op replacement,
literal swap, ref swap, output swap, and output wrapping. Every
mutation is type-checked before being yielded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aria.runtime.ops import OpSignature, all_ops, get_op, has_op
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import Bind, Call, Expr, Literal, Program, Ref, Type


_CALLABLE_TYPES = frozenset({
    Type.PREDICATE, Type.OBJ_TRANSFORM, Type.GRID_TRANSFORM, Type.CALLABLE,
})


@dataclass(frozen=True)
class Mutation:
    program: Program
    edit_kind: str
    detail: str


def mutate_program(
    program: Program,
    literal_pool: dict[Type, tuple[Literal, ...]],
    *,
    max_mutations: int = 50,
    rng: random.Random | None = None,
) -> list[Mutation]:
    """Generate type-safe mutations of a program."""
    rng = rng or random.Random(42)
    seen: set[str] = {program_to_text(program)}
    mutations: list[Mutation] = []

    all_candidates: list[Mutation] = []
    all_candidates.extend(_replace_op_mutations(program))
    all_candidates.extend(_replace_literal_mutations(program, literal_pool))
    all_candidates.extend(_replace_ref_mutations(program))
    all_candidates.extend(_replace_output_mutations(program))
    all_candidates.extend(_wrap_output_mutations(program))

    rng.shuffle(all_candidates)

    for m in all_candidates:
        if len(mutations) >= max_mutations:
            break
        text = program_to_text(m.program)
        if text in seen:
            continue
        seen.add(text)
        errors = type_check(
            m.program, initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
        )
        if not errors:
            mutations.append(m)

    return mutations


def _replace_op_mutations(program: Program) -> list[Mutation]:
    """Replace an op call with a type-compatible alternative."""
    ops = all_ops()
    results: list[Mutation] = []

    for step_idx, step in enumerate(program.steps):
        if not isinstance(step, Bind) or not isinstance(step.expr, Call):
            continue
        call = step.expr
        if not has_op(call.op):
            continue
        orig_sig, _ = get_op(call.op)

        for alt_name, alt_sig in ops.items():
            if alt_name == call.op:
                continue
            if alt_sig.return_type != orig_sig.return_type:
                continue
            if len(alt_sig.params) != len(orig_sig.params):
                continue
            if not all(
                _types_assignable(alt_p[1], orig_p[1])
                for alt_p, orig_p in zip(alt_sig.params, orig_sig.params)
            ):
                continue

            new_expr = Call(op=alt_name, args=call.args)
            new_step = Bind(name=step.name, typ=step.typ, expr=new_expr)
            new_steps = (
                program.steps[:step_idx] + (new_step,) + program.steps[step_idx + 1:]
            )
            results.append(Mutation(
                program=Program(steps=new_steps, output=program.output),
                edit_kind="replace_op",
                detail=f"{call.op} -> {alt_name} at step {step_idx}",
            ))

    return results


def _replace_literal_mutations(
    program: Program,
    literal_pool: dict[Type, tuple[Literal, ...]],
) -> list[Mutation]:
    """Replace a literal argument with another plausible literal."""
    results: list[Mutation] = []

    for step_idx, step in enumerate(program.steps):
        if not isinstance(step, Bind) or not isinstance(step.expr, Call):
            continue
        call = step.expr

        for arg_idx, arg in enumerate(call.args):
            if not isinstance(arg, Literal):
                continue
            pool = literal_pool.get(arg.typ, ())
            for alt in pool:
                if alt.value == arg.value:
                    continue
                new_args = (
                    call.args[:arg_idx] + (alt,) + call.args[arg_idx + 1:]
                )
                new_expr = Call(op=call.op, args=new_args)
                new_step = Bind(name=step.name, typ=step.typ, expr=new_expr)
                new_steps = (
                    program.steps[:step_idx]
                    + (new_step,)
                    + program.steps[step_idx + 1:]
                )
                results.append(Mutation(
                    program=Program(steps=new_steps, output=program.output),
                    edit_kind="replace_literal",
                    detail=f"{arg.value} -> {alt.value} at step {step_idx} arg {arg_idx}",
                ))

    return results


def _replace_ref_mutations(program: Program) -> list[Mutation]:
    """Replace a ref argument with another type-compatible ref in scope."""
    results: list[Mutation] = []
    env: dict[str, Type] = {"input": Type.GRID, "ctx": Type.TASK_CTX}

    for step_idx, step in enumerate(program.steps):
        if isinstance(step, Bind):
            if isinstance(step.expr, Call):
                call = step.expr
                for arg_idx, arg in enumerate(call.args):
                    if not isinstance(arg, Ref):
                        continue
                    arg_type = env.get(arg.name)
                    if arg_type is None:
                        continue
                    for name, typ in env.items():
                        if name == arg.name:
                            continue
                        if not _types_assignable(arg_type, typ):
                            continue
                        new_args = (
                            call.args[:arg_idx]
                            + (Ref(name),)
                            + call.args[arg_idx + 1:]
                        )
                        new_expr = Call(op=call.op, args=new_args)
                        new_step = Bind(
                            name=step.name, typ=step.typ, expr=new_expr,
                        )
                        new_steps = (
                            program.steps[:step_idx]
                            + (new_step,)
                            + program.steps[step_idx + 1:]
                        )
                        results.append(Mutation(
                            program=Program(
                                steps=new_steps, output=program.output,
                            ),
                            edit_kind="replace_ref",
                            detail=f"{arg.name} -> {name} at step {step_idx} arg {arg_idx}",
                        ))
            env[step.name] = step.typ

    return results


def _replace_output_mutations(program: Program) -> list[Mutation]:
    """Try yielding a different GRID binding as output."""
    results: list[Mutation] = []

    for step in program.steps:
        if not isinstance(step, Bind):
            continue
        if step.typ != Type.GRID:
            continue
        if step.name == program.output:
            continue
        results.append(Mutation(
            program=Program(steps=program.steps, output=step.name),
            edit_kind="replace_output",
            detail=f"{program.output} -> {step.name}",
        ))

    return results


def _wrap_output_mutations(program: Program, *, cap: int = 100) -> list[Mutation]:
    """Append a step that wraps the current output with a compatible op."""
    results: list[Mutation] = []
    ops = all_ops()

    output_type = Type.GRID
    for step in program.steps:
        if isinstance(step, Bind) and step.name == program.output:
            output_type = step.typ
            break

    if output_type != Type.GRID:
        return results

    env_types: dict[str, Type] = {"input": Type.GRID, "ctx": Type.TASK_CTX}
    for step in program.steps:
        if isinstance(step, Bind):
            env_types[step.name] = step.typ

    next_name = f"v{len(program.steps)}"
    output_ref = Ref(program.output)

    for op_name, sig in ops.items():
        if sig.return_type != Type.GRID:
            continue
        if len(results) >= cap:
            break

        if len(sig.params) == 1:
            _, param_type = sig.params[0]
            if _types_assignable(param_type, output_type):
                new_step = Bind(
                    name=next_name,
                    typ=Type.GRID,
                    expr=Call(op=op_name, args=(output_ref,)),
                )
                results.append(Mutation(
                    program=Program(
                        steps=program.steps + (new_step,),
                        output=next_name,
                    ),
                    edit_kind="wrap_output",
                    detail=f"wrap with {op_name}(output)",
                ))

        elif len(sig.params) == 2:
            for slot in (0, 1):
                _, p_type = sig.params[slot]
                _, other_type = sig.params[1 - slot]
                if not _types_assignable(p_type, output_type):
                    continue
                for env_name, env_type in env_types.items():
                    if not _types_assignable(other_type, env_type):
                        continue
                    if len(results) >= cap:
                        break
                    if slot == 0:
                        args: tuple[Expr, ...] = (output_ref, Ref(env_name))
                    else:
                        args = (Ref(env_name), output_ref)
                    new_step = Bind(
                        name=next_name,
                        typ=Type.GRID,
                        expr=Call(op=op_name, args=args),
                    )
                    results.append(Mutation(
                        program=Program(
                            steps=program.steps + (new_step,),
                            output=next_name,
                        ),
                        edit_kind="wrap_output",
                        detail=f"wrap with {op_name}",
                    ))

    return results


def _types_assignable(expected: Type, actual: Type) -> bool:
    if expected == actual:
        return True
    if {expected, actual} == {Type.INT, Type.COLOR}:
        return True
    if expected == Type.CALLABLE and actual in _CALLABLE_TYPES:
        return True
    if actual == Type.CALLABLE and expected in _CALLABLE_TYPES:
        return True
    return False
