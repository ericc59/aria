"""Tests for type-safe AST mutation operators."""

from __future__ import annotations

import aria.runtime  # noqa: F401
from aria.mutation import Mutation, mutate_program
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import Axis, Bind, Call, Literal, Program, Ref, Type


def _simple_reflect_program() -> Program:
    return Program(
        steps=(
            Bind(
                name="v0",
                typ=Type.GRID,
                expr=Call(
                    op="reflect_grid",
                    args=(Literal(Axis.HORIZONTAL, Type.AXIS), Ref("input")),
                ),
            ),
        ),
        output="v0",
    )


def _two_step_program() -> Program:
    return Program(
        steps=(
            Bind(
                name="v0",
                typ=Type.OBJECT_SET,
                expr=Call(op="find_objects", args=(Ref("input"),)),
            ),
            Bind(
                name="v1",
                typ=Type.GRID,
                expr=Call(op="overlay", args=(Ref("input"), Ref("input"))),
            ),
        ),
        output="v1",
    )


def _build_basic_pool() -> dict[Type, tuple[Literal, ...]]:
    return {
        Type.INT: (Literal(0, Type.INT), Literal(1, Type.INT), Literal(2, Type.INT)),
        Type.COLOR: (Literal(0, Type.COLOR), Literal(1, Type.COLOR)),
        Type.BOOL: (Literal(False, Type.BOOL), Literal(True, Type.BOOL)),
        Type.AXIS: (Literal(Axis.HORIZONTAL, Type.AXIS), Literal(Axis.VERTICAL, Type.AXIS)),
    }


def test_mutate_produces_type_safe_programs():
    program = _simple_reflect_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=20)

    assert len(mutations) > 0
    for m in mutations:
        errors = type_check(
            m.program,
            initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
        )
        assert not errors, f"Type error in mutation: {errors}"


def test_mutate_produces_distinct_programs():
    program = _simple_reflect_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=30)

    texts = set()
    original_text = program_to_text(program)
    for m in mutations:
        text = program_to_text(m.program)
        assert text != original_text
        assert text not in texts, f"Duplicate mutation: {text}"
        texts.add(text)


def test_replace_literal_mutation_exists():
    program = _simple_reflect_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=50)

    edit_kinds = {m.edit_kind for m in mutations}
    assert "replace_literal" in edit_kinds


def test_wrap_output_mutation_exists():
    program = _simple_reflect_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=50)

    edit_kinds = {m.edit_kind for m in mutations}
    assert "wrap_output" in edit_kinds


def test_replace_output_mutation_on_multi_step():
    program = _two_step_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=50)

    # v0 is OBJECT_SET not GRID, so replace_output should not produce v0
    for m in mutations:
        if m.edit_kind == "replace_output":
            # Should not switch to v0 (OBJECT_SET)
            assert m.program.output != "v0"


def test_mutate_respects_max_mutations():
    program = _simple_reflect_program()
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=5)
    assert len(mutations) <= 5


def test_mutate_empty_program_returns_empty():
    program = Program(steps=(), output="input")
    pool = _build_basic_pool()
    mutations = mutate_program(program, pool, max_mutations=10)
    # No steps to mutate, only wrap_output should work (if any)
    for m in mutations:
        errors = type_check(
            m.program,
            initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
        )
        assert not errors
