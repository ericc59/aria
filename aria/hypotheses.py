"""Hypothesis generation from retrieved abstractions.

Composes library entries into concrete typed program skeletons that can
be verified before falling into enumeration.  Every generator is derived
from the type system — no op names are hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

from aria.library.store import Library
from aria.offline_search import build_literal_pool
from aria.retrieval import AbstractionHint
from aria.runtime.ops import OpSignature, all_ops, get_op, has_op
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Expr,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.verify.verifier import verify


@dataclass(frozen=True)
class SkeletonResult:
    """Outcome of testing skeleton hypotheses before search."""

    solved: bool
    winning_program: Program | None
    skeletons_tested: int
    skeletons_generated: int
    hypotheses: tuple[SkeletonHypothesis, ...]


@dataclass(frozen=True)
class SkeletonHypothesis:
    """A single hypothesis: a program skeleton and whether it verified."""

    program_text: str
    passed: bool
    source: str  # e.g. "single:lib_X", "compose:lib_A+lib_B", "repair:..."
    error_type: str | None = None


# Available ref bindings for skeleton instantiation
_AVAILABLE_REFS: dict[str, Type] = {"input": Type.GRID, "ctx": Type.TASK_CTX}


def check_skeleton_hypotheses(
    demos: tuple[DemoPair, ...],
    hints: tuple[AbstractionHint, ...],
    library: Library,
    *,
    max_skeletons: int = 64,
    max_repairs: int = 32,
) -> SkeletonResult:
    """Generate and verify skeleton programs from retrieved abstractions.

    Phase 1: verify skeletons composed from hints.
    Phase 2: if near-misses exist (right dims, wrong pixels), try
             type-derived wraps and diff-guided literal repairs.
    Stops on the first exact-verification pass.
    """
    if not hints:
        return SkeletonResult(
            solved=False, winning_program=None,
            skeletons_tested=0, skeletons_generated=0, hypotheses=(),
        )

    literal_pool = build_literal_pool(demos)
    skeletons = list(islice(
        _generate_skeletons(hints, literal_pool),
        max_skeletons,
    ))

    tested = 0
    hypotheses: list[SkeletonHypothesis] = []
    near_misses: list[tuple[Program, str, dict]] = []

    for program, source in skeletons:
        if type_check(program, initial_env=dict(_AVAILABLE_REFS)):
            continue
        tested += 1
        result = verify(program, demos)
        text = program_to_text(program)

        hypotheses.append(SkeletonHypothesis(
            program_text=text, passed=result.passed,
            source=source, error_type=result.error_type,
        ))
        if result.passed:
            return SkeletonResult(
                solved=True, winning_program=program,
                skeletons_tested=tested,
                skeletons_generated=len(skeletons),
                hypotheses=tuple(hypotheses),
            )
        if (
            result.error_type == "wrong_output"
            and result.diff is not None
            and result.diff.get("pixel_diff_count") is not None
        ):
            near_misses.append((program, source, result.diff))

    # Phase 2: targeted repair of near-misses
    if near_misses:
        repairs = list(islice(
            _generate_repairs(near_misses, literal_pool),
            max_repairs,
        ))
        seen = {program_to_text(p) for p, _ in skeletons}
        for repair_program, repair_source in repairs:
            text = program_to_text(repair_program)
            if text in seen:
                continue
            seen.add(text)
            if type_check(repair_program, initial_env=dict(_AVAILABLE_REFS)):
                continue
            tested += 1
            result = verify(repair_program, demos)
            hypotheses.append(SkeletonHypothesis(
                program_text=text, passed=result.passed,
                source=repair_source, error_type=result.error_type,
            ))
            if result.passed:
                return SkeletonResult(
                    solved=True, winning_program=repair_program,
                    skeletons_tested=tested,
                    skeletons_generated=len(skeletons) + len(repairs),
                    hypotheses=tuple(hypotheses),
                )

    return SkeletonResult(
        solved=False, winning_program=None,
        skeletons_tested=tested,
        skeletons_generated=len(skeletons) + (len(near_misses) if near_misses else 0),
        hypotheses=tuple(hypotheses),
    )


# ---------------------------------------------------------------------------
# Skeleton generation — all type-derived, no hardcoded op names
# ---------------------------------------------------------------------------


def _generate_skeletons(
    hints: tuple[AbstractionHint, ...],
    literal_pool: dict[Type, tuple[Literal, ...]],
):
    """Yield (Program, source_tag) pairs in priority order.

    For each hint, generates all type-valid ways to call the op with
    the available refs (input, ctx) and plausible literals.
    Then generates pairwise compositions of GRID→GRID hints.
    """
    grid_g2g_hints: list[AbstractionHint] = []

    for hint in hints:
        if not has_op(hint.name):
            continue
        sig, _ = get_op(hint.name)
        if sig.return_type != Type.GRID:
            continue

        # Generate all typed instantiations of this op
        for program, tag in _instantiate_op(hint.name, sig, literal_pool):
            yield program, tag

        # Track GRID→GRID hints for composition
        if len(sig.params) == 1 and sig.params[0][1] == Type.GRID:
            grid_g2g_hints.append(hint)

    # Composition tier 1: sequential chains — lib_A(input) → lib_B(v0)
    for i, hint_a in enumerate(grid_g2g_hints):
        for j, hint_b in enumerate(grid_g2g_hints):
            if i == j:
                continue
            yield (
                _make_program([
                    Bind("v0", Type.GRID, Call(hint_a.name, (Ref("input"),))),
                    Bind("v1", Type.GRID, Call(hint_b.name, (Ref("v0"),))),
                ], "v1"),
                f"compose:{hint_a.name}+{hint_b.name}",
            )

    # Composition tier 2: parallel merge — lib_A(input), lib_B(input), merge(v0, v1)
    # Uses any registered (GRID, GRID) → GRID op as the merge step.
    if len(grid_g2g_hints) >= 2:
        merge_ops = [
            (name, sig) for name, sig in all_ops().items()
            if sig.return_type == Type.GRID
            and len(sig.params) == 2
            and sig.params[0][1] == Type.GRID
            and sig.params[1][1] == Type.GRID
        ]
        for i, hint_a in enumerate(grid_g2g_hints[:4]):
            for j, hint_b in enumerate(grid_g2g_hints[:4]):
                if i >= j:
                    continue
                for merge_name, _ in merge_ops[:6]:
                    yield (
                        _make_program([
                            Bind("v0", Type.GRID, Call(hint_a.name, (Ref("input"),))),
                            Bind("v1", Type.GRID, Call(hint_b.name, (Ref("input"),))),
                            Bind("v2", Type.GRID, Call(merge_name, (Ref("v0"), Ref("v1")))),
                        ], "v2"),
                        f"parallel:{hint_a.name}+{hint_b.name}>{merge_name}",
                    )

    # Composition tier 3: 3-step chains — lib_A → lib_B → lib_C
    for i, hint_a in enumerate(grid_g2g_hints[:3]):
        for j, hint_b in enumerate(grid_g2g_hints[:3]):
            if i == j:
                continue
            for k, hint_c in enumerate(grid_g2g_hints[:3]):
                if k == i or k == j:
                    continue
                yield (
                    _make_program([
                        Bind("v0", Type.GRID, Call(hint_a.name, (Ref("input"),))),
                        Bind("v1", Type.GRID, Call(hint_b.name, (Ref("v0"),))),
                        Bind("v2", Type.GRID, Call(hint_c.name, (Ref("v1"),))),
                    ], "v2"),
                    f"chain3:{hint_a.name}+{hint_b.name}+{hint_c.name}",
                )


def _instantiate_op(
    op_name: str,
    sig: OpSignature,
    literal_pool: dict[Type, tuple[Literal, ...]],
):
    """Yield all type-valid skeleton instantiations of an op.

    For each parameter, either use a matching ref from _AVAILABLE_REFS
    or a literal from the pool. The product of all choices is capped.
    """
    param_choices: list[list[Expr]] = []
    for param_name, param_type in sig.params:
        choices: list[Expr] = []
        # Refs that match this param type
        for ref_name, ref_type in _AVAILABLE_REFS.items():
            if _types_assignable(param_type, ref_type):
                choices.append(Ref(ref_name))
        # Literals that match
        for lit in _limited_literals(param_type, literal_pool):
            choices.append(lit)
        if not choices:
            break
        param_choices.append(choices)

    if len(param_choices) != len(sig.params):
        return

    # Enumerate all combinations, capped
    for args in _capped_product(param_choices, cap=12):
        yield (
            _make_program([
                Bind("v0", sig.return_type, Call(op_name, tuple(args))),
            ], "v0"),
            f"single:{op_name}",
        )


def _capped_product(
    choices: list[list[Expr]],
    cap: int,
):
    """Yield tuples from the cartesian product of choices, up to cap."""
    if not choices:
        return
    yielded = 0
    indices = [0] * len(choices)
    while True:
        yield tuple(choices[i][indices[i]] for i in range(len(choices)))
        yielded += 1
        if yielded >= cap:
            return
        # Increment odometer
        pos = len(indices) - 1
        while pos >= 0:
            indices[pos] += 1
            if indices[pos] < len(choices[pos]):
                break
            indices[pos] = 0
            pos -= 1
        if pos < 0:
            return


def _types_assignable(expected: Type, actual: Type) -> bool:
    if expected == actual:
        return True
    if {expected, actual} == {Type.INT, Type.COLOR}:
        return True
    callable_types = {Type.PREDICATE, Type.OBJ_TRANSFORM, Type.GRID_TRANSFORM, Type.CALLABLE}
    if expected == Type.CALLABLE and actual in callable_types:
        return True
    if actual == Type.CALLABLE and expected in callable_types:
        return True
    return False


def _limited_literals(
    typ: Type,
    literal_pool: dict[Type, tuple[Literal, ...]],
    cap: int = 8,
) -> list[Literal]:
    return list(literal_pool.get(typ, ()))[:cap]


def _make_program(steps: list[Bind], output: str) -> Program:
    return Program(steps=tuple(steps), output=output)


# ---------------------------------------------------------------------------
# Near-miss repair — type-derived wraps + diff-guided literal fixes
# ---------------------------------------------------------------------------


def _generate_repairs(
    near_misses: list[tuple[Program, str, dict]],
    literal_pool: dict[Type, tuple[Literal, ...]],
):
    """Yield (Program, source_tag) repair candidates for near-miss skeletons.

    Two generic repair strategies:
    1. Type-derived wraps: any registered op with (GRID)→GRID or
       (GRID,GRID)→GRID or (X,GRID)→GRID can wrap the near-miss output.
    2. Diff-guided literal fixes: if the diff shows specific colors are
       wrong (palette_missing/palette_extra), try substituting those
       specific literals in the skeleton AST.
    No op names are hardcoded.
    """
    all_op_sigs = all_ops()

    for base_program, base_source, diff in near_misses:
        base_output = base_program.output
        base_steps = list(base_program.steps)
        next_idx = len(base_steps)
        wrap_name = f"v{next_idx}"
        base_ref = Ref(base_output)
        input_ref = Ref("input")

        # Strategy 1: type-derived wraps using any GRID→GRID op
        for op_name, sig in all_op_sigs.items():
            if sig.return_type != Type.GRID:
                continue

            if len(sig.params) == 1 and sig.params[0][1] == Type.GRID:
                yield (
                    _make_program(
                        base_steps + [Bind(wrap_name, Type.GRID, Call(op_name, (base_ref,)))],
                        wrap_name,
                    ),
                    f"repair:{op_name}+{base_source}",
                )

            if len(sig.params) == 2:
                p0t = sig.params[0][1]
                p1t = sig.params[1][1]
                if p0t == Type.GRID and p1t == Type.GRID:
                    yield (
                        _make_program(
                            base_steps + [Bind(wrap_name, Type.GRID, Call(op_name, (base_ref, input_ref)))],
                            wrap_name,
                        ),
                        f"repair:{op_name}(skel,input)+{base_source}",
                    )
                    yield (
                        _make_program(
                            base_steps + [Bind(wrap_name, Type.GRID, Call(op_name, (input_ref, base_ref)))],
                            wrap_name,
                        ),
                        f"repair:{op_name}(input,skel)+{base_source}",
                    )
                elif p0t != Type.GRID and p1t == Type.GRID:
                    for lit in _limited_literals(p0t, literal_pool, cap=4):
                        yield (
                            _make_program(
                                base_steps + [Bind(wrap_name, Type.GRID, Call(op_name, (lit, base_ref)))],
                                wrap_name,
                            ),
                            f"repair:{op_name}+{base_source}",
                        )

        # Strategy 2: diff-guided literal substitution
        palette_missing = diff.get("palette_missing", [])
        palette_extra = diff.get("palette_extra", [])
        if palette_missing and palette_extra:
            for fix_program, fix_source in _diff_literal_fixes(
                base_program, base_source, palette_missing, palette_extra,
            ):
                yield fix_program, fix_source


def _diff_literal_fixes(
    program: Program,
    source: str,
    palette_missing: list[int],
    palette_extra: list[int],
):
    """Try substituting wrong-color literals with correct-color literals.

    If the diff says color 3 is extra and color 7 is missing, try
    replacing every literal 3 in the program with literal 7.
    Generic over any program structure.
    """
    for wrong_color in palette_extra:
        for right_color in palette_missing:
            new_steps = tuple(
                _replace_literal_in_step(step, wrong_color, right_color)
                for step in program.steps
            )
            if new_steps == program.steps:
                continue
            yield (
                Program(steps=new_steps, output=program.output),
                f"litfix:{wrong_color}->{right_color}+{source}",
            )


def _replace_literal_in_step(step: Bind, old_val: int, new_val: int) -> Bind:
    new_expr = _replace_literal_in_expr(step.expr, old_val, new_val)
    if new_expr is step.expr:
        return step
    return Bind(name=step.name, typ=step.typ, expr=new_expr, declared=step.declared)


def _replace_literal_in_expr(expr: Expr, old_val: int, new_val: int) -> Expr:
    if isinstance(expr, Literal):
        if isinstance(expr.value, int) and expr.value == old_val:
            return Literal(value=new_val, typ=expr.typ)
        return expr
    if isinstance(expr, Call):
        new_args = tuple(
            _replace_literal_in_expr(arg, old_val, new_val)
            for arg in expr.args
        )
        if new_args == expr.args:
            return expr
        return Call(op=expr.op, args=new_args)
    if isinstance(expr, Ref):
        return expr
    return expr
