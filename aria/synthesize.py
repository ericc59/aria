"""Observation-driven program synthesis.

Instead of enumerating programs and checking if they match, this module
observes what changed between input and output and directly constructs
candidate programs from those observations. Each observation is a
hypothesis derived from the data, not from search.

The pipeline:
1. Direct transforms: does output == known_op(input)?
2. Color map inference: does output == apply_color_map(inferred_map, input)?
3. Difference analysis: what's the overlay between input and output?
4. Object-level observations: how did individual objects change?

Every candidate is verified exactly. No heuristics bypass the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.offline_search import build_literal_pool
from aria.runtime.ops import all_ops, get_op, has_op
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import (
    Axis,
    Bind,
    Call,
    DemoPair,
    Dir,
    Expr,
    Grid,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.verify.verifier import verify


@dataclass(frozen=True)
class SynthesisResult:
    solved: bool
    winning_program: Program | None
    candidates_tested: int
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Observation:
    """A single observation about the input→output relationship."""

    kind: str  # "direct_transform", "color_map", "composed", ...
    program_text: str
    passed: bool
    source: str  # human-readable description of what was observed


_ENV = {"input": Type.GRID, "ctx": Type.TASK_CTX}


def synthesize_from_observations(
    demos: tuple[DemoPair, ...],
) -> SynthesisResult:
    """Derive candidate programs from observations, verify each.

    This is the core reasoning step: look at what changed, construct
    the program that would produce that change, and verify it.
    """
    tested = 0
    observations: list[Observation] = []
    literal_pool = build_literal_pool(demos)

    # Phase 1: direct single-op transforms
    for program, source in _direct_transforms(literal_pool):
        if type_check(program, initial_env=dict(_ENV)):
            continue
        tested += 1
        result = verify(program, demos)
        text = program_to_text(program)
        observations.append(Observation(
            kind="direct_transform", program_text=text,
            passed=result.passed, source=source,
        ))
        if result.passed:
            return SynthesisResult(
                solved=True, winning_program=program,
                candidates_tested=tested,
                observations=tuple(observations),
            )

    # Phase 2: infer color map from first demo, verify on all
    for program, source in _color_map_inference(demos):
        if type_check(program, initial_env=dict(_ENV)):
            continue
        tested += 1
        result = verify(program, demos)
        text = program_to_text(program)
        observations.append(Observation(
            kind="color_map", program_text=text,
            passed=result.passed, source=source,
        ))
        if result.passed:
            return SynthesisResult(
                solved=True, winning_program=program,
                candidates_tested=tested,
                observations=tuple(observations),
            )

    # Phase 3: composed transforms — apply transform A, then transform B
    for program, source in _composed_transforms(literal_pool):
        if type_check(program, initial_env=dict(_ENV)):
            continue
        tested += 1
        result = verify(program, demos)
        text = program_to_text(program)
        observations.append(Observation(
            kind="composed", program_text=text,
            passed=result.passed, source=source,
        ))
        if result.passed:
            return SynthesisResult(
                solved=True, winning_program=program,
                candidates_tested=tested,
                observations=tuple(observations),
            )

    # Phase 4: difference-based synthesis for same-size additive tasks
    for program, source in _difference_synthesis(demos, literal_pool):
        if type_check(program, initial_env=dict(_ENV)):
            continue
        tested += 1
        result = verify(program, demos)
        text = program_to_text(program)
        observations.append(Observation(
            kind="difference", program_text=text,
            passed=result.passed, source=source,
        ))
        if result.passed:
            return SynthesisResult(
                solved=True, winning_program=program,
                candidates_tested=tested,
                observations=tuple(observations),
            )

    return SynthesisResult(
        solved=False, winning_program=None,
        candidates_tested=tested,
        observations=tuple(observations),
    )


# ---------------------------------------------------------------------------
# Phase 1: Direct single-op transforms
# ---------------------------------------------------------------------------


def _direct_transforms(literal_pool):
    """Try every single-op transform that takes GRID → GRID."""
    all_op_sigs = all_ops()

    # 1-param GRID→GRID: op(input)
    for name, sig in sorted(all_op_sigs.items()):
        if sig.return_type != Type.GRID:
            continue
        if len(sig.params) != 1:
            continue
        if sig.params[0][1] != Type.GRID:
            continue
        yield (
            _prog([Bind("v0", Type.GRID, Call(name, (Ref("input"),)))], "v0"),
            f"{name}(input)",
        )

    # 2-param with one GRID and one literal
    for name, sig in sorted(all_op_sigs.items()):
        if sig.return_type != Type.GRID:
            continue
        if len(sig.params) != 2:
            continue

        p0n, p0t = sig.params[0]
        p1n, p1t = sig.params[1]

        if p0t != Type.GRID and p1t == Type.GRID:
            for lit in _full_lits(name, p0n, p0t, literal_pool):
                yield (
                    _prog([Bind("v0", Type.GRID, Call(name, (lit, Ref("input"))))], "v0"),
                    f"{name}({_lit_str(lit)}, input)",
                )
        elif p0t == Type.GRID and p1t != Type.GRID:
            for lit in _full_lits(name, p1n, p1t, literal_pool):
                yield (
                    _prog([Bind("v0", Type.GRID, Call(name, (Ref("input"), lit)))], "v0"),
                    f"{name}(input, {_lit_str(lit)})",
                )


# ---------------------------------------------------------------------------
# Phase 2: Color map inference
# ---------------------------------------------------------------------------


def _color_map_inference(demos: tuple[DemoPair, ...]):
    """Infer a color mapping from the first demo, construct the program."""
    if not demos:
        return

    demo = demos[0]
    inp, out = demo.input, demo.output

    if inp.shape != out.shape:
        return

    # Build the color map by observing every pixel
    color_map: dict[int, int] = {}
    consistent = True
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            iv, ov = int(inp[r, c]), int(out[r, c])
            if iv in color_map:
                if color_map[iv] != ov:
                    consistent = False
                    break
            else:
                color_map[iv] = ov
        if not consistent:
            break

    if not consistent:
        return

    # Only yield if the map actually changes something
    if all(k == v for k, v in color_map.items()):
        return

    # Construct: apply_color_map({...}, input)
    if has_op("apply_color_map"):
        map_literal = Literal(color_map, Type.COLOR_MAP)
        yield (
            _prog([
                Bind("v0", Type.GRID, Call("apply_color_map", (map_literal, Ref("input")))),
            ], "v0"),
            f"color_map({color_map})",
        )


# ---------------------------------------------------------------------------
# Phase 3: Composed transforms — A(B(input))
# ---------------------------------------------------------------------------


def _composed_transforms(literal_pool):
    """Try pairs of simple transforms: v0 = A(input), v1 = B(v0)."""
    all_op_sigs = all_ops()

    # Collect all (GRID)→GRID single-param ops
    g2g = [
        (name, sig) for name, sig in sorted(all_op_sigs.items())
        if sig.return_type == Type.GRID
        and len(sig.params) == 1
        and sig.params[0][1] == Type.GRID
    ]

    # Collect (literal, GRID)→GRID ops with their literals
    lg2g: list[tuple[str, Literal]] = []
    for name, sig in sorted(all_op_sigs.items()):
        if sig.return_type != Type.GRID or len(sig.params) != 2:
            continue
        p0t, p1t = sig.params[0][1], sig.params[1][1]
        if p0t != Type.GRID and p1t == Type.GRID:
            for lit in _full_lits(name, sig.params[0][0], p0t, literal_pool):
                lg2g.append((name, lit))
        elif p0t == Type.GRID and p1t != Type.GRID:
            for lit in _full_lits(name, sig.params[1][0], p1t, literal_pool):
                lg2g.append((name, lit))

    # Build step1 options: all ways to get a GRID from input
    step1_options: list[tuple[Bind, str]] = []
    for name, sig in g2g:
        step1_options.append((
            Bind("v0", Type.GRID, Call(name, (Ref("input"),))),
            name,
        ))
    for name, lit in lg2g[:20]:  # cap
        if all_op_sigs[name].params[0][1] == Type.GRID:
            step1_options.append((
                Bind("v0", Type.GRID, Call(name, (Ref("input"), lit))),
                f"{name}(input,{_lit_str(lit)})",
            ))
        else:
            step1_options.append((
                Bind("v0", Type.GRID, Call(name, (lit, Ref("input")))),
                f"{name}({_lit_str(lit)},input)",
            ))

    # Build step2 options: all ways to get a GRID from v0
    for step1, s1_desc in step1_options[:15]:  # cap step1
        for name, sig in g2g:
            yield (
                _prog([step1, Bind("v1", Type.GRID, Call(name, (Ref("v0"),)))], "v1"),
                f"{name}({s1_desc}(input))",
            )
        for name, lit in lg2g[:10]:  # cap step2 literals
            if all_op_sigs[name].params[0][1] == Type.GRID:
                yield (
                    _prog([step1, Bind("v1", Type.GRID, Call(name, (Ref("v0"), lit)))], "v1"),
                    f"{name}({s1_desc}(input), {_lit_str(lit)})",
                )
            else:
                yield (
                    _prog([step1, Bind("v1", Type.GRID, Call(name, (lit, Ref("v0"))))], "v1"),
                    f"{name}({_lit_str(lit)}, {s1_desc}(input))",
                )


# ---------------------------------------------------------------------------
# Phase 4: Difference-based synthesis
# ---------------------------------------------------------------------------


def _difference_synthesis(demos: tuple[DemoPair, ...], literal_pool):
    """For same-size tasks: compute what changed, try to explain the diff.

    If the output is mostly the same as the input with some changes,
    try: overlay(input, constructed_diff).
    """
    if not demos:
        return

    demo = demos[0]
    inp, out = demo.input, demo.output

    if inp.shape != out.shape:
        return

    changed_mask = inp != out
    if not np.any(changed_mask):
        return

    preserved_frac = 1.0 - np.sum(changed_mask) / changed_mask.size
    if preserved_frac < 0.3:
        return  # too much changed for overlay to be the right model

    # The diff is where output differs from input.
    # Try overlay(input, fill_region(...)) type constructions.
    # First: try overlay with simple grid ops applied to input
    all_op_sigs = all_ops()

    if not has_op("overlay"):
        return

    # overlay(input, op(input)) for various ops
    for name, sig in sorted(all_op_sigs.items()):
        if sig.return_type != Type.GRID:
            continue
        if len(sig.params) == 1 and sig.params[0][1] == Type.GRID:
            yield (
                _prog([
                    Bind("v0", Type.GRID, Call(name, (Ref("input"),))),
                    Bind("v1", Type.GRID, Call("overlay", (Ref("input"), Ref("v0")))),
                ], "v1"),
                f"overlay(input, {name}(input))",
            )
            yield (
                _prog([
                    Bind("v0", Type.GRID, Call(name, (Ref("input"),))),
                    Bind("v1", Type.GRID, Call("overlay", (Ref("v0"), Ref("input")))),
                ], "v1"),
                f"overlay({name}(input), input)",
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(steps: list[Bind], output: str) -> Program:
    return Program(steps=tuple(steps), output=output)


def _lits(typ: Type, pool: dict[Type, tuple[Literal, ...]], cap: int = 8) -> list[Literal]:
    return list(pool.get(typ, ()))[:cap]


def _full_lits(
    op_name: str, param_name: str, typ: Type,
    pool: dict[Type, tuple[Literal, ...]],
) -> list[Literal]:
    """Literals for a specific op+param, including domain knowledge.

    For rotation degrees, axis values, etc. — things the demo-derived
    literal pool won't contain but the op semantics require.
    """
    base = list(pool.get(typ, ()))

    if typ == Type.INT:
        # Rotation degrees
        extras = [90, 180, 270]
        existing = {int(l.value) for l in base if isinstance(l.value, int)}
        for v in extras:
            if v not in existing:
                base.append(Literal(v, Type.INT))

    return base[:16]


def _lit_str(lit: Literal) -> str:
    v = lit.value
    if hasattr(v, "name"):
        return v.name
    return repr(v)
