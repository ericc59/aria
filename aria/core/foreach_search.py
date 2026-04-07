"""ForEach program search: enumerate entity-conditional programs.

Generates ForEach programs with bounded entity selectors and body templates,
then verifies exactly on all demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.types import (
    Bind,
    Call,
    DemoPair,
    ForEach,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.runtime.program import program_to_text
from aria.verify.verifier import verify


@dataclass(frozen=True)
class ForEachSearchResult:
    solved: bool
    winning_program: Program | None = None
    winning_text: str = ""
    candidates_tested: int = 0


# ---------------------------------------------------------------------------
# Entity selectors: expressions that produce OBJECT_SET
# ---------------------------------------------------------------------------


def _entity_selectors(color_pool: list[int]) -> list[tuple[str, tuple]]:
    """Generate (description, (prefix_steps, source_expr)) pairs.

    Each selector is a pair of prefix Bind steps + a source expression
    that evaluates to OBJECT_SET.
    """
    selectors = []

    # Base: find_objects(input)
    find_all = Call(op="find_objects", args=(Ref(name="input"),))
    find_step = Bind(name="objs", typ=Type.OBJECT_SET, expr=find_all)

    # Unfiltered
    selectors.append(("all_objects", ((find_step,), Ref(name="objs"))))

    # Filter by size
    for threshold in (1, 2, 3, 4, 5):
        filt = Call(op="filter_by_min_size", args=(
            Ref(name="objs"), Literal(value=threshold, typ=Type.INT)))
        step = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt)
        selectors.append((
            f"min_size_{threshold}",
            ((find_step, step), Ref(name="filtered")),
        ))

        filt2 = Call(op="filter_by_max_size", args=(
            Ref(name="objs"), Literal(value=threshold, typ=Type.INT)))
        step2 = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt2)
        selectors.append((
            f"max_size_{threshold}",
            ((find_step, step2), Ref(name="filtered")),
        ))

    # Filter by color
    for color in color_pool:
        filt = Call(op="filter_by_color", args=(
            Ref(name="objs"), Literal(value=color, typ=Type.INT)))
        step = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt)
        selectors.append((
            f"color_{color}",
            ((find_step, step), Ref(name="filtered")),
        ))

        filt2 = Call(op="filter_by_not_color", args=(
            Ref(name="objs"), Literal(value=color, typ=Type.INT)))
        step2 = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt2)
        selectors.append((
            f"not_color_{color}",
            ((find_step, step2), Ref(name="filtered")),
        ))

    # Singletons / non-singletons
    filt_s = Call(op="filter_singletons", args=(Ref(name="objs"),))
    step_s = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt_s)
    selectors.append(("singletons", ((find_step, step_s), Ref(name="filtered"))))

    filt_ns = Call(op="filter_non_singletons", args=(Ref(name="objs"),))
    step_ns = Bind(name="filtered", typ=Type.OBJECT_SET, expr=filt_ns)
    selectors.append(("non_singletons", ((find_step, step_ns), Ref(name="filtered"))))

    return selectors


# ---------------------------------------------------------------------------
# Body templates: per-entity grid mutations
# ---------------------------------------------------------------------------


def _body_templates(color_pool: list[int], bg: int) -> list[tuple[str, tuple]]:
    """Generate (description, body_steps) pairs.

    Each body is a tuple of Bind steps using Ref("e") for the entity
    and Ref("input") for the accumulator grid.
    """
    templates = []
    e = Ref(name="e")
    acc = Ref(name="input")  # accumulator is "input"

    # 1. Fill entity with color
    for c in color_pool:
        step = Bind(name="input", typ=Type.GRID,
                     expr=Call(op="fill_entity", args=(e, Literal(value=c, typ=Type.INT), acc)))
        templates.append((f"fill_entity({c})", (step,)))

    # 2. Erase entity
    step = Bind(name="input", typ=Type.GRID,
                 expr=Call(op="erase_entity", args=(e, acc)))
    templates.append(("erase_entity", (step,)))

    # 3. Fill enclosed regions within entity bbox
    for c in color_pool:
        step = Bind(name="input", typ=Type.GRID,
                     expr=Call(op="fill_entity_enclosed", args=(
                         e, Literal(value=c, typ=Type.INT), acc)))
        templates.append((f"fill_entity_enclosed({c})", (step,)))

    # 4. Recolor within entity bbox
    for fc in color_pool:
        for tc in color_pool:
            if fc == tc:
                continue
            step = Bind(name="input", typ=Type.GRID,
                         expr=Call(op="recolor_entity", args=(
                             e,
                             Literal(value=fc, typ=Type.INT),
                             Literal(value=tc, typ=Type.INT),
                             acc)))
            templates.append((f"recolor_entity({fc}->{tc})", (step,)))

    # 5. Fill bg in entity bbox with color
    for c in color_pool:
        if c == bg:
            continue
        step = Bind(name="input", typ=Type.GRID,
                     expr=Call(op="fill_entity_bbox_bg", args=(
                         e, Literal(value=c, typ=Type.INT), acc)))
        templates.append((f"fill_entity_bbox_bg({c})", (step,)))

    return templates


# ---------------------------------------------------------------------------
# ForEach program enumeration
# ---------------------------------------------------------------------------


def search_foreach_programs(
    demos: tuple[DemoPair, ...],
    *,
    max_candidates: int = 5000,
) -> ForEachSearchResult:
    """Enumerate and verify ForEach programs.

    1. Compute color pool and bg from demos
    2. Generate entity selectors
    3. For each selector × body template, build and verify a ForEach program
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return ForEachSearchResult(solved=False, candidates_tested=0)

    # Color pool from all demos
    all_colors: set[int] = set()
    for d in demos:
        all_colors.update(int(v) for v in np.unique(d.input))
        all_colors.update(int(v) for v in np.unique(d.output))
    bg = int(np.bincount(demos[0].input.ravel()).argmax())
    color_pool = sorted(all_colors)

    selectors = _entity_selectors(color_pool)
    bodies = _body_templates(color_pool, bg)

    total_tested = 0
    seen: set[str] = set()

    for sel_desc, (prefix_steps, source_expr) in selectors:
        for body_desc, body_steps in bodies:
            if total_tested >= max_candidates:
                return ForEachSearchResult(
                    solved=False, candidates_tested=total_tested)

            # Build the ForEach program
            foreach = ForEach(
                iter_name="e",
                source=source_expr,
                body=body_steps,
                accumulator="input",
                output_name="v_out",
            )

            all_steps = prefix_steps + (foreach,)
            prog = Program(steps=all_steps, output="v_out")

            # Deduplicate
            key = program_to_text(prog)
            if key in seen:
                continue
            seen.add(key)

            total_tested += 1
            vr = verify(prog, demos)
            if vr.passed:
                return ForEachSearchResult(
                    solved=True,
                    winning_program=prog,
                    winning_text=key,
                    candidates_tested=total_tested,
                )

    return ForEachSearchResult(
        solved=False, candidates_tested=total_tested)
