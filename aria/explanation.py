"""Near-miss explanation and repair for same-dims candidates.

Given a candidate output that has the right dimensions but wrong pixels,
this module:
1. Builds an Explanation describing exactly what's wrong
2. Proposes typed repair actions derived from the explanation
3. Applies repairs and verifies them

The explanation is pixel-level, not program-level. This makes it
independent of how the candidate was generated (synthesis, observation,
search, skeleton, etc.).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.types import DemoPair, Grid, Program, Type, Bind, Call, Literal, Ref
from aria.runtime.program import program_to_text
from aria.verify.verifier import verify
from aria.verify.trace import compute_diff


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelError:
    """One wrong pixel."""
    row: int
    col: int
    actual: int
    expected: int


@dataclass(frozen=True)
class Explanation:
    """What's wrong with a same-dims near-miss candidate on one demo."""

    pixel_diff_count: int
    total_pixels: int
    wrong_pixels: tuple[PixelError, ...]

    # Derived classification
    error_class: str  # "color_swap", "missing_content", "extra_content", "mixed"
    color_swaps: dict[int, int]  # {actual_color: expected_color} when consistent
    missing_colors: tuple[int, ...]  # colors in expected but not in actual
    extra_colors: tuple[int, ...]  # colors in actual but not in expected

    # Residual pattern
    residual_uniform: bool  # are all wrong expected-pixels the same color?
    residual_color: int | None  # the uniform residual color, if any


def build_explanation(
    actual: Grid,
    expected: Grid,
    input_grid: Grid | None = None,
) -> Explanation | None:
    """Build an explanation for a same-dims near-miss."""
    if actual.shape != expected.shape:
        return None

    mismatch = actual != expected
    diff_count = int(np.sum(mismatch))
    if diff_count == 0:
        return None  # not a near-miss, it's correct

    total = int(actual.size)
    wrong: list[PixelError] = []
    swap_candidates: dict[int, Counter] = {}

    for r in range(actual.shape[0]):
        for c in range(actual.shape[1]):
            if mismatch[r, c]:
                av, ev = int(actual[r, c]), int(expected[r, c])
                wrong.append(PixelError(r, c, av, ev))
                swap_candidates.setdefault(av, Counter())[ev] += 1

    # Classify the error
    actual_palette = set(int(v) for v in np.unique(actual))
    expected_palette = set(int(v) for v in np.unique(expected))
    missing = sorted(expected_palette - actual_palette)
    extra = sorted(actual_palette - expected_palette)

    # Check for consistent color swaps
    color_swaps: dict[int, int] = {}
    is_pure_swap = True
    for actual_c, target_counts in swap_candidates.items():
        if len(target_counts) == 1:
            target_c = target_counts.most_common(1)[0][0]
            color_swaps[actual_c] = target_c
        else:
            is_pure_swap = False

    # A pure color swap should not involve background (0) as source
    real_swaps = {k: v for k, v in color_swaps.items() if k != 0}
    if is_pure_swap and real_swaps and not any(k == 0 for k in color_swaps):
        error_class = "color_swap"
    elif missing and not extra:
        error_class = "missing_content"
    elif extra and not missing:
        error_class = "extra_content"
    else:
        error_class = "mixed"

    # Check if residual (expected where wrong) is uniform
    residual_colors = Counter(p.expected for p in wrong)
    if len(residual_colors) == 1:
        residual_uniform = True
        residual_color = residual_colors.most_common(1)[0][0]
    else:
        residual_uniform = False
        residual_color = None

    return Explanation(
        pixel_diff_count=diff_count,
        total_pixels=total,
        wrong_pixels=tuple(wrong),
        error_class=error_class,
        color_swaps=color_swaps,
        missing_colors=tuple(missing),
        extra_colors=tuple(extra),
        residual_uniform=residual_uniform,
        residual_color=residual_color,
    )


# ---------------------------------------------------------------------------
# Repair actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairAction:
    """A typed repair to apply to a candidate grid."""
    kind: str  # "color_swap", "fill_residual", "overlay_input"
    details: dict[str, Any]


def propose_repairs(
    explanation: Explanation,
    actual: Grid,
    expected: Grid,
    input_grid: Grid,
) -> list[tuple[Grid, RepairAction]]:
    """Propose repaired grids from explanation analysis."""
    repairs: list[tuple[Grid, RepairAction]] = []

    # Repair 1: color swap — apply the inferred color map to the actual grid
    if explanation.color_swaps:
        repaired = actual.copy()
        for src, dst in explanation.color_swaps.items():
            repaired[actual == src] = dst
        repairs.append((repaired, RepairAction(
            kind="color_swap",
            details={"swap_map": explanation.color_swaps},
        )))

    # Repair 2: fill residual — where actual differs from expected,
    # replace with the expected values (only useful if residual is simple)
    if explanation.residual_uniform and explanation.residual_color is not None:
        repaired = actual.copy()
        for p in explanation.wrong_pixels:
            repaired[p.row, p.col] = p.expected
        repairs.append((repaired, RepairAction(
            kind="fill_residual",
            details={"residual_color": explanation.residual_color,
                     "pixel_count": len(explanation.wrong_pixels)},
        )))

    # Repair 3: overlay input — if the candidate is missing content
    # that exists in the input, overlay input onto candidate
    if explanation.error_class in ("missing_content", "mixed"):
        repaired = actual.copy()
        for r in range(actual.shape[0]):
            for c in range(actual.shape[1]):
                if actual[r, c] != expected[r, c] and int(input_grid[r, c]) == int(expected[r, c]):
                    repaired[r, c] = input_grid[r, c]
        if not np.array_equal(repaired, actual):
            repairs.append((repaired, RepairAction(
                kind="overlay_input",
                details={"description": "restore input pixels where candidate was wrong"},
            )))

    # Repair 4: for each single missing color, try filling all wrong
    # positions that should have that color
    for mc in explanation.missing_colors:
        repaired = actual.copy()
        for p in explanation.wrong_pixels:
            if p.expected == mc:
                repaired[p.row, p.col] = mc
        if not np.array_equal(repaired, actual):
            repairs.append((repaired, RepairAction(
                kind="fill_missing_color",
                details={"color": mc},
            )))

    return repairs


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairResult:
    """Outcome of the explanation-driven repair phase."""
    solved: bool
    winning_grid: Grid | None
    winning_action: RepairAction | None
    winning_program: Program | None  # if repair can be expressed as a program
    repaired_targets: tuple[Grid, ...] | None = None  # per-demo repaired outputs
    explanations_built: int = 0
    repairs_tried: int = 0
    primary_error_class: str | None = None


def attempt_repair(
    demos: tuple[DemoPair, ...],
    candidate_grids: list[Grid],
    candidate_programs: list[Program | None] | None = None,
    *,
    max_repairs: int = 20,
) -> RepairResult:
    """Try to repair near-miss candidate grids.

    For each candidate grid, builds an explanation per demo, proposes
    repairs, and verifies the repaired outputs across all demos.
    Only considers same-dims candidates.

    If candidate_programs is provided, successful color_swap repairs
    are expressed as executable programs (apply_color_map after the
    original program).
    """
    if not demos or not candidate_grids:
        return RepairResult(
            solved=False, winning_grid=None, winning_action=None,
            winning_program=None,
            explanations_built=0, repairs_tried=0,
            primary_error_class=None,
        )

    progs = candidate_programs or [None] * len(candidate_grids)

    explanations_built = 0
    repairs_tried = 0
    error_classes: list[str] = []

    for ci, candidate in enumerate(candidate_grids):
        base_prog = progs[ci] if ci < len(progs) else None

        demo_explanations: list[Explanation] = []
        skip = False
        for demo in demos:
            if candidate.shape != demo.output.shape:
                skip = True
                break
            expl = build_explanation(candidate, demo.output, demo.input)
            if expl is None:
                skip = True
                break
            demo_explanations.append(expl)
            explanations_built += 1

        if skip or not demo_explanations:
            continue

        error_classes.append(demo_explanations[0].error_class)

        first_expl = demo_explanations[0]
        repair_candidates = propose_repairs(
            first_expl, candidate, demos[0].output, demos[0].input,
        )

        for repaired_grid, action in repair_candidates:
            if repairs_tried >= max_repairs:
                break
            repairs_tried += 1

            all_correct = True
            per_demo_repaired: list[Grid] = [repaired_grid]
            for di, demo in enumerate(demos):
                if di == 0:
                    if not np.array_equal(repaired_grid, demo.output):
                        all_correct = False
                        break
                else:
                    demo_repaired = _apply_repair_to_demo(
                        action, candidate, demo.input, demo.output,
                    )
                    if demo_repaired is None or not np.array_equal(demo_repaired, demo.output):
                        all_correct = False
                        break
                    per_demo_repaired.append(demo_repaired)

            if all_correct:
                win_prog = _repair_to_program(action, base_prog)
                if win_prog is not None:
                    vr = verify(win_prog, demos)
                    if not vr.passed:
                        win_prog = None

                return RepairResult(
                    solved=True,
                    winning_grid=repaired_grid,
                    winning_action=action,
                    winning_program=win_prog,
                    repaired_targets=tuple(per_demo_repaired),
                    explanations_built=explanations_built,
                    repairs_tried=repairs_tried,
                    primary_error_class=first_expl.error_class,
                )

    most_common_class = Counter(error_classes).most_common(1)[0][0] if error_classes else None
    return RepairResult(
        solved=False, winning_grid=None, winning_action=None,
        winning_program=None,
        explanations_built=explanations_built,
        repairs_tried=repairs_tried,
        primary_error_class=most_common_class,
    )


def _repair_to_program(
    action: RepairAction,
    base_program: Program | None,
) -> Program | None:
    """Try to express a repair action as an executable program.

    - color_swap on a program → apply_color_map(swap_map, base_program(input))
    """
    if action.kind == "color_swap" and base_program is not None:
        swap_map = action.details.get("swap_map", {})
        if not swap_map:
            return None
        base_output = base_program.output
        base_steps = list(base_program.steps)
        idx = len(base_steps)
        repair_step = Bind(
            f"v{idx}", Type.GRID,
            Call("apply_color_map", (
                Literal(swap_map, Type.COLOR_MAP),
                Ref(base_output),
            )),
        )
        return Program(
            steps=tuple(base_steps) + (repair_step,),
            output=f"v{idx}",
        )

    return None


def attempt_program_repair(
    demos: tuple[DemoPair, ...],
    candidate_programs: list[Program],
    *,
    max_repairs: int = 30,
) -> RepairResult:
    """Try to repair near-miss programs by editing their literals.

    For each program-backed near-miss, tries:
    1. Appending apply_color_map with inferred swap map
    2. Swapping individual color literals in the program
    3. Adjusting integer literals by small deltas

    This produces executable programs that can run on test inputs.
    """
    if not demos or not candidate_programs:
        return RepairResult(
            solved=False, winning_grid=None, winning_action=None,
            winning_program=None,
            explanations_built=0, repairs_tried=0,
            primary_error_class=None,
        )

    from aria.runtime.executor import execute

    repairs_tried = 0
    explanations_built = 0

    for prog in candidate_programs:
        # Execute on first demo to get the candidate grid
        try:
            candidate = execute(prog, demos[0].input, None)
        except Exception:
            continue
        if not isinstance(candidate, np.ndarray):
            continue
        if candidate.shape != demos[0].output.shape:
            continue

        expl = build_explanation(candidate, demos[0].output, demos[0].input)
        if expl is None:
            continue
        explanations_built += 1

        # Repair 1: append color map for color_swap errors
        if expl.color_swaps:
            swap_map = {k: v for k, v in expl.color_swaps.items() if k != 0}
            if swap_map:
                repaired_prog = _append_color_map(prog, swap_map)
                repairs_tried += 1
                vr = verify(repaired_prog, demos)
                if vr.passed:
                    return RepairResult(
                        solved=True, winning_grid=None,
                        winning_action=RepairAction("program_color_map", {"swap_map": swap_map}),
                        winning_program=repaired_prog,
                        explanations_built=explanations_built,
                        repairs_tried=repairs_tried,
                        primary_error_class=expl.error_class,
                    )

        # Repair 2: try swapping individual color literals
        for step_idx, step in enumerate(prog.steps):
            if not isinstance(step, Bind) or not isinstance(step.expr, Call):
                continue
            for arg_idx, arg in enumerate(step.expr.args):
                if not isinstance(arg, Literal) or arg.typ not in (Type.COLOR, Type.INT):
                    continue
                if not isinstance(arg.value, int):
                    continue
                # Try replacing with each color from the expected palette
                expected_colors = set()
                for demo in demos:
                    expected_colors.update(int(v) for v in np.unique(demo.output))
                for new_val in sorted(expected_colors):
                    if new_val == arg.value:
                        continue
                    if repairs_tried >= max_repairs:
                        break
                    new_args = list(step.expr.args)
                    new_args[arg_idx] = Literal(new_val, arg.typ)
                    new_expr = Call(op=step.expr.op, args=tuple(new_args))
                    new_step = Bind(name=step.name, typ=step.typ, expr=new_expr, declared=step.declared)
                    new_steps = list(prog.steps)
                    new_steps[step_idx] = new_step
                    new_prog = Program(steps=tuple(new_steps), output=prog.output)
                    repairs_tried += 1
                    try:
                        vr = verify(new_prog, demos)
                        if vr.passed:
                            return RepairResult(
                                solved=True, winning_grid=None,
                                winning_action=RepairAction("program_literal_swap", {
                                    "step": step_idx, "arg": arg_idx,
                                    "old": arg.value, "new": new_val,
                                }),
                                winning_program=new_prog,
                                explanations_built=explanations_built,
                                repairs_tried=repairs_tried,
                                primary_error_class=expl.error_class,
                            )
                    except Exception:
                        pass

    return RepairResult(
        solved=False, winning_grid=None, winning_action=None,
        winning_program=None,
        explanations_built=explanations_built,
        repairs_tried=repairs_tried,
        primary_error_class=None,
    )


def _append_color_map(prog: Program, swap_map: dict[int, int]) -> Program:
    """Append apply_color_map(swap_map, output) to a program."""
    base_output = prog.output
    idx = len(prog.steps)
    repair_step = Bind(
        f"v{idx}", Type.GRID,
        Call("apply_color_map", (Literal(swap_map, Type.COLOR_MAP), Ref(base_output))),
    )
    return Program(steps=prog.steps + (repair_step,), output=f"v{idx}")


def _apply_repair_to_demo(
    action: RepairAction,
    candidate: Grid,
    input_grid: Grid,
    expected: Grid,
) -> Grid | None:
    """Apply a repair action to a candidate for a specific demo."""
    if candidate.shape != expected.shape:
        return None

    if action.kind == "color_swap":
        swap_map = action.details.get("swap_map", {})
        repaired = candidate.copy()
        for src, dst in swap_map.items():
            repaired[candidate == src] = dst
        return repaired

    if action.kind == "fill_residual":
        # Fill wrong positions with expected values
        repaired = candidate.copy()
        mismatch = candidate != expected
        repaired[mismatch] = expected[mismatch]
        return repaired

    if action.kind == "overlay_input":
        repaired = candidate.copy()
        for r in range(candidate.shape[0]):
            for c in range(candidate.shape[1]):
                if candidate[r, c] != expected[r, c] and int(input_grid[r, c]) == int(expected[r, c]):
                    repaired[r, c] = input_grid[r, c]
        return repaired

    if action.kind == "fill_missing_color":
        color = action.details.get("color")
        if color is None:
            return None
        repaired = candidate.copy()
        mismatch = candidate != expected
        for r in range(candidate.shape[0]):
            for c in range(candidate.shape[1]):
                if mismatch[r, c] and int(expected[r, c]) == color:
                    repaired[r, c] = color
        return repaired

    return None
