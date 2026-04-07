"""Depth-2 near-miss composition search.

Takes the best depth-1 near-miss programs and searches for a residual-local
corrective step 2. No new ops — just better search over depth-2 compositions
seeded by real depth-1 near-misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.library.store import Library
from aria.legacy.offline_search import SearchTraceEntry, search_program
from aria.runtime.executor import execute
from aria.runtime.ops import all_ops
from aria.runtime.program import program_to_text
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Grid,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NearMissCandidate:
    """A depth-1 program that almost solves the task."""
    program: Program
    program_text: str
    total_diff: int  # pixel diff summed across all demos
    per_demo_diffs: tuple[int, ...]
    residual_colors: tuple[tuple[int, int], ...]  # (wrong_color, expected_color) pairs
    residual_fill_colors: tuple[int, ...]  # colors that appear in expected but not actual


@dataclass(frozen=True)
class CompositionResult:
    """Result of depth-2 near-miss composition search."""
    solved: bool
    winning_program: Program | None = None
    winning_seed_text: str = ""
    winning_step2_desc: str = ""
    candidates_tested: int = 0
    best_residual: int | None = None
    seeds_tried: int = 0


# ---------------------------------------------------------------------------
# Part A: Collect depth-1 near-misses
# ---------------------------------------------------------------------------


def collect_depth1_near_misses(
    demos: tuple[DemoPair, ...],
    max_seeds: int = 10,
    search_budget: int = 500,
) -> list[NearMissCandidate]:
    """Run depth-1 search and collect the best near-misses by all-demo residual.

    Returns up to max_seeds candidates, ranked by total pixel diff (ascending).
    Only includes candidates where dims match on ALL demos.
    """
    lib = Library()
    traces: list[SearchTraceEntry] = []

    search_program(
        demos, lib,
        max_steps=1,
        max_candidates=search_budget,
        include_core_ops=True,
        observer=lambda e: traces.append(e),
    )

    # Filter to wrong_output entries with diff info
    candidates: list[NearMissCandidate] = []
    seen_diffs: set[int] = set()

    for t in traces:
        if t.error_type != "wrong_output":
            continue
        if t.diff is None or not isinstance(t.diff, dict):
            continue

        pdc = t.diff.get("pixel_diff_count")
        if pdc is None:
            continue  # dims mismatch on demo 0

        # Reconstruct the program and check ALL demos
        try:
            prog = _reconstruct_depth1_program(t.program_text)
        except Exception:
            continue

        per_demo_diffs, residual_colors, fill_colors = _compute_all_demo_residuals(
            prog, demos
        )
        if per_demo_diffs is None:
            continue  # dims mismatch on some demo

        total_diff = sum(per_demo_diffs)
        if total_diff == 0:
            continue  # already solved (shouldn't happen but guard)

        # Deduplicate by total_diff to avoid trying the same residual pattern
        if total_diff in seen_diffs:
            continue
        seen_diffs.add(total_diff)

        candidates.append(NearMissCandidate(
            program=prog,
            program_text=t.program_text,
            total_diff=total_diff,
            per_demo_diffs=tuple(per_demo_diffs),
            residual_colors=tuple(residual_colors),
            residual_fill_colors=tuple(fill_colors),
        ))

    # Sort by total_diff ascending, take top max_seeds
    candidates.sort(key=lambda c: c.total_diff)
    return candidates[:max_seeds]


def _reconstruct_depth1_program(program_text: str) -> Program:
    """Reconstruct a Program AST from search trace text.

    Depth-1 programs have the form:
        let v0: GRID = op_name(input, arg1, arg2, ...)
        -> v0
    """
    lines = program_text.strip().split("\n")
    # First line: "let v0: GRID = op_name(input, ...)"
    let_line = lines[0].strip()

    # Parse: "let v0: GRID = op_name(arg1, arg2, ...)"
    eq_idx = let_line.index("=")
    lhs = let_line[:eq_idx].strip()  # "let v0: GRID"
    rhs = let_line[eq_idx + 1:].strip()  # "op_name(arg1, ...)"

    # Extract var name
    parts = lhs.split()
    var_name = parts[1].rstrip(":")

    # Parse RHS: op_name(arg1, arg2, ...)
    paren_idx = rhs.index("(")
    op_name = rhs[:paren_idx].strip()
    args_str = rhs[paren_idx + 1:rhs.rindex(")")].strip()

    # Parse args
    args: list[Any] = []
    if args_str:
        for arg_s in args_str.split(","):
            arg_s = arg_s.strip()
            if arg_s == "input":
                args.append(Ref(name="input"))
            elif arg_s == "ctx":
                args.append(Ref(name="ctx"))
            elif arg_s.startswith("v"):
                args.append(Ref(name=arg_s))
            elif arg_s.lstrip("-").isdigit():
                args.append(Literal(value=int(arg_s), typ=Type.INT))
            else:
                args.append(Literal(value=int(arg_s), typ=Type.INT))

    step = Bind(
        name=var_name,
        typ=Type.GRID,
        expr=Call(op=op_name, args=tuple(args)),
    )
    return Program(steps=(step,), output=var_name)


def _compute_all_demo_residuals(
    prog: Program,
    demos: tuple[DemoPair, ...],
) -> tuple[list[int] | None, list[tuple[int, int]], list[int]]:
    """Compute per-demo pixel diffs and residual color patterns.

    Returns (per_demo_diffs, residual_color_pairs, fill_colors).
    Returns (None, ...) if dims mismatch on any demo.
    """
    per_demo_diffs: list[int] = []
    color_pairs: set[tuple[int, int]] = set()
    fill_colors: set[int] = set()

    for d in demos:
        try:
            result = execute(prog, d.input, None)
        except Exception:
            return None, [], []
        if result.shape != d.output.shape:
            return None, [], []

        diff_mask = result != d.output
        n_diff = int(np.sum(diff_mask))
        per_demo_diffs.append(n_diff)

        # Collect wrong→expected color pairs
        if n_diff > 0:
            wrong_vals = result[diff_mask]
            expected_vals = d.output[diff_mask]
            for w, e in zip(wrong_vals.ravel(), expected_vals.ravel()):
                color_pairs.add((int(w), int(e)))
                fill_colors.add(int(e))

    return per_demo_diffs, sorted(color_pairs), sorted(fill_colors)


# ---------------------------------------------------------------------------
# Part B: Residual-local step-2 operators
# ---------------------------------------------------------------------------


def _generate_step2_corrections(
    seed: NearMissCandidate,
    demos: tuple[DemoPair, ...],
) -> list[tuple[Program, str]]:
    """Generate bounded depth-2 programs from a seed near-miss.

    Each program freezes step 1 (the seed) and adds a corrective step 2
    that targets the residual.

    Returns list of (composed_program, step2_description).
    """
    programs: list[tuple[Program, str]] = []
    step1 = seed.program.steps[0]
    step1_var = seed.program.output

    # Collect residual info
    residual_colors = seed.residual_colors  # (wrong, expected) pairs
    fill_colors = seed.residual_fill_colors

    # Collect all colors present in any demo input/output
    all_input_colors: set[int] = set()
    all_output_colors: set[int] = set()
    for d in demos:
        all_input_colors.update(int(v) for v in np.unique(d.input))
        all_output_colors.update(int(v) for v in np.unique(d.output))

    def _make_prog(step2_expr, desc: str) -> None:
        step2 = Bind(name="v1", typ=Type.GRID, expr=step2_expr)
        prog = Program(steps=(step1, step2), output="v1")
        programs.append((prog, desc))

    v0 = Ref(name=step1_var)

    # 1. Recolor: for each (wrong_color, expected_color) pair in residual
    for wrong_c, expected_c in residual_colors:
        # conditional_fill(result, wrong_color, expected_color)
        _make_prog(
            Call(op="conditional_fill", args=(v0,
                Literal(value=wrong_c, typ=Type.INT),
                Literal(value=expected_c, typ=Type.INT))),
            f"conditional_fill({wrong_c}->{expected_c})",
        )
        # erase_color if expected is 0 (background)
        if expected_c == 0:
            _make_prog(
                Call(op="erase_color", args=(v0,
                    Literal(value=wrong_c, typ=Type.INT))),
                f"erase_color({wrong_c})",
            )

    # 2. Fill enclosed: if any fill_color is needed
    for fc in fill_colors:
        _make_prog(
            Call(op="fill_enclosed", args=(v0,
                Literal(value=fc, typ=Type.INT))),
            f"fill_enclosed({fc})",
        )
        _make_prog(
            Call(op="fill_enclosed_regions", args=(v0,
                Literal(value=fc, typ=Type.INT))),
            f"fill_enclosed_regions({fc})",
        )

    # 3. Fill between: for each color pair
    for wrong_c, expected_c in residual_colors:
        if wrong_c != 0:  # fill_between needs non-bg boundary color
            _make_prog(
                Call(op="fill_between", args=(v0,
                    Literal(value=wrong_c, typ=Type.INT),
                    Literal(value=expected_c, typ=Type.INT))),
                f"fill_between({wrong_c},{expected_c})",
            )

    # 4. Symmetry completion (if residual could be symmetric)
    _make_prog(
        Call(op="complete_symmetry_h", args=(v0,)),
        "complete_symmetry_h",
    )
    _make_prog(
        Call(op="complete_symmetry_v", args=(v0,)),
        "complete_symmetry_v",
    )

    # 5. Erase/keep by size (if residual involves extra objects)
    for threshold in (1, 2, 3, 4, 5):
        _make_prog(
            Call(op="erase_by_max_size", args=(v0,
                Literal(value=threshold, typ=Type.INT))),
            f"erase_by_max_size({threshold})",
        )
        _make_prog(
            Call(op="keep_by_min_size", args=(v0,
                Literal(value=threshold, typ=Type.INT))),
            f"keep_by_min_size({threshold})",
        )

    # 6. Extract markers (remove large objects, keep small)
    _make_prog(
        Call(op="extract_markers", args=(v0,)),
        "extract_markers",
    )

    # 7. Compact to origin (shift content)
    _make_prog(
        Call(op="compact_to_origin", args=(v0,)),
        "compact_to_origin",
    )

    # 8. Crop operations
    _make_prog(
        Call(op="crop_to_content", args=(v0,)),
        "crop_to_content",
    )
    _make_prog(
        Call(op="crop_frame_interior", args=(v0,)),
        "crop_frame_interior",
    )

    # 9. Keep single color (for each output color)
    for c in all_output_colors:
        if c == 0:
            continue
        _make_prog(
            Call(op="keep_color", args=(v0,
                Literal(value=c, typ=Type.INT))),
            f"keep_color({c})",
        )
        _make_prog(
            Call(op="erase_color", args=(v0,
                Literal(value=c, typ=Type.INT))),
            f"erase_color({c})",
        )

    # 10. Geometric transforms
    for code in range(-1, 8):
        _make_prog(
            Call(op="apply_geometric_transform", args=(v0,
                Literal(value=code, typ=Type.INT))),
            f"apply_geometric_transform({code})",
        )

    # 11. Fill enclosed auto
    _make_prog(
        Call(op="fill_enclosed_regions_auto", args=(v0,)),
        "fill_enclosed_regions_auto",
    )

    # 12. Propagate for each color pair
    for wrong_c, expected_c in residual_colors:
        if wrong_c == 0:
            # Need to fill bg with expected color, propagating from source
            for src_c in all_output_colors:
                if src_c == expected_c or src_c == 0:
                    continue
                _make_prog(
                    Call(op="propagate", args=(v0,
                        Literal(value=src_c, typ=Type.INT),
                        Literal(value=expected_c, typ=Type.INT),
                        Literal(value=0, typ=Type.INT))),
                    f"propagate(src={src_c},fill={expected_c},bg=0)",
                )

    # 13. Repair operations
    _make_prog(
        Call(op="repair_framed_2d_motif", args=(v0,)),
        "repair_framed_2d_motif",
    )
    for axis in (0, 1):
        for period in (2, 3, 4, 5):
            _make_prog(
                Call(op="repair_periodic", args=(v0,
                    Literal(value=axis, typ=Type.INT),
                    Literal(value=period, typ=Type.INT))),
                f"repair_periodic(axis={axis},period={period})",
            )

    # 14. Peel frame
    _make_prog(
        Call(op="peel_frame", args=(v0,)),
        "peel_frame",
    )

    # 15. Detect frame
    _make_prog(
        Call(op="detect_frame", args=(v0,)),
        "detect_frame",
    )

    # 16. Fill where neighbor count
    for fc in fill_colors:
        for nc in all_output_colors:
            if nc == 0 or nc == fc:
                continue
            for mc in (1, 2, 3):
                _make_prog(
                    Call(op="fill_where_neighbor_count", args=(v0,
                        Literal(value=nc, typ=Type.INT),
                        Literal(value=mc, typ=Type.INT),
                        Literal(value=fc, typ=Type.INT))),
                    f"fill_where_neighbor_count(nb={nc},min={mc},fill={fc})",
                )

    # 17. Select and separate
    for thresh in (1, 2, 3, 5, 10):
        _make_prog(
            Call(op="select_and_separate", args=(v0,
                Literal(value=thresh, typ=Type.INT))),
            f"select_and_separate({thresh})",
        )

    # 18. Repair masked region
    for tc in range(8):
        _make_prog(
            Call(op="repair_masked_region", args=(v0,
                Literal(value=tc, typ=Type.INT))),
            f"repair_masked_region({tc})",
        )

    return programs


# ---------------------------------------------------------------------------
# Part C: Compose and verify
# ---------------------------------------------------------------------------


def compose_and_search(
    demos: tuple[DemoPair, ...],
    *,
    max_seeds: int = 10,
    search_budget: int = 500,
) -> CompositionResult:
    """Run the full depth-2 near-miss composition search.

    1. Collect depth-1 near-misses
    2. For each seed, generate residual-local step-2 corrections
    3. Verify each composition on all demos
    """
    near_misses = collect_depth1_near_misses(
        demos, max_seeds=max_seeds, search_budget=search_budget,
    )

    if not near_misses:
        return CompositionResult(solved=False, candidates_tested=0)

    total_tested = 0
    best_residual: int | None = None

    for seed in near_misses:
        corrections = _generate_step2_corrections(seed, demos)

        for prog, desc in corrections:
            total_tested += 1
            vr = verify(prog, demos)
            if vr.passed:
                return CompositionResult(
                    solved=True,
                    winning_program=prog,
                    winning_seed_text=seed.program_text,
                    winning_step2_desc=desc,
                    candidates_tested=total_tested,
                    best_residual=0,
                    seeds_tried=near_misses.index(seed) + 1,
                )

            # Track best residual for reporting
            if vr.diff and isinstance(vr.diff, dict):
                pdc = vr.diff.get("pixel_diff_count")
                if pdc is not None:
                    if best_residual is None or pdc < best_residual:
                        best_residual = pdc

    return CompositionResult(
        solved=False,
        candidates_tested=total_tested,
        best_residual=best_residual,
        seeds_tried=len(near_misses),
    )
