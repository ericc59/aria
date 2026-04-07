"""Fitter seed format and repair loop.

Fitters produce typed intermediate hypotheses (seeds) that feed:
1. Direct compile+verify (existing path, kept intact)
2. Seed-guided repair: small modifications to near-miss programs
3. Seed-guided composition: 2-step programs built around the seed's decomposition

Seeds are explicit and inspectable. They carry the decomposition hypothesis,
scope, candidate op, and the near-miss residual if direct verification failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.types import DemoPair, Grid, Program, Bind, Call, Literal, Ref, Type


# ---------------------------------------------------------------------------
# Seed format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitterSeed:
    """A typed intermediate hypothesis from a fitter."""
    decomposition: str       # "frame_interior", "periodic_repair", "grid_transform", etc.
    scope: str               # "frame_interior", "full_grid", "partition_cell", etc.
    candidate_op: str        # op name that was tried
    candidate_args: tuple    # literal args to the op
    confidence: float        # 0-1
    near_miss_program: Program | None  # the compiled program if it exists
    near_miss_diff: int | None  # pixel diff if near-miss
    reason: str              # why this seed was emitted


# ---------------------------------------------------------------------------
# Seed-guided repair
# ---------------------------------------------------------------------------


def repair_from_seed(
    seed: FitterSeed,
    demos: tuple[DemoPair, ...],
    max_repair_candidates: int = 20,
) -> Program | None:
    """Try small modifications to a near-miss seed program.

    Repair strategies:
    1. Add a color-map step after the seed op
    2. Try the same op with different parameters
    3. Chain two ops: seed op + a complementary op
    """
    from aria.verify.verifier import verify

    if seed.near_miss_program is None:
        return None

    # Strategy 1: seed op + global color map
    # Find what colors are still wrong after the seed op
    repair_maps = _infer_residual_color_maps(seed.near_miss_program, demos)
    for cm_pairs in repair_maps:
        prog = _chain_with_color_map(seed.near_miss_program, cm_pairs)
        try:
            vr = verify(prog, demos)
            if vr.passed:
                return prog
        except Exception:
            continue

    # Strategy 2: seed op + fill_enclosed
    for fill_color in range(10):
        prog = _chain_with_fill(seed.near_miss_program, fill_color)
        try:
            vr = verify(prog, demos)
            if vr.passed:
                return prog
        except Exception:
            continue

    # Strategy 3: fill_enclosed + seed op (reversed order)
    if seed.candidate_op == "apply_global_color_map":
        for fill_color in range(10):
            prog = _prepend_fill(seed.candidate_op, seed.candidate_args, fill_color)
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    return prog
            except Exception:
                continue

    return None


def _infer_residual_color_maps(
    program: Program,
    demos: tuple[DemoPair, ...],
) -> list[list[tuple[int, int]]]:
    """Find color maps that would fix the residual of a near-miss program."""
    from aria.verify.trace import traced_execute

    # Collect all residual (got, expected) pairs
    residual_pairs: dict[int, set[int]] = {}
    for d in demos:
        try:
            out, _ = traced_execute(program, d.input, None, d.output)
            if out is None:
                return []
            diff = out != d.output
            for r, c in zip(*np.where(diff)):
                got = int(out[r, c])
                expected = int(d.output[r, c])
                residual_pairs.setdefault(got, set()).add(expected)
        except Exception:
            return []

    # Build consistent repair maps
    maps = []
    # Simple: one-to-one residual color map
    simple_map = []
    consistent = True
    for got, expected_set in residual_pairs.items():
        if len(expected_set) == 1:
            simple_map.append((got, next(iter(expected_set))))
        else:
            consistent = False

    if consistent and simple_map:
        maps.append(simple_map)

    return maps


def _chain_with_color_map(
    base_program: Program,
    color_pairs: list[tuple[int, int]],
) -> Program:
    """Append a global color map step to the base program."""
    n_pairs = len(color_pairs)
    padded = list(color_pairs) + [(-1, -1)] * (10 - n_pairs)
    cm_args = [Ref(base_program.output), Literal(n_pairs, Type.INT)]
    for fc, tc in padded:
        cm_args.extend([Literal(fc, Type.INT), Literal(tc, Type.INT)])

    return Program(
        steps=base_program.steps + (
            Bind("v_repair", Type.GRID, Call("apply_global_color_map", tuple(cm_args))),
        ),
        output="v_repair",
    )


def _chain_with_fill(base_program: Program, fill_color: int) -> Program:
    """Append a fill_enclosed_regions step to the base program."""
    return Program(
        steps=base_program.steps + (
            Bind("v_fill", Type.GRID, Call("fill_enclosed_regions",
                (Ref(base_program.output), Literal(fill_color, Type.INT)))),
        ),
        output="v_fill",
    )


def _prepend_fill(op_name: str, op_args: tuple, fill_color: int) -> Program:
    """Prepend a fill step before the main op."""
    fill_step = Bind("v_prefill", Type.GRID, Call("fill_enclosed_regions",
        (Ref("input"), Literal(fill_color, Type.INT))))

    # Rebuild the main op to read from v_prefill instead of input
    new_args = []
    for arg in op_args:
        if isinstance(arg, Ref) and arg.name == "input":
            new_args.append(Ref("v_prefill"))
        else:
            new_args.append(arg)

    main_step = Bind("v_main", Type.GRID, Call(op_name, tuple(new_args)))
    return Program(steps=(fill_step, main_step), output="v_main")


# ---------------------------------------------------------------------------
# Collect seeds from fitters
# ---------------------------------------------------------------------------


def collect_fitter_seeds(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> list[FitterSeed]:
    """Run all fitters and collect seeds (both direct-wins and near-misses)."""
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess
    from aria.verify.trace import traced_execute
    from aria.core.grid_perception import perceive_grid
    from aria.decomposition import detect_bg

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    seeds: list[FitterSeed] = []

    # Frame-interior seeds
    if all(d.input.shape == d.output.shape for d in demos):
        states = [perceive_grid(d.input) for d in demos]
        if all(s.framed_regions for s in states):
            # Check change concentration
            all_in_frame = True
            for d, s in zip(demos, states):
                diff = d.input != d.output
                n_ch = int(np.sum(diff))
                if n_ch == 0:
                    continue
                in_frame = sum(
                    int(np.sum(diff[fr.row:fr.row+fr.height, fr.col:fr.col+fr.width]))
                    for fr in s.framed_regions
                )
                if in_frame < n_ch * 0.5:
                    all_in_frame = False

            if all_in_frame:
                # Try fill_enclosed candidates
                diff_colors = set()
                for d in demos:
                    diff = d.input != d.output
                    if np.any(diff):
                        for r, c in zip(*np.where(diff)):
                            diff_colors.add(int(d.output[r, c]))

                for fc in sorted(diff_colors):
                    prog = Program(
                        steps=(Bind("v0", Type.GRID, Call("fill_enclosed_regions",
                            (Ref("input"), Literal(fc, Type.INT)))),),
                        output="v0",
                    )
                    try:
                        vr = verifier.verify(prog, demos)
                        if vr.passed:
                            seeds.append(FitterSeed(
                                decomposition="frame_interior",
                                scope="frame_interior",
                                candidate_op="fill_enclosed_regions",
                                candidate_args=(fc,),
                                confidence=1.0,
                                near_miss_program=prog,
                                near_miss_diff=0,
                                reason="direct verify",
                            ))
                            return seeds  # direct win

                        # Measure residual
                        total_diff = 0
                        for d in demos:
                            out, _ = traced_execute(prog, d.input, None, d.output)
                            if out is not None:
                                total_diff += int(np.sum(out != d.output))
                        if total_diff > 0:
                            seeds.append(FitterSeed(
                                decomposition="frame_interior",
                                scope="frame_interior",
                                candidate_op="fill_enclosed_regions",
                                candidate_args=(fc,),
                                confidence=max(0, 1.0 - total_diff / 100),
                                near_miss_program=prog,
                                near_miss_diff=total_diff,
                                reason=f"near-miss: {total_diff} pixels off",
                            ))
                    except Exception:
                        pass

                # Also try fill_enclosed_auto
                prog = Program(
                    steps=(Bind("v0", Type.GRID, Call("fill_enclosed_regions_auto",
                        (Ref("input"),))),),
                    output="v0",
                )
                try:
                    total_diff = 0
                    for d in demos:
                        out, _ = traced_execute(prog, d.input, None, d.output)
                        if out is not None:
                            total_diff += int(np.sum(out != d.output))
                    if total_diff > 0:
                        seeds.append(FitterSeed(
                            decomposition="frame_interior",
                            scope="frame_interior",
                            candidate_op="fill_enclosed_regions_auto",
                            candidate_args=(),
                            confidence=max(0, 1.0 - total_diff / 100),
                            near_miss_program=prog,
                            near_miss_diff=total_diff,
                            reason=f"near-miss: {total_diff} pixels off",
                        ))
                except Exception:
                    pass

    # Sort seeds by confidence (best first)
    seeds.sort(key=lambda s: (-s.confidence, s.near_miss_diff or 0))
    return seeds


# ---------------------------------------------------------------------------
# Seed-guided solve
# ---------------------------------------------------------------------------


def solve_from_seeds(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> tuple[Program | None, list[FitterSeed]]:
    """Collect seeds, try direct verify, then repair near-misses.

    Returns (winning_program, seeds_collected).
    """
    seeds = collect_fitter_seeds(demos, task_id)

    # Direct wins
    for seed in seeds:
        if seed.near_miss_diff == 0 and seed.near_miss_program is not None:
            return seed.near_miss_program, seeds

    # Repair near-misses (best first)
    for seed in seeds[:5]:  # top 5 seeds only
        if seed.near_miss_program is not None and seed.near_miss_diff is not None:
            repaired = repair_from_seed(seed, demos)
            if repaired is not None:
                return repaired, seeds

    return None, seeds
