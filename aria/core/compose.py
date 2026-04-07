"""Bounded two-stage pipeline composition.

Composes at most two existing lane operations in sequence, gated by
structural evidence. Tries a small ranked set of compositions, not
all pairs.

Allowed compositions (justified by composition audit):
  periodic_repair -> relocate_objects
  periodic_repair -> replicate_templates
  periodic_repair -> grid_transform (rotate/reflect/fill)

Each composition produces a two-step Program verified exactly.
Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from aria.types import Bind, Call, DemoPair, Grid, Literal, Program, Ref, Type


@dataclass(frozen=True)
class CompositionResult:
    """Result of trying a two-stage composition."""
    solved: bool
    program: Program | None = None
    stage1: str = ""
    stage2: str = ""
    description: str = ""
    diff: int = -1


def try_compositions(
    demos: tuple[DemoPair, ...],
    *,
    task_id: str = "",
    max_attempts: int = 30,
) -> CompositionResult:
    """Try bounded two-stage compositions, evidence-gated.

    Only tries compositions justified by structural evidence.
    Returns the first verifying composition, or the best near-miss.
    """
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.runtime.ops import has_op
    from aria.runtime.executor import execute
    from aria.verify.verifier import verify

    ev, ranking = compute_evidence_and_rank(demos)

    # Build the candidate list, gated by evidence
    candidates: list[tuple[str, Program]] = []

    # Stage 1 candidates: region isolation (highest priority for dims-change)
    if not ev.same_dims:
        if has_op("crop_to_content"):
            s1 = Bind("s1", Type.GRID, Call("crop_to_content", (Ref("input"),)))
            candidates.append(("crop_to_content", Program(steps=(s1,), output="s1")))

        if has_op("crop_frame_interior"):
            s1 = Bind("s1", Type.GRID, Call("crop_frame_interior", (Ref("input"),)))
            candidates.append(("crop_frame_interior", Program(steps=(s1,), output="s1")))

        if has_op("crop_to_content") and has_op("relocate_objects"):
            for mr in (0, 6):
                prog = Program(
                    steps=(
                        Bind("s1", Type.GRID, Call("crop_to_content", (Ref("input"),))),
                        Bind("v0", Type.GRID, Call("relocate_objects", (
                            Ref("s1"), Literal(mr, Type.INT), Literal(0, Type.INT),
                        ))),
                    ),
                    output="v0",
                )
                candidates.append((f"crop_to_content->relocate(mr={mr})", prog))

        if has_op("crop_to_content") and ev.output_grows_shapes and has_op("replicate_templates"):
            prog = Program(
                steps=(
                    Bind("s1", Type.GRID, Call("crop_to_content", (Ref("input"),))),
                    Bind("v0", Type.GRID, Call("replicate_templates", (
                        Ref("s1"), Literal(0, Type.INT),
                        Literal(0, Type.INT), Literal(0, Type.INT),
                    ))),
                ),
                output="v0",
            )
            candidates.append(("crop_to_content->replicate", prog))

    # Same-dims framed: crop_frame_interior alone
    if ev.same_dims and ev.has_framed_region and has_op("crop_frame_interior"):
        s1 = Bind("s1", Type.GRID, Call("crop_frame_interior", (Ref("input"),)))
        candidates.append(("crop_frame_interior", Program(steps=(s1,), output="s1")))

    # Stage 1 candidates: subset filtering (erase singletons, keep shapes)
    if ev.n_input_singles > 0 and has_op("erase_by_max_size"):
        # Erase singletons, then try downstream
        s1 = Bind("s1", Type.GRID, Call("erase_by_max_size", (Ref("input"), Literal(1, Type.INT))))
        candidates.append(("erase_singletons", Program(steps=(s1,), output="s1")))

        # erase_singletons -> relocation
        if has_op("relocate_objects"):
            prog = Program(
                steps=(
                    Bind("s1", Type.GRID, Call("erase_by_max_size", (Ref("input"), Literal(1, Type.INT)))),
                    Bind("v0", Type.GRID, Call("relocate_objects", (
                        Ref("s1"), Literal(0, Type.INT), Literal(0, Type.INT),
                    ))),
                ),
                output="v0",
            )
            candidates.append(("erase_singletons->relocate", prog))

    if ev.n_input_shapes > 0 and has_op("keep_by_min_size"):
        s1 = Bind("s1", Type.GRID, Call("keep_by_min_size", (Ref("input"), Literal(1, Type.INT))))
        candidates.append(("keep_shapes_only", Program(steps=(s1,), output="s1")))

    # Stage 1 candidates: normalize position (same dims)
    if ev.same_dims and has_op("normalize_to_grid"):
        s1 = Bind("s1", Type.GRID, Call("normalize_to_grid", (Ref("input"),)))
        candidates.append(("normalize_to_grid", Program(steps=(s1,), output="s1")))

    if not ev.same_dims and has_op("compact_to_origin"):
        s1 = Bind("s1", Type.GRID, Call("compact_to_origin", (Ref("input"),)))
        candidates.append(("compact_to_origin", Program(steps=(s1,), output="s1")))

    # Stage 1 candidates: periodic repair (if framed)
    if ev.has_framed_region and has_op("periodic_repair"):
        for axis in (0, 1):
            for period in (2, 3, 4):
                for mode in (0, 1, 2):
                    s1_label = f"periodic(axis={axis},period={period},mode={mode})"
                    s1_prog = Program(
                        steps=(Bind("s1", Type.GRID, Call("periodic_repair", (
                            Ref("input"), Literal(axis, Type.INT),
                            Literal(period, Type.INT), Literal(mode, Type.INT),
                        ))),),
                        output="s1",
                    )

                    # Stage 2: relocation
                    if ev.n_input_singles > 0 and has_op("relocate_objects"):
                        for mr in (0, 1, 6):  # shape_nearest, marker_nearest, mutual_nearest
                            for al in (0, 1, 6):  # center, at_marker, marker_interior
                                prog = Program(
                                    steps=(
                                        Bind("s1", Type.GRID, Call("periodic_repair", (
                                            Ref("input"), Literal(axis, Type.INT),
                                            Literal(period, Type.INT), Literal(mode, Type.INT),
                                        ))),
                                        Bind("v0", Type.GRID, Call("relocate_objects", (
                                            Ref("s1"), Literal(mr, Type.INT), Literal(al, Type.INT),
                                        ))),
                                    ),
                                    output="v0",
                                )
                                candidates.append((f"{s1_label}->relocate(mr={mr},al={al})", prog))

                    # Stage 2: replication
                    if ev.output_grows_shapes and has_op("replicate_templates"):
                        for kr in (0, 1):
                            prog = Program(
                                steps=(
                                    Bind("s1", Type.GRID, Call("periodic_repair", (
                                        Ref("input"), Literal(axis, Type.INT),
                                        Literal(period, Type.INT), Literal(mode, Type.INT),
                                    ))),
                                    Bind("v0", Type.GRID, Call("replicate_templates", (
                                        Ref("s1"), Literal(kr, Type.INT),
                                        Literal(0, Type.INT), Literal(0, Type.INT),
                                    ))),
                                ),
                                output="v0",
                            )
                            candidates.append((f"{s1_label}->replicate(kr={kr})", prog))

    # Cap the candidate list
    if len(candidates) > max_attempts:
        candidates = candidates[:max_attempts]

    # Try each composition, with early consensus filtering for two-stage programs
    best_diff = float("inf")
    best_desc = ""

    # Cache stage-1 outputs to avoid re-execution for two-stage candidates
    from aria.core.scene_solve import get_consensus_enabled
    _consensus_on = get_consensus_enabled()
    _stage1_cache: dict[str, list[Grid | None]] = {}

    for desc, prog in candidates:
        try:
            # For two-stage compositions, check stage-1 output consistency
            if _consensus_on and "->" in desc:
                s1_key = desc.split("->")[0]
                if s1_key not in _stage1_cache:
                    s1_outputs: list[Grid | None] = []
                    for d in demos:
                        try:
                            # Execute just the first step
                            s1_result = execute(
                                Program(steps=prog.steps[:1], output=prog.steps[0].name),
                                d.input, None,
                            )
                            s1_outputs.append(s1_result)
                        except Exception:
                            s1_outputs.append(None)
                    _stage1_cache[s1_key] = s1_outputs

                s1_outs = _stage1_cache[s1_key]
                # Consensus: if stage-1 fails on any demo, skip all stage-2 variants
                if any(o is None for o in s1_outs):
                    continue

            vr = verify(prog, demos)
            if vr.passed:
                return CompositionResult(
                    solved=True, program=prog,
                    stage1=desc.split("->")[0],
                    stage2=desc.split("->")[1] if "->" in desc else "",
                    description=desc, diff=0,
                )
            diff = sum(int(np.sum(execute(prog, d.input) != d.output)) for d in demos)
            if diff < best_diff:
                best_diff = diff
                best_desc = desc
        except Exception:
            pass

    return CompositionResult(
        solved=False,
        description=best_desc or "no composition tried",
        diff=int(best_diff) if best_diff < float("inf") else -1,
    )
