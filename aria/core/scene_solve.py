"""Scene program inference and verification.

Infers multi-step scene programs from demo pairs and verifies them.
Uses generic selection, correspondence, and map-over operators rather
than bespoke per-family code.

Stepwise all-demo consensus: before building/verifying full programs,
checks cross-demo structural consistency at each partial step to
prune branches that cannot represent one shared rule.
"""

from __future__ import annotations

import numpy as np

from aria.consensus import (
    build_initial_branch,
    should_prune,
    try_select_on_demo,
    update_branch_after_select,
)
from aria.consensus_trace import ConsensusTrace
from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.core.scene_executor import (
    _MAP_PROPERTY_EXTRACTORS,
    _SELECTOR_PREDICATES,
    execute_scene_program,
    make_scene_program,
    make_step,
)
from aria.scene_ir import EntityKind, SceneProgram, StepOp
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Consensus toggle — disable for A/B comparison
# ---------------------------------------------------------------------------

_CONSENSUS_ENABLED = True
_CONSENSUS_PRUNE_COUNT = 0
_VERIFY_CALL_COUNT = 0


def set_consensus_enabled(enabled: bool) -> None:
    """Enable or disable consensus gating globally for A/B comparison."""
    global _CONSENSUS_ENABLED
    _CONSENSUS_ENABLED = enabled


def get_consensus_enabled() -> bool:
    return _CONSENSUS_ENABLED


def reset_consensus_counters() -> None:
    """Reset prune and verify counters for A/B measurement."""
    global _CONSENSUS_PRUNE_COUNT, _VERIFY_CALL_COUNT
    _CONSENSUS_PRUNE_COUNT = 0
    _VERIFY_CALL_COUNT = 0


def get_consensus_counters() -> dict[str, int]:
    """Return current prune and verify counts."""
    return {
        "pruned": _CONSENSUS_PRUNE_COUNT,
        "verified": _VERIFY_CALL_COUNT,
    }


# ---------------------------------------------------------------------------
# Consensus-aware perception cache
# ---------------------------------------------------------------------------

def _perceive_all_demos(
    demos: tuple[DemoPair, ...],
) -> tuple[GridPerceptionState, ...]:
    """Perceive all demo inputs and return cached states."""
    return tuple(perceive_grid(d.input) for d in demos)


def _consensus_select_check(
    perceptions: tuple[GridPerceptionState, ...],
    kind: str,
    predicate: str,
    rank: int,
    *,
    color_filter: int | None = None,
    trace: ConsensusTrace | None = None,
) -> bool:
    """Check if a selector is cross-demo consistent without building a program.

    Returns True if the selector can find entities in all demos.
    Returns False (prune) if any demo has no entity for this selector.
    When consensus is disabled, always returns True (no pruning).
    """
    if not _CONSENSUS_ENABLED:
        return True

    results = [
        try_select_on_demo(p, kind, predicate, rank, color_filter=color_filter)
        for p in perceptions
    ]

    # Hard prune: selector doesn't find anything in some demo
    if not all(r.found for r in results):
        global _CONSENSUS_PRUNE_COUNT
        _CONSENSUS_PRUNE_COUNT += 1
        if trace is not None:
            branch = build_initial_branch(list(perceptions))
            branch = update_branch_after_select(
                branch, kind, predicate, rank, results,
            )
            trace.record(
                branch,
                f"select({kind}, {predicate}, rank={rank}) — pruned",
            )
        return False

    return True


def infer_scene_programs(
    demos: tuple[DemoPair, ...],
    *,
    consensus_trace: ConsensusTrace | None = None,
    retrieval_guidance: object | None = None,
) -> tuple[SceneProgram, ...]:
    """Try to infer multi-step scene programs that explain all demos.

    Uses stepwise all-demo consensus to prune selector/transform
    combinations that are structurally inconsistent across demos
    before building and verifying full programs.
    """
    if not demos:
        return ()

    # Perceive all demos once — shared across all families
    perceptions = _perceive_all_demos(demos)
    trace = consensus_trace or ConsensusTrace()

    candidates: list[SceneProgram] = []

    # Family 1: select entity + extract + transform
    prog = _try_select_extract_transform(demos, perceptions, trace)
    if prog is not None:
        candidates.append(prog)

    # Family 2: select entity + extract + color map
    prog = _try_select_extract_colormap(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 3: map over panels → summary grid (using generic MAP_OVER_ENTITIES)
    prog = _try_map_over_panels_summary(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 4: boolean combine panels
    prog = _try_boolean_combine(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 5: select panel by predicate + extract
    prog = _try_select_panel_extract(demos, perceptions, trace)
    if prog is not None:
        candidates.append(prog)

    # Family 6: select color_bbox entity + optional transform (dynamic output size)
    prog = _try_color_bbox_select(demos, perceptions, trace)
    if prog is not None:
        candidates.append(prog)

    # Family 7: per-cell partition operations (FOR_EACH over cells)
    prog = _try_per_cell_operation(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 8: per-object operations (FOR_EACH over objects)
    prog = _try_per_object_operation(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 9: combine cells → broadcast to all cells
    prog = _try_combine_broadcast(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 10: combine cells → cell-sized output
    prog = _try_combine_to_output(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 11: select sibling cell by property → cell-sized output
    prog = _try_sibling_cell_select(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 12: cell assignment / permutation
    prog = _try_cell_assignment(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 13: per-cell recolor (inferred from demo 0)
    prog = _try_per_cell_recolor(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 14: consensus-controlled compositional search
    try:
        from aria.consensus_search import consensus_compose_search
        compose_results = consensus_compose_search(
            demos, trace=trace,
        )
        for desc, prog, score in compose_results:
            candidates.append(prog)
    except Exception:
        pass

    # Family 15: factorized composition search
    try:
        from aria.core.factor_search import factor_composition_search
        factor_results = factor_composition_search(
            demos, consensus_trace=trace,
            retrieval_guidance=retrieval_guidance,
        )
        for factors, prog in factor_results:
            candidates.append(prog)
    except Exception:
        pass

    # Family 14: sub-region extraction (output == crop of input)
    prog = _try_subregion_extract(demos)
    if prog is not None:
        candidates.append(prog)

    # Family 16: role-based per-object conditional programs
    progs = _try_role_based_per_object(demos)
    for prog in progs:
        candidates.append(prog)

    # Family 17: observed cell-to-output synthesis (non-uniform ok)
    progs = _try_observed_cell_synthesis(demos)
    for prog in progs:
        candidates.append(prog)

    # Family 18: generalized periodicity synthesis
    progs = _try_periodicity_synthesis(demos)
    for prog in progs:
        candidates.append(prog)

    return tuple(candidates)


def verify_scene_program(
    program,
    demos: tuple[DemoPair, ...],
) -> bool:
    """Verify a scene program produces correct output on all demos."""
    global _VERIFY_CALL_COUNT
    _VERIFY_CALL_COUNT += 1
    all_dims_ok = True
    total_diff = 0
    total_pixels = 0
    for d in demos:
        try:
            # Support CorrespondenceProgram: verify by re-running
            # correspondence search on this demo's input/output pair
            if hasattr(program, 'verify_on_demo'):
                if not program.verify_on_demo(d.input, d.output):
                    _record_near_miss(program, False)
                    return False
                continue
            elif hasattr(program, 'execute'):
                result = program.execute(d.input)
            else:
                result = execute_scene_program(program, d.input)
        except Exception:
            _record_near_miss(program, False)
            return False
        if result.shape != d.output.shape:
            all_dims_ok = False
            _record_near_miss(program, False)
            return False
        diff = int(np.sum(result != d.output))
        total_diff += diff
        total_pixels += d.output.size
        if diff > 0:
            # Don't return early — continue to collect full diff info
            pass

    if total_diff == 0:
        return True

    # Near-miss: dims correct but some pixels wrong
    if all_dims_ok and total_pixels > 0:
        accuracy = (total_pixels - total_diff) / total_pixels
        _record_near_miss(program, accuracy >= 0.5)

    return False


# ---------------------------------------------------------------------------
# Near-miss sink — collects unverified candidates during inference
# ---------------------------------------------------------------------------

_NEAR_MISS_SINK: list[SceneProgram] | None = None


def _record_near_miss(program: SceneProgram, is_near_miss: bool) -> None:
    """Record a near-miss candidate if the sink is active."""
    if _NEAR_MISS_SINK is not None and is_near_miss:
        _NEAR_MISS_SINK.append(program)


def infer_and_repair_scene_programs(
    demos: tuple[DemoPair, ...],
    *,
    consensus_trace: ConsensusTrace | None = None,
) -> tuple[SceneProgram, ...]:
    """Infer scene programs, then attempt repair on near-misses.

    Wraps infer_scene_programs(). If no exact match is found, scores
    the near-miss candidates collected during inference and attempts
    bounded local repair.
    """
    global _NEAR_MISS_SINK

    # Enable near-miss collection
    _NEAR_MISS_SINK = []
    try:
        verified = infer_scene_programs(demos, consensus_trace=consensus_trace)
    finally:
        near_misses = _NEAR_MISS_SINK
        _NEAR_MISS_SINK = None

    if verified:
        return verified

    # No exact match — try repair on near-misses
    if not near_misses:
        return ()

    from aria.repair import repair_near_misses

    # Deduplicate by step signature
    seen: set[tuple] = set()
    unique: list[SceneProgram] = []
    for prog in near_misses:
        sig = prog.step_names()
        if sig not in seen:
            seen.add(sig)
            unique.append(prog)

    result = repair_near_misses(unique, demos)
    if result.solved and result.winning_program is not None:
        return (result.winning_program,)

    return ()


# ---------------------------------------------------------------------------
# Family 1: select entity by predicate + extract + geometric transform
# ---------------------------------------------------------------------------

_TRANSFORMS = [
    "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose",
]


def _try_select_extract_transform(
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...] | None = None,
    trace: ConsensusTrace | None = None,
) -> SceneProgram | None:
    """parse → select entity → extract → transform → render.

    Uses stepwise consensus to skip (kind, predicate, rank) combos
    where the selector cannot find entities in all demos.
    """
    if perceptions is None:
        perceptions = _perceive_all_demos(demos)
    s0 = perceptions[0]

    # Try selecting from different entity kinds
    kinds_to_try = []
    if s0.objects.objects:
        kinds_to_try.append("object")
    if s0.framed_regions:
        kinds_to_try.append("interior_region")
        kinds_to_try.append("boundary")

    predicates = [
        "largest_bbox_area", "smallest_bbox_area", "most_non_bg",
        "top_left", "bottom_right", "unique_shape", "not_touches_border",
        "unique_shape_largest", "unique_shape_most_pixels",
        "largest_pixel_count",
        "isolated", "isolated_largest", "isolated_most_pixels",
        "inside_frame", "inside_frame_largest",
    ]

    for kind in kinds_to_try:
        for predicate in predicates:
            for rank in range(min(3, len(s0.objects.objects) if kind == "object" else 3)):
                # Consensus gate: check selector across all demos
                if not _consensus_select_check(
                    perceptions, kind, predicate, rank, trace=trace,
                ):
                    continue  # prune this (kind, predicate, rank) combo

                # Without transform (identity)
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.SELECT_ENTITY,
                        kind=kind,
                        predicate=predicate,
                        rank=rank,
                        output_id="sel",
                    ),
                    make_step(StepOp.RENDER_SCENE, source="sel_grid"),
                )
                if verify_scene_program(prog, demos):
                    return prog

                # With transforms
                for transform in _TRANSFORMS:
                    prog = make_scene_program(
                        make_step(StepOp.PARSE_SCENE),
                        make_step(
                            StepOp.SELECT_ENTITY,
                            kind=kind,
                            predicate=predicate,
                            rank=rank,
                            output_id="sel",
                        ),
                        make_step(
                            StepOp.CANONICALIZE_OBJECT,
                            source="sel_grid",
                            transform=transform,
                            output_id="transformed",
                        ),
                        make_step(StepOp.RENDER_SCENE, source="transformed"),
                    )
                    if verify_scene_program(prog, demos):
                        return prog

    # Also try named entity sources (frame_N_interior, object_N)
    sources = []
    for idx in range(len(s0.framed_regions)):
        sources.append(f"frame_{idx}_interior")
    objs_by_area = sorted(
        enumerate(s0.objects.objects),
        key=lambda x: x[1].bbox_h * x[1].bbox_w,
        reverse=True,
    )
    for idx, _ in objs_by_area[:5]:
        sources.append(f"object_{idx}")

    for source in sources:
        for transform in _TRANSFORMS:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.EXTRACT_TEMPLATE, source=source, output_id="t1"),
                make_step(
                    StepOp.CANONICALIZE_OBJECT,
                    source="t1",
                    transform=transform,
                    output_id="t2",
                ),
                make_step(StepOp.RENDER_SCENE, source="t2"),
            )
            if verify_scene_program(prog, demos):
                return prog

    return None


# ---------------------------------------------------------------------------
# Family 2: select entity + extract + color map
# ---------------------------------------------------------------------------


def _try_select_extract_colormap(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → select → extract → stamp + recolor → render."""
    s0 = perceive_grid(demos[0].input)
    sources = []
    for idx in range(len(s0.framed_regions)):
        sources.append(f"frame_{idx}_interior")
    for idx in range(len(s0.boxed_regions)):
        sources.append(f"boxed_{idx}_interior")

    for source in sources:
        prog_extract = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.EXTRACT_TEMPLATE, source=source, output_id="t1"),
            make_step(StepOp.RENDER_SCENE, source="t1"),
        )
        try:
            extracted = execute_scene_program(prog_extract, demos[0].input)
        except Exception:
            continue
        if extracted.size == 0 or extracted.shape != demos[0].output.shape:
            continue

        cmap = _find_color_map(extracted, demos[0].output)
        if cmap is None or not cmap:
            continue

        steps = [
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.EXTRACT_TEMPLATE, source=source, output_id="t1"),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=demos[0].output.shape),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(StepOp.STAMP_TEMPLATE, source="t1", row=0, col=0),
        ]
        for fc, tc in cmap.items():
            steps.append(make_step(StepOp.RECOLOR_OBJECT, from_color=fc, to_color=tc))
        steps.append(make_step(StepOp.RENDER_SCENE))

        prog = make_scene_program(*steps)
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 3: map over panels → summary grid (generic)
# ---------------------------------------------------------------------------


def _try_map_over_panels_summary(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → build_correspondence → map_over(panel, property) → render."""
    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    oH, oW = demos[0].output.shape
    if (oH, oW) != (s0.partition.n_rows, s0.partition.n_cols):
        return None

    for prop_name in _MAP_PROPERTY_EXTRACTORS:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(
                StepOp.BUILD_CORRESPONDENCE,
                source_kind="panel",
                mode="row_col_index",
                output_id="corr",
            ),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=(oH, oW)),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(
                StepOp.MAP_OVER_ENTITIES,
                kind="panel",
                property=prop_name,
                layout="grid",
                correspondence="corr",
            ),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 4: boolean combine panels
# ---------------------------------------------------------------------------


def _try_boolean_combine(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → boolean combine panels → render."""
    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None
    if not s0.partition.is_uniform_partition:
        return None

    cell_dims = s0.partition.cells[0].dims

    operations: list[str] = ["overlay", "and", "xor", "or"]

    # Also try OR with explicit target color — infer from output palette
    out_colors = set(int(v) for v in np.unique(demos[0].output))
    bg = s0.bg_color
    for c in sorted(out_colors - {bg}):
        operations.append(f"or_color_{c}")

    for operation in operations:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=cell_dims),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(StepOp.BOOLEAN_COMBINE_PANELS, operation=operation),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 5: select panel by predicate + extract
# ---------------------------------------------------------------------------


def _try_select_panel_extract(
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...] | None = None,
    trace: ConsensusTrace | None = None,
) -> SceneProgram | None:
    """parse → select panel by predicate → extract → render.

    Consensus-gated: checks selector consistency across demos first.
    """
    if perceptions is None:
        perceptions = _perceive_all_demos(demos)
    s0 = perceptions[0]
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    predicates = [
        "most_non_bg", "least_non_bg_gt0", "most_colors",
        "largest_bbox_area", "smallest_bbox_area",
        "unique_shape", "not_touches_border",
    ]

    for predicate in predicates:
        for rank in range(min(3, len(s0.partition.cells))):
            # Consensus gate
            if not _consensus_select_check(
                perceptions, "panel", predicate, rank, trace=trace,
            ):
                continue

            # Direct extract
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.SELECT_ENTITY,
                    kind="panel",
                    predicate=predicate,
                    rank=rank,
                    output_id="sel",
                ),
                make_step(StepOp.RENDER_SCENE, source="sel_grid"),
            )
            if verify_scene_program(prog, demos):
                return prog

            # Extract + transform
            for transform in _TRANSFORMS:
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.SELECT_ENTITY,
                        kind="panel",
                        predicate=predicate,
                        rank=rank,
                        output_id="sel",
                    ),
                    make_step(
                        StepOp.CANONICALIZE_OBJECT,
                        source="sel_grid",
                        transform=transform,
                        output_id="transformed",
                    ),
                    make_step(StepOp.RENDER_SCENE, source="transformed"),
                )
                if verify_scene_program(prog, demos):
                    return prog

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Family 9: combine cells → broadcast to all cells
# ---------------------------------------------------------------------------


def _try_combine_broadcast(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → combine_cells → broadcast_to_cells → render.

    Output = input with every cell replaced by the combined cell.
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or not s0.partition.is_uniform_partition:
        return None
    if len(s0.partition.cells) < 2:
        return None

    for operation in ("overlay", "xor", "and", "or_any_color"):
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.COMBINE_CELLS, operation=operation, output_to="value",
                      output_id="combined"),
            make_step(StepOp.BROADCAST_TO_CELLS, source="combined"),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 10: combine cells → cell-sized output
# ---------------------------------------------------------------------------


def _try_combine_to_output(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → combine_cells → render (output = cell-sized combined result)."""
    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or not s0.partition.is_uniform_partition:
        return None
    if len(s0.partition.cells) < 2:
        return None

    cell_dims = s0.partition.cells[0].dims
    if demos[0].output.shape != cell_dims:
        return None

    for operation in ("overlay", "xor", "and", "or_any_color"):
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.COMBINE_CELLS, operation=operation, output_to="output"),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 11: select sibling cell by property → cell-sized output
# ---------------------------------------------------------------------------


def _try_sibling_cell_select(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → select_entity(panel, predicate) → render.

    Output = the partition cell selected by a sibling-relative predicate.
    Uses SELECT_ENTITY with panel kind.
    """
    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    cell_dims = s0.partition.cells[0].dims
    if not s0.partition.is_uniform_partition:
        return None
    if demos[0].output.shape != cell_dims:
        return None

    predicates = [
        "most_non_bg", "least_non_bg_gt0", "most_colors", "fewest_colors_gt0",
        "largest_bbox_area", "smallest_bbox_area", "unique_shape",
    ]

    for pred in predicates:
        for rank in range(min(4, len(s0.partition.cells))):
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.SELECT_ENTITY, kind="panel", predicate=pred,
                          rank=rank, output_id="sel"),
                make_step(StepOp.RENDER_SCENE, source="sel_grid"),
            )
            if verify_scene_program(prog, demos):
                return prog

    return None


# ---------------------------------------------------------------------------
# Family 8: per-object operations (FOR_EACH over objects)
# ---------------------------------------------------------------------------


def _try_per_object_operation(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → for_each_entity(object, rule) → render.

    Handles same-dims tasks where each object's bbox region is modified.
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    # fill_bbox_holes with various colors
    for color in range(10):
        for conn in [4, 8, None]:
            params = {"kind": "object", "rule": "fill_bbox_holes", "fill_color": color}
            if conn is not None:
                params["connectivity"] = conn
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.FOR_EACH_ENTITY, **params),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    # fill_enclosed_bbox with various colors
    for color in range(10):
        for conn in [4, 8, None]:
            params = {"kind": "object", "rule": "fill_enclosed_bbox", "fill_color": color}
            if conn is not None:
                params["connectivity"] = conn
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.FOR_EACH_ENTITY, **params),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    return None


# ---------------------------------------------------------------------------
# Family 6: select color_bbox entity + optional transform (dynamic output size)
# ---------------------------------------------------------------------------


def _try_color_bbox_select(
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...] | None = None,
    trace: ConsensusTrace | None = None,
) -> SceneProgram | None:
    """parse → select color_bbox by color/predicate → optional transform → render.

    Uses dynamic output size from the selected entity's grid.
    Consensus-gated: checks selector consistency across demos first.
    """
    if perceptions is None:
        perceptions = _perceive_all_demos(demos)
    s0 = perceptions[0]
    if not s0.non_bg_colors:
        return None

    # Try each non-bg color as a color filter
    for color in sorted(s0.non_bg_colors):
        predicates = ["largest_bbox_area", "largest_pixel_count"]
        for pred in predicates:
            # Consensus gate: check selector with color_filter across demos
            if not _consensus_select_check(
                perceptions, "object", pred, 0,
                color_filter=color, trace=trace,
            ):
                continue

            # Identity: just extract the color bbox
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.SELECT_ENTITY,
                    kind="object",
                    predicate=pred,
                    color_filter=color,
                    rank=0,
                    output_id="sel",
                ),
                make_step(StepOp.RENDER_SCENE, source="sel_grid"),
            )
            if verify_scene_program(prog, demos):
                return prog

            # With transforms
            for transform in _TRANSFORMS:
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.SELECT_ENTITY,
                        kind="object",
                        predicate=pred,
                        color_filter=color,
                        rank=0,
                        output_id="sel",
                    ),
                    make_step(
                        StepOp.CANONICALIZE_OBJECT,
                        source="sel_grid",
                        transform=transform,
                        output_id="t",
                    ),
                    make_step(StepOp.RENDER_SCENE, source="t"),
                )
                if verify_scene_program(prog, demos):
                    return prog

    return None


# ---------------------------------------------------------------------------
# Family 7: per-cell partition operations (FOR_EACH over cells)
# ---------------------------------------------------------------------------


def _try_per_cell_operation(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → apply_per_cell(rule, filter) → render.

    Handles tasks where the output is the input with a uniform
    per-cell transformation applied to each partition cell.
    """
    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None
    if demos[0].input.shape != demos[0].output.shape:
        return None

    rules = [
        "swap_dominant_with_bg",
        "fill_bg_with_dominant",
        "clear_non_bg",
        "invert_non_bg",
        "fill_enclosed_with_dominant",
        "replace_minority_with_dominant",
    ]
    filters = ["all", "has_non_bg", "has_multiple_non_bg"]

    for rule in rules:
        for cell_filter in filters:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.APPLY_PER_CELL, rule=rule, filter=cell_filter),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    # Also try fill_bg_with_color and fill_enclosed_with_color for each possible color
    for color in range(10):
        for cell_filter in filters:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.APPLY_PER_CELL, rule="fill_enclosed_with_color",
                          fill_color=color, filter=cell_filter),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    for color in range(10):
        for cell_filter in filters:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.APPLY_PER_CELL, rule="fill_bg_with_color",
                          fill_color=color, filter=cell_filter),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    return None


# ---------------------------------------------------------------------------
# Family 12: cell assignment / permutation
# ---------------------------------------------------------------------------


def _try_cell_assignment(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → assign_cells(mode) → render.

    Handles tasks where output cells are rearranged/permuted input cells.
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or not s0.partition.is_uniform_partition:
        return None
    if len(s0.partition.cells) < 2:
        return None

    modes = [
        "sort_by_non_bg_asc", "sort_by_non_bg_desc",
        "sort_by_color_count_asc", "sort_by_color_count_desc",
        "sort_by_dominant_color_asc", "sort_by_dominant_color_desc",
        "rotate_cell_grid_90", "rotate_cell_grid_180", "rotate_cell_grid_270",
        "flip_cell_grid_lr", "flip_cell_grid_ud", "transpose_cell_grid",
        "broadcast_most_non_bg", "broadcast_unique_color",
        "row_broadcast_most_non_bg", "row_broadcast_least_non_bg_gt0", "row_broadcast_most_colors",
        "col_broadcast_most_non_bg", "col_broadcast_least_non_bg_gt0", "col_broadcast_most_colors",
    ]

    for mode in modes:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.ASSIGN_CELLS, mode=mode),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            return prog

    return None


# ---------------------------------------------------------------------------
# Family 13: per-cell recolor (inferred color map from demo 0)
# ---------------------------------------------------------------------------


def _try_per_cell_recolor(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """parse → apply_per_cell(recolor, from_color, to_color, filter) → render.

    Infers a consistent color map from changed cells in demo 0,
    then verifies it works across all demos.
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    s0 = perceive_grid(demos[0].input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    bg = s0.bg_color
    p = s0.partition

    # Find the consistent color map from changed cells
    cmap: dict[int, int] = {}
    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        ic = demos[0].input[r0 : r1 + 1, c0 : c1 + 1]
        oc = demos[0].output[r0 : r1 + 1, c0 : c1 + 1]
        diff = ic != oc
        if not np.any(diff):
            continue
        for r, c in zip(*np.where(diff)):
            iv, ov = int(ic[r, c]), int(oc[r, c])
            if iv in cmap and cmap[iv] != ov:
                return None  # Inconsistent
            cmap[iv] = ov

    if not cmap:
        return None

    # Try each (from, to) pair with various filters
    filters = ["all", "has_non_bg", "has_multiple_non_bg"]
    for from_c, to_c in cmap.items():
        for filt in filters:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.APPLY_PER_CELL, rule="recolor",
                          from_color=from_c, to_color=to_c, filter=filt),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                return prog

    # Try applying ALL cmap pairs together
    if len(cmap) > 1:
        for filt in filters:
            steps = [make_step(StepOp.PARSE_SCENE)]
            for from_c, to_c in cmap.items():
                steps.append(make_step(StepOp.APPLY_PER_CELL, rule="recolor",
                                       from_color=from_c, to_color=to_c, filter=filt))
            steps.append(make_step(StepOp.RENDER_SCENE))
            prog = make_scene_program(*steps)
            if verify_scene_program(prog, demos):
                return prog

    return None


def _find_color_map(source: Grid, target: Grid) -> dict[int, int] | None:
    if source.shape != target.shape:
        return None
    cmap: dict[int, int] = {}
    diff = source != target
    if not np.any(diff):
        return {}
    rows, cols = np.where(diff)
    for r, c in zip(rows, cols):
        sc = int(source[r, c])
        tc = int(target[r, c])
        if sc in cmap and cmap[sc] != tc:
            return None
        cmap[sc] = tc
    return cmap


# ---------------------------------------------------------------------------
# Family 14: sub-region extraction (output == crop of input)
# ---------------------------------------------------------------------------


def _try_subregion_extract(
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """Detect tasks where output is an exact sub-region of input.

    Tries to find the crop using object bboxes, then falls back to
    brute-force sliding window on demo 0 + verification on all demos.
    """
    if not demos:
        return None
    if any(d.input.shape == d.output.shape for d in demos):
        return None  # Same dims → not extraction

    # First: try existing entity selectors (fastest)
    # Already covered by _try_select_extract_transform — skip here

    # Brute force: find output as sub-region in demo 0
    d0 = demos[0]
    out = d0.output
    oH, oW = out.shape
    iH, iW = d0.input.shape
    if oH > iH or oW > iW:
        return None

    # Find position in demo 0
    pos0 = None
    for r in range(iH - oH + 1):
        for c in range(iW - oW + 1):
            if np.array_equal(d0.input[r : r + oH, c : c + oW], out):
                pos0 = (r, c)
                break
        if pos0:
            break

    if pos0 is None:
        return None

    r0, c0 = pos0

    # Check: is the crop position CONSISTENT across demos?
    # Mode 1: same absolute position
    all_fixed = True
    for d in demos[1:]:
        doH, doW = d.output.shape
        if r0 + doH > d.input.shape[0] or c0 + doW > d.input.shape[1]:
            all_fixed = False
            break
        if not np.array_equal(d.input[r0 : r0 + doH, c0 : c0 + doW], d.output):
            all_fixed = False
            break

    if all_fixed:
        # Build a scene program that crops at fixed position
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.EXTRACT_TEMPLATE, source="scene", output_id="full"),
            make_step(StepOp.RENDER_SCENE, source="full"),
        )
        # Can't use fixed position easily with current ops
        # Instead, use the direct-execute approach
        pass

    # Mode 2: position is relative to an object/structure
    # Find what structural feature the crop aligns to
    s0 = perceive_grid(d0.input)

    # Check: does the crop align to a non-singleton object's bbox?
    for conn, objs_list in [(4, s0.objects.non_singletons), (8, s0.objects8.non_singletons)]:
        for obj in objs_list:
            if (obj.bbox_h, obj.bbox_w) != (oH, oW):
                continue
            if obj.row != r0 or obj.col != c0:
                continue
            # Found matching object. Use selector to find it across demos.
            # Already tried by _try_select_extract_transform
            break

    # Check: does the crop align to a framed region interior?
    for fr in s0.framed_regions:
        if (fr.height, fr.width) == (oH, oW) and fr.row == r0 and fr.col == c0:
            pass  # Already tried

    # Check: does the crop align to a boxed region interior?
    for br in s0.boxed_regions:
        if (br.height, br.width) == (oH, oW) and br.row == r0 and br.col == c0:
            pass  # Already tried

    # Check: crop aligns to tight non-bg bbox
    if s0.tight_non_bg_bbox is not None:
        tr0, tc0, tr1, tc1 = s0.tight_non_bg_bbox
        if (r0, c0) == (tr0, tc0) and (oH, oW) == (tr1 - tr0 + 1, tc1 - tc0 + 1):
            # Verify across demos
            all_ok = True
            for d in demos[1:]:
                sp = perceive_grid(d.input)
                if sp.tight_non_bg_bbox is None:
                    all_ok = False
                    break
                dr0, dc0, dr1, dc1 = sp.tight_non_bg_bbox
                region = d.input[dr0 : dr1 + 1, dc0 : dc1 + 1]
                if not np.array_equal(region, d.output):
                    all_ok = False
                    break
            if all_ok:
                # Build program: extract tight non-bg bbox
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.SELECT_ENTITY,
                        kind="object",
                        predicate="largest_bbox_area",
                        rank=0,
                        require_non_bg=True,
                        output_id="sel",
                    ),
                    make_step(StepOp.RENDER_SCENE, source="sel_grid"),
                )
                if verify_scene_program(prog, demos):
                    return prog

    # Mode 3: crop position varies per demo but output is always findable
    # Try each non-singleton object across demos
    for pred in ["largest_bbox_area", "smallest_bbox_area", "most_non_bg",
                  "unique_shape", "largest_pixel_count", "not_singleton"]:
        for rank in range(3):
            for conn_tag in [None, 4, 8]:
                params: dict = {"kind": "object", "predicate": pred, "rank": rank, "output_id": "sel"}
                if conn_tag is not None:
                    # Filter by connectivity — need to use entity attrs
                    pass  # Skip for now, already tried by general search
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(StepOp.SELECT_ENTITY, **params),
                    make_step(StepOp.RENDER_SCENE, source="sel_grid"),
                )
                if verify_scene_program(prog, demos):
                    return prog

    return None


# ---------------------------------------------------------------------------
# Family 16: role-based per-object conditional programs
# ---------------------------------------------------------------------------

# Color roles to try for fill/recolor operations
_COLOR_ROLES = (
    "singleton_color",
    "rarest_non_bg",
    "most_frequent_non_bg",
    "dominant_object_color",
    "minority_object_color",
    "boundary_color",
)

# Role-based rules to try
_ROLE_RULES: list[tuple[str, bool]] = [
    # (rule_name, needs_color_role)
    ("fill_bbox_holes_role", True),
    ("fill_enclosed_role", True),
    ("fill_bg_role", True),
    ("recolor_to_role", True),
    ("recolor_dominant_to_minority", False),
]


def _relevant_guards(demos: tuple[DemoPair, ...]) -> list[str]:
    """Determine which guard predicates are structurally relevant for these demos.

    Only returns guards that make sense given the scene structure, keeping
    enumeration bounded to ~50-100 extra verify calls.
    """
    guards: list[str] = []
    # Always try enclosed-bg guards (common pattern)
    guards.extend(["has_enclosed_bg", "no_enclosed_bg"])

    perceptions = [perceive_grid(d.input) for d in demos]

    has_frames = any(len(p.framed_regions) > 0 for p in perceptions)
    if has_frames:
        guards.extend(["not_frame_like", "is_frame_like"])

    # Check for border-touching objects
    for p in perceptions:
        h, w = p.dims
        for obj in p.objects.objects:
            r0, c0 = obj.row, obj.col
            r1, c1 = r0 + obj.bbox_h - 1, c0 + obj.bbox_w - 1
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                guards.extend(["touches_border", "not_touches_border"])
                break
        else:
            continue
        break

    # Check for multi-color bboxes
    for p in perceptions:
        bg = p.bg_color
        for obj in p.objects.objects:
            r0, c0 = obj.row, obj.col
            r1, c1 = r0 + obj.bbox_h - 1, c0 + obj.bbox_w - 1
            region = demos[0].input[r0:r1 + 1, c0:c1 + 1]  # approximate
            non_bg = set(int(v) for v in region.ravel() if int(v) != bg)
            if len(non_bg) > 1:
                guards.extend(["multi_color_bbox", "single_color_bbox"])
                break
        else:
            continue
        break

    return guards


def _try_role_based_per_object(
    demos: tuple[DemoPair, ...],
) -> list[SceneProgram]:
    """Generate and verify programs using role-based color resolution.

    Handles tasks where:
    - each object needs a different fill/recolor color
    - the color is determined by a structural role (singleton, rarest, etc.)
    - the role is consistent across demos even when literal colors differ

    Also tries global FILL_ENCLOSED_REGIONS with color roles,
    guarded per-object rules, and OBJECT_GROUP iterations.
    """
    if any(d.input.shape != d.output.shape for d in demos):
        return []

    verified: list[SceneProgram] = []

    # --- Priority 1: Global fill enclosed with color role ---
    for role in _COLOR_ROLES:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, color_role=role),
            make_step(StepOp.RENDER_SCENE),
        )
        if verify_scene_program(prog, demos):
            verified.append(prog)
            return verified

    # --- Priority 2: Guarded object rules ---
    guards = _relevant_guards(demos)
    if guards:
        for guard in guards:
            for conn_val in (4, 8, None):
                for rule_name, needs_role in _ROLE_RULES:
                    if needs_role:
                        for role in _COLOR_ROLES:
                            params: dict = {
                                "kind": "object",
                                "rule": rule_name,
                                "color_role": role,
                                "guard": guard,
                            }
                            if conn_val is not None:
                                params["connectivity"] = conn_val
                            prog = make_scene_program(
                                make_step(StepOp.PARSE_SCENE),
                                make_step(StepOp.FOR_EACH_ENTITY, **params),
                                make_step(StepOp.RENDER_SCENE),
                            )
                            if verify_scene_program(prog, demos):
                                verified.append(prog)
                                return verified
                    else:
                        params = {
                            "kind": "object",
                            "rule": rule_name,
                            "guard": guard,
                        }
                        if conn_val is not None:
                            params["connectivity"] = conn_val
                        prog = make_scene_program(
                            make_step(StepOp.PARSE_SCENE),
                            make_step(StepOp.FOR_EACH_ENTITY, **params),
                            make_step(StepOp.RENDER_SCENE),
                        )
                        if verify_scene_program(prog, demos):
                            verified.append(prog)
                            return verified

    # --- Priority 3: OBJECT_GROUP rules ---
    for rule_name, needs_role in _ROLE_RULES:
        if needs_role:
            for role in _COLOR_ROLES:
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(StepOp.FOR_EACH_ENTITY,
                              kind="object_group", rule=rule_name,
                              color_role=role),
                    make_step(StepOp.RENDER_SCENE),
                )
                if verify_scene_program(prog, demos):
                    verified.append(prog)
                    return verified
        else:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.FOR_EACH_ENTITY,
                          kind="object_group", rule=rule_name),
                make_step(StepOp.RENDER_SCENE),
            )
            if verify_scene_program(prog, demos):
                verified.append(prog)
                return verified

    # --- Priority 4: Unguarded per-object rules (existing) ---
    for entity_kind in ("object", "panel"):
        for conn_val in (4, 8, None):
            for rule_name, needs_role in _ROLE_RULES:
                if needs_role:
                    for role in _COLOR_ROLES:
                        params: dict = {
                            "kind": entity_kind,
                            "rule": rule_name,
                            "color_role": role,
                        }
                        if conn_val is not None:
                            params["connectivity"] = conn_val
                        prog = make_scene_program(
                            make_step(StepOp.PARSE_SCENE),
                            make_step(StepOp.FOR_EACH_ENTITY, **params),
                            make_step(StepOp.RENDER_SCENE),
                        )
                        if verify_scene_program(prog, demos):
                            verified.append(prog)
                            return verified
                else:
                    params = {
                        "kind": entity_kind,
                        "rule": rule_name,
                    }
                    if conn_val is not None:
                        params["connectivity"] = conn_val
                    prog = make_scene_program(
                        make_step(StepOp.PARSE_SCENE),
                        make_step(StepOp.FOR_EACH_ENTITY, **params),
                        make_step(StepOp.RENDER_SCENE),
                    )
                    if verify_scene_program(prog, demos):
                        verified.append(prog)
                        return verified

    return verified


# ---------------------------------------------------------------------------
# Family 17: observed cell-to-output synthesis
# ---------------------------------------------------------------------------
#
# Operates on partitioned grids (non-uniform OK). Observes the
# relationship between partition cells and the output grid across all
# demos, then synthesizes a structural program from the observed mapping.
#
# Three synthesis strategies:
#   A. Cell selection — output equals one cell's content
#   B. Cell combination — output is element-wise combine of cells
#   C. Per-cell predicate summary — output is a small grid where each
#      pixel represents a property of the corresponding cell


class _ObservedCellProgram:
    """Executable program synthesized from observed cell→output mappings."""

    def __init__(self, strategy: str, params: dict):
        self.strategy = strategy
        self.params = params

    def execute(self, input_grid: Grid) -> Grid:
        s = perceive_grid(input_grid)
        if s.partition is None:
            return input_grid.copy()

        p = s.partition
        bg = s.bg_color
        cells = p.cells

        if self.strategy == "select_cell":
            idx = self.params["cell_index_fn"](cells, bg, input_grid)
            if idx is None or idx >= len(cells):
                return input_grid.copy()
            r0, c0, r1, c1 = cells[idx].bbox
            return input_grid[r0:r1 + 1, c0:c1 + 1].copy()

        if self.strategy == "select_cell_transform":
            idx = self.params["cell_index_fn"](cells, bg, input_grid)
            if idx is None or idx >= len(cells):
                return input_grid.copy()
            r0, c0, r1, c1 = cells[idx].bbox
            cell = input_grid[r0:r1 + 1, c0:c1 + 1].copy()
            transform = self.params["transform"]
            return _apply_observed_transform(cell, transform)

        if self.strategy == "combine_two_cells":
            op = self.params["operation"]
            i0, i1 = self.params["cell_indices"]
            if i0 >= len(cells) or i1 >= len(cells):
                return input_grid.copy()
            r0a, c0a, r1a, c1a = cells[i0].bbox
            r0b, c0b, r1b, c1b = cells[i1].bbox
            ca = input_grid[r0a:r1a + 1, c0a:c1a + 1]
            cb = input_grid[r0b:r1b + 1, c0b:c1b + 1]
            return _combine_observed_grids(ca, cb, bg, op)

        if self.strategy == "predicate_summary":
            pred_fn = self.params["predicate_fn"]
            n_rows, n_cols = p.n_rows, p.n_cols
            out = np.full((n_rows, n_cols), bg, dtype=np.int32)
            for cell in cells:
                r0, c0, r1, c1 = cell.bbox
                cg = input_grid[r0:r1 + 1, c0:c1 + 1]
                out[cell.row_idx, cell.col_idx] = pred_fn(cg, bg)
            return out

        return input_grid.copy()

    def verify_on_demo(self, input_grid: Grid, output_grid: Grid) -> bool:
        try:
            result = self.execute(input_grid)
            return np.array_equal(result, output_grid)
        except Exception:
            return False


def _apply_observed_transform(grid: Grid, transform: str) -> Grid:
    if transform == "rot90":
        return np.rot90(grid, 1).copy()
    if transform == "rot180":
        return np.rot90(grid, 2).copy()
    if transform == "rot270":
        return np.rot90(grid, 3).copy()
    if transform == "flip_lr":
        return np.fliplr(grid).copy()
    if transform == "flip_ud":
        return np.flipud(grid).copy()
    if transform == "transpose":
        return grid.T.copy()
    return grid.copy()


def _combine_observed_grids(a: Grid, b: Grid, bg: int, op: str) -> Grid:
    if a.shape != b.shape:
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])
        pa = np.full((h, w), bg, dtype=np.int32)
        pb = np.full((h, w), bg, dtype=np.int32)
        pa[:a.shape[0], :a.shape[1]] = a
        pb[:b.shape[0], :b.shape[1]] = b
        a, b = pa, pb

    if op == "overlay_a_on_b":
        result = b.copy()
        result[a != bg] = a[a != bg]
        return result
    if op == "overlay_b_on_a":
        result = a.copy()
        result[b != bg] = b[b != bg]
        return result
    if op == "xor":
        result = np.full_like(a, bg)
        result[(a != bg) & (b == bg)] = a[(a != bg) & (b == bg)]
        result[(a == bg) & (b != bg)] = b[(a == bg) & (b != bg)]
        return result
    if op == "and":
        result = np.full_like(a, bg)
        both = (a != bg) & (b != bg)
        result[both] = a[both]
        return result
    return a.copy()


def _try_observed_cell_synthesis(
    demos: tuple[DemoPair, ...],
) -> list:
    """Synthesize programs by observing cell→output relationships."""
    if not demos:
        return []

    results = []

    prog = _try_obs_cell_selection(demos)
    if prog is not None:
        return [prog]

    prog = _try_obs_cell_selection_transform(demos)
    if prog is not None:
        return [prog]

    prog = _try_obs_cell_combination(demos)
    if prog is not None:
        return [prog]

    prog = _try_obs_predicate_summary(demos)
    if prog is not None:
        return [prog]

    return results


def _try_obs_cell_selection(demos: tuple[DemoPair, ...]):
    d0 = demos[0]
    s0 = perceive_grid(d0.input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    match_idx = None
    for ci, cell in enumerate(s0.partition.cells):
        r0, c0, r1, c1 = cell.bbox
        cg = d0.input[r0:r1 + 1, c0:c1 + 1]
        if cg.shape == d0.output.shape and np.array_equal(cg, d0.output):
            match_idx = ci
            break

    if match_idx is None:
        return None

    bg = s0.bg_color
    cells = s0.partition.cells

    for name, fn in _OBS_CELL_SELECTORS:
        if fn(cells, bg, d0.input) == match_idx:
            prog = _ObservedCellProgram("select_cell", {
                "cell_index_fn": fn, "selector_name": name,
            })
            if all(prog.verify_on_demo(d.input, d.output) for d in demos):
                return prog
    return None


def _try_obs_cell_selection_transform(demos: tuple[DemoPair, ...]):
    d0 = demos[0]
    s0 = perceive_grid(d0.input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None

    transforms = ["rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose"]
    bg = s0.bg_color
    cells = s0.partition.cells

    for ci, cell in enumerate(cells):
        r0, c0, r1, c1 = cell.bbox
        cg = d0.input[r0:r1 + 1, c0:c1 + 1]
        for t in transforms:
            tg = _apply_observed_transform(cg, t)
            if tg.shape == d0.output.shape and np.array_equal(tg, d0.output):
                for name, fn in _OBS_CELL_SELECTORS:
                    if fn(cells, bg, d0.input) == ci:
                        prog = _ObservedCellProgram("select_cell_transform", {
                            "cell_index_fn": fn, "selector_name": name,
                            "transform": t,
                        })
                        if all(prog.verify_on_demo(d.input, d.output) for d in demos):
                            return prog
    return None


def _try_obs_cell_combination(demos: tuple[DemoPair, ...]):
    d0 = demos[0]
    s0 = perceive_grid(d0.input)
    if s0.partition is None or len(s0.partition.cells) < 2:
        return None
    if len(s0.partition.cells) > 6:
        return None

    bg = s0.bg_color
    cells = s0.partition.cells
    ops = ["overlay_a_on_b", "overlay_b_on_a", "xor", "and"]

    for i in range(len(cells)):
        for j in range(len(cells)):
            if i == j:
                continue
            r0a, c0a, r1a, c1a = cells[i].bbox
            r0b, c0b, r1b, c1b = cells[j].bbox
            ca = d0.input[r0a:r1a + 1, c0a:c1a + 1]
            cb = d0.input[r0b:r1b + 1, c0b:c1b + 1]
            for op in ops:
                result = _combine_observed_grids(ca, cb, bg, op)
                if result.shape == d0.output.shape and np.array_equal(result, d0.output):
                    prog = _ObservedCellProgram("combine_two_cells", {
                        "cell_indices": (i, j), "operation": op,
                    })
                    if all(prog.verify_on_demo(d.input, d.output) for d in demos):
                        return prog
    return None


def _try_obs_predicate_summary(demos: tuple[DemoPair, ...]):
    d0 = demos[0]
    s0 = perceive_grid(d0.input)
    if s0.partition is None:
        return None
    p = s0.partition
    if d0.output.shape != (p.n_rows, p.n_cols):
        return None

    bg = s0.bg_color
    for name, fn in _OBS_CELL_PREDICATES:
        prog = _ObservedCellProgram("predicate_summary", {
            "predicate_fn": fn, "predicate_name": name,
        })
        if all(prog.verify_on_demo(d.input, d.output) for d in demos):
            return prog
    return None


# Cell selectors
def _obs_sel_most_non_bg(cells, bg, grid):
    best_i, best_n = 0, -1
    for i, c in enumerate(cells):
        r0, c0, r1, c1 = c.bbox
        n = int(np.sum(grid[r0:r1+1, c0:c1+1] != bg))
        if n > best_n:
            best_n = n
            best_i = i
    return best_i

def _obs_sel_least_non_bg_gt0(cells, bg, grid):
    best_i, best_n = None, float('inf')
    for i, c in enumerate(cells):
        r0, c0, r1, c1 = c.bbox
        n = int(np.sum(grid[r0:r1+1, c0:c1+1] != bg))
        if 0 < n < best_n:
            best_n = n
            best_i = i
    return best_i

def _obs_sel_most_colors(cells, bg, grid):
    best_i, best_n = 0, -1
    for i, c in enumerate(cells):
        r0, c0, r1, c1 = c.bbox
        cg = grid[r0:r1+1, c0:c1+1]
        n = len(set(int(v) for v in np.unique(cg)) - {bg})
        if n > best_n:
            best_n = n
            best_i = i
    return best_i

def _obs_sel_fewest_colors_gt0(cells, bg, grid):
    best_i, best_n = None, float('inf')
    for i, c in enumerate(cells):
        r0, c0, r1, c1 = c.bbox
        cg = grid[r0:r1+1, c0:c1+1]
        n = len(set(int(v) for v in np.unique(cg)) - {bg})
        if 0 < n < best_n:
            best_n = n
            best_i = i
    return best_i

def _obs_sel_unique_content(cells, bg, grid):
    contents = []
    for c in cells:
        r0, c0, r1, c1 = c.bbox
        contents.append(grid[r0:r1+1, c0:c1+1].tobytes())
    for i, cb in enumerate(contents):
        if contents.count(cb) == 1:
            return i
    return None

def _obs_sel_has_singleton(cells, bg, grid):
    for i, c in enumerate(cells):
        r0, c0, r1, c1 = c.bbox
        cg = grid[r0:r1+1, c0:c1+1]
        for r in range(cg.shape[0]):
            for col in range(cg.shape[1]):
                if cg[r, col] != bg:
                    neighbors = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, col+dc
                        if 0 <= nr < cg.shape[0] and 0 <= nc < cg.shape[1]:
                            if cg[nr, nc] != bg:
                                neighbors += 1
                    if neighbors == 0:
                        return i
    return None

_OBS_CELL_SELECTORS = [
    ("most_non_bg", _obs_sel_most_non_bg),
    ("least_non_bg_gt0", _obs_sel_least_non_bg_gt0),
    ("most_colors", _obs_sel_most_colors),
    ("fewest_colors_gt0", _obs_sel_fewest_colors_gt0),
    ("unique_content", _obs_sel_unique_content),
    ("has_singleton", _obs_sel_has_singleton),
]

# Cell predicates
def _obs_pred_dominant_non_bg(cg, bg):
    non_bg = cg[cg != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[np.argmax(counts)])

def _obs_pred_has_non_bg(cg, bg):
    return 1 if np.any(cg != bg) else 0

def _obs_pred_count_non_bg(cg, bg):
    return int(np.sum(cg != bg))

def _obs_pred_count_colors(cg, bg):
    return len(set(int(v) for v in np.unique(cg)) - {bg})

def _obs_pred_minority_color(cg, bg):
    non_bg = cg[cg != bg]
    if len(non_bg) < 2:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    if len(vals) < 2:
        return bg
    return int(vals[np.argmin(counts)])

_OBS_CELL_PREDICATES = [
    ("dominant_non_bg", _obs_pred_dominant_non_bg),
    ("has_non_bg", _obs_pred_has_non_bg),
    ("count_non_bg", _obs_pred_count_non_bg),
    ("count_colors", _obs_pred_count_colors),
    ("minority_color", _obs_pred_minority_color),
]


# ---------------------------------------------------------------------------
# Family 18: generalized periodicity synthesis
# ---------------------------------------------------------------------------
#
# Uses the periodicity core (aria.periodicity) to detect, complete, and
# extend periodic patterns in structural sequences extracted from the grid.
#
# Three strategies:
#   A. Per-row/col periodic repair (unframed) — detect period per line, fix violations
#   B. Separator-relative pattern extension — read one side, tile other side
#   C. 2D periodic anomaly recolor — detect 2D motif, find anomaly, recolor
#
# All strategies operate on generic sequences, not task-specific structures.


class _PeriodicProgram:
    """Executable program synthesized from periodic pattern analysis."""

    def __init__(self, strategy: str, params: dict):
        self.strategy = strategy
        self.params = params

    def execute(self, input_grid: Grid) -> Grid:
        from aria.periodicity import (
            detect_1d_period, complete_sequence,
            detect_seed_and_extend_row, tile_from_seed_row,
            detect_column_seed_row, detect_2d_anomaly,
        )
        from aria.decomposition import detect_bg

        bg = detect_bg(input_grid)

        if self.strategy == "per_line_repair":
            return self._execute_per_line_repair(input_grid, bg)
        if self.strategy == "separator_extend":
            return self._execute_separator_extend(input_grid, bg)
        if self.strategy == "column_seed_tile":
            return self._execute_column_seed_tile(input_grid, bg)
        if self.strategy == "anomaly_recolor":
            return self._execute_anomaly_recolor(input_grid, bg)
        return input_grid.copy()

    def _execute_per_line_repair(self, grid, bg):
        from aria.periodicity import detect_1d_period, complete_sequence
        axis = self.params["axis"]
        result = grid.copy()
        rows, cols = grid.shape
        if axis == "row":
            for r in range(rows):
                pat = detect_1d_period(grid[r], require_violations=True)
                if pat is not None and pat.confidence >= 0.7:
                    result[r] = complete_sequence(grid[r], pat)
        else:
            for c in range(cols):
                pat = detect_1d_period(grid[:, c], require_violations=True)
                if pat is not None and pat.confidence >= 0.7:
                    result[:, c] = complete_sequence(grid[:, c], pat)
        return result

    def _execute_separator_extend(self, grid, bg):
        sep_col = self.params.get("separator_col")
        sep_row = self.params.get("separator_row")
        result = grid.copy()
        rows, cols = grid.shape

        if sep_col is not None:
            # Find separator column dynamically (same color for all rows)
            actual_sep = _find_separator_col(grid)
            if actual_sep is None:
                actual_sep = sep_col
            for r in range(rows):
                extended = _extend_row_across_separator(grid[r], actual_sep, bg)
                if extended is not None:
                    result[r] = extended
        return result

    def _execute_column_seed_tile(self, grid, bg):
        from aria.periodicity import detect_column_seed_row, tile_from_seed_row
        seed = detect_column_seed_row(grid, bg)
        if seed is None:
            return grid.copy()
        return tile_from_seed_row(grid, seed, bg)

    def _execute_anomaly_recolor(self, grid, bg):
        from aria.periodicity import detect_2d_anomaly
        recolor = self.params["recolor_to"]
        result_pair = detect_2d_anomaly(grid, bg)
        if result_pair is None:
            return grid.copy()
        motif, anomalies = result_pair
        result = grid.copy()
        for r, c in anomalies:
            result[r, c] = recolor
        return result

    def verify_on_demo(self, input_grid, output_grid):
        try:
            result = self.execute(input_grid)
            return np.array_equal(result, output_grid)
        except Exception:
            return False


def _find_separator_col(grid: Grid) -> int | None:
    """Find a vertical separator column (same color in every row)."""
    rows, cols = grid.shape
    for c in range(cols):
        vals = set(int(grid[r, c]) for r in range(rows))
        if len(vals) == 1 and vals != {0}:  # uniform non-bg column
            return c
    return None


def _extend_row_across_separator(row: np.ndarray, sep_col: int, bg: int) -> np.ndarray | None:
    """Extend the left-side pattern of a row across the separator to the right."""
    n = len(row)
    if sep_col <= 0 or sep_col >= n - 1:
        return None

    left = row[:sep_col]
    sep_val = int(row[sep_col])

    # Check if left side has non-bg content
    non_bg = [i for i in range(len(left)) if int(left[i]) != bg]
    if not non_bg:
        return None  # empty left — nothing to extend

    result = row.copy()

    # Find the pattern in the left side
    from aria.periodicity import detect_1d_period
    pat = detect_1d_period(left, bg=bg)

    if pat is not None and pat.confidence >= 0.6:
        # Use the detected period to tile the right side
        for c in range(sep_col + 1, n):
            result[c] = pat.pattern[c % pat.period]
    else:
        # Direct tiling: repeat the left pattern
        left_len = len(left)
        for c in range(sep_col + 1, n):
            result[c] = int(left[c % left_len])

    return result


def _try_periodicity_synthesis(demos: tuple[DemoPair, ...]) -> list:
    """Synthesize programs using generalized periodicity detection."""
    if not demos:
        return []
    if not all(d.input.shape == d.output.shape for d in demos):
        return []

    results = []

    # Strategy A: per-row periodic repair (unframed)
    prog = _try_per_line_periodic_repair(demos)
    if prog is not None:
        return [prog]

    # Strategy B: separator-relative extension
    prog = _try_separator_extension(demos)
    if prog is not None:
        return [prog]

    # Strategy C: column-seed tiling
    prog = _try_column_seed_tiling(demos)
    if prog is not None:
        return [prog]

    # Strategy D: 2D anomaly recolor
    prog = _try_2d_anomaly_recolor(demos)
    if prog is not None:
        return [prog]

    return results


def _try_per_line_periodic_repair(demos):
    """Try repairing per-row or per-col periodic violations."""
    for axis in ["row", "col"]:
        prog = _PeriodicProgram("per_line_repair", {"axis": axis})
        if all(prog.verify_on_demo(d.input, d.output) for d in demos):
            return prog
    return None


def _try_separator_extension(demos):
    """Try extending patterns across a vertical separator."""
    d0 = demos[0]
    sep = _find_separator_col(d0.input)
    if sep is None:
        return None

    prog = _PeriodicProgram("separator_extend", {"separator_col": sep})
    if all(prog.verify_on_demo(d.input, d.output) for d in demos):
        return prog
    return None


def _try_column_seed_tiling(demos):
    """Try tiling columns from a detected seed row."""
    from aria.periodicity import detect_column_seed_row

    d0 = demos[0]
    from aria.decomposition import detect_bg
    bg = detect_bg(d0.input)
    seed = detect_column_seed_row(d0.input, bg)
    if seed is None:
        return None

    prog = _PeriodicProgram("column_seed_tile", {"seed_row": seed})
    if all(prog.verify_on_demo(d.input, d.output) for d in demos):
        return prog
    return None


def _try_2d_anomaly_recolor(demos):
    """Try detecting 2D periodic anomaly and recoloring it."""
    d0 = demos[0]
    from aria.decomposition import detect_bg
    bg = detect_bg(d0.input)

    from aria.periodicity import detect_2d_anomaly
    result = detect_2d_anomaly(d0.input, bg)
    if result is None:
        return None

    motif, anomalies = result
    if not anomalies:
        return None

    # Find what color the anomalies should be recolored to
    # Check the output to determine the recolor target
    anomaly_output_colors = set()
    for r, c in anomalies:
        anomaly_output_colors.add(int(d0.output[r, c]))

    if len(anomaly_output_colors) != 1:
        return None  # anomalies map to different colors — too complex

    recolor_to = next(iter(anomaly_output_colors))

    prog = _PeriodicProgram("anomaly_recolor", {"recolor_to": recolor_to})
    if all(prog.verify_on_demo(d.input, d.output) for d in demos):
        return prog
    return None
