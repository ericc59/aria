"""Scene-program proposer — automated proposal of short scene programs.

Proposes bounded, reusable multi-step scene programs for same_as_input tasks
from perception evidence. Each proposal is a SceneProgram that can be
compiled and verified by the executor.

Proposal templates:
1. Partition boolean combine (overlay/AND/OR/XOR panels)
2. Scoped fill enclosed (fill bg inside framed/partition regions)
3. Scoped color map (color substitution inside selected regions)
4. Partition cell property map (summary property per cell)

No task-id logic. No benchmark hacks. Exact verification is final arbiter.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.core.scene_executor import execute_scene_program, SceneExecutionError
from aria.decomposition import detect_bg
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Proposal result
# ---------------------------------------------------------------------------


def propose_scene_programs(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose scene programs from perception evidence.

    Returns list of (template_name, program) pairs. The caller should
    compile/verify each against the demos.

    Pre-filters partition-dependent proposals using cross-demo structural
    consistency: if partition cell counts differ across demos, partition-
    based proposals are skipped entirely.
    """
    if not demos:
        return []
    if not all(d.input.shape == d.output.shape for d in demos):
        return []

    # Cross-demo partition consistency check
    from aria.core.scene_solve import get_consensus_enabled
    partition_consistent = (
        _check_partition_consistency(demos) if get_consensus_enabled() else True
    )

    proposals: list[tuple[str, SceneProgram]] = []

    # 0. Partition per-cell conditional (highest priority for partition tasks)
    if partition_consistent:
        proposals.extend(_propose_partition_per_cell(demos))

    # 1. Partition boolean combine
    if partition_consistent:
        proposals.extend(_propose_partition_boolean_combine(demos))

    # 2. Scoped fill enclosed (whole grid)
    proposals.extend(_propose_fill_enclosed_variants(demos))

    # 3. Scoped color map (inside objects, inside frame, inside partition cells)
    proposals.extend(_propose_scoped_color_map(demos))

    # 4. Partition cell map (write property per cell to output)
    if partition_consistent:
        proposals.extend(_propose_partition_map(demos))

    # 5. Relational partition: combine cells into target
    if partition_consistent:
        proposals.extend(_propose_combine_cells(demos))

    # 6. Relational partition: broadcast source to targets
    if partition_consistent:
        proposals.extend(_propose_broadcast_to_cells(demos))

    # 7. Object-scoped transforms (no partition needed)
    proposals.extend(_propose_object_transforms(demos))

    # 8. Frame-interior proposals
    proposals.extend(_propose_frame_interior(demos))

    return proposals


def _check_partition_consistency(demos: tuple[DemoPair, ...]) -> bool:
    """Check that all demos have consistent partition structure.

    Returns True if all demos either have no partition or have
    partitions with the same cell count.
    """
    if len(demos) < 2:
        return True

    perceptions = [perceive_grid(d.input) for d in demos]
    has_partition = [p.partition is not None for p in perceptions]

    if not any(has_partition):
        return True  # no partitions at all

    if not all(has_partition):
        return False  # some have, some don't

    cell_counts = {len(p.partition.cells) for p in perceptions}
    return len(cell_counts) == 1


# ---------------------------------------------------------------------------
# Template 0: Partition per-cell conditional
# ---------------------------------------------------------------------------


def _propose_partition_per_cell(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose per-cell conditional operations over partition cells."""
    state = perceive_grid(demos[0].input)
    if state.partition is None or len(state.partition.cells) < 2:
        return []

    proposals = []

    rules = [
        "swap_dominant_with_bg",
        "fill_bg_with_dominant",
        "clear_non_bg",
        "invert_non_bg",
        "fill_solid_or_clear",
        "fill_solid_argmax_or_clear",
    ]
    filters = ["all", "has_non_bg"]

    for rule in rules:
        for filt in filters:
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                SceneStep(
                    op=StepOp.APPLY_PER_CELL,
                    params={"rule": rule, "filter": filt},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "per_cell_result"}),
            ))
            proposals.append((f"per_cell_{rule}_{filt}", prog))

    # Also try fill_bg_with_color for each color present
    bg = detect_bg(demos[0].input)
    palette = set(int(v) for v in np.unique(demos[0].input)) - {bg}
    for c in sorted(palette):
        for filt in filters:
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                SceneStep(
                    op=StepOp.APPLY_PER_CELL,
                    params={"rule": "fill_bg_with_color", "filter": filt, "fill_color": c},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "per_cell_result"}),
            ))
            proposals.append((f"per_cell_fill_bg_c{c}_{filt}", prog))

    # Threshold-based fill_solid_if_enough_or_clear
    for threshold in (2, 3):
        for filt in ["all"]:
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                SceneStep(
                    op=StepOp.APPLY_PER_CELL,
                    params={"rule": "fill_solid_if_enough_or_clear", "filter": filt, "threshold": threshold},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "per_cell_result"}),
            ))
            proposals.append((f"per_cell_fill_solid_thresh{threshold}_{filt}", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 1: Partition boolean combine
# ---------------------------------------------------------------------------


def _propose_partition_boolean_combine(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose boolean combination of partition panels."""
    state = perceive_grid(demos[0].input)
    if state.partition is None or len(state.partition.cells) < 2:
        return []

    proposals = []
    for mode in ("or", "and", "xor", "stack"):
        prog = SceneProgram(steps=(
            SceneStep(op=StepOp.PARSE_SCENE),
            SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
            SceneStep(
                op=StepOp.BOOLEAN_COMBINE_PANELS,
                params={"mode": mode},
            ),
            SceneStep(op=StepOp.RENDER_SCENE, params={"source": "combined"}),
        ))
        proposals.append((f"partition_boolean_{mode}", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 2: Fill enclosed variants
# ---------------------------------------------------------------------------


def _propose_fill_enclosed_variants(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose fill-enclosed with different fill colors."""
    d0 = demos[0]
    bg = detect_bg(d0.input)
    diff = d0.input != d0.output
    if not np.any(diff):
        return []

    # Only propose if changes are in bg positions
    changed_pos = list(zip(*np.where(diff)))
    if not all(int(d0.input[r, c]) == bg for r, c in changed_pos):
        return []

    proposals = []
    # Try each non-bg color that appears in the output diff
    fill_colors = set(int(d0.output[r, c]) for r, c in changed_pos)
    for fc in sorted(fill_colors):
        prog = SceneProgram(steps=(
            SceneStep(op=StepOp.PARSE_SCENE),
            SceneStep(
                op=StepOp.FILL_ENCLOSED_REGIONS,
                params={"fill_color": fc},
            ),
            SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
        ))
        proposals.append((f"fill_enclosed_c{fc}", prog))

    # Auto-fill (boundary color)
    prog = SceneProgram(steps=(
        SceneStep(op=StepOp.PARSE_SCENE),
        SceneStep(
            op=StepOp.FILL_ENCLOSED_REGIONS,
            params={"mode": "boundary_color"},
        ),
        SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
    ))
    proposals.append(("fill_enclosed_auto", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 3: Scoped color map
# ---------------------------------------------------------------------------


def _propose_scoped_color_map(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose color substitution scoped to specific regions."""
    d0 = demos[0]
    diff = d0.input != d0.output
    if not np.any(diff):
        return []

    # Build the consistent color map across all demos
    color_map: dict[int, int] = {}
    consistent = True
    for demo in demos:
        for r in range(demo.input.shape[0]):
            for c in range(demo.input.shape[1]):
                ic, oc = int(demo.input[r, c]), int(demo.output[r, c])
                if ic != oc:
                    if ic in color_map and color_map[ic] != oc:
                        consistent = False
                        break
                    color_map[ic] = oc
            if not consistent:
                break
        if not consistent:
            break

    if not consistent or not color_map or len(color_map) > 5:
        return []

    proposals = []
    pairs = sorted(color_map.items())

    # Whole-grid scoped color map (already handled by global_color_map render spec)
    # Instead, try object-scoped and region-scoped

    # Object-interior scoped: apply only inside non-bg objects
    prog = _make_scoped_color_map_program(pairs, scope="objects")
    proposals.append(("scoped_color_map_objects", prog))

    # Object-bbox scoped: apply within bounding boxes of objects (includes bg within bbox)
    prog = _make_scoped_color_map_program(pairs, scope="object_bboxes")
    proposals.append(("scoped_color_map_object_bboxes", prog))

    # Frame-interior scoped: apply only inside framed regions
    state = perceive_grid(d0.input)
    if state.framed_regions:
        prog = _make_scoped_color_map_program(pairs, scope="frame_interior")
        proposals.append(("scoped_color_map_frame", prog))

    # Partition-cell scoped
    if state.partition is not None:
        prog = _make_scoped_color_map_program(pairs, scope="partition_cells")
        proposals.append(("scoped_color_map_partition", prog))

    return proposals


def _make_scoped_color_map_program(
    pairs: list[tuple[int, int]],
    scope: str,
) -> SceneProgram:
    """Build a scoped color map scene program."""
    return SceneProgram(steps=(
        SceneStep(op=StepOp.PARSE_SCENE),
        SceneStep(
            op=StepOp.RECOLOR_OBJECT,
            params={
                "color_pairs": pairs,
                "scope": scope,
            },
        ),
        SceneStep(op=StepOp.RENDER_SCENE, params={"source": "recolored"}),
    ))


# ---------------------------------------------------------------------------
# Template 4: Partition property map
# ---------------------------------------------------------------------------


def _propose_partition_map(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose map-over-entities for partition-based summary tasks."""
    state = perceive_grid(demos[0].input)
    if state.partition is None or len(state.partition.cells) < 4:
        return []

    proposals = []
    for prop in ("dominant_non_bg_color", "has_non_bg", "unique_color_count",
                 "minority_color", "non_bg_count"):
        for corr_mode in ("row_col_index", "positional_order"):
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                SceneStep(
                    op=StepOp.INFER_OUTPUT_SIZE,
                    params={"mode": "partition_grid_shape"},
                ),
                SceneStep(op=StepOp.INFER_OUTPUT_BACKGROUND),
                SceneStep(op=StepOp.INITIALIZE_OUTPUT_SCENE),
                SceneStep(
                    op=StepOp.BUILD_CORRESPONDENCE,
                    params={
                        "source_kind": "panel",
                        "mode": corr_mode,
                    },
                    output_id="corr",
                ),
                SceneStep(
                    op=StepOp.MAP_OVER_ENTITIES,
                    params={
                        "kind": "panel",
                        "property": prop,
                        "layout": "grid",
                        "correspondence": "corr",
                    },
                ),
                SceneStep(op=StepOp.RENDER_SCENE),
            ))
            proposals.append((f"partition_map_{prop}_{corr_mode}", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 5: Combine cells into target
# ---------------------------------------------------------------------------


def _propose_combine_cells(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose overlay-into-target-cell programs."""
    state = perceive_grid(demos[0].input)
    if state.partition is None or len(state.partition.cells) < 2:
        return []

    # Check cells have uniform shape
    shapes = set()
    for cell in state.partition.cells:
        r0, c0, r1, c1 = cell.bbox
        shapes.add((r1 - r0 + 1, c1 - c0 + 1))
    if len(shapes) != 1:
        return []

    proposals = []
    for mode in ("or", "and", "xor"):
        for target in ("empty", "most_non_bg", "least_non_bg", "first", "last"):
            for keep in (True, False):
                suffix = "" if keep else "_clear"
                prog = SceneProgram(steps=(
                    SceneStep(op=StepOp.PARSE_SCENE),
                    SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                    SceneStep(
                        op=StepOp.COMBINE_CELLS,
                        params={"mode": mode, "target": target, "keep_others": keep},
                    ),
                    SceneStep(op=StepOp.RENDER_SCENE, params={"source": "combined"}),
                ))
                proposals.append((f"combine_{mode}_into_{target}{suffix}", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 6: Broadcast source to targets
# ---------------------------------------------------------------------------


def _propose_broadcast_to_cells(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose broadcast-source-to-targets programs."""
    state = perceive_grid(demos[0].input)
    if state.partition is None or len(state.partition.cells) < 2:
        return []

    shapes = set()
    for cell in state.partition.cells:
        r0, c0, r1, c1 = cell.bbox
        shapes.add((r1 - r0 + 1, c1 - c0 + 1))
    if len(shapes) != 1:
        return []

    proposals = []
    sources = (
        "most_non_bg", "most_colors", "unique_color",
        "unique_non_bg_count", "unique_palette", "differs_from_majority",
        "first_non_empty",
    )
    targets = (
        "empty", "all_others", "different_dominant", "same_dominant",
        "non_empty", "same_row", "same_col", "adjacent_4",
    )
    for source in sources:
        for target in targets:
            for overlay in ("replace", "overlay_non_bg"):
                prog = SceneProgram(steps=(
                    SceneStep(op=StepOp.PARSE_SCENE),
                    SceneStep(op=StepOp.SPLIT_BY_SEPARATOR),
                    SceneStep(
                        op=StepOp.BROADCAST_TO_CELLS,
                        params={
                            "source": source,
                            "target": target,
                            "overlay_mode": overlay,
                        },
                    ),
                    SceneStep(op=StepOp.RENDER_SCENE, params={"source": "broadcast_result"}),
                ))
                proposals.append((f"broadcast_{source}_to_{target}_{overlay}", prog))

    return proposals


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Template 7: Object-scoped transforms
# ---------------------------------------------------------------------------


def _propose_object_transforms(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose object-scoped transforms for non-partition tasks."""
    d0 = demos[0]
    bg = detect_bg(d0.input)
    state = perceive_grid(d0.input)

    # Only propose if there are non-bg objects
    n_obj = len([o for o in state.objects.objects if o.color != bg and o.size > 1])
    if n_obj == 0:
        return []

    proposals = []

    # For each geometric transform, try applying it to the whole grid
    for transform_name, transform_code in [
        ("rot90", 0), ("rot180", 1), ("rot270", 2),
        ("flipLR", 3), ("flipUD", 4), ("transpose", 5),
    ]:
        prog = SceneProgram(steps=(
            SceneStep(op=StepOp.PARSE_SCENE),
            SceneStep(
                op=StepOp.RECOLOR_OBJECT,
                params={"color_pairs": [], "scope": "global"},
            ),
            SceneStep(op=StepOp.RENDER_SCENE, params={"source": "recolored"}),
        ))
        # This is redundant with geometric_transform render spec, skip

    # Object recolor: try swapping each pair of non-bg colors
    non_bg_colors = sorted(set(int(v) for v in np.unique(d0.input)) - {bg})
    if len(non_bg_colors) >= 2:
        for i, c1 in enumerate(non_bg_colors):
            for c2 in non_bg_colors[i + 1:]:
                pairs = [(c1, c2), (c2, c1)]
                for scope in ("objects", "global"):
                    prog = _make_scoped_color_map_program(pairs, scope=scope)
                    proposals.append((f"swap_{c1}_{c2}_{scope}", prog))

    # Single-color recolor: try mapping each non-bg color to bg and vice versa
    for c in non_bg_colors[:3]:
        for scope in ("objects", "global"):
            pairs = [(c, bg)]
            prog = _make_scoped_color_map_program(pairs, scope=scope)
            proposals.append((f"remove_color_{c}_{scope}", prog))

    return proposals


# ---------------------------------------------------------------------------
# Template 8: Frame-interior proposals
# ---------------------------------------------------------------------------


def _propose_frame_interior(
    demos: tuple[DemoPair, ...],
) -> list[tuple[str, SceneProgram]]:
    """Propose frame-interior transforms."""
    d0 = demos[0]
    bg = detect_bg(d0.input)
    state = perceive_grid(d0.input)

    if not state.framed_regions:
        return []

    proposals = []

    # Fill enclosed regions inside frames
    diff = d0.input != d0.output
    if np.any(diff):
        fill_colors = set(int(d0.output[r, c]) for r, c in zip(*np.where(diff)))
        for fc in sorted(fill_colors):
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(
                    op=StepOp.FILL_ENCLOSED_REGIONS,
                    params={"fill_color": fc},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
            ))
            proposals.append((f"fill_enclosed_c{fc}_frame", prog))

    # Fill enclosed auto (boundary color)
    prog = SceneProgram(steps=(
        SceneStep(op=StepOp.PARSE_SCENE),
        SceneStep(
            op=StepOp.FILL_ENCLOSED_REGIONS,
            params={"mode": "boundary_color"},
        ),
        SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
    ))
    proposals.append(("fill_enclosed_auto_frame", prog))

    # Frame-scoped color maps
    color_map: dict[int, int] = {}
    consistent = True
    for demo in demos:
        for r in range(demo.input.shape[0]):
            for c in range(demo.input.shape[1]):
                ic, oc = int(demo.input[r, c]), int(demo.output[r, c])
                if ic != oc:
                    if ic in color_map and color_map[ic] != oc:
                        consistent = False
                        break
                    color_map[ic] = oc
            if not consistent:
                break
        if not consistent:
            break

    if consistent and color_map and len(color_map) <= 5:
        pairs = sorted(color_map.items())
        prog = _make_scoped_color_map_program(pairs, scope="frame_interior")
        proposals.append(("scoped_color_map_frame_interior", prog))

    return proposals


# ---------------------------------------------------------------------------
# Verify proposals against demos
# ---------------------------------------------------------------------------


def verify_scene_proposals(
    demos: tuple[DemoPair, ...],
    proposals: list[tuple[str, SceneProgram]],
) -> list[tuple[str, SceneProgram, bool]]:
    """Verify each proposed scene program against all demos.

    Returns list of (template_name, program, verified) triples.
    """
    results = []
    for template_name, prog in proposals:
        verified = True
        for demo in demos:
            try:
                output = execute_scene_program(prog, demo.input)
                if not np.array_equal(output, demo.output):
                    verified = False
                    break
            except (SceneExecutionError, Exception):
                verified = False
                break
        results.append((template_name, prog, verified))
    return results


def propose_and_verify(
    demos: tuple[DemoPair, ...],
) -> tuple[str, SceneProgram] | None:
    """Propose and verify scene programs, return first verified one.

    Runs both the template-based proposer and the consensus-controlled
    compositional search. Returns the first verified program from either.
    """
    # Template-based proposals
    proposals = propose_scene_programs(demos)
    for template_name, prog, verified in verify_scene_proposals(demos, proposals):
        if verified:
            return (template_name, prog)

    # Consensus-controlled compositional search (2-step)
    try:
        from aria.consensus_search import consensus_compose_search
        compose_results = consensus_compose_search(demos)
        for desc, prog, score in compose_results:
            return (f"consensus:{desc}", prog)
    except Exception:
        pass

    return None
