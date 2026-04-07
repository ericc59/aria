"""Factor → symbolic instantiation mapping.

Given a FactorSet, produces SceneProgram skeletons grounded in the
existing executor/IR. Each skeleton is a short (1-3 step) SceneProgram
that the executor can run and the verifier can check.

This is the heart of the factorized architecture turn: it bridges
the learned factor predictions to the concrete symbolic system.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aria.core.grid_perception import GridPerceptionState
from aria.core.scene_executor import make_scene_program, make_step
from aria.factors import (
    Correspondence,
    Decomposition,
    Depth,
    FactorSet,
    Op,
    Scope,
    Selector,
    is_compatible,
)
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Selector predicates to try per selector type
# ---------------------------------------------------------------------------

_OBJECT_PREDICATES = [
    "largest_bbox_area", "smallest_bbox_area", "most_non_bg",
    "top_left", "bottom_right", "unique_shape", "not_touches_border",
]

_TRANSFORMS = [
    "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def instantiate_factor_set(
    factors: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Produce concrete SceneProgram candidates for a factor combination.

    Returns zero or more programs. Each is ready for exact verification.
    The number is bounded by the factor values — at most ~20 per FactorSet
    (from iterating predicates/transforms/colors).
    """
    if not is_compatible(factors):
        return []
    if not demos:
        return []

    programs: list[SceneProgram] = []

    if factors.depth == Depth.ONE:
        programs.extend(_instantiate_depth1(factors, demos, perceptions))
    elif factors.depth == Depth.TWO:
        programs.extend(_instantiate_depth2(factors, demos, perceptions))
    elif factors.depth == Depth.THREE:
        programs.extend(_instantiate_depth3(factors, demos, perceptions))

    return programs


# ---------------------------------------------------------------------------
# Depth-1 skeletons
# ---------------------------------------------------------------------------


def _instantiate_depth1(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    results: list[SceneProgram] = []

    if f.op == Op.EXTRACT:
        results.extend(_skel_select_extract(f, demos, perceptions))
    elif f.op == Op.TRANSFORM:
        results.extend(_skel_global_transform(f, demos, perceptions))
    elif f.op == Op.RECOLOR:
        results.extend(_skel_scope_recolor(f, demos, perceptions))
    elif f.op == Op.FILL:
        results.extend(_skel_scope_fill(f, demos, perceptions))
    elif f.op == Op.COMBINE:
        results.extend(_skel_combine(f, demos, perceptions))
    elif f.op == Op.REPAIR:
        results.extend(_skel_repair(f, demos, perceptions))
    elif f.op == Op.GROW_PROPAGATE:
        results.extend(_skel_grow(f, demos, perceptions))

    return results


# ---------------------------------------------------------------------------
# Depth-2 skeletons
# ---------------------------------------------------------------------------


def _instantiate_depth2(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    results: list[SceneProgram] = []

    if f.op == Op.TRANSFORM and f.selector != Selector.NONE:
        results.extend(_skel_select_then_transform(f, demos, perceptions))
    elif f.op == Op.RECOLOR:
        results.extend(_skel_decompose_then_recolor(f, demos, perceptions))
    elif f.op == Op.FILL:
        results.extend(_skel_scope_fill(f, demos, perceptions))
        # Also try per-entity fill
        if f.decomposition in (Decomposition.OBJECT, Decomposition.REGION):
            results.extend(_skel_per_entity_fill(f, demos, perceptions))
    elif f.op == Op.COPY_STAMP:
        results.extend(_skel_correspond_then_stamp(f, demos, perceptions))
    elif f.op == Op.EXTRACT:
        results.extend(_skel_select_extract(f, demos, perceptions))
    elif f.op == Op.COMBINE:
        results.extend(_skel_combine(f, demos, perceptions))

    return results


# ---------------------------------------------------------------------------
# Depth-3 skeletons
# ---------------------------------------------------------------------------


def _instantiate_depth3(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    results: list[SceneProgram] = []

    if f.op == Op.TRANSFORM:
        results.extend(_skel_select_then_transform(f, demos, perceptions))
    elif f.op == Op.RECOLOR:
        results.extend(_skel_decompose_then_recolor(f, demos, perceptions))

    return results


# ---------------------------------------------------------------------------
# Skeleton: select + extract
# ---------------------------------------------------------------------------


def _skel_select_extract(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → select entity → extract as output."""
    results: list[SceneProgram] = []
    kinds = _entity_kinds_for_decomp(f.decomposition)
    predicates = _predicates_for_selector(f.selector)

    for kind in kinds:
        for pred in predicates:
            for rank in range(min(3, _max_rank(perceptions[0], kind))):
                # Identity (no transform)
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.SELECT_ENTITY,
                        kind=kind, predicate=pred, rank=rank,
                        output_id="sel",
                    ),
                    make_step(StepOp.RENDER_SCENE, source="sel_grid"),
                )
                results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: global/scoped transform
# ---------------------------------------------------------------------------


def _skel_global_transform(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → (optionally select) → transform → render."""
    results: list[SceneProgram] = []

    if f.selector == Selector.NONE and f.scope == Scope.GLOBAL:
        # Global transform: apply to whole grid
        for transform in _TRANSFORMS:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.CANONICALIZE_OBJECT,
                    source="input",
                    transform=transform,
                    output_id="transformed",
                ),
                make_step(StepOp.RENDER_SCENE, source="transformed"),
            )
            results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: select → transform (depth=2+)
# ---------------------------------------------------------------------------


def _skel_select_then_transform(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → select → extract → transform → render."""
    results: list[SceneProgram] = []
    kinds = _entity_kinds_for_decomp(f.decomposition)
    predicates = _predicates_for_selector(f.selector)

    for kind in kinds:
        for pred in predicates:
            for rank in range(min(3, _max_rank(perceptions[0], kind))):
                for transform in _TRANSFORMS:
                    prog = make_scene_program(
                        make_step(StepOp.PARSE_SCENE),
                        make_step(
                            StepOp.SELECT_ENTITY,
                            kind=kind, predicate=pred, rank=rank,
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
                    results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: scoped recolor
# ---------------------------------------------------------------------------


def _skel_scope_recolor(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → infer color map → recolor in scope → render."""
    results: list[SceneProgram] = []

    # 1. Global/scoped color map
    cmap = _infer_color_map(demos)
    if cmap:
        pairs = sorted(cmap.items())
        scope_names = _scope_names_for_scope(f.scope)
        for scope_name in scope_names:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.RECOLOR_OBJECT,
                    color_pairs=pairs, scope=scope_name,
                ),
                make_step(StepOp.RENDER_SCENE, source="recolored"),
            )
            results.append(prog)

    # 2. Per-object conditional recolor (no global color map needed)
    d0 = demos[0]
    if d0.input.shape == d0.output.shape:
        results.extend(_skel_per_object_conditional_recolor(f, demos, perceptions))

    return results


def _skel_per_object_conditional_recolor(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → per-object conditional recolor → render.

    Tries recoloring specific objects matching a predicate.
    Infers from-color/to-color from demo diffs.
    """
    results: list[SceneProgram] = []
    d0 = demos[0]
    if d0.input.shape != d0.output.shape:
        return results

    bg = perceptions[0].bg_color
    diff = d0.input != d0.output
    if not np.any(diff):
        return results

    # Collect per-color changes
    color_changes: dict[tuple[int, int], int] = {}
    for r, c in zip(*np.where(diff)):
        ic, oc = int(d0.input[r, c]), int(d0.output[r, c])
        color_changes[(ic, oc)] = color_changes.get((ic, oc), 0) + 1

    # Try each change pair as a per-object conditional recolor
    for (fc, tc), count in sorted(color_changes.items(), key=lambda x: -x[1]):
        if fc == bg:
            continue  # additive, not recolor
        # Try per-object with different predicates
        for predicate in ("all", "has_color"):
            params = {
                "per_object": True,
                "from_color": fc,
                "to_color": tc,
                "predicate": predicate,
            }
            if predicate == "has_color":
                params["match_color"] = fc
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.RECOLOR_OBJECT, **params),
                make_step(StepOp.RENDER_SCENE, source="recolored"),
            )
            results.append(prog)

    # Also try: for each object, recolor from_color→to_color
    for (fc, tc), count in sorted(color_changes.items(), key=lambda x: -x[1]):
        if fc == bg:
            continue
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(
                StepOp.FOR_EACH_ENTITY,
                kind="object", rule="recolor",
                from_color=fc, to_color=tc,
            ),
            make_step(StepOp.RENDER_SCENE),
        )
        results.append(prog)

    return results


def _skel_decompose_then_recolor(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Depth-2: scope_recolor + per-entity conditional recolor."""
    return _skel_scope_recolor(f, demos, perceptions)


# ---------------------------------------------------------------------------
# Skeleton: scoped fill
# ---------------------------------------------------------------------------


def _skel_scope_fill(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → fill enclosed regions / adjacent bg → render."""
    results: list[SceneProgram] = []
    d0 = demos[0]

    if d0.input.shape != d0.output.shape:
        return results

    # Infer fill colors from demo diff
    diff = d0.input != d0.output
    if not np.any(diff):
        return results

    bg = perceptions[0].bg_color
    fill_colors = sorted(set(int(d0.output[r, c]) for r, c in zip(*np.where(diff))))

    # 1. Fill enclosed regions
    for fc in fill_colors:
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, fill_color=fc),
            make_step(StepOp.RENDER_SCENE, source="filled"),
        )
        results.append(prog)

    # Auto-fill by boundary color
    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.FILL_ENCLOSED_REGIONS, mode="boundary_color"),
        make_step(StepOp.RENDER_SCENE, source="filled"),
    )
    results.append(prog)

    # 2. Per-object fill adjacent bg
    is_additive = all(int(d0.input[r, c]) == bg for r, c in zip(*np.where(diff)))
    if is_additive:
        for fc in fill_colors:
            if fc == bg:
                continue
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.FOR_EACH_ENTITY,
                    kind="object", rule="fill_adjacent_bg",
                    fill_color=fc,
                ),
                make_step(StepOp.RENDER_SCENE),
            )
            results.append(prog)

    # 3. Per-object conditional fill (fill bg within bbox using object's dominant color)
    for rule in ("fill_bbox_holes", "fill_enclosed_bbox"):
        # Use dominant non-bg colors from output diff
        for fc in fill_colors:
            if fc == bg:
                continue
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.FOR_EACH_ENTITY,
                    kind="object", rule=rule,
                    fill_color=fc,
                ),
                make_step(StepOp.RENDER_SCENE),
            )
            results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: combine (partition boolean combine)
# ---------------------------------------------------------------------------


def _skel_combine(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → boolean combine panels → render."""
    results: list[SceneProgram] = []

    if f.decomposition != Decomposition.PARTITION:
        return results

    s0 = perceptions[0]
    if s0.partition is None:
        return results

    for mode in ("overlay", "and", "or", "xor"):
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.BOOLEAN_COMBINE_PANELS, mode=mode),
            make_step(StepOp.RENDER_SCENE, source="combined"),
        )
        results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: repair (periodic pattern repair)
# ---------------------------------------------------------------------------


def _skel_repair(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → extend periodic pattern → render."""
    results: list[SceneProgram] = []

    prog = make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(StepOp.EXTEND_PERIODIC_PATTERN),
        make_step(StepOp.RENDER_SCENE, source="repaired"),
    )
    results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: grow/propagate
# ---------------------------------------------------------------------------


def _skel_grow(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Growth/propagation: extend object pixels along axes or into adjacent bg."""
    results: list[SceneProgram] = []
    d0 = demos[0]

    if d0.input.shape != d0.output.shape:
        return results

    # Grow per-object along row and column axes
    for axis in ("row", "col"):
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(
                StepOp.FOR_EACH_ENTITY,
                kind="object", rule="grow_along_axis",
                axis=axis,
            ),
            make_step(StepOp.RENDER_SCENE),
        )
        results.append(prog)

    # Fill adjacent bg as a form of growth
    diff = d0.input != d0.output
    if np.any(diff):
        bg = perceptions[0].bg_color
        fill_colors = sorted(set(int(d0.output[r, c]) for r, c in zip(*np.where(diff))) - {bg})
        for fc in fill_colors[:3]:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.FOR_EACH_ENTITY,
                    kind="object", rule="fill_adjacent_bg",
                    fill_color=fc,
                ),
                make_step(StepOp.RENDER_SCENE),
            )
            results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Skeleton: correspond + stamp
# ---------------------------------------------------------------------------


def _skel_correspond_then_stamp(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → build correspondence → copy/stamp by match → render.

    Uses BUILD_CORRESPONDENCE + FOR_EACH_ENTITY or MAP_OVER_ENTITIES.
    """
    results: list[SceneProgram] = []

    # Object-to-object correspondence based stamp
    kinds = _entity_kinds_for_decomp(f.decomposition)
    corr_modes = _correspondence_modes(f.correspondence)

    for kind in kinds:
        for corr_mode in corr_modes:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.BUILD_CORRESPONDENCE,
                    kind=kind, mode=corr_mode,
                    output_id="corr",
                ),
                make_step(
                    StepOp.FOR_EACH_ENTITY,
                    kind=kind, rule="stamp_by_correspondence",
                    correspondence="corr",
                ),
                make_step(StepOp.RENDER_SCENE),
            )
            results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Per-entity fill (FOR_EACH object fill variants)
# ---------------------------------------------------------------------------


def _skel_per_entity_fill(
    f: FactorSet,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[SceneProgram]:
    """Parse → for each object → fill enclosed in bbox → render."""
    results: list[SceneProgram] = []

    for color in range(10):
        for conn in [4, 8]:
            for rule in ("fill_bbox_holes", "fill_enclosed_bbox"):
                prog = make_scene_program(
                    make_step(StepOp.PARSE_SCENE),
                    make_step(
                        StepOp.FOR_EACH_ENTITY,
                        kind="object", rule=rule,
                        fill_color=color, connectivity=conn,
                    ),
                    make_step(StepOp.RENDER_SCENE),
                )
                results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Helpers: map factors to IR constructs
# ---------------------------------------------------------------------------


def _entity_kinds_for_decomp(d: Decomposition) -> list[str]:
    """Map decomposition factor to entity kind strings."""
    if d == Decomposition.OBJECT:
        return ["object"]
    if d == Decomposition.FRAME:
        return ["interior_region", "boundary"]
    if d == Decomposition.REGION:
        return ["object", "interior_region"]
    if d == Decomposition.PARTITION:
        return ["panel"]
    if d == Decomposition.ZONE:
        return ["panel"]
    if d == Decomposition.MASK:
        return ["object"]
    if d == Decomposition.PROPAGATION:
        return ["object"]
    return ["object"]


def _predicates_for_selector(s: Selector) -> list[str]:
    """Map selector factor to predicate names."""
    if s == Selector.OBJECT_SELECT:
        return _OBJECT_PREDICATES
    if s == Selector.REGION_SELECT:
        return ["largest_bbox_area", "smallest_bbox_area"]
    if s == Selector.FRAME_INTERIOR:
        return ["largest_bbox_area"]
    if s == Selector.CELL_PANEL:
        return ["top_left", "bottom_right", "most_non_bg"]
    if s == Selector.MARKER:
        return ["smallest_pixel_count"]
    if s == Selector.ENCLOSED:
        return ["largest_bbox_area"]
    if s == Selector.NONE:
        return ["largest_bbox_area"]
    return _OBJECT_PREDICATES


def _scope_names_for_scope(s: Scope) -> list[str]:
    """Map scope factor to executor scope parameter values."""
    if s == Scope.GLOBAL:
        return ["global"]
    if s == Scope.OBJECT:
        return ["objects"]
    if s == Scope.OBJECT_BBOX:
        return ["object_bboxes"]
    if s == Scope.FRAME_INTERIOR:
        return ["frame_interior"]
    if s == Scope.PARTITION_CELL:
        return ["partition_cells"]
    if s == Scope.ENCLOSED_SUBSET:
        return ["enclosed"]
    if s == Scope.LOCAL_SUPPORT:
        return ["local"]
    if s == Scope.REGION_LOCAL:
        return ["region"]
    return ["global"]


def _correspondence_modes(c: Correspondence) -> list[str]:
    """Map correspondence factor to BUILD_CORRESPONDENCE mode strings."""
    if c == Correspondence.POSITIONAL:
        return ["positional"]
    if c == Correspondence.SHAPE_BASED:
        return ["shape"]
    if c == Correspondence.SOURCE_TARGET:
        return ["source_target"]
    if c == Correspondence.ASSIGNMENT:
        return ["assignment"]
    if c == Correspondence.OBJECT_MATCH:
        return ["color_match", "size_match"]
    return []


def _max_rank(perception: GridPerceptionState, kind: str) -> int:
    """Max entities of a kind available in the first demo."""
    if kind == "object":
        return max(len(perception.objects.objects), 1)
    if kind == "panel":
        if perception.partition is not None:
            return max(len(perception.partition.cells), 1)
        return 0
    if kind in ("interior_region", "boundary"):
        return max(len(perception.framed_regions), 1)
    return 1


def _infer_color_map(demos: tuple[DemoPair, ...]) -> dict[int, int] | None:
    """Infer a consistent color map across all demos."""
    cmap: dict[int, int] = {}
    for demo in demos:
        if demo.input.shape != demo.output.shape:
            return None
        diff = demo.input != demo.output
        if not np.any(diff):
            continue
        for r, c in zip(*np.where(diff)):
            ic, oc = int(demo.input[r, c]), int(demo.output[r, c])
            if ic in cmap and cmap[ic] != oc:
                return None
            cmap[ic] = oc
    return cmap if cmap and len(cmap) <= 5 else None
