"""Scene program executor.

Executes multi-step ScenePrograms over typed intermediate scene state.
Each step operates on a SceneState that carries entities, relations,
the input/output grids, and named intermediate values.

The executor is the composition layer between perception (which detects
structures) and rendering (which produces output grids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.scene_ir import (
    BBox,
    EntityKind,
    OutputGridSpec,
    RelationKind,
    Scene,
    SceneEntity,
    SceneProgram,
    SceneRelation,
    SceneStep,
    StepOp,
)
from aria.types import Grid


# ---------------------------------------------------------------------------
# Scene state
# ---------------------------------------------------------------------------


@dataclass
class SceneState:
    """Mutable state carried through scene program execution."""

    input_grid: Grid
    perception: GridPerceptionState
    output_spec: OutputGridSpec | None = None
    output_grid: Grid | None = None
    scene: Scene | None = None
    entities: dict[str, SceneEntity] = field(default_factory=dict)
    relations: list[SceneRelation] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: SceneEntity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: SceneRelation) -> None:
        self.relations.append(relation)

    def get_entity(self, entity_id: str) -> SceneEntity | None:
        return self.entities.get(entity_id)

    def get_grid_region(self, bbox: BBox) -> Grid:
        r0, c0, r1, c1 = bbox
        return self.input_grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def entities_of_kind(self, kind: EntityKind) -> list[SceneEntity]:
        return [e for e in self.entities.values() if e.kind == kind]


# ---------------------------------------------------------------------------
# Step handler registry
# ---------------------------------------------------------------------------

_STEP_HANDLERS: dict[StepOp, Callable[[SceneState, SceneStep], None]] = {}


def register_step(op: StepOp):
    """Decorator to register a step handler."""

    def decorator(fn: Callable[[SceneState, SceneStep], None]):
        _STEP_HANDLERS[op] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class SceneExecutionError(Exception):
    pass


def execute_scene_program(
    program: SceneProgram,
    input_grid: Grid,
) -> Grid:
    """Execute a scene program and return the output grid.

    Execution contract (Option B):
    Scene programs may solve tasks even when stage-1 size inference fails.
    They must produce output_grid through one of:
    1. Explicit: INFER_OUTPUT_SIZE → INITIALIZE_OUTPUT_SCENE → ... → RENDER_SCENE
    2. Implicit: EXTRACT_TEMPLATE → ... → RENDER_SCENE(source=value)
    The executor does NOT require stage-1 size as a precondition.
    Exact verification against demos is the final arbiter.
    """
    state = SceneState(
        input_grid=input_grid,
        perception=perceive_grid(input_grid),
    )

    for idx, step in enumerate(program.steps):
        handler = _STEP_HANDLERS.get(step.op)
        if handler is None:
            raise SceneExecutionError(
                f"Step {idx}: no handler for {step.op.value}"
            )
        try:
            handler(state, step)
        except SceneExecutionError:
            raise
        except Exception as exc:
            raise SceneExecutionError(
                f"Step {idx} ({step.op.value}): {exc}"
            ) from exc

    if state.output_grid is not None:
        return state.output_grid

    raise SceneExecutionError("Program did not produce an output grid")


# ---------------------------------------------------------------------------
# Canonical shape key
# ---------------------------------------------------------------------------


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Compute a canonical shape key for an object mask.

    Normalizes the mask by trying all 8 orientations (4 rotations × 2 flips)
    and returning the lexicographically smallest bytes representation.
    This makes shape comparison invariant to rotation and reflection.
    """
    m = mask.astype(np.uint8)
    variants = []
    for rot in range(4):
        mr = np.rot90(m, rot)
        variants.append(mr.tobytes())
        variants.append(np.fliplr(mr).tobytes())
    return min(variants)


# ---------------------------------------------------------------------------
# Object group creation (frame-interior + proximity-based)
# ---------------------------------------------------------------------------


def _add_object_groups(state: SceneState) -> None:
    """Create OBJECT_GROUP entities from framed regions and proximity clusters."""
    bg = state.perception.bg_color
    grid = state.input_grid
    grouped_obj_ids: set[str] = set()
    group_idx = 0

    # B1: Frame-interior groups — one per BOUNDARY entity
    boundaries = state.entities_of_kind(EntityKind.BOUNDARY)
    for boundary in boundaries:
        if boundary.bbox is None:
            continue
        br0, bc0, br1, bc1 = boundary.bbox
        # Find all OBJECT entities whose bbox is fully inside the frame bbox
        member_ids = [boundary.id]
        for eid, ent in state.entities.items():
            if ent.kind != EntityKind.OBJECT or ent.bbox is None:
                continue
            er0, ec0, er1, ec1 = ent.bbox
            if er0 >= br0 and ec0 >= bc0 and er1 <= br1 and ec1 <= bc1:
                member_ids.append(eid)
                grouped_obj_ids.add(eid)
        # Count distinct non-bg colors
        region = grid[br0:br1 + 1, bc0:bc1 + 1]
        non_bg_colors = set(int(v) for v in region.ravel() if int(v) != bg)
        gid = f"object_group_{group_idx}"
        group_idx += 1
        state.add_entity(SceneEntity(
            id=gid,
            kind=EntityKind.OBJECT_GROUP,
            bbox=(br0, bc0, br1, bc1),
            attrs={
                "member_object_ids": tuple(member_ids),
                "n_colors": len(non_bg_colors),
                "has_frame": True,
                "group_source": "frame",
            },
        ))

    # B2: Proximity-based groups — cluster ungrouped objects by bbox adjacency
    ungrouped = [
        (eid, ent) for eid, ent in state.entities.items()
        if ent.kind == EntityKind.OBJECT
        and ent.bbox is not None
        and eid not in grouped_obj_ids
        and not ent.attrs.get("is_singleton", False)
    ]
    if ungrouped:
        # Union-Find for clustering by bbox proximity (within 1 pixel)
        id_list = [eid for eid, _ in ungrouped]
        ent_map = {eid: ent for eid, ent in ungrouped}
        parent: dict[str, str] = {eid: eid for eid in id_list}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def _bboxes_adjacent(b1: tuple, b2: tuple) -> bool:
            r0a, c0a, r1a, c1a = b1
            r0b, c0b, r1b, c1b = b2
            # Check if bboxes overlap or are within 1 pixel
            if r1a + 1 < r0b or r1b + 1 < r0a:
                return False
            if c1a + 1 < c0b or c1b + 1 < c0a:
                return False
            return True

        for i, (eid_a, ent_a) in enumerate(ungrouped):
            for eid_b, ent_b in ungrouped[i + 1:]:
                if _bboxes_adjacent(ent_a.bbox, ent_b.bbox):
                    union(eid_a, eid_b)

        # Collect clusters with >1 member
        from collections import defaultdict as _defaultdict
        clusters: dict[str, list[str]] = _defaultdict(list)
        for eid in id_list:
            clusters[find(eid)].append(eid)

        for members in clusters.values():
            if len(members) < 2:
                continue
            # Compute union bbox
            all_bboxes = [ent_map[m].bbox for m in members]
            ur0 = min(b[0] for b in all_bboxes)
            uc0 = min(b[1] for b in all_bboxes)
            ur1 = max(b[2] for b in all_bboxes)
            uc1 = max(b[3] for b in all_bboxes)
            region = grid[ur0:ur1 + 1, uc0:uc1 + 1]
            non_bg_colors = set(int(v) for v in region.ravel() if int(v) != bg)
            gid = f"object_group_{group_idx}"
            group_idx += 1
            state.add_entity(SceneEntity(
                id=gid,
                kind=EntityKind.OBJECT_GROUP,
                bbox=(ur0, uc0, ur1, uc1),
                attrs={
                    "member_object_ids": tuple(members),
                    "n_colors": len(non_bg_colors),
                    "has_frame": False,
                    "group_source": "proximity",
                },
            ))


# ---------------------------------------------------------------------------
# Guard predicates for FOR_EACH_ENTITY filtering
# ---------------------------------------------------------------------------


def _evaluate_guard(guard_name: str, entity: SceneEntity, state: SceneState) -> bool:
    """Evaluate a guard predicate on an entity. Returns True if entity passes."""
    fn = _ENTITY_GUARDS.get(guard_name)
    if fn is None:
        return True  # unknown guard → don't filter
    return fn(entity, state)


def _guard_has_enclosed_bg(entity: SceneEntity, state: SceneState) -> bool:
    """Entity bbox contains bg not touching bbox border."""
    if entity.bbox is None:
        return False
    from scipy import ndimage
    r0, c0, r1, c1 = entity.bbox
    region = state.input_grid[r0:r1 + 1, c0:c1 + 1]
    bg = state.perception.bg_color
    bg_mask = region == bg
    if not np.any(bg_mask):
        return False
    labeled, n = ndimage.label(bg_mask)
    if n == 0:
        return False
    h, w = region.shape
    border_labels = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])
    return any(lbl not in border_labels for lbl in range(1, n + 1))


def _guard_is_frame_like(entity: SceneEntity, state: SceneState) -> bool:
    """Entity bbox overlaps a BOUNDARY entity."""
    if entity.bbox is None:
        return False
    er0, ec0, er1, ec1 = entity.bbox
    for b in state.entities_of_kind(EntityKind.BOUNDARY):
        if b.bbox is None:
            continue
        br0, bc0, br1, bc1 = b.bbox
        if er0 <= br1 and er1 >= br0 and ec0 <= bc1 and ec1 >= bc0:
            return True
    return False


def _guard_has_interior(entity: SceneEntity, state: SceneState) -> bool:
    """Entity encloses a bg region (ring/frame shape)."""
    return _guard_has_enclosed_bg(entity, state)


def _guard_touches_border(entity: SceneEntity, state: SceneState) -> bool:
    """Entity bbox touches grid border."""
    if entity.bbox is None:
        return False
    r0, c0, r1, c1 = entity.bbox
    h, w = state.input_grid.shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _guard_is_largest(entity: SceneEntity, state: SceneState) -> bool:
    """Largest by size among entities of same kind."""
    size = entity.attrs.get("size", 0)
    for e in state.entities_of_kind(entity.kind):
        if e.id != entity.id and e.attrs.get("size", 0) > size:
            return False
    return True


def _guard_is_smallest(entity: SceneEntity, state: SceneState) -> bool:
    """Smallest by size among entities of same kind."""
    size = entity.attrs.get("size", float("inf"))
    for e in state.entities_of_kind(entity.kind):
        if e.id != entity.id:
            other_size = e.attrs.get("size", float("inf"))
            if other_size < size:
                return False
    return True


def _guard_multi_color_bbox(entity: SceneEntity, state: SceneState) -> bool:
    """Entity bbox contains >1 non-bg color."""
    if entity.bbox is None:
        return False
    r0, c0, r1, c1 = entity.bbox
    region = state.input_grid[r0:r1 + 1, c0:c1 + 1]
    bg = state.perception.bg_color
    non_bg_colors = set(int(v) for v in region.ravel() if int(v) != bg)
    return len(non_bg_colors) > 1


_ENTITY_GUARDS: dict[str, Callable[[SceneEntity, SceneState], bool]] = {
    "has_enclosed_bg": _guard_has_enclosed_bg,
    "no_enclosed_bg": lambda e, s: not _guard_has_enclosed_bg(e, s),
    "is_frame_like": _guard_is_frame_like,
    "not_frame_like": lambda e, s: not _guard_is_frame_like(e, s),
    "has_interior": _guard_has_interior,
    "no_interior": lambda e, s: not _guard_has_interior(e, s),
    "touches_border": _guard_touches_border,
    "not_touches_border": lambda e, s: not _guard_touches_border(e, s),
    "is_largest": _guard_is_largest,
    "is_smallest": _guard_is_smallest,
    "multi_color_bbox": _guard_multi_color_bbox,
    "single_color_bbox": lambda e, s: not _guard_multi_color_bbox(e, s),
}


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


@register_step(StepOp.PARSE_SCENE)
def _parse_scene(state: SceneState, step: SceneStep) -> None:
    """Parse the input grid into scene entities using perception."""
    s = state.perception
    bg = s.bg_color

    # Add scene entity
    scene_entity = SceneEntity(
        id="scene",
        kind=EntityKind.SCENE,
        bbox=(0, 0, s.dims[0] - 1, s.dims[1] - 1),
        attrs={"background": bg, "shape": s.dims},
    )
    state.add_entity(scene_entity)

    # Add partition cells as panel entities
    if s.partition is not None:
        sep_entity = SceneEntity(
            id="separator",
            kind=EntityKind.SEPARATOR,
            attrs={
                "color": s.partition.separator_color,
                "n_rows": s.partition.n_rows,
                "n_cols": s.partition.n_cols,
            },
        )
        state.add_entity(sep_entity)

        for cell in s.partition.cells:
            panel_id = f"panel_{cell.row_idx}_{cell.col_idx}"
            panel = SceneEntity(
                id=panel_id,
                kind=EntityKind.PANEL,
                bbox=cell.bbox,
                attrs={
                    "row_idx": cell.row_idx,
                    "col_idx": cell.col_idx,
                    "dims": cell.dims,
                },
            )
            state.add_entity(panel)
            state.add_relation(SceneRelation(
                kind=RelationKind.CONTAINS,
                source_id="scene",
                target_id=panel_id,
            ))

    # Add objects (4-conn) with canonical shape keys
    from collections import Counter as _Counter

    shape_keys_4: list[bytes] = []
    for obj in s.objects.objects:
        shape_keys_4.append(_canonical_shape_key(obj.mask))
    sk_counts_4 = _Counter(shape_keys_4)

    for idx, obj in enumerate(s.objects.objects):
        obj_id = f"object_{idx}"
        sk = shape_keys_4[idx]
        obj_entity = SceneEntity(
            id=obj_id,
            kind=EntityKind.OBJECT,
            bbox=(obj.row, obj.col, obj.row + obj.bbox_h - 1, obj.col + obj.bbox_w - 1),
            colors=(int(obj.color),),
            attrs={
                "size": obj.size,
                "is_singleton": obj.is_singleton,
                "bbox_h": obj.bbox_h,
                "bbox_w": obj.bbox_w,
                "shape_key": sk,
                "shape_unique": sk_counts_4[sk] == 1,
                "connectivity": 4,
            },
        )
        state.add_entity(obj_entity)

    # Add objects (8-conn) with canonical shape keys
    shape_keys_8: list[bytes] = []
    for obj in s.objects8.objects:
        shape_keys_8.append(_canonical_shape_key(obj.mask))
    sk_counts_8 = _Counter(shape_keys_8)

    for idx, obj in enumerate(s.objects8.objects):
        obj_id = f"object8_{idx}"
        sk = shape_keys_8[idx]
        obj_entity = SceneEntity(
            id=obj_id,
            kind=EntityKind.OBJECT,
            bbox=(obj.row, obj.col, obj.row + obj.bbox_h - 1, obj.col + obj.bbox_w - 1),
            colors=(int(obj.color),),
            attrs={
                "size": obj.size,
                "is_singleton": obj.is_singleton,
                "bbox_h": obj.bbox_h,
                "bbox_w": obj.bbox_w,
                "shape_key": sk,
                "shape_unique": sk_counts_8[sk] == 1,
                "connectivity": 8,
            },
        )
        state.add_entity(obj_entity)

    # Add framed regions
    # FramedRegion.(row, col, height, width) describe the INTERIOR region
    for idx, fr in enumerate(s.framed_regions):
        fr_id = f"frame_{idx}"
        # Outer frame bbox includes the 1-pixel border
        outer_r0 = max(0, fr.row - 1)
        outer_c0 = max(0, fr.col - 1)
        outer_r1 = min(s.dims[0] - 1, fr.row + fr.height)
        outer_c1 = min(s.dims[1] - 1, fr.col + fr.width)
        fr_entity = SceneEntity(
            id=fr_id,
            kind=EntityKind.BOUNDARY,
            bbox=(outer_r0, outer_c0, outer_r1, outer_c1),
            colors=(int(fr.frame_color),),
            attrs={"height": fr.height, "width": fr.width},
        )
        state.add_entity(fr_entity)
        # Interior bbox
        interior_id = f"frame_{idx}_interior"
        state.add_entity(SceneEntity(
            id=interior_id,
            kind=EntityKind.INTERIOR_REGION,
            bbox=(fr.row, fr.col, fr.row + fr.height - 1, fr.col + fr.width - 1),
        ))
        state.add_relation(SceneRelation(
            kind=RelationKind.ENCLOSES,
            source_id=fr_id,
            target_id=interior_id,
        ))

    # Add color bbox entities (one per non-bg color)
    for color, bbox in s.color_bboxes.items():
        if color == bg:
            continue
        r0, c0, r1, c1 = bbox
        cbbox_id = f"color_bbox_{color}"
        cbbox_entity = SceneEntity(
            id=cbbox_id,
            kind=EntityKind.OBJECT,
            bbox=(r0, c0, r1, c1),
            colors=(int(color),),
            attrs={
                "selector_kind": "color_bbox",
                "color": int(color),
                "pixel_count": int(s.color_pixel_counts.get(color, 0)),
            },
        )
        state.add_entity(cbbox_entity)

    # Add legend if present
    if s.legend is not None:
        leg = s.legend
        legend_entity = SceneEntity(
            id="legend",
            kind=EntityKind.LEGEND_MAP,
            bbox=leg.region_bbox,
            attrs={
                "edge": leg.edge,
                "entries": tuple(
                    (int(e.key_color), int(e.value_color)) for e in leg.entries
                ),
            },
        )
        state.add_entity(legend_entity)

    # Add object groups (frame-interior and proximity-based)
    _add_object_groups(state)

    state.scene = Scene(
        shape=s.dims,
        background=bg,
        entities=tuple(state.entities.values()),
        relations=tuple(state.relations),
    )


@register_step(StepOp.INFER_OUTPUT_SIZE)
def _infer_output_size(state: SceneState, step: SceneStep) -> None:
    """Infer output grid size from step params, entity, or selected value.

    Params:
        shape: explicit (rows, cols) tuple
        source: entity id to get bbox from
        from_value: name of a value in state.values (entity id or grid)
            If the value is a string, look up that entity's bbox.
            If the value is a grid (ndarray), use its shape.
    """
    shape = step.params.get("shape")
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        state.output_spec = OutputGridSpec(shape=(int(shape[0]), int(shape[1])))
        return

    # Try explicit entity reference
    source = step.params.get("source")
    if isinstance(source, str):
        entity = state.get_entity(source)
        if entity is not None and entity.bbox is not None:
            r0, c0, r1, c1 = entity.bbox
            state.output_spec = OutputGridSpec(shape=(r1 - r0 + 1, c1 - c0 + 1))
            return

    # Try value reference (from SELECT_ENTITY output)
    from_value = step.params.get("from_value")
    if isinstance(from_value, str) and from_value in state.values:
        val = state.values[from_value]
        if isinstance(val, np.ndarray):
            state.output_spec = OutputGridSpec(shape=val.shape)
            return
        if isinstance(val, str):
            # It's an entity id
            entity = state.get_entity(val)
            if entity is not None and entity.bbox is not None:
                r0, c0, r1, c1 = entity.bbox
                state.output_spec = OutputGridSpec(shape=(r1 - r0 + 1, c1 - c0 + 1))
                return

    raise SceneExecutionError("Cannot infer output size")


@register_step(StepOp.INFER_OUTPUT_BACKGROUND)
def _infer_output_background(state: SceneState, step: SceneStep) -> None:
    """Set output background color."""
    bg = step.params.get("background")
    if bg is not None:
        if state.output_spec is not None:
            state.output_spec = OutputGridSpec(
                shape=state.output_spec.shape,
                background=int(bg),
            )
        return

    # Default to input background
    if state.output_spec is not None:
        state.output_spec = OutputGridSpec(
            shape=state.output_spec.shape,
            background=state.perception.bg_color,
        )


@register_step(StepOp.INITIALIZE_OUTPUT_SCENE)
def _initialize_output(state: SceneState, step: SceneStep) -> None:
    """Create the output grid filled with background."""
    if state.output_spec is None:
        raise SceneExecutionError("No output spec")
    h, w = state.output_spec.shape
    bg = state.output_spec.background
    if bg is None:
        bg = state.perception.bg_color
    state.output_grid = np.full((h, w), bg, dtype=state.input_grid.dtype)


@register_step(StepOp.SPLIT_BY_SEPARATOR)
def _split_by_separator(state: SceneState, step: SceneStep) -> None:
    """Ensure partition cells are parsed as panel entities."""
    # Already done by PARSE_SCENE if partition exists
    pass


@register_step(StepOp.EXTRACT_TEMPLATE)
def _extract_template(state: SceneState, step: SceneStep) -> None:
    """Extract a named entity's grid content as a template value."""
    source = step.params.get("source")
    if not isinstance(source, str):
        return
    entity = state.get_entity(source)
    if entity is None or entity.bbox is None:
        return
    grid = state.get_grid_region(entity.bbox)
    output_id = step.output_id or f"template_{source}"
    state.values[output_id] = grid
    state.add_entity(SceneEntity(
        id=output_id,
        kind=EntityKind.TEMPLATE,
        bbox=entity.bbox,
        attrs={"source": source},
    ))


@register_step(StepOp.LOOKUP_COLOR)
def _lookup_color(state: SceneState, step: SceneStep) -> None:
    """Look up a color from a legend mapping."""
    legend = state.get_entity("legend")
    if legend is None:
        return
    entries = legend.attrs.get("entries", ())
    key_color = step.params.get("key_color")
    if key_color is None:
        return
    for kc, vc in entries:
        if kc == int(key_color):
            output_id = step.output_id or "looked_up_color"
            state.values[output_id] = vc
            return


@register_step(StepOp.RECOLOR_OBJECT)
def _recolor_object(state: SceneState, step: SceneStep) -> None:
    """Recolor pixels in the output grid.

    Supports two modes:
    1. Single pair: from_color -> to_color (optionally region-scoped)
    2. Multi-pair with scope: color_pairs + scope (objects/frame_interior/partition_cells)
    """
    if state.output_grid is None:
        # If no output grid yet, start from input copy
        state.output_grid = state.input_grid.copy()

    color_pairs = step.params.get("color_pairs")
    scope = step.params.get("scope")

    if isinstance(color_pairs, (list, tuple)) and color_pairs:
        _apply_scoped_color_map(state, color_pairs, scope or "global")
        state.values["recolored"] = state.output_grid.copy()
        return

    from_color = step.params.get("from_color")
    to_color = step.params.get("to_color")
    region = step.params.get("region")  # optional bbox
    per_object = step.params.get("per_object")  # per-object conditional

    if per_object:
        # Per-object conditional recolor: recolor only objects matching a predicate
        predicate = step.params.get("predicate", "all")
        bg = state.perception.bg_color
        for obj in state.perception.objects.objects:
            if obj.color == bg:
                continue
            r0o, c0o = obj.row, obj.col
            r1o, c1o = r0o + obj.bbox_h, c0o + obj.bbox_w
            orig = state.input_grid[r0o:r1o, c0o:c1o]
            non_bg = orig[orig != bg]
            if len(non_bg) == 0:
                continue
            dom = int(np.bincount(non_bg.astype(int), minlength=10).argmax())
            match = (
                predicate == "all"
                or (predicate == "has_color" and dom == step.params.get("match_color"))
                or (predicate == "is_singleton" and obj.size == 1)
                or (predicate == "is_largest" and obj.size == max(o.size for o in state.perception.objects.objects if o.color != bg))
            )
            if not match:
                continue
            out_region = state.output_grid[r0o:r1o, c0o:c1o]
            if from_color is not None and to_color is not None:
                out_region[orig == int(from_color)] = int(to_color)
            elif to_color is not None:
                # Recolor all non-bg pixels to to_color
                out_region[orig != bg] = int(to_color)
        state.values["recolored"] = state.output_grid.copy()
        return

    if from_color is None or to_color is None:
        return

    if region is not None and isinstance(region, (tuple, list)) and len(region) == 4:
        r0, c0, r1, c1 = [int(x) for x in region]
        sub = state.output_grid[r0 : r1 + 1, c0 : c1 + 1]
        sub[sub == int(from_color)] = int(to_color)
    else:
        state.output_grid[state.output_grid == int(from_color)] = int(to_color)


def _apply_scoped_color_map(
    state: SceneState,
    color_pairs: list,
    scope: str,
) -> None:
    """Apply color substitution with scope control."""
    grid = state.output_grid
    if grid is None:
        return

    if scope == "global":
        temp = grid.copy()
        for fc, tc in color_pairs:
            grid[temp == int(fc)] = int(tc)
    elif scope == "objects":
        # Apply only to non-bg pixels
        bg = state.perception.bg_color
        temp = grid.copy()
        for fc, tc in color_pairs:
            mask = (temp == int(fc)) & (temp != bg)
            grid[mask] = int(tc)
    elif scope == "object_bboxes":
        # Apply within bounding boxes of all non-bg objects (includes bg within bbox)
        bg = state.perception.bg_color
        temp = grid.copy()
        applied = np.zeros(grid.shape, dtype=bool)
        for obj in state.perception.objects.objects:
            if obj.color == bg:
                continue
            r0, c0 = obj.row, obj.col
            r1, c1 = r0 + obj.bbox_h, c0 + obj.bbox_w
            applied[r0:r1, c0:c1] = True
        for fc, tc in color_pairs:
            mask = (temp == int(fc)) & applied
            grid[mask] = int(tc)
    elif scope == "frame_interior":
        # Apply only inside framed regions
        temp = grid.copy()
        for fr in state.perception.framed_regions:
            r0, c0 = fr.row, fr.col
            r1, c1 = r0 + fr.height, c0 + fr.width
            sub_temp = temp[r0:r1, c0:c1]
            sub_grid = grid[r0:r1, c0:c1]
            for fc, tc in color_pairs:
                sub_grid[sub_temp == int(fc)] = int(tc)
    elif scope == "partition_cells":
        # Apply only inside partition cell interiors
        if state.perception.partition is None:
            return
        temp = grid.copy()
        for cell in state.perception.partition.cells:
            r0, c0, r1, c1 = cell.bbox
            sub_temp = temp[r0:r1+1, c0:c1+1]
            sub_grid = grid[r0:r1+1, c0:c1+1]
            for fc, tc in color_pairs:
                sub_grid[sub_temp == int(fc)] = int(tc)


@register_step(StepOp.BOOLEAN_COMBINE_PANELS)
def _boolean_combine_panels(state: SceneState, step: SceneStep) -> None:
    """Combine partition panels with a boolean operation."""
    if state.output_grid is None:
        # Initialize from input
        state.output_grid = state.input_grid.copy()

    op = step.params.get("operation") or step.params.get("mode", "overlay")
    panels = state.entities_of_kind(EntityKind.PANEL)
    if len(panels) < 2:
        return

    bg = state.perception.bg_color

    # Get all panel grids
    panel_grids = []
    for panel in panels:
        if panel.bbox is None:
            continue
        g = state.get_grid_region(panel.bbox)
        panel_grids.append(g)

    if not panel_grids:
        return

    # All panels must be same size
    shape0 = panel_grids[0].shape
    if not all(g.shape == shape0 for g in panel_grids):
        return

    if op == "overlay":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        for g in panel_grids:
            mask = g != bg
            result[mask] = g[mask]
    elif op == "and":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        # Pixels that are non-bg in ALL panels
        all_non_bg = np.ones(shape0, dtype=bool)
        for g in panel_grids:
            all_non_bg &= (g != bg)
        # Use color from first panel where all agree
        result[all_non_bg] = panel_grids[0][all_non_bg]
    elif op == "xor":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        # Pixels non-bg in exactly 1 panel
        count = np.zeros(shape0, dtype=int)
        for g in panel_grids:
            count += (g != bg).astype(int)
        xor_mask = count == 1
        for g in panel_grids:
            mask = (g != bg) & xor_mask
            result[mask] = g[mask]
    elif op == "or" or op == "or_any_color":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        # Pixels non-bg in ANY panel
        any_non_bg = np.zeros(shape0, dtype=bool)
        for g in panel_grids:
            any_non_bg |= (g != bg)
        # Use color from last panel that has non-bg at each position
        for g in panel_grids:
            mask = (g != bg) & any_non_bg
            result[mask] = g[mask]
    elif op.startswith("or_color_"):
        # OR with explicit target color: "or_color_3" → fill OR mask with color 3
        try:
            target_color = int(op.split("_")[-1])
        except ValueError:
            return
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        any_non_bg = np.zeros(shape0, dtype=bool)
        for g in panel_grids:
            any_non_bg |= (g != bg)
        result[any_non_bg] = target_color
    else:
        return

    # Write to output
    h, w = state.output_grid.shape
    rh, rw = result.shape
    if (h, w) == (rh, rw):
        state.output_grid[:] = result
    elif h >= rh and w >= rw:
        state.output_grid[:rh, :rw] = result

    # Store result for RENDER_SCENE source="combined"
    state.values["combined"] = state.output_grid.copy()


@register_step(StepOp.RENDER_SCENE)
def _render_scene(state: SceneState, step: SceneStep) -> None:
    """Finalize the output grid. Copy a named value if specified."""
    source = step.params.get("source")
    if isinstance(source, str) and source in state.values:
        val = state.values[source]
        if isinstance(val, np.ndarray):
            state.output_grid = val.copy()
            return

    # If output grid is already set, we're done
    if state.output_grid is not None:
        return

    raise SceneExecutionError("No output to render")


@register_step(StepOp.FILL_ENCLOSED_REGIONS)
def _fill_enclosed_regions(state: SceneState, step: SceneStep) -> None:
    """Fill enclosed regions in the output grid.

    Modes:
    - fill_color=N: fill all interior bg regions with color N
    - mode="boundary_color": fill each interior region with its boundary's dominant non-bg color
    """
    if state.output_grid is None:
        # Start from input copy
        state.output_grid = state.input_grid.copy()

    from scipy import ndimage

    bg = state.perception.bg_color
    grid = state.output_grid
    bg_mask = grid == bg
    labeled, n = ndimage.label(bg_mask)
    if n < 2:
        return

    h, w = grid.shape
    border_labels: set[int] = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    mode = step.params.get("mode")
    fill_color = step.params.get("fill_color")

    if mode == "boundary_color":
        # Each interior region gets its boundary's dominant non-bg color
        for region_id in range(1, n + 1):
            if region_id in border_labels:
                continue
            region_mask = labeled == region_id
            dilated = ndimage.binary_dilation(region_mask)
            boundary = dilated & ~region_mask
            boundary_colors = grid[boundary]
            boundary_non_bg = boundary_colors[boundary_colors != bg]
            if len(boundary_non_bg) == 0:
                continue
            vals, counts = np.unique(boundary_non_bg, return_counts=True)
            fc = int(vals[np.argmax(counts)])
            state.output_grid[region_mask] = fc
    elif fill_color is not None:
        for region_id in range(1, n + 1):
            if region_id not in border_labels:
                state.output_grid[labeled == region_id] = int(fill_color)
    else:
        # Try color_role
        color_role = step.params.get("color_role")
        if color_role is not None:
            resolved = resolve_color_role(color_role, state)
            if resolved is not None:
                for region_id in range(1, n + 1):
                    if region_id not in border_labels:
                        state.output_grid[labeled == region_id] = resolved

    # Store result for RENDER_SCENE source="filled"
    state.values["filled"] = state.output_grid.copy()


@register_step(StepOp.CANONICALIZE_OBJECT)
def _canonicalize_object(state: SceneState, step: SceneStep) -> None:
    """Apply a geometric transform to a named grid value."""
    source = step.params.get("source")
    transform = step.params.get("transform")
    if not isinstance(source, str) or source not in state.values:
        return

    grid = state.values[source]
    if not isinstance(grid, np.ndarray):
        return

    if transform == "rot90":
        result = np.rot90(grid, 1)
    elif transform == "rot180":
        result = np.rot90(grid, 2)
    elif transform == "rot270":
        result = np.rot90(grid, 3)
    elif transform == "flip_lr":
        result = np.fliplr(grid)
    elif transform == "flip_ud":
        result = np.flipud(grid)
    elif transform == "transpose":
        result = grid.T
    else:
        return

    output_id = step.output_id or f"{source}_transformed"
    state.values[output_id] = result.copy()


@register_step(StepOp.STAMP_TEMPLATE)
def _stamp_template(state: SceneState, step: SceneStep) -> None:
    """Stamp a template grid value onto the output at a given position."""
    if state.output_grid is None:
        return

    source = step.params.get("source")
    if not isinstance(source, str):
        return

    # Try named value first, then entity grid region
    template = state.values.get(source)
    if template is None:
        entity = state.get_entity(source)
        if entity is not None and entity.bbox is not None:
            template = state.get_grid_region(entity.bbox)
    if template is None:
        return
    if not isinstance(template, np.ndarray):
        return

    row = int(step.params.get("row", 0))
    col = int(step.params.get("col", 0))
    th, tw = template.shape
    oh, ow = state.output_grid.shape

    # Clip to output bounds
    r_end = min(row + th, oh)
    c_end = min(col + tw, ow)
    state.output_grid[row:r_end, col:c_end] = template[: r_end - row, : c_end - col]


# ---------------------------------------------------------------------------
# Scene program builder helpers
# ---------------------------------------------------------------------------


def make_scene_program(*steps: SceneStep) -> SceneProgram:
    """Convenience: build a SceneProgram from step arguments."""
    return SceneProgram(steps=steps)


def make_step(
    op: StepOp,
    inputs: tuple[str, ...] = (),
    output_id: str | None = None,
    **params: Any,
) -> SceneStep:
    """Convenience: build a SceneStep with keyword params."""
    return SceneStep(op=op, inputs=inputs, params=params, output_id=output_id)


# ---------------------------------------------------------------------------
# SELECT_ENTITY — generic entity selection by predicate
# ---------------------------------------------------------------------------

# Selector predicates. Each takes (entity, state) -> sortable key.
# The selector picks the entity that maximizes the key (or minimizes for _asc).


def _entity_bbox_area(entity: SceneEntity, state: SceneState) -> int:
    if entity.bbox is None:
        return 0
    r0, c0, r1, c1 = entity.bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _entity_non_bg_count(entity: SceneEntity, state: SceneState) -> int:
    if entity.bbox is None:
        return 0
    g = state.get_grid_region(entity.bbox)
    return int(np.sum(g != state.perception.bg_color))


def _entity_color_count(entity: SceneEntity, state: SceneState) -> int:
    if entity.bbox is None:
        return 0
    g = state.get_grid_region(entity.bbox)
    bg = state.perception.bg_color
    return len(set(int(v) for v in g.ravel() if int(v) != bg))


def _entity_pixel_count(entity: SceneEntity, state: SceneState) -> int:
    return int(entity.attrs.get("size", _entity_non_bg_count(entity, state)))


def _entity_is_singleton(entity: SceneEntity, state: SceneState) -> int:
    return 1 if entity.attrs.get("is_singleton", False) else 0


def _entity_shape_unique(entity: SceneEntity, state: SceneState) -> int:
    return 1 if entity.attrs.get("shape_unique", False) else 0


def _entity_touches_border(entity: SceneEntity, state: SceneState) -> int:
    if entity.bbox is None:
        return 0
    r0, c0, r1, c1 = entity.bbox
    h, w = state.perception.dims
    return 1 if (r0 == 0 or c0 == 0 or r1 >= h - 1 or c1 >= w - 1) else 0


def _entity_is_isolated(entity: SceneEntity, state: SceneState) -> int:
    """1 if no non-bg pixel in entity's bbox is 4-adjacent to non-bg outside bbox."""
    if entity.bbox is None:
        return 0
    r0, c0, r1, c1 = entity.bbox
    grid = state.input_grid
    bg = state.perception.bg_color
    iH, iW = grid.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) == bg:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < iH and 0 <= nc < iW:
                    if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                        if int(grid[nr, nc]) != bg:
                            return 0
    return 1


def _entity_inside_frame(entity: SceneEntity, state: SceneState) -> int:
    """1 if entity's bbox is fully contained within any framed region's outer bbox."""
    if entity.bbox is None:
        return 0
    er0, ec0, er1, ec1 = entity.bbox
    for fr in state.perception.framed_regions:
        fr_r0 = max(0, fr.row - 1)
        fr_c0 = max(0, fr.col - 1)
        fr_r1 = fr.row + fr.height
        fr_c1 = fr.col + fr.width
        if fr_r0 <= er0 and fr_c0 <= ec0 and er1 <= fr_r1 and ec1 <= fr_c1:
            return 1
    return 0


_SELECTOR_PREDICATES = {
    # area selectors
    "largest_bbox_area": (lambda e, s: _entity_bbox_area(e, s), True),
    "smallest_bbox_area": (lambda e, s: _entity_bbox_area(e, s), False),
    # pixel count selectors
    "most_non_bg": (lambda e, s: _entity_non_bg_count(e, s), True),
    "least_non_bg_gt0": (lambda e, s: _entity_non_bg_count(e, s), False),
    "largest_pixel_count": (lambda e, s: _entity_pixel_count(e, s), True),
    "smallest_pixel_count": (lambda e, s: _entity_pixel_count(e, s), False),
    # color count selectors
    "most_colors": (lambda e, s: _entity_color_count(e, s), True),
    "fewest_colors_gt0": (lambda e, s: _entity_color_count(e, s), False),
    # positional selectors
    "top_left": (lambda e, s: (-(e.bbox[0] if e.bbox else 999), -(e.bbox[1] if e.bbox else 999)), True),
    "bottom_right": (lambda e, s: ((e.bbox[2] if e.bbox else -1), (e.bbox[3] if e.bbox else -1)), True),
    # singleton selector (useful for marker objects)
    "is_singleton": (lambda e, s: _entity_is_singleton(e, s), True),
    "not_singleton": (lambda e, s: _entity_is_singleton(e, s), False),
    # shape-based selectors
    "unique_shape": (lambda e, s: _entity_shape_unique(e, s), True),
    "not_unique_shape": (lambda e, s: _entity_shape_unique(e, s), False),
    # compound: unique shape + largest area tiebreak
    "unique_shape_largest": (lambda e, s: (
        _entity_shape_unique(e, s) * 100000 + _entity_bbox_area(e, s)
    ), True),
    # compound: unique shape + most pixels tiebreak
    "unique_shape_most_pixels": (lambda e, s: (
        _entity_shape_unique(e, s) * 100000 + _entity_pixel_count(e, s)
    ), True),
    # border-touching selectors
    "touches_border": (lambda e, s: _entity_touches_border(e, s), True),
    "not_touches_border": (lambda e, s: _entity_touches_border(e, s), False),
    # isolation selectors (no adjacent non-bg outside bbox)
    "isolated": (lambda e, s: _entity_is_isolated(e, s), True),
    "not_isolated": (lambda e, s: _entity_is_isolated(e, s), False),
    # compound: isolated + largest area tiebreak
    "isolated_largest": (lambda e, s: (
        _entity_is_isolated(e, s) * 100000 + _entity_bbox_area(e, s)
    ), True),
    # compound: isolated + most pixels tiebreak
    "isolated_most_pixels": (lambda e, s: (
        _entity_is_isolated(e, s) * 100000 + _entity_pixel_count(e, s)
    ), True),
    # frame containment selectors
    "inside_frame": (lambda e, s: _entity_inside_frame(e, s), True),
    "not_inside_frame": (lambda e, s: _entity_inside_frame(e, s), False),
    # compound: inside_frame + largest
    "inside_frame_largest": (lambda e, s: (
        _entity_inside_frame(e, s) * 100000 + _entity_bbox_area(e, s)
    ), True),
}


@register_step(StepOp.SELECT_ENTITY)
def _select_entity(state: SceneState, step: SceneStep) -> None:
    """Select one entity by kind + predicate, store its id and grid in state.values.

    Params:
        kind: EntityKind value to filter by (e.g. "panel", "object")
        predicate: selector predicate name from _SELECTOR_PREDICATES
        rank: which rank to select (0 = best, 1 = second best, ...)
        require_non_bg: if True, filter to entities with at least 1 non-bg pixel
        color_filter: if set, only consider entities whose colors include this value
        exclude_color: if set, exclude entities whose colors include this value
    """
    kind_str = step.params.get("kind")
    predicate_name = step.params.get("predicate", "largest_bbox_area")
    rank = int(step.params.get("rank", 0))
    require_non_bg = bool(step.params.get("require_non_bg", False))
    color_filter = step.params.get("color_filter")
    exclude_color = step.params.get("exclude_color")
    connectivity_filter = step.params.get("connectivity")

    if not isinstance(kind_str, str):
        return

    try:
        kind = EntityKind(kind_str)
    except ValueError:
        return

    pred_info = _SELECTOR_PREDICATES.get(predicate_name)
    if pred_info is None:
        return
    key_fn, maximize = pred_info

    candidates = state.entities_of_kind(kind)

    # Exclude color_bbox entities unless color_filter is explicitly set
    if color_filter is None:
        candidates = [
            e for e in candidates
            if e.attrs.get("selector_kind") != "color_bbox"
        ]

    # Connectivity filter: only 4-conn or only 8-conn entities
    if connectivity_filter is not None:
        candidates = [
            e for e in candidates
            if e.attrs.get("connectivity") == int(connectivity_filter)
        ]

    if require_non_bg:
        bg = state.perception.bg_color
        candidates = [
            e for e in candidates
            if e.bbox is not None and np.any(state.get_grid_region(e.bbox) != bg)
        ]

    if color_filter is not None:
        cf = int(color_filter)
        candidates = [e for e in candidates if cf in e.colors]

    if exclude_color is not None:
        ec = int(exclude_color)
        candidates = [e for e in candidates if ec not in e.colors]

    if not candidates:
        return

    # Stable sort: add entity id as tiebreaker
    scored = [(key_fn(e, state), e.id, e) for e in candidates]
    scored.sort(key=lambda x: x[0], reverse=maximize)

    # For "least_non_bg_gt0", filter out zero-count
    if predicate_name == "least_non_bg_gt0":
        scored = [(k, eid, e) for k, eid, e in scored if k > 0]
    if predicate_name == "fewest_colors_gt0":
        scored = [(k, eid, e) for k, eid, e in scored if k > 0]

    if rank >= len(scored):
        return

    _, _, selected = scored[rank]
    output_id = step.output_id or "selected"
    state.values[output_id] = selected.id
    # Also store the grid region for convenience
    if selected.bbox is not None:
        state.values[f"{output_id}_grid"] = state.get_grid_region(selected.bbox)


# ---------------------------------------------------------------------------
# BUILD_CORRESPONDENCE — generic correspondence builder
# ---------------------------------------------------------------------------


@register_step(StepOp.BUILD_CORRESPONDENCE)
def _build_correspondence(state: SceneState, step: SceneStep) -> None:
    """Build a correspondence between two sets of entities.

    Params:
        source_kind: EntityKind for source entities
        target_kind: EntityKind for target entities (or "output_cells")
        mode: correspondence mode
            "positional_order": sort both by (row, col), match by index
            "row_col_index": match by (row_idx, col_idx) attrs
            "color_key": match by dominant non-bg color
            "zone_order": match by spatial position order
    """
    source_kind_str = step.params.get("source_kind")
    target_kind_str = step.params.get("target_kind", "output_cells")
    mode = step.params.get("mode", "positional_order")

    if not isinstance(source_kind_str, str):
        return

    try:
        source_kind = EntityKind(source_kind_str)
    except ValueError:
        return

    sources = state.entities_of_kind(source_kind)
    if not sources:
        return

    # Sort sources by position
    sources_sorted = sorted(
        sources,
        key=lambda e: (e.bbox[0] if e.bbox else 0, e.bbox[1] if e.bbox else 0),
    )

    # Build correspondence entries
    entries: list[tuple[str, int, int]] = []  # (entity_id, target_row, target_col)

    if mode == "positional_order":
        for idx, entity in enumerate(sources_sorted):
            entries.append((entity.id, idx, 0))
    elif mode == "row_col_index":
        for entity in sources_sorted:
            ri = entity.attrs.get("row_idx")
            ci = entity.attrs.get("col_idx")
            if isinstance(ri, int) and isinstance(ci, int):
                entries.append((entity.id, ri, ci))
    elif mode == "zone_order":
        for idx, entity in enumerate(sources_sorted):
            entries.append((entity.id, idx, 0))
    else:
        return

    output_id = step.output_id or "correspondence"
    state.values[output_id] = entries


# ---------------------------------------------------------------------------
# MAP_OVER_ENTITIES — iterate entities, compute property, write to output
# ---------------------------------------------------------------------------


def _entity_prop_dominant_non_bg(grid: Grid, bg: int) -> int:
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _entity_prop_has_multiple_non_bg(grid: Grid, bg: int) -> int:
    return 1 if int(np.sum(grid != bg)) >= 2 else 0


def _entity_prop_non_bg_count(grid: Grid, bg: int) -> int:
    return int(np.sum(grid != bg))


def _entity_prop_unique_color_count(grid: Grid, bg: int) -> int:
    return len(set(int(v) for v in grid.ravel() if int(v) != bg))


def _entity_prop_has_non_bg(grid: Grid, bg: int) -> int:
    return 1 if np.any(grid != bg) else 0


def _entity_prop_minority_color(grid: Grid, bg: int) -> int:
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[int(np.argmin(counts))])


_MAP_PROPERTY_EXTRACTORS = {
    "dominant_non_bg_color": _entity_prop_dominant_non_bg,
    "has_multiple_non_bg": _entity_prop_has_multiple_non_bg,
    "non_bg_count": _entity_prop_non_bg_count,
    "unique_color_count": _entity_prop_unique_color_count,
    "has_non_bg": _entity_prop_has_non_bg,
    "minority_color": _entity_prop_minority_color,
}


@register_step(StepOp.MAP_OVER_ENTITIES)
def _map_over_entities(state: SceneState, step: SceneStep) -> None:
    """Iterate entities, compute a property per entity, write results.

    Params:
        kind: EntityKind to iterate
        property: property extractor name
        layout: how to arrange results
            "grid": write to output grid at (row_idx, col_idx) from entity attrs
            "list": store as list in state.values
        correspondence: optional name of a correspondence to use for positioning
    """
    kind_str = step.params.get("kind")
    prop_name = step.params.get("property")
    layout = step.params.get("layout", "grid")
    corr_name = step.params.get("correspondence")

    if not isinstance(kind_str, str) or not isinstance(prop_name, str):
        return

    try:
        kind = EntityKind(kind_str)
    except ValueError:
        return

    extractor = _MAP_PROPERTY_EXTRACTORS.get(prop_name)
    if extractor is None:
        return

    entities = state.entities_of_kind(kind)
    if not entities:
        return

    bg = state.perception.bg_color

    if layout == "grid" and state.output_grid is not None:
        # Use correspondence if provided
        if isinstance(corr_name, str) and corr_name in state.values:
            corr = state.values[corr_name]
            for entity_id, target_r, target_c in corr:
                entity = state.get_entity(entity_id)
                if entity is None or entity.bbox is None:
                    continue
                grid = state.get_grid_region(entity.bbox)
                val = extractor(grid, bg)
                oH, oW = state.output_grid.shape
                if 0 <= target_r < oH and 0 <= target_c < oW:
                    state.output_grid[target_r, target_c] = val
        else:
            # Use row_idx/col_idx attrs from entities
            for entity in entities:
                ri = entity.attrs.get("row_idx")
                ci = entity.attrs.get("col_idx")
                if not isinstance(ri, int) or not isinstance(ci, int):
                    continue
                if entity.bbox is None:
                    continue
                grid = state.get_grid_region(entity.bbox)
                val = extractor(grid, bg)
                oH, oW = state.output_grid.shape
                if 0 <= ri < oH and 0 <= ci < oW:
                    state.output_grid[ri, ci] = val

    elif layout == "list":
        results = []
        entities_sorted = sorted(
            entities,
            key=lambda e: (e.bbox[0] if e.bbox else 0, e.bbox[1] if e.bbox else 0),
        )
        for entity in entities_sorted:
            if entity.bbox is None:
                results.append(bg)
                continue
            grid = state.get_grid_region(entity.bbox)
            results.append(extractor(grid, bg))
        output_id = step.output_id or "map_results"
        state.values[output_id] = results


# ---------------------------------------------------------------------------
# APPLY_PER_CELL — conditional per-cell operation for partition tasks
# ---------------------------------------------------------------------------


def _cell_dominant_non_bg(cell_grid: Grid, bg: int) -> int:
    non_bg = cell_grid[cell_grid != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[np.argmax(counts)])


@register_step(StepOp.APPLY_PER_CELL)
def _apply_per_cell(state: SceneState, step: SceneStep) -> None:
    """Apply a per-cell transformation over partition cells.

    Each cell is independently transformed based on a rule that reads
    the cell's own content. The result is written back in place.

    Params:
        rule: transformation rule name
            "swap_dominant_with_bg": swap cell's dominant non-bg color with bg
            "fill_bg_with_dominant": fill bg pixels with cell's dominant non-bg color
            "clear_non_bg": set all non-bg pixels to bg
            "fill_bg_with_color": fill bg pixels with params["fill_color"]
        filter: optional predicate to select which cells are affected
            "has_non_bg": only cells with at least one non-bg pixel
            "has_multiple_non_bg": only cells with 2+ non-bg pixels
            "all": all cells (default)
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    rule = step.params.get("rule", "swap_dominant_with_bg")
    cell_filter = step.params.get("filter", "all")
    fill_color = step.params.get("fill_color")

    p = state.perception.partition
    if p is None or len(p.cells) < 2:
        return

    bg = state.perception.bg_color

    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        cell_grid = state.output_grid[r0 : r1 + 1, c0 : c1 + 1]
        orig_cell = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]

        # Apply filter
        non_bg_mask = orig_cell != bg
        n_non_bg = int(np.sum(non_bg_mask))

        if cell_filter == "has_non_bg" and n_non_bg == 0:
            continue
        if cell_filter == "has_multiple_non_bg" and n_non_bg < 2:
            continue

        dom = _cell_dominant_non_bg(orig_cell, bg)

        if rule == "swap_dominant_with_bg":
            temp = orig_cell.copy()
            cell_grid[temp == dom] = bg
            cell_grid[temp == bg] = dom
        elif rule == "fill_bg_with_dominant":
            cell_grid[orig_cell == bg] = dom
        elif rule == "clear_non_bg":
            cell_grid[orig_cell != bg] = bg
        elif rule == "fill_bg_with_color" and fill_color is not None:
            cell_grid[orig_cell == bg] = int(fill_color)
        elif rule == "invert_non_bg":
            # Replace each non-bg pixel with bg, and each bg pixel with cell's dominant
            temp = orig_cell.copy()
            cell_grid[temp != bg] = bg
            cell_grid[temp == bg] = dom
        elif rule == "fill_enclosed_with_dominant":
            # Fill bg regions NOT touching the cell border with dominant color
            _fill_enclosed_in_cell(cell_grid, orig_cell, bg, dom)
        elif rule == "fill_enclosed_with_color" and fill_color is not None:
            _fill_enclosed_in_cell(cell_grid, orig_cell, bg, int(fill_color))
        elif rule == "replace_minority_with_dominant":
            # Replace the minority non-bg color with the dominant
            non_bg = orig_cell[orig_cell != bg]
            if len(non_bg) > 0:
                vals, counts = np.unique(non_bg, return_counts=True)
                if len(vals) >= 2:
                    minority = int(vals[np.argmin(counts)])
                    cell_grid[orig_cell == minority] = dom
        elif rule == "recolor":
            # Replace from_color with to_color in the cell
            from_c = step.params.get("from_color")
            to_c = step.params.get("to_color")
            if from_c is not None and to_c is not None:
                cell_grid[orig_cell == int(from_c)] = int(to_c)
        elif rule == "fill_solid_or_clear":
            # Threshold rule: fill cell solid if meets filter, clear to bg otherwise
            cell_grid[:] = dom if n_non_bg > 0 else bg
        elif rule == "fill_solid_if_enough_or_clear":
            # Fill solid with dominant if n_non_bg >= threshold, else clear
            threshold = int(step.params.get("threshold", 2))
            if n_non_bg >= threshold:
                cell_grid[:] = dom
            else:
                cell_grid[:] = bg
        elif rule == "fill_solid_argmax_or_clear":
            pass  # handled below after loop

    # Argmax rule: need to know max across all cells
    if rule == "fill_solid_argmax_or_clear":
        # Find max non-bg count
        max_nbg = 0
        for cell2 in p.cells:
            r2, c2, r3, c3 = cell2.bbox
            g2 = state.input_grid[r2 : r3 + 1, c2 : c3 + 1]
            n2 = int(np.sum(g2 != bg))
            if n2 > max_nbg:
                max_nbg = n2
        # Fill cells at max, clear others
        for cell2 in p.cells:
            r2, c2, r3, c3 = cell2.bbox
            g2 = state.input_grid[r2 : r3 + 1, c2 : c3 + 1]
            n2 = int(np.sum(g2 != bg))
            cg2 = state.output_grid[r2 : r3 + 1, c2 : c3 + 1]
            if n2 == max_nbg and n2 > 0:
                dom2 = _cell_dominant_non_bg(g2, bg)
                cg2[:] = dom2
            else:
                cg2[:] = bg

    state.values["per_cell_result"] = state.output_grid.copy()


# ---------------------------------------------------------------------------
# COMBINE_CELLS — overlay/combine all partition cells into one target cell
# ---------------------------------------------------------------------------


@register_step(StepOp.COMBINE_CELLS)
def _combine_cells(state: SceneState, step: SceneStep) -> None:
    """Combine partition cells via overlay/boolean/copy.

    Two usage modes:
    1. Target-cell mode (same-as-input): write overlay into one target cell
       Params: mode, target, keep_others
    2. Cell-sized output mode (different-size): produce cell-sized result
       Params: operation, output_to
    """
    p = state.perception.partition
    if p is None or len(p.cells) < 2:
        return

    bg = state.perception.bg_color

    # Detect which mode: legacy (operation+output_to) vs new (mode+target)
    operation = step.params.get("operation")
    output_to = step.params.get("output_to")

    # Extract cell grids
    cell_data: list[tuple[Any, Grid]] = []
    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        cell_data.append((cell, state.input_grid[r0 : r1 + 1, c0 : c1 + 1]))

    shapes = [g.shape for _, g in cell_data]
    if len(set(shapes)) != 1:
        return
    shape0 = shapes[0]

    # Compute overlay using whichever param is present
    mode = step.params.get("mode") or operation or "or"

    if mode in ("or", "overlay", "stack"):
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        for _, g in cell_data:
            result[g != bg] = g[g != bg]
    elif mode == "and":
        all_nb = np.ones(shape0, dtype=bool)
        for _, g in cell_data:
            all_nb &= (g != bg)
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        if np.any(all_nb):
            result[all_nb] = cell_data[0][1][all_nb]
    elif mode == "xor":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        count = np.zeros(shape0, dtype=int)
        for _, g in cell_data:
            count += (g != bg).astype(int)
        xor_mask = count == 1
        for _, g in cell_data:
            mask = (g != bg) & xor_mask
            result[mask] = g[mask]
    elif mode == "or_any_color":
        result = np.full(shape0, bg, dtype=state.input_grid.dtype)
        for r in range(shape0[0]):
            for c in range(shape0[1]):
                colors = [int(g[r, c]) for _, g in cell_data if int(g[r, c]) != bg]
                if colors:
                    from collections import Counter as _C
                    result[r, c] = _C(colors).most_common(1)[0][0]
    else:
        return

    # Legacy mode: cell-sized output
    if operation is not None or output_to is not None:
        output_id = step.output_id or "combined"
        state.values[output_id] = result
        if (output_to or "output") == "output":
            state.output_grid = result
        return

    # New mode: write overlay into a target cell, optionally clear others
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    target_sel = step.params.get("target", "most_non_bg")
    keep_others = step.params.get("keep_others", True)

    target_cell = _select_target_cell(cell_data, bg, target_sel)
    if target_cell is None:
        return

    # Clear non-target cells if requested
    if not keep_others:
        for cell_obj, _ in cell_data:
            if cell_obj is not target_cell:
                cr0, cc0, cr1, cc1 = cell_obj.bbox
                state.output_grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = bg

    r0, c0, r1, c1 = target_cell.bbox
    rh, rw = result.shape
    state.output_grid[r0 : r0 + rh, c0 : c0 + rw] = result

    state.values["combined"] = state.output_grid.copy()


def _select_target_cell(cell_data, bg, selector):
    """Select a target cell by predicate."""
    if selector == "most_non_bg":
        return max(cell_data, key=lambda x: int(np.sum(x[1] != bg)))[0]
    elif selector == "least_non_bg":
        candidates = [(c, g) for c, g in cell_data if np.any(g != bg)]
        if candidates:
            return min(candidates, key=lambda x: int(np.sum(x[1] != bg)))[0]
    elif selector == "empty":
        empties = [c for c, g in cell_data if not np.any(g != bg)]
        if len(empties) == 1:
            return empties[0]
    elif selector == "first":
        return cell_data[0][0]
    elif selector == "last":
        return cell_data[-1][0]
    return None


# ---------------------------------------------------------------------------
# BROADCAST_TO_CELLS — copy source cell content into target cells
# ---------------------------------------------------------------------------


@register_step(StepOp.BROADCAST_TO_CELLS)
def _broadcast_to_cells(state: SceneState, step: SceneStep) -> None:
    """Select a source cell by predicate and copy into target cells.

    Params:
        source: predicate to select source cell
            "unique_color": cell with a unique dominant non-bg color
            "most_non_bg": cell with most non-bg pixels
            "most_colors": cell with most unique non-bg colors
        target: which cells receive the source content
            "empty": all-bg cells
            "different_dominant": cells whose dominant color differs from source
            "all_others": all non-source cells
        overlay_mode: "replace" (default) or "overlay_non_bg"
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    p = state.perception.partition
    if p is None or len(p.cells) < 2:
        return

    bg = state.perception.bg_color
    source_sel = step.params.get("source", "most_non_bg")
    target_sel = step.params.get("target", "empty")
    overlay_mode = step.params.get("overlay_mode", "replace")

    cell_data = []
    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        g = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]
        non_bg = g[g != bg]
        n_nb = len(non_bg)
        dom = int(np.bincount(non_bg.astype(int)).argmax()) if n_nb > 0 else bg
        n_colors = len(set(int(v) for v in non_bg)) if n_nb > 0 else 0
        cell_data.append((cell, g, n_nb, dom, n_colors))

    # Select source
    source = _select_source_cell(cell_data, bg, source_sel)

    if source is None:
        return

    source_cell, source_grid = source
    source_dom = int(np.bincount(source_grid[source_grid != bg].astype(int)).argmax()) if np.any(source_grid != bg) else bg
    source_ri = source_cell.row_idx
    source_ci = source_cell.col_idx

    # Select targets
    targets = _select_target_cells(cell_data, bg, target_sel, source_cell, source_dom, source_ri, source_ci)

    # Write source into targets
    for tc in targets:
        r0, c0, r1, c1 = tc.bbox
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        sh, sw = source_grid.shape
        if th != sh or tw != sw:
            continue
        if overlay_mode == "replace":
            state.output_grid[r0 : r1 + 1, c0 : c1 + 1] = source_grid
        elif overlay_mode == "overlay_non_bg":
            mask = source_grid != bg
            state.output_grid[r0 : r1 + 1, c0 : c1 + 1][mask] = source_grid[mask]

    state.values["broadcast_result"] = state.output_grid.copy()


def _select_source_cell(cell_data, bg, selector):
    """Select source cell by predicate. Returns (cell, grid) or None."""
    non_empty = [(c, g, n, d, nc) for c, g, n, d, nc in cell_data if n > 0]

    if selector == "most_non_bg":
        if non_empty:
            best = max(non_empty, key=lambda x: x[2])
            return (best[0], best[1])
    elif selector == "most_colors":
        if non_empty:
            best = max(non_empty, key=lambda x: x[4])
            return (best[0], best[1])
    elif selector == "unique_color":
        dom_counts: dict[int, list] = {}
        for c, g, n, d, nc in cell_data:
            dom_counts.setdefault(d, []).append((c, g))
        for d, entries in dom_counts.items():
            if d != bg and len(entries) == 1:
                return entries[0]
    elif selector == "unique_non_bg_count":
        count_groups: dict[int, list] = {}
        for c, g, n, d, nc in cell_data:
            count_groups.setdefault(n, []).append((c, g))
        for cnt, entries in sorted(count_groups.items(), reverse=True):
            if cnt > 0 and len(entries) == 1:
                return entries[0]
    elif selector == "unique_palette":
        pal_groups: dict[tuple, list] = {}
        for c, g, n, d, nc in cell_data:
            pal = tuple(sorted(set(int(v) for v in g.ravel() if int(v) != bg)))
            pal_groups.setdefault(pal, []).append((c, g))
        for pal, entries in pal_groups.items():
            if pal and len(entries) == 1:
                return entries[0]
    elif selector == "differs_from_majority":
        # Find majority dominant color, select the cell that differs
        dom_list = [d for _, _, n, d, _ in cell_data if n > 0]
        if dom_list:
            from collections import Counter as _C
            majority = _C(dom_list).most_common(1)[0][0]
            for c, g, n, d, nc in cell_data:
                if n > 0 and d != majority:
                    return (c, g)
    elif selector == "first_non_empty":
        if non_empty:
            return (non_empty[0][0], non_empty[0][1])
    elif selector == "last_non_empty":
        if non_empty:
            return (non_empty[-1][0], non_empty[-1][1])
    return None


def _select_target_cells(cell_data, bg, selector, source_cell, source_dom, source_ri, source_ci):
    """Select target cells by predicate. Returns list of cells."""
    targets = []
    for c, g, n, d, nc in cell_data:
        if c is source_cell:
            continue
        match = False
        if selector == "empty":
            match = (n == 0)
        elif selector == "all_others":
            match = True
        elif selector == "different_dominant":
            match = (d != source_dom)
        elif selector == "same_dominant":
            match = (d == source_dom and n > 0)
        elif selector == "non_empty":
            match = (n > 0)
        elif selector == "same_row":
            match = (c.row_idx == source_ri)
        elif selector == "same_col":
            match = (c.col_idx == source_ci)
        elif selector == "adjacent_4":
            dr = abs(c.row_idx - source_ri)
            dc = abs(c.col_idx - source_ci)
            match = ((dr == 1 and dc == 0) or (dr == 0 and dc == 1))
        elif selector == "adjacent_8":
            dr = abs(c.row_idx - source_ri)
            dc = abs(c.col_idx - source_ci)
            match = (0 < max(dr, dc) <= 1)
        elif selector == "diagonal":
            dr = abs(c.row_idx - source_ri)
            dc = abs(c.col_idx - source_ci)
            match = (dr == dc and dr > 0)
        if match:
            targets.append(c)
    return targets


# ---------------------------------------------------------------------------
# FOR_EACH_ENTITY — generic for-each with sub-operation
# ---------------------------------------------------------------------------


@register_step(StepOp.FOR_EACH_ENTITY)
def _for_each_entity(state: SceneState, step: SceneStep) -> None:
    """Iterate entities and apply a sub-operation to each.

    Params:
        kind: EntityKind to iterate
        filter: predicate name (optional)
            "has_non_bg", "has_multiple_non_bg", "empty"
        operation: what to do per entity
            "fill_bg_with_dominant": fill bg pixels with entity's dominant color
            "recolor_to_source_dominant": recolor to match source cell's dominant
        source: name of a value in state.values to use as source data
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    kind_str = step.params.get("kind", "panel")
    filt = step.params.get("filter", "all")
    operation = step.params.get("operation", "fill_bg_with_dominant")

    try:
        kind = EntityKind(kind_str)
    except ValueError:
        return

    entities = state.entities_of_kind(kind)
    bg = state.perception.bg_color

    for entity in entities:
        if entity.bbox is None:
            continue
        r0, c0, r1, c1 = entity.bbox
        orig = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]
        non_bg = orig[orig != bg]
        n_nb = len(non_bg)

        # Apply filter
        if filt == "has_non_bg" and n_nb == 0:
            continue
        if filt == "has_multiple_non_bg" and n_nb < 2:
            continue
        if filt == "empty" and n_nb > 0:
            continue

        dom = int(np.bincount(non_bg.astype(int)).argmax()) if n_nb > 0 else bg

        cell_grid = state.output_grid[r0 : r1 + 1, c0 : c1 + 1]

        if operation == "fill_bg_with_dominant":
            cell_grid[orig == bg] = dom
        elif operation == "clear_non_bg":
            cell_grid[orig != bg] = bg

    state.values["for_each_result"] = state.output_grid.copy()


@register_step(StepOp.FOR_EACH_ENTITY)
def _for_each_entity(state: SceneState, step: SceneStep) -> None:
    """Apply a rule to each entity of a given kind.

    Params:
        kind: EntityKind to iterate
        rule: operation to apply per entity
            "fill_bbox_holes": fill bg pixels within entity bbox with fill_color
            "recolor": replace from_color with to_color within entity bbox
        fill_color: color for fill operations
        from_color / to_color: for recolor
        filter: optional predicate
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    kind_str = step.params.get("kind", "object")
    rule = step.params.get("rule", "fill_bbox_holes")
    fill_color = step.params.get("fill_color")
    from_color = step.params.get("from_color")
    to_color = step.params.get("to_color")
    connectivity = step.params.get("connectivity")

    try:
        kind = EntityKind(kind_str)
    except ValueError:
        return

    bg = state.perception.bg_color
    entities = state.entities_of_kind(kind)

    # Optional connectivity filter
    if connectivity is not None:
        entities = [e for e in entities if e.attrs.get("connectivity") == connectivity]

    # Filter out singletons by default for object operations
    # Exception: predicate_dispatch handles its own filtering
    if kind == EntityKind.OBJECT and rule != "predicate_dispatch":
        entities = [e for e in entities if not e.attrs.get("is_singleton", False)]

    guard = step.params.get("guard")

    for entity in entities:
        if entity.bbox is None:
            continue

        # Guard predicate: skip non-matching entities
        if guard is not None:
            if not _evaluate_guard(guard, entity, state):
                continue

        r0, c0, r1, c1 = entity.bbox
        region = state.output_grid[r0 : r1 + 1, c0 : c1 + 1]
        orig = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]

        if rule == "fill_bbox_holes":
            if fill_color is not None:
                region[orig == bg] = int(fill_color)
        elif rule == "fill_enclosed_bbox":
            if fill_color is not None:
                _fill_enclosed_in_cell(region, orig, bg, int(fill_color))
        elif rule == "recolor":
            if from_color is not None and to_color is not None:
                region[orig == int(from_color)] = int(to_color)
        elif rule == "recolor_by_role":
            # Recolor object pixels based on a role (e.g., match to
            # another object's color by positional correspondence)
            role_map = step.params.get("role_map")
            if isinstance(role_map, dict):
                for fc, tc in role_map.items():
                    region[orig == int(fc)] = int(tc)
        elif rule == "conditional_recolor":
            # Recolor only objects matching a predicate
            predicate = step.params.get("predicate", "all")
            color = entity.attrs.get("dominant_color")
            non_bg = orig[orig != bg]
            if len(non_bg) == 0:
                continue
            dom = int(np.bincount(non_bg.astype(int), minlength=10).argmax())
            match = (
                predicate == "all"
                or (predicate == "is_singleton" and entity.attrs.get("is_singleton"))
                or (predicate == "has_color" and dom == step.params.get("match_color"))
                or (predicate == "size_above" and len(non_bg) > step.params.get("size_threshold", 0))
            )
            if match and from_color is not None and to_color is not None:
                region[orig == int(from_color)] = int(to_color)
        elif rule == "fill_adjacent_bg":
            # Fill bg pixels adjacent to object pixels with fill_color
            if fill_color is None:
                continue
            non_bg_mask = orig != bg
            for dr in range(orig.shape[0]):
                for dc in range(orig.shape[1]):
                    if orig[dr, dc] == bg:
                        for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = dr+ddr, dc+ddc
                            if 0 <= nr < orig.shape[0] and 0 <= nc < orig.shape[1]:
                                if non_bg_mask[nr, nc]:
                                    region[dr, dc] = int(fill_color)
                                    break
        elif rule == "grow_along_axis":
            # Grow/extend object pixels along a row or column
            axis = step.params.get("axis", "row")
            non_bg = orig != bg
            if not np.any(non_bg):
                continue
            if axis == "row":
                for dr in range(orig.shape[0]):
                    if np.any(non_bg[dr]):
                        colors_in_row = orig[dr][non_bg[dr]]
                        fill_c = int(colors_in_row[0])
                        region[dr, :] = fill_c
            elif axis == "col":
                for dc in range(orig.shape[1]):
                    if np.any(non_bg[:, dc]):
                        colors_in_col = orig[:, dc][non_bg[:, dc]]
                        fill_c = int(colors_in_col[0])
                        region[:, dc] = fill_c

        elif rule == "predicate_dispatch":
            # Evaluate a symbolic predicate on this object and apply
            # the action only if the predicate holds.
            from aria.predicate_dispatch import deserialize_predicate
            from aria.predicates import ObjectContext, evaluate_predicate
            from aria.decomposition import extract_objects as _extract_objs

            pred_dict = step.params.get("predicate")
            action_name = step.params.get("action", "recolor")
            if not isinstance(pred_dict, dict):
                continue

            pred = deserialize_predicate(pred_dict)
            all_objs = _extract_objs(state.input_grid, bg)
            # Find raw object matching this entity's bbox
            raw_obj = None
            for o in all_objs:
                if (o.row == r0 and o.col == c0
                        and o.bbox_h == r1 - r0 + 1
                        and o.bbox_w == c1 - c0 + 1):
                    raw_obj = o
                    break
            if raw_obj is None:
                continue

            ctx = ObjectContext(
                obj=raw_obj, grid=state.input_grid, bg=bg,
                all_objects=all_objs, grid_shape=state.input_grid.shape,
                framed_regions=state.perception.framed_regions,
            )

            if not evaluate_predicate(pred, ctx):
                continue  # predicate doesn't match

            if action_name == "recolor":
                fc = step.params.get("from_color")
                tc = step.params.get("to_color")
                if fc is not None and tc is not None:
                    region[orig == int(fc)] = int(tc)
            elif action_name == "recolor_dominant_to":
                # Recolor the object's dominant color to a fixed target
                tc = step.params.get("to_color")
                if tc is not None:
                    non_bg_vals = orig[orig != bg]
                    if len(non_bg_vals) > 0:
                        dom = int(np.bincount(non_bg_vals.astype(int), minlength=10).argmax())
                        region[orig == dom] = int(tc)
            elif action_name == "recolor_to_adjacent":
                # Recolor object pixels to the most common adjacent non-bg non-self color
                adj_counts: dict[int, int] = {}
                for dr2 in range(raw_obj.bbox_h):
                    for dc2 in range(raw_obj.bbox_w):
                        if not raw_obj.mask[dr2, dc2]:
                            continue
                        ar, ac = raw_obj.row + dr2, raw_obj.col + dc2
                        for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr2, nc2 = ar+ddr, ac+ddc
                            if 0 <= nr2 < state.input_grid.shape[0] and 0 <= nc2 < state.input_grid.shape[1]:
                                v = int(state.input_grid[nr2, nc2])
                                if v != bg and v != raw_obj.color:
                                    adj_counts[v] = adj_counts.get(v, 0) + 1
                if adj_counts:
                    adj_color = max(adj_counts, key=adj_counts.get)
                    region[orig == raw_obj.color] = adj_color
            elif action_name == "fill_adjacent":
                fc = step.params.get("fill_color")
                if fc is not None:
                    nb_mask = orig != bg
                    for dr2 in range(orig.shape[0]):
                        for dc2 in range(orig.shape[1]):
                            if orig[dr2, dc2] == bg:
                                for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                    nr2, nc2 = dr2+ddr, dc2+ddc
                                    if 0 <= nr2 < orig.shape[0] and 0 <= nc2 < orig.shape[1]:
                                        if nb_mask[nr2, nc2]:
                                            region[dr2, dc2] = int(fc)
                                            break
            elif action_name == "recolor_to_nearest_neighbor":
                # Find nearest object of a different color, recolor to its color
                my_color = raw_obj.color
                best_dist = float('inf')
                best_color = None
                for other in all_objs:
                    if other is raw_obj or other.color == my_color or other.color == bg:
                        continue
                    d = abs(raw_obj.center_row - other.center_row) + abs(raw_obj.center_col - other.center_col)
                    if d < best_dist:
                        best_dist = d
                        best_color = other.color
                if best_color is not None:
                    region[orig == my_color] = best_color
            elif action_name == "erase_to_bg":
                # Erase all non-bg pixels in this object to background
                for dr2 in range(orig.shape[0]):
                    for dc2 in range(orig.shape[1]):
                        r_abs, c_abs = r0 + dr2, c0 + dc2
                        if (0 <= r_abs < state.input_grid.shape[0]
                                and 0 <= c_abs < state.input_grid.shape[1]):
                            if orig[dr2, dc2] != bg:
                                region[dr2, dc2] = bg
            elif action_name == "relocate":
                # Erase original + stamp at derived position
                relocate_mode = step.params.get("relocate_mode", "nearest_anchor")
                obj_color = raw_obj.color
                grid = state.output_grid
                grows, gcols = grid.shape

                target_r, target_c = None, None
                if relocate_mode == "nearest_anchor":
                    best_d = float('inf')
                    for oth in all_objs:
                        if oth is raw_obj or oth.color == obj_color or oth.color == bg:
                            continue
                        if oth.size <= raw_obj.size:
                            continue  # anchor should be larger
                        dd = abs(raw_obj.center_row - oth.center_row) + abs(raw_obj.center_col - oth.center_col)
                        if dd < best_d:
                            best_d = dd
                            # Place inside anchor's bbox (at its center)
                            target_r = oth.center_row
                            target_c = oth.center_col
                elif relocate_mode == "mirror":
                    cr, cc = grows // 2, gcols // 2
                    target_r = 2 * cr - raw_obj.row - raw_obj.bbox_h
                    target_c = 2 * cc - raw_obj.col - raw_obj.bbox_w
                elif relocate_mode == "host_gap":
                    # Find bg gaps inside the nearest larger host object
                    # and stamp the singleton there
                    best_host_d = float('inf')
                    best_host = None
                    for oth in all_objs:
                        if oth is raw_obj or oth.color == obj_color or oth.color == bg:
                            continue
                        if oth.size <= raw_obj.size * 3:
                            continue  # host must be substantially larger
                        dd = abs(raw_obj.center_row - oth.center_row) + abs(raw_obj.center_col - oth.center_col)
                        if dd < best_host_d:
                            best_host_d = dd
                            best_host = oth
                    if best_host is not None:
                        # Find bg cells inside host bbox
                        hr0, hc0 = best_host.row, best_host.col
                        hr1, hc1 = hr0 + best_host.bbox_h, hc0 + best_host.bbox_w
                        bg_gaps = []
                        for hr in range(hr0, min(hr1, grows)):
                            for hc in range(hc0, min(hc1, gcols)):
                                if int(state.input_grid[hr, hc]) == bg:
                                    # Check if surrounded by host color (true interior gap)
                                    adj_host = 0
                                    for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                        nr2, nc2 = hr+ddr, hc+ddc
                                        if 0<=nr2<grows and 0<=nc2<gcols:
                                            if int(state.input_grid[nr2, nc2]) == best_host.color:
                                                adj_host += 1
                                    if adj_host >= 2:  # at least 2 adjacent host-color pixels
                                        bg_gaps.append((hr, hc))
                        if bg_gaps:
                            target_r, target_c = bg_gaps[0]
                elif relocate_mode == "fixed_position":
                    target_r = step.params.get("target_row")
                    target_c = step.params.get("target_col")

                if target_r is not None and target_c is not None:
                    target_r = max(0, min(int(target_r), grows - 1))
                    target_c = max(0, min(int(target_c), gcols - 1))
                    # Erase original
                    for dr2 in range(orig.shape[0]):
                        for dc2 in range(orig.shape[1]):
                            if raw_obj.mask[dr2, dc2]:
                                region[dr2, dc2] = bg
                    # Stamp at target (single pixel for singletons, shape for larger)
                    for dr2 in range(raw_obj.bbox_h):
                        for dc2 in range(raw_obj.bbox_w):
                            if raw_obj.mask[dr2, dc2]:
                                tr = target_r + dr2
                                tc = target_c + dc2
                                if 0 <= tr < grows and 0 <= tc < gcols:
                                    grid[tr, tc] = obj_color
            elif action_name == "erase_at_reference_rows":
                # Erase pixels at rows where other non-self objects exist
                ref_rows = set()
                for oth in all_objs:
                    if oth is raw_obj or oth.color == raw_obj.color or oth.color == bg:
                        continue
                    for odr in range(oth.bbox_h):
                        for odc in range(oth.bbox_w):
                            if oth.mask[odr, odc]:
                                ref_rows.add(oth.row + odr)
                for dr2 in range(orig.shape[0]):
                    abs_row = r0 + dr2
                    if abs_row in ref_rows:
                        for dc2 in range(orig.shape[1]):
                            if raw_obj.mask[dr2, dc2] if dr2 < raw_obj.bbox_h and dc2 < raw_obj.bbox_w else False:
                                region[dr2, dc2] = bg
            elif action_name == "erase":
                ec = step.params.get("erase_color")
                tc = step.params.get("to_color", bg)
                if ec is not None:
                    region[orig == int(ec)] = int(tc)

        # ----- Role-based rules -----

        elif rule == "fill_bbox_holes_role":
            # Fill bg within entity bbox using a color role instead of literal
            color_role = step.params.get("color_role", "singleton_color")
            resolved = resolve_color_role(color_role, state, entity)
            if resolved is not None:
                region[orig == bg] = resolved

        elif rule == "fill_enclosed_role":
            # Fill enclosed bg regions using a color role
            color_role = step.params.get("color_role", "singleton_color")
            resolved = resolve_color_role(color_role, state, entity)
            if resolved is not None:
                _fill_enclosed_in_cell(region, orig, bg, resolved)

        elif rule == "recolor_dominant_to_minority":
            # Swap dominant and minority non-bg colors within entity bbox
            non_bg = orig[orig != bg]
            if len(non_bg) < 2:
                continue
            counts = np.bincount(non_bg.astype(int), minlength=10)
            non_bg_colors = [c for c in range(10) if c != bg and counts[c] > 0]
            if len(non_bg_colors) < 2:
                continue
            non_bg_colors.sort(key=lambda c: -counts[c])
            dom_c = non_bg_colors[0]
            min_c = non_bg_colors[-1]
            # Swap: dominant ↔ minority
            temp = orig.copy()
            region[temp == dom_c] = min_c
            region[temp == min_c] = dom_c

        elif rule == "recolor_to_role":
            # Recolor all non-bg pixels in entity bbox to a resolved color role
            color_role = step.params.get("color_role", "singleton_color")
            resolved = resolve_color_role(color_role, state, entity)
            if resolved is not None:
                region[orig != bg] = resolved

        elif rule == "fill_bg_role":
            # Fill bg pixels with a resolved color role (global or entity-local)
            color_role = step.params.get("color_role", "dominant_object_color")
            resolved = resolve_color_role(color_role, state, entity)
            if resolved is not None:
                region[orig == bg] = resolved


def _fill_enclosed_in_cell(cell_grid, orig_cell, bg, fill_color):
    """Fill bg pixels enclosed within the cell (not touching cell border)."""
    from scipy import ndimage
    bg_mask = orig_cell == bg
    labeled, n = ndimage.label(bg_mask)
    if n == 0:
        return
    h, w = orig_cell.shape
    border_labels = set()
    for r in range(h):
        if labeled[r, 0] > 0: border_labels.add(labeled[r, 0])
        if labeled[r, w-1] > 0: border_labels.add(labeled[r, w-1])
    for c in range(w):
        if labeled[0, c] > 0: border_labels.add(labeled[0, c])
        if labeled[h-1, c] > 0: border_labels.add(labeled[h-1, c])
    for label in range(1, n + 1):
        if label not in border_labels:
            cell_grid[labeled == label] = fill_color


# ---------------------------------------------------------------------------
# Color role resolution
# ---------------------------------------------------------------------------

# Color roles: symbolic names resolved per-demo from perception/entity context.
# Used by conditional FOR_EACH_ENTITY rules instead of literal color IDs.

COLOR_ROLES = (
    "singleton_color",        # color of singleton (1-pixel) objects
    "rarest_non_bg",          # least frequent non-bg color in the grid
    "most_frequent_non_bg",   # most frequent non-bg color
    "dominant_object_color",  # most common non-bg color within an entity bbox
    "minority_object_color",  # least common non-bg color within an entity bbox
    "boundary_color",         # color of enclosing frame if present
    "interior_bg",            # bg color within an enclosed region
)


def resolve_color_role(
    role: str,
    state: SceneState,
    entity: SceneEntity | None = None,
) -> int | None:
    """Resolve a symbolic color role to a concrete color value.

    Returns None if the role cannot be resolved in the current context.
    """
    bg = state.perception.bg_color

    if role == "singleton_color":
        for obj in state.perception.objects.objects:
            if obj.is_singleton and obj.color != bg:
                return int(obj.color)
        return None

    if role == "rarest_non_bg":
        color_counts = state.perception.color_pixel_counts
        best_color, best_count = None, float("inf")
        for c, cnt in color_counts.items():
            if c == bg:
                continue
            if cnt < best_count:
                best_count = cnt
                best_color = c
        return int(best_color) if best_color is not None else None

    if role == "most_frequent_non_bg":
        color_counts = state.perception.color_pixel_counts
        best_color, best_count = None, 0
        for c, cnt in color_counts.items():
            if c == bg:
                continue
            if cnt > best_count:
                best_count = cnt
                best_color = c
        return int(best_color) if best_color is not None else None

    if role == "boundary_color":
        if state.perception.framed_regions:
            return int(state.perception.framed_regions[0].frame_color)
        return None

    # Entity-local roles
    if entity is not None and entity.bbox is not None:
        r0, c0, r1, c1 = entity.bbox
        region = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]
        non_bg = region[region != bg]

        if role == "dominant_object_color":
            if len(non_bg) == 0:
                return None
            return int(np.bincount(non_bg.astype(int), minlength=10).argmax())

        if role == "minority_object_color":
            if len(non_bg) == 0:
                return None
            counts = np.bincount(non_bg.astype(int), minlength=10)
            # Find the least frequent non-zero non-bg color
            best_c, best_cnt = None, float("inf")
            for c in range(10):
                if c == bg or counts[c] == 0:
                    continue
                if counts[c] < best_cnt:
                    best_cnt = counts[c]
                    best_c = c
            return best_c

        if role == "interior_bg":
            return bg

    return None


# The COMBINE_CELLS handler above (line ~1198) handles both:
# - "mode" param: overlay into a target cell (same-as-input tasks)
# - "operation" param: combine into cell-sized output (different-size tasks)
# Legacy callers using "operation"+"output_to" are handled via fallback in
# the handler above.


# ---------------------------------------------------------------------------
# BROADCAST_TO_CELLS — stamp a value into every partition cell position
# ---------------------------------------------------------------------------


@register_step(StepOp.BROADCAST_TO_CELLS)
def _broadcast_to_cells(state: SceneState, step: SceneStep) -> None:
    """Stamp a cell-sized grid into every partition cell position.

    Params:
        source: value name or "combined" for the combined result
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    source = step.params.get("source", "combined")
    template = state.values.get(source)
    if template is None or not isinstance(template, np.ndarray):
        return

    p = state.perception.partition
    if p is None:
        return

    for cell in p.cells:
        r0, c0, r1, c1 = cell.bbox
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        th, tw = template.shape
        if (ch, cw) == (th, tw):
            state.output_grid[r0 : r1 + 1, c0 : c1 + 1] = template


# ---------------------------------------------------------------------------
# ASSIGN_CELLS — permute/rearrange partition cells by structural rule
# ---------------------------------------------------------------------------


def _cell_non_bg_count(grid, bg):
    return int(np.sum(grid != bg))


def _cell_dominant_color(grid, bg):
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return bg
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[np.argmax(counts)])


def _cell_n_unique_colors(grid, bg):
    return len(set(int(v) for v in grid.ravel() if int(v) != bg))


@register_step(StepOp.ASSIGN_CELLS)
def _assign_cells(state: SceneState, step: SceneStep) -> None:
    """Permute partition cells by a structural assignment rule.

    Reads input cells, computes an assignment (source→target mapping),
    and writes source cell content into target positions in the output.

    Params:
        mode: assignment rule
            "sort_by_non_bg_asc": place cells sorted by non-bg count ascending
            "sort_by_non_bg_desc": place cells sorted by non-bg count descending
            "sort_by_color_count_asc": sort by distinct non-bg colors ascending
            "sort_by_color_count_desc": sort by distinct non-bg colors descending
            "sort_by_dominant_color_asc": sort by dominant non-bg color value
            "sort_by_dominant_color_desc": sort by dominant color descending
            "rotate_cell_grid_90": rot90 the cell grid arrangement
            "rotate_cell_grid_180": rot180 the cell grid arrangement
            "rotate_cell_grid_270": rot270 the cell grid arrangement
            "flip_cell_grid_lr": flip cell grid left-right
            "flip_cell_grid_ud": flip cell grid up-down
            "transpose_cell_grid": transpose cell grid
            "broadcast_most_non_bg": stamp the cell with most non-bg into all cells
            "broadcast_unique_color": stamp the cell with unique dominant color into all
    """
    if state.output_grid is None:
        state.output_grid = state.input_grid.copy()

    p = state.perception.partition
    if p is None or not p.is_uniform_partition or len(p.cells) < 2:
        return

    bg = state.perception.bg_color
    mode = step.params.get("mode", "sort_by_non_bg_asc")
    nr, nc = p.n_rows, p.n_cols

    # Build cell grid indexed by (row, col)
    cell_map: dict[tuple[int, int], object] = {}
    cell_grids: dict[tuple[int, int], np.ndarray] = {}
    for cell in p.cells:
        pos = (cell.row_idx, cell.col_idx)
        cell_map[pos] = cell
        r0, c0, r1, c1 = cell.bbox
        cell_grids[pos] = state.input_grid[r0 : r1 + 1, c0 : c1 + 1]

    # Compute assignment: target_pos -> source_pos
    assignment: dict[tuple[int, int], tuple[int, int]] = {}

    # Positional order: row-major list of (r, c)
    positions = [(r, c) for r in range(nr) for c in range(nc)]

    if mode.startswith("sort_by_"):
        # Sort cells by property, place in positional order
        prop = mode[len("sort_by_"):]
        if prop == "non_bg_asc":
            key_fn = lambda pos: _cell_non_bg_count(cell_grids[pos], bg)
            reverse = False
        elif prop == "non_bg_desc":
            key_fn = lambda pos: _cell_non_bg_count(cell_grids[pos], bg)
            reverse = True
        elif prop == "color_count_asc":
            key_fn = lambda pos: _cell_n_unique_colors(cell_grids[pos], bg)
            reverse = False
        elif prop == "color_count_desc":
            key_fn = lambda pos: _cell_n_unique_colors(cell_grids[pos], bg)
            reverse = True
        elif prop == "dominant_color_asc":
            key_fn = lambda pos: _cell_dominant_color(cell_grids[pos], bg)
            reverse = False
        elif prop == "dominant_color_desc":
            key_fn = lambda pos: _cell_dominant_color(cell_grids[pos], bg)
            reverse = True
        else:
            return
        sorted_sources = sorted(positions, key=key_fn, reverse=reverse)
        for target_pos, source_pos in zip(positions, sorted_sources):
            assignment[target_pos] = source_pos

    elif mode == "rotate_cell_grid_90":
        for r in range(nr):
            for c in range(nc):
                sr, sc = nc - 1 - c, r
                if (sr, sc) in cell_grids:
                    assignment[(r, c)] = (sr, sc)
    elif mode == "rotate_cell_grid_180":
        for r in range(nr):
            for c in range(nc):
                assignment[(r, c)] = (nr - 1 - r, nc - 1 - c)
    elif mode == "rotate_cell_grid_270":
        for r in range(nr):
            for c in range(nc):
                sr, sc = c, nr - 1 - r
                if (sr, sc) in cell_grids:
                    assignment[(r, c)] = (sr, sc)
    elif mode == "flip_cell_grid_lr":
        for r in range(nr):
            for c in range(nc):
                assignment[(r, c)] = (r, nc - 1 - c)
    elif mode == "flip_cell_grid_ud":
        for r in range(nr):
            for c in range(nc):
                assignment[(r, c)] = (nr - 1 - r, c)
    elif mode == "transpose_cell_grid":
        for r in range(nr):
            for c in range(nc):
                if (c, r) in cell_grids:
                    assignment[(r, c)] = (c, r)
    elif mode.startswith("broadcast_"):
        # Select one source cell, copy to all targets
        sel = mode[len("broadcast_"):]
        if sel == "most_non_bg":
            source_pos = max(positions, key=lambda p: _cell_non_bg_count(cell_grids[p], bg))
        elif sel == "unique_color":
            doms = {pos: _cell_dominant_color(cell_grids[pos], bg) for pos in positions}
            dom_counts: dict[int, list] = {}
            for pos, d in doms.items():
                dom_counts.setdefault(d, []).append(pos)
            source_pos = None
            for d, ps in dom_counts.items():
                if d != bg and len(ps) == 1:
                    source_pos = ps[0]; break
            if source_pos is None:
                return
        else:
            return
        for pos in positions:
            assignment[pos] = source_pos
    elif mode.startswith("row_broadcast_"):
        # Per row: select source cell by property, broadcast to all cols in that row
        prop = mode[len("row_broadcast_"):]
        for r in range(nr):
            row_positions = [(r, c) for c in range(nc) if (r, c) in cell_grids]
            if not row_positions:
                continue
            if prop == "most_non_bg":
                src = max(row_positions, key=lambda p: _cell_non_bg_count(cell_grids[p], bg))
            elif prop == "least_non_bg_gt0":
                gt0 = [p for p in row_positions if _cell_non_bg_count(cell_grids[p], bg) > 0]
                src = min(gt0, key=lambda p: _cell_non_bg_count(cell_grids[p], bg)) if gt0 else row_positions[0]
            elif prop == "most_colors":
                src = max(row_positions, key=lambda p: _cell_n_unique_colors(cell_grids[p], bg))
            else:
                return
            for c in range(nc):
                if (r, c) in cell_grids:
                    assignment[(r, c)] = src
    elif mode.startswith("col_broadcast_"):
        # Per column: select source cell by property, broadcast to all rows in that col
        prop = mode[len("col_broadcast_"):]
        for c in range(nc):
            col_positions = [(r, c) for r in range(nr) if (r, c) in cell_grids]
            if not col_positions:
                continue
            if prop == "most_non_bg":
                src = max(col_positions, key=lambda p: _cell_non_bg_count(cell_grids[p], bg))
            elif prop == "least_non_bg_gt0":
                gt0 = [p for p in col_positions if _cell_non_bg_count(cell_grids[p], bg) > 0]
                src = min(gt0, key=lambda p: _cell_non_bg_count(cell_grids[p], bg)) if gt0 else col_positions[0]
            elif prop == "most_colors":
                src = max(col_positions, key=lambda p: _cell_n_unique_colors(cell_grids[p], bg))
            else:
                return
            for r in range(nr):
                if (r, c) in cell_grids:
                    assignment[(r, c)] = src
    else:
        return

    # Execute assignment: write source cell into target position
    for target_pos, source_pos in assignment.items():
        if target_pos not in cell_map or source_pos not in cell_grids:
            continue
        target_cell = cell_map[target_pos]
        source_grid = cell_grids[source_pos]
        r0, c0, r1, c1 = target_cell.bbox
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        sh, sw = source_grid.shape
        if (th, tw) == (sh, sw):
            state.output_grid[r0 : r1 + 1, c0 : c1 + 1] = source_grid
