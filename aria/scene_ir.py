from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


Cell = tuple[int, int]
BBox = tuple[int, int, int, int]


class EntityKind(str, Enum):
    SCENE = "scene"
    PANEL = "panel"
    OBJECT = "object"
    TEMPLATE = "template"
    BOUNDARY = "boundary"
    INTERIOR_REGION = "interior_region"
    SEPARATOR = "separator"
    SLOT = "slot"
    SLOT_GRID = "slot_grid"
    ORDERED_CHAIN = "ordered_chain"
    CLONE_SERIES = "clone_series"
    REFERENCE_OBJECT = "reference_object"
    LEGEND_MAP = "legend_map"
    SHAPE_KEY = "shape_key"
    CANONICAL_FORM = "canonical_form"
    OBJECT_GROUP = "object_group"


class RelationKind(str, Enum):
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    ENCLOSES = "encloses"
    CORRESPONDS_TO = "corresponds_to"
    INDEXED_BY = "indexed_by"
    ORDERED_BEFORE = "ordered_before"
    COPIES_FROM = "copies_from"
    USES_KEY = "uses_key"
    ALIGNED_WITH = "aligned_with"
    EXTENDS_ALONG = "extends_along"


class StepOp(str, Enum):
    INFER_OUTPUT_SIZE = "infer_output_size"
    INFER_OUTPUT_BACKGROUND = "infer_output_background"
    INITIALIZE_OUTPUT_SCENE = "initialize_output_scene"
    PARSE_SCENE = "parse_scene"
    EXTRACT_TEMPLATE = "extract_template"
    INFER_TILE_LAYOUT = "infer_tile_layout"
    STAMP_TEMPLATE = "stamp_template"
    EXTRACT_REFERENCE_KEY = "extract_reference_key"
    LOOKUP_COLOR = "lookup_color"
    RECOLOR_OBJECT = "recolor_object"
    DETECT_CLOSED_BOUNDARIES = "detect_closed_boundaries"
    FILL_ENCLOSED_REGIONS = "fill_enclosed_regions"
    SPLIT_BY_SEPARATOR = "split_by_separator"
    BOOLEAN_COMBINE_PANELS = "boolean_combine_panels"
    CANONICALIZE_OBJECT = "canonicalize_object"
    EXTEND_PERIODIC_PATTERN = "extend_periodic_pattern"
    ORDER_OBJECTS_INTO_CHAIN = "order_objects_into_chain"
    PLACE_CHAIN = "place_chain"
    CLONE_ALONG_PATH = "clone_along_path"
    RENDER_SCENE = "render_scene"
    # Generic control/data-flow operators
    SELECT_ENTITY = "select_entity"
    BUILD_CORRESPONDENCE = "build_correspondence"
    MAP_OVER_ENTITIES = "map_over_entities"
    APPLY_PER_CELL = "apply_per_cell"
    FOR_EACH_ENTITY = "for_each_entity"
    COMBINE_CELLS = "combine_cells"
    BROADCAST_TO_CELLS = "broadcast_to_cells"
    ASSIGN_CELLS = "assign_cells"


@dataclass(frozen=True)
class OutputGridSpec:
    shape: tuple[int, int]
    background: int | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneEntity:
    id: str
    kind: EntityKind
    bbox: BBox | None = None
    cells: tuple[Cell, ...] = ()
    colors: tuple[int, ...] = ()
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneRelation:
    kind: RelationKind
    source_id: str
    target_id: str
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    shape: tuple[int, int]
    background: int
    entities: tuple[SceneEntity, ...] = ()
    relations: tuple[SceneRelation, ...] = ()


@dataclass(frozen=True)
class SceneStep:
    op: StepOp
    inputs: tuple[str, ...] = ()
    params: Mapping[str, object] = field(default_factory=dict)
    output_id: str | None = None


@dataclass(frozen=True)
class SceneProgram:
    steps: tuple[SceneStep, ...]

    def step_names(self) -> tuple[str, ...]:
        return tuple(step.op.value for step in self.steps)

    def starts_with_output_spec(self) -> bool:
        if len(self.steps) < 2:
            return False
        return (
            self.steps[0].op is StepOp.INFER_OUTPUT_SIZE
            and self.steps[1].op is StepOp.INFER_OUTPUT_BACKGROUND
        )
