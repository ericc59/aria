"""Core types for the ARIA system.

Every component imports from here. All types are immutable.
Grid is the only mutable-shaped type (numpy array), but we treat it
as a value — operations return new arrays, never mutate in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Primitive domain types
# ---------------------------------------------------------------------------

Grid = NDArray[np.uint8]  # shape (rows, cols), values 0-9

Color = int  # 0-9


class Shape(Enum):
    RECT = auto()
    LINE = auto()
    L = auto()
    T = auto()
    CROSS = auto()
    DOT = auto()
    IRREGULAR = auto()


class Dir(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class Axis(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAG_MAIN = auto()
    DIAG_ANTI = auto()


class SortDir(Enum):
    ASC = auto()
    DESC = auto()


class SizeRank(Enum):
    LARGEST = auto()
    SMALLEST = auto()


class ZoneRole(Enum):
    RULE = auto()
    DATA = auto()
    FRAME = auto()
    BORDER = auto()


class SceneRole(Enum):
    SEPARATOR = auto()
    LEGEND = auto()
    FRAME = auto()
    MARKER = auto()


# ---------------------------------------------------------------------------
# Symmetry flags
# ---------------------------------------------------------------------------

class Symmetry(Enum):
    ROT90 = auto()
    ROT180 = auto()
    REFL_H = auto()
    REFL_V = auto()
    REFL_D = auto()


class GlobalSymmetry(Enum):
    GLOBAL_ROT = auto()
    GLOBAL_REFL = auto()
    PERIODIC = auto()


# ---------------------------------------------------------------------------
# Spatial / topological relation tags
# ---------------------------------------------------------------------------

class SpatialRel(Enum):
    ABOVE = auto()
    BELOW = auto()
    LEFT = auto()
    RIGHT = auto()
    DIAGONAL = auto()


class TopoRel(Enum):
    ADJACENT = auto()
    CONTAINS = auto()
    OVERLAPS = auto()
    DISJOINT = auto()


class AlignRel(Enum):
    ALIGN_H = auto()
    ALIGN_V = auto()
    ALIGN_DIAG = auto()


class MatchRel(Enum):
    SAME_COLOR = auto()
    SAME_SHAPE = auto()
    SAME_SIZE = auto()
    MIRROR = auto()


class Property(Enum):
    COLOR = auto()
    SIZE = auto()
    SHAPE = auto()
    POS_X = auto()
    POS_Y = auto()
    SYMMETRY = auto()


# ---------------------------------------------------------------------------
# State graph nodes and edges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectNode:
    id: int
    color: int  # 0-9
    mask: NDArray[np.bool_]  # shape matches bbox (h, w)
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    shape: Shape
    symmetry: frozenset[Symmetry]
    size: int  # pixel count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectNode):
            return NotImplemented
        return (
            self.id == other.id
            and self.color == other.color
            and self.bbox == other.bbox
            and self.shape == other.shape
            and self.symmetry == other.symmetry
            and self.size == other.size
            and np.array_equal(self.mask, other.mask)
        )

    def __hash__(self) -> int:
        return hash((self.id, self.color, self.bbox, self.shape, self.size))


@dataclass(frozen=True)
class RelationEdge:
    src: int  # ObjectNode.id
    dst: int  # ObjectNode.id
    spatial: frozenset[SpatialRel]
    topo: frozenset[TopoRel]
    align: frozenset[AlignRel]
    match: frozenset[MatchRel]


@dataclass(frozen=True)
class GridContext:
    dims: tuple[int, int]  # (rows, cols)
    bg_color: int
    is_tiled: tuple[int, int] | None  # (tile_rows, tile_cols) or None
    symmetry: frozenset[GlobalSymmetry]
    palette: frozenset[int]
    obj_count: int


@dataclass(frozen=True)
class PartitionCell:
    row_idx: int
    col_idx: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1)
    dims: tuple[int, int]
    background: int
    palette: frozenset[int]
    obj_count: int


@dataclass(frozen=True)
class PartitionScene:
    separator_color: int
    separator_rows: tuple[int, ...]
    separator_cols: tuple[int, ...]
    cells: tuple[PartitionCell, ...]
    n_rows: int
    n_cols: int
    cell_shapes: tuple[tuple[int, int], ...]
    is_uniform_partition: bool


@dataclass(frozen=True)
class LegendEntry:
    key_color: int
    value_color: int


@dataclass(frozen=True)
class LegendInfo:
    region_bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1)
    entries: tuple[LegendEntry, ...]
    edge: str


@dataclass(frozen=True)
class RoleBinding:
    role: SceneRole
    object_id: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    color: int | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Delta:
    added: tuple[ObjectNode, ...]
    removed: tuple[int, ...]  # ObjectNode.ids
    modified: tuple[tuple[int, str, Any, Any], ...]  # (id, field, old, new)
    dims_changed: tuple[tuple[int, int], tuple[int, int]] | None  # (old, new)


@dataclass(frozen=True)
class StateGraph:
    objects: tuple[ObjectNode, ...]
    relations: tuple[RelationEdge, ...]
    context: GridContext
    grid: Grid  # the raw grid, kept for reference
    partition: PartitionScene | None = None
    legend: LegendInfo | None = None
    roles: tuple[RoleBinding, ...] = ()


# ---------------------------------------------------------------------------
# Zone (sub-grid region)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    grid: Grid
    x: int
    y: int
    w: int
    h: int


# ---------------------------------------------------------------------------
# Step machine types
# ---------------------------------------------------------------------------

class Type(Enum):
    """Types in the ARIA step language."""
    GRID = auto()
    OBJECT = auto()
    OBJECT_SET = auto()
    OBJECT_LIST = auto()
    COLOR = auto()
    INT = auto()
    BOOL = auto()
    DIMS = auto()
    DIR = auto()
    AXIS = auto()
    SHAPE = auto()
    PROPERTY = auto()
    COLOR_MAP = auto()
    INT_LIST = auto()
    REGION = auto()
    ZONE = auto()
    ZONE_LIST = auto()
    PAIR = auto()
    TASK_CTX = auto()
    SORT_DIR = auto()
    SIZE_RANK = auto()
    ZONE_ROLE = auto()
    # Function types for higher-order ops
    PREDICATE = auto()     # Object -> Bool
    OBJ_TRANSFORM = auto()  # Object -> Object
    GRID_TRANSFORM = auto()  # Grid -> Grid
    # Callable for generic lambdas
    CALLABLE = auto()


# ---------------------------------------------------------------------------
# Program AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """Reference to a bound name in the environment."""
    name: str


@dataclass(frozen=True)
class Literal:
    """A literal value (int, bool, Color, enum variant, etc.)."""
    value: Any
    typ: Type


@dataclass(frozen=True)
class Call:
    """Call a named operation with arguments."""
    op: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Lambda:
    """A single-parameter lambda node for higher-order ops.

    Multi-parameter surface syntax is lowered to nested Lambda nodes.
    """
    param: str
    param_type: Type
    body: Expr


Expr = Ref | Literal | Call | Lambda


@dataclass(frozen=True)
class Bind:
    """Bind a name to a typed expression."""
    name: str
    typ: Type
    expr: Expr
    declared: bool = True


@dataclass(frozen=True)
class Assert:
    """Fail-fast guard."""
    pred: Expr


Step = Bind | Assert


@dataclass(frozen=True)
class Program:
    steps: tuple[Step, ...]
    output: str  # name of the binding to yield


# ---------------------------------------------------------------------------
# Task and demo types (ARC format)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class Task:
    train: tuple[DemoPair, ...]
    test: tuple[DemoPair, ...]  # output may be None at test time


@dataclass(frozen=True)
class TaskContext:
    """Read-only view of demo pairs, available to context-reading ops."""
    demos: tuple[DemoPair, ...]


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

class VerifyMode(Enum):
    STATELESS = auto()
    LEAVE_ONE_OUT = auto()
    SEQUENTIAL = auto()


@dataclass(frozen=True)
class StepTraceEntry:
    step_name: str
    value: Any
    ok: bool
    suspect: str | None = None


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    mode: VerifyMode
    failed_demo: int | None = None
    error_type: str | None = None
    diff: dict[str, Any] | None = None
    step_trace: tuple[StepTraceEntry, ...] = ()


# ---------------------------------------------------------------------------
# Library types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LibraryEntry:
    name: str
    params: tuple[tuple[str, Type], ...]
    return_type: Type
    steps: tuple[Step, ...]
    output: str
    level: int  # 1 = permanent, 2 = task-specific
    use_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    return np.full((rows, cols), fill, dtype=np.uint8)


def grid_eq(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def grid_from_list(data: list[list[int]]) -> Grid:
    return np.array(data, dtype=np.uint8)
