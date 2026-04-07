"""Factorized target space for program synthesis.

Six independent factor dimensions that jointly describe a program skeleton.
Each factor is a small enum. Valid combinations are filtered by an explicit
compatibility table.

The factorization is grounded in the existing executor/IR — every factor
value maps to concrete scene-IR constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterator


# ---------------------------------------------------------------------------
# Factor 1: Decomposition — how to parse the input grid
# ---------------------------------------------------------------------------


class Decomposition(str, Enum):
    OBJECT = "object"
    FRAME = "frame"
    REGION = "region"
    PARTITION = "partition"
    ZONE = "zone"
    MASK = "mask"
    PROPAGATION = "propagation"


# ---------------------------------------------------------------------------
# Factor 2: Selector — which entities to operate on
# ---------------------------------------------------------------------------


class Selector(str, Enum):
    OBJECT_SELECT = "object_select"
    REGION_SELECT = "region_select"
    FRAME_INTERIOR = "frame_interior"
    ENCLOSED = "enclosed"
    MARKER = "marker"
    CELL_PANEL = "cell_panel"
    NONE = "none"


# ---------------------------------------------------------------------------
# Factor 3: Scope — spatial extent of the operation
# ---------------------------------------------------------------------------


class Scope(str, Enum):
    GLOBAL = "global"
    OBJECT = "object"
    OBJECT_BBOX = "object_bbox"
    FRAME_INTERIOR = "frame_interior"
    PARTITION_CELL = "partition_cell"
    ENCLOSED_SUBSET = "enclosed_subset"
    LOCAL_SUPPORT = "local_support"
    REGION_LOCAL = "region_local"


# ---------------------------------------------------------------------------
# Factor 4: Op — what transformation to apply
# ---------------------------------------------------------------------------


class Op(str, Enum):
    RECOLOR = "recolor"
    FILL = "fill"
    TRANSFORM = "transform"
    COPY_STAMP = "copy_stamp"
    COMBINE = "combine"
    REPAIR = "repair"
    GROW_PROPAGATE = "grow_propagate"
    EXTRACT = "extract"


# ---------------------------------------------------------------------------
# Factor 5: Correspondence — how entities are matched across input/output
# ---------------------------------------------------------------------------


class Correspondence(str, Enum):
    NONE = "none"
    POSITIONAL = "positional"
    SHAPE_BASED = "shape_based"
    SOURCE_TARGET = "source_target"
    ASSIGNMENT = "assignment"
    OBJECT_MATCH = "object_match"


# ---------------------------------------------------------------------------
# Factor 6: Depth — number of composition steps
# ---------------------------------------------------------------------------


class Depth(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3


# ---------------------------------------------------------------------------
# FactorSet — a complete factor combination
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSet:
    decomposition: Decomposition
    selector: Selector
    scope: Scope
    op: Op
    correspondence: Correspondence
    depth: Depth

    def as_tuple(self) -> tuple:
        return (
            self.decomposition,
            self.selector,
            self.scope,
            self.op,
            self.correspondence,
            self.depth,
        )

    def __repr__(self) -> str:
        return (
            f"F({self.decomposition.value}, {self.selector.value}, "
            f"{self.scope.value}, {self.op.value}, "
            f"{self.correspondence.value}, d={self.depth.value})"
        )


# ---------------------------------------------------------------------------
# Compatibility constraints — prune structurally impossible combos
# ---------------------------------------------------------------------------

# Decomposition → allowed selectors
_DECOMP_SELECTORS: dict[Decomposition, frozenset[Selector]] = {
    Decomposition.OBJECT: frozenset({
        Selector.OBJECT_SELECT, Selector.NONE, Selector.MARKER,
        Selector.ENCLOSED,
    }),
    Decomposition.FRAME: frozenset({
        Selector.FRAME_INTERIOR, Selector.NONE, Selector.ENCLOSED,
    }),
    Decomposition.REGION: frozenset({
        Selector.REGION_SELECT, Selector.OBJECT_SELECT, Selector.NONE,
        Selector.ENCLOSED,
    }),
    Decomposition.PARTITION: frozenset({
        Selector.CELL_PANEL, Selector.NONE,
    }),
    Decomposition.ZONE: frozenset({
        Selector.CELL_PANEL, Selector.NONE,
    }),
    Decomposition.MASK: frozenset({
        Selector.MARKER, Selector.NONE, Selector.REGION_SELECT,
    }),
    Decomposition.PROPAGATION: frozenset({
        Selector.NONE, Selector.OBJECT_SELECT, Selector.REGION_SELECT,
    }),
}

# Decomposition → allowed scopes
_DECOMP_SCOPES: dict[Decomposition, frozenset[Scope]] = {
    Decomposition.OBJECT: frozenset({
        Scope.GLOBAL, Scope.OBJECT, Scope.OBJECT_BBOX, Scope.REGION_LOCAL,
    }),
    Decomposition.FRAME: frozenset({
        Scope.GLOBAL, Scope.FRAME_INTERIOR,
    }),
    Decomposition.REGION: frozenset({
        Scope.GLOBAL, Scope.OBJECT, Scope.OBJECT_BBOX,
        Scope.ENCLOSED_SUBSET, Scope.REGION_LOCAL,
    }),
    Decomposition.PARTITION: frozenset({
        Scope.GLOBAL, Scope.PARTITION_CELL,
    }),
    Decomposition.ZONE: frozenset({
        Scope.GLOBAL, Scope.PARTITION_CELL,
    }),
    Decomposition.MASK: frozenset({
        Scope.GLOBAL, Scope.LOCAL_SUPPORT, Scope.REGION_LOCAL,
    }),
    Decomposition.PROPAGATION: frozenset({
        Scope.GLOBAL, Scope.OBJECT, Scope.REGION_LOCAL,
    }),
}

# Selector → allowed scopes (further restriction)
_SELECTOR_SCOPES: dict[Selector, frozenset[Scope]] = {
    Selector.OBJECT_SELECT: frozenset({
        Scope.OBJECT, Scope.OBJECT_BBOX, Scope.GLOBAL, Scope.REGION_LOCAL,
    }),
    Selector.REGION_SELECT: frozenset({
        Scope.REGION_LOCAL, Scope.GLOBAL, Scope.ENCLOSED_SUBSET,
    }),
    Selector.FRAME_INTERIOR: frozenset({
        Scope.FRAME_INTERIOR, Scope.GLOBAL,
    }),
    Selector.ENCLOSED: frozenset({
        Scope.ENCLOSED_SUBSET, Scope.GLOBAL,
    }),
    Selector.MARKER: frozenset({
        Scope.LOCAL_SUPPORT, Scope.GLOBAL, Scope.OBJECT,
    }),
    Selector.CELL_PANEL: frozenset({
        Scope.PARTITION_CELL, Scope.GLOBAL,
    }),
    Selector.NONE: frozenset(Scope),  # any scope
}

# Op → requires correspondence?
_OP_NEEDS_CORRESPONDENCE: frozenset[Op] = frozenset({
    Op.COPY_STAMP,
})

# Correspondence != NONE only makes sense with certain ops
_CORRESPONDENCE_OPS: frozenset[Op] = frozenset({
    Op.COPY_STAMP, Op.RECOLOR, Op.TRANSFORM, Op.EXTRACT,
})

# Depth constraints: depth=3 only with certain ops
_DEPTH3_OPS: frozenset[Op] = frozenset({
    Op.TRANSFORM, Op.RECOLOR, Op.COPY_STAMP,
})


def is_compatible(fs: FactorSet) -> bool:
    """Check if a factor combination is structurally valid."""
    # Decomposition → selector compatibility
    allowed_sel = _DECOMP_SELECTORS.get(fs.decomposition)
    if allowed_sel is not None and fs.selector not in allowed_sel:
        return False

    # Decomposition → scope compatibility
    allowed_scope = _DECOMP_SCOPES.get(fs.decomposition)
    if allowed_scope is not None and fs.scope not in allowed_scope:
        return False

    # Selector → scope compatibility
    allowed_scope2 = _SELECTOR_SCOPES.get(fs.selector)
    if allowed_scope2 is not None and fs.scope not in allowed_scope2:
        return False

    # Correspondence constraints
    if fs.correspondence != Correspondence.NONE:
        if fs.op not in _CORRESPONDENCE_OPS:
            return False
    if fs.op in _OP_NEEDS_CORRESPONDENCE:
        if fs.correspondence == Correspondence.NONE:
            return False

    # Depth=3 only for certain ops
    if fs.depth == Depth.THREE and fs.op not in _DEPTH3_OPS:
        return False

    # Depth=1 + correspondence makes no sense (need select+correspond+apply = 2+)
    if fs.depth == Depth.ONE and fs.correspondence != Correspondence.NONE:
        return False

    # Selector.NONE + Op.EXTRACT makes no sense (extract what?)
    # But CELL_PANEL + EXTRACT is valid (extract from a cell)
    if fs.selector == Selector.NONE and fs.op == Op.EXTRACT:
        return False
    # MARKER + EXTRACT also doesn't make sense
    if fs.selector == Selector.MARKER and fs.op == Op.EXTRACT:
        return False

    # GROW_PROPAGATE only with PROPAGATION or MASK decomposition
    if fs.op == Op.GROW_PROPAGATE:
        if fs.decomposition not in (Decomposition.PROPAGATION, Decomposition.MASK):
            return False

    # COMBINE only with PARTITION or OBJECT decomposition
    if fs.op == Op.COMBINE:
        if fs.decomposition not in (Decomposition.PARTITION, Decomposition.OBJECT):
            return False

    # REPAIR only with MASK, FRAME, or PROPAGATION decomposition
    if fs.op == Op.REPAIR:
        if fs.decomposition not in (
            Decomposition.MASK, Decomposition.FRAME, Decomposition.PROPAGATION,
        ):
            return False

    # FILL only makes sense with region/enclosed/frame/partition scopes
    if fs.op == Op.FILL:
        if fs.scope in (Scope.OBJECT, Scope.LOCAL_SUPPORT):
            return False

    # TRANSFORM at GLOBAL scope with Selector.NONE is the only global transform
    # (object-level transforms need a selector)
    if fs.op == Op.TRANSFORM and fs.scope != Scope.GLOBAL:
        if fs.selector == Selector.NONE:
            return False

    # Correspondence only meaningful with certain decompositions
    if fs.correspondence != Correspondence.NONE:
        if fs.decomposition not in (
            Decomposition.OBJECT, Decomposition.REGION,
            Decomposition.PARTITION, Decomposition.ZONE,
        ):
            return False

    return True


def enumerate_compatible() -> list[FactorSet]:
    """Enumerate all structurally valid factor combinations."""
    result = []
    for d, sel, sc, op, corr, dep in product(
        Decomposition, Selector, Scope, Op, Correspondence, Depth,
    ):
        fs = FactorSet(d, sel, sc, op, corr, dep)
        if is_compatible(fs):
            result.append(fs)
    return result


def enumerate_compatible_for(
    *,
    decomposition: Decomposition | None = None,
    selector: Selector | None = None,
    scope: Scope | None = None,
    op: Op | None = None,
    correspondence: Correspondence | None = None,
    depth: Depth | None = None,
) -> list[FactorSet]:
    """Enumerate compatible combos with some factors pinned."""
    result = []
    for d, sel, sc, o, corr, dep in product(
        [decomposition] if decomposition else Decomposition,
        [selector] if selector else Selector,
        [scope] if scope else Scope,
        [op] if op else Op,
        [correspondence] if correspondence else Correspondence,
        [depth] if depth else Depth,
    ):
        fs = FactorSet(d, sel, sc, o, corr, dep)
        if is_compatible(fs):
            result.append(fs)
    return result


# ---------------------------------------------------------------------------
# Factor metadata — for feature extraction and labeling
# ---------------------------------------------------------------------------

FACTOR_NAMES = (
    "decomposition", "selector", "scope", "op", "correspondence", "depth",
)

FACTOR_ENUMS = {
    "decomposition": Decomposition,
    "selector": Selector,
    "scope": Scope,
    "op": Op,
    "correspondence": Correspondence,
    "depth": Depth,
}

FACTOR_SIZES = {name: len(enum) for name, enum in FACTOR_ENUMS.items()}
