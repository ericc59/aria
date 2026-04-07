"""Extract factor labels from existing solved traces.

Maps each solved task's family/skeleton to a FactorSet label.
Sources: guidance_proposer SKELETON_TYPES, scene_solve families,
stage-1 specs, scene programs.

No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from aria.factors import (
    Correspondence,
    Decomposition,
    Depth,
    FactorSet,
    Op,
    Scope,
    Selector,
    FACTOR_NAMES,
)
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Skeleton → FactorSet mapping
# ---------------------------------------------------------------------------

# Maps guidance_proposer SKELETON_TYPES to factor labels
SKELETON_TO_FACTORS: dict[str, FactorSet] = {
    "derivation_clone": FactorSet(
        Decomposition.REGION, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.EXTRACT, Correspondence.POSITIONAL, Depth.TWO,
    ),
    "derivation_interior": FactorSet(
        Decomposition.FRAME, Selector.FRAME_INTERIOR,
        Scope.FRAME_INTERIOR, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
    ),
    "tiled_input": FactorSet(
        Decomposition.REGION, Selector.NONE,
        Scope.GLOBAL, Op.TRANSFORM, Correspondence.NONE, Depth.ONE,
    ),
    "geometric_transform": FactorSet(
        Decomposition.OBJECT, Selector.NONE,
        Scope.GLOBAL, Op.TRANSFORM, Correspondence.NONE, Depth.ONE,
    ),
    "global_color_map": FactorSet(
        Decomposition.OBJECT, Selector.NONE,
        Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    ),
    "scoped_color_map": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT_BBOX, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    ),
    "fill_enclosed": FactorSet(
        Decomposition.REGION, Selector.ENCLOSED,
        Scope.ENCLOSED_SUBSET, Op.FILL, Correspondence.NONE, Depth.ONE,
    ),
    "zone_summary_grid": FactorSet(
        Decomposition.ZONE, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.POSITIONAL, Depth.TWO,
    ),
    "partition_cell_select": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
    ),
    "partition_per_cell": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.RECOLOR, Correspondence.NONE, Depth.TWO,
    ),
    "partition_combine": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.GLOBAL, Op.COMBINE, Correspondence.NONE, Depth.ONE,
    ),
    "mask_repair": FactorSet(
        Decomposition.MASK, Selector.MARKER,
        Scope.LOCAL_SUPPORT, Op.REPAIR, Correspondence.NONE, Depth.ONE,
    ),
    "object_transform": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.TRANSFORM, Correspondence.NONE, Depth.TWO,
    ),
    "frame_interior_edit": FactorSet(
        Decomposition.FRAME, Selector.FRAME_INTERIOR,
        Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    ),
    "bbox_extract": FactorSet(
        Decomposition.REGION, Selector.OBJECT_SELECT,
        Scope.OBJECT_BBOX, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
    ),
}

# Scene-solve family index → factor labels
SCENE_FAMILY_TO_FACTORS: dict[str, FactorSet] = {
    "select_extract_transform": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.TRANSFORM, Correspondence.NONE, Depth.TWO,
    ),
    "select_extract_colormap": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.TWO,
    ),
    "map_over_panels_summary": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.POSITIONAL, Depth.TWO,
    ),
    "boolean_combine": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.GLOBAL, Op.COMBINE, Correspondence.NONE, Depth.ONE,
    ),
    "select_panel_extract": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
    ),
    "color_bbox_select": FactorSet(
        Decomposition.REGION, Selector.OBJECT_SELECT,
        Scope.OBJECT_BBOX, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
    ),
    "per_cell_operation": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.RECOLOR, Correspondence.NONE, Depth.TWO,
    ),
    "per_object_operation": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT_BBOX, Op.FILL, Correspondence.NONE, Depth.TWO,
    ),
    "combine_broadcast": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.GLOBAL, Op.COMBINE, Correspondence.NONE, Depth.TWO,
    ),
    "combine_to_output": FactorSet(
        Decomposition.PARTITION, Selector.CELL_PANEL,
        Scope.PARTITION_CELL, Op.COMBINE, Correspondence.NONE, Depth.ONE,
    ),
    "consensus_compose:objects": FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT_BBOX, Op.FILL, Correspondence.NONE, Depth.TWO,
    ),
    "consensus_compose:enclosed": FactorSet(
        Decomposition.REGION, Selector.ENCLOSED,
        Scope.ENCLOSED_SUBSET, Op.FILL, Correspondence.NONE, Depth.ONE,
    ),
    "consensus_compose:frame": FactorSet(
        Decomposition.FRAME, Selector.FRAME_INTERIOR,
        Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    ),
}


# ---------------------------------------------------------------------------
# Factor label extraction
# ---------------------------------------------------------------------------


@dataclass
class FactorLabel:
    """Factor labels for one solved task."""
    task_id: str
    factors: FactorSet
    source: str  # "skeleton", "scene_family", "inferred"
    confidence: float = 1.0


@dataclass
class FactorLabelSet:
    """All factor labels extracted from solved tasks."""
    labels: list[FactorLabel] = field(default_factory=list)

    def factor_distribution(self, factor_name: str) -> dict[str, int]:
        """Count occurrences of each value for a factor dimension."""
        counts: Counter = Counter()
        for label in self.labels:
            val = getattr(label.factors, factor_name)
            counts[val.value if hasattr(val, 'value') else str(val)] += 1
        return dict(counts)

    def coverage_report(self) -> dict[str, dict[str, int]]:
        """Factor coverage across all labels."""
        return {name: self.factor_distribution(name) for name in FACTOR_NAMES}


def label_from_skeleton(task_id: str, skeleton_name: str) -> FactorLabel | None:
    """Map a skeleton name to factor labels."""
    factors = SKELETON_TO_FACTORS.get(skeleton_name)
    if factors is None:
        return None
    return FactorLabel(task_id=task_id, factors=factors, source="skeleton")


def label_from_scene_family(task_id: str, family_desc: str) -> FactorLabel | None:
    """Map a scene-solve family description to factor labels."""
    # Try exact match first
    factors = SCENE_FAMILY_TO_FACTORS.get(family_desc)
    if factors is not None:
        return FactorLabel(task_id=task_id, factors=factors, source="scene_family")

    # Try prefix match for consensus_compose families
    for prefix, fset in SCENE_FAMILY_TO_FACTORS.items():
        if family_desc.startswith(prefix):
            return FactorLabel(task_id=task_id, factors=fset, source="scene_family")

    return None
