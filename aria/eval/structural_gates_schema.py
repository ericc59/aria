"""Structural gates ontology and gold annotation schema.

Defines a deliberately small, stable label set for measuring
structural induction quality at each pipeline stage.

The ontology is for evaluation only — never imported by the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Part A: Small ontology
# ---------------------------------------------------------------------------


class DecompLabel(str, Enum):
    OBJECT = "object"
    FRAME = "frame"
    PANEL = "panel"
    PARTITION = "partition"
    REGION = "region"
    HOST_SLOT = "host_slot"


class EntityKind(str, Enum):
    OBJECT = "object"
    MARKER = "marker"
    HOST = "host"
    GAP = "gap"
    PANEL = "panel"
    REGION = "region"


class RelationKind(str, Enum):
    CONTAINS = "contains"
    PAIRED_WITH = "paired_with"
    ALIGNED_ROW = "aligned_row"
    ALIGNED_COL = "aligned_col"
    HOST_OF = "host_of"
    GAP_OF = "gap_of"
    ADJACENT_TO = "adjacent_to"


class TemplateFamily(str, Enum):
    MATCH_RECOLOR = "match_recolor"
    HOST_SLOT_PLACE = "host_slot_place"
    EXTRACT_MODIFY = "extract_modify"
    PANEL_COMBINE_REWRITE = "panel_combine_rewrite"
    REGION_FILL = "region_fill"
    SWAP = "swap"


# ---------------------------------------------------------------------------
# Part B: Gold annotation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldEntity:
    name: str
    kind: EntityKind
    selector_note: str = ""

    def matches_kind(self, other_kind: str) -> bool:
        """Coarse kind match against system-produced entity kinds."""
        return self.kind.value == other_kind


@dataclass(frozen=True)
class GoldRelation:
    kind: RelationKind
    source: str  # entity name
    target: str  # entity name


@dataclass(frozen=True)
class GoldTask:
    task_id: str
    decomposition: DecompLabel
    entities: tuple[GoldEntity, ...]
    relations: tuple[GoldRelation, ...]
    template: TemplateFamily
    critical_slots: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_names(self) -> frozenset[str]:
        return frozenset(e.name for e in self.entities)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _parse_entity(d: dict) -> GoldEntity:
    return GoldEntity(
        name=d["name"],
        kind=EntityKind(d["kind"]),
        selector_note=d.get("selector_note", ""),
    )


def _parse_relation(d: dict) -> GoldRelation:
    return GoldRelation(
        kind=RelationKind(d["kind"]),
        source=d["source"],
        target=d["target"],
    )


def _parse_task(d: dict) -> GoldTask:
    return GoldTask(
        task_id=d["task_id"],
        decomposition=DecompLabel(d["decomposition"]),
        entities=tuple(_parse_entity(e) for e in d.get("entities", [])),
        relations=tuple(_parse_relation(r) for r in d.get("relations", [])),
        template=TemplateFamily(d["template"]),
        critical_slots=dict(d.get("critical_slots", {})),
    )


def load_gold_tasks(path: str | Path) -> list[GoldTask]:
    """Load gold annotations from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    tasks_data = data if isinstance(data, list) else data.get("tasks", [])
    return [_parse_task(t) for t in tasks_data]


def load_gold_tasks_map(path: str | Path) -> dict[str, GoldTask]:
    """Load gold annotations keyed by task_id."""
    return {t.task_id: t for t in load_gold_tasks(path)}
