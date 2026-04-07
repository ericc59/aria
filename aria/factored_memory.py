"""Factored memory records for retrieval over symbolic abstractions.

Stores decomposition types, selector/scope/op/correspondence families,
perception signatures, and repair paths from solved programs. Retrieval
operates over these abstractions to bias search, never to bypass
symbolic verification.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Perception key — compact structural features for retrieval matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionKey:
    """Compact structural features extracted from cross-demo perception."""

    dims_relation: str          # "same", "grow", "shrink", "reshape"
    palette_relation: str       # "subset", "same", "new_colors"
    change_type: str            # "additive", "bg_preserved", "dense"
    object_count_bucket: str    # "none", "single", "few", "many"
    has_partition: bool
    has_frame: bool
    has_marker: bool
    has_legend: bool
    symmetry_tags: tuple[str, ...]
    color_count_bucket: str     # "few", "medium", "many"
    partition_cell_count: int | None

    def field_overlap(self, other: PerceptionKey) -> int:
        """Count of matching scalar fields between two keys."""
        score = 0
        if self.dims_relation == other.dims_relation:
            score += 1
        if self.palette_relation == other.palette_relation:
            score += 1
        if self.change_type == other.change_type:
            score += 1
        if self.object_count_bucket == other.object_count_bucket:
            score += 1
        if self.has_partition == other.has_partition:
            score += 1
        if self.has_frame == other.has_frame:
            score += 1
        if self.has_marker == other.has_marker:
            score += 1
        if self.has_legend == other.has_legend:
            score += 1
        if self.color_count_bucket == other.color_count_bucket:
            score += 1
        if self.partition_cell_count == other.partition_cell_count:
            score += 1
        # Symmetry tags: count intersection
        shared = set(self.symmetry_tags) & set(other.symmetry_tags)
        score += len(shared)
        return score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerceptionKey:
        return cls(
            dims_relation=d["dims_relation"],
            palette_relation=d["palette_relation"],
            change_type=d["change_type"],
            object_count_bucket=d["object_count_bucket"],
            has_partition=d["has_partition"],
            has_frame=d["has_frame"],
            has_marker=d["has_marker"],
            has_legend=d["has_legend"],
            symmetry_tags=tuple(d.get("symmetry_tags", ())),
            color_count_bucket=d["color_count_bucket"],
            partition_cell_count=d.get("partition_cell_count"),
        )


# ---------------------------------------------------------------------------
# Repair path — how a near-miss became a solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairPath:
    """How a near-miss was turned into a solve."""

    initial_error_type: str         # "pixel_mismatch", "dims_mismatch"
    repair_kind: str                # "beam_mutation", "sketch_recompile", "edit_search"
    mutations_applied: tuple[str, ...]
    rounds_to_solve: int
    initial_pixel_diff: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RepairPath:
        return cls(
            initial_error_type=d["initial_error_type"],
            repair_kind=d["repair_kind"],
            mutations_applied=tuple(d.get("mutations_applied", ())),
            rounds_to_solve=d["rounds_to_solve"],
            initial_pixel_diff=d.get("initial_pixel_diff"),
        )


# ---------------------------------------------------------------------------
# Factored record — one solved program's abstract knowledge
# ---------------------------------------------------------------------------


def _compute_record_id(
    decomposition_type: str,
    selector_family: str,
    scope_family: str,
    op_family: str,
    correspondence: str,
    composition_depth: int,
    perception_key: PerceptionKey,
) -> str:
    """Deterministic content-based record ID."""
    content = (
        f"{decomposition_type}|{selector_family}|{scope_family}|"
        f"{op_family}|{correspondence}|{composition_depth}|"
        f"{perception_key.dims_relation}|{perception_key.palette_relation}|"
        f"{perception_key.change_type}|{perception_key.object_count_bucket}|"
        f"{perception_key.color_count_bucket}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FactoredRecord:
    """One solved factorized program's abstract knowledge."""

    record_id: str
    # Provenance
    task_ids: tuple[str, ...]
    source: str  # "scene_solve", "sketch", "consensus", "observation", "search"
    # Factor tuple
    decomposition_type: str     # "framed_periodic", "composite_role", "partition_cell", etc.
    selector_family: str        # "by_color", "by_size_rank", "largest_bbox", etc.
    scope_family: str           # "same_dims", "extract", "partition_cell", "frame_interior"
    op_family: str              # "recolor", "fill", "transform", "stamp", "color_map"
    correspondence: str         # "1:1_object", "cell_to_cell", "none"
    composition_depth: int
    # Perception signature
    perception_key: PerceptionKey
    # Task signatures (existing system)
    task_signatures: tuple[str, ...]
    # Verification
    verified: bool
    verify_mode: str            # "stateless", "leave_one_out", "sequential"
    # Optional repair path
    repair_path: RepairPath | None

    @property
    def distinct_task_count(self) -> int:
        return len(self.task_ids)

    def factor_tuple(self) -> tuple[str, str, str, str, str]:
        """The canonical factor tuple for this record."""
        return (
            self.decomposition_type,
            self.selector_family,
            self.scope_family,
            self.op_family,
            self.correspondence,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "record_id": self.record_id,
            "task_ids": list(self.task_ids),
            "source": self.source,
            "decomposition_type": self.decomposition_type,
            "selector_family": self.selector_family,
            "scope_family": self.scope_family,
            "op_family": self.op_family,
            "correspondence": self.correspondence,
            "composition_depth": self.composition_depth,
            "perception_key": self.perception_key.to_dict(),
            "task_signatures": list(self.task_signatures),
            "verified": self.verified,
            "verify_mode": self.verify_mode,
            "repair_path": self.repair_path.to_dict() if self.repair_path else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FactoredRecord:
        repair = d.get("repair_path")
        return cls(
            record_id=d["record_id"],
            task_ids=tuple(d.get("task_ids", ())),
            source=d.get("source", "unknown"),
            decomposition_type=d.get("decomposition_type", "unknown"),
            selector_family=d.get("selector_family", "unknown"),
            scope_family=d.get("scope_family", "unknown"),
            op_family=d.get("op_family", "unknown"),
            correspondence=d.get("correspondence", "none"),
            composition_depth=d.get("composition_depth", 1),
            perception_key=PerceptionKey.from_dict(d["perception_key"]),
            task_signatures=tuple(d.get("task_signatures", ())),
            verified=d.get("verified", True),
            verify_mode=d.get("verify_mode", "stateless"),
            repair_path=RepairPath.from_dict(repair) if repair else None,
        )


# ---------------------------------------------------------------------------
# Factored memory store — JSON-persisted, deduped by record_id
# ---------------------------------------------------------------------------


class FactoredMemoryStore:
    """Deduplicated set of factored records, persisted to disk."""

    def __init__(self) -> None:
        self._records: dict[str, FactoredRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add_record(self, record: FactoredRecord) -> FactoredRecord:
        """Add or merge a record. Merges task_ids and task_signatures."""
        existing = self._records.get(record.record_id)
        if existing is None:
            self._records[record.record_id] = record
            return record

        # Merge provenance
        merged_task_ids = tuple(sorted(set(existing.task_ids) | set(record.task_ids)))
        merged_sigs = tuple(sorted(set(existing.task_signatures) | set(record.task_signatures)))
        merged = FactoredRecord(
            record_id=record.record_id,
            task_ids=merged_task_ids,
            source=existing.source,
            decomposition_type=existing.decomposition_type,
            selector_family=existing.selector_family,
            scope_family=existing.scope_family,
            op_family=existing.op_family,
            correspondence=existing.correspondence,
            composition_depth=existing.composition_depth,
            perception_key=existing.perception_key,
            task_signatures=merged_sigs,
            verified=existing.verified or record.verified,
            verify_mode=existing.verify_mode,
            repair_path=existing.repair_path or record.repair_path,
        )
        self._records[record.record_id] = merged
        return merged

    def all_records(self) -> list[FactoredRecord]:
        return list(self._records.values())

    def get(self, record_id: str) -> FactoredRecord | None:
        return self._records.get(record_id)

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "records": [r.to_dict() for r in self._records.values()],
        }
        tmp_path = output.with_name(f".{output.name}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output)

    @classmethod
    def load_json(cls, path: str | Path) -> FactoredMemoryStore:
        source = Path(path)
        store = cls()
        if not source.exists():
            return store
        with open(source) as f:
            data = json.load(f)
        if data.get("version") != 1 or not isinstance(data.get("records"), list):
            raise ValueError(f"Unsupported factored memory format: {source}")
        for item in data["records"]:
            try:
                record = FactoredRecord.from_dict(item)
                store._records[record.record_id] = record
            except (KeyError, TypeError):
                continue
        return store
