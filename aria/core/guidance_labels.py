"""Guidance label schemas — typed targets for future learned guidance.

Defines explicit label types for each guidance target the symbolic system
can produce. Labels are extracted from the same perception/stage-1/evidence
pipeline as the export, but structured for direct consumption by classifiers
and rankers.

Does not overfit to current rules. Grounded in the current symbolic system.
No task-id logic. No solver changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Individual label types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputSizeLabel:
    """Label for output-size prediction."""
    mode: str
    params: dict
    verified: bool

    def to_dict(self) -> dict:
        return {"mode": self.mode, "params": self.params, "verified": self.verified}

    @staticmethod
    def from_dict(d: dict) -> OutputSizeLabel:
        return OutputSizeLabel(mode=d["mode"], params=d["params"], verified=d["verified"])


@dataclass(frozen=True)
class DerivationLabel:
    """Label for output-derivation prediction."""
    candidate_kind: str
    relation: str
    selector: str
    params: dict
    verified: bool

    def to_dict(self) -> dict:
        return {
            "candidate_kind": self.candidate_kind,
            "relation": self.relation,
            "selector": self.selector,
            "params": self.params,
            "verified": self.verified,
        }

    @staticmethod
    def from_dict(d: dict) -> DerivationLabel:
        return DerivationLabel(
            candidate_kind=d["candidate_kind"],
            relation=d["relation"],
            selector=d["selector"],
            params=d["params"],
            verified=d["verified"],
        )


@dataclass(frozen=True)
class RoleLabel:
    """Label for structural role assignment."""
    bg_color: int
    has_frame: bool
    frame_color: int | None
    has_marker: bool
    marker_color: int | None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> RoleLabel:
        return RoleLabel(**d)


@dataclass(frozen=True)
class LegendLabel:
    """Label for legend detection."""
    present: bool
    edge: str | None = None
    n_entries: int = 0
    region_bbox: tuple[int, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "edge": self.edge,
            "n_entries": self.n_entries,
            "region_bbox": list(self.region_bbox) if self.region_bbox else None,
        }

    @staticmethod
    def from_dict(d: dict) -> LegendLabel:
        bbox = tuple(d["region_bbox"]) if d.get("region_bbox") else None
        return LegendLabel(
            present=d["present"],
            edge=d.get("edge"),
            n_entries=d.get("n_entries", 0),
            region_bbox=bbox,
        )


@dataclass(frozen=True)
class SlotGridLabel:
    """Label for slot-grid detection."""
    present: bool
    n_rows: int | None = None
    n_cols: int | None = None
    slot_dims: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "slot_dims": list(self.slot_dims) if self.slot_dims else None,
        }

    @staticmethod
    def from_dict(d: dict) -> SlotGridLabel:
        sd = tuple(d["slot_dims"]) if d.get("slot_dims") else None
        return SlotGridLabel(
            present=d["present"],
            n_rows=d.get("n_rows"),
            n_cols=d.get("n_cols"),
            slot_dims=sd,
        )


@dataclass(frozen=True)
class CorrespondenceLabel:
    """Label for correspondence detection."""
    has_correspondences: bool
    kind: str | None = None
    n_pairs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> CorrespondenceLabel:
        return CorrespondenceLabel(**d)


@dataclass(frozen=True)
class ProgramFamilyLabel:
    """Label for program family prediction."""
    top_family: str
    top_score: float
    alt_families: tuple[str, ...]
    gate_passed: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "top_family": self.top_family,
            "top_score": self.top_score,
            "alt_families": list(self.alt_families),
            "gate_passed": list(self.gate_passed),
        }

    @staticmethod
    def from_dict(d: dict) -> ProgramFamilyLabel:
        return ProgramFamilyLabel(
            top_family=d["top_family"],
            top_score=d["top_score"],
            alt_families=tuple(d["alt_families"]),
            gate_passed=tuple(d["gate_passed"]),
        )


# ---------------------------------------------------------------------------
# Composite label for a task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskGuidanceLabels:
    """All guidance labels for a single task."""
    task_id: str
    output_size: OutputSizeLabel | None
    derivation: DerivationLabel | None
    roles: RoleLabel
    legend: LegendLabel
    slot_grid: SlotGridLabel
    correspondences: CorrespondenceLabel
    program_family: ProgramFamilyLabel

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "output_size": self.output_size.to_dict() if self.output_size else None,
            "derivation": self.derivation.to_dict() if self.derivation else None,
            "roles": self.roles.to_dict(),
            "legend": self.legend.to_dict(),
            "slot_grid": self.slot_grid.to_dict(),
            "correspondences": self.correspondences.to_dict(),
            "program_family": self.program_family.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> TaskGuidanceLabels:
        return TaskGuidanceLabels(
            task_id=d["task_id"],
            output_size=OutputSizeLabel.from_dict(d["output_size"]) if d.get("output_size") else None,
            derivation=DerivationLabel.from_dict(d["derivation"]) if d.get("derivation") else None,
            roles=RoleLabel.from_dict(d["roles"]),
            legend=LegendLabel.from_dict(d["legend"]),
            slot_grid=SlotGridLabel.from_dict(d["slot_grid"]),
            correspondences=CorrespondenceLabel.from_dict(d["correspondences"]),
            program_family=ProgramFamilyLabel.from_dict(d["program_family"]),
        )


# ---------------------------------------------------------------------------
# Label extraction from demos
# ---------------------------------------------------------------------------


def extract_labels(
    task_id: str,
    demos: tuple[DemoPair, ...],
) -> TaskGuidanceLabels:
    """Extract guidance labels from task demos.

    Runs perception, stage-1, and mechanism evidence to produce typed labels.
    """
    from aria.core.grid_perception import perceive_grid
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.output_stage1 import infer_output_stage1_spec
    from aria.core.relations import build_legend_mapping, detect_slot_grid

    # Perception on first demo
    state = perceive_grid(demos[0].input) if demos else None

    # Stage-1
    stage1 = infer_output_stage1_spec(demos)

    output_size = None
    if stage1 is not None:
        output_size = OutputSizeLabel(
            mode=stage1.size_spec.mode,
            params=dict(stage1.size_spec.params),
            verified=True,
        )

    derivation = None
    if stage1 is not None and stage1.derivation_spec is not None:
        ds = stage1.derivation_spec
        derivation = DerivationLabel(
            candidate_kind=ds.candidate_kind,
            relation=ds.relation,
            selector=ds.selector,
            params=dict(ds.params),
            verified=True,
        )

    # Roles
    bg_color = state.bg_color if state else 0
    has_frame = len(state.framed_regions) > 0 if state else False
    frame_color = None
    has_marker = False
    marker_color = None
    if state:
        for r in state.roles:
            if r.role.name == "FRAME" and r.color is not None:
                frame_color = r.color
            if r.role.name == "MARKER" and r.color is not None:
                has_marker = True
                marker_color = r.color
    roles = RoleLabel(
        bg_color=bg_color,
        has_frame=has_frame,
        frame_color=frame_color,
        has_marker=has_marker,
        marker_color=marker_color,
    )

    # Legend
    legend_label = LegendLabel(present=False)
    if state:
        lm = build_legend_mapping(state)
        if lm is not None:
            legend_label = LegendLabel(
                present=True,
                edge=lm.edge,
                n_entries=len(lm.key_to_value),
                region_bbox=tuple(lm.legend_bbox),
            )

    # Slot grid
    slot_grid_label = SlotGridLabel(present=False)
    if state:
        sg = detect_slot_grid(state)
        if sg is not None:
            slot_grid_label = SlotGridLabel(
                present=True,
                n_rows=sg.n_rows,
                n_cols=sg.n_cols,
                slot_dims=(sg.slot_height, sg.slot_width),
            )

    # Correspondences
    from aria.core.relations import verify_zone_summary_grid
    zsm = verify_zone_summary_grid(demos)
    correspondences = CorrespondenceLabel(
        has_correspondences=zsm is not None,
        kind=zsm.mapping_kind if zsm else None,
        n_pairs=len(zsm.zone_order) if zsm else 0,
    )

    # Mechanism evidence / program family
    evidence, ranking = compute_evidence_and_rank(demos)
    top = ranking.lanes[0] if ranking.lanes else None
    program_family = ProgramFamilyLabel(
        top_family=top.name if top else "",
        top_score=round(top.final_score, 3) if top else 0.0,
        alt_families=tuple(
            c.name for c in ranking.lanes[1:] if c.gate_pass and c.final_score > 0.1
        ),
        gate_passed=tuple(c.name for c in ranking.lanes if c.gate_pass),
    )

    return TaskGuidanceLabels(
        task_id=task_id,
        output_size=output_size,
        derivation=derivation,
        roles=roles,
        legend=legend_label,
        slot_grid=slot_grid_label,
        correspondences=correspondences,
        program_family=program_family,
    )


# ---------------------------------------------------------------------------
# Batch / IO
# ---------------------------------------------------------------------------


def save_labels(
    labels: list[TaskGuidanceLabels],
    path: str | Path,
) -> None:
    """Save labels as JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for l in labels:
            f.write(json.dumps(l.to_dict(), sort_keys=True) + "\n")


def load_labels(path: str | Path) -> list[TaskGuidanceLabels]:
    """Load labels from JSONL."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(TaskGuidanceLabels.from_dict(json.loads(line)))
    return labels


def label_coverage(labels: list[TaskGuidanceLabels]) -> dict[str, Any]:
    """Report how many tasks have each label type populated."""
    n = len(labels)
    if n == 0:
        return {"total": 0}

    return {
        "total": n,
        "has_size_spec": sum(1 for l in labels if l.output_size is not None),
        "has_derivation": sum(1 for l in labels if l.derivation is not None),
        "has_frame": sum(1 for l in labels if l.roles.has_frame),
        "has_marker": sum(1 for l in labels if l.roles.has_marker),
        "has_legend": sum(1 for l in labels if l.legend.present),
        "has_slot_grid": sum(1 for l in labels if l.slot_grid.present),
        "has_correspondences": sum(1 for l in labels if l.correspondences.has_correspondences),
        "has_program_family": sum(1 for l in labels if l.program_family.top_family),
    }
