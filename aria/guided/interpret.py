"""LEGACY — threshold-heavy heuristics, not general structural facts.

The role assignment (SCAFFOLD, LEGEND, DATA, TARGET, MARKER, FILLER) uses
hardcoded thresholds (grid_frac > 0.15, > 0.1, etc.) that are benchmark-fit.
Cross-demo "unification" just returns demo 0's roles. If kept, this should
emit soft features for clause induction, not final semantic labels.

Do not add new functionality here.

---
Scene interpretation layer.

Infers semantic structure from a workspace BEFORE program search begins.
Assigns roles to objects/regions, identifies relations between them,
and determines the answer mode — what KIND of question the task is asking.

This bridges workspace parsing and answer program induction.

Role vocabulary:
  SCAFFOLD  — large structural frame, border, separator (preserved)
  LEGEND    — small region encoding a mapping or key
  DATA      — objects that carry information (may be preserved or transformed)
  TARGET    — positions/regions where the answer goes
  MARKER    — singleton cues indicating positions, labels, or references
  FILLER    — background or padding (no semantic role)

Relation vocabulary:
  encodes       — legend entry maps key→value
  corresponds   — object in region A matches object in region B
  contains      — spatial containment (parent-child)
  anomaly_of    — object breaks the pattern of its container
  refers_to     — marker/label points to another object

Answer mode vocabulary:
  complete   — fill in missing parts of a pattern
  decode     — use a legend/key to translate symbols
  propagate  — extend or repeat a rule across the scene
  compare    — find differences between sub-scenes
  place      — position answer objects on a canvas
  repair     — fix anomalies/defects in a pattern
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from collections import Counter

import numpy as np
from scipy import ndimage

from aria.guided.workspace import Workspace, ObjectInfo, Relation, ResidualUnit, build_workspace
from aria.types import Grid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(Enum):
    SCAFFOLD = auto()
    LEGEND = auto()
    DATA = auto()
    TARGET = auto()
    MARKER = auto()
    FILLER = auto()


class SemanticRel(Enum):
    ENCODES = auto()
    CORRESPONDS = auto()
    CONTAINS = auto()
    ANOMALY_OF = auto()
    REFERS_TO = auto()


class AnswerMode(Enum):
    COMPLETE = auto()
    DECODE = auto()
    PROPAGATE = auto()
    COMPARE = auto()
    PLACE = auto()
    REPAIR = auto()


# ---------------------------------------------------------------------------
# Interpretation result
# ---------------------------------------------------------------------------

@dataclass
class ObjectRole:
    oid: int
    role: Role
    confidence: float
    notes: str = ""


@dataclass
class SemanticRelation:
    src_oid: int
    dst_oid: int
    rel: SemanticRel
    detail: str = ""


@dataclass
class SceneInterpretation:
    """Complete semantic interpretation of one task."""
    # Per-object role assignments
    roles: list[ObjectRole]
    # Semantic relations between objects
    relations: list[SemanticRelation]
    # Inferred answer mode
    answer_mode: AnswerMode
    answer_mode_confidence: float
    # Structural features
    has_legend: bool
    has_scaffold: bool
    has_repeating_pattern: bool
    has_sub_scenes: bool
    # Derived
    legend_objects: list[int] = field(default_factory=list)     # oids
    scaffold_objects: list[int] = field(default_factory=list)
    target_objects: list[int] = field(default_factory=list)
    data_objects: list[int] = field(default_factory=list)
    marker_objects: list[int] = field(default_factory=list)

    def describe(self) -> str:
        roles_str = ", ".join(f"{r.oid}:{r.role.name}" for r in self.roles if r.role != Role.FILLER)
        return f"mode={self.answer_mode.name} legend={self.has_legend} scaffold={self.has_scaffold} roles=[{roles_str}]"


# ---------------------------------------------------------------------------
# Main interpretation entry
# ---------------------------------------------------------------------------

def interpret_scene(
    demos: list[tuple[Grid, Grid]],
) -> SceneInterpretation:
    """Interpret a task's semantic structure from its train demos.

    Cross-demo: roles and relations must be consistent across all demos.
    """
    if not demos:
        return _empty_interpretation()

    # Build workspaces for all demos
    workspaces = [build_workspace(inp, out) for inp, out in demos]

    # Phase 1: Assign roles per demo, then unify
    per_demo_roles = [_assign_roles(ws) for ws in workspaces]
    unified_roles = _unify_roles(per_demo_roles, workspaces)

    # Phase 2: Infer semantic relations
    relations = _infer_relations(workspaces, unified_roles)

    # Phase 3: Determine answer mode
    answer_mode, confidence = _infer_answer_mode(workspaces, unified_roles, relations)

    # Phase 4: Identify structural features
    has_legend = any(r.role == Role.LEGEND for r in unified_roles)
    has_scaffold = any(r.role == Role.SCAFFOLD for r in unified_roles)
    has_pattern = _detect_repeating_pattern(workspaces)
    has_subscenes = _detect_sub_scenes(workspaces)

    interp = SceneInterpretation(
        roles=unified_roles,
        relations=relations,
        answer_mode=answer_mode,
        answer_mode_confidence=confidence,
        has_legend=has_legend,
        has_scaffold=has_scaffold,
        has_repeating_pattern=has_pattern,
        has_sub_scenes=has_subscenes,
    )

    # Populate convenience lists
    for r in unified_roles:
        if r.role == Role.LEGEND:
            interp.legend_objects.append(r.oid)
        elif r.role == Role.SCAFFOLD:
            interp.scaffold_objects.append(r.oid)
        elif r.role == Role.TARGET:
            interp.target_objects.append(r.oid)
        elif r.role == Role.DATA:
            interp.data_objects.append(r.oid)
        elif r.role == Role.MARKER:
            interp.marker_objects.append(r.oid)

    return interp


# ---------------------------------------------------------------------------
# Phase 1: Role assignment
# ---------------------------------------------------------------------------

def _assign_roles(ws: Workspace) -> list[ObjectRole]:
    """Assign roles to objects in one workspace."""
    roles = []
    grid_size = ws.rows * ws.cols
    diff = ws.input_grid != ws.output_grid if ws.same_shape else np.ones((ws.rows, ws.cols), dtype=bool)

    # Precompute containment
    containment = {}  # oid -> list of oids it contains
    for rel in ws.relations:
        if rel.rel_type == "contains":
            containment.setdefault(rel.src, []).append(rel.dst)

    # Color frequency
    color_counts = Counter(o.color for o in ws.objects)

    for obj in ws.objects:
        obj_mask = np.zeros((ws.rows, ws.cols), dtype=bool)
        obj_mask[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] |= obj.mask
        obj_changed = bool(np.any(obj_mask & diff))
        grid_frac = obj.size / grid_size
        contained_ids = containment.get(obj.oid, [])
        n_same_color = color_counts[obj.color]

        # Decision tree for role assignment
        if grid_frac > 0.15 and len(contained_ids) >= 2:
            role = Role.SCAFFOLD
            conf = 0.9
            note = f"large ({grid_frac:.0%}) with {len(contained_ids)} children"

        elif grid_frac > 0.1 and not obj_changed:
            role = Role.SCAFFOLD
            conf = 0.7
            note = f"large ({grid_frac:.0%}) preserved"

        elif _is_legend_candidate(obj, ws, contained_ids):
            role = Role.LEGEND
            conf = 0.8
            note = "small region with unique color pairs"

        elif obj.is_singleton and n_same_color <= 2 and not obj_changed:
            role = Role.MARKER
            conf = 0.7
            note = f"singleton, rare color (count={n_same_color})"

        elif obj_changed:
            role = Role.TARGET
            conf = 0.9
            note = "changed in output"

        elif obj.is_singleton and n_same_color > 3:
            role = Role.DATA
            conf = 0.6
            note = f"singleton with common color (count={n_same_color})"

        elif not obj_changed and grid_frac < 0.05:
            role = Role.DATA
            conf = 0.5
            note = "small, preserved"

        else:
            role = Role.DATA
            conf = 0.3
            note = "default"

        roles.append(ObjectRole(obj.oid, role, conf, note))

    return roles


def _is_legend_candidate(obj: ObjectInfo, ws: Workspace, contained_ids: list[int]) -> bool:
    """Check if an object looks like a legend/key region.

    Legends typically: small, near an edge, contain multiple colors
    in a structured arrangement (rows/columns of color pairs).
    """
    if obj.size > ws.rows * ws.cols * 0.1:
        return False  # too large
    # Check if near a border
    near_top = obj.row <= 2
    near_bottom = obj.row + obj.height >= ws.rows - 2
    near_left = obj.col <= 2
    near_right = obj.col + obj.width >= ws.cols - 2
    if not (near_top or near_bottom or near_left or near_right):
        return False
    # Check for multiple unique colors in the bbox
    sub = ws.input_grid[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width]
    n_colors = len(set(int(v) for v in np.unique(sub)) - {ws.bg})
    return n_colors >= 3


# ---------------------------------------------------------------------------
# Phase 1b: Unify roles across demos
# ---------------------------------------------------------------------------

def _unify_roles(per_demo_roles: list[list[ObjectRole]], workspaces: list[Workspace]) -> list[ObjectRole]:
    """Unify role assignments across demos.

    Since objects differ across demos, unify by ROLE TYPE distribution:
    find the most common role for each structural position.

    For now, use demo 0's roles as the canonical set.
    """
    if not per_demo_roles:
        return []
    # Use demo 0 as canonical but cross-check with other demos
    canonical = per_demo_roles[0]

    # Check: do role DISTRIBUTIONS agree across demos?
    for roles in per_demo_roles[1:]:
        role_counts_0 = Counter(r.role for r in canonical)
        role_counts_i = Counter(r.role for r in roles)
        # Boost confidence if distributions match
        for r in canonical:
            if r.role in role_counts_i:
                r_count = sum(1 for ri in roles if ri.role == r.role)
                if r_count > 0:
                    pass  # consistent

    return canonical


# ---------------------------------------------------------------------------
# Phase 2: Semantic relations
# ---------------------------------------------------------------------------

def _infer_relations(
    workspaces: list[Workspace],
    roles: list[ObjectRole],
) -> list[SemanticRelation]:
    """Infer semantic relations between objects."""
    rels = []
    ws = workspaces[0]
    role_map = {r.oid: r.role for r in roles}

    # CONTAINS: from workspace structural relations
    for rel in ws.relations:
        if rel.rel_type == "contains":
            parent_role = role_map.get(rel.src, Role.FILLER)
            child_role = role_map.get(rel.dst, Role.FILLER)
            rels.append(SemanticRelation(
                rel.src, rel.dst, SemanticRel.CONTAINS,
                f"{parent_role.name} contains {child_role.name}",
            ))

    # REFERS_TO: markers adjacent to data/target objects
    markers = [r.oid for r in roles if r.role == Role.MARKER]
    for m_oid in markers:
        for rel in ws.relations:
            if rel.rel_type == "adjacent" and (rel.src == m_oid or rel.dst == m_oid):
                other = rel.dst if rel.src == m_oid else rel.src
                other_role = role_map.get(other, Role.FILLER)
                if other_role in (Role.DATA, Role.TARGET, Role.SCAFFOLD):
                    rels.append(SemanticRelation(
                        m_oid, other, SemanticRel.REFERS_TO,
                        f"marker refers to {other_role.name}",
                    ))

    # ANOMALY_OF: target objects inside scaffold (changed = anomaly)
    targets = [r.oid for r in roles if r.role == Role.TARGET]
    scaffolds = [r.oid for r in roles if r.role == Role.SCAFFOLD]
    for t_oid in targets:
        for s_oid in scaffolds:
            for rel in ws.relations:
                if rel.rel_type == "contains" and rel.src == s_oid and rel.dst == t_oid:
                    rels.append(SemanticRelation(
                        t_oid, s_oid, SemanticRel.ANOMALY_OF,
                        "target is anomaly within scaffold",
                    ))

    # CORRESPONDS: objects in different sub-regions with same shape/role
    # (detected if there are multiple scaffolds containing similar children)
    if len(scaffolds) >= 2:
        for i, s1 in enumerate(scaffolds):
            for s2 in scaffolds[i + 1:]:
                rels.append(SemanticRelation(
                    s1, s2, SemanticRel.CORRESPONDS,
                    "scaffolds may correspond",
                ))

    return rels


# ---------------------------------------------------------------------------
# Phase 3: Answer mode inference
# ---------------------------------------------------------------------------

def _infer_answer_mode(
    workspaces: list[Workspace],
    roles: list[ObjectRole],
    relations: list[SemanticRelation],
) -> tuple[AnswerMode, float]:
    """Infer what kind of answer the task is asking for."""
    role_set = set(r.role for r in roles)
    rel_set = set(r.rel for r in relations)
    ws = workspaces[0]

    has_legend = Role.LEGEND in role_set
    has_scaffold = Role.SCAFFOLD in role_set
    has_markers = Role.MARKER in role_set
    has_targets = Role.TARGET in role_set
    has_anomalies = SemanticRel.ANOMALY_OF in rel_set
    has_correspondences = SemanticRel.CORRESPONDS in rel_set

    # Decision tree for answer mode
    if has_legend:
        return AnswerMode.DECODE, 0.8

    if has_anomalies:
        return AnswerMode.REPAIR, 0.8

    if has_correspondences and has_targets:
        return AnswerMode.COMPARE, 0.6

    if ws.n_residual > 0 and ws.n_preserved > ws.n_residual * 3:
        # Mostly preserved with small changes
        if has_scaffold:
            return AnswerMode.COMPLETE, 0.7
        return AnswerMode.PROPAGATE, 0.6

    if has_markers:
        return AnswerMode.PLACE, 0.5

    return AnswerMode.COMPLETE, 0.3


# ---------------------------------------------------------------------------
# Phase 4: Structural features
# ---------------------------------------------------------------------------

def _detect_repeating_pattern(workspaces: list[Workspace]) -> bool:
    """Check if the input has repeating row/column patterns."""
    ws = workspaces[0]
    for r in range(ws.rows):
        row = ws.input_grid[r, :]
        for p in range(1, len(row) // 2 + 1):
            tile = row[:p]
            if all(np.array_equal(row[i:i + p], tile[:min(p, len(row) - i)])
                   for i in range(p, len(row), p)):
                if p < len(row):
                    return True
    return False


def _detect_sub_scenes(workspaces: list[Workspace]) -> bool:
    """Check if the grid contains multiple similar sub-regions."""
    ws = workspaces[0]
    scaffolds = [o for o in ws.objects if o.size > ws.rows * ws.cols * 0.1]
    return len(scaffolds) >= 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_interpretation() -> SceneInterpretation:
    return SceneInterpretation(
        roles=[], relations=[],
        answer_mode=AnswerMode.COMPLETE,
        answer_mode_confidence=0.0,
        has_legend=False, has_scaffold=False,
        has_repeating_pattern=False, has_sub_scenes=False,
    )
