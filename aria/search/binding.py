"""Typed role/binding substrate for aria/search.

Sits between coarse perception and execution. Assigns semantic roles
to perceived entities (panels, regions, objects, strips) and expresses
relations between them. Used at derive time to guide program construction.

NOT an execution layer. NOT part of the AST. This is search-time reasoning
that narrows the space of programs to try.

Entities are references to perceived structure (panels, regions, objects).
Roles label what an entity does (legend, query, workspace, prototype, anchor).
Relations express how entities interact (maps_to, encodes, registered_at).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class Role(Enum):
    """Semantic role an entity plays in a task."""
    LEGEND = auto()       # defines a mapping/rule/codebook
    QUERY = auto()        # region to be decoded/transformed
    WORKSPACE = auto()    # region where transformation happens in-place
    PROTOTYPE = auto()    # template to be copied/transferred
    ANCHOR = auto()       # position where prototype lands
    CONTROL = auto()      # provides parameters (colors, directions)
    SOURCE = auto()       # origin of content transfer
    TARGET = auto()       # destination for transferred content
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Entity references
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityRef:
    """Reference to a perceived entity. Stable identity for binding."""
    kind: str             # 'panel', 'region', 'object', 'strip', 'cell_grid'
    index: int            # position index (0-based, in extraction order)
    demo_idx: int = -1    # which demo this was extracted from (-1 = cross-demo)

    def __repr__(self):
        return f"{self.kind}[{self.index}]"


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

class Relation(Enum):
    """Typed relation between two entities."""
    MAPS_TO = auto()          # content of A determines/modifies B
    ENCODES = auto()          # A encodes a rule applied to B
    REGISTERED_AT = auto()    # A (prototype) placed at B (anchor)
    ALIGNED_WITH = auto()     # A and B share an axis
    CONTAINS = auto()         # A spatially contains B
    SAME_STRUCTURE = auto()   # A and B have the same shape/pattern


@dataclass
class EntityRelation:
    """A typed relation between two entities."""
    source: EntityRef
    target: EntityRef
    relation: Relation
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scene binding
# ---------------------------------------------------------------------------

@dataclass
class RoleAssignment:
    """Assigns a role to an entity with confidence."""
    entity: EntityRef
    role: Role
    confidence: float = 1.0
    reason: str = ''


@dataclass
class SceneBinding:
    """Complete role/relation assignment for a scene.

    Represents the latent structure of one demo or a cross-demo consensus.
    """
    roles: list[RoleAssignment] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
    entity_grids: dict[EntityRef, np.ndarray] = field(default_factory=dict)

    def entities_with_role(self, role: Role) -> list[EntityRef]:
        return [ra.entity for ra in self.roles if ra.role == role]

    def role_of(self, entity: EntityRef) -> Role | None:
        for ra in self.roles:
            if ra.entity == entity:
                return ra.role
        return None

    def relations_from(self, entity: EntityRef) -> list[EntityRelation]:
        return [r for r in self.relations if r.source == entity]

    def relations_to(self, entity: EntityRef) -> list[EntityRelation]:
        return [r for r in self.relations if r.target == entity]


# ---------------------------------------------------------------------------
# Panel role classification
# ---------------------------------------------------------------------------

def classify_panel_roles(
    panels: list[np.ndarray],
    bg: int,
    out_shape: tuple[int, int] | None = None,
) -> list[RoleAssignment]:
    """Classify panels into roles based on structural properties.

    Heuristics:
    - Panel matching output shape → QUERY or WORKSPACE
    - Smallest panel with diverse colors → LEGEND
    - Panel with least content → CONTROL
    - Panels with similar content → one is SOURCE, other TARGET
    """
    if not panels:
        return []

    assignments = []
    n = len(panels)

    # Compute per-panel stats
    stats = []
    for i, p in enumerate(panels):
        non_bg = np.sum(p != bg)
        n_colors = len(set(int(p[r, c]) for r in range(p.shape[0])
                           for c in range(p.shape[1]) if p[r, c] != bg))
        stats.append({
            'index': i, 'shape': p.shape, 'non_bg': non_bg,
            'n_colors': n_colors, 'density': non_bg / max(p.size, 1),
        })

    # If output shape matches one panel → that panel is likely the answer/query
    if out_shape:
        for s in stats:
            if s['shape'] == out_shape:
                assignments.append(RoleAssignment(
                    entity=EntityRef('panel', s['index']),
                    role=Role.QUERY,
                    confidence=0.8,
                    reason=f"shape matches output {out_shape}",
                ))

    # Smallest panel with high color diversity → LEGEND
    by_size = sorted(stats, key=lambda s: s['non_bg'])
    for s in by_size:
        if s['n_colors'] >= 3 and s['non_bg'] < max(st['non_bg'] for st in stats) * 0.5:
            ref = EntityRef('panel', s['index'])
            if not any(ra.entity == ref for ra in assignments):
                assignments.append(RoleAssignment(
                    entity=ref, role=Role.LEGEND,
                    confidence=0.6,
                    reason=f"small+diverse: {s['n_colors']} colors, {s['non_bg']} cells",
                ))
            break

    # Remaining panels → WORKSPACE
    assigned = {ra.entity for ra in assignments}
    for s in stats:
        ref = EntityRef('panel', s['index'])
        if ref not in assigned:
            assignments.append(RoleAssignment(
                entity=ref, role=Role.WORKSPACE,
                confidence=0.4,
                reason="unassigned panel",
            ))

    return assignments


# ---------------------------------------------------------------------------
# Strip/legend detection
# ---------------------------------------------------------------------------

def find_legend_candidates(
    grid: np.ndarray,
    bg: int,
    objects: list,
    separators: list,
) -> list[RoleAssignment]:
    """Find the best legend candidate (dense color-pair block).

    Returns at most 1-2 candidates — the densest, most diverse blocks.
    """
    h, w = grid.shape
    candidates = []  # (score, r0, c0, rows, cols, n_colors)

    # Scan 2-row blocks
    for r0 in range(h - 1):
        for c0 in range(w):
            best_w = 0
            for width in range(2, min(12, w - c0 + 1)):
                block = grid[r0:r0 + 2, c0:c0 + width]
                if np.any(block == bg):
                    break  # stop extending at first bg
                colors = set(int(block[r, c]) for r in range(2) for c in range(width))
                if len(colors) >= 3:
                    best_w = width
            if best_w >= 2:
                block = grid[r0:r0 + 2, c0:c0 + best_w]
                nc = len(set(int(block[r, c]) for r in range(2) for c in range(best_w)))
                candidates.append((nc * best_w, r0, c0, 2, best_w, nc))

    # Scan 2-col blocks
    for c0 in range(w - 1):
        for r0 in range(h):
            best_h = 0
            for height in range(2, min(12, h - r0 + 1)):
                block = grid[r0:r0 + height, c0:c0 + 2]
                if np.any(block == bg):
                    break
                colors = set(int(block[r, c]) for r in range(height) for c in range(2))
                if len(colors) >= 3:
                    best_h = height
            if best_h >= 2:
                block = grid[r0:r0 + best_h, c0:c0 + 2]
                nc = len(set(int(block[r, c]) for r in range(best_h) for c in range(2)))
                candidates.append((nc * best_h, r0, c0, best_h, 2, nc))

    if not candidates:
        return []

    # Return the single best candidate (highest score)
    candidates.sort(reverse=True)
    best = candidates[0]
    score, r0, c0, rows, cols, nc = best
    return [RoleAssignment(
        entity=EntityRef('strip', 0),
        role=Role.LEGEND,
        confidence=0.7,
        reason=f"dense {rows}x{cols} block at ({r0},{c0}) with {nc} colors",
    )]


# ---------------------------------------------------------------------------
# Cross-demo consistency
# ---------------------------------------------------------------------------

def check_panel_consistency(
    demos: list[tuple[np.ndarray, np.ndarray]],
    perceive_fn,
) -> dict:
    """Check if panel structure is consistent across demos.

    Returns dict with:
    - 'consistent': bool
    - 'n_panels': int (if consistent)
    - 'panel_shapes': list of shapes per panel slot
    - 'separator_axes': consistent separator structure
    """
    all_sep_info = []

    for inp, _ in demos:
        facts = perceive_fn(inp)
        col_seps = sorted([s for s in facts.separators if s.axis == 'col'],
                           key=lambda s: s.index)
        row_seps = sorted([s for s in facts.separators if s.axis == 'row'],
                           key=lambda s: s.index)
        all_sep_info.append({
            'n_col_seps': len(col_seps),
            'n_row_seps': len(row_seps),
        })

    if not all_sep_info:
        return {'consistent': False}

    # Check consistency
    n_col = all_sep_info[0]['n_col_seps']
    n_row = all_sep_info[0]['n_row_seps']

    consistent = all(
        info['n_col_seps'] == n_col and info['n_row_seps'] == n_row
        for info in all_sep_info
    )

    return {
        'consistent': consistent,
        'n_col_seps': n_col,
        'n_row_seps': n_row,
        'n_panels': n_col + 1 if n_col > 0 else n_row + 1 if n_row > 0 else 1,
    }


# ---------------------------------------------------------------------------
# Panel extraction helper
# ---------------------------------------------------------------------------

def extract_panels_from_separators(
    grid: np.ndarray,
    separators: list,
) -> list[np.ndarray]:
    """Extract panels by slicing at separator positions.

    Returns list of panel subgrids, in order.
    """
    h, w = grid.shape
    col_seps = sorted([s.index for s in separators if s.axis == 'col'])
    row_seps = sorted([s.index for s in separators if s.axis == 'row'])

    if col_seps:
        boundaries = [0] + col_seps + [w]
        panels = []
        for i in range(len(boundaries) - 1):
            c0 = boundaries[i]
            c1 = boundaries[i + 1]
            if i > 0:
                c0 += 1  # skip separator column
            if c1 > c0:
                panels.append(grid[:, c0:c1])
        return panels

    if row_seps:
        boundaries = [0] + row_seps + [h]
        panels = []
        for i in range(len(boundaries) - 1):
            r0 = boundaries[i]
            r1 = boundaries[i + 1]
            if i > 0:
                r0 += 1
            if r1 > r0:
                panels.append(grid[r0:r1, :])
        return panels

    return [grid]


# ---------------------------------------------------------------------------
# Binding derivation entry point
# ---------------------------------------------------------------------------

def derive_scene_binding(
    demos: list[tuple[np.ndarray, np.ndarray]],
) -> SceneBinding | None:
    """Derive a cross-demo consistent scene binding.

    Attempts to assign roles to perceived entities and express
    relations between them. Returns None if no consistent binding found.

    Works for both separator-based and non-separator scenes.
    """
    from aria.guided.perceive import perceive

    if not demos:
        return None

    inp0, out0 = demos[0]
    facts0 = perceive(inp0)

    binding = SceneBinding()

    # --- Strategy 1: Separator-based panels ---
    panels = extract_panels_from_separators(inp0, facts0.separators)
    if len(panels) >= 2:
        panel_roles = classify_panel_roles(panels, facts0.bg, out0.shape)
        binding.roles.extend(panel_roles)
        for i, p in enumerate(panels):
            binding.entity_grids[EntityRef('panel', i)] = p

    # --- Strategy 2: Legend strip detection (works with or without separators) ---
    legend_cands = find_legend_candidates(inp0, facts0.bg, facts0.objects, facts0.separators)
    binding.roles.extend(legend_cands)

    # --- Strategy 3: Object-based entities (for non-separator tasks) ---
    if len(panels) < 2 and facts0.objects:
        # Group objects by color
        from collections import defaultdict
        color_groups = defaultdict(list)
        for obj in facts0.objects:
            color_groups[obj.color].append(obj)

        # Each color group is a potential entity
        for i, (color, objs) in enumerate(sorted(color_groups.items(),
                                                    key=lambda x: -len(x[1]))):
            ref = EntityRef('object_group', i)
            # Large groups → WORKSPACE, small groups → CONTROL/ANCHOR
            total_size = sum(o.size for o in objs)
            if total_size > inp0.size * 0.1:
                binding.roles.append(RoleAssignment(
                    entity=ref, role=Role.WORKSPACE, confidence=0.3,
                    reason=f"color {color}: {len(objs)} objects, {total_size} cells",
                ))
            elif len(objs) <= 3 and total_size < inp0.size * 0.05:
                binding.roles.append(RoleAssignment(
                    entity=ref, role=Role.CONTROL, confidence=0.3,
                    reason=f"color {color}: small group ({total_size} cells)",
                ))

    # --- Build relations ---
    legends = binding.entities_with_role(Role.LEGEND)
    workspaces = binding.entities_with_role(Role.WORKSPACE)
    queries = binding.entities_with_role(Role.QUERY)
    targets = queries or workspaces

    for leg in legends:
        for tgt in targets:
            binding.relations.append(EntityRelation(
                source=leg, target=tgt, relation=Relation.MAPS_TO,
            ))

    # Only return binding if we found at least some structure
    if not binding.roles:
        return None

    return binding
