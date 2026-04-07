"""Entity graph: explicit relational structure over parsed scenes.

Extends the existing StateGraph extraction with:
- Entity role classification (host, marker, frame, gap, anchor)
- Containment / gap detection
- Cross-demo structural correspondence
- Query interface for slot-conditioned templates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.graph.extract import extract, extract_with_delta
from aria.graph.relations import compute_relations
from aria.types import (
    AlignRel,
    DemoPair,
    Grid,
    MatchRel,
    ObjectNode,
    RelationEdge,
    SpatialRel,
    StateGraph,
    TopoRel,
)


# ---------------------------------------------------------------------------
# Entity roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityRole:
    """Role assigned to an object in the entity graph."""
    object_id: int
    role: str  # "host", "marker", "frame", "gap", "anchor", "content"
    size: int
    color: int
    bbox: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Entity graph — enriched view of a single grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityGraph:
    """Explicit entity graph for one grid.

    Extends StateGraph with role assignments, containment edges,
    and gap detection.
    """
    state_graph: StateGraph
    roles: tuple[EntityRole, ...]
    containment: tuple[tuple[int, int], ...]  # (container_id, contained_id)
    gaps: tuple[GapEntity, ...]  # detected gaps (bg holes in hosts)
    marker_host_pairs: tuple[tuple[int, int], ...]  # (marker_id, host_id)


@dataclass(frozen=True)
class GapEntity:
    """A background gap inside a host object."""
    host_id: int
    gap_rows: tuple[int, ...]
    gap_cols: tuple[int, ...]
    gap_color: int  # expected fill color (from output, if known)
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_entity_graph(grid: Grid) -> EntityGraph:
    """Build an EntityGraph from a grid.

    1. Extract StateGraph (objects, relations)
    2. Classify roles (host, marker, frame)
    3. Detect containment
    4. Detect gaps
    """
    sg = extract(grid)
    bg = sg.context.bg_color

    # Classify roles
    roles = _classify_roles(sg, bg)

    # Detect containment
    containment = _detect_containment(sg)

    # Detect gaps (bg holes inside hosts)
    hosts = [r for r in roles if r.role == "host"]
    gaps = _detect_gaps(grid, hosts, bg)

    # Pair markers to nearest hosts
    markers = [r for r in roles if r.role == "marker"]
    pairs = _pair_markers_to_hosts(markers, hosts, sg)

    return EntityGraph(
        state_graph=sg,
        roles=tuple(roles),
        containment=tuple(containment),
        gaps=tuple(gaps),
        marker_host_pairs=tuple(pairs),
    )


def _classify_roles(sg: StateGraph, bg: int) -> list[EntityRole]:
    """Classify each object as host, marker, frame, or content."""
    roles: list[EntityRole] = []
    sizes = [o.size for o in sg.objects]
    if not sizes:
        return roles

    median_size = sorted(sizes)[len(sizes) // 2]

    for obj in sg.objects:
        if obj.color == bg:
            continue

        role = "content"  # default
        if obj.size == 1:
            role = "marker"
        elif obj.size <= 3 and obj.size < median_size / 2:
            role = "marker"
        elif _is_frame_shape(obj):
            role = "frame"
        elif obj.size >= median_size:
            role = "host"

        roles.append(EntityRole(
            object_id=obj.id,
            role=role,
            size=obj.size,
            color=obj.color,
            bbox=obj.bbox,
        ))
    return roles


def _is_frame_shape(obj: ObjectNode) -> bool:
    """Check if object has a frame-like shape (hollow rectangle)."""
    if obj.mask is None:
        return False
    h, w = obj.mask.shape
    if h < 3 or w < 3:
        return False
    interior = obj.mask[1:-1, 1:-1]
    # Frame if border is full but interior has holes
    border_count = obj.size
    interior_filled = int(np.sum(interior))
    total_interior = (h - 2) * (w - 2)
    if total_interior == 0:
        return False
    # Frame-like if interior is mostly empty
    return interior_filled < total_interior * 0.5 and border_count > interior_filled


def _detect_containment(sg: StateGraph) -> list[tuple[int, int]]:
    """Find containment relationships: which objects are inside which."""
    containment: list[tuple[int, int]] = []
    for edge in sg.relations:
        if TopoRel.CONTAINS in edge.topo:
            containment.append((edge.src, edge.dst))
    return containment


def _detect_gaps(
    grid: Grid, hosts: list[EntityRole], bg: int,
) -> list[GapEntity]:
    """Find background gaps inside host objects."""
    from scipy import ndimage

    gaps: list[GapEntity] = []
    for host in hosts:
        x, y, w, h = host.bbox
        r0, c0 = y, x
        r1, c1 = y + h, x + w
        # Clamp to grid bounds
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(grid.shape[0], r1)
        c1 = min(grid.shape[1], c1)

        region = grid[r0:r1, c0:c1]
        bg_mask = region == bg
        if not np.any(bg_mask):
            continue

        # Label connected bg regions within the bbox
        labeled, n = ndimage.label(bg_mask)
        rh, rw = region.shape

        # Find enclosed bg regions (not touching bbox border)
        border_labels = set()
        for r in range(rh):
            if labeled[r, 0] > 0:
                border_labels.add(labeled[r, 0])
            if labeled[r, rw - 1] > 0:
                border_labels.add(labeled[r, rw - 1])
        for c in range(rw):
            if labeled[0, c] > 0:
                border_labels.add(labeled[0, c])
            if labeled[rh - 1, c] > 0:
                border_labels.add(labeled[rh - 1, c])

        for lbl in range(1, n + 1):
            if lbl in border_labels:
                continue
            # This is an enclosed gap
            gap_mask = labeled == lbl
            gap_positions = np.argwhere(gap_mask)
            gap_rows = tuple(sorted(set(int(r0 + p[0]) for p in gap_positions)))
            gap_cols = tuple(sorted(set(int(c0 + p[1]) for p in gap_positions)))
            gr0 = int(r0 + gap_positions[:, 0].min())
            gc0 = int(c0 + gap_positions[:, 1].min())
            gr1 = int(r0 + gap_positions[:, 0].max())
            gc1 = int(c0 + gap_positions[:, 1].max())

            gaps.append(GapEntity(
                host_id=host.object_id,
                gap_rows=gap_rows,
                gap_cols=gap_cols,
                gap_color=bg,
                bbox=(gr0, gc0, gr1, gc1),
            ))

    return gaps


def _pair_markers_to_hosts(
    markers: list[EntityRole],
    hosts: list[EntityRole],
    sg: StateGraph,
) -> list[tuple[int, int]]:
    """Pair each marker to its nearest host."""
    if not markers or not hosts:
        return []

    obj_map = {o.id: o for o in sg.objects}
    pairs: list[tuple[int, int]] = []
    used_hosts: set[int] = set()

    for marker in markers:
        m_obj = obj_map.get(marker.object_id)
        if m_obj is None:
            continue
        best_host = None
        best_dist = float("inf")
        for host in hosts:
            h_obj = obj_map.get(host.object_id)
            if h_obj is None:
                continue
            # L1 distance between centers
            mc = (m_obj.bbox[1] + m_obj.bbox[3] / 2, m_obj.bbox[0] + m_obj.bbox[2] / 2)
            hc = (h_obj.bbox[1] + h_obj.bbox[3] / 2, h_obj.bbox[0] + h_obj.bbox[2] / 2)
            d = abs(mc[0] - hc[0]) + abs(mc[1] - hc[1])
            if d < best_dist:
                best_dist = d
                best_host = host.object_id
        if best_host is not None:
            pairs.append((marker.object_id, best_host))

    return pairs


# ---------------------------------------------------------------------------
# Cross-demo structural correspondence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoGraphSet:
    """Entity graphs for all demos in a task, with cross-demo consistency."""
    graphs: tuple[EntityGraph, ...]
    consistent_roles: tuple[str, ...]  # roles present in ALL demos
    n_markers_consistent: bool  # same number of markers across demos
    n_hosts_consistent: bool
    n_gaps_consistent: bool


def build_demo_graphs(demos: tuple[DemoPair, ...]) -> DemoGraphSet:
    """Build entity graphs for all demo inputs and check consistency."""
    graphs = tuple(build_entity_graph(d.input) for d in demos)

    # Check role consistency
    role_sets = [set(r.role for r in g.roles) for g in graphs]
    if role_sets:
        consistent = role_sets[0]
        for rs in role_sets[1:]:
            consistent &= rs
    else:
        consistent = set()

    # Check count consistency
    marker_counts = [sum(1 for r in g.roles if r.role == "marker") for g in graphs]
    host_counts = [sum(1 for r in g.roles if r.role == "host") for g in graphs]
    gap_counts = [len(g.gaps) for g in graphs]

    return DemoGraphSet(
        graphs=graphs,
        consistent_roles=tuple(sorted(consistent)),
        n_markers_consistent=len(set(marker_counts)) <= 1,
        n_hosts_consistent=len(set(host_counts)) <= 1,
        n_gaps_consistent=len(set(gap_counts)) <= 1,
    )


# ---------------------------------------------------------------------------
# Graph queries for slot filling
# ---------------------------------------------------------------------------


def query_objects_by_role(graph: EntityGraph, role: str) -> list[ObjectNode]:
    """Get all objects with a given role."""
    obj_map = {o.id: o for o in graph.state_graph.objects}
    return [obj_map[r.object_id] for r in graph.roles
            if r.role == role and r.object_id in obj_map]


def query_pairs_by_relation(
    graph: EntityGraph,
    relation: str,
) -> list[tuple[ObjectNode, ObjectNode]]:
    """Get object pairs connected by a specific relation type."""
    obj_map = {o.id: o for o in graph.state_graph.objects}
    pairs: list[tuple[ObjectNode, ObjectNode]] = []

    for edge in graph.state_graph.relations:
        match = False
        if relation == "contains" and TopoRel.CONTAINS in edge.topo:
            match = True
        elif relation == "adjacent" and TopoRel.ADJACENT in edge.topo:
            match = True
        elif relation == "same_color" and MatchRel.SAME_COLOR in edge.match:
            match = True
        elif relation == "same_shape" and MatchRel.SAME_SHAPE in edge.match:
            match = True
        elif relation == "same_size" and MatchRel.SAME_SIZE in edge.match:
            match = True
        elif relation == "aligned_h" and AlignRel.ALIGN_H in edge.align:
            match = True
        elif relation == "aligned_v" and AlignRel.ALIGN_V in edge.align:
            match = True
        elif relation == "nearest":
            match = True  # all edges are nearest candidates

        if match:
            src = obj_map.get(edge.src)
            dst = obj_map.get(edge.dst)
            if src is not None and dst is not None:
                pairs.append((src, dst))

    return pairs


def query_marker_host_pairs(graph: EntityGraph) -> list[tuple[ObjectNode, ObjectNode]]:
    """Get (marker, host) pairs from the entity graph."""
    obj_map = {o.id: o for o in graph.state_graph.objects}
    pairs = []
    for m_id, h_id in graph.marker_host_pairs:
        m = obj_map.get(m_id)
        h = obj_map.get(h_id)
        if m is not None and h is not None:
            pairs.append((m, h))
    return pairs
