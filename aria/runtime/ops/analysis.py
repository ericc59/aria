"""Analysis operations: zones, counting, grouping, sorting."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from aria.types import (
    Grid,
    ObjectNode,
    Property,
    SortDir,
    Type,
    Zone,
    ZoneRole,
)
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Zone operations
# ---------------------------------------------------------------------------


def _find_zones(grid: Grid) -> list[Zone]:
    """Detect rectangular zones in a grid.

    Looks for lines of a single color that partition the grid into
    rectangular sub-regions. Falls back to returning the entire grid
    as a single zone if no separators are found.
    """
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return []

    # Try to detect horizontal and vertical separator lines.
    # A separator row is one where all cells share the same color.
    h_seps: list[int] = []
    for r in range(rows):
        if np.all(grid[r, :] == grid[r, 0]):
            h_seps.append(r)

    v_seps: list[int] = []
    for c in range(cols):
        if np.all(grid[:, c] == grid[0, c]):
            v_seps.append(c)

    # Filter to only separators that actually divide (same color as each other).
    if h_seps and len(h_seps) < rows:
        sep_color = int(grid[h_seps[0], 0])
        h_seps = [r for r in h_seps if int(grid[r, 0]) == sep_color]
    else:
        h_seps = []

    if v_seps and len(v_seps) < cols:
        sep_color_v = int(grid[0, v_seps[0]])
        v_seps = [c for c in v_seps if int(grid[0, c]) == sep_color_v]
    else:
        v_seps = []

    # Build row and col ranges between separators.
    def ranges_between(seps: list[int], total: int) -> list[tuple[int, int]]:
        if not seps:
            return [(0, total)]
        result: list[tuple[int, int]] = []
        prev = 0
        for s in sorted(seps):
            if s > prev:
                result.append((prev, s))
            prev = s + 1
        if prev < total:
            result.append((prev, total))
        return result

    row_ranges = ranges_between(h_seps, rows)
    col_ranges = ranges_between(v_seps, cols)

    zones: list[Zone] = []
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            sub = grid[r_start:r_end, c_start:c_end].copy()
            zones.append(Zone(
                grid=sub,
                x=c_start,
                y=r_start,
                w=c_end - c_start,
                h=r_end - r_start,
            ))

    if not zones:
        zones.append(Zone(grid=grid.copy(), x=0, y=0, w=cols, h=rows))

    return zones


def _zone_by_role(zones: list[Zone], role: ZoneRole) -> Zone:
    """Select a zone by role.

    Heuristic: RULE = smallest zone, DATA = largest zone,
    FRAME/BORDER = first zone. Proper role classification
    is left to the proposer layer.
    """
    if not zones:
        raise ValueError("zone_by_role: empty zone list")
    by_area = sorted(zones, key=lambda z: z.w * z.h)
    if role == ZoneRole.RULE:
        return by_area[0]
    if role == ZoneRole.DATA:
        return by_area[-1]
    return zones[0]


def _zone_to_grid(zone: Zone) -> Grid:
    """Extract the grid from a zone."""
    return zone.grid.copy()


def _extract_map(zone: Zone) -> dict[int, int]:
    """Extract a color mapping from a zone.

    Assumes the zone encodes a mapping as pairs of colored cells:
    left-column color -> right-column color.
    """
    grid = zone.grid
    rows, cols = grid.shape
    mapping: dict[int, int] = {}
    if cols >= 2:
        for r in range(rows):
            left = int(grid[r, 0])
            right = int(grid[r, cols - 1])
            if left != 0 or right != 0:
                mapping[left] = right
    return mapping


# ---------------------------------------------------------------------------
# Counting and aggregation
# ---------------------------------------------------------------------------


def _count(objects: set[ObjectNode]) -> int:
    """Count objects in a set."""
    return len(objects)


def _length(obj_list: list[Any]) -> int:
    """Return the length of a list."""
    return len(obj_list)


# ---------------------------------------------------------------------------
# Grouping and sorting
# ---------------------------------------------------------------------------


def _get_property(prop: Property, obj: ObjectNode) -> Any:
    """Extract a property value from an object."""
    if prop == Property.COLOR:
        return obj.color
    if prop == Property.SIZE:
        return obj.size
    if prop == Property.SHAPE:
        return obj.shape
    if prop == Property.POS_X:
        return obj.bbox[0]
    if prop == Property.POS_Y:
        return obj.bbox[1]
    if prop == Property.SYMMETRY:
        return obj.symmetry
    raise ValueError(f"Unknown property: {prop}")


def _group_by(prop: Property, objects: set[ObjectNode]) -> list[set[ObjectNode]]:
    """Group objects by a property value."""
    groups: dict[Any, set[ObjectNode]] = {}
    for obj in objects:
        key = _get_property(prop, obj)
        # frozenset is not hashable if used as key, convert
        if isinstance(key, frozenset):
            key = tuple(sorted(key))
        groups.setdefault(key, set()).add(obj)
    return list(groups.values())


def _sort_by(
    prop: Property, direction: SortDir, objects: set[ObjectNode]
) -> list[ObjectNode]:
    """Sort objects by a property value."""
    reverse = direction == SortDir.DESC
    return sorted(
        objects,
        key=lambda o: _get_property(prop, o),
        reverse=reverse,
    )


def _unique_colors(objects: set[ObjectNode]) -> list[int]:
    """Return the sorted list of unique colors across objects."""
    return sorted({obj.color for obj in objects})


def _max_val(ints: list[int]) -> int:
    """Return the maximum value in a list."""
    if not ints:
        raise ValueError("max_val: empty list")
    return max(ints)


def _min_val(ints: list[int]) -> int:
    """Return the minimum value in a list."""
    if not ints:
        raise ValueError("min_val: empty list")
    return min(ints)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "find_zones",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.ZONE_LIST),
    _find_zones,
)

register(
    "zone_by_role",
    OpSignature(
        params=(("zones", Type.ZONE_LIST), ("role", Type.ZONE_ROLE)),
        return_type=Type.ZONE,
    ),
    _zone_by_role,
)

register(
    "zone_to_grid",
    OpSignature(params=(("zone", Type.ZONE),), return_type=Type.GRID),
    _zone_to_grid,
)

register(
    "extract_map",
    OpSignature(params=(("zone", Type.ZONE),), return_type=Type.COLOR_MAP),
    _extract_map,
)

register(
    "count",
    OpSignature(params=(("objects", Type.OBJECT_SET),), return_type=Type.INT),
    _count,
)

register(
    "length",
    OpSignature(params=(("obj_list", Type.OBJECT_LIST),), return_type=Type.INT),
    _length,
)

register(
    "group_by",
    OpSignature(
        params=(("prop", Type.PROPERTY), ("objects", Type.OBJECT_SET)),
        return_type=Type.OBJECT_LIST,
    ),
    _group_by,
)

register(
    "sort_by",
    OpSignature(
        params=(
            ("prop", Type.PROPERTY),
            ("direction", Type.SORT_DIR),
            ("objects", Type.OBJECT_SET),
        ),
        return_type=Type.OBJECT_LIST,
    ),
    _sort_by,
)

register(
    "unique_colors",
    OpSignature(
        params=(("objects", Type.OBJECT_SET),),
        return_type=Type.INT_LIST,
    ),
    _unique_colors,
)

register(
    "max_val",
    OpSignature(params=(("ints", Type.INT_LIST),), return_type=Type.INT),
    _max_val,
)

register(
    "min_val",
    OpSignature(params=(("ints", Type.INT_LIST),), return_type=Type.INT),
    _min_val,
)
