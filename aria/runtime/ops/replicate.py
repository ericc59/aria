"""Template replication from anchor-bearing exemplars to compatible targets.

Mechanism:
  1. Extract multi-pixel exemplar shapes and single-pixel singletons
  2. For each exemplar, find its anchor: the adjacent singleton whose
     color differs from the exemplar's color
  3. The anchor's color is the key
  4. Find all other singletons with the same key color — these are targets
  5. Clone the exemplar to each target, preserving the anchor's relative
     offset inside the exemplar
  6. Paint clones onto a clean grid (optionally erasing sources)

Parameters:
  key_rule (int):
    0 = adjacent_diff_color — anchor is adjacent singleton with different color
    1 = adjacent_any        — anchor is nearest adjacent singleton regardless of color

  source_policy (int):
    0 = erase_sources — source exemplars are not painted in the output
    1 = keep_sources  — source exemplars are preserved in the output

  placement_rule (int):
    0 = anchor_offset — clone placed so anchor offset from exemplar top-left is preserved
    1 = center_on_target — clone center aligns with target position

All parameters are explicit integers visible in programs and specialization.
"""

from __future__ import annotations

import numpy as np

from aria.runtime.ops import OpSignature, register
from aria.runtime.ops.selection import _find_objects
from aria.types import Grid, ObjectNode, Type


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_ADJACENT_DIFF_COLOR = 0
KEY_ADJACENT_ANY = 1

KEY_NAMES = {
    KEY_ADJACENT_DIFF_COLOR: "adjacent_diff_color",
    KEY_ADJACENT_ANY: "adjacent_any",
}
ALL_KEY_RULES = (KEY_ADJACENT_DIFF_COLOR, KEY_ADJACENT_ANY)

SOURCE_ERASE = 0
SOURCE_KEEP = 1

SOURCE_NAMES = {SOURCE_ERASE: "erase_sources", SOURCE_KEEP: "keep_sources"}
ALL_SOURCE_POLICIES = (SOURCE_ERASE, SOURCE_KEEP)

PLACE_ANCHOR_OFFSET = 0
PLACE_CENTER = 1

PLACE_NAMES = {PLACE_ANCHOR_OFFSET: "anchor_offset", PLACE_CENTER: "center_on_target"}
ALL_PLACE_RULES = (PLACE_ANCHOR_OFFSET, PLACE_CENTER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_bg(grid: Grid) -> int:
    unique, counts = np.unique(grid, return_counts=True)
    return int(unique[np.argmax(counts)])


def _is_adjacent(sx: int, sy: int, sw: int, sh: int, mx: int, my: int) -> bool:
    """Check if point (mx, my) [col, row] is adjacent to or inside bbox."""
    return sx - 1 <= mx <= sx + sw and sy - 1 <= my <= sy + sh


def _find_anchor(
    shape: ObjectNode, singletons: list[ObjectNode], key_rule: int,
) -> tuple[ObjectNode | None, tuple[int, int]]:
    """Find the anchor singleton for an exemplar shape.

    Returns (anchor_object, (row_offset, col_offset)) or (None, (0,0)).
    Offset is marker_row - shape_top_row, marker_col - shape_left_col.
    """
    sx, sy, sw, sh = shape.bbox  # x=col, y=row
    for m in singletons:
        mx, my = m.bbox[0], m.bbox[1]
        if not _is_adjacent(sx, sy, sw, sh, mx, my):
            continue
        if key_rule == KEY_ADJACENT_DIFF_COLOR and m.color == shape.color:
            continue
        offset = (my - sy, mx - sx)
        return m, offset
    return None, (0, 0)


# ---------------------------------------------------------------------------
# Core: replicate_templates
# ---------------------------------------------------------------------------


def _replicate_templates(
    grid: Grid,
    key_rule: int = KEY_ADJACENT_DIFF_COLOR,
    source_policy: int = SOURCE_ERASE,
    placement_rule: int = PLACE_ANCHOR_OFFSET,
) -> Grid:
    """Template replication: clone exemplars to compatible singleton targets.

    1. Find exemplar shapes and singletons
    2. For each exemplar, derive anchor + key color
    3. Find targets with matching key color
    4. Clone exemplar to each target, preserving anchor offset
    5. Paint onto clean grid
    """
    if grid.size == 0:
        return grid.copy()

    bg = _detect_bg(grid)
    all_objs = list(_find_objects(grid))
    shapes = [o for o in all_objs if o.color != bg and o.size > 1]
    singletons = [o for o in all_objs if o.color != bg and o.size == 1]

    if not shapes or not singletons:
        return grid.copy()

    result = np.full_like(grid, bg)
    used_singletons: set[int] = set()  # indices of singletons used as anchors

    # Phase 1: find exemplar-anchor pairs and collect clone instructions
    # An exemplar is the full bounding region around shape + anchor (may be multi-color)
    clones: list[tuple[np.ndarray, int, int, int]] = []  # (region_grid, bg, target_row, target_col)
    anchored_shapes: set[int] = set()
    clone_target_indices: set[int] = set()

    for si_idx, shape in enumerate(shapes):
        anchor, offset = _find_anchor(shape, singletons, key_rule)
        if anchor is None:
            continue
        anchored_shapes.add(si_idx)

        key_color = anchor.color
        anchor_idx = next(j for j, m in enumerate(singletons) if m is anchor)
        used_singletons.add(anchor_idx)

        # Extract the full exemplar region: all non-bg pixels reachable
        # from the shape/anchor via non-bg adjacency (8-connected)
        sx, sy, sw, sh = shape.bbox  # col, row, w, h
        ax, ay = anchor.bbox[0], anchor.bbox[1]

        # Flood-fill from shape and anchor pixels to find full exemplar
        visited = set()
        stack = []
        # Seed from shape mask pixels
        mask = shape.mask
        for dr in range(mask.shape[0]):
            for dc in range(mask.shape[1]):
                if mask[dr, dc]:
                    stack.append((sy + dr, sx + dc))
        # Seed from anchor
        stack.append((ay, ax))

        rows_g, cols_g = grid.shape
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= rows_g or c < 0 or c >= cols_g:
                continue
            if int(grid[r, c]) == bg:
                continue
            visited.add((r, c))
            # Expand to 8-connected neighbors
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r + dr, c + dc))

        if not visited:
            continue

        # Compute bounding box of the exemplar
        min_row = min(r for r, c in visited)
        max_row = max(r for r, c in visited) + 1
        min_col = min(c for r, c in visited)
        max_col = max(c for r, c in visited) + 1
        region = grid[min_row:max_row, min_col:max_col].copy()
        anchor_row_in_region = ay - min_row
        anchor_col_in_region = ax - min_col

        # Find all targets with the same key color
        for j, target in enumerate(singletons):
            if j == anchor_idx:
                continue
            if target.color != key_color:
                continue

            clone_target_indices.add(j)
            tx, ty = target.bbox[0], target.bbox[1]

            if placement_rule == PLACE_ANCHOR_OFFSET:
                clone_row = ty - anchor_row_in_region
                clone_col = tx - anchor_col_in_region
            else:  # PLACE_CENTER
                rh, rw = region.shape
                clone_row = ty - rh // 2
                clone_col = tx - rw // 2

            clones.append((region, bg, clone_row, clone_col))

    # Phase 2: paint only target singletons (not anchors, not unused)
    for j, m in enumerate(singletons):
        if j not in clone_target_indices:
            continue
        mx, my = m.bbox[0], m.bbox[1]
        if 0 <= my < result.shape[0] and 0 <= mx < result.shape[1]:
            result[my, mx] = m.color

    # Phase 3: paint source shapes if keeping
    if source_policy == SOURCE_KEEP:
        for si_idx, shape in enumerate(shapes):
            sx, sy = shape.bbox[0], shape.bbox[1]
            _paint_mask(result, shape, sy, sx)

    # Phase 4: paint clones (full regions, not just shape masks)
    for region, region_bg, cr, cc in clones:
        rh, rw = region.shape
        rows, cols = result.shape
        for dr in range(rh):
            for dc in range(rw):
                r, c = cr + dr, cc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    pixel = int(region[dr, dc])
                    if pixel != region_bg:
                        result[r, c] = pixel

    return result


def _paint_mask(result: Grid, shape: ObjectNode, row: int, col: int) -> None:
    """Paint a shape's mask onto result at (row, col)."""
    mask = shape.mask
    mh, mw = mask.shape
    rows, cols = result.shape
    for dr in range(mh):
        for dc in range(mw):
            if mask[dr, dc]:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    result[r, c] = shape.color


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "replicate_templates",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("key_rule", Type.INT),
            ("source_policy", Type.INT),
            ("placement_rule", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _replicate_templates,
)
