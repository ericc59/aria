"""Object correspondence: map output objects to input sources.

This is THE correspondence engine for the active clause-based pipeline.
All other modules that need output→input object mapping should import from here.

For each output object, finds the best matching input object and classifies
the relationship (identical, recolored, moved, moved_recolored, new).
Uses ObjFact from perceive.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.guided.perceive import GridFacts, ObjFact


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ObjMapping:
    """How one output object relates to its input source."""
    out_obj: ObjFact
    in_obj: ObjFact | None
    match_type: str       # "identical", "recolored", "moved", "moved_recolored", "transformed", "modified", "new"
    color_from: int       # input color (-1 if new)
    color_to: int         # output color
    transform: str = None # for "transformed": which transform (flip_h, flip_v, rot90, etc.)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def map_output_to_input(
    out_facts: GridFacts,
    in_facts: GridFacts,
) -> list[ObjMapping]:
    """Map each output object to its best input match.

    Uses global (Hungarian) assignment when feasible, falls back to greedy.
    """
    n_out = len(out_facts.objects)
    n_in = len(in_facts.objects)

    # Use global assignment for reasonable sizes
    if 1 <= n_out <= 50 and 1 <= n_in <= 50:
        result = _map_global(out_facts, in_facts)
        if result is not None:
            return result

    # Fallback: greedy
    return _map_greedy(out_facts.objects, in_facts.objects)


def _map_greedy(out_objs, in_objs):
    """Greedy correspondence: process output objects largest-first."""
    mappings = []
    used = set()
    for out_obj in out_objs:
        best, mtype, xform = _best_match(out_obj, in_objs, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj,
            in_obj=best,
            match_type=mtype,
            color_from=best.color if best else -1,
            color_to=out_obj.color,
            transform=xform,
        ))
    return mappings


def _map_global(out_facts, in_facts):
    """Global (Hungarian) assignment minimizing a cost matrix.

    Cost encodes: shape mismatch, color mismatch, position distance.
    Lower cost = better match. Unmatched outputs become 'new'.
    """
    from scipy.optimize import linear_sum_assignment

    out_objs = out_facts.objects
    in_objs = in_facts.objects
    n_out = len(out_objs)
    n_in = len(in_objs)

    # Build cost matrix: n_out rows × (n_in + n_out) cols
    # Extra n_out columns are "dummy" (no match = new), with high cost
    DUMMY_COST = 1000.0
    n_cols = n_in + n_out
    cost = np.full((n_out, n_cols), DUMMY_COST, dtype=np.float64)

    for i, out_obj in enumerate(out_objs):
        for j, in_obj in enumerate(in_objs):
            cost[i, j] = _match_cost(out_obj, in_obj)

    row_idx, col_idx = linear_sum_assignment(cost)

    mappings = [None] * n_out
    used_in = set()
    for r, c in zip(row_idx, col_idx):
        out_obj = out_objs[r]
        if c < n_in:
            in_obj = in_objs[c]
            used_in.add(in_obj.oid)
            mtype, xform = _classify_match(out_obj, in_obj)
            mappings[r] = ObjMapping(
                out_obj=out_obj, in_obj=in_obj,
                match_type=mtype,
                color_from=in_obj.color, color_to=out_obj.color,
                transform=xform,
            )
        else:
            mappings[r] = ObjMapping(
                out_obj=out_obj, in_obj=None,
                match_type="new",
                color_from=-1, color_to=out_obj.color,
            )

    return mappings


def _match_cost(out_obj, in_obj):
    """Cost of matching out_obj to in_obj. Lower = better."""
    same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
    same_shape = (in_obj.height == out_obj.height and
                  in_obj.width == out_obj.width and
                  np.array_equal(in_obj.mask, out_obj.mask))
    same_color = (in_obj.color == out_obj.color)

    # Perfect match
    if same_pos and same_shape and same_color:
        return 0.0
    if same_pos and same_shape:
        return 1.0   # recolored
    if same_shape and same_color:
        return 2.0   # moved
    if same_shape:
        return 3.0   # moved + recolored

    # Transform match
    if same_pos and same_color and out_obj.size >= 4:
        xform = _is_transform_of(in_obj.mask, out_obj.mask)
        if xform:
            return 4.0

    # Near-shape match: same color, similar size, bbox overlap or containment.
    # Catches objects that grew/shrank slightly during movement.
    if same_color and in_obj.size >= 2 and out_obj.size >= 2:
        size_ratio = min(in_obj.size, out_obj.size) / max(in_obj.size, out_obj.size)
        if size_ratio >= 0.5:
            iou = _masks_overlap_iou(in_obj, out_obj)
            if iou > 0.3:
                return 5.0 + (1.0 - iou) * 5  # 5.0-8.5 range
            # No overlap but same color + similar size → candidate move
            pos_dist = abs(in_obj.center_row - out_obj.center_row) + \
                       abs(in_obj.center_col - out_obj.center_col)
            if pos_dist < max(in_obj.height + out_obj.height,
                              in_obj.width + out_obj.width) * 2:
                return 15.0 + pos_dist * 0.5

    # Feature-match tier: different color but similar shape/size.
    # Mask IoU >= 0.5 OR perimeter similarity, size ratio >= 0.5.
    # Catches recolored objects that also moved.
    if not same_color and in_obj.size >= 2 and out_obj.size >= 2:
        size_ratio = min(in_obj.size, out_obj.size) / max(in_obj.size, out_obj.size)
        if size_ratio >= 0.5:
            iou = _masks_overlap_iou(in_obj, out_obj)
            if iou >= 0.5:
                return 20.0 + (1.0 - iou) * 5
            # No overlap — check mask shape similarity (perimeter ratio)
            perim_a = _mask_perimeter(in_obj)
            perim_b = _mask_perimeter(out_obj)
            if perim_a > 0 and perim_b > 0:
                perim_ratio = min(perim_a, perim_b) / max(perim_a, perim_b)
                if perim_ratio >= 0.5:
                    pos_dist = abs(in_obj.center_row - out_obj.center_row) + \
                               abs(in_obj.center_col - out_obj.center_col)
                    return 25.0 + pos_dist * 0.3

    # Modified (same color, overlapping)
    if same_color:
        iou = _masks_overlap_iou(in_obj, out_obj)
        if iou > 0.3:
            return 10.0 - iou * 5  # 5.0-8.5 range

    # Poor match
    pos_dist = abs(in_obj.row - out_obj.row) + abs(in_obj.col - out_obj.col)
    size_diff = abs(in_obj.size - out_obj.size)
    color_pen = 0 if same_color else 50
    return 100.0 + pos_dist + size_diff + color_pen


def _classify_match(out_obj, in_obj):
    """Classify the match type and transform for an assigned pair."""
    same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
    same_shape = (in_obj.height == out_obj.height and
                  in_obj.width == out_obj.width and
                  np.array_equal(in_obj.mask, out_obj.mask))
    same_color = (in_obj.color == out_obj.color)

    if same_pos and same_shape and same_color:
        return "identical", None
    if same_pos and same_shape:
        return "recolored", None
    if same_shape and same_color:
        return "moved", None
    if same_shape:
        return "moved_recolored", None

    if same_pos and same_color and out_obj.size >= 4:
        xform = _is_transform_of(in_obj.mask, out_obj.mask)
        if xform:
            return "transformed", xform

    # Near-shape movement: same color, similar size, not same position.
    # Object may have grown/shrunk slightly during placement.
    if same_color and not same_pos and in_obj.size >= 2 and out_obj.size >= 2:
        size_ratio = min(in_obj.size, out_obj.size) / max(in_obj.size, out_obj.size)
        if size_ratio >= 0.5:
            return "moved", None

    # Feature-match: different color, similar shape/size → moved_recolored
    if not same_color and in_obj.size >= 2 and out_obj.size >= 2:
        size_ratio = min(in_obj.size, out_obj.size) / max(in_obj.size, out_obj.size)
        if size_ratio >= 0.5:
            iou = _masks_overlap_iou(in_obj, out_obj)
            if iou >= 0.5:
                return "moved_recolored", None
            perim_a = _mask_perimeter(in_obj)
            perim_b = _mask_perimeter(out_obj)
            if perim_a > 0 and perim_b > 0:
                perim_ratio = min(perim_a, perim_b) / max(perim_a, perim_b)
                if perim_ratio >= 0.5:
                    return "moved_recolored", None

    if same_color:
        iou = _masks_overlap_iou(in_obj, out_obj)
        if iou > 0.3:
            return "modified", None

    return "moved_recolored", None  # fallback for assigned but poor matches


def map_output_to_input_topk(
    out_facts: GridFacts,
    in_facts: GridFacts,
    k: int = 3,
) -> list[list[ObjMapping]]:
    """Generate K correspondence hypotheses using different strategies.

    Returns up to K distinct mapping lists. The first is always the default
    (largest-first greedy). Others use different tie-breaking or ordering.
    """
    hypotheses = []
    seen = set()

    # Strategy 1: default (largest first)
    m1 = map_output_to_input(out_facts, in_facts)
    key1 = _mapping_key(m1)
    hypotheses.append(m1)
    seen.add(key1)

    # Strategy 2: smallest output objects first
    if len(out_facts.objects) > 1:
        m2 = _map_with_order(out_facts, in_facts, reverse=True)
        key2 = _mapping_key(m2)
        if key2 not in seen:
            hypotheses.append(m2)
            seen.add(key2)

    # Strategy 3: prioritize same-color matches over same-shape
    if len(hypotheses) < k:
        m3 = _map_color_priority(out_facts, in_facts)
        key3 = _mapping_key(m3)
        if key3 not in seen:
            hypotheses.append(m3)
            seen.add(key3)

    return hypotheses[:k]


def _mapping_key(mappings):
    """Hashable key for a correspondence (for dedup)."""
    return tuple(
        (m.out_obj.oid, m.in_obj.oid if m.in_obj else -1, m.match_type)
        for m in mappings
    )


def _map_with_order(out_facts, in_facts, reverse=False):
    """Map with output objects in size order (reverse=True → smallest first)."""
    ordered = sorted(out_facts.objects, key=lambda o: o.size, reverse=not reverse)
    mappings = []
    used = set()
    for out_obj in ordered:
        best, mtype, xform = _best_match(out_obj, in_facts.objects, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj, in_obj=best, match_type=mtype,
            color_from=best.color if best else -1, color_to=out_obj.color,
            transform=xform,
        ))
    return mappings


def _map_color_priority(out_facts, in_facts):
    """Map prioritizing color match over shape match."""
    mappings = []
    used = set()
    for out_obj in out_facts.objects:
        best, mtype, xform = _best_match_color_first(out_obj, in_facts.objects, used)
        if best:
            used.add(best.oid)
        mappings.append(ObjMapping(
            out_obj=out_obj, in_obj=best, match_type=mtype,
            color_from=best.color if best else -1, color_to=out_obj.color,
            transform=xform,
        ))
    return mappings


def _best_match_color_first(out_obj, in_objs, used):
    """Like _best_match but prioritizes color match over position match."""
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
        same_shape = (in_obj.height == out_obj.height and
                      in_obj.width == out_obj.width and
                      np.array_equal(in_obj.mask, out_obj.mask))
        same_color = (in_obj.color == out_obj.color)
        if same_pos and same_shape and same_color:
            return in_obj, "identical", None
        if same_shape and same_color:
            return in_obj, "moved", None
        if same_pos and same_shape:
            return in_obj, "recolored", None
        if same_shape:
            return in_obj, "moved_recolored", None
    # Fallback: modified (same color, overlapping)
    best_modified = None
    best_iou = 0.0
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        if in_obj.color != out_obj.color:
            continue
        iou = _masks_overlap_iou(in_obj, out_obj)
        if iou > 0.3 and iou > best_iou:
            best_modified = in_obj
            best_iou = iou
    if best_modified:
        return best_modified, "modified", None
    return None, "new", None


def find_removed_objects(
    in_facts: GridFacts,
    mappings: list[ObjMapping],
) -> list[ObjFact]:
    """Find input objects that have no corresponding output object."""
    matched_oids = {m.in_obj.oid for m in mappings if m.in_obj is not None}
    return [o for o in in_facts.objects if o.oid not in matched_oids]


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _best_match(
    out_obj: ObjFact,
    in_objs: list[ObjFact],
    used: set[int],
) -> tuple[ObjFact | None, str, str | None]:
    """Find the best input match for one output object.

    Returns (matched_obj, match_type, transform_name).
    transform_name is set only for "transformed" matches.

    Match priority (strictest first):
    1. identical        — same pos + shape + color
    2. recolored        — same pos + shape, different color
    3. moved            — same shape + color, different pos
    4. moved_recolored  — same shape, different pos + color
    5. transformed      — geometric transform of mask, same/nearby pos
    6. modified         — same color + overlapping position, shape changed
    """
    # --- Tier 1-4: strict shape matching (existing) ---
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
        same_shape = (in_obj.height == out_obj.height and
                      in_obj.width == out_obj.width and
                      np.array_equal(in_obj.mask, out_obj.mask))
        same_color = (in_obj.color == out_obj.color)

        if same_pos and same_shape and same_color:
            return in_obj, "identical", None
        if same_pos and same_shape:
            return in_obj, "recolored", None
        if same_shape and same_color:
            return in_obj, "moved", None
        if same_shape:
            return in_obj, "moved_recolored", None

    # --- Tier 5: transform match (flip/rot of mask at same position, same color) ---
    if out_obj.size >= 4:
        for in_obj in in_objs:
            if in_obj.oid in used:
                continue
            same_pos = (in_obj.row == out_obj.row and in_obj.col == out_obj.col)
            same_color = (in_obj.color == out_obj.color)
            if not same_pos or not same_color:
                continue
            xform = _is_transform_of(in_obj.mask, out_obj.mask)
            if xform:
                return in_obj, "transformed", xform

    # --- Tier 6: modified match (same color, overlapping bbox) ---
    best_modified = None
    best_iou = 0.0
    for in_obj in in_objs:
        if in_obj.oid in used:
            continue
        if in_obj.color != out_obj.color:
            continue
        iou = _masks_overlap_iou(in_obj, out_obj)
        if iou > 0.3 and iou > best_iou:
            best_modified = in_obj
            best_iou = iou

    if best_modified:
        return best_modified, "modified", None

    return None, "new", None


def _is_transform_of(mask_a, mask_b):
    """Check if mask_b is a geometric transform of mask_a.

    Returns transform name or None. Only checks when dimensions match.
    """
    for xform_name, xfn in [('flip_h', lambda m: m[:, ::-1]),
                              ('flip_v', lambda m: m[::-1, :]),
                              ('flip_hv', lambda m: m[::-1, ::-1])]:
        if mask_a.shape == mask_b.shape and np.array_equal(xfn(mask_a), mask_b):
            return xform_name
    # Rotations: 90/270 swap dimensions
    if mask_a.shape[0] == mask_b.shape[1] and mask_a.shape[1] == mask_b.shape[0]:
        for xform_name, xfn in [('rot90', lambda m: np.rot90(m)),
                                  ('rot270', lambda m: np.rot90(m, 3))]:
            if np.array_equal(xfn(mask_a), mask_b):
                return xform_name
    if mask_a.shape == mask_b.shape:
        if np.array_equal(np.rot90(mask_a, 2), mask_b):
            return 'rot180'
    return None


def _masks_overlap_iou(a, b):
    """Compute IoU of two objects' masks in absolute grid coordinates."""
    # Bounding box overlap
    r0 = max(a.row, b.row)
    c0 = max(a.col, b.col)
    r1 = min(a.row + a.height, b.row + b.height)
    c1 = min(a.col + a.width, b.col + b.width)
    if r0 >= r1 or c0 >= c1:
        return 0.0

    # Count overlapping mask pixels
    intersection = 0
    for r in range(r0, r1):
        for c in range(c0, c1):
            a_on = a.mask[r - a.row, c - a.col] if (0 <= r - a.row < a.height and 0 <= c - a.col < a.width) else False
            b_on = b.mask[r - b.row, c - b.col] if (0 <= r - b.row < b.height and 0 <= c - b.col < b.width) else False
            if a_on and b_on:
                intersection += 1

    union = a.size + b.size - intersection
    return intersection / union if union > 0 else 0.0


def _mask_perimeter(obj):
    """Count perimeter pixels (mask cells with at least one non-mask neighbor)."""
    m = obj.mask
    h, w = m.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not m[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                count += 1
            elif not (m[r-1, c] and m[r+1, c] and m[r, c-1] and m[r, c+1]):
                count += 1
    return count
