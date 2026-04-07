"""Local rule induction from aligned input↔output substructure pairs.

Given an aligned (input_sub, output_sub) pair, induces a small local
rule that transforms input_sub → output_sub using root-level primitives.

Rules are represented as composable operations, not named families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.prototype.region_alignment import RegionAlignment
from aria.types import Grid


# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalRule:
    """A small compositional rule that transforms an input subgrid to an output subgrid."""
    op: str                    # operation name
    params: dict[str, Any]     # operation parameters
    confidence: float          # how well it explains the alignment
    description: str           # human-readable description

    def apply(self, input_sub: Grid, bg: int = 0) -> Grid:
        """Apply this rule to an input subgrid."""
        fn = _RULE_EXECUTORS.get(self.op)
        if fn is None:
            return input_sub.copy()
        return fn(input_sub, bg, self.params)


# ---------------------------------------------------------------------------
# Rule induction
# ---------------------------------------------------------------------------


def induce_local_rules(
    alignment: RegionAlignment,
    bg: int = 0,
) -> list[LocalRule]:
    """Induce candidate local rules from one aligned region pair.

    Returns a list of candidate rules, ranked by confidence.
    """
    inp = alignment.input_content
    out = alignment.output_content

    if inp.shape != out.shape:
        return []

    candidates: list[LocalRule] = []

    # Identity check
    if np.array_equal(inp, out):
        candidates.append(LocalRule("identity", {}, 1.0, "no change"))
        return candidates

    # Rule 1: Global color map
    rule = _try_color_map(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 2: Fill enclosed bg regions with a color
    rule = _try_fill_enclosed(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 2b: Per-object bbox fill (fill holes in each object's bbox)
    rule = _try_per_object_bbox_fill(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 3: Recolor specific positions (mask-based)
    rule = _try_mask_recolor(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 4: Copy from elsewhere (constant output)
    rule = _try_constant_output(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 5: Mirror/reflect then patch
    rule = _try_mirror_repair(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 8: Singleton-marker erasure + corresponding object rewrite
    rule = _try_singleton_marker_rewrite(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 6: Per-row/col periodic completion
    rule = _try_periodic_complete(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Rule 7: Gravity/shift non-bg pixels
    rule = _try_gravity(inp, out, bg)
    if rule is not None:
        candidates.append(rule)

    # Sort by confidence
    candidates.sort(key=lambda r: -r.confidence)
    return candidates


# ---------------------------------------------------------------------------
# Rule inductors
# ---------------------------------------------------------------------------


def _try_color_map(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output = input with some colors remapped."""
    diff = inp != out
    if not np.any(diff):
        return None

    cmap: dict[int, int] = {}
    for r, c in zip(*np.where(diff)):
        ic, oc = int(inp[r, c]), int(out[r, c])
        if ic in cmap and cmap[ic] != oc:
            return None  # inconsistent
        cmap[ic] = oc

    # Verify: applying the map reproduces the output
    result = inp.copy()
    for from_c, to_c in cmap.items():
        result[inp == from_c] = to_c
    if not np.array_equal(result, out):
        return None

    return LocalRule(
        "color_map",
        {"map": cmap},
        1.0,
        f"recolor {cmap}",
    )


def _try_fill_enclosed(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output fills enclosed bg regions with a specific color."""
    from scipy import ndimage

    diff = inp != out
    if not np.any(diff):
        return None

    # All changed pixels should be bg→something
    changed_from = set(int(inp[r, c]) for r, c in zip(*np.where(diff)))
    if changed_from != {bg}:
        return None

    changed_to = set(int(out[r, c]) for r, c in zip(*np.where(diff)))
    if len(changed_to) != 1:
        return None
    fill_color = next(iter(changed_to))

    # Check: are the changed positions an enclosed bg region?
    bg_mask = inp == bg
    labeled, n = ndimage.label(bg_mask)
    rows, cols = inp.shape
    border_labels = set()
    for r in range(rows):
        if labeled[r, 0] > 0: border_labels.add(labeled[r, 0])
        if labeled[r, cols-1] > 0: border_labels.add(labeled[r, cols-1])
    for c in range(cols):
        if labeled[0, c] > 0: border_labels.add(labeled[0, c])
        if labeled[rows-1, c] > 0: border_labels.add(labeled[rows-1, c])

    # Check if changed positions are exactly the enclosed bg regions
    result = inp.copy()
    for lbl in range(1, n + 1):
        if lbl not in border_labels:
            result[labeled == lbl] = fill_color

    if np.array_equal(result, out):
        return LocalRule(
            "fill_enclosed",
            {"fill_color": fill_color},
            1.0,
            f"fill enclosed bg with {fill_color}",
        )
    return None


def _try_per_object_bbox_fill(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output fills bg holes inside each object's bounding box.

    For each connected non-bg component, fill bg pixels within its bbox
    with the component's color. Verify against output.
    """
    from scipy import ndimage

    diff = inp != out
    if not np.any(diff):
        return None

    # All changes must be bg -> non-bg
    changed_from = set(int(inp[r, c]) for r, c in zip(*np.where(diff)))
    if changed_from != {bg}:
        return None

    # Find connected non-bg objects
    non_bg_mask = inp != bg
    labeled, n = ndimage.label(non_bg_mask)

    result = inp.copy()
    for lbl in range(1, n + 1):
        component = labeled == lbl
        rows_with, cols_with = np.where(component)
        if len(rows_with) == 0:
            continue
        r0, r1 = int(rows_with.min()), int(rows_with.max())
        c0, c1 = int(cols_with.min()), int(cols_with.max())

        # Object's dominant color
        obj_colors = inp[component]
        from collections import Counter
        color_counts = Counter(int(v) for v in obj_colors)
        dominant = color_counts.most_common(1)[0][0]

        # Fill bg within bbox
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(result[r, c]) == bg:
                    result[r, c] = dominant

    if np.array_equal(result, out):
        return LocalRule(
            "per_object_bbox_fill",
            {},
            0.95,
            "fill bg holes in each object bbox with object color",
        )
    return None


def _try_mask_recolor(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if a fixed set of positions get recolored.

    Represents the change as: at each (dr, dc) offset within the bbox,
    change color X to color Y. This captures spatial mask patterns.
    """
    diff = inp != out
    if not np.any(diff):
        return None

    n_diff = int(np.sum(diff))
    total = inp.size
    if n_diff > total // 2:
        return None  # too many changes for a mask

    # Collect the changes
    changes: list[tuple[int, int, int, int]] = []  # (dr, dc, from_color, to_color)
    for r, c in zip(*np.where(diff)):
        changes.append((int(r), int(c), int(inp[r, c]), int(out[r, c])))

    return LocalRule(
        "mask_recolor",
        {"changes": changes},
        0.8,  # lower confidence — this is a memorized mask, not a real rule
        f"recolor {n_diff} specific positions",
    )


def _try_constant_output(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output is independent of input (constant content)."""
    # This is only useful for diff-shape tasks where the output is a known pattern
    return LocalRule(
        "constant",
        {"content": out.copy()},
        0.3,  # very low confidence — this just memorizes the output
        "constant output (memorized)",
    )


def _try_mirror_repair(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output is input with mirror-symmetry violations repaired."""
    rows, cols = inp.shape
    diff = inp != out
    if not np.any(diff):
        return None

    # Try horizontal mirror
    h_mirror = np.fliplr(inp)
    h_repaired = inp.copy()
    for r in range(rows):
        for c in range(cols):
            if int(inp[r, c]) != int(h_mirror[r, c]):
                # Which one is "wrong"? The one that the output overrides
                if int(out[r, c]) == int(h_mirror[r, c]):
                    h_repaired[r, c] = int(h_mirror[r, c])

    if np.array_equal(h_repaired, out):
        return LocalRule("mirror_repair", {"axis": "horizontal"}, 0.95,
                         "repair horizontal mirror violations")

    # Try vertical mirror
    v_mirror = np.flipud(inp)
    v_repaired = inp.copy()
    for r in range(rows):
        for c in range(cols):
            if int(inp[r, c]) != int(v_mirror[r, c]):
                if int(out[r, c]) == int(v_mirror[r, c]):
                    v_repaired[r, c] = int(v_mirror[r, c])

    if np.array_equal(v_repaired, out):
        return LocalRule("mirror_repair", {"axis": "vertical"}, 0.95,
                         "repair vertical mirror violations")

    return None


def _try_periodic_complete(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if output completes periodic patterns, optionally within framed regions.

    Tries two modes:
    1. Whole-grid per-row/col periodic completion
    2. Per-framed-region periodic completion (strips borders before detecting period)
    """
    from aria.periodicity import detect_1d_period, complete_sequence

    rows, cols = inp.shape
    diff = inp != out
    if not np.any(diff):
        return None

    # Mode 1: whole-grid
    for axis in ("row", "col"):
        result = inp.copy()
        any_repaired = False
        if axis == "row":
            for r in range(rows):
                pat = detect_1d_period(inp[r], require_violations=True)
                if pat is not None and pat.confidence >= 0.7:
                    result[r] = complete_sequence(inp[r], pat)
                    any_repaired = True
        else:
            for c in range(cols):
                pat = detect_1d_period(inp[:, c], require_violations=True)
                if pat is not None and pat.confidence >= 0.7:
                    result[:, c] = complete_sequence(inp[:, c], pat)
                    any_repaired = True
        if any_repaired and np.array_equal(result, out):
            return LocalRule("periodic_complete", {"axis": axis, "mode": "whole_grid"},
                             0.95, f"complete {axis}-wise periodic patterns")

    # Mode 2: per-framed-region (border-aware)
    result = _try_framed_periodic_complete(inp, out, bg)
    if result is not None:
        return result

    return None


def _try_framed_periodic_complete(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Complete periodic patterns inside each framed region, respecting borders.

    For each framed region:
    1. Extract interior rows
    2. Strip the first/last cell if they match the inner-border color
    3. Detect period on the stripped content
    4. Repair violations in the content zone only
    """
    from aria.decomposition import detect_framed_regions
    from aria.periodicity import detect_1d_period, complete_sequence

    framed = detect_framed_regions(inp, bg)
    if not framed:
        return None

    result = inp.copy()
    any_repaired = False

    for fr in framed:
        interior = result[fr.row:fr.row + fr.height, fr.col:fr.col + fr.width]
        iH, iW = interior.shape

        for ri in range(iH):
            row = interior[ri].copy()
            n = len(row)
            if n < 4:
                continue

            # Detect inner border: check if first/last values are a uniform
            # "inner border" color distinct from the content
            border_color = None
            if int(row[0]) == int(row[-1]) and int(row[0]) != bg:
                # Check: is this color only at the edges of this row?
                edge_color = int(row[0])
                interior_vals = set(int(row[i]) for i in range(1, n - 1))
                if edge_color not in interior_vals or len(interior_vals) <= 2:
                    border_color = edge_color

            if border_color is not None:
                # Strip borders, detect period on content only
                content = row[1:-1]
                pat = detect_1d_period(content, require_violations=True)
                if pat is not None and pat.confidence >= 0.8 and len(pat.violations) <= max(1, len(content) // 4):
                    repaired = complete_sequence(content, pat)
                    if not np.array_equal(content, repaired):
                        interior[ri, 1:-1] = repaired
                        any_repaired = True
            else:
                pat = detect_1d_period(row, require_violations=True)
                if pat is not None and pat.confidence >= 0.8 and len(pat.violations) <= max(1, len(row) // 4):
                    repaired = complete_sequence(row, pat)
                    if not np.array_equal(row, repaired):
                        interior[ri] = repaired
                        any_repaired = True

    if any_repaired and np.array_equal(result, out):
        return LocalRule("periodic_complete", {"axis": "row", "mode": "framed_region"},
                         0.98, "complete periodic patterns inside framed regions")

    return None


def _try_singleton_marker_rewrite(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if singletons are erased and nearby objects are rewritten.

    Pattern: isolated single-pixel markers indicate a rewrite action on
    the nearest non-singleton object. The marker gets erased, and the
    object (or part of it) gets recolored to the marker's color.

    This is the backward-solver formulation of correspondence-based
    recoloring: we observe which singletons vanish and which object
    pixels change, then check if the marker color matches the new color.

    Compositional representation:
    - Selector: singletons (1-pixel non-bg connected components)
    - Context: nearest non-singleton object per singleton
    - Condition: singleton is erased in output
    - Action: recolor object pixels that changed to the marker's color
    """
    from aria.decomposition import decompose_objects
    from scipy import ndimage

    diff = inp != out
    if not np.any(diff):
        return None

    objs = decompose_objects(inp, bg)
    if not objs.singletons or not objs.non_singletons:
        return None

    # Check: are any singletons erased?
    erased_singletons = []
    for s in objs.singletons:
        if int(out[s.row, s.col]) == bg:
            erased_singletons.append(s)

    if not erased_singletons:
        return None

    # For each erased singleton, check if nearby object pixels changed
    # to the singleton's color
    result = inp.copy()

    for marker in erased_singletons:
        mc = marker.color
        # Erase the marker
        result[marker.row, marker.col] = bg

        # Find the nearest non-singleton object
        best_obj = None
        best_dist = float('inf')
        for obj in objs.non_singletons:
            dist = abs(marker.center_row - obj.center_row) + abs(marker.center_col - obj.center_col)
            if dist < best_dist:
                best_dist = dist
                best_obj = obj

        if best_obj is None:
            continue

        # Check: did pixels in this object's bbox change to the marker color?
        r0, r1 = best_obj.row, best_obj.row + best_obj.bbox_h
        c0, c1 = best_obj.col, best_obj.col + best_obj.bbox_w
        for r in range(r0, r1):
            for c in range(c0, c1):
                if int(out[r, c]) == mc and int(inp[r, c]) != mc:
                    result[r, c] = mc

    if np.array_equal(result, out):
        return LocalRule(
            "singleton_marker_rewrite",
            {},
            0.95,
            "erase singletons and recolor nearest object to marker color",
        )

    return None


def _try_gravity(inp: Grid, out: Grid, bg: int) -> LocalRule | None:
    """Check if non-bg pixels have been shifted/gravitied in one direction."""
    diff = inp != out
    if not np.any(diff):
        return None

    rows, cols = inp.shape

    # Try gravity down: for each column, non-bg pixels sink to bottom
    for direction in ["down", "up", "left", "right"]:
        result = _apply_gravity(inp, bg, direction)
        if np.array_equal(result, out):
            return LocalRule("gravity", {"direction": direction}, 0.95,
                             f"gravity {direction}")
    return None


def _apply_gravity(grid: Grid, bg: int, direction: str) -> Grid:
    """Shift non-bg pixels in one direction within each row/col."""
    result = np.full_like(grid, bg)
    rows, cols = grid.shape

    if direction == "down":
        for c in range(cols):
            non_bg = [int(grid[r, c]) for r in range(rows) if int(grid[r, c]) != bg]
            for i, v in enumerate(reversed(non_bg)):
                result[rows - 1 - i, c] = v
    elif direction == "up":
        for c in range(cols):
            non_bg = [int(grid[r, c]) for r in range(rows) if int(grid[r, c]) != bg]
            for i, v in enumerate(non_bg):
                result[i, c] = v
    elif direction == "right":
        for r in range(rows):
            non_bg = [int(grid[r, c]) for c in range(cols) if int(grid[r, c]) != bg]
            for i, v in enumerate(reversed(non_bg)):
                result[r, cols - 1 - i] = v
    elif direction == "left":
        for r in range(rows):
            non_bg = [int(grid[r, c]) for c in range(cols) if int(grid[r, c]) != bg]
            for i, v in enumerate(non_bg):
                result[r, i] = v
    return result


# ---------------------------------------------------------------------------
# Rule executors
# ---------------------------------------------------------------------------


def _exec_identity(inp: Grid, bg: int, params: dict) -> Grid:
    return inp.copy()


def _exec_color_map(inp: Grid, bg: int, params: dict) -> Grid:
    result = inp.copy()
    for from_c, to_c in params["map"].items():
        result[inp == from_c] = to_c
    return result


def _exec_fill_enclosed(inp: Grid, bg: int, params: dict) -> Grid:
    from scipy import ndimage
    fill_color = params["fill_color"]
    bg_mask = inp == bg
    labeled, n = ndimage.label(bg_mask)
    rows, cols = inp.shape
    border_labels = set()
    for r in range(rows):
        if labeled[r, 0] > 0: border_labels.add(labeled[r, 0])
        if labeled[r, cols-1] > 0: border_labels.add(labeled[r, cols-1])
    for c in range(cols):
        if labeled[0, c] > 0: border_labels.add(labeled[0, c])
        if labeled[rows-1, c] > 0: border_labels.add(labeled[rows-1, c])
    result = inp.copy()
    for lbl in range(1, n + 1):
        if lbl not in border_labels:
            result[labeled == lbl] = fill_color
    return result


def _exec_mask_recolor(inp: Grid, bg: int, params: dict) -> Grid:
    result = inp.copy()
    for dr, dc, from_c, to_c in params["changes"]:
        if 0 <= dr < result.shape[0] and 0 <= dc < result.shape[1]:
            if int(result[dr, dc]) == from_c:
                result[dr, dc] = to_c
    return result


def _exec_mirror_repair(inp: Grid, bg: int, params: dict) -> Grid:
    axis = params["axis"]
    if axis == "horizontal":
        mirror = np.fliplr(inp)
    else:
        mirror = np.flipud(inp)
    result = inp.copy()
    rows, cols = inp.shape
    for r in range(rows):
        for c in range(cols):
            if int(inp[r, c]) != int(mirror[r, c]):
                # Use the majority value between original and mirror
                # (repairs the minority position)
                result[r, c] = int(mirror[r, c])
    # This is wrong for asymmetric cases — need to use output-directed repair
    # For now, fall back to direct application
    return result


def _exec_periodic_complete(inp: Grid, bg: int, params: dict) -> Grid:
    from aria.periodicity import detect_1d_period, complete_sequence
    mode = params.get("mode", "whole_grid")
    axis = params.get("axis", "row")

    if mode == "framed_region":
        return _exec_framed_periodic_complete(inp, bg)

    result = inp.copy()
    rows, cols = inp.shape
    if axis == "row":
        for r in range(rows):
            pat = detect_1d_period(inp[r], require_violations=True)
            if pat is not None and pat.confidence >= 0.7:
                result[r] = complete_sequence(inp[r], pat)
    else:
        for c in range(cols):
            pat = detect_1d_period(inp[:, c], require_violations=True)
            if pat is not None and pat.confidence >= 0.7:
                result[:, c] = complete_sequence(inp[:, c], pat)
    return result


def _exec_framed_periodic_complete(inp: Grid, bg: int) -> Grid:
    """Execute framed-region periodic completion."""
    from aria.decomposition import detect_framed_regions
    from aria.periodicity import detect_1d_period, complete_sequence

    framed = detect_framed_regions(inp, bg)
    result = inp.copy()

    for fr in framed:
        interior = result[fr.row:fr.row + fr.height, fr.col:fr.col + fr.width]
        iH, iW = interior.shape

        for ri in range(iH):
            row = interior[ri].copy()
            n = len(row)
            if n < 4:
                continue

            border_color = None
            if int(row[0]) == int(row[-1]) and int(row[0]) != bg:
                edge_color = int(row[0])
                interior_vals = set(int(row[i]) for i in range(1, n - 1))
                if edge_color not in interior_vals or len(interior_vals) <= 2:
                    border_color = edge_color

            if border_color is not None:
                content = row[1:-1]
                pat = detect_1d_period(content, require_violations=True)
                if pat is not None and pat.confidence >= 0.8 and len(pat.violations) <= max(1, len(content) // 4):
                    interior[ri, 1:-1] = complete_sequence(content, pat)
            else:
                pat = detect_1d_period(row, require_violations=True)
                if pat is not None and pat.confidence >= 0.7:
                    interior[ri] = complete_sequence(row, pat)

    return result


def _exec_gravity(inp: Grid, bg: int, params: dict) -> Grid:
    return _apply_gravity(inp, bg, params["direction"])


def _exec_constant(inp: Grid, bg: int, params: dict) -> Grid:
    return params["content"].copy()


def _exec_per_object_bbox_fill(inp: Grid, bg: int, params: dict) -> Grid:
    from scipy import ndimage
    from collections import Counter
    non_bg_mask = inp != bg
    labeled, n = ndimage.label(non_bg_mask)
    result = inp.copy()
    for lbl in range(1, n + 1):
        component = labeled == lbl
        rows_with, cols_with = np.where(component)
        if len(rows_with) == 0:
            continue
        r0, r1 = int(rows_with.min()), int(rows_with.max())
        c0, c1 = int(cols_with.min()), int(cols_with.max())
        obj_colors = inp[component]
        dominant = Counter(int(v) for v in obj_colors).most_common(1)[0][0]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(result[r, c]) == bg:
                    result[r, c] = dominant
    return result


def _exec_singleton_marker_rewrite(inp: Grid, bg: int, params: dict) -> Grid:
    """Erase singletons and recolor nearest object to marker color."""
    from aria.decomposition import decompose_objects

    objs = decompose_objects(inp, bg)
    result = inp.copy()

    for marker in objs.singletons:
        if marker.color == bg:
            continue
        mc = marker.color

        # Erase marker
        result[marker.row, marker.col] = bg

        # Find nearest non-singleton object
        best_obj = None
        best_dist = float('inf')
        for obj in objs.non_singletons:
            dist = abs(marker.center_row - obj.center_row) + abs(marker.center_col - obj.center_col)
            if dist < best_dist:
                best_dist = dist
                best_obj = obj

        if best_obj is None:
            continue

        # Recolor object pixels that are not the object's dominant color
        # to the marker's color. This handles "the singleton indicates
        # which color to rewrite the anomalous cells to."
        from collections import Counter
        obj_pixels = []
        for dr in range(best_obj.bbox_h):
            for dc in range(best_obj.bbox_w):
                if best_obj.mask[dr, dc]:
                    obj_pixels.append(int(inp[best_obj.row + dr, best_obj.col + dc]))
        dominant = Counter(obj_pixels).most_common(1)[0][0]

        for dr in range(best_obj.bbox_h):
            for dc in range(best_obj.bbox_w):
                if best_obj.mask[dr, dc]:
                    r, c = best_obj.row + dr, best_obj.col + dc
                    if int(inp[r, c]) != dominant and int(inp[r, c]) != bg:
                        result[r, c] = mc

    return result


_RULE_EXECUTORS = {
    "identity": _exec_identity,
    "color_map": _exec_color_map,
    "fill_enclosed": _exec_fill_enclosed,
    "per_object_bbox_fill": _exec_per_object_bbox_fill,
    "singleton_marker_rewrite": _exec_singleton_marker_rewrite,
    "mask_recolor": _exec_mask_recolor,
    "mirror_repair": _exec_mirror_repair,
    "periodic_complete": _exec_periodic_complete,
    "gravity": _exec_gravity,
    "constant": _exec_constant,
}
