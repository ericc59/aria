"""Backward explanation engine: find a primitive graph for each output unit.

For each output unit, searches for a small DAG of root primitives that,
when executed on the input, produces that exact output unit.

Search strategies:
1. Global single-primitive (color map, fill enclosed, reflection, periodic)
2. Per-object rule induction (the workhorse for ARC):
   - Track what happens to each input object in the output
   - Induce a predicate that selects affected objects
   - Induce the action (recolor, move, delete, fill-enclosed-with-own-color)
   - Unify predicate+action into one rule graph
3. Diff-shape explanations (crop, upscale, tile)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.ngs.ir import PrimitiveGraph, Leaf, PrimCall, VType
from aria.ngs.output_units import OutputUnit, UnitType
from aria.ngs.primitives import execute_prim
from aria.types import Grid


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class Explanation:
    """A primitive graph that explains one output unit."""
    __slots__ = ("graph", "output_unit", "confidence", "description")

    def __init__(self, graph: PrimitiveGraph, output_unit: OutputUnit,
                 confidence: float, description: str):
        self.graph = graph
        self.output_unit = output_unit
        self.confidence = confidence
        self.description = description

    def __repr__(self) -> str:
        return f"Explanation({self.description}, conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def explain_unit(
    inp: Grid,
    out: Grid,
    unit: OutputUnit,
    bg: int,
    max_depth: int = 2,
) -> list[Explanation]:
    """Search for primitive graphs that explain one output unit."""
    candidates: list[Explanation] = []

    if unit.unit_type == UnitType.WHOLE and inp.shape == out.shape:
        candidates.extend(_explain_same_shape_whole(inp, out, unit, bg))
    elif unit.unit_type == UnitType.DIFF_REGION:
        candidates.extend(_explain_diff_region(inp, out, unit, bg))
    elif unit.unit_type == UnitType.WHOLE:
        candidates.extend(_explain_diff_shape_whole(inp, out, unit, bg))

    candidates.sort(key=lambda e: (-e.confidence, e.graph.size()))
    return candidates


def explain_all_units(
    inp: Grid,
    out: Grid,
    units: list[OutputUnit],
    bg: int,
) -> list[list[Explanation]]:
    """Explain every output unit."""
    return [explain_unit(inp, out, u, bg) for u in units]


# ---------------------------------------------------------------------------
# Same-shape whole-grid explanations
# ---------------------------------------------------------------------------

def _explain_same_shape_whole(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
) -> list[Explanation]:
    results: list[Explanation] = []

    if np.array_equal(inp, out):
        g = PrimitiveGraph()
        i_inp = g.add_leaf("input", VType.GRID, inp)
        g.set_output(i_inp)
        results.append(Explanation(g, unit, 1.0, "identity"))
        return results

    # 1. Global color map
    cmap = execute_prim("compute_color_map", inp, out)
    if cmap is not None:
        predicted = execute_prim("render_color_map", inp, cmap)
        if np.array_equal(predicted, out):
            g = _make_color_map_graph(cmap)
            results.append(Explanation(g, unit, 0.95, f"color_map {cmap}"))

    # 2. Fill enclosed (all bg -> single color)
    for fill_c in range(10):
        try:
            predicted = execute_prim("render_fill_enclosed", inp, fill_c, bg)
            if np.array_equal(predicted, out):
                g = PrimitiveGraph()
                i_inp = g.add_leaf("input", VType.GRID, inp)
                i_fc = g.add_leaf("fill_color", VType.COLOR, fill_c)
                i_bg = g.add_leaf("bg", VType.COLOR, bg)
                i_out = g.add_prim("render_fill_enclosed", (i_inp, i_fc, i_bg), VType.GRID)
                g.set_output(i_out)
                results.append(Explanation(g, unit, 0.9, f"fill_enclosed color={fill_c}"))
        except Exception:
            pass

    # 3. Reflection / rotation
    for prim_name, desc in [
        ("render_reflect_h", "reflect_h"),
        ("render_reflect_v", "reflect_v"),
        ("render_rotate_90", "rotate_90"),
    ]:
        predicted = execute_prim(prim_name, inp)
        if np.array_equal(predicted, out):
            g = PrimitiveGraph()
            i_inp = g.add_leaf("input", VType.GRID, inp)
            i_out = g.add_prim(prim_name, (i_inp,), VType.GRID)
            g.set_output(i_out)
            results.append(Explanation(g, unit, 0.9, desc))

    # 4. Overlay reflected (mirror repair)
    for prim_name, desc in [
        ("render_reflect_h", "overlay_reflect_h"),
        ("render_reflect_v", "overlay_reflect_v"),
    ]:
        reflected = execute_prim(prim_name, inp)
        for order, suffix in [(True, ""), (False, "_rev")]:
            if order:
                predicted = execute_prim("render_overlay", reflected, inp, bg)
            else:
                predicted = execute_prim("render_overlay", inp, reflected, bg)
            if np.array_equal(predicted, out):
                g = PrimitiveGraph()
                i_inp = g.add_leaf("input", VType.GRID, inp)
                i_bg = g.add_leaf("bg", VType.COLOR, bg)
                i_ref = g.add_prim(prim_name, (i_inp,), VType.GRID)
                if order:
                    i_out = g.add_prim("render_overlay", (i_ref, i_inp, i_bg), VType.GRID)
                else:
                    i_out = g.add_prim("render_overlay", (i_inp, i_ref, i_bg), VType.GRID)
                g.set_output(i_out)
                results.append(Explanation(g, unit, 0.85, f"{desc}{suffix}"))

    # 5. Periodic line repair
    _try_periodic_line_repair(inp, out, unit, bg, results)

    # 6. Per-object rule induction (the key strategy)
    _try_per_object_rules(inp, out, unit, bg, results)

    # 7. Fill enclosed per-object (each object fills its own enclosed regions)
    _try_fill_enclosed_per_object(inp, out, unit, bg, results)

    return results


# ---------------------------------------------------------------------------
# Per-object rule induction
# ---------------------------------------------------------------------------

def _try_per_object_rules(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    """Induce rules by tracking what happens to each input object.

    For each input object, determine:
    - Does it survive unchanged?
    - Does it get recolored?
    - Does it get deleted (erased to bg)?
    - Does it move?
    - Does it get new pixels added?

    Then find a predicate that separates affected from unaffected objects,
    and encode the rule as: for_each(select(predicate), action).
    """
    objs_in = extract_objects(inp, bg, connectivity=4)
    if not objs_in:
        return

    # Classify each object's fate
    fates: list[dict[str, Any]] = []
    for obj in objs_in:
        fate = _classify_object_fate(obj, inp, out, bg)
        fates.append(fate)

    # Group objects by fate type
    fate_groups: dict[str, list[int]] = defaultdict(list)
    for i, fate in enumerate(fates):
        fate_groups[fate["type"]].append(i)

    # --- Strategy A: Some objects get deleted ---
    if "deleted" in fate_groups and "unchanged" in fate_groups:
        deleted_indices = fate_groups["deleted"]
        unchanged_indices = fate_groups["unchanged"]
        if len(deleted_indices) + len(unchanged_indices) == len(objs_in):
            # All objects are either deleted or unchanged
            pred = _find_separating_predicate(
                [objs_in[i] for i in deleted_indices],
                [objs_in[i] for i in unchanged_indices],
                objs_in, inp, bg,
            )
            if pred is not None:
                # Verify: delete objects matching predicate
                predicted = inp.copy()
                for i in deleted_indices:
                    obj = objs_in[i]
                    for r in range(obj.bbox_h):
                        for c in range(obj.bbox_w):
                            if obj.mask[r, c]:
                                predicted[obj.row + r, obj.col + c] = bg
                if np.array_equal(predicted, out):
                    g = _make_per_object_graph("delete", pred, {})
                    results.append(Explanation(g, unit, 0.85,
                        f"delete_objects where {pred}"))

    # --- Strategy B: Some objects get recolored ---
    if "recolored" in fate_groups:
        recolored_indices = fate_groups["recolored"]
        other_indices = [i for i in range(len(objs_in)) if i not in recolored_indices]

        # Check if all recolored objects get the same target color
        target_colors = set(fates[i]["to_color"] for i in recolored_indices)
        if len(target_colors) == 1:
            target_color = target_colors.pop()
            pred = _find_separating_predicate(
                [objs_in[i] for i in recolored_indices],
                [objs_in[i] for i in other_indices],
                objs_in, inp, bg,
            )
            if pred is not None:
                predicted = inp.copy()
                for i in recolored_indices:
                    obj = objs_in[i]
                    for r in range(obj.bbox_h):
                        for c in range(obj.bbox_w):
                            if obj.mask[r, c]:
                                predicted[obj.row + r, obj.col + c] = target_color
                if np.array_equal(predicted, out):
                    g = _make_per_object_graph("recolor", pred, {"color": target_color})
                    results.append(Explanation(g, unit, 0.85,
                        f"recolor_objects to {target_color} where {pred}"))

    # --- Strategy C: Objects swap positions ---
    if "moved" in fate_groups:
        moved_indices = fate_groups["moved"]
        if len(moved_indices) >= 2:
            _try_object_swap(objs_in, fates, moved_indices, inp, out, unit, bg, results)

    # --- Strategy D: Conditional recolor (color depends on object property) ---
    if "recolored" in fate_groups:
        recolored_indices = fate_groups["recolored"]
        _try_conditional_recolor(objs_in, fates, recolored_indices, inp, out, unit, bg, results)


def _classify_object_fate(
    obj: RawObject, inp: Grid, out: Grid, bg: int,
) -> dict[str, Any]:
    """Determine what happened to one input object in the output."""
    # Check pixels at the object's original position
    all_same = True
    all_bg = True
    recolor_target = None
    recolor_consistent = True

    for r in range(obj.bbox_h):
        for c in range(obj.bbox_w):
            if obj.mask[r, c]:
                gr, gc = obj.row + r, obj.col + c
                if gr >= out.shape[0] or gc >= out.shape[1]:
                    return {"type": "out_of_bounds"}
                out_val = int(out[gr, gc])
                in_val = int(inp[gr, gc])
                if out_val != in_val:
                    all_same = False
                if out_val != bg:
                    all_bg = False
                if out_val != in_val and out_val != bg:
                    if recolor_target is None:
                        recolor_target = out_val
                    elif recolor_target != out_val:
                        recolor_consistent = False

    if all_same:
        return {"type": "unchanged"}

    if all_bg:
        # Check if the object appeared elsewhere (moved)
        # Search for same-shape, same-mask object in output
        objs_out = extract_objects(out, bg, connectivity=4)
        for out_obj in objs_out:
            if (out_obj.mask.shape == obj.mask.shape
                    and np.array_equal(out_obj.mask, obj.mask)
                    and out_obj.color == obj.color
                    and (out_obj.row != obj.row or out_obj.col != obj.col)):
                return {
                    "type": "moved",
                    "to_row": out_obj.row,
                    "to_col": out_obj.col,
                    "dr": out_obj.row - obj.row,
                    "dc": out_obj.col - obj.col,
                }
        return {"type": "deleted"}

    if recolor_consistent and recolor_target is not None:
        return {"type": "recolored", "to_color": recolor_target}

    return {"type": "complex"}


def _find_separating_predicate(
    positive: list[RawObject],
    negative: list[RawObject],
    all_objs: list[RawObject],
    inp: Grid, bg: int,
) -> str | None:
    """Find a simple predicate that separates positive from negative objects.

    Returns a string description of the predicate, or None.
    """
    if not positive or not negative:
        return None

    # Predicate 1: by color
    pos_colors = set(o.color for o in positive)
    neg_colors = set(o.color for o in negative)
    if not pos_colors & neg_colors:
        return f"color_in({sorted(pos_colors)})"

    # Predicate 2: by size (singleton vs non-singleton)
    pos_singleton = all(o.is_singleton for o in positive)
    neg_singleton = all(o.is_singleton for o in negative)
    if pos_singleton and not neg_singleton:
        return "is_singleton"
    if not pos_singleton and neg_singleton:
        return "not_singleton"

    # Predicate 3: by size threshold
    pos_sizes = [o.size for o in positive]
    neg_sizes = [o.size for o in negative]
    max_pos_size = max(pos_sizes)
    min_neg_size = min(neg_sizes)
    if max_pos_size < min_neg_size:
        return f"size<={max_pos_size}"
    min_pos_size = min(pos_sizes)
    max_neg_size = max(neg_sizes)
    if min_pos_size > max_neg_size:
        return f"size>={min_pos_size}"

    # Predicate 4: by containment (positive objects are inside negative objects)
    # Check if every positive object is spatially inside some negative object
    pos_enclosed = True
    for po in positive:
        is_inside = False
        for no in negative:
            if (po.row >= no.row and po.col >= no.col
                    and po.row + po.bbox_h <= no.row + no.bbox_h
                    and po.col + po.bbox_w <= no.col + no.bbox_w):
                is_inside = True
                break
        if not is_inside:
            pos_enclosed = False
            break
    if pos_enclosed:
        return "enclosed_by_larger"

    # Predicate 5: has unique color (color appears exactly once in the grid)
    color_counts = Counter(o.color for o in all_objs)
    pos_unique = all(color_counts[o.color] == 1 for o in positive)
    neg_unique = all(color_counts[o.color] == 1 for o in negative)
    if pos_unique and not neg_unique:
        return "unique_color"

    # Predicate 6: touches border
    rows, cols = inp.shape
    pos_touches_border = all(
        o.row == 0 or o.col == 0 or
        o.row + o.bbox_h >= rows or o.col + o.bbox_w >= cols
        for o in positive
    )
    neg_touches_border = any(
        o.row == 0 or o.col == 0 or
        o.row + o.bbox_h >= rows or o.col + o.bbox_w >= cols
        for o in negative
    )
    if pos_touches_border and not neg_touches_border:
        return "touches_border"

    return None


def _try_object_swap(
    objs: list[RawObject], fates: list[dict], moved_indices: list[int],
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    """Try to explain object movements as systematic swaps or shifts."""
    # Check if moved objects share a common offset
    offsets = set()
    for i in moved_indices:
        dr = fates[i].get("dr", 0)
        dc = fates[i].get("dc", 0)
        offsets.add((dr, dc))

    if len(offsets) == 1:
        dr, dc = offsets.pop()
        # All moved objects shift by same amount
        predicted = inp.copy()
        for i in moved_indices:
            obj = objs[i]
            # Erase old position
            for r in range(obj.bbox_h):
                for c in range(obj.bbox_w):
                    if obj.mask[r, c]:
                        predicted[obj.row + r, obj.col + c] = bg
        for i in moved_indices:
            obj = objs[i]
            # Paint new position
            for r in range(obj.bbox_h):
                for c in range(obj.bbox_w):
                    if obj.mask[r, c]:
                        nr = obj.row + r + dr
                        nc = obj.col + c + dc
                        if 0 <= nr < predicted.shape[0] and 0 <= nc < predicted.shape[1]:
                            predicted[nr, nc] = obj.color
        if np.array_equal(predicted, out):
            g = _make_per_object_graph("move", f"moved_set", {"dr": dr, "dc": dc})
            results.append(Explanation(g, unit, 0.8,
                f"move_objects by ({dr},{dc})"))


def _try_conditional_recolor(
    objs: list[RawObject], fates: list[dict], recolored_indices: list[int],
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    """Try rules where recolor target depends on an object property.

    E.g., "recolor each object to the color of its nearest neighbor"
    or "recolor based on size rank."
    """
    # Check if the target color correlates with any simple property
    for i in recolored_indices:
        fate = fates[i]
        obj = objs[i]
        # Does the target color appear as a neighbor?
        to_color = fate.get("to_color")
        if to_color is None:
            return

    # For now, skip complex conditional recolors
    pass


def _try_fill_enclosed_per_object(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    """Fill enclosed bg regions within each object's bbox with that object's color."""
    objs = extract_objects(inp, bg, connectivity=4)
    if not objs:
        return

    predicted = inp.copy()
    for obj in objs:
        r0, c0 = obj.row, obj.col
        r1, c1 = r0 + obj.bbox_h, c0 + obj.bbox_w
        # Get the subgrid including the object
        sub = inp[r0:r1, c0:c1].copy()
        # Find enclosed bg cells in this subgrid relative to this object
        obj_mask = np.zeros_like(sub, dtype=bool)
        obj_mask[obj.mask] = True
        # Create a binary image: object pixels = wall, bg = open
        wall = sub != bg
        # Find bg cells not reachable from the sub-boundary
        h, w = sub.shape
        if h < 3 or w < 3:
            continue
        # Flood fill from edges
        reachable = np.zeros((h, w), dtype=bool)
        from collections import deque
        q = deque()
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and not wall[r, c]:
                    reachable[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not wall[nr, nc]:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        enclosed = (~wall) & (~reachable)
        if np.any(enclosed):
            for r in range(h):
                for c in range(w):
                    if enclosed[r, c]:
                        predicted[r0 + r, c0 + c] = obj.color

    if np.array_equal(predicted, out) and not np.array_equal(predicted, inp):
        g = PrimitiveGraph()
        i_inp = g.add_leaf("input", VType.GRID, inp)
        i_bg = g.add_leaf("bg", VType.COLOR, bg)
        i_out = g.add_prim("fill_enclosed_per_object", (i_inp, i_bg), VType.GRID)
        g.set_output(i_out)
        results.append(Explanation(g, unit, 0.88,
            "fill_enclosed_per_object"))


# ---------------------------------------------------------------------------
# Periodic line repair
# ---------------------------------------------------------------------------

def _try_periodic_line_repair(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    rows, cols = inp.shape
    if inp.shape != out.shape:
        return

    # Per-row
    repaired = inp.copy()
    ok = True
    for r in range(rows):
        if np.array_equal(inp[r, :], out[r, :]):
            continue
        period = execute_prim("detect_period", out[r, :])
        if period < cols:
            completed = execute_prim("render_periodic_complete", inp[r, :], period)
            if np.array_equal(completed, out[r, :]):
                repaired[r, :] = completed
                continue
        ok = False
        break
    if ok and np.array_equal(repaired, out):
        g = PrimitiveGraph()
        i_inp = g.add_leaf("input", VType.GRID, inp)
        i_out = g.add_prim("periodic_row_repair", (i_inp,), VType.GRID, {"axis": "row"})
        g.set_output(i_out)
        results.append(Explanation(g, unit, 0.85, "periodic_row_repair"))

    # Per-col
    repaired = inp.copy()
    ok = True
    for c in range(cols):
        if np.array_equal(inp[:, c], out[:, c]):
            continue
        period = execute_prim("detect_period", out[:, c])
        if period < rows:
            completed = execute_prim("render_periodic_complete", inp[:, c], period)
            if np.array_equal(completed, out[:, c]):
                repaired[:, c] = completed
                continue
        ok = False
        break
    if ok and np.array_equal(repaired, out):
        g = PrimitiveGraph()
        i_inp = g.add_leaf("input", VType.GRID, inp)
        i_out = g.add_prim("periodic_col_repair", (i_inp,), VType.GRID, {"axis": "col"})
        g.set_output(i_out)
        results.append(Explanation(g, unit, 0.85, "periodic_col_repair"))


# ---------------------------------------------------------------------------
# Diff-region explanations
# ---------------------------------------------------------------------------

def _explain_diff_region(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
) -> list[Explanation]:
    results: list[Explanation] = []
    assert unit.input_support is not None
    in_sub = unit.input_support
    out_sub = unit.content

    # 1. Local color map
    cmap = execute_prim("compute_color_map", in_sub, out_sub)
    if cmap is not None:
        predicted = execute_prim("render_color_map", in_sub, cmap)
        if np.array_equal(predicted, out_sub):
            g = PrimitiveGraph()
            i_inp = g.add_leaf("input", VType.GRID, inp)
            i_bbox = g.add_leaf("bbox", VType.BBOX, unit.bbox)
            i_sub = g.add_prim("extract_subgrid", (i_inp, i_bbox), VType.GRID)
            i_cmap = g.add_leaf("color_map", VType.COLOR_MAP, cmap)
            i_out = g.add_prim("render_color_map", (i_sub, i_cmap), VType.GRID)
            g.set_output(i_out)
            results.append(Explanation(g, unit, 0.9, f"local_color_map {cmap}"))

    # 2. Fill enclosed
    diff = in_sub != out_sub
    if np.any(diff):
        changed_vals = out_sub[diff]
        unique_changed = np.unique(changed_vals)
        if len(unique_changed) == 1:
            fill_c = int(unique_changed[0])
            enc = execute_prim("detect_enclosed", in_sub, bg)
            if np.any(enc) and np.array_equal(enc, diff):
                g = PrimitiveGraph()
                i_inp = g.add_leaf("input", VType.GRID, inp)
                i_bbox = g.add_leaf("bbox", VType.BBOX, unit.bbox)
                i_sub = g.add_prim("extract_subgrid", (i_inp, i_bbox), VType.GRID)
                i_fc = g.add_leaf("fill_color", VType.COLOR, fill_c)
                i_bg = g.add_leaf("bg", VType.COLOR, bg)
                i_out = g.add_prim("render_fill_enclosed", (i_sub, i_fc, i_bg), VType.GRID)
                g.set_output(i_out)
                results.append(Explanation(g, unit, 0.85, f"enclosed_fill={fill_c}"))

    # 3. Stamp from elsewhere
    _try_stamp_from_input(inp, out, unit, bg, results)

    return results


def _try_stamp_from_input(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
    results: list[Explanation],
) -> None:
    r0, c0, r1, c1 = unit.bbox
    target = out[r0:r1 + 1, c0:c1 + 1]
    th, tw = target.shape
    rows, cols = inp.shape

    for sr in range(rows - th + 1):
        for sc in range(cols - tw + 1):
            if np.array_equal(inp[sr:sr + th, sc:sc + tw], target):
                g = PrimitiveGraph()
                i_inp = g.add_leaf("input", VType.GRID, inp)
                i_src_bbox = g.add_leaf("src_bbox", VType.BBOX, (sr, sc, sr + th - 1, sc + tw - 1))
                i_stamp = g.add_prim("extract_subgrid", (i_inp, i_src_bbox), VType.GRID)
                g.set_output(i_stamp)
                results.append(Explanation(g, unit, 0.8,
                    f"stamp_from ({sr},{sc})->({r0},{c0})"))
                return


# ---------------------------------------------------------------------------
# Diff-shape whole explanations
# ---------------------------------------------------------------------------

def _explain_diff_shape_whole(
    inp: Grid, out: Grid, unit: OutputUnit, bg: int,
) -> list[Explanation]:
    results: list[Explanation] = []
    iH, iW = inp.shape
    oH, oW = out.shape

    # 1. Crop
    if oH <= iH and oW <= iW:
        for r in range(iH - oH + 1):
            for c in range(iW - oW + 1):
                if np.array_equal(inp[r:r + oH, c:c + oW], out):
                    g = PrimitiveGraph()
                    i_inp = g.add_leaf("input", VType.GRID, inp)
                    i_out = g.add_prim("render_crop", (i_inp,), VType.GRID,
                                       {"r0": r, "c0": c, "r1": r + oH - 1, "c1": c + oW - 1})
                    g.set_output(i_out)
                    results.append(Explanation(g, unit, 0.9, f"crop ({r},{c})"))
                    return results

    # 2. Upscale
    if oH > iH and oW > iW and oH % iH == 0 and oW % iW == 0:
        fr, fc = oH // iH, oW // iW
        if fr == fc:
            predicted = execute_prim("render_upscale", inp, fr)
            if np.array_equal(predicted, out):
                g = PrimitiveGraph()
                i_inp = g.add_leaf("input", VType.GRID, inp)
                i_f = g.add_leaf("factor", VType.INT, fr)
                i_out = g.add_prim("render_upscale", (i_inp, i_f), VType.GRID)
                g.set_output(i_out)
                results.append(Explanation(g, unit, 0.9, f"upscale {fr}x"))
                return results

    # 3. Tile
    if oH >= iH and oW >= iW and oH % iH == 0 and oW % iW == 0:
        tr, tc = oH // iH, oW // iW
        predicted = execute_prim("render_tile", inp, tr, tc)
        if np.array_equal(predicted, out):
            g = PrimitiveGraph()
            i_inp = g.add_leaf("input", VType.GRID, inp)
            i_tr = g.add_leaf("tile_rows", VType.INT, tr)
            i_tc = g.add_leaf("tile_cols", VType.INT, tc)
            i_out = g.add_prim("render_tile", (i_inp, i_tr, i_tc), VType.GRID)
            g.set_output(i_out)
            results.append(Explanation(g, unit, 0.9, f"tile {tr}x{tc}"))
            return results

    return results


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _make_color_map_graph(cmap: dict[int, int]) -> PrimitiveGraph:
    g = PrimitiveGraph()
    i_inp = g.add_leaf("input", VType.GRID, None)
    i_cmap = g.add_leaf("color_map", VType.COLOR_MAP, cmap)
    i_out = g.add_prim("render_color_map", (i_inp, i_cmap), VType.GRID)
    g.set_output(i_out)
    return g


def _make_per_object_graph(
    action: str, predicate: str, params: dict[str, Any],
) -> PrimitiveGraph:
    """Build a graph encoding a per-object rule.

    Structure: select_objects → filter(predicate) → for_each(action)
    """
    g = PrimitiveGraph()
    i_inp = g.add_leaf("input", VType.GRID, None)
    i_bg = g.add_leaf("bg", VType.COLOR, None)
    i_objs = g.add_prim("select_objects", (i_inp, i_bg), VType.OBJECTS)
    i_filtered = g.add_prim("filter_by_predicate", (i_objs,), VType.OBJECTS,
                            {"predicate": predicate})
    i_out = g.add_prim(f"for_each_{action}", (i_inp, i_filtered, i_bg), VType.GRID,
                       params)
    g.set_output(i_out)
    return g
