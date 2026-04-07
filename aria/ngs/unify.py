"""Cross-demo rule unification for the next-generation solver.

Takes per-demo local explanation graphs and unifies them into one
shared abstract rule. This is the central inference step.

Strategy:
1. Canonicalize each demo's explanation graph
2. Group explanations by canonical structure
3. For each candidate group, check if bindings differ only in
   data-dependent leaves (input grid, colors) not in structure
4. Verify the abstract rule on all demos exactly
5. Return the best unified rule

Rejects demo-specific memorization by requiring structural agreement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from aria.decomposition import detect_bg
from aria.ngs.ir import (
    PrimitiveGraph, Leaf, PrimCall, VType,
    AbstractRule, DemoBinding, UnifiedRule,
)
from aria.ngs.backward_explain import Explanation
from aria.ngs.output_units import OutputUnit, UnitType
from aria.ngs.primitives import execute_prim
from aria.types import Grid, DemoPair


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def unify_across_demos(
    per_demo_explanations: list[list[Explanation]],
    demos: tuple[DemoPair, ...],
) -> UnifiedRule | None:
    """Find one abstract rule that explains all demos.

    per_demo_explanations[i] is the list of candidate explanations for demo i.
    Returns the best UnifiedRule or None if no unification succeeds.
    """
    if not per_demo_explanations:
        return None

    n_demos = len(demos)
    if n_demos == 0:
        return None

    # Each demo may have multiple units. For Stage 1, handle the case where
    # each demo produces exactly one WHOLE-unit explanation.
    # Group explanations by canonical key and try to unify.

    # Build all possible combinations: pick one explanation per demo
    # For efficiency, limit to top-k per demo
    MAX_PER_DEMO = 5
    trimmed = [exps[:MAX_PER_DEMO] if len(exps) > MAX_PER_DEMO else exps
               for exps in per_demo_explanations]

    # Filter to non-empty
    if any(len(exps) == 0 for exps in trimmed):
        return None

    # Group by canonical key
    key_to_explanations: dict[tuple, list[tuple[int, Explanation]]] = defaultdict(list)
    for di, exps in enumerate(trimmed):
        for exp in exps:
            key = exp.graph.canon_key()
            key_to_explanations[key].append((di, exp))

    # Find keys that appear in ALL demos
    best_rule: UnifiedRule | None = None
    best_score = -1.0

    for key, demo_exps in key_to_explanations.items():
        demo_indices = set(di for di, _ in demo_exps)
        if len(demo_indices) < n_demos:
            continue

        # Pick one explanation per demo (the first one)
        per_demo: dict[int, Explanation] = {}
        for di, exp in demo_exps:
            if di not in per_demo:
                per_demo[di] = exp

        if len(per_demo) < n_demos:
            continue

        # Try to build an abstract rule
        rule = _build_abstract_rule(per_demo, demos)
        if rule is not None and rule.train_verified:
            score = rule.rule.graph.size()  # prefer simpler rules
            inv_score = -score  # lower size = better
            if best_rule is None or inv_score > best_score:
                best_rule = rule
                best_score = inv_score

    return best_rule


def unify_multi_unit(
    per_demo_per_unit_explanations: list[list[list[Explanation]]],
    demos: tuple[DemoPair, ...],
) -> UnifiedRule | None:
    """Unify when each demo has multiple output units.

    per_demo_per_unit_explanations[demo_idx][unit_idx] = list[Explanation]

    For now, fall back to requiring same number of units per demo,
    and unify each unit independently, then compose.
    """
    if not per_demo_per_unit_explanations:
        return None

    n_demos = len(demos)
    n_units_per_demo = [len(units) for units in per_demo_per_unit_explanations]

    # All demos must have same number of units
    if len(set(n_units_per_demo)) != 1:
        return None

    n_units = n_units_per_demo[0]
    if n_units == 0:
        return None

    # If single unit, delegate to unify_across_demos
    if n_units == 1:
        flat = [exps[0] for exps in per_demo_per_unit_explanations]
        return unify_across_demos(flat, demos)

    # Multiple units: unify each independently
    unified_per_unit: list[UnifiedRule | None] = []
    for ui in range(n_units):
        per_demo_for_unit = [exps[ui] for exps in per_demo_per_unit_explanations]
        unified = unify_across_demos(per_demo_for_unit, demos)
        unified_per_unit.append(unified)

    # If all units unified, compose them
    if all(u is not None and u.train_verified for u in unified_per_unit):
        # Return the first unit's rule as a proxy (composition is TODO)
        return unified_per_unit[0]

    return None


# ---------------------------------------------------------------------------
# Abstract rule construction
# ---------------------------------------------------------------------------

def _build_abstract_rule(
    per_demo: dict[int, Explanation],
    demos: tuple[DemoPair, ...],
) -> UnifiedRule | None:
    """Build and verify an abstract rule from per-demo explanations.

    All explanations must have the same canonical structure.
    """
    # Use demo 0's graph as the template
    demo_0_exp = per_demo[0]
    template = demo_0_exp.graph

    # Extract leaf bindings per demo
    bindings: list[DemoBinding] = []
    for di in sorted(per_demo.keys()):
        exp = per_demo[di]
        leaf_vals: dict[str, Any] = {}
        for node in exp.graph.nodes:
            if isinstance(node, Leaf):
                leaf_vals[node.name] = node.value
        bindings.append(DemoBinding(demo_idx=di, leaf_values=leaf_vals))

    abstract_rule = AbstractRule(
        graph=template,
        description=demo_0_exp.description,
    )

    # Verify: execute the abstract rule on each demo's input
    # by substituting that demo's leaf bindings
    train_diff = 0
    all_verified = True

    for di, demo in enumerate(demos):
        bg = detect_bg(demo.input)
        predicted = _execute_abstract_rule(abstract_rule, demo.input, bg, di, demos)
        if predicted is None or not np.array_equal(predicted, demo.output):
            all_verified = False
            if predicted is not None and predicted.shape == demo.output.shape:
                train_diff += int(np.sum(predicted != demo.output))
            else:
                train_diff += demo.output.size
        # else: exact match

    return UnifiedRule(
        rule=abstract_rule,
        bindings=bindings,
        train_verified=all_verified,
        train_diff=train_diff,
    )


def _execute_abstract_rule(
    rule: AbstractRule,
    inp: Grid,
    bg: int,
    demo_idx: int,
    demos: tuple[DemoPair, ...],
) -> Grid | None:
    """Execute an abstract rule on a specific input.

    Substitutes the input grid and bg color into the graph's leaves,
    then evaluates the graph.
    """
    graph = rule.graph
    values: dict[int, Any] = {}

    for i, node in enumerate(graph.nodes):
        if isinstance(node, Leaf):
            if node.name == "input" or node.name == "input_copy":
                values[i] = inp
            elif node.name == "bg":
                values[i] = bg
            elif node.vtype == VType.COLOR_MAP:
                # Color maps are structural, keep them
                values[i] = node.value
            elif node.vtype == VType.COLOR:
                values[i] = node.value
            elif node.vtype == VType.INT:
                values[i] = node.value
            elif node.vtype == VType.BBOX:
                values[i] = node.value
            else:
                values[i] = node.value
        else:
            # PrimCall: execute with resolved args
            args = []
            for arg_idx in node.args:
                if arg_idx not in values:
                    return None
                args.append(values[arg_idx])

            try:
                # Composite prims not in the atomic registry
                if node.prim in _COMPOSITE_EXECUTORS:
                    result = _COMPOSITE_EXECUTORS[node.prim](args, node.params)
                else:
                    result = execute_prim(node.prim, *args, **node.params)
                values[i] = result
            except Exception:
                return None

    out_idx = graph.output_idx
    if out_idx < 0 or out_idx not in values:
        return None

    result = values[out_idx]
    if isinstance(result, np.ndarray):
        return result
    return None


def _exec_periodic_row_repair(args: list, params: dict) -> Grid:
    grid = args[0]
    out = grid.copy()
    for r in range(grid.shape[0]):
        row = grid[r, :]
        period = execute_prim("detect_period", row)
        if period < len(row):
            out[r, :] = execute_prim("render_periodic_complete", row, period)
    return out


def _exec_periodic_col_repair(args: list, params: dict) -> Grid:
    grid = args[0]
    out = grid.copy()
    for c in range(grid.shape[1]):
        col = grid[:, c]
        period = execute_prim("detect_period", col)
        if period < len(col):
            out[:, c] = execute_prim("render_periodic_complete", col, period)
    return out


def _exec_fill_enclosed_per_object(args: list, params: dict) -> Grid:
    """Fill enclosed bg regions within each object's bbox with that object's color."""
    from collections import deque as _deque
    grid, bg = args[0], args[1]
    predicted = grid.copy()
    objs = execute_prim("select_objects", grid, bg)
    for obj in objs:
        r0, c0 = obj.row, obj.col
        r1, c1 = r0 + obj.bbox_h, c0 + obj.bbox_w
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape
        if h < 3 or w < 3:
            continue
        wall = sub != bg
        reachable = np.zeros((h, w), dtype=bool)
        q = _deque()
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
        for r in range(h):
            for c in range(w):
                if enclosed[r, c]:
                    predicted[r0 + r, c0 + c] = obj.color
    return predicted


def _exec_for_each_delete(args: list, params: dict) -> Grid:
    """Delete objects matching a predicate."""
    grid, filtered_objs, bg = args[0], args[1], args[2]
    out = grid.copy()
    for obj in filtered_objs:
        for r in range(obj.bbox_h):
            for c in range(obj.bbox_w):
                if obj.mask[r, c]:
                    out[obj.row + r, obj.col + c] = bg
    return out


def _exec_for_each_recolor(args: list, params: dict) -> Grid:
    """Recolor objects matching a predicate."""
    grid, filtered_objs, bg = args[0], args[1], args[2]
    target_color = params.get("color", 0)
    out = grid.copy()
    for obj in filtered_objs:
        for r in range(obj.bbox_h):
            for c in range(obj.bbox_w):
                if obj.mask[r, c]:
                    out[obj.row + r, obj.col + c] = target_color
    return out


def _exec_filter_by_predicate(args: list, params: dict) -> list:
    """Filter objects by a predicate string."""
    objects = args[0]
    pred = params.get("predicate", "")

    if pred.startswith("color_in("):
        # Parse color_in([1, 2, 3])
        import ast
        colors_str = pred[len("color_in("):-1]
        try:
            colors = set(ast.literal_eval(colors_str))
        except Exception:
            return objects
        return [o for o in objects if o.color in colors]

    if pred == "is_singleton":
        return [o for o in objects if o.is_singleton]

    if pred == "not_singleton":
        return [o for o in objects if not o.is_singleton]

    if pred.startswith("size<="):
        threshold = int(pred[len("size<="):])
        return [o for o in objects if o.size <= threshold]

    if pred.startswith("size>="):
        threshold = int(pred[len("size>="):])
        return [o for o in objects if o.size >= threshold]

    if pred == "enclosed_by_larger":
        # Objects whose bbox is fully inside some larger object's bbox
        result = []
        for o in objects:
            for other in objects:
                if other is o:
                    continue
                if (o.row >= other.row and o.col >= other.col
                        and o.row + o.bbox_h <= other.row + other.bbox_h
                        and o.col + o.bbox_w <= other.col + other.bbox_w
                        and other.size > o.size):
                    result.append(o)
                    break
        return result

    if pred == "unique_color":
        from collections import Counter as _Counter
        color_counts = _Counter(o.color for o in objects)
        return [o for o in objects if color_counts[o.color] == 1]

    if pred == "touches_border":
        # Need grid shape — not available here. Return all as fallback.
        return objects

    return objects


def _exec_for_each_move(args: list, params: dict) -> Grid:
    """Move objects by a fixed offset."""
    grid, filtered_objs, bg = args[0], args[1], args[2]
    dr, dc = params.get("dr", 0), params.get("dc", 0)
    out = grid.copy()
    # Erase old positions
    for obj in filtered_objs:
        for r in range(obj.bbox_h):
            for c in range(obj.bbox_w):
                if obj.mask[r, c]:
                    out[obj.row + r, obj.col + c] = bg
    # Paint new positions
    for obj in filtered_objs:
        for r in range(obj.bbox_h):
            for c in range(obj.bbox_w):
                if obj.mask[r, c]:
                    nr = obj.row + r + dr
                    nc = obj.col + c + dc
                    if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                        out[nr, nc] = obj.color
    return out


# Composite executor registry
_COMPOSITE_EXECUTORS = {
    "periodic_row_repair": _exec_periodic_row_repair,
    "periodic_col_repair": _exec_periodic_col_repair,
    "fill_enclosed_per_object": _exec_fill_enclosed_per_object,
    "for_each_delete": _exec_for_each_delete,
    "for_each_recolor": _exec_for_each_recolor,
    "for_each_move": _exec_for_each_move,
    "filter_by_predicate": _exec_filter_by_predicate,
}
