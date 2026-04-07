"""Residual explanation engine for the NGS solver.

After preservation factoring, explains only the residual using
support context from the preserved region.

Key strategy: instead of searching over the full grid, we explain
each residual region using its local structural context:
- what color encloses it
- what colors are adjacent
- what objects are nearby
- whether it's enclosed

This is where the residual output color is determined.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.ngs.ir import PrimitiveGraph, Leaf, PrimCall, VType, AbstractRule, UnifiedRule, DemoBinding
from aria.ngs.preservation import PreservationFactor, ResidualRegion, SupportContext
from aria.types import Grid, DemoPair


# ---------------------------------------------------------------------------
# Residual rule: a single rule explaining all residual regions in a demo
# ---------------------------------------------------------------------------

@dataclass
class ResidualRule:
    """A rule explaining the residual of one demo."""
    rule_type: str    # canonical name of the rule pattern
    params: dict[str, Any]  # parameters (may be abstract or concrete)
    confidence: float
    description: str

    def canon_key(self) -> tuple:
        """Canonical form for cross-demo comparison."""
        return (self.rule_type, tuple(sorted(
            (k, v) for k, v in self.params.items()
            if k != "concrete_changes"  # strip demo-specific data
        )))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def explain_residual(
    pf: PreservationFactor,
    inp: Grid,
    out: Grid,
    bg: int,
) -> list[ResidualRule]:
    """Find rules that explain all residual regions in one demo.

    Returns candidate rules ranked by confidence.
    """
    if pf.n_residual == 0:
        return [ResidualRule("identity", {}, 1.0, "no residual")]

    candidates: list[ResidualRule] = []

    # Strategy 1: fill_enclosed_with_adjacent_color
    # Each enclosed bg region gets filled with the color of its enclosing frame
    _try_fill_enclosed_adjacent(pf, inp, out, bg, candidates)

    # Strategy 2: recolor_enclosed_to_adjacent
    # Each enclosed non-bg region gets recolored to match adjacent singletons
    _try_recolor_to_adjacent(pf, inp, out, bg, candidates)

    # Strategy 3: delete_enclosed_anomalies
    # Enclosed pixels that break local symmetry get deleted
    _try_delete_enclosed(pf, inp, out, bg, candidates)

    # Strategy 4: swap_by_enclosure
    # Objects enclosed by different frames swap colors
    _try_swap_enclosed(pf, inp, out, bg, candidates)

    # Strategy 5: fill_enclosed_with_enclosing_color
    # Enclosed bg gets the color of the enclosing object itself
    _try_fill_with_enclosing_color(pf, inp, out, bg, candidates)

    candidates.sort(key=lambda r: -r.confidence)
    return candidates


# ---------------------------------------------------------------------------
# Strategy 1: fill enclosed bg with adjacent frame color
# ---------------------------------------------------------------------------

def _try_fill_enclosed_adjacent(
    pf: PreservationFactor, inp: Grid, out: Grid, bg: int,
    candidates: list[ResidualRule],
) -> None:
    """Each bg residual cell fills with the color of its surrounding frame."""
    if not pf.residual_regions:
        return

    # Check: all residual regions are add-only (bg -> non-bg)
    if not all(r.change_type == "add" for r in pf.residual_regions):
        return

    # Check: each region's output color matches an adjacent color
    for reg in pf.residual_regions:
        if len(reg.output_colors) != 1:
            return
        out_c = next(iter(reg.output_colors))
        if out_c not in reg.adjacent_colors:
            return

    # Verify
    predicted = inp.copy()
    for reg in pf.residual_regions:
        out_c = next(iter(reg.output_colors))
        r0, c0, r1, c1 = reg.bbox
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                if reg.mask[r, c]:
                    gr, gc = r0 + r, c0 + c
                    if inp[gr, gc] == bg:
                        predicted[gr, gc] = out_c

    if np.array_equal(predicted, out):
        conf = 0.92 if all(r.is_enclosed for r in pf.residual_regions) else 0.85
        candidates.append(ResidualRule(
            "fill_bg_with_adjacent_color",
            {},
            conf,
            "fill bg residual with adjacent non-bg color",
        ))


# ---------------------------------------------------------------------------
# Strategy 2: recolor enclosed objects to adjacent singleton color
# ---------------------------------------------------------------------------

def _try_recolor_to_adjacent(
    pf: PreservationFactor, inp: Grid, out: Grid, bg: int,
    candidates: list[ResidualRule],
) -> None:
    """Recolored objects get the color of an adjacent non-bg neighbor."""
    if not pf.residual_regions:
        return

    if not all(r.change_type == "recolor" for r in pf.residual_regions):
        return

    for reg in pf.residual_regions:
        if len(reg.output_colors) != 1:
            return
        out_c = next(iter(reg.output_colors))
        if out_c not in reg.adjacent_colors:
            return

    # Verify
    predicted = inp.copy()
    for reg in pf.residual_regions:
        out_c = next(iter(reg.output_colors))
        r0, c0, r1, c1 = reg.bbox
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                if reg.mask[r, c]:
                    gr, gc = r0 + r, c0 + c
                    if inp[gr, gc] != bg:
                        predicted[gr, gc] = out_c

    if np.array_equal(predicted, out):
        candidates.append(ResidualRule(
            "recolor_to_adjacent",
            {},
            0.85,
            "recolor objects to adjacent non-bg color",
        ))


# ---------------------------------------------------------------------------
# Strategy 3: delete enclosed anomaly pixels
# ---------------------------------------------------------------------------

def _try_delete_enclosed(
    pf: PreservationFactor, inp: Grid, out: Grid, bg: int,
    candidates: list[ResidualRule],
) -> None:
    """Enclosed pixels that are anomalous get deleted (set to bg)."""
    if not pf.residual_regions:
        return

    if not all(r.change_type == "delete" for r in pf.residual_regions):
        return

    # All residual output colors should be bg
    for reg in pf.residual_regions:
        if reg.output_colors != {bg}:
            return

    # The rule is: enclosed pixels of certain colors get erased
    # Check what input colors are being deleted
    deleted_colors: set[int] = set()
    for reg in pf.residual_regions:
        deleted_colors |= reg.input_colors

    # Simple case: all deleted pixels share some property
    # For now, just record the rule
    predicted = inp.copy()
    for reg in pf.residual_regions:
        r0, c0, r1, c1 = reg.bbox
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                if reg.mask[r, c]:
                    gr, gc = r0 + r, c0 + c
                    predicted[gr, gc] = bg

    if np.array_equal(predicted, out):
        candidates.append(ResidualRule(
            "delete_enclosed_anomalies",
            {"deleted_colors": sorted(deleted_colors)},
            0.7,
            f"delete enclosed pixels of colors {sorted(deleted_colors)}",
        ))


# ---------------------------------------------------------------------------
# Strategy 4: swap colors of enclosed objects
# ---------------------------------------------------------------------------

def _try_swap_enclosed(
    pf: PreservationFactor, inp: Grid, out: Grid, bg: int,
    candidates: list[ResidualRule],
) -> None:
    """Enclosed objects swap colors."""
    if not pf.residual_regions:
        return
    if len(pf.residual_regions) < 2:
        return

    if not all(r.change_type == "recolor" for r in pf.residual_regions):
        return

    # Check if there's a consistent swap pattern
    swap_map: dict[int, int] = {}
    for reg in pf.residual_regions:
        if len(reg.input_colors) != 1 or len(reg.output_colors) != 1:
            return
        from_c = next(iter(reg.input_colors))
        to_c = next(iter(reg.output_colors))
        if from_c in swap_map and swap_map[from_c] != to_c:
            return
        swap_map[from_c] = to_c

    # Verify it's actually a swap (A->B and B->A)
    is_swap = False
    for a, b in swap_map.items():
        if b in swap_map and swap_map[b] == a and a != b:
            is_swap = True
            break

    if not is_swap:
        return

    # Verify
    predicted = inp.copy()
    for reg in pf.residual_regions:
        from_c = next(iter(reg.input_colors))
        to_c = swap_map[from_c]
        r0, c0, r1, c1 = reg.bbox
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                if reg.mask[r, c]:
                    gr, gc = r0 + r, c0 + c
                    if inp[gr, gc] == from_c:
                        predicted[gr, gc] = to_c

    if np.array_equal(predicted, out):
        candidates.append(ResidualRule(
            "swap_enclosed_colors",
            {"swap": dict(swap_map)},
            0.85,
            f"swap enclosed object colors: {swap_map}",
        ))


# ---------------------------------------------------------------------------
# Strategy 5: fill enclosed bg with enclosing object's own color
# ---------------------------------------------------------------------------

def _try_fill_with_enclosing_color(
    pf: PreservationFactor, inp: Grid, out: Grid, bg: int,
    candidates: list[ResidualRule],
) -> None:
    """Enclosed bg cells get filled with the color of their enclosing object."""
    if not pf.residual_regions:
        return

    if not all(r.change_type == "add" and r.is_enclosed for r in pf.residual_regions):
        return

    for reg in pf.residual_regions:
        if len(reg.output_colors) != 1:
            return
        out_c = next(iter(reg.output_colors))
        if reg.enclosing_color is None or out_c != reg.enclosing_color:
            return

    # Verify
    predicted = inp.copy()
    for reg in pf.residual_regions:
        r0, c0, r1, c1 = reg.bbox
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                if reg.mask[r, c]:
                    gr, gc = r0 + r, c0 + c
                    if inp[gr, gc] == bg:
                        predicted[gr, gc] = reg.enclosing_color

    if np.array_equal(predicted, out):
        candidates.append(ResidualRule(
            "fill_enclosed_with_enclosing_color",
            {},
            0.88,
            "fill enclosed bg with enclosing object color",
        ))


# ---------------------------------------------------------------------------
# Cross-demo unification of residual rules
# ---------------------------------------------------------------------------

def unify_residual_rules(
    per_demo_rules: list[list[ResidualRule]],
    per_demo_factors: list[PreservationFactor],
    demos: tuple[DemoPair, ...],
) -> UnifiedRule | None:
    """Unify residual rules across demos.

    All demos must agree on the rule type (canonical key).
    The unified rule is then verified on all demos.
    """
    if not per_demo_rules or any(len(r) == 0 for r in per_demo_rules):
        return None

    n_demos = len(demos)

    # Group by rule type: find types that appear in all demos
    from collections import defaultdict
    type_to_demos: dict[str, list[tuple[int, ResidualRule]]] = defaultdict(list)

    for di, rules in enumerate(per_demo_rules):
        for rule in rules:
            type_to_demos[rule.rule_type].append((di, rule))

    best: UnifiedRule | None = None
    best_conf = -1.0

    for rule_type, demo_rules in type_to_demos.items():
        demo_indices = set(di for di, _ in demo_rules)
        if len(demo_indices) < n_demos:
            continue

        # Take first rule per demo
        per_demo: dict[int, ResidualRule] = {}
        for di, rule in demo_rules:
            if di not in per_demo:
                per_demo[di] = rule

        if len(per_demo) < n_demos:
            continue

        # Build and verify the unified rule
        unified = _verify_residual_rule(rule_type, per_demo, demos)
        if unified is not None and unified.train_verified:
            avg_conf = sum(per_demo[di].confidence for di in range(n_demos)) / n_demos
            if avg_conf > best_conf:
                best = unified
                best_conf = avg_conf

    return best


def _verify_residual_rule(
    rule_type: str,
    per_demo: dict[int, ResidualRule],
    demos: tuple[DemoPair, ...],
) -> UnifiedRule | None:
    """Verify a residual rule on all demos by re-executing it."""
    from aria.ngs.preservation import factor_preservation

    train_diff = 0
    all_ok = True

    for di, demo in enumerate(demos):
        bg = detect_bg(demo.input)
        predicted = _execute_residual_rule(rule_type, per_demo.get(di), demo.input, bg)
        if predicted is None or not np.array_equal(predicted, demo.output):
            all_ok = False
            if predicted is not None and predicted.shape == demo.output.shape:
                train_diff += int(np.sum(predicted != demo.output))
            else:
                train_diff += demo.output.size

    # Build the abstract rule as a primitive graph
    g = PrimitiveGraph()
    i_inp = g.add_leaf("input", VType.GRID, None)
    i_bg = g.add_leaf("bg", VType.COLOR, None)
    i_out = g.add_prim(rule_type, (i_inp, i_bg), VType.GRID)
    g.set_output(i_out)

    abstract = AbstractRule(graph=g, description=rule_type)
    bindings = [DemoBinding(demo_idx=di, leaf_values={}) for di in range(len(demos))]

    return UnifiedRule(
        rule=abstract,
        bindings=bindings,
        train_verified=all_ok,
        train_diff=train_diff,
    )


def _execute_residual_rule(
    rule_type: str,
    rule: ResidualRule | None,
    inp: Grid,
    bg: int,
) -> Grid | None:
    """Execute a residual rule on an input grid."""
    from aria.ngs.preservation import factor_preservation

    if rule_type == "identity":
        return inp.copy()

    pf = factor_preservation(inp, inp, bg)  # dummy — we need the real output
    # Actually, we need to APPLY the rule to a new input where we don't know the output
    # This means the rule must be executable from input alone

    if rule_type in ("fill_enclosed_adjacent", "fill_bg_with_adjacent_color"):
        return _exec_fill_enclosed_adjacent(inp, bg)
    elif rule_type == "fill_enclosed_with_enclosing_color":
        return _exec_fill_enclosed_with_enclosing(inp, bg)
    elif rule_type in ("recolor_enclosed_to_adjacent", "recolor_to_adjacent"):
        return _exec_recolor_to_adjacent(inp, bg)
    elif rule_type == "delete_enclosed_anomalies":
        return _exec_delete_enclosed_anomalies(inp, bg, rule)
    elif rule_type == "swap_enclosed_colors":
        return _exec_swap_enclosed(inp, bg, rule)

    return None


def _exec_fill_enclosed_adjacent(inp: Grid, bg: int) -> Grid:
    """Fill each enclosed bg region with the non-bg color of its boundary."""
    from collections import deque

    predicted = inp.copy()
    rows, cols = inp.shape

    # Find all bg cells not reachable from the border
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and inp[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and inp[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))

    enclosed = (inp == bg) & (~reachable)
    if not np.any(enclosed):
        return predicted

    # Label enclosed regions
    from scipy import ndimage
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))

    for label_id in range(1, n + 1):
        component = labeled == label_id
        # Find the adjacent non-bg color
        struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        dilated = ndimage.binary_dilation(component, structure=struct4)
        border = dilated & ~component
        border_colors = set(int(v) for v in np.unique(inp[border])) - {bg}

        if len(border_colors) == 1:
            fill_c = next(iter(border_colors))
            predicted[component] = fill_c
        elif len(border_colors) > 1:
            # Use most common non-bg border color
            border_vals = inp[border]
            border_vals = border_vals[border_vals != bg]
            if len(border_vals) > 0:
                fill_c = int(Counter(border_vals.tolist()).most_common(1)[0][0])
                predicted[component] = fill_c

    return predicted


def _exec_fill_enclosed_with_enclosing(inp: Grid, bg: int) -> Grid:
    """Fill enclosed bg within each object's bbox with that object's color."""
    from collections import deque

    predicted = inp.copy()
    objects = extract_objects(inp, bg, connectivity=4)

    for obj in objects:
        r0, c0 = obj.row, obj.col
        r1, c1 = r0 + obj.bbox_h, c0 + obj.bbox_w
        sub = inp[r0:r1, c0:c1]
        h, w = sub.shape
        if h < 3 or w < 3:
            continue
        wall = sub != bg
        reachable = np.zeros((h, w), dtype=bool)
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
        for r in range(h):
            for c in range(w):
                if enclosed[r, c]:
                    predicted[r0 + r, c0 + c] = obj.color

    return predicted


def _exec_recolor_to_adjacent(inp: Grid, bg: int) -> Grid:
    """Recolor each object that has a unique adjacent different-colored neighbor.

    For each object, find adjacent non-bg colors (excluding own color).
    If there's exactly one such color, recolor the object to it.
    """
    from scipy import ndimage

    predicted = inp.copy()
    rows, cols = inp.shape
    objects = extract_objects(inp, bg, connectivity=4)

    for obj in objects:
        obj_mask = np.zeros((rows, cols), dtype=bool)
        obj_mask[obj.row:obj.row + obj.bbox_h, obj.col:obj.col + obj.bbox_w] |= obj.mask
        struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        dilated = ndimage.binary_dilation(obj_mask, structure=struct4)
        border = dilated & ~obj_mask
        if not np.any(border):
            continue
        adj_colors = set(int(v) for v in np.unique(inp[border])) - {bg, obj.color}

        if len(adj_colors) == 1:
            target = next(iter(adj_colors))
            for r in range(obj.bbox_h):
                for c in range(obj.bbox_w):
                    if obj.mask[r, c]:
                        predicted[obj.row + r, obj.col + c] = target

    return predicted


def _exec_delete_enclosed_anomalies(inp: Grid, bg: int, rule: ResidualRule | None) -> Grid:
    """Delete enclosed pixels — placeholder, needs anomaly detection."""
    # Without knowing WHICH enclosed pixels are anomalous, we can't execute this generically.
    # This rule type needs more structural analysis to be executable.
    return inp.copy()


def _exec_swap_enclosed(inp: Grid, bg: int, rule: ResidualRule | None) -> Grid:
    """Swap colors of enclosed objects."""
    if rule is None or "swap" not in rule.params:
        return inp.copy()

    swap_map = rule.params["swap"]
    predicted = inp.copy()

    # Find enclosed non-bg cells and apply swap
    from collections import deque
    rows, cols = inp.shape
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and inp[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and inp[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            val = int(inp[r, c])
            if val != bg and not reachable[r, c] and val in swap_map:
                predicted[r, c] = swap_map[val]

    return predicted
