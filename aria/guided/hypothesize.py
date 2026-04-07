"""Hypothesis engine: accumulate rules across demos, test, and apply.

For each task:
1. Analyze residual objects in each demo
2. For each residual object, generate hypotheses about WHY it's there
3. Accumulate hypotheses across demos into a unified rule
4. Verify the rule reproduces ALL demos
5. Apply to test

A hypothesis is: "object X became Y because of Z"
A rule is: the set of all such hypotheses that are consistent across demos.

Key insight: each demo may reveal DIFFERENT aspects of the rule.
Demo 1 might show the rule for red objects, demo 2 for blue.
The full rule is the accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict

import numpy as np
from scipy import ndimage

from aria.guided.residual_objects import (
    analyze_residual_objects, ResidualObject, DemoResidual,
)
from aria.guided.construct import construct_canvas, ConstructedCanvas
from aria.guided.workspace import _detect_bg, _extract_objects, ObjectInfo
from aria.types import Grid


# ---------------------------------------------------------------------------
# Hypothesis: one observation about one object in one demo
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """Why one residual object exists."""
    cause: str                    # SAME_POS_RECOLOR, MOVED, FILLED, NEW, etc.
    source_color: int             # original color (-1 if none)
    target_color: int             # output color
    condition: str                # what determines this (structural description)
    condition_key: tuple          # hashable key for matching across demos
    demo_idx: int


# ---------------------------------------------------------------------------
# Rule: accumulated hypotheses that generalize across demos
# ---------------------------------------------------------------------------

@dataclass
class AccumulatedRule:
    """A rule built from cross-demo observation."""
    cause: str
    mappings: dict[tuple, Any]    # condition_key -> action
    description: str
    n_demos_supporting: int

    def apply(self, inp: Grid) -> Grid:
        """Apply this rule to an input to produce output."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete rule types
# ---------------------------------------------------------------------------

class RecolorRule(AccumulatedRule):
    """Recolor objects based on accumulated color mappings."""
    def __init__(self):
        super().__init__("SAME_POS_RECOLOR", {}, "recolor", 0)
        self.color_map: dict[int, int] = {}  # source_color -> target_color

    def add_mapping(self, source_color: int, target_color: int, demo_idx: int):
        if source_color in self.color_map:
            if self.color_map[source_color] != target_color:
                return False  # conflict
        self.color_map[source_color] = target_color
        self.n_demos_supporting = max(self.n_demos_supporting, demo_idx + 1)
        return True

    def apply(self, inp: Grid) -> Grid:
        out = inp.copy()
        # Apply simultaneously using LUT
        lut = np.arange(256, dtype=np.uint8)
        for fc, tc in self.color_map.items():
            lut[fc] = tc
        return lut[inp]


class ContextRecolorRule(AccumulatedRule):
    """Recolor based on structural context (adjacent, enclosing, etc.)."""
    def __init__(self, context_fn: Callable, description: str):
        super().__init__("SAME_POS_RECOLOR", {}, description, 0)
        self._context_fn = context_fn

    def apply(self, inp: Grid) -> Grid:
        return self._context_fn(inp)


class FillRule(AccumulatedRule):
    """Fill bg regions based on accumulated observations."""
    def __init__(self):
        super().__init__("FILLED", {}, "fill", 0)
        self.fill_map: dict[int, int] = {}  # enclosing_color -> fill_color
        self.single_fill: int | None = None

    def add_mapping(self, enclosing_color: int, fill_color: int, demo_idx: int):
        if enclosing_color >= 0:
            if enclosing_color in self.fill_map:
                if self.fill_map[enclosing_color] != fill_color:
                    return False
            self.fill_map[enclosing_color] = fill_color
        if self.single_fill is None:
            self.single_fill = fill_color
        elif self.single_fill != fill_color:
            self.single_fill = -1  # varies
        self.n_demos_supporting = max(self.n_demos_supporting, demo_idx + 1)
        return True

    def apply(self, inp: Grid) -> Grid:
        bg = _detect_bg(inp)
        out = inp.copy()
        enclosed = _get_enclosed(inp, bg)
        if not np.any(enclosed):
            return out

        if self.fill_map:
            # Fill each enclosed region with its enclosing color's mapped value
            labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
            struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
            for lid in range(1, n + 1):
                comp = labeled == lid
                dilated = ndimage.binary_dilation(comp, structure=struct4)
                border = dilated & ~comp
                vals = inp[border]
                non_bg = vals[vals != bg]
                if len(non_bg) > 0:
                    from collections import Counter
                    enc_c = int(Counter(non_bg.tolist()).most_common(1)[0][0])
                    if enc_c in self.fill_map:
                        out[comp] = self.fill_map[enc_c]
                    elif self.single_fill and self.single_fill >= 0:
                        out[comp] = self.single_fill
        elif self.single_fill is not None and self.single_fill >= 0:
            out[enclosed] = self.single_fill

        return out


class MoveRule(AccumulatedRule):
    """Move objects based on accumulated observations."""
    def __init__(self):
        super().__init__("MOVED", {}, "move", 0)
        self.offset: tuple[int, int] | None = None
        self.offsets_consistent: bool = True

    def add_offset(self, dr: int, dc: int, demo_idx: int):
        if self.offset is None:
            self.offset = (dr, dc)
        elif self.offset != (dr, dc):
            self.offsets_consistent = False
        self.n_demos_supporting = max(self.n_demos_supporting, demo_idx + 1)

    def apply(self, inp: Grid) -> Grid:
        if not self.offsets_consistent or self.offset is None:
            return inp.copy()
        bg = _detect_bg(inp)
        out = inp.copy()
        dr, dc = self.offset
        objs = _extract_objects(inp, bg)
        # Move all objects
        for obj in objs:
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.mask[r, c]:
                        out[obj.row + r, obj.col + c] = bg
        for obj in objs:
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.mask[r, c]:
                        nr, nc = obj.row + r + dr, obj.col + c + dc
                        if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                            out[nr, nc] = obj.color
        return out


class CompositeRule(AccumulatedRule):
    """Multiple rules applied in sequence."""
    def __init__(self, steps: list[AccumulatedRule]):
        desc = " + ".join(s.description for s in steps)
        super().__init__("COMPOSITE", {}, desc, 0)
        self.steps = steps

    def apply(self, inp: Grid) -> Grid:
        result = inp.copy()
        for step in self.steps:
            result = step.apply(result)
        return result


# ---------------------------------------------------------------------------
# Main: hypothesize, accumulate, verify
# ---------------------------------------------------------------------------

def hypothesize_and_verify(
    demos: list[tuple[Grid, Grid]],
) -> AccumulatedRule | None:
    """Build rules from cross-demo observation and verify."""
    if not demos:
        return None

    residuals = analyze_residual_objects(demos)

    # Collect all cause types across demos
    all_causes = set()
    for dr in residuals:
        for obj in dr.objects:
            all_causes.add(obj.cause)

    # Try building rules for each cause type
    candidate_rules = []

    if 'SAME_POS_RECOLOR' in all_causes:
        rule = _build_recolor_rule(residuals, demos)
        if rule:
            candidate_rules.append(rule)

    if 'FILLED' in all_causes:
        rule = _build_fill_rule(residuals, demos)
        if rule:
            candidate_rules.append(rule)

    if 'MOVED' in all_causes:
        rule = _build_move_rule(residuals, demos)
        if rule:
            candidate_rules.append(rule)

    # Try single rules first
    for rule in candidate_rules:
        ok, diff = _verify_rule(rule, demos)
        if ok:
            return rule

    # Try composites (pairs)
    for i, r1 in enumerate(candidate_rules):
        for r2 in candidate_rules[i + 1:]:
            composite = CompositeRule([r1, r2])
            ok, diff = _verify_rule(composite, demos)
            if ok:
                return composite
            # Try reverse order
            composite2 = CompositeRule([r2, r1])
            ok2, diff2 = _verify_rule(composite2, demos)
            if ok2:
                return composite2

    # Return best partial
    if candidate_rules:
        best = min(candidate_rules, key=lambda r: _verify_rule(r, demos)[1])
        return best

    return None


# ---------------------------------------------------------------------------
# Rule builders
# ---------------------------------------------------------------------------

def _build_recolor_rule(residuals, demos):
    """Accumulate color mappings across demos."""
    rule = RecolorRule()
    for di, dr in enumerate(residuals):
        for obj in dr.objects:
            if obj.cause == 'SAME_POS_RECOLOR' and obj.source_color >= 0:
                if not rule.add_mapping(obj.source_color, obj.new_color, di):
                    return None  # conflict
    if not rule.color_map:
        return None
    rule.description = f"recolor {rule.color_map}"
    return rule


def _build_fill_rule(residuals, demos):
    """Accumulate fill observations across demos."""
    rule = FillRule()
    for di, (dr, (inp, out)) in enumerate(zip(residuals, demos)):
        bg = _detect_bg(inp)
        for obj in dr.objects:
            if obj.cause != 'FILLED':
                continue
            # Find enclosing color
            obj_mask = np.zeros(inp.shape, dtype=bool)
            r0, c0 = obj.row, obj.col
            r1 = min(r0 + obj.height, inp.shape[0])
            c1 = min(c0 + obj.width, inp.shape[1])
            sub_mask = obj.mask[:r1 - r0, :c1 - c0]
            obj_mask[r0:r1, c0:c1] |= sub_mask
            struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
            dilated = ndimage.binary_dilation(obj_mask, structure=struct4)
            border = dilated & ~obj_mask
            vals = inp[border]
            non_bg = vals[vals != bg]
            enc_c = -1
            if len(non_bg) > 0:
                from collections import Counter
                enc_c = int(Counter(non_bg.tolist()).most_common(1)[0][0])
            rule.add_mapping(enc_c, obj.new_color, di)

    if rule.single_fill is None and not rule.fill_map:
        return None
    rule.description = f"fill enclosed (map={rule.fill_map}, single={rule.single_fill})"
    return rule


def _build_move_rule(residuals, demos):
    """Accumulate movement offsets across demos."""
    rule = MoveRule()
    for di, dr in enumerate(residuals):
        for obj in dr.objects:
            if obj.cause == 'MOVED' and obj.source_obj is not None:
                dr_off = obj.row - obj.source_obj.row
                dc_off = obj.col - obj.source_obj.col
                rule.add_offset(dr_off, dc_off, di)
    if rule.offset is None:
        return None
    rule.description = f"move by {rule.offset}"
    return rule


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_rule(rule, demos):
    """Verify rule on all train demos. Returns (exact_match, total_diff)."""
    total_diff = 0
    for inp, out in demos:
        try:
            pred = rule.apply(inp)
        except Exception:
            return False, sum(o.size for _, o in demos)
        if pred.shape != out.shape:
            return False, sum(o.size for _, o in demos)
        d = int(np.sum(pred != out))
        total_diff += d
    return total_diff == 0, total_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_enclosed(grid, bg):
    from collections import deque
    rows, cols = grid.shape
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return (grid == bg) & ~reachable
