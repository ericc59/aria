"""Small symbolic predicate language over objects/regions.

Predicates are composable, inspectable, and grounded in perception.
They are used for predicate-parameterized dispatch: apply op O only
to entities satisfying predicate P, where P is inferred from
cross-demo structural evidence.

Predicate language:
  Atomic:
    HasColor(c)             — object's dominant color is c
    IsSize(op, threshold)   — size comparison (>, <, ==)
    IsSingleton()           — size == 1
    AdjacentTo(color=c)     — has neighbor pixel of color c
    InsideFrame()           — contained within a framed region
    TouchesBorder()         — touches grid edge
    IsLargestOfColor()      — largest object of its color
    IsSmallestOfColor()     — smallest object of its color
    NearestTo(color=c)      — closest to any pixel of color c
    HasNeighborKind(kind)   — adjacent to object of kind (singleton, large, etc)

  Composite:
    And(p1, p2)             — both predicates hold
    Not(p)                  — predicate does not hold
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from aria.decomposition import RawObject, detect_bg, extract_objects
from aria.types import Grid


# ---------------------------------------------------------------------------
# Predicate types
# ---------------------------------------------------------------------------


class PredicateKind(str, Enum):
    HAS_COLOR = "has_color"
    IS_SIZE = "is_size"
    IS_SINGLETON = "is_singleton"
    ADJACENT_TO = "adjacent_to"
    ADJACENT_EDGE = "adjacent_edge"  # adjacent with >=N contact pixels
    INSIDE_FRAME = "inside_frame"
    TOUCHES_BORDER = "touches_border"
    IS_LARGEST_OF_COLOR = "is_largest_of_color"
    IS_SMALLEST_OF_COLOR = "is_smallest_of_color"
    NEAREST_TO = "nearest_to"
    HAS_NEIGHBOR_KIND = "has_neighbor_kind"
    NOT_SINGLETON = "not_singleton"
    SIZE_ABOVE_MEAN = "size_above_mean"
    SIZE_BELOW_MEAN = "size_below_mean"
    AND = "and"
    NOT = "not"
    TRUE = "true"


@dataclass(frozen=True)
class Predicate:
    kind: PredicateKind
    params: tuple[tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.params:
            if k == key:
                return v
        return default

    def __repr__(self) -> str:
        if self.kind == PredicateKind.TRUE:
            return "True"
        if self.kind == PredicateKind.AND:
            return f"({self.get('left')} & {self.get('right')})"
        if self.kind == PredicateKind.NOT:
            return f"!{self.get('inner')}"
        params_str = ", ".join(f"{k}={v}" for k, v in self.params)
        return f"{self.kind.value}({params_str})"


# Convenience constructors
def P_true() -> Predicate:
    return Predicate(PredicateKind.TRUE)

def P_has_color(c: int) -> Predicate:
    return Predicate(PredicateKind.HAS_COLOR, (("color", c),))

def P_is_singleton() -> Predicate:
    return Predicate(PredicateKind.IS_SINGLETON)

def P_adjacent_to(color: int) -> Predicate:
    return Predicate(PredicateKind.ADJACENT_TO, (("color", color),))

def P_is_size(op: str, threshold: int) -> Predicate:
    return Predicate(PredicateKind.IS_SIZE, (("op", op), ("threshold", threshold),))

def P_touches_border() -> Predicate:
    return Predicate(PredicateKind.TOUCHES_BORDER)

def P_inside_frame() -> Predicate:
    return Predicate(PredicateKind.INSIDE_FRAME)

def P_is_largest_of_color() -> Predicate:
    return Predicate(PredicateKind.IS_LARGEST_OF_COLOR)

def P_is_smallest_of_color() -> Predicate:
    return Predicate(PredicateKind.IS_SMALLEST_OF_COLOR)

def P_nearest_to(color: int) -> Predicate:
    return Predicate(PredicateKind.NEAREST_TO, (("color", color),))

def P_adjacent_edge(color: int, min_contact: int = 2) -> Predicate:
    return Predicate(PredicateKind.ADJACENT_EDGE, (("color", color), ("min_contact", min_contact),))

def P_not_singleton() -> Predicate:
    return Predicate(PredicateKind.NOT_SINGLETON)

def P_size_above_mean() -> Predicate:
    return Predicate(PredicateKind.SIZE_ABOVE_MEAN)

def P_size_below_mean() -> Predicate:
    return Predicate(PredicateKind.SIZE_BELOW_MEAN)

def P_and(left: Predicate, right: Predicate) -> Predicate:
    return Predicate(PredicateKind.AND, (("left", left), ("right", right),))

def P_not(inner: Predicate) -> Predicate:
    return Predicate(PredicateKind.NOT, (("inner", inner),))


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


@dataclass
class ObjectContext:
    """Everything needed to evaluate a predicate on one object."""
    obj: RawObject
    grid: Grid
    bg: int
    all_objects: list[RawObject]
    grid_shape: tuple[int, int]
    framed_regions: tuple = ()


def evaluate_predicate(pred: Predicate, ctx: ObjectContext) -> bool:
    """Evaluate a predicate on one object in context."""
    kind = pred.kind

    if kind == PredicateKind.TRUE:
        return True

    if kind == PredicateKind.HAS_COLOR:
        return ctx.obj.color == pred.get("color")

    if kind == PredicateKind.IS_SINGLETON:
        return ctx.obj.size == 1

    if kind == PredicateKind.IS_SIZE:
        op = pred.get("op", ">")
        thr = pred.get("threshold", 1)
        if op == ">": return ctx.obj.size > thr
        if op == "<": return ctx.obj.size < thr
        if op == ">=": return ctx.obj.size >= thr
        if op == "<=": return ctx.obj.size <= thr
        if op == "==": return ctx.obj.size == thr
        return False

    if kind == PredicateKind.ADJACENT_TO:
        target_color = pred.get("color")
        return _is_adjacent_to_color(ctx.obj, ctx.grid, target_color)

    if kind == PredicateKind.ADJACENT_EDGE:
        target_color = pred.get("color")
        min_contact = pred.get("min_contact", 2)
        return _adjacent_contact_count(ctx.obj, ctx.grid, target_color) >= min_contact

    if kind == PredicateKind.NOT_SINGLETON:
        return ctx.obj.size > 1

    if kind == PredicateKind.SIZE_ABOVE_MEAN:
        sizes = [o.size for o in ctx.all_objects if o.color != ctx.bg]
        return ctx.obj.size > (sum(sizes) / max(len(sizes), 1))

    if kind == PredicateKind.SIZE_BELOW_MEAN:
        sizes = [o.size for o in ctx.all_objects if o.color != ctx.bg]
        return ctx.obj.size < (sum(sizes) / max(len(sizes), 1))

    if kind == PredicateKind.TOUCHES_BORDER:
        return _touches_border(ctx.obj, ctx.grid_shape)

    if kind == PredicateKind.INSIDE_FRAME:
        return _inside_frame(ctx.obj, ctx.framed_regions)

    if kind == PredicateKind.IS_LARGEST_OF_COLOR:
        same_color = [o for o in ctx.all_objects if o.color == ctx.obj.color]
        return ctx.obj.size == max(o.size for o in same_color)

    if kind == PredicateKind.IS_SMALLEST_OF_COLOR:
        same_color = [o for o in ctx.all_objects if o.color == ctx.obj.color]
        return ctx.obj.size == min(o.size for o in same_color)

    if kind == PredicateKind.NEAREST_TO:
        target_color = pred.get("color")
        return _is_nearest_to_color(ctx.obj, ctx.grid, ctx.all_objects, target_color)

    if kind == PredicateKind.AND:
        left = pred.get("left")
        right = pred.get("right")
        return evaluate_predicate(left, ctx) and evaluate_predicate(right, ctx)

    if kind == PredicateKind.NOT:
        inner = pred.get("inner")
        return not evaluate_predicate(inner, ctx)

    return True


def _is_adjacent_to_color(obj: RawObject, grid: Grid, color: int) -> bool:
    """Check if any pixel of the object is adjacent to a pixel of the given color."""
    rows, cols = grid.shape
    for dr in range(obj.bbox_h):
        for dc in range(obj.bbox_w):
            if not obj.mask[dr, dc]:
                continue
            r, c = obj.row + dr, obj.col + dc
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + ddr, c + ddc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if int(grid[nr, nc]) == color:
                        return True
    return False


def _adjacent_contact_count(obj: RawObject, grid: Grid, color: int) -> int:
    """Count how many object boundary pixels are adjacent to the given color."""
    rows, cols = grid.shape
    count = 0
    for dr in range(obj.bbox_h):
        for dc in range(obj.bbox_w):
            if not obj.mask[dr, dc]:
                continue
            r, c = obj.row + dr, obj.col + dc
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + ddr, c + ddc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if int(grid[nr, nc]) == color:
                        count += 1
                        break  # count each object pixel once
    return count


def _touches_border(obj: RawObject, shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return obj.row == 0 or obj.col == 0 or obj.row + obj.bbox_h >= rows or obj.col + obj.bbox_w >= cols


def _inside_frame(obj: RawObject, framed_regions: tuple) -> bool:
    for fr in framed_regions:
        if (fr.row <= obj.row and fr.col <= obj.col
                and obj.row + obj.bbox_h <= fr.row + fr.height
                and obj.col + obj.bbox_w <= fr.col + fr.width):
            return True
    return False


def _is_nearest_to_color(obj: RawObject, grid: Grid, all_objects: list, color: int) -> bool:
    """Is this object the nearest to any pixel of the given color?"""
    color_positions = np.argwhere(grid == color)
    if len(color_positions) == 0:
        return False

    def _min_dist(o):
        min_d = float('inf')
        for dr in range(o.bbox_h):
            for dc in range(o.bbox_w):
                if o.mask[dr, dc]:
                    r, c = o.row + dr, o.col + dc
                    for cr, cc in color_positions:
                        d = abs(r - cr) + abs(c - cc)
                        min_d = min(min_d, d)
        return min_d

    my_dist = _min_dist(obj)
    for other in all_objects:
        if other is obj or other.color == obj.color:
            continue
        if _min_dist(other) < my_dist:
            return False
    return True


# ---------------------------------------------------------------------------
# Predicate induction from demos
# ---------------------------------------------------------------------------


def induce_predicates(
    demos: tuple,
    *,
    max_predicates: int = 20,
) -> list[Predicate]:
    """Induce candidate predicates that separate changed from unchanged objects.

    For each demo, identifies which objects changed (input != output at object
    pixels). Then finds predicates that are True for changed objects and False
    for unchanged across ALL demos.

    Returns predicates ranked by cross-demo consistency.
    """
    from aria.core.grid_perception import perceive_grid

    # Collect changed/unchanged classification per demo
    per_demo_labels: list[list[tuple[ObjectContext, bool]]] = []

    for d in demos:
        if d.input.shape != d.output.shape:
            continue
        bg = detect_bg(d.input)
        objs = extract_objects(d.input, bg)
        perception = perceive_grid(d.input)
        framed = perception.framed_regions

        labeled = []
        for obj in objs:
            if obj.color == bg:
                continue
            # Check if this object changed
            changed = False
            for dr in range(obj.bbox_h):
                for dc in range(obj.bbox_w):
                    if obj.mask[dr, dc]:
                        r, c = obj.row + dr, obj.col + dc
                        if int(d.input[r, c]) != int(d.output[r, c]):
                            changed = True
                            break
                if changed:
                    break

            ctx = ObjectContext(
                obj=obj, grid=d.input, bg=bg,
                all_objects=objs, grid_shape=d.input.shape,
                framed_regions=framed,
            )
            labeled.append((ctx, changed))

        if labeled:
            per_demo_labels.append(labeled)

    if not per_demo_labels:
        return []

    # Generate candidate predicates
    candidates = _generate_candidate_predicates(per_demo_labels)

    # Score each candidate by cross-demo separability
    scored: list[tuple[Predicate, float]] = []
    for pred in candidates:
        score = _score_predicate(pred, per_demo_labels)
        if score > 0:
            scored.append((pred, score))

    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored[:max_predicates]]


def _generate_candidate_predicates(
    per_demo_labels: list[list[tuple[ObjectContext, bool]]],
) -> list[Predicate]:
    """Generate atomic and simple composite predicate candidates."""
    candidates: list[Predicate] = []

    # Collect all colors and sizes seen
    all_colors: set[int] = set()
    all_sizes: set[int] = set()
    all_adj_colors: set[int] = set()

    for demo_labels in per_demo_labels:
        for ctx, _ in demo_labels:
            all_colors.add(ctx.obj.color)
            all_sizes.add(ctx.obj.size)
            for c in range(10):
                if c != ctx.bg and _is_adjacent_to_color(ctx.obj, ctx.grid, c):
                    all_adj_colors.add(c)

    # Atomic predicates
    for c in all_colors:
        candidates.append(P_has_color(c))
    candidates.append(P_is_singleton())
    candidates.append(P_not_singleton())
    candidates.append(P_touches_border())
    candidates.append(P_inside_frame())
    candidates.append(P_is_largest_of_color())
    candidates.append(P_is_smallest_of_color())
    candidates.append(P_size_above_mean())
    candidates.append(P_size_below_mean())

    for c in all_adj_colors:
        candidates.append(P_adjacent_to(c))
        candidates.append(P_adjacent_edge(c, 2))  # >=2 contact pixels

    for c in all_colors:
        candidates.append(P_nearest_to(c))

    # Size predicates
    for s in sorted(all_sizes):
        if s > 1:
            candidates.append(P_is_size(">", s - 1))
            candidates.append(P_is_size("<", s + 1))

    # Negations of the best atomics
    atomics = list(candidates)
    for p in atomics:
        candidates.append(P_not(p))

    # AND of top pairs: score atomics first, combine top-5
    atom_scored = []
    for p in atomics:
        s = _score_predicate(p, per_demo_labels)
        if s > 0.1:
            atom_scored.append((p, s))
    atom_scored.sort(key=lambda x: -x[1])
    top_atoms = [p for p, _ in atom_scored[:5]]
    for i, p1 in enumerate(top_atoms):
        for p2 in top_atoms[i+1:]:
            candidates.append(P_and(p1, p2))
        # Also And with negations
        for p2 in top_atoms:
            if p2 is not p1:
                candidates.append(P_and(p1, P_not(p2)))

    return candidates


def _score_predicate(
    pred: Predicate,
    per_demo_labels: list[list[tuple[ObjectContext, bool]]],
) -> float:
    """Score a predicate by how well it separates changed from unchanged.

    Uses precision-weighted F-score (Fbeta with beta=0.5) to penalize
    false positives more heavily. A predicate that fires on unchanged
    objects is worse than one that misses a changed object, because
    false positives cause wrong pixel edits while false negatives
    only leave pixels unchanged.

    Returns min across demos (not average) to require consistency.
    """
    if not per_demo_labels:
        return 0.0

    demo_scores = []
    for demo_labels in per_demo_labels:
        if not demo_labels:
            continue

        tp = fp = tn = fn = 0
        for ctx, is_changed in demo_labels:
            pred_true = evaluate_predicate(pred, ctx)
            if is_changed and pred_true:
                tp += 1
            elif is_changed and not pred_true:
                fn += 1
            elif not is_changed and pred_true:
                fp += 1
            else:
                tn += 1

        # Fbeta with beta=0.5 (precision-weighted)
        beta = 0.5
        if tp + fp == 0 or tp + fn == 0:
            demo_scores.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                demo_scores.append(0.0)
            else:
                fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                demo_scores.append(fbeta)

    if not demo_scores:
        return 0.0
    # Use min across demos — requires consistency in ALL demos
    return min(demo_scores)
