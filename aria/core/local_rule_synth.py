"""Seeded local-rule synthesis.

Given a seed (decomposition/scope), synthesize a small shared symbolic
rule that transforms each pixel/cell inside the scope. The rule is:
  for each pixel in scope:
    if predicate(pixel, neighborhood, roles): apply action

Rules are shared across demos; role bindings resolve from each input.
Verified exactly against all demos.

DSL:
  Predicate = conjunction of atomic predicates over pixel features
  Action = set_bg | set_role_color | keep
  Role = bg | dominant_neighbor | boundary_color | frame_color
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Pixel feature extraction
# ---------------------------------------------------------------------------


def _pixel_features(grid: Grid, r: int, c: int, bg: int, obj_map: np.ndarray | None = None) -> dict[str, int]:
    """Compute local features for one pixel, including object-aware features."""
    h, w = grid.shape
    color = int(grid[r, c])
    is_bg = int(color == bg)

    # 4-neighbor counts
    n4_same = 0
    n4_bg = 0
    n4_nbg = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            nc_val = int(grid[nr, nc])
            if nc_val == color:
                n4_same += 1
            if nc_val == bg:
                n4_bg += 1
            else:
                n4_nbg += 1

    # 8-neighbor counts
    n8_same = 0
    n8_bg = 0
    n8_diff_nbg = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nv = int(grid[nr, nc])
                if nv == color:
                    n8_same += 1
                if nv == bg:
                    n8_bg += 1
                elif nv != color:
                    n8_diff_nbg += 1

    # Border adjacency
    at_border = int(r == 0 or r == h - 1 or c == 0 or c == w - 1)

    # Object-aware features (only if obj_map provided)
    n4_same_obj = 0
    on_obj_border = 0
    obj_size_bucket = 0  # 0=no obj, 1=small(<=4), 2=medium(5-20), 3=large(>20)

    if obj_map is not None:
        oi = int(obj_map[r, c])
        if oi >= 0:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(obj_map[nr, nc]) == oi:
                    n4_same_obj += 1
            on_obj_border = int(n4_same_obj < 4)
            # Object size
            obj_pixels = int(np.sum(obj_map == oi))
            if obj_pixels <= 4:
                obj_size_bucket = 1
            elif obj_pixels <= 20:
                obj_size_bucket = 2
            else:
                obj_size_bucket = 3

    return {
        "is_bg": is_bg,
        "n4_same": n4_same,
        "n4_bg": n4_bg,
        "n4_nbg": n4_nbg,
        "n8_same": n8_same,
        "n8_bg": n8_bg,
        "n8_diff_nbg": n8_diff_nbg,
        "at_border": at_border,
        "n4_same_obj": n4_same_obj,
        "on_obj_border": on_obj_border,
        "obj_size_bucket": obj_size_bucket,
    }


# ---------------------------------------------------------------------------
# Atomic predicates
# ---------------------------------------------------------------------------

# Each predicate is (feature_name, comparator, threshold)
# Comparators: "==" "!=" "<=" ">="

_ATOMIC_PREDICATES = [
    # Pixel-level
    ("is_bg", "==", 0),
    ("is_bg", "==", 1),
    ("n4_same", "<=", 0),
    ("n4_same", "<=", 1),
    ("n4_same", ">=", 2),
    ("n4_same", ">=", 3),
    ("n4_bg", ">=", 3),
    ("n4_bg", ">=", 2),
    ("n4_bg", "<=", 1),
    ("n4_bg", "==", 0),
    ("n8_same", "<=", 1),
    ("n8_same", "<=", 2),
    ("n8_same", ">=", 3),
    ("n8_same", ">=", 5),
    ("n8_bg", ">=", 5),
    ("n8_bg", ">=", 6),
    ("n8_bg", "==", 0),
    ("n8_diff_nbg", ">=", 1),
    ("n8_diff_nbg", ">=", 2),
    ("n8_diff_nbg", "==", 0),
    ("at_border", "==", 0),
    ("at_border", "==", 1),
    # Object-aware
    ("n4_same_obj", "<=", 1),
    ("n4_same_obj", "<=", 2),
    ("n4_same_obj", ">=", 3),
    ("on_obj_border", "==", 1),
    ("on_obj_border", "==", 0),
    ("obj_size_bucket", "==", 1),  # small object (<=4)
    ("obj_size_bucket", "==", 2),  # medium (5-20)
    ("obj_size_bucket", ">=", 2),  # medium or large
]


def _eval_predicate(features: dict, pred: tuple) -> bool:
    fname, cmp, thresh = pred
    val = features.get(fname, 0)
    if cmp == "==":
        return val == thresh
    elif cmp == "!=":
        return val != thresh
    elif cmp == "<=":
        return val <= thresh
    elif cmp == ">=":
        return val >= thresh
    return False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

# Actions: what to set the pixel to
# "set_bg": set to bg color
# "keep": don't change
# Roles are resolved per-demo from input

_ACTIONS = ["set_bg"]


# ---------------------------------------------------------------------------
# Rule: conjunction of predicates + action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalRule:
    predicates: tuple[tuple[str, str, int], ...]  # conjunction
    action: str
    description: str

    def matches(self, features: dict) -> bool:
        return all(_eval_predicate(features, p) for p in self.predicates)


def _dominant_neighbor_color(grid: Grid, r: int, c: int, bg: int) -> int:
    """Find the most common non-bg color in the 8-neighborhood."""
    h, w = grid.shape
    from collections import Counter
    colors = Counter()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(grid[nr, nc])
                if v != bg:
                    colors[v] += 1
    if colors:
        return colors.most_common(1)[0][0]
    return bg


def _build_obj_map(grid: Grid, bg: int) -> np.ndarray:
    """Build pixel→object_index map using 8-connected decomposition."""
    from aria.decomposition import decompose_objects
    h, w = grid.shape
    obj_map = np.full((h, w), -1, dtype=int)
    objs = decompose_objects(grid, bg, connectivity=8)
    for oi, obj in enumerate(objs.objects):
        for dr in range(obj.bbox_h):
            for dc in range(obj.bbox_w):
                if obj.mask[dr, dc]:
                    obj_map[obj.row + dr, obj.col + dc] = oi
    return obj_map


def apply_rule(grid: Grid, bg: int, rule: LocalRule, obj_map: np.ndarray | None = None) -> Grid:
    """Apply a local rule to every pixel in the grid."""
    h, w = grid.shape
    result = grid.copy()

    # Build obj_map if any predicate needs it
    needs_obj = any(p[0] in ("n4_same_obj", "on_obj_border", "obj_size_bucket")
                    for p in rule.predicates)
    if needs_obj and obj_map is None:
        obj_map = _build_obj_map(grid, bg)

    for r in range(h):
        for c in range(w):
            features = _pixel_features(grid, r, c, bg, obj_map)
            if rule.matches(features):
                if rule.action == "set_bg":
                    result[r, c] = bg
                elif rule.action == "set_dominant_neighbor":
                    result[r, c] = _dominant_neighbor_color(grid, r, c, bg)
                elif rule.action == "set_minority_neighbor":
                    result[r, c] = _minority_neighbor_color(grid, r, c, bg)
                elif rule.action.startswith("set_color_"):
                    color = int(rule.action.split("_")[-1])
                    result[r, c] = color

    return result


def _minority_neighbor_color(grid: Grid, r: int, c: int, bg: int) -> int:
    """Find the least common non-bg color in the 8-neighborhood."""
    h, w = grid.shape
    from collections import Counter
    colors = Counter()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(grid[nr, nc])
                if v != bg:
                    colors[v] += 1
    if colors:
        return colors.most_common()[-1][0]  # least common
    return bg


# ---------------------------------------------------------------------------
# Synthesis: enumerate rules, verify across demos
# ---------------------------------------------------------------------------


def _should_try_synthesis(demos: tuple[DemoPair, ...]) -> bool:
    """Gate: only try synthesis on tasks where it's likely to help."""
    if not demos:
        return False
    if not all(d.input.shape == d.output.shape for d in demos):
        return False
    # Skip only if grid is large AND nearly all pixels change
    for d in demos:
        diff = d.input != d.output
        if d.input.size > 100 and np.sum(diff) > d.input.size * 0.9:
            return False
    return True


def synthesize_local_rule(
    demos: tuple[DemoPair, ...],
    max_conjunction_size: int = 2,
    max_candidates: int = 500,
) -> LocalRule | None:
    """Synthesize a shared local rule that transforms input→output across all demos.

    Enumerates conjunctions of atomic predicates × actions.
    Returns the first rule that verifies exactly on all demos, or None.
    """
    from aria.decomposition import detect_bg

    if not _should_try_synthesis(demos):
        return None

    # Build candidate predicate conjunctions
    pred_candidates: list[tuple[tuple, ...]] = []
    for p in _ATOMIC_PREDICATES:
        pred_candidates.append((p,))
    if max_conjunction_size >= 2:
        for i, p1 in enumerate(_ATOMIC_PREDICATES):
            for p2 in _ATOMIC_PREDICATES[i + 1:]:
                pred_candidates.append((p1, p2))
    pred_candidates = pred_candidates[:max_candidates]

    # Determine action candidates from demo evidence
    actions = ["set_bg", "set_dominant_neighbor", "set_minority_neighbor"]
    # Add set_color_N for colors that appear in output diffs
    diff_colors: set[int] = set()
    for d in demos:
        if d.input.shape == d.output.shape:
            diff = d.input != d.output
            if np.any(diff):
                for r, c in zip(*np.where(diff)):
                    diff_colors.add(int(d.output[r, c]))
    for color in sorted(diff_colors):
        actions.append(f"set_color_{color}")

    # Pre-compute all features + targets for all demos
    demo_data = []
    for d in demos:
        bg = detect_bg(d.input)
        h, w = d.input.shape
        obj_map = _build_obj_map(d.input, bg)
        pixels = []
        for r in range(h):
            for c in range(w):
                features = _pixel_features(d.input, r, c, bg, obj_map)
                should_change = d.input[r, c] != d.output[r, c]
                out_val = int(d.output[r, c])
                in_val = int(d.input[r, c])
                pixels.append((features, should_change, in_val, out_val, bg, r, c))
        demo_data.append((pixels, d.input, bg))

    # Search: for each (preds, action), check if the rule is exact
    for action in actions:
        for preds in pred_candidates:
            all_match = True
            for pixels, grid, bg in demo_data:
                ok = True
                for features, should_change, in_val, out_val, bg_v, r, c in pixels:
                    matches = all(_eval_predicate(features, p) for p in preds)
                    if matches:
                        # Compute what the action would produce
                        if action == "set_bg":
                            new_val = bg_v
                        elif action == "set_dominant_neighbor":
                            new_val = _dominant_neighbor_color(grid, r, c, bg_v)
                        elif action == "set_minority_neighbor":
                            new_val = _minority_neighbor_color(grid, r, c, bg_v)
                        elif action.startswith("set_color_"):
                            new_val = int(action.split("_")[-1])
                        else:
                            new_val = in_val

                        if new_val != out_val:
                            ok = False
                            break
                    else:
                        # Rule doesn't match: pixel should stay unchanged
                        if in_val != out_val:
                            ok = False
                            break
                if not ok:
                    all_match = False
                    break

            if all_match:
                desc = " AND ".join(f"{f}{c}{t}" for f, c, t in preds) + f" → {action}"
                return LocalRule(predicates=preds, action=action, description=desc)

    return None


def synthesize_local_rule_scoped(
    demos: tuple[DemoPair, ...],
    scope: str = "full",
    max_conjunction_size: int = 2,
) -> LocalRule | None:
    """Synthesize a local rule constrained to a specific scope.

    scope="full": apply to entire grid
    scope="frame_interior": apply only inside detected framed regions
    scope="non_bg": apply only to non-bg pixels (predicate must include is_bg==0)
    """
    from aria.decomposition import detect_bg

    if scope == "non_bg":
        # Pre-filter: only consider predicates that start with is_bg==0
        rule = synthesize_local_rule(demos, max_conjunction_size)
        if rule and any(p == ("is_bg", "==", 0) for p in rule.predicates):
            return rule
        return None

    if scope == "frame_interior":
        # Apply rule only inside frame interiors
        from aria.core.grid_perception import perceive_grid

        candidates: list[tuple[tuple, ...]] = []
        for p in _ATOMIC_PREDICATES:
            candidates.append((p,))
        if max_conjunction_size >= 2:
            for i, p1 in enumerate(_ATOMIC_PREDICATES):
                for p2 in _ATOMIC_PREDICATES[i + 1:]:
                    candidates.append((p1, p2))
        candidates = candidates[:500]

        for preds in candidates:
            rule = LocalRule(predicates=preds, action="set_bg", description="")

            all_match = True
            for d in demos:
                bg = detect_bg(d.input)
                state = perceive_grid(d.input)
                result = d.input.copy()

                for fr in state.framed_regions:
                    r0, c0 = fr.row, fr.col
                    h, w = fr.height, fr.width
                    for r in range(r0, r0 + h):
                        for c in range(c0, c0 + w):
                            features = _pixel_features(d.input, r, c, bg)
                            if rule.matches(features):
                                result[r, c] = bg

                if not np.array_equal(result, d.output):
                    all_match = False
                    break

            if all_match:
                desc = " AND ".join(f"{f}{c}{t}" for f, c, t in preds) + " → set_bg [frame_interior]"
                return LocalRule(predicates=preds, action="set_bg", description=desc)

    return synthesize_local_rule(demos, max_conjunction_size)


# ---------------------------------------------------------------------------
# Seeded synthesis: use FitterSeed to constrain
# ---------------------------------------------------------------------------


def synthesize_from_seed(
    demos: tuple[DemoPair, ...],
    seed_decomposition: str = "",
    seed_scope: str = "full",
) -> LocalRule | None:
    """Synthesize a local rule constrained by a fitter seed."""
    if seed_scope == "frame_interior" or seed_decomposition == "frame_interior":
        return synthesize_local_rule_scoped(demos, scope="frame_interior")
    return synthesize_local_rule(demos)
