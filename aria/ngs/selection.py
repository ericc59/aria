"""Residual-target selection induction for the NGS solver.

Given:
- preservation factors for all train demos (with known residual masks)
- structural context of each demo

Induce a selection function that predicts the residual mask for a new input.

Approach: cross-demo structural alignment.
For each residual cell, extract a local structural descriptor from the
preserved input context. Find descriptors that are consistent across demos
for residual vs non-residual cells. Use those as the selection function.

This is NOT exhaustive predicate search. It works by:
1. Computing a fixed set of structural features for every cell
2. Finding feature values that are consistent with residual membership
3. Applying those feature values to new inputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.ngs.preservation import PreservationFactor, factor_preservation
from aria.types import Grid, DemoPair


# ---------------------------------------------------------------------------
# Selection function representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellFeatures:
    """Structural features of one cell for selection."""
    color: int              # cell's own color
    is_bg: bool             # is this cell bg?
    n4_colors: tuple[int, ...]  # sorted 4-neighbor colors
    n4_non_bg_count: int    # count of non-bg 4-neighbors
    enclosing_color: int    # color of the object whose bbox contains this cell (-1 if none)
    obj_size: int           # size of the object this cell belongs to (0 if bg)
    obj_color: int          # color of the object this cell belongs to (-1 if bg)
    obj_is_singleton: bool  # is this cell part of a singleton object?
    dist_to_border: int     # min manhattan distance to grid border
    same_color_neighbor_count: int  # how many 4-neighbors have the same color
    is_enclosed_bg: bool    # bg cell not reachable from border through bg


@dataclass
class SelectionRule:
    """A rule predicting which cells should be in the residual."""
    rule_type: str  # name of the selection method
    feature_constraints: dict[str, Any]  # feature -> required value(s)
    confidence: float
    description: str

    def predict_mask(self, inp: Grid, bg: int) -> np.ndarray:
        """Predict the residual mask for a new input."""
        rows, cols = inp.shape
        features = _compute_all_cell_features(inp, bg)
        mask = np.zeros((rows, cols), dtype=bool)

        for r in range(rows):
            for c in range(cols):
                if _matches_constraints(features[r][c], self.feature_constraints):
                    mask[r, c] = True
        return mask


@dataclass
class SelectionResult:
    """Result of selection induction."""
    rule: SelectionRule | None
    train_verified: bool
    train_mask_recall: float     # fraction of true residual cells found
    train_mask_precision: float  # fraction of predicted cells that are true residual
    description: str


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_all_cell_features(
    grid: Grid, bg: int,
) -> list[list[CellFeatures]]:
    """Compute structural features for every cell in the grid."""
    rows, cols = grid.shape
    objects = extract_objects(grid, bg, connectivity=4)

    # Build object membership map
    obj_map: dict[tuple[int, int], RawObject] = {}
    for obj in objects:
        for r in range(obj.bbox_h):
            for c in range(obj.bbox_w):
                if obj.mask[r, c]:
                    obj_map[(obj.row + r, obj.col + c)] = obj

    # Compute enclosed-bg mask: flood from border through bg
    from collections import deque
    reachable_bg = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == bg:
                reachable_bg[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable_bg[nr, nc] and grid[nr, nc] == bg:
                reachable_bg[nr, nc] = True
                q.append((nr, nc))
    enclosed_bg_mask = (grid == bg) & ~reachable_bg

    features = []
    for r in range(rows):
        row_feats = []
        for c in range(cols):
            color = int(grid[r, c])
            is_bg = color == bg

            # 4-neighbors
            n4 = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    n4.append(int(grid[nr, nc]))
                else:
                    n4.append(-1)  # border
            n4_non_bg = sum(1 for v in n4 if v != bg and v != -1)
            same_color_n = sum(1 for v in n4 if v == color and v != -1)

            # Object membership
            obj = obj_map.get((r, c))
            obj_size = obj.size if obj else 0
            obj_color = obj.color if obj else -1
            obj_singleton = obj.is_singleton if obj else False

            # Enclosing object
            enc_color = -1
            for o in objects:
                if (o.row <= r < o.row + o.bbox_h and
                    o.col <= c < o.col + o.bbox_w and
                    o is not obj):
                    enc_color = o.color
                    break

            dist_border = min(r, c, rows - 1 - r, cols - 1 - c)

            row_feats.append(CellFeatures(
                color=color,
                is_bg=is_bg,
                n4_colors=tuple(sorted(n4)),
                n4_non_bg_count=n4_non_bg,
                enclosing_color=enc_color,
                obj_size=obj_size,
                obj_color=obj_color,
                obj_is_singleton=obj_singleton,
                dist_to_border=dist_border,
                same_color_neighbor_count=same_color_n,
                is_enclosed_bg=bool(enclosed_bg_mask[r, c]),
            ))
        features.append(row_feats)
    return features


def _matches_constraints(
    feat: CellFeatures,
    constraints: dict[str, Any],
) -> bool:
    """Check if a cell's features match all constraints."""
    for key, val in constraints.items():
        actual = getattr(feat, key, None)
        if actual is None:
            return False
        if isinstance(val, set):
            if actual not in val:
                return False
        elif isinstance(val, tuple) and len(val) == 2 and val[0] == ">=":
            if actual < val[1]:
                return False
        elif isinstance(val, tuple) and len(val) == 2 and val[0] == "<=":
            if actual > val[1]:
                return False
        elif isinstance(val, tuple) and len(val) == 2 and val[0] == "!=":
            if actual == val[1]:
                return False
        else:
            if actual != val:
                return False
    return True


# ---------------------------------------------------------------------------
# Selection induction: cross-demo feature alignment
# ---------------------------------------------------------------------------

def induce_selection(
    factors: list[PreservationFactor],
    demos: tuple[DemoPair, ...],
) -> SelectionResult:
    """Induce a selection rule from train demos.

    Strategy: for each structural feature, check if there's a value
    (or small set of values) that's NECESSARY AND SUFFICIENT for
    residual membership across all demos.
    """
    if not factors or not demos:
        return SelectionResult(None, False, 0.0, 0.0, "no data")

    n_demos = len(demos)

    # Compute features and residual labels for each demo
    per_demo_data: list[tuple[list[list[CellFeatures]], np.ndarray]] = []
    for di, (pf, demo) in enumerate(zip(factors, demos)):
        bg = detect_bg(demo.input)
        feats = _compute_all_cell_features(demo.input, bg)
        residual = pf.residual_mask
        per_demo_data.append((feats, residual))

    # Phase 1: try single-feature rules
    candidates: list[SelectionRule] = []
    _try_color_selection(per_demo_data, demos, candidates)
    _try_size_selection(per_demo_data, demos, candidates)
    _try_singleton_selection(per_demo_data, demos, candidates)
    _try_neighbor_selection(per_demo_data, demos, candidates)
    _try_enclosed_bg_selection(per_demo_data, demos, candidates)

    # Phase 2: conjunctive rules — combine features that individually have high recall
    _try_conjunctive_selection(per_demo_data, demos, candidates)

    # Verify each candidate
    best: SelectionResult | None = None
    for rule in candidates:
        result = _verify_selection(rule, factors, demos)
        if result.train_verified:
            if best is None or rule.confidence > best.rule.confidence:
                best = result

    if best is not None:
        return best

    # Best partial: maximize F1 = 2*P*R / (P+R)
    if candidates:
        results = [_verify_selection(r, factors, demos) for r in candidates]
        def f1(r):
            p, rec = r.train_mask_precision, r.train_mask_recall
            return 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        results.sort(key=lambda r: -f1(r))
        return results[0]

    return SelectionResult(None, False, 0.0, 0.0, "no selection rule found")


def _try_color_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Check if residual cells share a common color across demos."""
    # Collect residual colors per demo
    per_demo_res_colors: list[set[int]] = []
    per_demo_nonres_colors: list[set[int]] = []

    for feats, residual in data:
        res_colors = set()
        nonres_colors = set()
        rows, cols = residual.shape
        for r in range(rows):
            for c in range(cols):
                if residual[r, c]:
                    res_colors.add(feats[r][c].color)
                else:
                    nonres_colors.add(feats[r][c].color)
        per_demo_res_colors.append(res_colors)
        per_demo_nonres_colors.append(nonres_colors)

    # Colors that are ONLY in residual (not in non-residual) in ALL demos
    exclusive_res = None
    for res_c, nonres_c in zip(per_demo_res_colors, per_demo_nonres_colors):
        exclusive = res_c - nonres_c
        if exclusive_res is None:
            exclusive_res = exclusive
        else:
            exclusive_res &= exclusive

    if exclusive_res:
        # Check union covers all residual
        covers_all = True
        for feats, residual in data:
            rows, cols = residual.shape
            for r in range(rows):
                for c in range(cols):
                    if residual[r, c] and feats[r][c].color not in exclusive_res:
                        covers_all = False
                        break
                if not covers_all:
                    break
            if not covers_all:
                break

        if covers_all:
            candidates.append(SelectionRule(
                "color_membership",
                {"color": exclusive_res},
                0.9,
                f"select cells with color in {sorted(exclusive_res)}",
            ))

    # Also try: cells with obj_color in specific set
    per_demo_res_obj_colors: list[set[int]] = []
    for feats, residual in data:
        oc = set()
        rows, cols = residual.shape
        for r in range(rows):
            for c in range(cols):
                if residual[r, c]:
                    oc.add(feats[r][c].obj_color)
        per_demo_res_obj_colors.append(oc)

    common_obj_colors = per_demo_res_obj_colors[0]
    for oc in per_demo_res_obj_colors[1:]:
        common_obj_colors &= oc

    if common_obj_colors and -1 not in common_obj_colors:
        candidates.append(SelectionRule(
            "obj_color_membership",
            {"obj_color": common_obj_colors},
            0.85,
            f"select cells belonging to objects with color in {sorted(common_obj_colors)}",
        ))


def _try_size_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Check if residual cells belong to objects above/below a size threshold."""
    for threshold_type in ["min", "max"]:
        per_demo_ok = True
        threshold_val = None

        for feats, residual in data:
            res_sizes = set()
            nonres_sizes = set()
            rows, cols = residual.shape
            for r in range(rows):
                for c in range(cols):
                    sz = feats[r][c].obj_size
                    if sz == 0:
                        continue
                    if residual[r, c]:
                        res_sizes.add(sz)
                    else:
                        nonres_sizes.add(sz)

            if not res_sizes:
                per_demo_ok = False
                break

            if threshold_type == "min":
                # All residual sizes > some threshold > all non-residual sizes
                min_res = min(res_sizes)
                max_nonres = max(nonres_sizes) if nonres_sizes else 0
                if min_res <= max_nonres:
                    per_demo_ok = False
                    break
                demo_thresh = max_nonres + 1
                if threshold_val is None:
                    threshold_val = demo_thresh
                else:
                    threshold_val = max(threshold_val, demo_thresh)
            else:
                max_res = max(res_sizes)
                min_nonres = min(nonres_sizes) if nonres_sizes else float('inf')
                if max_res >= min_nonres:
                    per_demo_ok = False
                    break
                demo_thresh = min_nonres - 1
                if threshold_val is None:
                    threshold_val = demo_thresh
                else:
                    threshold_val = min(threshold_val, demo_thresh)

        if per_demo_ok and threshold_val is not None:
            if threshold_type == "min":
                candidates.append(SelectionRule(
                    "obj_size_min",
                    {"obj_size": (">=", threshold_val)},
                    0.8,
                    f"select cells in objects with size >= {threshold_val}",
                ))
            else:
                candidates.append(SelectionRule(
                    "obj_size_max",
                    {"obj_size": ("<=", threshold_val)},
                    0.8,
                    f"select cells in objects with size <= {threshold_val}",
                ))


def _try_singleton_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Check if residual cells are all singletons or all non-singletons."""
    for target_val in [True, False]:
        all_ok = True
        for feats, residual in data:
            rows, cols = residual.shape
            for r in range(rows):
                for c in range(cols):
                    if residual[r, c] and feats[r][c].obj_size > 0:
                        if feats[r][c].obj_is_singleton != target_val:
                            all_ok = False
                            break
                if not all_ok:
                    break
            if not all_ok:
                break

        if all_ok:
            desc = "singletons" if target_val else "non-singletons"
            candidates.append(SelectionRule(
                f"singleton_{target_val}",
                {"obj_is_singleton": target_val},
                0.75,
                f"select {desc}",
            ))


def _try_neighbor_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Check if residual cells have a consistent non-bg neighbor count."""
    for target_count in range(1, 5):
        all_ok = True
        for feats, residual in data:
            rows, cols = residual.shape
            for r in range(rows):
                for c in range(cols):
                    if residual[r, c]:
                        if feats[r][c].n4_non_bg_count < target_count:
                            all_ok = False
                            break
                if not all_ok:
                    break
            if not all_ok:
                break

        if all_ok:
            candidates.append(SelectionRule(
                f"min_non_bg_neighbors_{target_count}",
                {"n4_non_bg_count": (">=", target_count)},
                0.7,
                f"select cells with >= {target_count} non-bg neighbors",
            ))


def _try_enclosed_bg_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Check if residual cells are bg cells that are enclosed (non-bg on all sides)."""
    all_ok = True
    for feats, residual in data:
        rows, cols = residual.shape
        for r in range(rows):
            for c in range(cols):
                if residual[r, c]:
                    f = feats[r][c]
                    if not f.is_bg:
                        all_ok = False
                        break
                    if f.n4_non_bg_count < 2:
                        all_ok = False
                        break
            if not all_ok:
                break
        if not all_ok:
            break

    if all_ok:
        candidates.append(SelectionRule(
            "enclosed_bg",
            {"is_bg": True, "n4_non_bg_count": (">=", 2)},
            0.8,
            "select bg cells with >= 2 non-bg neighbors",
        ))

    # Also try: enclosed bg (flood-fill based, more precise)
    all_ok2 = True
    for feats, residual in data:
        rows, cols = residual.shape
        for r in range(rows):
            for c in range(cols):
                if residual[r, c]:
                    if not feats[r][c].is_enclosed_bg:
                        all_ok2 = False
                        break
            if not all_ok2:
                break
        if not all_ok2:
            break

    if all_ok2:
        candidates.append(SelectionRule(
            "flood_enclosed_bg",
            {"is_enclosed_bg": True},
            0.92,
            "select bg cells enclosed by non-bg (flood fill)",
        ))


def _try_conjunctive_selection(
    data: list[tuple[list[list[CellFeatures]], np.ndarray]],
    demos: tuple[DemoPair, ...],
    candidates: list[SelectionRule],
) -> None:
    """Find conjunctions of 2-3 feature constraints that separate residual.

    Strategy: for each demo, profile the feature distributions of residual
    vs non-residual cells. Find feature values that are COMMON in residual
    and RARE in non-residual across ALL demos. Combine into conjunctions.
    """
    # Collect feature profiles
    FEATURE_NAMES = [
        "color", "is_bg", "n4_non_bg_count", "obj_size", "obj_color",
        "obj_is_singleton", "same_color_neighbor_count", "is_enclosed_bg",
    ]

    # For each feature, find values that appear in residual of EVERY demo
    # and the values that appear in non-residual
    per_feature_res: dict[str, list[set]] = {f: [] for f in FEATURE_NAMES}
    per_feature_nonres: dict[str, list[set]] = {f: [] for f in FEATURE_NAMES}

    for feats, residual in data:
        rows, cols = residual.shape
        for fname in FEATURE_NAMES:
            res_vals = set()
            nonres_vals = set()
            for r in range(rows):
                for c in range(cols):
                    val = getattr(feats[r][c], fname)
                    if residual[r, c]:
                        res_vals.add(val)
                    else:
                        nonres_vals.add(val)
            per_feature_res[fname].append(res_vals)
            per_feature_nonres[fname].append(nonres_vals)

    # Build atomic constraints: feature=value that covers all residual in all demos
    atomic_constraints: list[tuple[str, Any, float, float]] = []  # (fname, constraint, recall, precision_boost)

    for fname in FEATURE_NAMES:
        # Values in residual of ALL demos
        common_res = per_feature_res[fname][0].copy()
        for s in per_feature_res[fname][1:]:
            common_res &= s

        for val in common_res:
            # What fraction of residual has this value (recall)?
            recall_scores = []
            precision_scores = []
            for (feats, residual), nonres_vals in zip(data, per_feature_nonres[fname]):
                rows, cols = residual.shape
                n_res_with = 0
                n_res_total = 0
                n_nonres_with = 0
                n_nonres_total = 0
                for r in range(rows):
                    for c in range(cols):
                        v = getattr(feats[r][c], fname)
                        if residual[r, c]:
                            n_res_total += 1
                            if v == val:
                                n_res_with += 1
                        else:
                            n_nonres_total += 1
                            if v == val:
                                n_nonres_with += 1
                rec = n_res_with / n_res_total if n_res_total > 0 else 0
                prec = n_res_with / (n_res_with + n_nonres_with) if (n_res_with + n_nonres_with) > 0 else 0
                recall_scores.append(rec)
                precision_scores.append(prec)

            avg_recall = sum(recall_scores) / len(recall_scores)
            avg_precision = sum(precision_scores) / len(precision_scores)

            if avg_recall >= 0.8:  # high recall constraint
                atomic_constraints.append((fname, val, avg_recall, avg_precision))

    # Sort by precision descending (higher precision = more discriminative)
    atomic_constraints.sort(key=lambda x: -x[3])

    # Try pairs of constraints (conjunction)
    for i in range(len(atomic_constraints)):
        for j in range(i + 1, min(len(atomic_constraints), i + 10)):
            f1, v1, r1, p1 = atomic_constraints[i]
            f2, v2, r2, p2 = atomic_constraints[j]
            if f1 == f2:
                continue

            constraint = {f1: v1, f2: v2}
            candidates.append(SelectionRule(
                "conjunction",
                constraint,
                min(p1, p2),
                f"select where {f1}={v1} AND {f2}={v2}",
            ))

    # Try triples
    for i in range(min(len(atomic_constraints), 5)):
        for j in range(i + 1, min(len(atomic_constraints), 8)):
            for k in range(j + 1, min(len(atomic_constraints), 10)):
                f1, v1, _, _ = atomic_constraints[i]
                f2, v2, _, _ = atomic_constraints[j]
                f3, v3, _, _ = atomic_constraints[k]
                if len({f1, f2, f3}) < 3:
                    continue

                constraint = {f1: v1, f2: v2, f3: v3}
                candidates.append(SelectionRule(
                    "conjunction_3",
                    constraint,
                    0.7,
                    f"select where {f1}={v1} AND {f2}={v2} AND {f3}={v3}",
                ))

    # Also try: color AND singleton
    for (feats, residual) in data:
        rows, cols = residual.shape
        res_colors = set()
        for r in range(rows):
            for c in range(cols):
                if residual[r, c]:
                    res_colors.add(feats[r][c].color)

    # color + is_singleton combinations
    for color_val in range(10):
        for singleton_val in [True, False]:
            constraint = {"obj_color": color_val, "obj_is_singleton": singleton_val}
            candidates.append(SelectionRule(
                "color_singleton",
                constraint,
                0.8,
                f"select obj_color={color_val} AND singleton={singleton_val}",
            ))

    # is_bg + specific n4_non_bg_count
    for nbg in range(1, 5):
        constraint = {"is_bg": True, "n4_non_bg_count": nbg}
        candidates.append(SelectionRule(
            "bg_neighbor_exact",
            constraint,
            0.75,
            f"select bg cells with exactly {nbg} non-bg neighbors",
        ))

    # same_color_neighbor_count thresholds
    for scn in range(0, 4):
        constraint = {"same_color_neighbor_count": scn}
        candidates.append(SelectionRule(
            "same_color_neighbors",
            constraint,
            0.7,
            f"select cells with exactly {scn} same-color neighbors",
        ))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_selection(
    rule: SelectionRule,
    factors: list[PreservationFactor],
    demos: tuple[DemoPair, ...],
) -> SelectionResult:
    """Verify a selection rule on all demos."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_exact = True

    for pf, demo in zip(factors, demos):
        bg = detect_bg(demo.input)
        predicted = rule.predict_mask(demo.input, bg)
        actual = pf.residual_mask

        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not np.array_equal(predicted, actual):
            all_exact = False

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    return SelectionResult(
        rule=rule,
        train_verified=all_exact,
        train_mask_recall=recall,
        train_mask_precision=precision,
        description=rule.description,
    )


# ---------------------------------------------------------------------------
# End-to-end: selection + rewrite = verified output
# ---------------------------------------------------------------------------

def apply_selection_and_rewrite(
    selection: SelectionRule,
    rewrite_type: str,
    inp: Grid,
    bg: int,
) -> Grid | None:
    """Apply selection mask + rewrite rule to produce output."""
    mask = selection.predict_mask(inp, bg)
    if not np.any(mask):
        return inp.copy()

    predicted = inp.copy()

    if rewrite_type == "delete":
        predicted[mask] = bg
        return predicted

    if rewrite_type == "recolor_to_adjacent":
        # For each selected cell, recolor to the most common adjacent non-bg non-self color
        rows, cols = inp.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    adj_colors = []
                    cell_color = int(inp[r, c])
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            v = int(inp[nr, nc])
                            if v != bg and v != cell_color:
                                adj_colors.append(v)
                    if adj_colors:
                        predicted[r, c] = Counter(adj_colors).most_common(1)[0][0]
        return predicted

    if rewrite_type == "fill_with_adjacent":
        rows, cols = inp.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and inp[r, c] == bg:
                    adj_colors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            v = int(inp[nr, nc])
                            if v != bg:
                                adj_colors.append(v)
                    if adj_colors:
                        predicted[r, c] = Counter(adj_colors).most_common(1)[0][0]
        return predicted

    return None
