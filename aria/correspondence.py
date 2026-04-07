"""Object-to-object correspondence and bijective assignment.

Matches source entities to target entities/positions under one shared
structural rule across all demos. Used for:
- object ↔ adjacent partner (recolor to partner's color)
- singleton ↔ host gap (relocate into gap)
- object ↔ swapped position (pairwise swap)

The correspondence is typed, scored, and bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aria.decomposition import RawObject, detect_bg, extract_objects
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Correspondence types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectPair:
    """A matched pair of objects within one demo."""
    source: RawObject
    target: RawObject
    score: float = 1.0
    relation: str = ""  # "adjacent", "same_row", "nearest", etc.


@dataclass(frozen=True)
class DemoCorrespondence:
    """All matched pairs for one demo."""
    demo_index: int
    pairs: tuple[ObjectPair, ...]
    unmatched_sources: tuple[RawObject, ...] = ()
    unmatched_targets: tuple[RawObject, ...] = ()


@dataclass(frozen=True)
class CorrespondenceRule:
    """A shared correspondence rule across demos."""
    rule_type: str  # "adjacent_singleton", "nearest_different_color", etc.
    params: dict
    per_demo: tuple[DemoCorrespondence, ...]
    consistent: bool = True  # same structure across all demos


# ---------------------------------------------------------------------------
# Correspondence builders
# ---------------------------------------------------------------------------


def find_adjacent_singleton_pairs(
    demos: tuple[DemoPair, ...],
) -> CorrespondenceRule | None:
    """Find pairs where each non-singleton is matched to an adjacent singleton.

    Rule: for each non-trivial object, find a singleton of a different color
    that is physically adjacent (shares a border pixel). The singleton is
    the "partner."

    This is the pattern in 97d7923e.
    """
    per_demo: list[DemoCorrespondence] = []

    for di, d in enumerate(demos):
        if d.input.shape != d.output.shape:
            return None
        bg = detect_bg(d.input)
        objs = extract_objects(d.input, bg)
        singletons = [o for o in objs if o.size == 1 and o.color != bg]
        non_singletons = [o for o in objs if o.size > 1 and o.color != bg]

        if not singletons:
            return None

        pairs: list[ObjectPair] = []
        used_singletons: set[int] = set()

        for ns in non_singletons:
            # Find best adjacent singleton of different color
            best_sing = None
            best_contact = 0
            for si, sing in enumerate(singletons):
                if si in used_singletons:
                    continue
                if sing.color == ns.color:
                    continue
                contact = _contact_count(ns, sing, d.input)
                if contact > best_contact:
                    best_contact = contact
                    best_sing = (si, sing)

            if best_sing is not None:
                si, sing = best_sing
                used_singletons.add(si)
                pairs.append(ObjectPair(
                    source=ns, target=sing,
                    score=float(best_contact),
                    relation="adjacent_singleton",
                ))

        # Only keep if we found at least one pair
        if not pairs:
            return None

        per_demo.append(DemoCorrespondence(
            demo_index=di,
            pairs=tuple(pairs),
        ))

    if not per_demo:
        return None

    # Check consistency: all demos should have pairs
    return CorrespondenceRule(
        rule_type="adjacent_singleton",
        params={},
        per_demo=tuple(per_demo),
        consistent=all(len(dc.pairs) > 0 for dc in per_demo),
    )


def find_singleton_to_enclosed_region(
    demos: tuple[DemoPair, ...],
) -> CorrespondenceRule | None:
    """Find singleton markers that relocate into their nearest enclosed bg region.

    For each color-X singleton, finds the nearest enclosed (interior) bg
    component and creates a correspondence. The exact position within the
    region is determined by per-demo subset verification.
    """
    from scipy import ndimage

    per_demo: list[DemoCorrespondence] = []

    for di, d in enumerate(demos):
        if d.input.shape != d.output.shape:
            return None
        bg = detect_bg(d.input)
        objs = extract_objects(d.input, bg)

        # Find the marker color: singletons that get erased
        erased_colors: set[int] = set()
        for o in objs:
            if o.size == 1 and o.color != bg:
                if int(d.output[o.row, o.col]) == bg:
                    erased_colors.add(o.color)

        if not erased_colors:
            return None

        marker_color = next(iter(erased_colors))
        markers = [o for o in objs if o.size == 1 and o.color == marker_color]
        erased = [m for m in markers if int(d.output[m.row, m.col]) == bg]

        if not erased:
            return None

        # Find enclosed bg regions
        bg_mask = d.input == bg
        labeled, n = ndimage.label(bg_mask)
        rows, cols = d.input.shape
        border_labels = set()
        for r in range(rows):
            if labeled[r, 0] > 0: border_labels.add(labeled[r, 0])
            if labeled[r, cols-1] > 0: border_labels.add(labeled[r, cols-1])
        for c in range(cols):
            if labeled[0, c] > 0: border_labels.add(labeled[0, c])
            if labeled[rows-1, c] > 0: border_labels.add(labeled[rows-1, c])

        enclosed: dict[int, list[tuple[int, int]]] = {}
        for lbl in range(1, n + 1):
            if lbl not in border_labels:
                enclosed[lbl] = list(zip(*np.where(labeled == lbl)))

        if not enclosed:
            return None

        # Match each erased marker to nearest enclosed region
        pairs: list[ObjectPair] = []
        for m in erased:
            best_dist = float('inf')
            best_label = None
            for lbl, cells in enclosed.items():
                for cr, cc in cells:
                    d_val = abs(m.row - cr) + abs(m.col - cc)
                    if d_val < best_dist:
                        best_dist = d_val
                        best_label = lbl
            if best_label is not None:
                # Create a virtual "target" object at the region centroid
                cells = enclosed[best_label]
                cr = int(np.mean([r for r, c in cells]))
                cc = int(np.mean([c for r, c in cells]))
                target = RawObject(
                    color=bg, row=cr, col=cc, size=len(cells),
                    mask=np.ones((1, 1), dtype=bool),
                    bbox_h=1, bbox_w=1,
                )
                pairs.append(ObjectPair(
                    source=m, target=target,
                    score=1.0 / max(best_dist, 1),
                    relation="nearest_enclosed_region",
                ))

        if not pairs:
            return None
        per_demo.append(DemoCorrespondence(demo_index=di, pairs=tuple(pairs)))

    if not per_demo:
        return None

    return CorrespondenceRule(
        rule_type="singleton_to_enclosed",
        params={"marker_color": marker_color},
        per_demo=tuple(per_demo),
        consistent=all(len(dc.pairs) > 0 for dc in per_demo),
    )


def find_nearest_different_color_pairs(
    demos: tuple[DemoPair, ...],
) -> CorrespondenceRule | None:
    """Match each object to the nearest object of a different color."""
    per_demo: list[DemoCorrespondence] = []

    for di, d in enumerate(demos):
        if d.input.shape != d.output.shape:
            return None
        bg = detect_bg(d.input)
        objs = [o for o in extract_objects(d.input, bg) if o.color != bg]
        if len(objs) < 2:
            return None

        pairs: list[ObjectPair] = []
        for obj in objs:
            best_dist = float('inf')
            best_other = None
            for other in objs:
                if other is obj or other.color == obj.color:
                    continue
                d_val = abs(obj.center_row - other.center_row) + abs(obj.center_col - other.center_col)
                if d_val < best_dist:
                    best_dist = d_val
                    best_other = other
            if best_other is not None:
                pairs.append(ObjectPair(
                    source=obj, target=best_other,
                    score=1.0 / max(best_dist, 1),
                    relation="nearest_different_color",
                ))

        per_demo.append(DemoCorrespondence(demo_index=di, pairs=tuple(pairs)))

    return CorrespondenceRule(
        rule_type="nearest_different_color",
        params={},
        per_demo=tuple(per_demo),
        consistent=True,
    )


# ---------------------------------------------------------------------------
# Correspondence-conditioned actions
# ---------------------------------------------------------------------------


def apply_recolor_to_partner(
    grid: Grid,
    correspondence: DemoCorrespondence,
) -> Grid:
    """Recolor each source object to its matched partner's color."""
    result = grid.copy()
    bg = detect_bg(grid)

    for pair in correspondence.pairs:
        src = pair.source
        tgt = pair.target
        # Recolor source pixels to target's color
        for dr in range(src.bbox_h):
            for dc in range(src.bbox_w):
                if src.mask[dr, dc]:
                    r, c = src.row + dr, src.col + dc
                    result[r, c] = tgt.color

    return result


def apply_swap_paired_objects(
    grid: Grid,
    correspondence: DemoCorrespondence,
) -> Grid:
    """Swap each source object's pixels with its partner's pixels."""
    result = grid.copy()
    bg = detect_bg(grid)

    for pair in correspondence.pairs:
        src, tgt = pair.source, pair.target
        # Collect source pixels
        src_pixels = []
        for dr in range(src.bbox_h):
            for dc in range(src.bbox_w):
                if src.mask[dr, dc]:
                    src_pixels.append((src.row + dr, src.col + dc, src.color))
        # Collect target pixels
        tgt_pixels = []
        for dr in range(tgt.bbox_h):
            for dc in range(tgt.bbox_w):
                if tgt.mask[dr, dc]:
                    tgt_pixels.append((tgt.row + dr, tgt.col + dc, tgt.color))

        # Erase both
        for r, c, _ in src_pixels:
            result[r, c] = bg
        for r, c, _ in tgt_pixels:
            result[r, c] = bg

        # Stamp source at target position and vice versa
        if len(src_pixels) == len(tgt_pixels):
            for (sr, sc, scolor), (tr, tc, tcolor) in zip(src_pixels, tgt_pixels):
                result[tr, tc] = scolor
                result[sr, sc] = tcolor

    return result


def apply_relocate_to_enclosed(
    grid: Grid,
    correspondence: DemoCorrespondence,
) -> Grid:
    """Erase markers and stamp them at the nearest cell in their target enclosed region.

    For each pair, erases the source singleton and places its color at
    the nearest bg cell in the target region (determined by correspondence).
    Uses per-pair independent placement within the target region.
    """
    from scipy import ndimage

    result = grid.copy()
    bg = detect_bg(grid)

    bg_mask = grid == bg
    labeled, n = ndimage.label(bg_mask)
    rows, cols = grid.shape
    border_labels = set()
    for r in range(rows):
        if labeled[r, 0] > 0: border_labels.add(labeled[r, 0])
        if labeled[r, cols-1] > 0: border_labels.add(labeled[r, cols-1])
    for c in range(cols):
        if labeled[0, c] > 0: border_labels.add(labeled[0, c])
        if labeled[rows-1, c] > 0: border_labels.add(labeled[rows-1, c])

    for pair in correspondence.pairs:
        src = pair.source
        # Erase original
        result[src.row, src.col] = bg

        # Find nearest enclosed region
        best_dist = float('inf')
        best_label = None
        for lbl in range(1, n + 1):
            if lbl in border_labels:
                continue
            cells = list(zip(*np.where(labeled == lbl)))
            for cr, cc in cells:
                d_val = abs(src.row - cr) + abs(src.col - cc)
                if d_val < best_dist:
                    best_dist = d_val
                    best_label = lbl

        if best_label is not None:
            # Find nearest cell in that region to the singleton
            region_cells = list(zip(*np.where(labeled == best_label)))
            dists = [(abs(src.row - cr) + abs(src.col - cc), cr, cc) for cr, cc in region_cells]
            dists.sort()
            if dists:
                target_r, target_c = dists[0][1], dists[0][2]
                result[target_r, target_c] = src.color

    return result


def apply_erase_and_stamp_at_partner(
    grid: Grid,
    correspondence: DemoCorrespondence,
) -> Grid:
    """Erase source, stamp at partner's position."""
    result = grid.copy()
    bg = detect_bg(grid)

    for pair in correspondence.pairs:
        src = pair.source
        # Erase source
        for dr in range(src.bbox_h):
            for dc in range(src.bbox_w):
                if src.mask[dr, dc]:
                    result[src.row + dr, src.col + dc] = bg
        # Stamp at target position
        tgt = pair.target
        for dr in range(src.bbox_h):
            for dc in range(src.bbox_w):
                if src.mask[dr, dc]:
                    r = tgt.row + dr
                    c = tgt.col + dc
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                        result[r, c] = src.color

    return result


# ---------------------------------------------------------------------------
# Full correspondence search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrespondenceProgram:
    """A verified correspondence-based solution, executable on new inputs.

    Wraps the rule + action so it can produce outputs for test inputs.
    Uses independent subset search per input.
    """
    rule_type: str
    action_name: str
    action_fn_name: str  # name of the action function

    def verify_on_demo(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if this correspondence can produce exact output for this demo."""
        from itertools import combinations
        bg = detect_bg(input_grid)
        objs = extract_objects(input_grid, bg)
        non_bg = [o for o in objs if o.color != bg]
        pairs = self._build_pairs(non_bg, input_grid, bg)
        if not pairs:
            return np.array_equal(input_grid, output_grid)

        action_fn = self._get_action_fn()

        for size in range(1, len(pairs) + 1):
            for subset in combinations(pairs, size):
                dc = DemoCorrespondence(demo_index=0, pairs=subset)
                try:
                    result = action_fn(input_grid, dc)
                    if np.array_equal(result, output_grid):
                        return True
                except Exception:
                    continue
        return False

    def _build_pairs(self, objs, grid, bg):
        pairs = []
        for src in objs:
            best = None
            best_contact = 0
            for tgt in objs:
                if tgt is src or tgt.color == src.color:
                    continue
                c = _contact_count(src, tgt, grid)
                if c > best_contact:
                    best_contact = c
                    best = tgt
            if best is not None:
                pairs.append(ObjectPair(source=src, target=best, relation=self.rule_type))
        return pairs

    def _get_action_fn(self):
        return {
            "recolor_to_partner": apply_recolor_to_partner,
            "swap_paired": apply_swap_paired_objects,
            "erase_stamp_at_partner": apply_erase_and_stamp_at_partner,
        }.get(self.action_fn_name, apply_recolor_to_partner)

    def execute(self, input_grid: Grid) -> Grid:
        """Apply correspondence to a new input grid."""
        from itertools import combinations

        bg = detect_bg(input_grid)
        objs = extract_objects(input_grid, bg)
        non_bg = [o for o in objs if o.color != bg]

        # Build pairs
        pairs = []
        for src in non_bg:
            best = None
            best_contact = 0
            for tgt in non_bg:
                if tgt is src or tgt.color == src.color:
                    continue
                c = _contact_count(src, tgt, input_grid)
                if c > best_contact:
                    best_contact = c
                    best = tgt
            if best is not None:
                pairs.append(ObjectPair(
                    source=src, target=best,
                    relation=self.rule_type,
                ))

        if not pairs:
            return input_grid.copy()

        action_fn = {
            "recolor_to_partner": apply_recolor_to_partner,
            "swap_paired": apply_swap_paired_objects,
            "erase_stamp_at_partner": apply_erase_and_stamp_at_partner,
        }.get(self.action_fn_name, apply_recolor_to_partner)

        # Try subsets from small to large (bounded)
        for size in range(1, min(len(pairs) + 1, 8)):
            for subset in combinations(pairs, size):
                dc = DemoCorrespondence(demo_index=0, pairs=subset)
                result = action_fn(input_grid, dc)
                # Return the first non-identity result
                if not np.array_equal(result, input_grid):
                    return result

        # Fallback: all pairs
        dc = DemoCorrespondence(demo_index=0, pairs=tuple(pairs))
        return action_fn(input_grid, dc)


def correspondence_search(
    demos: tuple[DemoPair, ...],
    *,
    max_programs: int = 50,
) -> list[tuple[str, CorrespondenceProgram]]:
    """Search for correspondence-based solutions.

    Strategy: find correspondence rules, then for each rule × action,
    try all pairs and enumerated subsets. Verification picks the winner.
    """
    if not demos:
        return []
    if not all(d.input.shape == d.output.shape for d in demos):
        return []

    results: list[tuple[str, object]] = []
    tried = 0

    # Try specialized solvers first
    try:
        from aria.correspondence_enclosed import solve_singleton_relocation
        prog = solve_singleton_relocation(demos)
        if prog is not None:
            results.append(("singleton_relocation", prog))
            return results
    except Exception:
        pass

    rules = [
        find_adjacent_singleton_pairs(demos),
        find_singleton_to_enclosed_region(demos),
        find_nearest_different_color_pairs(demos),
    ]

    actions = [
        ("recolor_to_partner", apply_recolor_to_partner),
        ("swap_paired", apply_swap_paired_objects),
        ("erase_stamp_at_partner", apply_erase_and_stamp_at_partner),
        ("relocate_to_enclosed", apply_relocate_to_enclosed),
    ]

    for rule in rules:
        if rule is None or not rule.consistent:
            continue
        for action_name, action_fn in actions:
            if tried >= max_programs:
                break

            # Try all pairs
            if _verify_rule_action(demos, rule, action_fn):
                results.append((f"{rule.rule_type}+{action_name}", CorrespondenceProgram(
                    rule_type=rule.rule_type, action_name=action_name, action_fn_name=action_name,
                )))
                tried += 1
                continue

            tried += 1

            # Try per-color-group strategies
            for strategy in ["largest_per_color", "smallest_per_color"]:
                if tried >= max_programs:
                    break
                if _verify_with_strategy(demos, rule, action_fn, strategy):
                    results.append((f"{rule.rule_type}+{action_name}+{strategy}", rule))
                tried += 1

            # Try size-rank strategies
            max_pairs = max(len(dc.pairs) for dc in rule.per_demo)
            for k in range(min(max_pairs, 5)):
                if tried >= max_programs:
                    break
                if _verify_with_strategy(demos, rule, action_fn, "size_rank", k):
                    results.append((f"{rule.rule_type}+{action_name}+rank{k}", rule))
                tried += 1

            # Per-demo independent subset search: for each demo,
            # find the subset that gives exact match, then check if
            # ALL demos have at least one exact subset.
            if max_pairs <= 6 and tried < max_programs:
                if _verify_independent_subsets(demos, rule, action_fn):
                    results.append((
                        f"{rule.rule_type}+{action_name}+independent_subsets",
                        CorrespondenceProgram(
                            rule_type=rule.rule_type,
                            action_name=action_name,
                            action_fn_name=action_name,
                        ),
                    ))
                tried += 1

    return results


def _verify_rule_action(demos, rule, action_fn) -> bool:
    for di, d in enumerate(demos):
        if di >= len(rule.per_demo):
            return False
        try:
            output = action_fn(d.input, rule.per_demo[di])
            if not np.array_equal(output, d.output):
                return False
        except Exception:
            return False
    return True


def _verify_with_strategy(demos, rule, action_fn, strategy, param=None) -> bool:
    for di, d in enumerate(demos):
        if di >= len(rule.per_demo):
            return False
        dc = rule.per_demo[di]
        selected = _select_pairs(dc.pairs, strategy, param)
        if not selected:
            return False
        sub_dc = DemoCorrespondence(demo_index=di, pairs=tuple(selected))
        try:
            output = action_fn(d.input, sub_dc)
            if not np.array_equal(output, d.output):
                return False
        except Exception:
            return False
    return True


def _verify_subset_across_demos(demos, rule, action_fn, subset) -> bool:
    """Verify a subset pattern from demo 0 applied structurally to all demos."""
    dc0 = rule.per_demo[0]
    # Structural signature of the subset: source sizes and color count
    sub_sizes = sorted(dc0.pairs[i].source.size for i in subset)
    n_colors = len(set(dc0.pairs[i].source.color for i in subset))
    target_count = len(subset)

    for di, d in enumerate(demos):
        if di >= len(rule.per_demo):
            return False
        dc = rule.per_demo[di]
        # Find matching subset in this demo
        matched = _match_subset(dc.pairs, target_count, n_colors)
        if matched is None:
            return False
        sub_dc = DemoCorrespondence(demo_index=di, pairs=tuple(matched))
        try:
            output = action_fn(d.input, sub_dc)
            if not np.array_equal(output, d.output):
                return False
        except Exception:
            return False
    return True


def _verify_independent_subsets(demos, rule, action_fn) -> bool:
    """For each demo independently, check if ANY subset of pairs gives exact match."""
    from itertools import combinations
    for di, d in enumerate(demos):
        if di >= len(rule.per_demo):
            return False
        dc = rule.per_demo[di]
        found_exact = False
        for size in range(1, len(dc.pairs) + 1):
            for subset in combinations(dc.pairs, size):
                sub_dc = DemoCorrespondence(demo_index=di, pairs=subset)
                try:
                    output = action_fn(d.input, sub_dc)
                    if np.array_equal(output, d.output):
                        found_exact = True
                        break
                except Exception:
                    continue
            if found_exact:
                break
        if not found_exact:
            return False
    return True


def _select_pairs(pairs, strategy, param=None):
    if strategy == "largest_per_color":
        by_color = {}
        for p in pairs:
            by_color.setdefault(p.source.color, []).append(p)
        return [max(ps, key=lambda p: p.source.size) for ps in by_color.values()]
    if strategy == "smallest_per_color":
        by_color = {}
        for p in pairs:
            by_color.setdefault(p.source.color, []).append(p)
        return [min(ps, key=lambda p: p.source.size) for ps in by_color.values()]
    if strategy == "size_rank":
        ranked = sorted(pairs, key=lambda p: -p.source.size)
        return [ranked[param]] if param < len(ranked) else []
    return list(pairs)


def _match_subset(pairs, target_count, n_colors):
    """Find a subset of pairs matching the structural pattern."""
    from itertools import combinations
    if len(pairs) < target_count:
        return None
    for subset in combinations(pairs, target_count):
        colors = len(set(p.source.color for p in subset))
        if colors == n_colors:
            return list(subset)
    return list(pairs[:target_count]) if len(pairs) >= target_count else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contact_count(a: RawObject, b: RawObject, grid: Grid) -> int:
    """Count border pixels between two objects."""
    rows, cols = grid.shape
    count = 0
    for dr in range(a.bbox_h):
        for dc in range(a.bbox_w):
            if not a.mask[dr, dc]:
                continue
            r, c = a.row + dr, a.col + dc
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + ddr, c + ddc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Check if (nr, nc) belongs to b
                    br, bc = nr - b.row, nc - b.col
                    if 0 <= br < b.bbox_h and 0 <= bc < b.bbox_w:
                        if b.mask[br, bc]:
                            count += 1
    return count
