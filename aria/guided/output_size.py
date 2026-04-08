"""Output size inference via perception facts.

The perception layer computes dim_candidates: named values that could
be output dimensions. This module finds which candidate names consistently
match the actual output dimensions across all demos.

If (candidate_X, candidate_Y) == (output_rows, output_cols) for every demo,
that's the size rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.guided.perceive import perceive, GridFacts
from aria.types import Grid


@dataclass
class SizeRule:
    mode: str                     # "same", "static", "dynamic"
    row_source: str               # name of the dim_candidate for rows
    col_source: str               # name of the dim_candidate for cols
    static_shape: tuple[int, int] | None
    description: str

    def predict(self, facts: GridFacts) -> tuple[int, int]:
        if self.mode == "same":
            return (facts.rows, facts.cols)
        if self.mode == "static" and self.static_shape:
            return self.static_shape
        # Dynamic: look up from dim_candidates or constant
        r = self._resolve(self.row_source, facts)
        c = self._resolve(self.col_source, facts)
        return (r, c)

    def _resolve(self, source: str, facts: GridFacts) -> int:
        if source.startswith('const_'):
            return int(source[6:])
        # Direct name match
        if source in facts.dim_candidates:
            return facts.dim_candidates[source]
        # Semantic group fallback: find the LARGEST value in the same group
        # (since we typically want the dominant object's dimension)
        group = _semantic_group(source)
        group_values = [val for name, val in facts.dim_candidates.items()
                        if _semantic_group(name) == group]
        if group_values:
            return max(group_values)
        return facts.rows


def infer_output_size(demos: list[tuple[Grid, Grid]]) -> SizeRule | None:
    """Infer output size rule from demo pairs using perception facts."""
    in_shapes = [inp.shape for inp, _ in demos]
    out_shapes = [out.shape for _, out in demos]

    # Category 1: SAME
    if all(i == o for i, o in zip(in_shapes, out_shapes)):
        return SizeRule("same", "rows", "cols", None, "same as input")

    # Category 2: STATIC
    if len(set(out_shapes)) == 1:
        static = out_shapes[0]
        return SizeRule("static", "", "", static, f"always {static}")

    # Category 2b: one dim constant, other varies with input
    out_rows = [o[0] for o in out_shapes]
    out_cols = [o[1] for o in out_shapes]
    if len(set(out_rows)) == 1 and len(set(out_cols)) > 1:
        const_r = out_rows[0]
        facts_list_pre = [perceive(inp) for inp, _ in demos]
        common_names = set(facts_list_pre[0].dim_candidates.keys())
        for f in facts_list_pre[1:]:
            common_names &= set(f.dim_candidates.keys())
        col_matches = [name for name in common_names
                       if all(f.dim_candidates[name] == o[1] for f, o in zip(facts_list_pre, out_shapes))]
        if col_matches:
            col_name = _best_name(col_matches)
            return SizeRule("dynamic", f"const_{const_r}", col_name, None,
                            f"rows=const({const_r}), cols={col_name}")
        # Semantic group fallback
        col_group = _semantic_group_match(facts_list_pre, out_cols)
        if col_group:
            candidate = SizeRule("dynamic", f"const_{const_r}", col_group, None,
                                 f"rows=const({const_r}), cols~{col_group}")
            if all(candidate.predict(f) == o for f, o in zip(facts_list_pre, out_shapes)):
                return candidate

    if len(set(out_cols)) == 1 and len(set(out_rows)) > 1:
        const_c = out_cols[0]
        facts_list_pre = [perceive(inp) for inp, _ in demos]
        common_names = set(facts_list_pre[0].dim_candidates.keys())
        for f in facts_list_pre[1:]:
            common_names &= set(f.dim_candidates.keys())
        row_matches = [name for name in common_names
                       if all(f.dim_candidates[name] == o[0] for f, o in zip(facts_list_pre, out_shapes))]
        if row_matches:
            row_name = _best_name(row_matches)
            return SizeRule("dynamic", row_name, f"const_{const_c}", None,
                            f"rows={row_name}, cols=const({const_c})")
        # Semantic group fallback
        row_group = _semantic_group_match(facts_list_pre, out_rows)
        if row_group:
            candidate = SizeRule("dynamic", row_group, f"const_{const_c}", None,
                                 f"rows~{row_group}, cols=const({const_c})")
            if all(candidate.predict(f) == o for f, o in zip(facts_list_pre, out_shapes)):
                return candidate

    # Category 2c: SQUARE output — both dims equal, find one candidate for size
    if all(r == c for r, c in out_shapes) and len(set(out_shapes)) > 1:
        side_values = [o[0] for o in out_shapes]
        facts_list_sq = [perceive(inp) for inp, _ in demos]
        common_names = set(facts_list_sq[0].dim_candidates.keys())
        for f in facts_list_sq[1:]:
            common_names &= set(f.dim_candidates.keys())
        sq_candidates = []
        for name in common_names:
            if all(f.dim_candidates[name] == s for f, s in zip(facts_list_sq, side_values)):
                sq_candidates.append(name)
        if sq_candidates:
            # For square outputs, prefer orientation-invariant names
            # (min_dim, max_dim, n_objects, etc.) over row/col-specific ones
            best = _best_name_square(sq_candidates)
            return SizeRule("dynamic", best, best, None,
                            f"square: side={best}")
        # Try semantic group matching for square
        side_group = _semantic_group_match(facts_list_sq, side_values)
        if side_group:
            candidate = SizeRule("dynamic", side_group, side_group, None,
                                 f"square: side~{side_group}")
            if all(candidate.predict(f) == o for f, o in zip(facts_list_sq, out_shapes)):
                return candidate

    # Category 3: DYNAMIC — find matching dim_candidates across demos
    facts_list = [perceive(inp) for inp, _ in demos]

    # For each pair of candidate names (row_name, col_name):
    # check if facts[i].dim_candidates[row_name] == out_shapes[i][0]
    # AND facts[i].dim_candidates[col_name] == out_shapes[i][1]
    # for ALL demos

    # Collect all candidate names present in ALL demos
    common_names = set(facts_list[0].dim_candidates.keys())
    for f in facts_list[1:]:
        common_names &= set(f.dim_candidates.keys())

    # Find row candidates: names where value matches output rows in all demos
    row_candidates = []
    for name in common_names:
        if all(f.dim_candidates[name] == o[0] for f, o in zip(facts_list, out_shapes)):
            row_candidates.append(name)

    # Find col candidates: names where value matches output cols in all demos
    col_candidates = []
    for name in common_names:
        if all(f.dim_candidates[name] == o[1] for f, o in zip(facts_list, out_shapes)):
            col_candidates.append(name)

    if row_candidates and col_candidates:
        # Prefer the most descriptive / specific name
        row_name = _best_name(row_candidates)
        col_name = _best_name(col_candidates)
        return SizeRule("dynamic", row_name, col_name, None,
                        f"rows={row_name}, cols={col_name}")

    # Fallback: semantic-group matching with train verification.
    # Find groups, then verify the predicted sizes match ALL train outputs.
    row_group = _semantic_group_match(facts_list, [o[0] for o in out_shapes])
    col_group = _semantic_group_match(facts_list, [o[1] for o in out_shapes])

    candidates_to_try = []
    if row_group and col_group:
        candidates_to_try.append((row_group, col_group, f"rows~{row_group}, cols~{col_group}"))
    if row_candidates and col_group:
        candidates_to_try.append((_best_name(row_candidates), col_group,
                                   f"rows={_best_name(row_candidates)}, cols~{col_group}"))
    if row_group and col_candidates:
        candidates_to_try.append((row_group, _best_name(col_candidates),
                                   f"rows~{row_group}, cols={_best_name(col_candidates)}"))

    for r_src, c_src, desc in candidates_to_try:
        candidate = SizeRule("dynamic", r_src, c_src, None, desc)
        # Verify on ALL train demos
        verified = all(
            candidate.predict(f) == o
            for f, o in zip(facts_list, out_shapes)
        )
        if verified:
            return candidate

    return None


def debug_output_size(demos):
    """Debug helper: show all train-consistent candidates and why the winner was chosen.

    Not called during normal solving — use from scripts or REPL.
    """
    in_shapes = [inp.shape for inp, _ in demos]
    out_shapes = [out.shape for _, out in demos]

    if all(i == o for i, o in zip(in_shapes, out_shapes)):
        print("Output size: SAME as input")
        return

    if len(set(out_shapes)) == 1:
        print(f"Output size: STATIC {out_shapes[0]}")
        return

    facts_list = [perceive(inp) for inp, _ in demos]
    common_names = set(facts_list[0].dim_candidates.keys())
    for f in facts_list[1:]:
        common_names &= set(f.dim_candidates.keys())

    # Find all train-consistent candidates for rows and cols
    row_matches = sorted(
        [n for n in common_names
         if all(f.dim_candidates[n] == o[0] for f, o in zip(facts_list, out_shapes))],
        key=_name_score, reverse=True)
    col_matches = sorted(
        [n for n in common_names
         if all(f.dim_candidates[n] == o[1] for f, o in zip(facts_list, out_shapes))],
        key=_name_score, reverse=True)

    print(f"Row candidates ({len(row_matches)}):")
    for n in row_matches[:10]:
        tier = _candidate_tier(n)
        score = _name_score(n)
        vals = [f.dim_candidates[n] for f in facts_list]
        print(f"  T{tier} score={score:4d}  {n:40s}  vals={vals}")

    print(f"\nCol candidates ({len(col_matches)}):")
    for n in col_matches[:10]:
        tier = _candidate_tier(n)
        score = _name_score(n)
        vals = [f.dim_candidates[n] for f in facts_list]
        print(f"  T{tier} score={score:4d}  {n:40s}  vals={vals}")

    rule = infer_output_size(demos)
    if rule:
        print(f"\nChosen: {rule.description}")
        print(f"  row_source: {rule.row_source} (tier {_candidate_tier(rule.row_source) if not rule.row_source.startswith('const_') else 'const'})")
        print(f"  col_source: {rule.col_source} (tier {_candidate_tier(rule.col_source) if not rule.col_source.startswith('const_') else 'const'})")
    else:
        print("\nNo rule found")


def _value_match_across_demos(facts_list, target_values):
    """Find a candidate name that matches target values in MOST demos,
    allowing different names in different demos as long as the VALUE is right.

    Returns the most common matching name, or None.
    """
    from collections import Counter

    # For each demo, find all names that match the target value
    per_demo_names = []
    for f, val in zip(facts_list, target_values):
        matching = {name for name, v in f.dim_candidates.items() if v == val}
        if not matching:
            return None  # value doesn't exist in this demo at all
        per_demo_names.append(matching)

    # Find the name that appears in the most demos
    name_counts = Counter()
    for names in per_demo_names:
        for n in names:
            name_counts[n] += 1

    if not name_counts:
        return None

    # Best name = appears in most demos
    best_name, best_count = name_counts.most_common(1)[0]

    # Require it appears in ALL demos to avoid false positives
    if best_count == len(facts_list):
        return best_name

    # Not reliable enough
    return None


def _semantic_group(name):
    """Group candidate names by what structural concept they represent."""
    if 'unique_color' in name: return 'unique_color_obj'
    if 'color_' in name and 'bbox' in name: return 'color_bbox'
    if 'largest' in name or 'obj_rank0' in name: return 'largest_obj'
    if 'second' in name or 'obj_rank1' in name: return 'second_obj'
    if 'smallest' in name or 'min_obj' in name: return 'smallest_obj'
    if 'most_common_color' in name: return 'most_common_obj'
    if 'n_objects' in name or 'n_colors' in name or 'n_unique' in name: return 'count'
    if 'n_color_' in name: return 'per_color_count'
    if 'region' in name or 'cell' in name or 'sep' in name: return 'partition'
    if 'rows' in name or 'cols' in name: return 'grid_arith'
    if 'holes' in name: return 'holes'
    if 'interior' in name: return 'interior_obj'
    if 'nonsing' in name: return 'nonsing_obj'
    return 'other'


def _semantic_group_match(facts_list, target_values):
    """Find a semantic group that consistently contains a matching candidate
    across all demos."""
    from collections import Counter as _Counter

    per_demo_groups = []
    for f, val in zip(facts_list, target_values):
        matching = [n for n, v in f.dim_candidates.items() if v == val]
        if not matching:
            return None
        groups = set(_semantic_group(n) for n in matching)
        per_demo_groups.append(groups)

    # Find groups common to ALL demos
    common = per_demo_groups[0]
    for g in per_demo_groups[1:]:
        common &= g

    if not common:
        return None

    # Prefer specific groups over generic ones
    preference = ['unique_color_obj', 'color_bbox', 'largest_obj', 'second_obj',
                   'smallest_obj', 'most_common_obj', 'interior_obj', 'nonsing_obj',
                   'count', 'partition', 'holes', 'per_color_count']
    for pref in preference:
        if pref in common:
            # Return a representative name from this group
            # Use the name from demo 0 that belongs to this group
            matching = [n for n, v in facts_list[0].dim_candidates.items()
                        if v == target_values[0] and _semantic_group(n) == pref]
            if matching:
                return matching[0]

    return None


def _best_name(names: list[str]) -> str:
    """Pick the best candidate name.

    Prefer simpler, more specific names. Penalize complex combinations
    that are likely spurious matches.
    """
    return max(names, key=_name_score)


def _best_name_square(names: list[str]) -> str:
    """Pick the best candidate for square output dimensions.

    For square outputs (rows==cols), prefer orientation-invariant names
    over row/col-specific ones since the test input might have
    different orientation than training.
    """
    def score(name):
        s = _name_score(name)
        # Boost orientation-invariant names
        if name in ('min_dim', 'max_dim', 'n_objects', 'n_colors',
                     'max_obj_size', 'min_obj_size', 'n_unique_shapes',
                     'sqrt_max_obj', 'sqrt_total_pixels'):
            s += 25
        # Penalize row/col-specific names (could be wrong orientation)
        if name.startswith('rows') or name.startswith('cols'):
            s -= 10
        return s
    return max(names, key=score)


def _candidate_tier(name):
    """Classify a candidate name into a quality tier.

    Tier 1 (best):  structural/semantic — role-based, orientation-aware
    Tier 2:         rank-based proxy — ordinal position among objects
    Tier 3:         color-literal proxy — names that bake in a color value
    Tier 4:         index-literal proxy — names that bake in a region/object index
    Tier 5 (worst): arithmetic fallback — grid dim arithmetic, multiplicative combos
    """
    # --- Tier 1: structural/semantic ---
    if name in ('rows', 'cols', 'n_objects', 'n_colors', 'n_unique_shapes',
                'n_singletons', 'n_non_singletons', 'n_rectangles', 'n_large_rectangles',
                'min_dim', 'max_dim', 'n_separators', 'n_row_separators', 'n_col_separators',
                'n_row_regions', 'n_col_regions', 'n_row_cells', 'n_col_cells',
                'total_holes', 'max_holes', 'min_holes',
                'max_obj_size', 'min_obj_size', 'max_obj_height', 'max_obj_width',
                'min_obj_height', 'min_obj_width', 'max_color_count', 'min_color_count',
                'total_nonbg_pixels', 'n_bg_pixels', 'sum_obj_heights', 'sum_obj_widths'):
        return 1
    for prefix in ('largest_', 'smallest_', 'second_largest_', 'second_frame_',
                    'unique_color_obj_',
                    'only_unique_color_', 'largest_unique_color_',
                    'interior_obj_', 'biggest_interior_', 'smallest_nonsing_',
                    'most_common_color_obj_', 'first_cell_', 'max_cell_', 'min_cell_'):
        if name.startswith(prefix):
            return 1
    # Simple grid arithmetic with structural names
    if name in ('rows//2', 'cols//2', 'rows//3', 'cols//3',
                'rows-1', 'cols-1', 'rows+1', 'cols+1',
                'rows*2', 'cols*2', 'rows*3', 'cols*3',
                'rows+cols', 'rows-cols', 'rows+cols-1',
                'n_objects+1', 'rows-2', 'cols-2', 'rows+2', 'cols+2'):
        return 1

    # --- Tier 2: rank-based proxy ---
    if name.startswith('obj_rank') or name.startswith('holes_rank'):
        return 2

    # --- Tier 3: color-literal proxy ---
    if name.startswith('color_') and ('bbox' in name or 'total_px' in name):
        return 3
    if name.startswith('n_color_'):
        return 3
    # Multiplicative with per-color count
    if 'n_c' in name and any(c.isdigit() for c in name):
        return 3

    # --- Tier 4: index-literal proxy ---
    if name.startswith('region_'):
        return 4

    # --- Tier 5: arithmetic fallback ---
    return 5


def _name_score(name):
    """Score a candidate name. Higher = preferred.

    Tier-based: structural > rank > color-literal > index > arithmetic.
    Within a tier, prefer shorter names and semantic specificity.
    """
    tier = _candidate_tier(name)
    # Tier bonus: 100 points per tier of advantage
    s = (5 - tier) * 100

    # Within-tier: prefer shorter names (less specific = less brittle)
    s -= len(name)

    # Within-tier bonuses for especially good structural names
    if tier == 1:
        # Prefer role-derived (largest, unique_color) over grid arithmetic
        for prefix in ('largest_', 'unique_color_obj_', 'biggest_interior_',
                        'interior_obj_', 'smallest_nonsing_', 'most_common_color_obj_'):
            if name.startswith(prefix):
                s += 30
                break
        # Simple grid dims
        if name in ('rows', 'cols', 'n_objects', 'n_colors'):
            s += 20
        # Separator/cell derived
        if 'cell_' in name or 'region' in name:
            s += 15
        # min/max dim (orientation-invariant)
        if name in ('min_dim', 'max_dim'):
            s += 25

    # Penalize multiplicative combinations (more likely spurious)
    if '*' in name and tier >= 3:
        s -= 50

    return s
