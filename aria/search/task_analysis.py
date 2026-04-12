"""Lightweight task analysis for search gating.

Runs once per task, classifies structural properties that guide
strategy selection and output-dims prediction. Cheap and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aria.guided.perceive import perceive


@dataclass(frozen=True)
class TaskAnalysis:
    dims_change: bool
    same_dims: bool
    diff_type: str  # 'recolor_only'|'additive'|'subtractive'|'rearrange'|'mixed'
    changed_pixel_fraction: float
    new_colors: frozenset[int]
    removed_colors: frozenset[int]
    has_separators: bool
    has_panels: bool
    is_extraction: bool   # output is subgrid of input in all demos
    is_construction: bool  # output shares <10% with input


def analyze_task(demos: list[tuple[np.ndarray, np.ndarray]]) -> TaskAnalysis:
    """Analyze structural properties of a task from its demos."""
    dims_change = any(inp.shape != out.shape for inp, out in demos)
    same_dims = all(inp.shape == out.shape for inp, out in demos)

    # Classify diff type
    diff_type = _classify_diff_type(demos, same_dims)

    # Changed pixel fraction
    fracs = []
    for inp, out in demos:
        if inp.shape == out.shape:
            fracs.append(float(np.sum(inp != out)) / max(inp.size, 1))
        else:
            fracs.append(1.0)
    changed_pixel_fraction = sum(fracs) / len(fracs) if fracs else 0.0

    # Color sets
    new_colors = set()
    removed_colors = set()
    for inp, out in demos:
        in_c = set(int(x) for x in np.unique(inp))
        out_c = set(int(x) for x in np.unique(out))
        new_colors |= (out_c - in_c)
        removed_colors |= (in_c - out_c)

    # Separators and panels
    has_seps = False
    has_panels = False
    for inp, out in demos:
        facts = perceive(inp)
        if facts.separators:
            has_seps = True
        if facts.regions and len(facts.regions) >= 2:
            has_panels = True

    # Extraction: output is a subgrid of input
    is_extraction = _check_extraction(demos)

    # Construction: output shares <10% with input
    is_construction = _check_construction(demos)

    return TaskAnalysis(
        dims_change=dims_change,
        same_dims=same_dims,
        diff_type=diff_type,
        changed_pixel_fraction=changed_pixel_fraction,
        new_colors=frozenset(new_colors),
        removed_colors=frozenset(removed_colors),
        has_separators=has_seps,
        has_panels=has_panels,
        is_extraction=is_extraction,
        is_construction=is_construction,
    )


def _classify_diff_type(demos, same_dims):
    """Classify the type of input→output transformation."""
    if not same_dims:
        return 'mixed'

    recolor = True
    additive = True
    subtractive = True
    rearrange = True

    for inp, out in demos:
        if inp.shape != out.shape:
            return 'mixed'
        bg = 0  # assume 0 is bg
        in_nz = inp != bg
        out_nz = out != bg

        # Recolor: same non-bg mask, different colors
        if not np.array_equal(in_nz, out_nz):
            recolor = False

        # Additive: output has extra non-bg pixels, none removed
        if np.any(in_nz & ~out_nz):
            additive = False

        # Subtractive: output missing non-bg pixels, none added
        if np.any(~in_nz & out_nz):
            subtractive = False

        # Rearrange: same color multiset
        from collections import Counter
        if Counter(inp.flat) != Counter(out.flat):
            rearrange = False

    # Both rearrange and recolor can be true (same mask + same multiset).
    # Distinguish: if multiset is preserved, it's rearrange (values moved).
    # If multiset changed but mask didn't, it's recolor_only.
    if rearrange:
        return 'rearrange'
    if recolor:
        return 'recolor_only'
    if additive:
        return 'additive'
    if subtractive:
        return 'subtractive'
    return 'mixed'


def _check_extraction(demos):
    """Check if output is an exact subgrid of input in all demos."""
    for inp, out in demos:
        oh, ow = out.shape
        ih, iw = inp.shape
        if oh > ih or ow > iw:
            return False
        found = False
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                if np.array_equal(inp[r:r+oh, c:c+ow], out):
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _check_construction(demos):
    """Check if output shares <10% pixels/colors with input."""
    for inp, out in demos:
        in_colors = set(int(x) for x in np.unique(inp)) - {0}
        out_colors = set(int(x) for x in np.unique(out)) - {0}
        if not out_colors:
            continue
        overlap = len(in_colors & out_colors) / len(out_colors)
        if overlap >= 0.1:
            return False
    return True
