"""Grid-analysis-based parameter proposal for structural templates.

Instead of brute-force enumerating all parameter combinations,
analyze the input/output grid structure to propose likely parameter
values for each op family.

This is the "learned slot proposal" component — currently rule-based,
designed to be replaced by a trained model later.
"""

from __future__ import annotations

import numpy as np

from aria.types import DemoPair, Grid


def propose_tiling_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose parameters for render_tiled_input_pattern.

    Params: (row_repeat, col_repeat, odd_row_transform, odd_col_transform)

    Infer repeat counts from input/output size ratio.
    Infer transforms from structural analysis.
    """
    d = demos[0]
    ih, iw = d.input.shape
    oh, ow = d.output.shape

    candidates = []

    # Infer repeat counts from size ratio
    if oh > ih and ow > iw and ih > 0 and iw > 0:
        # Exact tiling
        if oh % ih == 0 and ow % iw == 0:
            rr = oh // ih
            cr = ow // iw
            # Try all transform combos
            for t1 in range(-1, 8):
                for t2 in range(-1, 8):
                    candidates.append((rr, cr, t1, t2))
        # Near-tiling (off by 1)
        for rr in range(1, min(6, oh // max(1, ih) + 2)):
            for cr in range(1, min(6, ow // max(1, iw) + 2)):
                if rr * ih >= oh - 1 and rr * ih <= oh + 1:
                    if cr * iw >= ow - 1 and cr * iw <= ow + 1:
                        for t1 in range(-1, 8):
                            for t2 in range(-1, 8):
                                candidates.append((rr, cr, t1, t2))
    elif (ih, iw) == (oh, ow):
        # Same shape — might be self-tiling or pattern fill
        for rr in range(1, 4):
            for cr in range(1, 4):
                for t1 in range(-1, 4):
                    for t2 in range(-1, 4):
                        candidates.append((rr, cr, t1, t2))

    return _dedupe(candidates)


def propose_color_map_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose color mapping parameters.

    For tasks where the transformation is a pixel-level color remapping.
    """
    d = demos[0]
    if d.input.shape != d.output.shape:
        return []

    # Infer the mapping from demo 0
    mapping = {}
    for r in range(d.input.shape[0]):
        for c in range(d.input.shape[1]):
            iv = int(d.input[r, c])
            ov = int(d.output[r, c])
            if iv in mapping:
                if mapping[iv] != ov:
                    return []  # not a consistent mapping
            mapping[iv] = ov

    if not mapping:
        return []

    # Check consistency across all demos
    for demo in demos[1:]:
        if demo.input.shape != demo.output.shape:
            return []
        for r in range(demo.input.shape[0]):
            for c in range(demo.input.shape[1]):
                iv = int(demo.input[r, c])
                ov = int(demo.output[r, c])
                if iv in mapping and mapping[iv] != ov:
                    return []  # inconsistent across demos

    # Return the mapping as a single candidate
    # Format: pairs of (from, to)
    return [tuple(v for pair in sorted(mapping.items()) for v in pair)]


def propose_geometric_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose geometric transform codes."""
    d = demos[0]
    if d.input.shape != d.output.shape and d.input.shape != d.output.shape[::-1]:
        # Only propose for same or transposed shapes
        return list(range(-1, 8))
    return list(range(-1, 8))


def propose_fill_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose fill color parameters from output analysis."""
    d = demos[0]
    if d.input.shape != d.output.shape:
        return []

    bg = int(np.bincount(d.input.ravel()).argmax())
    diff_mask = d.input != d.output

    if not np.any(diff_mask):
        return []

    # Colors that appear in output but not input at changed positions
    fill_colors = set(int(d.output[r, c]) for r, c in zip(*np.where(diff_mask)))
    return [(c,) for c in sorted(fill_colors)]


def propose_crop_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose crop/extraction parameters."""
    d = demos[0]
    if d.input.shape == d.output.shape:
        return []

    oh, ow = d.output.shape
    bg = int(np.bincount(d.input.ravel()).argmax())

    candidates = []
    # For crop_to_object: try each color
    in_colors = sorted(set(int(v) for v in np.unique(d.input)))
    for c in in_colors:
        candidates.append((c,))

    return candidates


def propose_relocate_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose match_rule × align for relocate_objects."""
    candidates = []
    for match_rule in range(7):
        for align in range(7):
            candidates.append((match_rule, align))
    return candidates


def propose_repair_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose repair parameters."""
    d = demos[0]
    candidates = []
    # repair_masked_region: transform codes
    for tc in range(8):
        candidates.append(("repair_masked_region", tc))
    # repair_periodic: axis × period
    for axis in (0, 1):
        for period in range(2, min(8, max(d.input.shape))):
            candidates.append(("repair_periodic", axis, period))
    # repair_framed_lines: axis × period
    for axis in (0, 1):
        for period in range(2, 8):
            candidates.append(("repair_framed_lines", axis, period))
    return candidates


def propose_derive_output_params(demos: tuple[DemoPair, ...]) -> list[tuple[int, ...]]:
    """Propose parameters for derive_output_from_input (7-param op)."""
    candidates = []
    for kind in range(4):
        for rel in range(4):
            for sel in range(4):
                for a0 in range(3):
                    for a1 in range(3):
                        for a2 in range(3):
                            candidates.append((kind, rel, sel, a0, a1, a2))
    return candidates


def _dedupe(candidates: list) -> list:
    seen = set()
    result = []
    for c in candidates:
        key = tuple(c) if isinstance(c, (list, tuple)) else (c,)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
