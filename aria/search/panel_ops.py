"""Panel algebra operations for separator/panel tasks.

Canonical operations on panel grids extracted via separator detection.
These are reusable execution primitives, not family wrappers.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from aria.search.sketch import SearchStep, SearchProgram


# ---------------------------------------------------------------------------
# Panel grid extraction
# ---------------------------------------------------------------------------

def extract_panel_grid(grid: np.ndarray):
    """Extract a grid of panels from uniform separators.

    Returns (panels, panel_shape, row_spans, col_spans) or None.
    panels: list of list of np.ndarray (row-major)
    """
    h, w = grid.shape

    # Find uniform rows (all same color)
    row_seps = []
    for r in range(h):
        vals = set(int(grid[r, c]) for c in range(w))
        if len(vals) == 1:
            row_seps.append(r)

    # Find uniform cols
    col_seps = []
    for c in range(w):
        vals = set(int(grid[r, c]) for r in range(h))
        if len(vals) == 1:
            col_seps.append(c)

    # Compute spans between separators
    row_spans = _spans_from_seps(row_seps, h)
    col_spans = _spans_from_seps(col_seps, w)

    if not row_spans or not col_spans:
        return None

    # Check: all panels same size?
    panel_heights = set(r1 - r0 + 1 for r0, r1 in row_spans)
    panel_widths = set(c1 - c0 + 1 for c0, c1 in col_spans)
    if len(panel_heights) != 1 or len(panel_widths) != 1:
        return None

    ph = next(iter(panel_heights))
    pw = next(iter(panel_widths))

    # Extract panels
    panels = []
    for r0, r1 in row_spans:
        row_panels = []
        for c0, c1 in col_spans:
            panel = grid[r0:r1 + 1, c0:c1 + 1].copy()
            row_panels.append(panel)
        panels.append(row_panels)

    return panels, (ph, pw), row_spans, col_spans


def _get_sep_color(grid, row_spans, col_spans):
    """Get the separator color (from uniform rows/cols)."""
    h, w = grid.shape
    # Check first row
    if row_spans and row_spans[0][0] > 0:
        return int(grid[0, 0])
    # Check first col
    if col_spans and col_spans[0][0] > 0:
        return int(grid[0, 0])
    return 0


def _spans_from_seps(seps, length):
    """Compute content spans between separator positions."""
    if not seps:
        return [(0, length - 1)]
    spans = []
    # Before first sep
    if seps[0] > 0:
        spans.append((0, seps[0] - 1))
    # Between seps
    for i in range(len(seps) - 1):
        s = seps[i] + 1
        e = seps[i + 1] - 1
        if s <= e:
            spans.append((s, e))
    # After last sep
    if seps[-1] < length - 1:
        spans.append((seps[-1] + 1, length - 1))
    return spans


# ---------------------------------------------------------------------------
# Panel operations
# ---------------------------------------------------------------------------

def panel_odd_select(grid: np.ndarray) -> np.ndarray | None:
    """Select the 'odd one out' panel from each row band.

    For each row of panels, find the panel that appears uniquely
    (all others are duplicates). Output = stack of selected panels.
    """
    result = extract_panel_grid(grid)
    if result is None:
        return None

    panels, (ph, pw), row_spans, col_spans = result
    if len(col_spans) < 3:
        return None  # need at least 3 columns to have an "odd one out"

    selected = []
    for row_panels in panels:
        # Hash each panel
        keys = [p.tobytes() for p in row_panels]
        counts = Counter(keys)
        # Find the unique one
        odd_indices = [i for i, k in enumerate(keys) if counts[k] == 1]
        if len(odd_indices) != 1:
            return None  # no unique panel or multiple unique
        selected.append(row_panels[odd_indices[0]])

    if not selected:
        return None

    # Reconstruct output with separator borders
    # Output width = panel width + 2 (for left/right sep borders)
    # Output height = full input height (preserves row separators)
    sep_color = _get_sep_color(grid, row_spans, col_spans)
    out_h = grid.shape[0]
    out_w = pw + 2  # panel + left border + right border

    output = np.full((out_h, out_w), sep_color, dtype=grid.dtype)
    for band_idx, (r0, r1) in enumerate(row_spans):
        if band_idx < len(selected):
            panel = selected[band_idx]
            output[r0:r1 + 1, 1:pw + 1] = panel

    return output


def panel_majority_select(grid: np.ndarray) -> np.ndarray | None:
    """Select the majority (most common) panel from each row band."""
    result = extract_panel_grid(grid)
    if result is None:
        return None

    panels, (ph, pw), row_spans, col_spans = result
    if len(col_spans) < 2:
        return None

    selected = []
    for row_panels in panels:
        keys = [p.tobytes() for p in row_panels]
        counts = Counter(keys)
        majority_key = counts.most_common(1)[0][0]
        majority_idx = next(i for i, k in enumerate(keys) if k == majority_key)
        selected.append(row_panels[majority_idx])

    if not selected:
        return None

    return np.vstack(selected)


# ---------------------------------------------------------------------------
# Search derivation
# ---------------------------------------------------------------------------

def extract_frame_panels(grid: np.ndarray):
    """Extract panels from rectangular framed objects.

    Finds objects that form rectangular borders (frames), extracts their
    interiors as panels. Returns list of (r0, c0, r1, c1, interior) tuples,
    or None if no frames found.

    This unifies with separator-based extraction: both produce
    (region_bounds, content) that panel ops can work on.
    """
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_find_frame

    facts = perceive(grid)
    frames = []
    for obj in facts.objects:
        if obj.size < 8:
            continue
        frame = prim_find_frame(obj, grid)
        if frame is None:
            continue
        r0, c0, r1, c1 = frame
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        if ih < 1 or iw < 1:
            continue
        # Skip frames that contain other frames (only repair innermost)
        has_sub = False
        for other in facts.objects:
            if other.oid == obj.oid or other.size < 8:
                continue
            other_frame = prim_find_frame(other, grid)
            if other_frame is None:
                continue
            fr0, fc0, fr1, fc1 = other_frame
            if fr0 > r0 and fc0 > c0 and fr1 < r1 and fc1 < c1:
                if fr1 - fr0 > 2 and fc1 - fc0 > 2:
                    has_sub = True
                    break
        if has_sub:
            continue
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        frames.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1, interior))

    return frames if frames else None


def panel_periodic_repair(grid: np.ndarray) -> np.ndarray | None:
    """Repair periodic patterns within each panel.

    Tries two panel extraction methods:
    1. Separator-defined panels (uniform row/col separators)
    2. Frame-defined panels (rectangular bordered objects)

    For each panel, find the best periodic tile via majority vote,
    replace defective cells. Returns repaired grid or None if no repairs needed.
    """
    from aria.guided.dsl import prim_repair_region

    repaired = grid.copy()
    any_changed = False

    # Try frame-based extraction first (more specific — avoids corrupting borders)
    frames = extract_frame_panels(grid)
    if frames:
        for r0, c0, r1, c1, interior in frames:
            fixed = prim_repair_region(interior)
            if fixed is not None and not np.array_equal(fixed, interior):
                repaired[r0:r1 + 1, c0:c1 + 1] = fixed
                any_changed = True

    # Fall back to separator-based extraction
    if not any_changed:
        sep_result = extract_panel_grid(grid)
        if sep_result is not None:
            panels, (ph, pw), row_spans, col_spans = sep_result
            for r0, r1 in row_spans:
                for c0, c1 in col_spans:
                    panel = grid[r0:r1 + 1, c0:c1 + 1]
                    fixed = prim_repair_region(panel)
                    if fixed is not None and not np.array_equal(fixed, panel):
                        repaired[r0:r1 + 1, c0:c1 + 1] = fixed
                        any_changed = True

    return repaired if any_changed else None


def derive_panel_algebra_programs(demos):
    """Derive panel algebra programs from demos."""
    results = []

    # Try odd-one-out selection
    prog = _try_panel_select(demos, 'odd')
    if prog:
        results.append(prog)

    # Try majority selection
    prog = _try_panel_select(demos, 'majority')
    if prog:
        results.append(prog)

    # Try panel periodic repair
    prog = _try_panel_repair(demos)
    if prog:
        results.append(prog)

    return results


def _try_panel_repair(demos):
    """Try panel periodic repair across all demos."""
    for inp, out in demos:
        result = panel_periodic_repair(inp)
        if result is None or not np.array_equal(result, out):
            return None

    prog = SearchProgram(
        steps=[SearchStep('panel_repair', {})],
        provenance='panel:periodic_repair',
    )
    if prog.verify(demos):
        return prog
    return None


def _try_panel_select(demos, mode):
    """Try panel selection (odd or majority) across all demos."""
    for inp, out in demos:
        if mode == 'odd':
            result = panel_odd_select(inp)
        else:
            result = panel_majority_select(inp)
        if result is None or not np.array_equal(result, out):
            return None

    prog = SearchProgram(
        steps=[SearchStep(f'panel_{mode}_select', {})],
        provenance=f'panel:{mode}_select',
    )
    if prog.verify(demos):
        return prog
    return None
