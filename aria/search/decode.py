"""Region-to-region decode / transfer operations.

Handles tasks where the input has multiple panels/regions and the output
is computed by decoding a rule from one region and applying it to another.

Common patterns:
- Legend decode: small region defines a color→color or shape→action mapping
- Template transfer: one region is a template, output applies it to another
- Region combine: output combines content from multiple regions
- Conditional render: one region controls how another is rendered
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.search.sketch import SearchStep, SearchProgram, StepSelect
from aria.search.ast import ASTNode, Op


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def derive_region_programs(demos: list[tuple[np.ndarray, np.ndarray]]) -> list[SearchProgram]:
    """Derive programs that involve region-to-region decode/transfer.

    Analyzes separator structure, assigns region roles, attempts
    decode/apply patterns. Returns verified SearchPrograms.
    """
    from aria.guided.perceive import perceive

    results = []

    # Perceive all demos
    all_facts = []
    for inp, _ in demos:
        all_facts.append(perceive(inp))

    # Only tasks with separators / multiple regions
    if not all_facts[0].separators:
        return []

    # Strategy 1: 2-panel combine (split → binary op → render)
    progs = _try_panel_combine(demos, all_facts)
    results.extend(progs)

    # Strategy 2: Region select (output = one of the regions, possibly transformed)
    progs = _try_region_select(demos, all_facts)
    results.extend(progs)

    # Strategy 3: Legend-conditioned recolor (small region defines a color map)
    progs = _try_legend_recolor(demos, all_facts)
    results.extend(progs)

    # Strategy 4: Template overlay (one region overlaid on another)
    progs = _try_region_overlay(demos, all_facts)
    results.extend(progs)

    # Strategy 5: Cell-wise conditional (per-cell rule from one region applied to another)
    progs = _try_cellwise_conditional(demos, all_facts)
    results.extend(progs)

    return results


# ---------------------------------------------------------------------------
# Strategy 1: Panel combine
# ---------------------------------------------------------------------------

def _try_panel_combine(demos, all_facts):
    """Output = binary_op(panel_A, panel_B) rendered with a color."""
    from aria.guided.dsl import prim_combine, prim_render_mask

    results = []

    for demo_idx, (inp, out) in enumerate(demos):
        if demo_idx > 0:
            break  # derive from demo 0, verify on all

        facts = all_facts[0]
        regions = _extract_regions(inp, facts)
        if len(regions) < 2:
            return []

        # Try all pairs of same-sized regions
        for i, (name_a, region_a) in enumerate(regions):
            for name_b, region_b in regions[i+1:]:
                if region_a.shape != region_b.shape:
                    continue
                if region_a.shape != out.shape:
                    continue

                bg = facts.bg
                for op in ('and', 'or', 'xor', 'diff', 'rdiff'):
                    mask = prim_combine(region_a, region_b, op, bg)
                    for color in range(10):
                        rendered = prim_render_mask(mask, color, bg)
                        if np.array_equal(rendered, out):
                            prog = SearchProgram(
                                steps=[SearchStep('split_combine_render',
                                                   {'op': op, 'color': color})],
                                provenance=f'decode:panel_combine_{op}_{name_a}_{name_b}',
                            )
                            if prog.verify(demos):
                                results.append(prog)
                                return results  # found one

    return results


# ---------------------------------------------------------------------------
# Strategy 2: Region select
# ---------------------------------------------------------------------------

def _try_region_select(demos, all_facts):
    """Output = one of the input regions (possibly transformed)."""
    results = []

    for demo_idx, (inp, out) in enumerate(demos):
        if demo_idx > 0:
            break

        facts = all_facts[0]
        regions = _extract_regions(inp, facts)

        for name, region in regions:
            # Direct match
            if np.array_equal(region, out):
                # Find which region this is and build a program
                prog = _make_region_select_prog(name, None, demos)
                if prog:
                    results.append(prog)
                    return results

            # Transformed match
            for xform, xfn in [('flip_h', lambda r: r[:, ::-1]),
                                ('flip_v', lambda r: r[::-1, :]),
                                ('flip_hv', lambda r: r[::-1, ::-1]),
                                ('rot90', lambda r: np.rot90(r)),
                                ('rot180', lambda r: np.rot90(r, 2))]:
                transformed = xfn(region)
                if transformed.shape == out.shape and np.array_equal(transformed, out):
                    prog = _make_region_select_prog(name, xform, demos)
                    if prog:
                        results.append(prog)
                        return results

    return results


# ---------------------------------------------------------------------------
# Strategy 3: Legend-conditioned recolor
# ---------------------------------------------------------------------------

def _try_legend_recolor(demos, all_facts):
    """Small region defines a color→color mapping applied to the larger region."""
    results = []

    facts0 = all_facts[0]
    inp0, out0 = demos[0]
    regions = _extract_regions(inp0, facts0)
    if len(regions) < 2:
        return []

    bg = facts0.bg

    # Find legend (smaller) and data (larger) regions
    sorted_regions = sorted(regions, key=lambda r: r[1].size)
    for legend_idx in range(len(sorted_regions)):
        legend_name, legend = sorted_regions[legend_idx]
        for data_idx in range(len(sorted_regions)):
            if data_idx == legend_idx:
                continue
            data_name, data = sorted_regions[data_idx]

            # Extract color mapping from legend
            # Pattern: legend has pairs of colors (old→new)
            color_map = _extract_color_map(legend, bg)
            if not color_map:
                continue

            # Apply mapping to data region
            mapped = data.copy()
            for r in range(mapped.shape[0]):
                for c in range(mapped.shape[1]):
                    v = int(mapped[r, c])
                    if v in color_map:
                        mapped[r, c] = color_map[v]

            if mapped.shape == out0.shape and np.array_equal(mapped, out0):
                # Verify across demos
                all_ok = True
                for i, (inp, out) in enumerate(demos[1:], 1):
                    facts_i = all_facts[i]
                    regions_i = _extract_regions(inp, facts_i)
                    if len(regions_i) < 2:
                        all_ok = False
                        break
                    sorted_i = sorted(regions_i, key=lambda r: r[1].size)
                    if legend_idx >= len(sorted_i) or data_idx >= len(sorted_i):
                        all_ok = False
                        break
                    legend_i = sorted_i[legend_idx][1]
                    data_i = sorted_i[data_idx][1]
                    cmap_i = _extract_color_map(legend_i, facts_i.bg)
                    if not cmap_i:
                        all_ok = False
                        break
                    mapped_i = data_i.copy()
                    for r in range(mapped_i.shape[0]):
                        for c in range(mapped_i.shape[1]):
                            v = int(mapped_i[r, c])
                            if v in cmap_i:
                                mapped_i[r, c] = cmap_i[v]
                    if not np.array_equal(mapped_i, out):
                        all_ok = False
                        break

                if all_ok:
                    prog = SearchProgram(
                        steps=[SearchStep('legend_recolor',
                                           {'legend_idx': legend_idx, 'data_idx': data_idx})],
                        provenance=f'decode:legend_recolor_{legend_name}→{data_name}',
                    )
                    results.append(prog)
                    return results

    return results


# ---------------------------------------------------------------------------
# Strategy 4: Region overlay
# ---------------------------------------------------------------------------

def _try_region_overlay(demos, all_facts):
    """Output = one region with another overlaid on top."""
    results = []

    facts0 = all_facts[0]
    inp0, out0 = demos[0]
    bg = facts0.bg
    regions = _extract_regions(inp0, facts0)
    if len(regions) < 2:
        return []

    for i, (name_a, a) in enumerate(regions):
        for j, (name_b, b) in enumerate(regions):
            if i == j:
                continue
            if a.shape != b.shape or a.shape != out0.shape:
                continue

            # Overlay: a as base, b's non-bg pixels on top
            overlay = a.copy()
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    if b[r, c] != bg:
                        overlay[r, c] = b[r, c]
            if np.array_equal(overlay, out0):
                prog = SearchProgram(
                    steps=[SearchStep('region_overlay',
                                       {'base': name_a, 'top': name_b})],
                    provenance=f'decode:overlay_{name_a}+{name_b}',
                )
                if _verify_overlay(demos, all_facts, i, j):
                    results.append(prog)
                    return results

    return results


def _verify_overlay(demos, all_facts, base_idx, top_idx):
    """Verify overlay pattern across all demos."""
    for i, (inp, out) in enumerate(demos):
        facts = all_facts[i]
        regions = _extract_regions(inp, facts)
        if base_idx >= len(regions) or top_idx >= len(regions):
            return False
        base = regions[base_idx][1]
        top = regions[top_idx][1]
        if base.shape != out.shape:
            return False
        bg = facts.bg
        overlay = base.copy()
        for r in range(base.shape[0]):
            for c in range(base.shape[1]):
                if top[r, c] != bg:
                    overlay[r, c] = top[r, c]
        if not np.array_equal(overlay, out):
            return False
    return True


# ---------------------------------------------------------------------------
# Strategy 5: Cell-wise conditional
# ---------------------------------------------------------------------------

def _try_cellwise_conditional(demos, all_facts):
    """Per-cell: if control_region[r,c] has color X, output[r,c] = f(data_region[r,c])."""
    results = []

    facts0 = all_facts[0]
    inp0, out0 = demos[0]
    bg = facts0.bg
    regions = _extract_regions(inp0, facts0)
    if len(regions) < 2:
        return []

    for ctrl_idx, (ctrl_name, ctrl) in enumerate(regions):
        for data_idx, (data_name, data) in enumerate(regions):
            if ctrl_idx == data_idx:
                continue
            if ctrl.shape != data.shape or data.shape != out0.shape:
                continue

            # Infer cell-wise rule: where ctrl != bg, output = ctrl color; else output = data
            rule_works = True
            for r in range(out0.shape[0]):
                for c in range(out0.shape[1]):
                    if ctrl[r, c] != bg:
                        if out0[r, c] != ctrl[r, c]:
                            rule_works = False
                            break
                    else:
                        if out0[r, c] != data[r, c]:
                            rule_works = False
                            break
                if not rule_works:
                    break

            if rule_works:
                # Verify across demos
                all_ok = True
                for i, (inp, out) in enumerate(demos[1:], 1):
                    facts_i = all_facts[i]
                    regions_i = _extract_regions(inp, facts_i)
                    if ctrl_idx >= len(regions_i) or data_idx >= len(regions_i):
                        all_ok = False
                        break
                    ci = regions_i[ctrl_idx][1]
                    di = regions_i[data_idx][1]
                    if ci.shape != out.shape or di.shape != out.shape:
                        all_ok = False
                        break
                    bg_i = facts_i.bg
                    for r in range(out.shape[0]):
                        for c in range(out.shape[1]):
                            expected = ci[r, c] if ci[r, c] != bg_i else di[r, c]
                            if out[r, c] != expected:
                                all_ok = False
                                break
                        if not all_ok:
                            break
                    if not all_ok:
                        break

                if all_ok:
                    prog = SearchProgram(
                        steps=[SearchStep('cellwise_conditional',
                                           {'ctrl': ctrl_name, 'data': data_name})],
                        provenance=f'decode:cellwise_{ctrl_name}→{data_name}',
                    )
                    results.append(prog)
                    return results

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_regions(grid, facts):
    """Extract named subgrid regions from a grid with separators."""
    regions = []
    if not facts.separators:
        return regions

    # Use perceived regions
    for i, r in enumerate(facts.regions):
        sub = grid[r.r0:r.r1 + 1, r.c0:r.c1 + 1].copy()
        regions.append((f'region_{i}', sub))

    return regions


def _extract_color_map(legend, bg):
    """Extract a color→color mapping from a small legend region.

    Looks for row-pairs or column-pairs where one color maps to another.
    """
    h, w = legend.shape
    color_map = {}

    # Pattern 1: horizontal pairs — each row has [color_a, color_b]
    if w == 2:
        for r in range(h):
            a, b = int(legend[r, 0]), int(legend[r, 1])
            if a != bg and b != bg and a != b:
                if a in color_map and color_map[a] != b:
                    return {}  # inconsistent
                color_map[a] = b
        if color_map:
            return color_map

    # Pattern 2: vertical pairs — each col has [color_a; color_b]
    if h == 2:
        for c in range(w):
            a, b = int(legend[0, c]), int(legend[1, c])
            if a != bg and b != bg and a != b:
                if a in color_map and color_map[a] != b:
                    return {}
                color_map[a] = b
        if color_map:
            return color_map

    # Pattern 3: adjacent non-bg color pairs in any arrangement
    # Scan for pairs where unique colors appear exactly twice
    from collections import Counter
    colors = [int(legend[r, c]) for r in range(h) for c in range(w) if legend[r, c] != bg]
    counts = Counter(colors)
    if len(counts) >= 2 and all(v == 1 for v in counts.values()):
        # Pairs of unique colors — need spatial arrangement to determine direction
        # For now, try left→right reading order
        non_bg = [(r, c, int(legend[r, c])) for r in range(h) for c in range(w) if legend[r, c] != bg]
        if len(non_bg) % 2 == 0:
            for k in range(0, len(non_bg), 2):
                if k + 1 < len(non_bg):
                    a = non_bg[k][2]
                    b = non_bg[k + 1][2]
                    if a != b:
                        color_map[a] = b
            if color_map:
                return color_map

    return {}


def _make_region_select_prog(region_name, xform, demos):
    """Build a SearchProgram for 'output = region [possibly transformed]'."""
    steps = []
    if xform:
        steps.append(SearchStep('crop_bbox', {}, StepSelect('largest')))  # placeholder
        steps.append(SearchStep(xform, {}))
    else:
        steps.append(SearchStep('crop_bbox', {}, StepSelect('largest')))

    prog = SearchProgram(steps=steps, provenance=f'decode:region_select_{region_name}')
    # Verify
    if prog.verify(demos):
        return prog
    return None
