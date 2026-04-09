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

    # Strategy 6: Quadrant template decode (one quadrant is template, others mirror-apply)
    progs = _try_quadrant_template_decode(demos, all_facts)
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


# ---------------------------------------------------------------------------
# Strategy 6: Quadrant template decode
# ---------------------------------------------------------------------------

def _try_quadrant_template_decode(demos, all_facts):
    """One quadrant defines a spatial color template; others apply it mirrored.

    Requires exactly one row separator and one col separator creating 4 quadrants.
    One quadrant is the template (has the most complex pattern). Each other quadrant
    has a "seed" block of the template's central color. The template is mirrored
    (h for left/right, v for top/bottom, hv for diagonal) and applied to each seed,
    scaled by the seed's size.
    """
    results = []

    facts0 = all_facts[0]
    inp0, out0 = demos[0]
    bg = facts0.bg

    row_seps = sorted([s for s in facts0.separators if s.axis == 'row'], key=lambda s: s.index)
    col_seps = sorted([s for s in facts0.separators if s.axis == 'col'], key=lambda s: s.index)

    if len(row_seps) != 1 or len(col_seps) != 1:
        return []
    if inp0.shape != out0.shape:
        return []

    rs, cs = row_seps[0].index, col_seps[0].index
    h, w = inp0.shape

    # Extract 4 quadrants
    quads_rc = [(0, rs, 0, cs), (0, rs, cs + 1, w), (rs + 1, h, 0, cs), (rs + 1, h, cs + 1, w)]
    quads = [inp0[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]
    out_quads = [out0[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]

    # Require all same shape
    shapes = [q.shape for q in quads]
    if len(set(shapes)) != 1:
        return []

    qh, qw = shapes[0]

    # Find seed color: the non-bg color present in ALL 4 quadrants
    quad_colors = []
    for q in quads:
        colors = set(int(q[r, c]) for r in range(qh) for c in range(qw) if q[r, c] != bg)
        quad_colors.append(colors)

    common_colors = quad_colors[0]
    for cs_set in quad_colors[1:]:
        common_colors = common_colors & cs_set
    if not common_colors:
        return []

    # Find template quadrant: most distinct non-bg colors, and unchanged in output
    quad_complexity = []
    for i, q in enumerate(quads):
        quad_complexity.append((len(quad_colors[i]), i))
    quad_complexity.sort(reverse=True)

    for seed_color in common_colors:
        for _, tmpl_idx in quad_complexity:
            tmpl = quads[tmpl_idx]
            tmpl_colors = quad_colors[tmpl_idx]
            if len(tmpl_colors) < 2:
                continue

            # Template quadrant should be unchanged in output
            if not np.array_equal(tmpl, out_quads[tmpl_idx]):
                continue

            # Extract template pattern with known seed color
            tmpl_pattern = _extract_quadrant_pattern(tmpl, bg, seed_color)
            if tmpl_pattern is None:
                continue

            pat_colors, pat_bbox, pat_relative = tmpl_pattern
            central_color = seed_color

            # Try applying template to each other quadrant
            result = inp0.copy()
            all_ok = True

            for qi in range(4):
                if qi == tmpl_idx:
                    continue

                q = quads[qi]
                out_q = out_quads[qi]

                # Find seed block of central_color in this quadrant
                seed_cells = [(r, c) for r in range(qh) for c in range(qw) if q[r, c] == central_color]
                if not seed_cells:
                    all_ok = False
                    break

                # Determine mirror based on quadrant position relative to template
                tmpl_row = 0 if tmpl_idx < 2 else 1
                tmpl_col = tmpl_idx % 2
                qi_row = 0 if qi < 2 else 1
                qi_col = qi % 2
                flip_v = (qi_row != tmpl_row)
                flip_h = (qi_col != tmpl_col)

                # Apply mirrored template to seed
                applied = _apply_quadrant_template(q, pat_relative, seed_cells,
                                                    central_color, bg, flip_h, flip_v)
                if applied is None or not np.array_equal(applied, out_q):
                    all_ok = False
                    break

                # Write into result
                r0, r1, c0, c1 = quads_rc[qi]
                result[r0:r1, c0:c1] = applied

            if all_ok and np.array_equal(result, out0):
                # Verify on all demos
                verified = _verify_quadrant_template(demos, all_facts, tmpl_idx, seed_color)
                if verified:
                    prog = SearchProgram(
                        steps=[SearchStep('quadrant_template_decode',
                                           {'template_quadrant': tmpl_idx,
                                            'seed_color': seed_color})],
                        provenance=f'decode:quadrant_template_q{tmpl_idx}',
                    )
                    results.append(prog)
                    return results

    return results


def _extract_quadrant_pattern(tmpl, bg, seed_color=None):
    """Extract the spatial color pattern from a template quadrant.

    If seed_color is given, use it as the center. Otherwise heuristic.
    Returns dict with color roles and relative positions, or None.
    """
    h, w = tmpl.shape

    # Find non-bg bounding box
    non_bg = [(r, c, int(tmpl[r, c])) for r in range(h) for c in range(w) if tmpl[r, c] != bg]
    if len(non_bg) < 2:
        return None

    colors = set(v for _, _, v in non_bg)
    if len(colors) < 2:
        return None

    # Find contiguous color blocks
    from collections import defaultdict
    color_cells = defaultdict(list)
    for r, c, v in non_bg:
        color_cells[v].append((r, c))

    if seed_color is not None and seed_color in colors:
        best_center = seed_color
    else:
        # Heuristic: the color whose bounding box center is closest to the pattern center
        pattern_rows = [r for r, _, _ in non_bg]
        pattern_cols = [c for _, c, _ in non_bg]
        pr0, pr1 = min(pattern_rows), max(pattern_rows)
        pc0, pc1 = min(pattern_cols), max(pattern_cols)
        pat_center_r = (pr0 + pr1) / 2
        pat_center_c = (pc0 + pc1) / 2

        best_center = None
        best_score = 999
        for color in colors:
            cells = color_cells[color]
            cr = sum(r for r, c in cells) / len(cells)
            cc = sum(c for r, c in cells) / len(cells)
            dist = abs(cr - pat_center_r) + abs(cc - pat_center_c)
            if dist < best_score:
                best_score = dist
                best_center = color

    if best_center is None:
        return None

    pattern_rows = [r for r, _, _ in non_bg]
    pattern_cols = [c for _, c, _ in non_bg]
    pr0, pr1 = min(pattern_rows), max(pattern_rows)
    pc0, pc1 = min(pattern_cols), max(pattern_cols)

    center_cells = color_cells[best_center]
    center_r0 = min(r for r, c in center_cells)
    center_c0 = min(c for r, c in center_cells)
    center_r1 = max(r for r, c in center_cells)
    center_c1 = max(c for r, c in center_cells)
    center_h = center_r1 - center_r0 + 1
    center_w = center_c1 - center_c0 + 1

    # Center must form a filled rectangle (no multi-island merge)
    if len(center_cells) != center_h * center_w:
        return None

    # Build relative pattern: for each non-center color block, record
    # offset relative to center block (in units of center block size)
    relative = []
    for color in colors:
        if color == best_center:
            continue
        cells = color_cells[color]
        block_r0 = min(r for r, c in cells)
        block_c0 = min(c for r, c in cells)
        block_r1 = max(r for r, c in cells)
        block_c1 = max(c for r, c in cells)
        block_h = block_r1 - block_r0 + 1
        block_w = block_c1 - block_c0 + 1

        # Verify cells form a filled rectangle (no multi-island merge corruption)
        if len(cells) != block_h * block_w:
            return None

        # Offset in units of center block size
        dr = (block_r0 - center_r0) / center_h if center_h > 0 else 0
        dc = (block_c0 - center_c0) / center_w if center_w > 0 else 0

        relative.append({
            'color': color,
            'dr': dr,
            'dc': dc,
            'h_ratio': block_h / center_h if center_h > 0 else 1,
            'w_ratio': block_w / center_w if center_w > 0 else 1,
        })

    return {'center': best_center}, (pr0, pr1, pc0, pc1), relative


def _apply_quadrant_template(quad, pat_relative, seed_cells, central_color, bg, flip_h, flip_v):
    """Apply a mirrored template to a quadrant's seed block."""
    qh, qw = quad.shape
    result = quad.copy()

    # Find seed bounding box
    sr0 = min(r for r, c in seed_cells)
    sc0 = min(c for r, c in seed_cells)
    sr1 = max(r for r, c in seed_cells)
    sc1 = max(c for r, c in seed_cells)
    seed_h = sr1 - sr0 + 1
    seed_w = sc1 - sc0 + 1

    for block in pat_relative:
        dr = block['dr']
        dc = block['dc']
        color = block['color']
        h_ratio = block['h_ratio']
        w_ratio = block['w_ratio']

        # Mirror offsets
        if flip_v:
            dr = -dr - h_ratio + 1  # flip vertical offset
        if flip_h:
            dc = -dc - w_ratio + 1  # flip horizontal offset

        # Compute block position in seed coordinates
        br0 = sr0 + int(round(dr * seed_h))
        bc0 = sc0 + int(round(dc * seed_w))
        bh = max(1, int(round(h_ratio * seed_h)))
        bw = max(1, int(round(w_ratio * seed_w)))

        # Fill the block
        for r in range(bh):
            for c in range(bw):
                pr, pc = br0 + r, bc0 + c
                if 0 <= pr < qh and 0 <= pc < qw:
                    result[pr, pc] = color

    return result


def _verify_quadrant_template(demos, all_facts, tmpl_idx, seed_color=None):
    """Verify quadrant template decode across all demos."""
    for i, (inp, out) in enumerate(demos):
        facts = all_facts[i]
        bg = facts.bg

        row_seps = sorted([s for s in facts.separators if s.axis == 'row'], key=lambda s: s.index)
        col_seps = sorted([s for s in facts.separators if s.axis == 'col'], key=lambda s: s.index)
        if len(row_seps) != 1 or len(col_seps) != 1:
            return False
        if inp.shape != out.shape:
            return False

        rs, cs = row_seps[0].index, col_seps[0].index
        h, w = inp.shape

        quads_rc = [(0, rs, 0, cs), (0, rs, cs + 1, w), (rs + 1, h, 0, cs), (rs + 1, h, cs + 1, w)]
        quads = [inp[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]
        out_quads = [out[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]

        shapes = [q.shape for q in quads]
        if len(set(shapes)) != 1:
            return False

        tmpl = quads[tmpl_idx]
        if not np.array_equal(tmpl, out_quads[tmpl_idx]):
            return False

        tmpl_pattern = _extract_quadrant_pattern(tmpl, bg, seed_color)
        if tmpl_pattern is None:
            return False

        pat_colors, pat_bbox, pat_relative = tmpl_pattern
        central_color = seed_color if seed_color is not None else pat_colors.get('center')
        if central_color is None:
            return False

        result = inp.copy()
        tmpl_row = 0 if tmpl_idx < 2 else 1
        tmpl_col = tmpl_idx % 2

        for qi in range(4):
            if qi == tmpl_idx:
                continue

            q = quads[qi]
            out_q = out_quads[qi]
            seed_cells = [(r, c) for r in range(q.shape[0]) for c in range(q.shape[1]) if q[r, c] == central_color]
            if not seed_cells:
                return False

            qi_row = 0 if qi < 2 else 1
            qi_col = qi % 2
            flip_v = (qi_row != tmpl_row)
            flip_h = (qi_col != tmpl_col)

            applied = _apply_quadrant_template(q, pat_relative, seed_cells,
                                                central_color, bg, flip_h, flip_v)
            if applied is None or not np.array_equal(applied, out_q):
                return False

            r0, r1, c0, c1 = quads_rc[qi]
            result[r0:r1, c0:c1] = applied

        if not np.array_equal(result, out):
            return False

    return True
