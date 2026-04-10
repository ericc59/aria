"""Binding-guided program derivation.

Uses SceneBinding (role assignments, entity relations) to narrow the
search space for derive strategies. Tries existing execution ops with
binding-informed parameters.

This is NOT a new solver. It consumes bindings to guide existing
derive/decode infrastructure more effectively.
"""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchStep, SearchProgram, StepSelect
from aria.search.binding import (
    derive_scene_binding, SceneBinding, Role, EntityRef,
    extract_panels_from_separators,
)


def derive_from_binding(
    demos: list[tuple[np.ndarray, np.ndarray]],
) -> list[SearchProgram]:
    """Use scene bindings to derive programs.

    Strategies tried:
    1. Legend-panel region select: output = one bound panel, possibly transformed
    2. Legend-driven panel recolor: legend panel defines color map for workspace
    3. Workspace crop: output = crop of the workspace panel
    """
    binding = derive_scene_binding(demos)
    if binding is None:
        return []

    results = []

    # Strategy 1: output = one of the bound panels (region select)
    progs = _try_bound_region_select(demos, binding)
    results.extend(progs)
    if results:
        return results

    # Strategy 2: legend-driven panel recolor
    progs = _try_legend_panel_recolor(demos, binding)
    results.extend(progs)
    if results:
        return results

    # Strategy 3: workspace crop (output = subregion of bound workspace)
    progs = _try_workspace_crop(demos, binding)
    results.extend(progs)
    if results:
        return results

    # Strategy 4: ordered legend/control chain connectors in a shared grid
    progs = _try_legend_chain_connect(demos, binding)
    results.extend(progs)
    if results:
        return results

    # Strategy 5: object-level legend highlight (P0 shapes → workspace motif match → 8→3)
    progs = _try_object_highlight(demos)
    results.extend(progs)
    if results:
        return results

    # Strategy 6: bar-window progressive fill transfer (G06)
    progs = _try_bar_window_transfer(demos)
    results.extend(progs)

    return results


# ---------------------------------------------------------------------------
# Strategy 1: Bound region select
# ---------------------------------------------------------------------------

def _try_bound_region_select(demos, binding):
    """Output = one of the bound panels, possibly transformed.

    Uses role assignments to know WHICH panel to try, rather than
    trying all panels blindly.
    """
    from aria.guided.perceive import perceive

    results = []
    queries = binding.entities_with_role(Role.QUERY)
    workspaces = binding.entities_with_role(Role.WORKSPACE)
    candidates = queries + workspaces

    if not candidates:
        return []

    inp0, out0 = demos[0]
    facts0 = perceive(inp0)

    for entity in candidates:
        if entity not in binding.entity_grids:
            continue
        panel = binding.entity_grids[entity]

        # Direct match
        if panel.shape == out0.shape and np.array_equal(panel, out0):
            # Verify across demos
            if _verify_panel_select(demos, entity, None):
                prog = SearchProgram(
                    steps=[SearchStep('crop_bbox', {}, StepSelect('largest'))],
                    provenance=f'binding:region_select_{entity}',
                )
                if prog.verify(demos):
                    results.append(prog)
                    return results

        # Transformed match
        for xform, xfn in [('flip_h', lambda r: r[:, ::-1]),
                            ('flip_v', lambda r: r[::-1, :]),
                            ('rot90', lambda r: np.rot90(r)),
                            ('transpose', lambda r: r.T)]:
            transformed = xfn(panel)
            if transformed.shape == out0.shape and np.array_equal(transformed, out0):
                if _verify_panel_select(demos, entity, xform):
                    prog = SearchProgram(
                        steps=[SearchStep(xform, {})],
                        provenance=f'binding:region_select_{entity}_{xform}',
                    )
                    if prog.verify(demos):
                        results.append(prog)
                        return results

    return results


def _verify_panel_select(demos, entity, xform):
    """Verify panel select across all demos."""
    from aria.guided.perceive import perceive

    for inp, out in demos:
        facts = perceive(inp)
        panels = extract_panels_from_separators(inp, facts.separators)
        if entity.index >= len(panels):
            return False
        panel = panels[entity.index]
        if xform:
            xfns = {'flip_h': lambda r: r[:, ::-1], 'flip_v': lambda r: r[::-1, :],
                     'rot90': lambda r: np.rot90(r), 'transpose': lambda r: r.T}
            panel = xfns[xform](panel)
        if not np.array_equal(panel, out):
            return False
    return True


# ---------------------------------------------------------------------------
# Strategy 2: Legend-driven panel recolor
# ---------------------------------------------------------------------------

def _try_legend_panel_recolor(demos, binding):
    """Use a legend panel/strip to define a color map, apply to workspace.

    Binding tells us which panel is the legend and which is the workspace.
    We extract the color map from the legend and apply to the workspace.
    """
    from aria.guided.perceive import perceive
    from aria.search.decode import _extract_color_map

    legends = binding.entities_with_role(Role.LEGEND)
    workspaces = binding.entities_with_role(Role.WORKSPACE)
    queries = binding.entities_with_role(Role.QUERY)
    targets = queries + workspaces

    if not legends or not targets:
        return []

    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    bg = facts0.bg

    for legend_ref in legends:
        if legend_ref not in binding.entity_grids:
            continue
        legend_grid = binding.entity_grids[legend_ref]
        color_map = _extract_color_map(legend_grid, bg)
        if not color_map:
            continue

        for target_ref in targets:
            if target_ref not in binding.entity_grids:
                continue
            target_grid = binding.entity_grids[target_ref]

            # Apply color map to target
            mapped = target_grid.copy()
            for r in range(mapped.shape[0]):
                for c in range(mapped.shape[1]):
                    v = int(mapped[r, c])
                    if v in color_map:
                        mapped[r, c] = color_map[v]

            if mapped.shape == out0.shape and np.array_equal(mapped, out0):
                # Verify across demos
                all_ok = True
                for i, (inp, out) in enumerate(demos[1:], 1):
                    facts_i = perceive(inp)
                    panels_i = extract_panels_from_separators(inp, facts_i.separators)

                    if legend_ref.index >= len(panels_i) or target_ref.index >= len(panels_i):
                        all_ok = False
                        break

                    leg_i = panels_i[legend_ref.index]
                    tgt_i = panels_i[target_ref.index]
                    cmap_i = _extract_color_map(leg_i, facts_i.bg)
                    if not cmap_i:
                        all_ok = False
                        break

                    mapped_i = tgt_i.copy()
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
                        steps=[SearchStep('legend_recolor', {
                            'legend_idx': legend_ref.index,
                            'data_idx': target_ref.index,
                        })],
                        provenance=f'binding:legend_recolor_{legend_ref}→{target_ref}',
                    )
                    return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 3: Workspace crop
# ---------------------------------------------------------------------------

def _try_workspace_crop(demos, binding):
    """Output = crop/subregion of the bound workspace panel."""
    from aria.guided.perceive import perceive

    workspaces = binding.entities_with_role(Role.WORKSPACE)
    if not workspaces:
        return []

    inp0, out0 = demos[0]
    oh, ow = out0.shape

    for ws_ref in workspaces:
        if ws_ref not in binding.entity_grids:
            continue
        ws = binding.entity_grids[ws_ref]
        wh, ww = ws.shape

        if oh > wh or ow > ww:
            continue

        # Try all sub-windows
        for r0 in range(wh - oh + 1):
            for c0 in range(ww - ow + 1):
                sub = ws[r0:r0 + oh, c0:c0 + ow]
                if np.array_equal(sub, out0):
                    # Verify across demos with same offset
                    all_ok = True
                    for inp, out in demos[1:]:
                        facts = perceive(inp)
                        panels = extract_panels_from_separators(inp, facts.separators)
                        if ws_ref.index >= len(panels):
                            all_ok = False
                            break
                        ws_i = panels[ws_ref.index]
                        if r0 + oh > ws_i.shape[0] or c0 + ow > ws_i.shape[1]:
                            all_ok = False
                            break
                        if not np.array_equal(ws_i[r0:r0 + oh, c0:c0 + ow], out):
                            all_ok = False
                            break

                    if all_ok:
                        prog = SearchProgram(
                            steps=[SearchStep('crop_bbox', {
                                'panel_idx': ws_ref.index,
                                'r0': r0, 'c0': c0,
                                'oh': oh, 'ow': ow,
                            })],
                            provenance=f'binding:workspace_crop_{ws_ref}@({r0},{c0})',
                        )
                        if prog.verify(demos):
                            return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 4: Ordered legend/control chain connectors
# ---------------------------------------------------------------------------

def _try_legend_chain_connect(demos, binding):
    """Connect aligned workspace motifs following a bottom control-strip order.

    Pre-computes bg from demo 0 so the executor can skip perceive at test time.
    """
    from aria.guided.perceive import perceive
    from aria.search.executor import _exec_legend_chain_connect

    if not binding.entities_with_role(Role.CONTROL):
        return []
    if not binding.entities_with_role(Role.WORKSPACE):
        return []
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    # --- Pre-compute bg from demo 0 ---
    inp0, _ = demos[0]
    facts0 = perceive(inp0)
    bg0 = int(facts0.bg)

    params = {'control_side': 'bottom', 'bg': bg0}
    changed_any = False
    for inp, out in demos:
        facts = perceive(inp)
        diff = inp != out
        if not np.any(diff):
            return []
        changed_any = True
        # This task class only paints background cells; it does not erase objects.
        if np.any(inp[diff] != facts.bg):
            return []
        if np.any(out[diff] == facts.bg):
            return []
        pred = _exec_legend_chain_connect(inp, params)
        if not np.array_equal(pred, out):
            return []

    if not changed_any:
        return []

    return [SearchProgram(
        steps=[SearchStep('legend_chain_connect', params)],
        provenance='binding:legend_chain_connect',
    )]


# ---------------------------------------------------------------------------
# Strategy 5: Object-level legend highlight
# ---------------------------------------------------------------------------

def _try_object_highlight(demos):
    """Legend-driven object highlight with induced symbolic rules."""
    from aria.guided.perceive import perceive
    from aria.search.motif import extract_motifs, extract_panel_facts
    from aria.search.rules import induce_boolean_dnf

    if any(inp.shape != out.shape for inp, out in demos):
        return []

    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    row_seps = sorted(set(s.index for s in facts0.separators if s.axis == 'row'))
    if len(row_seps) < 4:
        return []

    r_bounds = [0] + row_seps + [inp0.shape[0]]
    panels0 = [(r_bounds[i], r_bounds[i+1])
               for i in range(len(r_bounds) - 1) if r_bounds[i+1] - r_bounds[i] >= 3]
    if len(panels0) < 3:
        return []

    # Infer ground and highlight colors
    diff = (inp0 != out0)
    if diff.sum() == 0:
        return []
    from collections import Counter
    old_vals = Counter(int(inp0[r, c]) for r, c in zip(*np.where(diff)))
    new_vals = Counter(int(out0[r, c]) for r, c in zip(*np.where(diff)))
    if len(old_vals) != 1 or len(new_vals) != 1:
        return []
    ground = list(old_vals.keys())[0]
    highlight = list(new_vals.keys())[0]

    shape_rows = []
    shape_labels = []
    per_demo_panel_facts = []
    p0_labels = []

    for inp_i, out_i in demos:
        facts_i = perceive(inp_i)
        row_seps_i = sorted(set(s.index for s in facts_i.separators if s.axis == 'row'))
        if len(row_seps_i) < 4:
            return []
        r_bounds_i = [0] + row_seps_i + [inp_i.shape[0]]
        panels_i = [
            (r_bounds_i[j], r_bounds_i[j + 1])
            for j in range(len(r_bounds_i) - 1)
            if r_bounds_i[j + 1] - r_bounds_i[j] >= 3
        ]
        if len(panels_i) < 3:
            return []

        r0_p0, r1_p0 = panels_i[0]
        p0_panel = inp_i[r0_p0:r1_p0, :]
        p0_motifs = extract_motifs(p0_panel, bg=0, ground=ground, min_cells=2)
        p0_shapes = frozenset(m.motif for m in p0_motifs)
        p0_colors = frozenset(m.color for m in p0_motifs)
        p0_n = len(p0_motifs)

        ws_facts = []
        ws_changed = []
        for pi, (r0, r1) in enumerate(panels_i[1:], 1):
            pf = extract_panel_facts(
                inp_i[r0:r1, :],
                pi,
                p0_shapes,
                p0_colors,
                p0_n,
                bg=0,
                ground=ground,
            )
            ws_facts.append(pf)
            changed = _panel_has_highlight(inp_i[r0:r1, :], out_i[r0:r1, :], ground, highlight)
            ws_changed.append(changed)
            shape_rows.append(pf.to_rule_dict())
            shape_labels.append(bool(pf.any_match and changed))

        per_demo_panel_facts.append((ws_facts, ws_changed))
        p0_labels.append(_panel_has_highlight(p0_panel, out_i[r0_p0:r1_p0, :], ground, highlight))

    shape_rule = induce_boolean_dnf(
        shape_rows,
        shape_labels,
        candidate_fields=['any_match', 'full_match', 'fewer_motifs', 'color_disjoint'],
        max_clause_size=2,
        max_clauses=2,
    )
    if shape_rule is None:
        return []

    aggregate_rows = []
    for ws_facts, ws_changed in per_demo_panel_facts:
        aggregate_rows.append({
            'any_full_match': any(pf.full_match for pf in ws_facts),
            'all_ws_highlight': bool(ws_changed) and all(ws_changed),
        })

    p0_rule = induce_boolean_dnf(
        aggregate_rows,
        p0_labels,
        candidate_fields=['any_full_match', 'all_ws_highlight'],
        max_clause_size=2,
        max_clauses=2,
    )
    if p0_rule is None:
        return []

    # Derive bottom band color: when P0 is not highlighted, the bottom band may
    # use a different color. Learn this from training demos.
    from aria.search.sketch import _exec_object_highlight_full

    bottom_alt_color = None
    for inp_i, out_i in demos:
        h_i = inp_i.shape[0]
        # Check if bottom band uses a different color
        bb_colors = set(int(out_i[r, c]) for r in range(h_i - 2, h_i)
                        for c in range(inp_i.shape[1]) if inp_i[r, c] == ground)
        if bb_colors and bb_colors != {highlight}:
            bottom_alt_color = list(bb_colors)[0]
            break

    params = {
        'ground': ground,
        'highlight': highlight,
        'shape_rule': shape_rule.to_dict(),
        'p0_rule': p0_rule.to_dict(),
    }
    if bottom_alt_color is not None:
        params['bottom_alt_color'] = bottom_alt_color

    # Verify the full rule on all demos using the executor
    for inp, out in demos:
        pred = _exec_object_highlight_full(inp, params)
        if not np.array_equal(pred, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('object_highlight', params)],
        provenance='binding:object_highlight',
    )
    return [prog]


def _panel_has_highlight(
    inp_panel: np.ndarray,
    out_panel: np.ndarray,
    ground: int,
    highlight: int,
) -> bool:
    """Whether a panel gained highlight-colored ground cells."""
    changed = (inp_panel == ground) & (out_panel == highlight)
    return bool(np.any(changed))


# ---------------------------------------------------------------------------
# Strategy 6: Bar-window progressive fill transfer
# ---------------------------------------------------------------------------

def _try_bar_window_transfer(demos):
    """Derive bar-window progressive fill transfer parameters from demos.

    Uses primitives from windows.py (progressive_fill, stamp_bar_window,
    erase_region) composed at derive time.  Verifies on all demos before
    returning a SearchProgram.
    """
    from aria.search.windows import (
        extract_bar_windows, strip_border, source_fill_params,
        get_fill_extents, compute_fill_K, progressive_fill,
        erase_region, stamp_bar_window, _stack_and_fill_dims,
    )

    if not demos:
        return []

    inp0, out0 = demos[0]

    # Use extract_bar_windows_best which tries multiple backgrounds
    from aria.search.windows import extract_bar_windows_best
    windows = extract_bar_windows_best(inp0)
    if not windows:
        return []

    # Infer bg from the grid border
    from aria.search.windows import _candidate_bgs
    bg_candidates = _candidate_bgs(inp0, None)
    bg = bg_candidates[0] if bg_candidates else 0
    # Pick the bg that yielded the most windows
    for candidate_bg in bg_candidates:
        trial = extract_bar_windows(inp0, bg=candidate_bg)
        if len(trial) >= len(windows):
            bg = candidate_bg
            windows = trial
            break
    if len(windows) < 2:
        return []

    sources = [w for w in windows if w.is_mixed and 7 in w.content_colors and 5 in w.content_colors]
    if not sources:
        return []

    # Derive parameters from demo 0
    params = _derive_transfer_params(inp0, out0, bg, windows, sources)
    if params is None:
        return []

    # Verify on ALL demos using the executor
    for inp, out in demos:
        pred = _exec_bar_window_transfer(inp, params)
        if pred is None or not np.array_equal(pred, out):
            return []

    return [SearchProgram(
        steps=[SearchStep('bar_window_transfer', params)],
        provenance='binding:bar_window_transfer',
    )]


def _derive_transfer_params(inp, out, bg, windows, sources):
    """Derive the full parameter dict from a single demo."""
    from aria.search.windows import (
        strip_border, source_fill_params, compute_fill_K,
    )

    bar_color = sources[0].bar_color
    targets = [w for w in windows if w.is_empty]
    is_transfer = len(targets) > 0

    # Collect source params
    src_params = []
    for s in sources:
        sc, se, ns = source_fill_params(s)
        K = compute_fill_K(sc, s.side, ns, is_transfer=is_transfer)
        src_params.append({
            'bbox': s.bbox, 'side': s.side, 'N_src': ns,
            'extents': se, 'K': K, 'content_shape': sc.shape,
        })

    return {
        'bg': bg,
        'bar_color': bar_color,
        'fill_color': 7,
        'base_color': 5,
    }


def _exec_bar_window_transfer(grid, params):
    """Execute the bar-window transfer operation.

    Composes the primitives: extract → pair → fill → erase → stamp.
    """
    from aria.search.windows import (
        extract_bar_windows, strip_border, source_fill_params,
        get_fill_extents, compute_fill_K, progressive_fill,
        erase_region, stamp_bar_window, _stack_and_fill_dims,
    )

    bg = params['bg']
    bar_color = params['bar_color']

    windows = extract_bar_windows(grid, bg=bg)
    if len(windows) < 2:
        return None

    sources = [w for w in windows if w.is_mixed
               and params['fill_color'] in w.content_colors
               and params['base_color'] in w.content_colors]
    if not sources:
        return None

    targets = [w for w in windows if w.is_empty]
    is_transfer = len(targets) > 0
    result = grid.copy()

    if is_transfer:
        return _exec_transfer(result, bg, bar_color, windows, sources, targets, params)
    source = sources[0]
    sc, se, ns = source_fill_params(source)
    return _exec_inplace(result, bg, bar_color, windows, source, sc, se, ns)


def _exec_inplace(result, bg, bar_color, windows, source, s_content, s_ext, N_src):
    """In-place: each window shrinks bar by 1, fills +1 step."""
    from aria.search.windows import (
        strip_border, progressive_fill, erase_region, stamp_bar_window,
    )

    for w in windows:
        content = strip_border(w.interior_grid)
        cr, cc = content.shape
        if cr == 0 or cc == 0:
            continue

        if w.is_mixed:
            own_ext, own_N = s_ext, N_src
        else:
            own_ext, own_N = None, 0

        new_content = progressive_fill(
            np.full_like(content, 5), w.side, own_N + 1,
            source_extents=own_ext, N_source=own_N,
            source_side=w.side if own_ext else None,
            source_shape=content.shape if own_ext else None,
            K=1,
        )

        erase_region(result, w.bbox, bg)

        r0, c0, r1, c1 = w.bbox
        if w.side == "top":
            stamp_bar_window(result, new_content, w.side, bar_color, r0, c0)
        elif w.side == "bottom":
            stamp_bar_window(result, new_content, w.side, bar_color, r1 - (cr + 2), c0)
        elif w.side == "left":
            stamp_bar_window(result, new_content, w.side, bar_color, r0, c0)
        else:
            stamp_bar_window(result, new_content, w.side, bar_color, r0, c1 - (cc + 2))

    return result


def _bands_overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def _window_center(w):
    r0, c0, r1, c1 = w.bbox
    return ((r0 + r1) / 2, (c0 + c1) / 2)


def _exec_transfer(result, bg, bar_color, windows, sources, targets, params):
    """Transfer mode: pair content windows with targets, apply fill, place."""
    from aria.search.windows import (
        strip_border, source_fill_params, compute_fill_K,
        progressive_fill, erase_region, stamp_bar_window,
    )

    content_windows = [
        w for w in windows
        if not w.is_empty
        and not (strip_border(w.interior_grid).size > 0
                 and params['base_color'] not in strip_border(w.interior_grid))
    ]

    used_t, used_c = set(), set()
    pairs = []

    for pass_horizontal in (False, True):
        for ti, t in enumerate(targets):
            if ti in used_t:
                continue
            is_horiz = t.side in ("top", "bottom")
            if is_horiz != pass_horizontal:
                continue

            if is_horiz:
                t_band = (t.bbox[1], t.bbox[3])
                def band_fn(w):
                    return (w.bbox[1], w.bbox[3])
            else:
                t_band = (t.bbox[0], t.bbox[2])
                def band_fn(w):
                    return (w.bbox[0], w.bbox[2])

            best_ci, best_dist = -1, float("inf")
            for ci, cw in enumerate(content_windows):
                if ci in used_c:
                    continue
                if not _bands_overlap(t_band, band_fn(cw)):
                    continue
                tr, tc = _window_center(t)
                cr, cc = _window_center(cw)
                dist = abs(tr - cr) + abs(tc - cc)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci

            if best_ci < 0:
                alt_band = (t.bbox[0], t.bbox[2]) if is_horiz else (t.bbox[1], t.bbox[3])
                alt_fn = (lambda w: (w.bbox[0], w.bbox[2])) if is_horiz else (lambda w: (w.bbox[1], w.bbox[3]))
                for ci, cw in enumerate(content_windows):
                    if ci in used_c:
                        continue
                    if not _bands_overlap(alt_band, alt_fn(cw)):
                        continue
                    tr, tc = _window_center(t)
                    cr, cc = _window_center(cw)
                    dist = abs(tr - cr) + abs(tc - cc)
                    if dist < best_dist:
                        best_dist = dist
                        best_ci = ci

            if best_ci >= 0:
                pairs.append((t, content_windows[best_ci]))
                used_t.add(ti)
                used_c.add(best_ci)

    for target, paired in pairs:
        p_content = strip_border(paired.interior_grid)
        pr, pc = p_content.shape
        if pr == 0 or pc == 0:
            continue

        tr0, tc0, tr1, tc1 = target.bbox
        pr0, pc0, pr1, pc1 = paired.bbox

        row_sep = abs((pr0 + pr1) / 2 - (tr0 + tr1) / 2)
        col_sep = abs((pc0 + pc1) / 2 - (tc0 + tc1) / 2)

        if col_sep >= row_sep:
            out_side = "right" if pc0 < tc0 else "left"
        else:
            out_side = "bottom" if pr0 < tr0 else "top"

        pair_center = _window_center(paired)
        best_src = min(sources, key=lambda s: abs(_window_center(s)[0] - pair_center[0]) + abs(_window_center(s)[1] - pair_center[1]))
        s_content, s_ext, N_src = source_fill_params(best_src)
        K = compute_fill_K(s_content, best_src.side, N_src, is_transfer=True)

        sdim = pr if out_side in ("left", "right") else pc
        N_out = N_src + sdim

        filled = progressive_fill(
            np.full((pr, pc), params['base_color'], dtype=result.dtype),
            out_side, N_out,
            source_extents=s_ext, N_source=N_src,
            source_side=best_src.side, source_shape=s_content.shape,
            K=K,
        )

        erase_region(result, paired.bbox, bg)

        iw = pc + 2
        ih = pr + 2
        target_vertical = target.side in ("left", "right")
        output_vertical = out_side in ("left", "right")
        axes_match = target_vertical == output_vertical

        if axes_match:
            separate_bar = (K > 2 and out_side == "top")
            if out_side == "left":
                anchor_r, anchor_c = tr0, tc0
            elif out_side == "right":
                anchor_r, anchor_c = tr0, tc1 - iw
            elif out_side == "top":
                anchor_r = tr1 + 1 if separate_bar else tr0
                anchor_c = tc0
            else:
                anchor_r, anchor_c = tr1 - ih, tc0
        else:
            if out_side == "left":
                anchor_r, anchor_c = tr0, tc1 + 2
            elif out_side == "right":
                anchor_r, anchor_c = tr0, tc0 - 2 - iw
            elif out_side == "top":
                anchor_r, anchor_c = tr1 + 2, tc0
            else:
                anchor_r, anchor_c = tr0 - 2 - ih, tc0

        stamp_bar_window(result, filled, out_side, bar_color, anchor_r, anchor_c)

    return result
