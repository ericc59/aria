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
    """Connect aligned workspace motifs following a bottom control-strip order."""
    from aria.guided.perceive import perceive
    from aria.search.executor import _exec_legend_chain_connect

    if not binding.entities_with_role(Role.CONTROL):
        return []
    if not binding.entities_with_role(Role.WORKSPACE):
        return []
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    params = {'control_side': 'bottom'}
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
