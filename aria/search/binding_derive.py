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
