"""Correspondence-derived parameter binding for search.

Analyzes input→output transitions to derive SearchStep parameters
structurally, not by blind enumeration. This is the bridge from
schema enumeration to real structural reasoning.

Uses aria.guided perception + correspondence as the analysis substrate,
but produces SearchProgram outputs that lower to AST.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from aria.search.sketch import SearchStep, SearchProgram, StepSelect


# ---------------------------------------------------------------------------
# Main entry: derive programs from correspondence
# ---------------------------------------------------------------------------

def derive_programs(demos: list[tuple[np.ndarray, np.ndarray]]) -> list[SearchProgram]:
    """Analyze demos via correspondence and derive candidate SearchPrograms.

    Returns a list of verified SearchPrograms, ordered by quality.
    Each program was derived from structural analysis, not blind enumeration.
    """
    from aria.guided.perceive import perceive
    from aria.guided.synthesize import compute_transitions

    # Shape-independent canonical strategies first
    progs = _derive_object_repack(demos)
    if progs:
        return progs

    progs = _derive_symmetry_repair(demos)
    if progs:
        return progs

    # Newer structural strategies (after canonical ones)
    progs = _derive_cross_stencil_recolor(demos)
    if progs:
        return progs

    progs = _derive_frame_bbox_pack(demos)
    if progs:
        return progs

    progs = _derive_anomaly_halo(demos)
    if progs:
        return progs

    progs = _derive_cavity_transfer(demos)
    if progs:
        return progs

    progs = _derive_masked_patch_transfer(demos)
    if progs:
        return progs

    progs = _derive_separator_motif_broadcast(demos)
    if progs:
        return progs

    progs = _derive_line_arith_broadcast(demos)
    if progs:
        return progs

    progs = _derive_barrier_port_transfer(demos)
    if progs:
        return progs

    progs = _derive_legend_frame_fill(demos)
    if progs:
        return progs

    # Shape-independent strategies (work for any input/output size)

    # Strategy 0a: Global color map (simple substitution)
    progs = _derive_color_map(demos)
    if progs:
        return progs

    # Strategy 0b: Per-color stencil stamp (each colored cell gets a pattern)
    progs = _derive_color_stencil(demos)
    if progs:
        return progs

    # Strategy 0c: Direct subgrid crop (output = exact rectangle from input)
    progs = _derive_direct_crop(demos)
    if progs:
        return progs

    # Strategy 0d: Scale-up or tile (output = repeated/scaled input)
    progs = _derive_scale_or_tile(demos)
    if progs:
        return progs

    # Strategy 0e: Template broadcast (output = kron(mask, template))
    progs = _derive_template_broadcast(demos)
    if progs:
        return progs

    # Remaining strategies require same-shape
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    progs = _derive_diagonal_collision_trace(demos)
    if progs:
        return progs

    # Compute transitions for all demos
    all_transitions = []
    all_facts = []
    for inp, out in demos:
        in_facts = perceive(inp)
        out_facts = perceive(out)
        all_facts.append(in_facts)
        all_transitions.append(compute_transitions(in_facts, out_facts))

    if not all_transitions[0]:
        return []

    results = []

    # Strategy 1: Uniform transition (all changed objects → same action)
    progs = _derive_uniform(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 2: Multi-group dispatch (different groups → different actions)
    progs = _derive_dispatch(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 2b: Conditional dispatch (same action type, different params per predicate group)
    progs = _derive_conditional_dispatch(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 3: Stamp/creation (new objects from input object shapes)
    progs = _derive_stamp(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 4: Marker stamp (small markers → learned template stamped at each)
    progs = _derive_marker_stamp(demos, all_facts)
    results.extend(progs)


    return results


# ---------------------------------------------------------------------------
# Strategy 1: Uniform transition
# ---------------------------------------------------------------------------

def _derive_scale_or_tile(demos):
    """Derive programs where the output is a scaled or tiled version of the input."""
    if not demos:
        return []

    for inp, out in demos:
        if out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]:
            return []

    # Try exact tile
    inp0, out0 = demos[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape

    if oh % ih == 0 and ow % iw == 0:
        rr, rc = oh // ih, ow // iw
        if np.array_equal(np.tile(inp0, (rr, rc)), out0):
            # Enforce SAME tile params on all demos
            all_ok = all(
                out.shape == (inp.shape[0] * rr, inp.shape[1] * rc)
                and np.array_equal(np.tile(inp, (rr, rc)), out)
                for inp, out in demos[1:]
            )
            if all_ok:
                return [SearchProgram(
                    steps=[SearchStep('tile', {'rows': rr, 'cols': rc})],
                    provenance='derive:exact_tile',
                )]

    # Try pixel scale (each cell → NxN block)
    if oh % ih == 0 and ow % iw == 0:
        sr, sc = oh // ih, ow // iw
        if sr == sc:
            scaled = np.repeat(np.repeat(inp0, sr, axis=0), sc, axis=1)
            if np.array_equal(scaled, out0):
                # Enforce SAME scale factor on all demos
                all_ok = all(
                    out.shape == (inp.shape[0] * sr, inp.shape[1] * sr)
                    and np.array_equal(
                        np.repeat(np.repeat(inp, sr, axis=0), sr, axis=1), out)
                    for inp, out in demos[1:]
                )
                if all_ok:
                    return [SearchProgram(
                        steps=[SearchStep('scale', {'factor': sr})],
                        provenance='derive:pixel_scale',
                    )]

    return []


def _derive_template_broadcast(demos):
    """Derive template broadcast: out = kron(input != bg, input).

    Verification:
    1. Output must be multiplicative in input size: oh == ih * ih, ow == iw * iw
    2. Partition output into ih×iw blocks
    3. Each block must be either the full input template or a bg-filled block
    4. Non-bg block positions must match the input's non-bg support mask
    """
    if not demos:
        return []

    for inp, out in demos:
        ih, iw = inp.shape
        oh, ow = out.shape
        # Output size must be ih*ih × iw*iw (template used as both mask and content)
        if oh != ih * ih or ow != iw * iw:
            return []

    # Verify block structure for every demo
    from aria.guided.perceive import perceive

    for inp, out in demos:
        ih, iw = inp.shape
        bg = int(perceive(inp).bg)

        bg_block = np.full((ih, iw), bg, dtype=inp.dtype)

        for br in range(ih):
            for bc in range(iw):
                block = out[br * ih:(br + 1) * ih, bc * iw:(bc + 1) * iw]
                is_bg_cell = (inp[br, bc] == bg)

                if is_bg_cell:
                    if not np.array_equal(block, bg_block):
                        return []
                else:
                    if not np.array_equal(block, inp):
                        return []

    return [SearchProgram(
        steps=[SearchStep('template_broadcast', {})],
        provenance='derive:template_broadcast',
    )]


def _derive_direct_crop(demos):
    """Derive programs where the output is an exact subgrid of the input.

    Finds a consistent crop rule across all demos. Tries:
    1. Fixed offset (same r0, c0 in every demo)
    2. Object-anchored crop (output = bbox of an object matching a predicate)
    3. Non-bg bounding box crop (output = tightest bbox around non-bg cells)
    """
    if not demos:
        return []

    # Check: all demos have output smaller than input
    for inp, out in demos:
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            return []

    # Try: non-bg bounding box crop (most common)
    all_match_nonbg = True
    for inp, out in demos:
        from aria.guided.perceive import perceive
        facts = perceive(inp)
        bg = facts.bg
        nonbg = np.argwhere(inp != bg)
        if len(nonbg) == 0:
            all_match_nonbg = False
            break
        r0, c0 = nonbg.min(axis=0)
        r1, c1 = nonbg.max(axis=0)
        crop = inp[r0:r1+1, c0:c1+1]
        if not np.array_equal(crop, out):
            all_match_nonbg = False
            break

    if all_match_nonbg:
        prog = SearchProgram(
            steps=[SearchStep('crop_nonbg', {})],
            provenance='derive:crop_nonbg',
        )
        return [prog]

    # Try: object bbox crop (output = bbox of largest/smallest/unique object)
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    for pred, role in [(Pred.IS_LARGEST, 'largest'), (Pred.IS_SMALLEST, 'smallest'),
                       (Pred.UNIQUE_COLOR, 'unique_color'),
                       (Pred.IS_TOPMOST, 'topmost'), (Pred.IS_BOTTOMMOST, 'bottommost'),
                       (Pred.IS_LEFTMOST, 'leftmost'), (Pred.IS_RIGHTMOST, 'rightmost'),
                       (Pred.NOT_TOUCHES_BORDER, 'interior'),
                       (Pred.TOUCHES_BORDER, 'touches_border')]:
        all_match = True
        for inp, out in demos:
            facts = perceive(inp)
            selected = prim_select(facts, [Predicate(pred)])
            if len(selected) != 1:
                all_match = False
                break
            obj = selected[0]
            crop = inp[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
            if not np.array_equal(crop, out):
                all_match = False
                break

        if all_match:
            prog = SearchProgram(
                steps=[SearchStep('crop_object', {'predicate': role})],
                provenance=f'derive:crop_object_{role}',
            )
            return [prog]

    # Try: color-based crop (output = bbox of unique object of a specific color)
    inp0_facts = perceive(demos[0][0])
    for color in sorted(set(o.color for o in inp0_facts.objects)):
        all_match = True
        for inp, out in demos:
            facts = perceive(inp)
            selected = prim_select(facts, [Predicate(Pred.COLOR_EQ, color)])
            if len(selected) != 1:
                all_match = False
                break
            obj = selected[0]
            crop = inp[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
            if not np.array_equal(crop, out):
                all_match = False
                break
        if all_match:
            return [SearchProgram(
                steps=[SearchStep('crop_object', {'predicate': f'color_{color}'})],
                provenance=f'derive:crop_object_color_{color}',
            )]

    # Try: fixed offset crop
    oh, ow = demos[0][1].shape
    offsets = set()
    for inp, out in demos:
        if out.shape != (oh, ow):
            return []  # inconsistent output size
        found = False
        for r in range(inp.shape[0] - oh + 1):
            for c in range(inp.shape[1] - ow + 1):
                if np.array_equal(inp[r:r+oh, c:c+ow], out):
                    offsets.add((r, c))
                    found = True
                    break
            if found:
                break
        if not found:
            return []

    if len(offsets) == 1:
        r0, c0 = next(iter(offsets))
        prog = SearchProgram(
            steps=[SearchStep('crop_fixed', {'r0': r0, 'c0': c0, 'h': oh, 'w': ow})],
            provenance='derive:crop_fixed',
        )
        return [prog]

    return []


def _derive_color_stencil(demos):
    """Derive a per-color stencil stamp.

    For each non-bg color in the input, extract the pattern that appears
    around each cell of that color in the output. If the pattern is
    consistent across all cells and all demos, emit a stencil_stamp program.
    """
    if not demos:
        return []

    inp0, out0 = demos[0]
    if inp0.shape != out0.shape:
        return []

    # Detect bg
    border = np.concatenate([inp0[0], inp0[-1], inp0[1:-1, 0], inp0[1:-1, -1]])
    bg = int(np.bincount(border).argmax())

    # Find non-bg colors
    colors = sorted(set(int(inp0[r, c]) for r in range(inp0.shape[0])
                        for c in range(inp0.shape[1]) if inp0[r, c] != bg))
    if not colors:
        return []

    # For each color, extract the stencil pattern from demo 0.
    # Try radii 1 then 2 (most ARC stencils are 3x3 or 5x5).
    for radius in (1, 2):
        stencils = _extract_stencils(inp0, out0, colors, bg, radius)
        if stencils:
            params = {'bg': bg, 'stencils': {str(c): {f"{dr},{dc}": v for (dr, dc), v in s.items()}
                                              for c, s in stencils.items()}}
            if all(np.array_equal(_exec_color_stencil(inp, params), out) for inp, out in demos):
                return [SearchProgram(
                    steps=[SearchStep('color_stencil', params)],
                    provenance='derive:color_stencil',
                )]

    return []


def _extract_stencils(inp, out, colors, bg, radius):
    """Extract per-color stencil patterns at given radius."""
    stencils = {}
    for color in colors:
        pattern = {}
        cells = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1])
                 if inp[r, c] == color]
        if not cells:
            continue

        for r, c in cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                        ov = int(out[nr, nc])
                        iv = int(inp[nr, nc])
                        if ov != iv and iv == bg:
                            key = (dr, dc)
                            if key in pattern:
                                if pattern[key] != ov:
                                    pattern = None
                                    break
                            else:
                                pattern[key] = ov
                if pattern is None:
                    break

            if pattern is None:
                break

        if pattern:
            stencils[color] = pattern

    return stencils


def _exec_color_stencil(grid, params):
    """Apply per-color stencil patterns to a grid."""
    bg = params.get('bg', 0)
    stencils = params.get('stencils', {})
    result = grid.copy()
    h, w = grid.shape

    for color_str, pattern in stencils.items():
        color = int(color_str)
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == color]
        for r, c in cells:
            for key, fill_color in pattern.items():
                dr, dc = map(int, key.split(','))
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == bg:
                    result[nr, nc] = fill_color

    return result


def _derive_color_map(demos):
    """Derive a global color substitution map.

    If every cell's change can be explained by a consistent color→color lookup
    table across all demos, emit a single-step recolor_map program.
    """
    color_map = {}
    for inp, out in demos:
        if inp.shape != out.shape:
            return []
        diff = inp != out
        if not np.any(diff):
            return []
        for r, c in zip(*np.where(diff)):
            ic, oc = int(inp[r, c]), int(out[r, c])
            if ic in color_map:
                if color_map[ic] != oc:
                    return []  # conflicting mapping
            else:
                color_map[ic] = oc

    if not color_map:
        return []

    # Verify: the map must explain ALL changes in every demo
    for inp, out in demos:
        pred = inp.copy()
        for r in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                v = int(pred[r, c])
                if v in color_map:
                    pred[r, c] = color_map[v]
        if not np.array_equal(pred, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('recolor_map', {'color_map': color_map})],
        provenance='derive:color_map',
    )
    return [prog]


def _derive_uniform(all_transitions, all_facts, demos):
    """All non-identical objects undergo the same action."""
    results = []

    # Check: all changed objects have the same match_type across all demos
    match_type = None
    for demo_trans in all_transitions:
        changed = [t for t in demo_trans if t.match_type != 'identical']
        if not changed:
            return []
        types = set(t.match_type for t in changed)
        if len(types) != 1:
            return []
        mt = next(iter(types))
        if match_type is None:
            match_type = mt
        elif mt != match_type:
            return []

    if match_type == 'moved':
        results.extend(_derive_uniform_move(all_transitions, all_facts, demos))
        if not results:
            results.extend(_derive_per_object_move(all_transitions, all_facts, demos))
    elif match_type == 'recolored':
        results.extend(_derive_uniform_recolor(all_transitions, all_facts, demos))
    elif match_type == 'moved_recolored':
        results.extend(_derive_uniform_move_recolor(all_transitions, all_facts, demos))
    elif match_type == 'transformed':
        results.extend(_derive_uniform_transform(all_transitions, all_facts, demos))
    elif match_type == 'removed':
        results.extend(_derive_uniform_remove(all_transitions, all_facts, demos))

    return results


def _derive_uniform_move(all_transitions, all_facts, demos):
    """Derive move programs: constant offset, gravity, or slide."""
    results = []

    # Check constant offset
    offsets = set()
    for demo_trans in all_transitions:
        moved = [t for t in demo_trans if t.match_type == 'moved']
        demo_offsets = set((t.dr, t.dc) for t in moved)
        if len(demo_offsets) != 1:
            break
        offsets.add(next(iter(demo_offsets)))
    else:
        if len(offsets) == 1:
            dr, dc = next(iter(offsets))
            sel = _find_selector_for_group(all_transitions, all_facts, 'moved')
            if sel:
                prog = SearchProgram(
                    steps=[SearchStep('move', {'dr': dr, 'dc': dc}, sel)],
                    provenance='derive:uniform_move',
                )
                if prog.verify(demos):
                    results.append(prog)

    # Check gravity (all to same border, varying offsets)
    for direction in ('down', 'up', 'left', 'right'):
        consistent = True
        any_moved = False
        for i, demo_trans in enumerate(all_transitions):
            moved = [t for t in demo_trans if t.match_type == 'moved']
            rows, cols = demos[i][0].shape
            for t in moved:
                o = t.out_obj
                at_border = _at_border(o, direction, rows, cols)
                toward = _moved_toward(t, direction)
                if not at_border:
                    consistent = False
                    break
                if toward:
                    any_moved = True
            if not consistent:
                break
        if consistent and any_moved:
            sel = _find_selector_for_group(all_transitions, all_facts, 'moved')
            if sel:
                prog = SearchProgram(
                    steps=[SearchStep('gravity', {'direction': direction}, sel)],
                    provenance=f'derive:uniform_gravity_{direction}',
                )
                if prog.verify(demos):
                    results.append(prog)

    return results


def _derive_per_object_move(all_transitions, all_facts, demos):
    """Derive per-object movement rules when each object moves differently.

    Checks structural offset patterns:
    1. Gravity to nearest border (each object goes to the closest edge)
    2. Slide in a consistent direction until collision
    3. Gravity to a specific border (all go same direction, varying distance)
    """
    results = []

    # All transitions must be 'moved'
    for demo_trans in all_transitions:
        if not all(t.match_type in ('moved', 'identical') for t in demo_trans):
            return []

    # Check: all objects move along the same axis?
    for axis in ['vertical', 'horizontal']:
        consistent = True
        for demo_trans in all_transitions:
            moved = [t for t in demo_trans if t.match_type == 'moved']
            for t in moved:
                if axis == 'vertical' and t.dc != 0:
                    consistent = False
                    break
                if axis == 'horizontal' and t.dr != 0:
                    consistent = False
                    break
            if not consistent:
                break
        if not consistent:
            continue

        # All on same axis. Try: gravity to nearest border
        prog = _try_gravity_nearest(all_transitions, all_facts, demos, axis)
        if prog:
            results.append(prog)
            return results

        # Try: slide in consistent direction until collision
        prog = _try_slide_until_collision(all_transitions, all_facts, demos, axis)
        if prog:
            results.append(prog)
            return results

    return results


def _try_gravity_nearest(all_transitions, all_facts, demos, axis):
    """Each object moves to the nearest border along the given axis."""
    for demo_idx, (inp, out) in enumerate(demos):
        demo_trans = all_transitions[demo_idx]
        moved = [t for t in demo_trans if t.match_type == 'moved']
        rows, cols = inp.shape

        for t in moved:
            obj = t.in_obj
            out_obj = t.out_obj
            if axis == 'vertical':
                dist_top = obj.center_row
                dist_bottom = rows - (obj.center_row + 1)
                if dist_top <= dist_bottom:
                    expected_row = 0
                else:
                    expected_row = rows - obj.height
                if out_obj.row != expected_row:
                    return None
            else:
                dist_left = obj.center_col
                dist_right = cols - (obj.center_col + 1)
                if dist_left <= dist_right:
                    expected_col = 0
                else:
                    expected_col = cols - obj.width
                if out_obj.col != expected_col:
                    return None

    # Verified across all demos
    sel = _find_selector_for_group(all_transitions, all_facts, 'moved')
    if sel is None:
        sel = StepSelect('all')

    prog = SearchProgram(
        steps=[SearchStep('gravity_nearest', {'axis': axis}, sel)],
        provenance=f'derive:gravity_nearest_{axis}',
    )
    if prog.verify(demos):
        return prog
    return None


def _try_slide_until_collision(all_transitions, all_facts, demos, axis):
    """Each object slides in a direction until it hits another object or border."""
    # Determine direction per object: toward which border?
    # Group: objects moving positive vs negative
    for demo_trans in all_transitions:
        moved = [t for t in demo_trans if t.match_type == 'moved']
        pos_count = sum(1 for t in moved if (t.dr > 0 if axis == 'vertical' else t.dc > 0))
        neg_count = sum(1 for t in moved if (t.dr < 0 if axis == 'vertical' else t.dc < 0))

        # If all same direction → simple slide
        if pos_count == len(moved) or neg_count == len(moved):
            direction = ('down' if pos_count > 0 else 'up') if axis == 'vertical' \
                else ('right' if pos_count > 0 else 'left')

            sel = _find_selector_for_group(all_transitions, all_facts, 'moved')
            if sel is None:
                sel = StepSelect('all')

            prog = SearchProgram(
                steps=[SearchStep('slide', {'direction': direction}, sel)],
                provenance=f'derive:slide_{direction}',
            )
            if prog.verify(demos):
                return prog

        # Mixed directions → gravity to nearest border (per-object direction)
        # Already handled by _try_gravity_nearest

    return None


def _derive_uniform_recolor(all_transitions, all_facts, demos):
    """Derive recolor programs: constant color or support-derived."""
    results = []

    # Check constant color
    colors = set()
    for demo_trans in all_transitions:
        recolored = [t for t in demo_trans if t.match_type == 'recolored']
        for t in recolored:
            colors.add(t.color_to)
    if len(colors) == 1:
        color = next(iter(colors))
        sel = _find_selector_for_group(all_transitions, all_facts, 'recolored')
        if sel:
            prog = SearchProgram(
                steps=[SearchStep('recolor', {'color': color}, sel)],
                provenance='derive:uniform_recolor_const',
            )
            if prog.verify(demos):
                results.append(prog)

    # Check support-derived color (color of largest/smallest/unique)
    if len(colors) > 1:
        for support_role in ('largest', 'smallest', 'unique_color'):
            consistent = True
            for i, demo_trans in enumerate(all_transitions):
                recolored = [t for t in demo_trans if t.match_type == 'recolored']
                if not recolored:
                    consistent = False
                    break
                expected_color = _get_support_color(all_facts[i], support_role)
                if expected_color is None:
                    consistent = False
                    break
                if not all(t.color_to == expected_color for t in recolored):
                    consistent = False
                    break
            if consistent:
                sel = _find_selector_for_group(all_transitions, all_facts, 'recolored')
                if sel:
                    # Use the support color from each demo at runtime
                    # For now: constant from demo 0 (TODO: derived color in AST)
                    c = _get_support_color(all_facts[0], support_role)
                    prog = SearchProgram(
                        steps=[SearchStep('recolor', {'color': c}, sel)],
                        provenance=f'derive:uniform_recolor_from_{support_role}',
                    )
                    if prog.verify(demos):
                        results.append(prog)

    return results


def _derive_uniform_move_recolor(all_transitions, all_facts, demos):
    """Derive move+recolor programs."""
    results = []

    offsets = set()
    colors = set()
    for demo_trans in all_transitions:
        changed = [t for t in demo_trans if t.match_type == 'moved_recolored']
        if not changed:
            return []
        demo_offsets = set((t.dr, t.dc) for t in changed)
        if len(demo_offsets) != 1:
            return []
        offsets.add(next(iter(demo_offsets)))
        for t in changed:
            colors.add(t.color_to)

    if len(offsets) == 1 and len(colors) == 1:
        dr, dc = next(iter(offsets))
        color = next(iter(colors))
        sel = _find_selector_for_group(all_transitions, all_facts, 'moved_recolored')
        if sel:
            prog = SearchProgram(
                steps=[
                    SearchStep('move', {'dr': dr, 'dc': dc}, sel),
                    SearchStep('recolor', {'color': color}, sel),
                ],
                provenance='derive:uniform_move_recolor',
            )
            if prog.verify(demos):
                results.append(prog)

    return results


def _derive_uniform_transform(all_transitions, all_facts, demos):
    """Derive in-place transform programs."""
    results = []

    xforms = set()
    for demo_trans in all_transitions:
        transformed = [t for t in demo_trans if t.match_type == 'transformed']
        if not transformed:
            return []
        demo_xforms = set(t.transform for t in transformed if t.transform)
        if len(demo_xforms) != 1:
            return []
        xforms.add(next(iter(demo_xforms)))

    if len(xforms) == 1:
        xform = next(iter(xforms))
        sel = _find_selector_for_group(all_transitions, all_facts, 'transformed')
        if sel:
            prog = SearchProgram(
                steps=[SearchStep('transform', {'xform': xform}, sel)],
                provenance=f'derive:uniform_transform_{xform}',
            )
            if prog.verify(demos):
                results.append(prog)

    return results


def _derive_uniform_remove(all_transitions, all_facts, demos):
    """Derive remove programs."""
    results = []
    sel = _find_selector_for_group(all_transitions, all_facts, 'removed')
    if sel:
        prog = SearchProgram(
            steps=[SearchStep('remove', {}, sel)],
            provenance='derive:uniform_remove',
        )
        if prog.verify(demos):
            results.append(prog)
    return results


# ---------------------------------------------------------------------------
# Strategy 2: Multi-group dispatch
# ---------------------------------------------------------------------------

def _derive_dispatch(all_transitions, all_facts, demos):
    """Different object groups undergo different transitions."""
    results = []

    # Group transitions by match_type in demo 0
    groups = defaultdict(list)
    for t in all_transitions[0]:
        if t.match_type == 'identical':
            continue
        groups[t.match_type].append(t)

    if len(groups) < 2:
        return []

    # Find selectors for each group
    steps = []
    for mtype, transitions in groups.items():
        oids = set(t.in_obj.oid for t in transitions if t.in_obj)
        if not oids:
            continue
        sel = _find_selector(oids, all_facts[0])
        if sel is None:
            return []  # can't select this group

        if mtype == 'removed':
            steps.append(SearchStep('remove', {}, sel))
        elif mtype == 'recolored':
            colors = set(t.color_to for t in transitions)
            if len(colors) == 1:
                steps.append(SearchStep('recolor', {'color': next(iter(colors))}, sel))
            else:
                return []
        elif mtype == 'moved':
            offsets = set((t.dr, t.dc) for t in transitions)
            if len(offsets) == 1:
                dr, dc = next(iter(offsets))
                steps.append(SearchStep('move', {'dr': dr, 'dc': dc}, sel))
            else:
                # Try gravity
                rows, cols = demos[0][0].shape
                gdir = _detect_gravity(transitions, rows, cols)
                if gdir:
                    steps.append(SearchStep('gravity', {'direction': gdir}, sel))
                else:
                    return []
        elif mtype == 'transformed':
            xforms = set(t.transform for t in transitions if t.transform)
            if len(xforms) == 1:
                steps.append(SearchStep('transform', {'xform': next(iter(xforms))}, sel))
            else:
                return []
        else:
            return []  # unhandled transition type

    if len(steps) >= 2:
        prog = SearchProgram(steps=steps, provenance='derive:dispatch')
        if prog.verify(demos):
            results.append(prog)

    return results


def _derive_conditional_dispatch(all_transitions, all_facts, demos):
    """Objects split into predicate-selected groups, each getting its own action.

    Unlike _derive_dispatch (groups by match_type), this finds cases where
    objects of the SAME match_type get different params based on a property.
    Example: 'color-2 objects move down, color-4 objects move up'.

    Strategy:
    1. In demo 0: group objects by a candidate selector
    2. For each group: derive a uniform action (recolor/move/remove/transform)
    3. Verify the program on all demos
    """
    results = []

    changed_0 = [t for t in all_transitions[0] if t.match_type != 'identical']
    if len(changed_0) < 2:
        return []

    all_oids = set(t.in_obj.oid for t in changed_0 if t.in_obj)
    if len(all_oids) < 2:
        return []

    # Derive bg from demo output (handles dense grids where perceive can't detect bg)
    bg = _infer_bg_from_demos(demos)

    for partition in _candidate_partitions(changed_0, all_facts[0]):
        # Cross-demo validation: verify the partition holds in ALL demos.
        # The same selector must pick objects with the same match_type in every demo.
        if not _partition_generalizes(partition, all_transitions, all_facts):
            continue

        steps = []
        ok = True
        for sel, group_oids in partition:
            group_trans_demo0 = [t for t in changed_0 if t.in_obj and t.in_obj.oid in group_oids]
            # Gather same-selector transitions across ALL demos
            all_group_trans = [group_trans_demo0]
            for di in range(1, len(all_transitions)):
                sel_oids = _resolve_selector(sel, all_facts[di])
                if sel_oids:
                    gt = [t for t in all_transitions[di]
                          if t.in_obj and t.in_obj.oid in sel_oids and t.match_type != 'identical']
                    all_group_trans.append(gt)
            step = _derive_group_step_multi(all_group_trans, sel, demos, bg)
            if step is None:
                ok = False
                break
            steps.append(step)

        if not ok or len(steps) < 2:
            continue

        prog = SearchProgram(steps=steps, provenance='derive:conditional_dispatch')
        if prog.verify(demos):
            results.append(prog)
            return results

    return results


def _partition_generalizes(partition, all_transitions, all_facts):
    """Check that a partition's selectors produce DISJOINT, consistent groups in ALL demos.

    For each demo beyond 0:
    1. Each selector must resolve to a non-empty set of changed objects
    2. All changed objects in a group must share the same match_type
    3. Groups must be DISJOINT (no object selected by two selectors)
    4. Groups must COVER all changed objects (nothing missed)
    """
    for demo_idx in range(1, len(all_transitions)):
        facts = all_facts[demo_idx]
        trans = all_transitions[demo_idx]
        trans_by_oid = {t.in_obj.oid: t for t in trans if t.in_obj}
        changed_oids = {oid for oid, t in trans_by_oid.items() if t.match_type != 'identical'}

        all_selected = set()
        for sel, _ in partition:
            selected = _resolve_selector(sel, facts)
            if selected is None:
                return False

            # Only consider changed objects in this group
            group_changed = selected & changed_oids
            if not group_changed:
                # Selector finds objects but none are changed — acceptable
                # (some groups may be empty in some demos)
                continue

            # Disjointness: no overlap with previously selected
            if group_changed & all_selected:
                return False
            all_selected |= group_changed

            # Uniformity: all changed objects in group share match_type
            group_mtypes = {trans_by_oid[oid].match_type for oid in group_changed}
            if len(group_mtypes) > 1:
                return False

    return True


def _infer_bg_from_demos(demos):
    """Infer background color from demo input/output comparison.

    Looks at cells that changed from non-zero to a uniform color → that's bg.
    Falls back to perceive if no removal detected.
    """
    from collections import Counter
    from aria.guided.perceive import perceive

    for inp, out in demos:
        removed_mask = (inp != out) & (inp != 0)
        if not np.any(removed_mask):
            continue
        out_colors = Counter(int(out[r, c]) for r, c in zip(*np.where(removed_mask)))
        if len(out_colors) == 1:
            return next(iter(out_colors))

    # Fallback: perceive bg from first output (output usually has clear bg)
    return int(perceive(demos[0][1]).bg)


def _candidate_partitions(changed_transitions, facts):
    """Generate candidate (selector, oid_set) partitions of changed objects."""
    from aria.guided.clause import Predicate, Pred

    # By color
    color_groups = defaultdict(set)
    for t in changed_transitions:
        if t.in_obj:
            color_groups[t.in_obj.color].add(t.in_obj.oid)
    if len(color_groups) >= 2:
        partition = []
        for c, oids in sorted(color_groups.items()):
            sel = StepSelect('by_color', {'color': c})
            partition.append((sel, oids))
        yield partition

    # By structural predicate pairs (e.g., largest vs rest)
    _pred_pairs = [
        (Pred.IS_LARGEST, Pred.NOT),
        (Pred.IS_SMALLEST, Pred.NOT),
        (Pred.TOUCHES_BORDER, Pred.NOT_TOUCHES_BORDER),
        (Pred.IS_RECTANGULAR, Pred.NOT),
        (Pred.IS_SINGLETON, Pred.NOT),
    ]
    all_oids = set(t.in_obj.oid for t in changed_transitions if t.in_obj)
    from aria.guided.dsl import prim_select

    for pos_pred, neg_pred in _pred_pairs:
        pos_preds = [Predicate(pos_pred)]
        pos_oids = set(o.oid for o in prim_select(facts, pos_preds))
        pos_in_changed = pos_oids & all_oids
        neg_in_changed = all_oids - pos_in_changed

        if pos_in_changed and neg_in_changed:
            if neg_pred == Pred.NOT:
                neg_sel = StepSelect('by_predicate', {'predicates': [Predicate(Pred.NOT, Predicate(pos_pred))]})
            else:
                neg_sel = StepSelect('by_predicate', {'predicates': [Predicate(neg_pred)]})

            # Map pos_pred to role name
            _pred_to_role = {
                Pred.IS_LARGEST: 'largest', Pred.IS_SMALLEST: 'smallest',
                Pred.TOUCHES_BORDER: 'touches_border',
                Pred.IS_RECTANGULAR: 'rectangular', Pred.IS_SINGLETON: 'singleton',
            }
            pos_sel = StepSelect(_pred_to_role.get(pos_pred, 'by_predicate'),
                                 {} if pos_pred in _pred_to_role else {'predicates': pos_preds})
            yield [(pos_sel, pos_in_changed), (neg_sel, neg_in_changed)]


def _derive_group_step(group_trans, sel, all_transitions, all_facts, demos, bg=None):
    """Derive a single SearchStep for a group of objects that share a selector.

    Handles:
    - removed: pass bg so executor uses correct background
    - recolored: constant target color
    - moved: fixed offset OR gravity/slide for variable offsets
    - transformed: uniform geometric transform
    """
    if not group_trans:
        return None

    mtypes = set(t.match_type for t in group_trans)
    if len(mtypes) != 1:
        return None
    mtype = next(iter(mtypes))

    if mtype == 'removed':
        params = {'bg': bg} if bg is not None else {}
        return SearchStep('remove', params, sel)

    if mtype == 'recolored':
        colors = set(t.color_to for t in group_trans)
        if len(colors) == 1:
            return SearchStep('recolor', {'color': next(iter(colors))}, sel)
        return None

    if mtype == 'moved':
        offsets = set((t.dr, t.dc) for t in group_trans)
        if len(offsets) == 1:
            dr, dc = next(iter(offsets))
            return SearchStep('move', {'dr': dr, 'dc': dc}, sel)
        # Try gravity (variable distances, consistent direction)
        rows, cols = demos[0][0].shape
        gdir = _detect_gravity(group_trans, rows, cols)
        if gdir:
            return SearchStep('gravity', {'direction': gdir}, sel)
        # Try slide (move until collision)
        sdir = _detect_slide_direction(group_trans)
        if sdir:
            return SearchStep('slide', {'direction': sdir}, sel)
        return None

    if mtype == 'moved_recolored':
        # Split: try move + recolor if uniform recolor
        colors = set(t.color_to for t in group_trans)
        offsets = set((t.dr, t.dc) for t in group_trans)
        if len(colors) == 1 and len(offsets) == 1:
            # Both uniform — return move (recolor will be a separate step... but we
            # can only return one step here. Return the move; recolor needs composition.)
            dr, dc = next(iter(offsets))
            return SearchStep('move', {'dr': dr, 'dc': dc}, sel)
        if len(colors) == 1:
            # Uniform recolor, variable move → try gravity
            rows, cols = demos[0][0].shape
            gdir = _detect_gravity(group_trans, rows, cols)
            if gdir:
                return SearchStep('gravity', {'direction': gdir}, sel)
        return None

    if mtype == 'transformed':
        xforms = set(t.transform for t in group_trans if t.transform)
        if len(xforms) == 1:
            return SearchStep('transform', {'xform': next(iter(xforms))}, sel)
        return None

    return None


def _derive_group_step_multi(all_group_trans, sel, demos, bg=None):
    """Derive a SearchStep from a group's transitions across ALL demos.

    Uses multi-demo evidence to detect variable-offset patterns (gravity/slide)
    that single-demo analysis would miss.
    """
    if not all_group_trans or not all_group_trans[0]:
        return None

    # All demos must have the same match_type for this group
    mtypes = set()
    for gt in all_group_trans:
        for t in gt:
            if t.match_type != 'identical':
                mtypes.add(t.match_type)
    if len(mtypes) != 1:
        return None
    mtype = next(iter(mtypes))

    if mtype == 'removed':
        params = {'bg': bg} if bg is not None else {}
        return SearchStep('remove', params, sel)

    if mtype == 'recolored':
        # Check: same target color across ALL demos
        all_colors = set()
        for gt in all_group_trans:
            for t in gt:
                all_colors.add(t.color_to)
        if len(all_colors) == 1:
            return SearchStep('recolor', {'color': next(iter(all_colors))}, sel)
        return None

    if mtype in ('moved', 'moved_recolored'):
        # Collect offsets from ALL demos
        all_offsets = []
        for gt in all_group_trans:
            for t in gt:
                all_offsets.append((t.dr, t.dc))

        offset_set = set(all_offsets)
        if len(offset_set) == 1:
            dr, dc = next(iter(offset_set))
            return SearchStep('move', {'dr': dr, 'dc': dc}, sel)

        # Variable offsets → try gravity (same direction, variable distance)
        flat = [t for gt in all_group_trans for t in gt]
        rows, cols = demos[0][0].shape
        gdir = _detect_gravity(flat, rows, cols)
        if gdir:
            return SearchStep('gravity', {'direction': gdir}, sel)

        sdir = _detect_slide_direction(flat)
        if sdir:
            return SearchStep('slide', {'direction': sdir}, sel)

        # Try gravity_nearest (each object moves to nearest border)
        return SearchStep('gravity_nearest', {'axis': 'both'}, sel)

    if mtype == 'transformed':
        all_xforms = set()
        for gt in all_group_trans:
            for t in gt:
                if t.transform:
                    all_xforms.add(t.transform)
        if len(all_xforms) == 1:
            return SearchStep('transform', {'xform': next(iter(all_xforms))}, sel)
        return None

    return None


def _detect_slide_direction(transitions):
    """Detect if all transitions slide in the same direction (variable distance)."""
    if not transitions:
        return None
    drs = set(1 if t.dr > 0 else (-1 if t.dr < 0 else 0) for t in transitions)
    dcs = set(1 if t.dc > 0 else (-1 if t.dc < 0 else 0) for t in transitions)

    if len(drs) == 1 and len(dcs) == 1:
        dr, dc = next(iter(drs)), next(iter(dcs))
        if dr != 0 and dc == 0:
            return 'down' if dr > 0 else 'up'
        if dc != 0 and dr == 0:
            return 'right' if dc > 0 else 'left'
    return None


def _resolve_selector(sel, facts):
    """Resolve a StepSelect to a set of object IDs."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    if sel.role == 'by_color':
        color = sel.params.get('color')
        preds = [Predicate(Pred.COLOR_EQ, color)]
        return set(o.oid for o in prim_select(facts, preds))

    # Map role names to predicates
    _role_to_pred = {
        'largest': Pred.IS_LARGEST,
        'smallest': Pred.IS_SMALLEST,
        'unique_color': Pred.UNIQUE_COLOR,
        'singleton': Pred.IS_SINGLETON,
        'rectangular': Pred.IS_RECTANGULAR,
        'line': Pred.IS_LINE,
        'topmost': Pred.IS_TOPMOST,
        'bottommost': Pred.IS_BOTTOMMOST,
        'leftmost': Pred.IS_LEFTMOST,
        'rightmost': Pred.IS_RIGHTMOST,
        'touches_border': Pred.TOUCHES_BORDER,
        'interior': Pred.NOT_TOUCHES_BORDER,
    }
    if sel.role in _role_to_pred:
        preds = [Predicate(_role_to_pred[sel.role])]
        return set(o.oid for o in prim_select(facts, preds))

    if sel.role == 'all':
        return set(o.oid for o in facts.objects)

    return None


def _transition_action_key(t):
    """Extract a hashable (match_type, param_key) for grouping transitions."""
    if t.match_type == 'recolored':
        return ('recolored', t.color_to)
    if t.match_type == 'moved':
        return ('moved', (t.dr, t.dc))
    if t.match_type == 'removed':
        return ('removed', None)
    if t.match_type == 'transformed':
        return ('transformed', t.transform)
    if t.match_type == 'moved_recolored':
        return ('moved_recolored', (t.dr, t.dc, t.color_to))
    return None


def _action_step_from_key(mtype, param_key, sel):
    """Build a SearchStep from an action key."""
    if mtype == 'recolored':
        return SearchStep('recolor', {'color': param_key}, sel)
    if mtype == 'moved':
        dr, dc = param_key
        return SearchStep('move', {'dr': dr, 'dc': dc}, sel)
    if mtype == 'removed':
        return SearchStep('remove', {}, sel)
    if mtype == 'transformed':
        return SearchStep('transform', {'xform': param_key}, sel)
    if mtype == 'moved_recolored':
        dr, dc, color = param_key
        return SearchStep('move', {'dr': dr, 'dc': dc}, sel)
        # NOTE: recolor step would need to be added separately
    return None


# ---------------------------------------------------------------------------
# Strategy 3: Stamp / creation
# ---------------------------------------------------------------------------

def _derive_stamp(all_transitions, all_facts, demos):
    """New objects whose shape matches input objects = stamp at offset."""
    results = []

    # Check all demos have new objects
    for demo_trans in all_transitions:
        if not any(t.match_type == 'new' for t in demo_trans):
            return []

    # For each new object in demo 0, find matching input object
    stamps = []
    for t in all_transitions[0]:
        if t.match_type != 'new':
            continue
        out_obj = t.out_obj
        for in_obj in all_facts[0].objects:
            if (in_obj.height == out_obj.height and in_obj.width == out_obj.width
                    and np.array_equal(in_obj.mask, out_obj.mask)):
                dr = out_obj.row - in_obj.row
                dc = out_obj.col - in_obj.col
                stamps.append((in_obj, dr, dc, out_obj.color))
                break

    if not stamps:
        return []

    # Check constant offset
    offsets = set((dr, dc) for _, dr, dc, _ in stamps)
    if len(offsets) == 1:
        dr, dc = next(iter(offsets))
        source_oids = set(s[0].oid for s in stamps)
        sel = _find_selector(source_oids, all_facts[0])
        if sel:
            prog = SearchProgram(
                steps=[SearchStep('stamp', {'dr': dr, 'dc': dc}, sel)],
                provenance='derive:stamp',
            )
            if prog.verify(demos):
                results.append(prog)

    return results


# ---------------------------------------------------------------------------
# Selector derivation
# ---------------------------------------------------------------------------

def _find_selector_for_group(all_transitions, all_facts, match_type):
    """Find a StepSelect that selects all objects of a given match_type in demo 0."""
    oids = set(t.in_obj.oid for t in all_transitions[0]
               if t.match_type == match_type and t.in_obj)
    if not oids:
        return None
    return _find_selector(oids, all_facts[0])


def _find_selector(oids, facts):
    """Find a StepSelect that selects exactly the given object IDs."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    target_set = set(oids)

    # Color-based
    target_objs = [o for o in facts.objects if o.oid in target_set]
    if target_objs:
        colors = set(o.color for o in target_objs)
        for c in colors:
            sel_preds = [Predicate(Pred.COLOR_EQ, c)]
            if set(o.oid for o in prim_select(facts, sel_preds)) == target_set:
                return StepSelect('by_color', {'color': c})

    # Structural predicates
    _pred_map = {
        Pred.IS_LARGEST: 'largest',
        Pred.IS_SMALLEST: 'smallest',
        Pred.UNIQUE_COLOR: 'unique_color',
        Pred.IS_SINGLETON: 'singleton',
        Pred.IS_RECTANGULAR: 'rectangular',
        Pred.IS_LINE: 'line',
        Pred.IS_TOPMOST: 'topmost',
        Pred.IS_BOTTOMMOST: 'bottommost',
        Pred.IS_LEFTMOST: 'leftmost',
        Pred.IS_RIGHTMOST: 'rightmost',
        Pred.TOUCHES_BORDER: 'touches_border',
        Pred.NOT_TOUCHES_BORDER: 'interior',
    }
    for pred, role in _pred_map.items():
        sel_preds = [Predicate(pred)]
        if set(o.oid for o in prim_select(facts, sel_preds)) == target_set:
            return StepSelect(role)

    # NOT predicates
    for pred in [Pred.IS_SINGLETON, Pred.IS_RECTANGULAR, Pred.IS_LINE,
                 Pred.TOUCHES_BORDER]:
        sel_preds = [Predicate(Pred.NOT, Predicate(pred))]
        if set(o.oid for o in prim_select(facts, sel_preds)) == target_set:
            return StepSelect('by_predicate', {'predicates': sel_preds})

    # Relational
    inner_options = [
        (Predicate(Pred.IS_LARGEST), 'largest'),
        (Predicate(Pred.IS_SMALLEST), 'smallest'),
        (Predicate(Pred.UNIQUE_COLOR), 'unique_color'),
    ]
    for o in facts.objects:
        inner_options.append((Predicate(Pred.COLOR_EQ, o.color), f'by_color_{o.color}'))

    for rel_pred, rel_role in [(Pred.CONTAINED_BY, 'contained_by'),
                                (Pred.ADJACENT_TO, 'adjacent_to'),
                                (Pred.CONTAINS, 'contains')]:
        for inner_pred, inner_name in inner_options:
            sel_preds = [Predicate(rel_pred, inner_pred)]
            if set(o.oid for o in prim_select(facts, sel_preds)) == target_set:
                return StepSelect(rel_role, {'inner': inner_name})

    # All objects
    if target_set == set(o.oid for o in facts.objects):
        return StepSelect('all')

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_marker_stamp(demos, all_facts):
    """Derive marker-stamp programs: small marker objects get a template stamped at their position.

    The template is learned from the diff (added pixels relative to marker centroid).
    Must be consistent across all demos.
    """
    results = []

    # Only same-shape tasks with additive changes
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    # For each demo, find added pixels (in output but not in input)
    for inp, out in demos:
        if np.sum(out != inp) == 0:
            return []  # no changes

    # Learn templates from demo 0
    inp0, out0 = demos[0]
    facts0 = all_facts[0]
    bg = facts0.bg

    # Find small objects (markers): size 1-2
    markers = [o for o in facts0.objects if o.size <= 2]
    if not markers:
        return []

    # Check: is the change additive (output = input + new pixels)?
    added = (out0 != inp0) & (inp0 == bg)
    removed = (out0 != inp0) & (out0 == bg)
    if np.sum(removed) > 0:
        return []  # not purely additive

    if np.sum(added) == 0:
        return []

    # Group markers by color
    from collections import defaultdict
    color_markers = defaultdict(list)
    for m in markers:
        color_markers[m.color].append(m)

    # For each marker color, learn template from added pixels near markers
    templates = {}  # color → {(dr, dc): pixel_color}
    for color, mlist in color_markers.items():
        per_marker_templates = []
        for m in mlist:
            cr = round(m.center_row)
            cc = round(m.center_col)
            template = {}
            for r in range(out0.shape[0]):
                for c in range(out0.shape[1]):
                    if added[r, c]:
                        dr, dc = r - cr, c - cc
                        # Only attribute pixels close to this marker
                        if abs(dr) <= max(out0.shape) // 2 and abs(dc) <= max(out0.shape) // 2:
                            template[(dr, dc)] = int(out0[r, c])
            per_marker_templates.append(template)

        # Check consistency: all markers of this color → same template
        if not per_marker_templates:
            continue
        ref = per_marker_templates[0]
        if all(t == ref for t in per_marker_templates) and ref:
            templates[color] = ref

    if not templates:
        return []

    # Verify across all demos
    all_ok = True
    for i, (inp, out) in enumerate(demos):
        facts_i = all_facts[i]
        result = _apply_marker_stamp(inp, facts_i, templates, bg)
        if not np.array_equal(result, out):
            all_ok = False
            break

    if all_ok:
        prog = SearchProgram(
            steps=[SearchStep('marker_stamp', {'templates': templates, 'bg': bg})],
            provenance='derive:marker_stamp',
        )
        if prog.verify(demos):
            results.append(prog)

    return results


def _apply_marker_stamp(grid, facts, templates, bg):
    """Apply learned marker-stamp templates to a grid."""
    result = grid.copy()
    rows, cols = grid.shape
    for obj in facts.objects:
        if obj.size > 2:
            continue
        if obj.color not in templates:
            continue
        template = templates[obj.color]
        cr = round(obj.center_row)
        cc = round(obj.center_col)
        for (dr, dc), pixel_color in template.items():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr, nc] == bg:
                result[nr, nc] = pixel_color
    return result



def _derive_symmetry_repair(demos):
    """Derive symmetry repair: find damage color that, when repaired by symmetry, produces output."""
    results = []

    # Must be same-shape
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    for damage_color in range(10):
        prog = SearchProgram(
            steps=[SearchStep('symmetry_repair', {'damage_color': damage_color})],
            provenance=f'derive:symmetry_repair_c{damage_color}',
        )
        if prog.verify(demos):
            results.append(prog)
            return results

    return results


def _derive_anomaly_halo(demos):
    """Detect isolated anomaly cells and decorate with a halo color.

    Pattern: grid has two dominant colors forming bands/regions.
    Isolated cells of the wrong color in each region are anomalies.
    In one region, anomalies get a halo of a new color around them.
    In the other region, anomalies are simply removed (filled with region color).
    """
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    inp0, out0 = demos[0]
    h, w = inp0.shape

    # Need exactly 2 dominant colors + 1 halo color in the output
    from collections import Counter
    in_counts = Counter(int(inp0[r, c]) for r in range(h) for c in range(w))
    out_counts = Counter(int(out0[r, c]) for r in range(h) for c in range(w))

    # Find the two dominant colors
    top2 = in_counts.most_common(2)
    if len(top2) < 2:
        return []
    c1, c2 = top2[0][0], top2[1][0]

    # Find the halo color (appears in output but not/barely in input)
    halo_color = None
    for color in out_counts:
        if color not in in_counts or in_counts[color] == 0:
            halo_color = color
            break
        if in_counts[color] < 5 and out_counts[color] > in_counts[color] * 3:
            halo_color = color
            break

    if halo_color is None:
        return []

    # Verify using the actual executor semantics only. Older derive-side
    # neighborhood heuristics were stricter than the runtime behavior and
    # created false negatives in search.
    from aria.search.executor import _exec_anomaly_halo
    params = {'c1': c1, 'c2': c2, 'halo_color': halo_color}
    for inp, out in demos:
        cand = _exec_anomaly_halo(inp, params)
        if not np.array_equal(cand, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('anomaly_halo', params)],
        provenance=f'derive:anomaly_halo_{c2}in{c1}_h{halo_color}',
    )
    return [prog]


def _derive_cavity_transfer(demos):
    """Transfer marker cells from inside a concave host to the opposite open end.

    Pattern: a concave host shape has marker cells inside its cavity. The output
    removes the markers and places them on the exterior, projected through the
    host's opening. The number of new cells = number of unique rows (vertical
    opening) or cols (horizontal opening) of the original markers. All new cells
    align at the opening's tip coordinate.
    """
    from aria.guided.perceive import perceive
    from collections import defaultdict

    if any(inp.shape != out.shape for inp, out in demos):
        return []

    results = []

    # Analyze demo 0 to find host-marker pairs and opening direction
    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    bg = facts0.bg
    h, w = inp0.shape

    color_cells = defaultdict(set)
    for r in range(h):
        for c in range(w):
            if inp0[r, c] != bg:
                color_cells[int(inp0[r, c])].add((r, c))

    # Find ALL host-marker pairs with their directions
    pairs = []  # list of (marker_color, host_color, direction)
    for marker_color in list(color_cells):
        mc_in = color_cells[marker_color]
        mc_out = set((r, c) for r in range(h) for c in range(w) if out0[r, c] == marker_color)
        removed = mc_in - mc_out
        added = mc_out - mc_in
        if not removed or not added:
            continue

        for host_color in list(color_cells):
            if host_color == marker_color:
                continue
            host = color_cells[host_color]
            if len(host) < 4:
                continue

            hr = [r for r, c in host]
            hc = [c for r, c in host]
            top, bot = min(hr), max(hr)
            left, right = min(hc), max(hc)

            inside = [(r, c) for r, c in removed
                      if top <= r <= bot and left <= c <= right]
            if not inside:
                continue

            for direction in ['up', 'down', 'left', 'right']:
                if direction == 'up':
                    top_cells = sorted([(r, c) for r, c in host if r == top])
                    if not top_cells:
                        continue
                    tip_col = top_cells[0][1]
                    n_unique = len(set(r for r, c in inside))
                    expected_new = set((top - 1 - i, tip_col) for i in range(n_unique))
                elif direction == 'down':
                    bot_cells = sorted([(r, c) for r, c in host if r == bot])
                    if not bot_cells:
                        continue
                    tip_col = bot_cells[0][1]
                    n_unique = len(set(r for r, c in inside))
                    expected_new = set((bot + 1 + i, tip_col) for i in range(n_unique))
                elif direction == 'right':
                    tip_row = max(r for r, c in host if c == right)
                    n_unique = len(set(c for r, c in inside))
                    expected_new = set((tip_row, right + 1 + i) for i in range(n_unique))
                elif direction == 'left':
                    tip_row = max(r for r, c in host if c == left)
                    n_unique = len(set(c for r, c in inside))
                    expected_new = set((tip_row, left - 1 - i) for i in range(n_unique))

                if expected_new and expected_new <= added:
                    pairs.append((marker_color, host_color, direction, expected_new))
                    break  # found direction for this marker/host pair

    if not pairs:
        return []

    # Emit pairs as params; verify using the parameterized executor on ALL demos.
    from aria.search.executor import _exec_cavity_transfer

    pair_tuples = [(mc, hc, d) for mc, hc, d, _ in pairs]
    cavity_params = {'pairs': pair_tuples}

    for inp, out in demos:
        result = _exec_cavity_transfer(inp, cavity_params)
        if not np.array_equal(result, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('cavity_transfer', cavity_params)],
        provenance='derive:cavity_transfer',
    )
    return [prog]


def _derive_masked_patch_transfer(demos):
    """Recover a solid rectangular mask from a transformed source patch elsewhere."""
    from aria.guided.perceive import perceive
    from aria.search.executor import _exec_masked_patch_transfer

    mask_colors: set[int] | None = None
    for inp, out in demos:
        if inp.shape == out.shape:
            return []
        facts = perceive(inp)
        candidates = {
            int(obj.color)
            for obj in facts.objects
            if obj.is_rectangular
            and np.all(obj.mask)
            and (obj.height, obj.width) == out.shape
            and obj.color != facts.bg
        }
        if not candidates:
            return []
        mask_colors = candidates if mask_colors is None else (mask_colors & candidates)
        if not mask_colors:
            return []

    for mask_color in sorted(mask_colors):
        params = {"mask_color": int(mask_color), "ring": 1}
        if all(np.array_equal(_exec_masked_patch_transfer(inp, params), out) for inp, out in demos):
            prog = SearchProgram(
                steps=[SearchStep("masked_patch_transfer", params)],
                provenance=f"derive:masked_patch_transfer_c{mask_color}",
            )
            return [prog]
    return []


def _derive_separator_motif_broadcast(demos):
    """Detect separator-line motifs broadcast into an empty opposite side.

    Computes sep_axis and sep_idx at derive time so the executor can skip
    auto-detection at test time.
    """
    from aria.search.executor import (
        _broadcast_run_score,
        _exec_separator_motif_broadcast,
        _mono_separator_candidates,
    )
    from aria.guided.perceive import perceive

    if any(inp.shape != out.shape for inp, out in demos):
        return []

    # --- Derive sep_axis and sep_idx from demo 0 ---
    inp0, _ = demos[0]
    facts0 = perceive(inp0)
    bg0 = int(facts0.bg)
    row_seps = sorted({s.index for s in facts0.separators if s.axis == "row"})
    col_seps = sorted({s.index for s in facts0.separators if s.axis == "col"})
    row_candidates = sorted(set(row_seps) | set(_mono_separator_candidates(inp0, "row", bg0)))
    col_candidates = sorted(set(col_seps) | set(_mono_separator_candidates(inp0, "col", bg0)))

    candidates: list[tuple[str, int, int]] = []
    for idx in row_candidates:
        candidates.append(("row", idx, _broadcast_run_score(inp0, "row", idx, bg0)))
    for idx in col_candidates:
        candidates.append(("col", idx, _broadcast_run_score(inp0, "col", idx, bg0)))
    candidates = [c for c in candidates if c[2] > 0]
    if not candidates:
        return []

    sep_axis, sep_idx, _ = max(candidates, key=lambda item: (item[2], item[0] == "col"))

    params = {"sep_axis": sep_axis, "sep_idx": int(sep_idx)}
    if all(np.array_equal(_exec_separator_motif_broadcast(inp, params), out) for inp, out in demos):
        prog = SearchProgram(
            steps=[SearchStep("separator_motif_broadcast", params)],
            provenance="derive:separator_motif_broadcast",
        )
        return [prog]
    return []


def _derive_line_arith_broadcast(demos):
    """Detect mixed-axis arithmetic line broadcast from sparse line scaffolds."""
    from aria.search.executor import _exec_line_arith_broadcast

    if any(inp.shape != out.shape for inp, out in demos):
        return []

    params = {"axis": "auto"}
    if all(np.array_equal(_exec_line_arith_broadcast(inp, params), out) for inp, out in demos):
        prog = SearchProgram(
            steps=[SearchStep("line_arith_broadcast", params)],
            provenance="derive:line_arith_broadcast",
        )
        return [prog]
    return []


def _derive_barrier_port_transfer(demos):
    """Detect object relocation through barrier opening families.

    Pre-computes barrier_color, barrier_orient, and bg from demo 0 so
    the executor can skip auto-detection at test time.
    """
    from aria.search.executor import _detect_barrier_params, _exec_barrier_port_transfer

    if any(inp.shape != out.shape for inp, out in demos):
        return []

    # --- Extract barrier structural params from demo 0 ---
    inp0, _ = demos[0]
    detected = _detect_barrier_params(inp0)
    if detected is None:
        return []

    barrier_color, barrier_orient, bg = detected
    params = {
        "barrier_color": int(barrier_color),
        "barrier_orient": barrier_orient,
        "bg": int(bg),
    }
    if all(np.array_equal(_exec_barrier_port_transfer(inp, params), out) for inp, out in demos):
        prog = SearchProgram(
            steps=[SearchStep("barrier_port_transfer", params)],
            provenance="derive:barrier_port_transfer",
        )
        return [prog]
    return []


def _derive_legend_frame_fill(demos):
    """Fill bg cells enclosed by colored boundaries.

    For each non-bg color, flood-fill bg from the grid border. Any bg cells
    NOT reached are enclosed by that color's boundary. Fill them with a
    derived color (from a legend, or the boundary color itself).

    Pre-computes legend_map and wall_colors from demo 0 so the executor
    can skip auto-detection at test time.
    """
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    inp0, out0 = demos[0]

    changed = (inp0 != out0)
    if changed.sum() == 0:
        return []

    # --- Extract legend_map and wall_colors from demo 0 ---
    from aria.search.executor import _extract_enclosed_fill_params, _exec_legend_frame_fill

    extracted = _extract_enclosed_fill_params(inp0)
    if extracted is None:
        return []

    legend_map, wall_colors = extracted
    params = {
        'strategy': 'enclosed_fill',
        'legend_map': {int(k): int(v) for k, v in legend_map.items()},
        'wall_colors': sorted(int(c) for c in wall_colors),
    }

    for inp, out in demos:
        cand = _exec_legend_frame_fill(inp, params)
        if not np.array_equal(cand, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('legend_frame_fill', params)],
        provenance='derive:legend_enclosed_fill',
    )
    return [prog]


def _derive_diagonal_collision_trace(demos):
    """Detect diagonal ray tasks with corner/point emitters and bar/point bounces."""
    from scipy import ndimage

    from aria.search.executor import _exec_diagonal_collision_trace

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    def _classify(coords):
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        if len(coords) == 1:
            return 'point'
        if len(coords) == 3 and h == 1 and w == 3:
            return 'hbar'
        if len(coords) == 3 and h == 3 and w == 1:
            return 'vbar'
        if len(coords) == 3 and h == 2 and w == 2:
            return 'corner'
        return None

    saw_emitter = False
    saw_reflector = False
    for inp, out in demos:
        if inp.shape != out.shape:
            return []
        supported = True
        for color in sorted(int(v) for v in np.unique(inp) if v != 0):
            labels, count = ndimage.label(inp == color, structure=structure)
            for idx in range(1, count + 1):
                coords = [tuple(x) for x in np.argwhere(labels == idx).tolist()]
                kind = _classify(coords)
                if kind is None:
                    supported = False
                    break
                saw_emitter = saw_emitter or (kind in ('corner', 'point'))
                saw_reflector = saw_reflector or (kind in ('hbar', 'vbar', 'point'))
            if not supported:
                break
        if not supported:
            return []

    if not (saw_emitter and saw_reflector):
        return []

    params = {'point_dir': 'up_right', 'include_direct_hit': True}
    for inp, out in demos:
        pred = _exec_diagonal_collision_trace(inp, params)
        if not np.array_equal(pred, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('diagonal_collision_trace', params)],
        provenance='derive:diagonal_collision_trace',
    )
    return [prog]


def _derive_cross_stencil_recolor(demos):
    """Detect plus/cross patterns and recolor them.

    Scans for cells where all 4 orthogonal neighbors have the same non-bg color.
    Recolors the 5-cell cross to a new color. Verifies across demos.
    """
    from aria.guided.perceive import perceive

    # Need same-shape IO
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    bg = facts0.bg

    # Find changed cells in demo 0
    diff_mask = (inp0 != out0)
    if diff_mask.sum() == 0:
        return []

    # All changes must be same old_color → same new_color
    changed = list(zip(*np.where(diff_mask)))
    old_colors = set(int(inp0[r, c]) for r, c in changed)
    new_colors = set(int(out0[r, c]) for r, c in changed)
    if len(old_colors) != 1 or len(new_colors) != 1:
        return []

    old_color = old_colors.pop()
    new_color = new_colors.pop()
    if old_color == bg or new_color == bg:
        return []

    # Find cross centers: cells where all 4 orthogonal neighbors = old_color
    h, w = inp0.shape
    cross_centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if inp0[r, c] != old_color:
                continue
            if (inp0[r-1, c] == old_color and inp0[r+1, c] == old_color and
                inp0[r, c-1] == old_color and inp0[r, c+1] == old_color):
                cross_centers.append((r, c))

    if not cross_centers:
        return []

    # Build expected output: recolor all cross cells
    candidate = inp0.copy()
    for cr, cc in cross_centers:
        candidate[cr, cc] = new_color
        candidate[cr-1, cc] = new_color
        candidate[cr+1, cc] = new_color
        candidate[cr, cc-1] = new_color
        candidate[cr, cc+1] = new_color

    if not np.array_equal(candidate, out0):
        return []

    # Verify on all demos
    for inp, out in demos[1:]:
        facts = perceive(inp)
        bg_i = facts.bg
        hi, wi = inp.shape

        # Find cross centers in this demo
        centers = []
        for r in range(1, hi - 1):
            for c in range(1, wi - 1):
                if inp[r, c] != old_color:
                    continue
                if (inp[r-1, c] == old_color and inp[r+1, c] == old_color and
                    inp[r, c-1] == old_color and inp[r, c+1] == old_color):
                    centers.append((r, c))

        cand = inp.copy()
        for cr, cc in centers:
            cand[cr, cc] = new_color
            cand[cr-1, cc] = new_color
            cand[cr+1, cc] = new_color
            cand[cr, cc-1] = new_color
            cand[cr, cc+1] = new_color

        if not np.array_equal(cand, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('cross_stencil_recolor',
                           {'old_color': old_color, 'new_color': new_color})],
        provenance=f'derive:cross_stencil_recolor_{old_color}to{new_color}',
    )
    return [prog]


def _derive_frame_bbox_pack(demos):
    """Derive programs that pack framed-object bboxes into a grid layout.

    Detects framed objects, extracts their bboxes, tries to arrange them
    into a grid matching the output dimensions.
    """
    from aria.search.frames import extract_rect_items, render_rect_family_side

    results = []

    # Analyze demo 0
    inp0, out0 = demos[0]
    bg0 = int(np.bincount(inp0.ravel()).argmax())
    frame_items0 = extract_rect_items(inp0, bg=bg0, min_span=4)
    frame_infos = [
        {
            'bbox': item.patch,
            'color': item.color,
            'row': item.row,
            'col': item.col,
            'interior_bg': item.interior_bg,
            'kind': item.kind,
        }
        for item in frame_items0
    ]

    if len(frame_infos) < 2:
        return []

    # All bboxes must be same shape
    bbox_shapes = set(f['bbox'].shape for f in frame_infos)
    if len(bbox_shapes) != 1:
        return []
    bh, bw = list(bbox_shapes)[0]

    oh, ow = out0.shape
    n_frames = len(frame_infos)

    family_colors0 = sorted({item.color for item in frame_items0})
    if inp0.shape == out0.shape and len(family_colors0) == 2:
        family_orders = [
            ("desc", sorted(family_colors0, reverse=True)),
            ("asc", sorted(family_colors0)),
        ]
        for family_order_name, family_order in family_orders:
            candidate = np.full_like(out0, int(np.bincount(out0.ravel()).argmax()))
            ok = True
            for side, color in zip(("left", "right"), family_order):
                group = [item for item in frame_items0 if item.color == color]
                if not group:
                    ok = False
                    break
                family_canvas = render_rect_family_side(group, shape=out0.shape, bg=bg0, side=side)
                mask = family_canvas != bg0
                candidate[mask] = family_canvas[mask]
            if ok and np.array_equal(candidate, out0):
                verified = True
                for inp, out in demos[1:]:
                    bg_i = int(np.bincount(inp.ravel()).argmax())
                    items_i = extract_rect_items(inp, bg=bg_i, min_span=4)
                    colors_i = sorted({item.color for item in items_i})
                    if sorted(colors_i) != sorted(family_order):
                        verified = False
                        break
                    cand = np.full_like(out, int(np.bincount(out.ravel()).argmax()))
                    for side, color in zip(("left", "right"), family_order):
                        group = [item for item in items_i if item.color == color]
                        if not group:
                            verified = False
                            break
                        family_canvas = render_rect_family_side(group, shape=out.shape, bg=bg_i, side=side)
                        mask = family_canvas != bg_i
                        cand[mask] = family_canvas[mask]
                    if not verified or not np.array_equal(cand, out):
                        verified = False
                        break
                if verified:
                    prog = SearchProgram(
                        steps=[SearchStep(
                            'frame_bbox_pack',
                            {
                                'mode': 'family_side_lanes',
                                'family_order': family_order_name,
                            },
                        )],
                        provenance=f'derive:frame_bbox_pack_family_side_lanes_{family_order_name}',
                    )
                    results.append(prog)
                    return results

    # Compute grid dimensions directly from output shape and block size
    if oh % bh != 0 or ow % bw != 0:
        return []
    nr, nc = oh // bh, ow // bw
    if nr * nc < n_frames:
        return []
    grid_configs = [(nr, nc)]

    # Try orderings: by row, by col, by color, by interior classification
    orderings = [
        ('row', lambda fs: sorted(fs, key=lambda f: (f['row'], f['col']))),
        ('col', lambda fs: sorted(fs, key=lambda f: (f['col'], f['row']))),
        ('color', lambda fs: sorted(fs, key=lambda f: f['color'])),
    ]

    # Also try: split into bg-interior and non-bg-interior groups, pack in 2 columns
    has_groups = any(f['interior_bg'] for f in frame_infos) and any(not f['interior_bg'] for f in frame_infos)

    if has_groups:
        empty = sorted([f for f in frame_infos if f['interior_bg']], key=lambda f: (f['row'], f['col']))
        filled = sorted([f for f in frame_infos if not f['interior_bg']], key=lambda f: (f['row'], f['col']))

        def _group_columns(fs):
            # Pack: row i = [empty[i], filled[i]]
            result = []
            for i in range(max(len(empty), len(filled))):
                result.append(empty[i] if i < len(empty) else None)
                result.append(filled[i] if i < len(filled) else None)
            return result

        orderings.append(('group_cols', _group_columns))

    for nr, nc in grid_configs:
        for ord_name, ord_fn in orderings:
            ordered = ord_fn(frame_infos)

            # Build output grid
            out_bg0 = int(np.bincount(out0.ravel()).argmax())
            candidate = np.full((oh, ow), out_bg0, dtype=out0.dtype)
            for idx in range(nr * nc):
                r_block = idx // nc
                c_block = idx % nc
                if idx < len(ordered) and ordered[idx] is not None:
                    bbox = ordered[idx]['bbox']
                    candidate[r_block * bh:(r_block + 1) * bh,
                              c_block * bw:(c_block + 1) * bw] = bbox

            if np.array_equal(candidate, out0):
                # Verify on all demos
                all_ok = True
                for di, (inp, out) in enumerate(demos[1:], 1):
                    bg_i = int(np.bincount(inp.ravel()).argmax())
                    finfos = [
                        {
                            'bbox': item.patch,
                            'color': item.color,
                            'row': item.row,
                            'col': item.col,
                            'interior_bg': item.interior_bg,
                            'kind': item.kind,
                        }
                        for item in extract_rect_items(inp, bg=bg_i, min_span=4)
                        if item.patch.shape == (bh, bw)
                    ]

                    if not finfos:
                        all_ok = False
                        break

                    if has_groups and ord_name == 'group_cols':
                        emp = sorted([f for f in finfos if f['interior_bg']], key=lambda f: (f['row'], f['col']))
                        fil = sorted([f for f in finfos if not f['interior_bg']], key=lambda f: (f['row'], f['col']))
                        o = []
                        for j in range(max(len(emp), len(fil))):
                            o.append(emp[j] if j < len(emp) else None)
                            o.append(fil[j] if j < len(fil) else None)
                    else:
                        o = ord_fn(finfos)

                    oh_i, ow_i = out.shape
                    nr_i = oh_i // bh
                    nc_i = ow_i // bw
                    out_bg = int(np.bincount(out.ravel()).argmax())
                    cand = np.full((oh_i, ow_i), out_bg, dtype=inp.dtype)
                    for idx in range(nr_i * nc_i):
                        rb = idx // nc_i
                        cb = idx % nc_i
                        if idx < len(o) and o[idx] is not None:
                            cand[rb * bh:(rb + 1) * bh, cb * bw:(cb + 1) * bw] = o[idx]['bbox']

                    if not np.array_equal(cand, out):
                        all_ok = False
                        break

                if all_ok:
                    prog = SearchProgram(
                        steps=[SearchStep('frame_bbox_pack',
                                           {'ordering': ord_name, 'block_h': bh, 'block_w': bw,
                                            'grid_cols': nc})],
                        provenance=f'derive:frame_bbox_pack_{ord_name}_nc{nc}',
                    )
                    results.append(prog)
                    return results

    return results


def _derive_object_repack(demos):
    """Derive object-repack programs: output = objects repacked into a new layout."""
    results = []

    # Try parameter combinations
    for ordering in ['chain', 'spatial', 'size']:
        for layout in ['column', 'row']:
            for payload in ['color_by_area']:
                params = {'ordering': ordering, 'layout': layout, 'payload': payload}
                prog = SearchProgram(
                    steps=[SearchStep('object_repack', params)],
                    provenance=f'derive:repack_{ordering}_{layout}_{payload}',
                )
                if prog.verify(demos):
                    results.append(prog)
                    return results  # first verified wins

    return results


def _at_border(obj, direction, rows, cols):
    if direction == 'down':
        return obj.row + obj.height == rows
    if direction == 'up':
        return obj.row == 0
    if direction == 'right':
        return obj.col + obj.width == cols
    if direction == 'left':
        return obj.col == 0
    return False


def _moved_toward(t, direction):
    if direction == 'down':
        return t.dr > 0
    if direction == 'up':
        return t.dr < 0
    if direction == 'right':
        return t.dc > 0
    if direction == 'left':
        return t.dc < 0
    return False


def _detect_gravity(transitions, rows, cols):
    for d in ('down', 'up', 'right', 'left'):
        if all(_at_border(t.out_obj, d, rows, cols) for t in transitions):
            if any(_moved_toward(t, d) for t in transitions):
                return d
    return None


def _get_support_color(facts, role):
    """Get the color of the support object identified by role."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred
    role_map = {
        'largest': [Predicate(Pred.IS_LARGEST)],
        'smallest': [Predicate(Pred.IS_SMALLEST)],
        'unique_color': [Predicate(Pred.UNIQUE_COLOR)],
    }
    preds = role_map.get(role)
    if preds is None:
        return None
    objs = prim_select(facts, preds)
    return objs[0].color if objs else None
