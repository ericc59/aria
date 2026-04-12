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
    # NOTE: mid-level benchmark-shaped strategies are quarantined from default
    # derive routing; they remain in the codebase for macro/replay use.
    progs = _derive_frame_bbox_pack(demos)
    if progs:
        return progs

    progs = _derive_masked_patch_transfer(demos)
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

    # Strategy: Object grid pack (repack objects into different-size output)
    progs = _derive_object_grid_pack_prescan(demos)
    if progs:
        return progs

    # Remaining strategies require same-shape
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    # Quarantined: diagonal_collision_trace (benchmark-shaped)

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

    # Strategy 1b: Rank/group recolor (all recolored by size rank — more
    # specific than dispatch, runs first to avoid accidental partition matches)
    if not results:
        progs = _derive_rank_recolor(all_transitions, all_facts, demos)
        results.extend(progs)

    # Strategy 2: Multi-group dispatch (different groups → different actions)
    progs = _derive_dispatch(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 2b: Conditional dispatch (same action type, different params per predicate group)
    progs = _derive_conditional_dispatch(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 2c: Action-first dispatch (group by observed action, find selectors)
    if not results:
        progs = _derive_action_first_dispatch(all_transitions, all_facts, demos)
        results.extend(progs)

    # Strategy 2f: Registration transfer (modules move into frame openings)
    if not results:
        progs = _derive_registration_transfer(all_transitions, all_facts, demos)
        results.extend(progs)

    # Strategy 2g: Anchor-based registration transfer (modules align to anchor sites)
    if not results:
        progs = _derive_anchor_registration_transfer(all_facts, demos)
        results.extend(progs)

    # Strategy 2h: Grid-cell broadcast (content replicated within grid rows/cols)
    if not results:
        progs = _derive_grid_broadcast(all_facts, demos)
        results.extend(progs)

    # Strategy 2i: Grid-cell pack (pack non-empty cells into a grid order)
    if not results:
        progs = _derive_grid_cell_pack(all_facts, demos)
        results.extend(progs)

    # Strategy 2j: Grid-slot transfer (move cell contents into empty slots)
    if not results:
        progs = _derive_grid_slot_transfer(all_facts, demos)
        results.extend(progs)

    # Strategy 2k: Grid-conditional transfer (fill empty cells by rule)
    if not results:
        progs = _derive_grid_conditional_transfer(all_facts, demos)
        results.extend(progs)

    # Strategy 2m: Panel legend map (legend → color mapping on target panel)
    if not results:
        progs = _derive_panel_legend_map(all_facts, demos)
        results.extend(progs)

    # Strategy 3: Stamp/creation (new objects from input object shapes)
    progs = _derive_stamp(all_transitions, all_facts, demos)
    results.extend(progs)

    # Strategy 4: Marker stamp (small markers → learned template stamped at each)
    # Quarantined from default derive (benchmark-shaped); keep for macro/replay.


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

    # Try: cross-demo rule-induction crop (output = bbox of an object
    # selected by a conjunction of structural features)
    target_oids_per_demo = []
    all_crop_facts = []
    all_crop_ok = True
    for inp, out in demos:
        facts = perceive(inp)
        all_crop_facts.append(facts)
        oh_d, ow_d = out.shape
        found_obj = None
        for obj in facts.objects:
            if obj.height == oh_d and obj.width == ow_d:
                crop = inp[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
                if np.array_equal(crop, out):
                    found_obj = obj
                    break
        if found_obj is None:
            all_crop_ok = False
            break
        target_oids_per_demo.append({found_obj.oid})

    if all_crop_ok and target_oids_per_demo:
        sel = _find_selector_for_oid_sets(target_oids_per_demo, all_crop_facts)
        if sel is not None:
            prog = SearchProgram(
                steps=[SearchStep('crop_object', {'predicate': 'by_rule'}, select=sel)],
                provenance='derive:crop_object_rule',
            )
            if prog.verify(demos):
                return [prog]

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


# ---------------------------------------------------------------------------
# Strategy 2c: Action-first dispatch
# ---------------------------------------------------------------------------

def _derive_action_first_dispatch(all_transitions, all_facts, demos):
    """Conditional dispatch via action-first grouping.

    Groups changed objects by their observed action across ALL demos,
    then finds cross-demo selectors for each action group.

    Handles cases where selector-first partitioning fails to generalize
    (e.g., color-based partitions in demo 0 that don't hold in demo 1).
    """
    # Group by match_type (coarse) across all demos
    per_demo_groups = []
    for trans in all_transitions:
        groups = defaultdict(set)
        for t in trans:
            if t.match_type == 'identical' or not t.in_obj:
                continue
            groups[t.match_type].add(t.in_obj.oid)
        per_demo_groups.append(groups)

    # Consistent match_type grouping across demos
    keys_0 = set(per_demo_groups[0].keys())
    if len(keys_0) < 2:
        return []
    for pdg in per_demo_groups[1:]:
        if set(pdg.keys()) != keys_0:
            return []

    bg = _infer_bg_from_demos(demos)
    steps = []

    for mtype in sorted(keys_0):
        target_oids_per_demo = [pdg.get(mtype, set()) for pdg in per_demo_groups]

        # Find a cross-demo selector
        sel = _find_selector_for_oid_sets(target_oids_per_demo, all_facts)
        if sel is None:
            return []

        # Derive the action for this group
        all_group_trans = []
        for di, trans in enumerate(all_transitions):
            oids = target_oids_per_demo[di]
            gt = [t for t in trans if t.in_obj and t.in_obj.oid in oids]
            all_group_trans.append(gt)

        step = _derive_group_step_multi(all_group_trans, sel, demos, bg)
        if step is None:
            return []
        steps.append(step)

    if len(steps) < 2:
        return []

    prog = SearchProgram(steps=steps, provenance='derive:action_first_dispatch')
    if prog.verify(demos):
        return [prog]
    return []


# ---------------------------------------------------------------------------
# Strategy 2d: Rank-recolor
# ---------------------------------------------------------------------------

def _derive_rank_recolor(all_transitions, all_facts, demos):
    """All changed objects recolored by size rank or size group.

    Detects: all objects get match_type='recolored', target colors
    map consistently to size rank across demos. Handles both:
    - strict rank (each object gets a unique color by rank)
    - size groups (objects with same size get same color)

    Emits N separate recolor steps, each with a per-group selector.
    """
    # Check: all changed objects are recolored
    for trans in all_transitions:
        changed = [t for t in trans if t.match_type != 'identical' and t.in_obj]
        if not changed:
            return []
        if not all(t.match_type == 'recolored' for t in changed):
            return []

    # In each demo, build size_group → color mapping (deduplicated)
    group_maps = []
    for trans in all_transitions:
        size_to_color = {}
        consistent = True
        for t in trans:
            if t.match_type != 'recolored' or not t.in_obj:
                continue
            sz = t.in_obj.size
            if sz in size_to_color:
                if size_to_color[sz] != t.color_to:
                    consistent = False
                    break
            else:
                size_to_color[sz] = t.color_to
        if not consistent:
            return []
        # Sort by size descending → color sequence (deduped)
        group_seq = tuple(c for _, c in sorted(size_to_color.items(), key=lambda x: -x[0]))
        group_maps.append((size_to_color, group_seq))

    if not group_maps:
        return []
    n_groups = len(group_maps[0][1])
    if n_groups < 2:
        return []
    ref_seq = group_maps[0][1]
    for _, seq in group_maps[1:]:
        if seq != ref_seq:
            return []

    # Build per-group selectors and recolor steps
    steps = []
    ref_sizes = sorted(group_maps[0][0].keys(), reverse=True)

    for gi, ref_size in enumerate(ref_sizes):
        target_color = group_maps[0][0][ref_size]

        # Collect OIDs for objects in this size group in each demo
        target_oids_per_demo = []
        for di, trans in enumerate(all_transitions):
            demo_size = sorted(group_maps[di][0].keys(), reverse=True)
            if gi >= len(demo_size):
                return []
            group_size = demo_size[gi]
            oids = {t.in_obj.oid for t in trans
                    if t.match_type == 'recolored' and t.in_obj
                    and t.in_obj.size == group_size}
            if not oids:
                return []
            target_oids_per_demo.append(oids)

        sel = _find_selector_for_oid_sets(target_oids_per_demo, all_facts)
        if sel is None:
            return []

        steps.append(SearchStep('recolor', {'color': target_color}, sel))

    prog = SearchProgram(steps=steps, provenance='derive:rank_recolor')
    if prog.verify(demos):
        return [prog]
    return []


# ---------------------------------------------------------------------------
# Frame opening detection (shared by derive + execution)
# ---------------------------------------------------------------------------

def _find_frame_openings(grid, frames, bg):
    """Find interior openings in frame objects.

    Excludes exterior bg cells by flood-filling from the bbox border.
    Only keeps bg regions fully enclosed by the frame.

    Returns list of (frame_obj, global_r0, global_c0, h, w, n_cells, mask).
    """
    from collections import deque
    from scipy import ndimage

    openings = []
    for f in frames:
        patch = grid[f.row:f.row + f.height, f.col:f.col + f.width]
        bg_mask = (patch == bg)
        if not np.any(bg_mask):
            continue

        # Flood-fill exterior bg from bbox borders
        h, w = patch.shape
        exterior = np.zeros((h, w), dtype=bool)
        q = deque()
        for r in range(h):
            for c in (0, w - 1):
                if bg_mask[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if bg_mask[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not exterior[nr, nc]:
                    exterior[nr, nc] = True
                    q.append((nr, nc))

        # Interior = bg minus exterior
        interior = bg_mask & ~exterior
        if not np.any(interior):
            continue

        labels, n = ndimage.label(interior)
        for li in range(1, n + 1):
            cells = list(zip(*np.where(labels == li)))
            r0 = min(r for r, c in cells)
            c0 = min(c for r, c in cells)
            r1 = max(r for r, c in cells)
            c1 = max(c for r, c in cells)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            omask = np.zeros((oh, ow), dtype=bool)
            for r, c in cells:
                omask[r - r0, c - c0] = True
            openings.append((f, r0 + f.row, c0 + f.col, oh, ow, len(cells), omask))

    return openings


# ---------------------------------------------------------------------------
# Strategy 2f: Registration transfer (modules into frame openings)
# ---------------------------------------------------------------------------

def _derive_registration_transfer(all_transitions, all_facts, demos):
    """Modules move into frame openings based on shape fit.

    Detects: large non-rectangular objects (frames) with bg-cell openings,
    plus small objects (modules) that move to fill those openings.
    Matches each module to the frame opening with the closest shape fit.
    Emits per-module move steps.
    """
    from scipy import ndimage

    # All demos must have moved objects
    for trans in all_transitions:
        moved = [t for t in trans if t.match_type in ('moved', 'moved_recolored') and t.in_obj]
        if not moved:
            return []

    # Verify pattern in each demo: frames with openings + modules that fill them
    all_move_steps = []  # per-demo list of (module_oid, dr, dc)

    for di, (inp, out) in enumerate(demos):
        facts = all_facts[di]
        trans = all_transitions[di]
        bg = facts.bg

        moved = [t for t in trans if t.match_type in ('moved', 'moved_recolored') and t.in_obj]
        if not moved:
            return []

        # Find frames: large non-rectangular objects
        frames = [o for o in facts.objects if not o.is_rectangular and o.size >= 8]
        if not frames:
            return []

        # Find interior openings in each frame (exclude exterior bg)
        frame_openings = _find_frame_openings(inp, frames, bg)

        if not frame_openings:
            return []

        # Match each moved module to nearest compatible frame opening
        # (same matching rule as execution: nearest by center distance)
        demo_moves = []
        used_openings = set()
        for t in moved:
            m = t.in_obj
            best_opening = None
            best_dist = float('inf')
            for oi, (frame, or0, oc0, oh, ow, nc, omask) in enumerate(frame_openings):
                if oi in used_openings:
                    continue
                if m.height == oh and m.width == ow and m.size == nc:
                    if np.array_equal(m.mask, omask):
                        dist = abs(m.center_row - (or0 + oh / 2)) + \
                               abs(m.center_col - (oc0 + ow / 2))
                        if dist < best_dist:
                            best_dist = dist
                            best_opening = oi

            if best_opening is None:
                return []

            used_openings.add(best_opening)
            frame, or0, oc0, oh, ow, nc, omask = frame_openings[best_opening]
            dr = or0 - m.row
            dc = oc0 - m.col
            demo_moves.append((m.oid, dr, dc))

        all_move_steps.append(demo_moves)

    # Check cross-demo consistency: same number of moves
    n_moves = len(all_move_steps[0])
    if any(len(dm) != n_moves for dm in all_move_steps):
        return []

    # Build per-module move steps
    # We can't use fixed offsets because they vary per demo.
    # Instead, emit a single 'registration_transfer' step that re-derives
    # at execution time.
    prog = SearchProgram(
        steps=[SearchStep('registration_transfer', {})],
        provenance='derive:registration_transfer',
    )
    if prog.verify(demos):
        return [prog]
    return []


# ---------------------------------------------------------------------------
# Strategy 2g: Anchor-based registration transfer (modules to anchor sites)
# ---------------------------------------------------------------------------

def _derive_anchor_registration_transfer(all_facts, demos):
    """Anchor-based registration using precomputed anchored shapes.

    Detects: a base shape with anchor sites plus a single movable module
    that relocates to a specific anchor site. Uses exact shape+anchor
    overlay candidates to match the output.
    """
    from aria.search.registration import (
        base_registration_patch,
        cluster_movable_modules,
        extract_anchored_shapes,
        module_anchor_patch,
        overlay_registration_candidates,
    )

    # Candidate color pairs from demo 0
    inp0 = demos[0][0]
    colors = sorted(set(int(v) for v in inp0.flatten()))
    if len(colors) < 2:
        return []

    for shape_color in colors:
        for anchor_color in colors:
            if anchor_color == shape_color:
                continue

            all_ok = True
            module_count_ref = None

            for di, (inp, out) in enumerate(demos):
                facts = all_facts[di]
                if shape_color == facts.bg or anchor_color == facts.bg:
                    all_ok = False
                    break
                shapes = extract_anchored_shapes(
                    inp, shape_color=shape_color, anchor_color=anchor_color,
                )
                if not any(bool(s.anchors_global) for s in shapes):
                    all_ok = False
                    break
                base_idx, modules = cluster_movable_modules(shapes)
                if base_idx is None or not modules:
                    all_ok = False
                    break
                if module_count_ref is None:
                    module_count_ref = len(modules)
                elif len(modules) != module_count_ref:
                    all_ok = False
                    break

                base_patch, target_sites = base_registration_patch(
                    shapes[base_idx], shape_color=shape_color,
                )
                if not shapes[base_idx].anchors_global:
                    all_ok = False
                    break

                # Nearest-anchor assignment: module anchors to target sites
                target_sites_global = [
                    (shapes[base_idx].row + r, shapes[base_idx].col + c)
                    for r, c in target_sites
                ]
                if len(modules) > len(target_sites_global):
                    all_ok = False
                    break
                from aria.search.registration import module_anchor_centroid, module_anchor_origin
                from scipy.optimize import linear_sum_assignment
                cost = np.zeros((len(modules), len(target_sites_global)), dtype=float)
                for mi, module in enumerate(modules):
                    mr, mc = module_anchor_centroid(shapes, module)
                    for ti, (tr, tc) in enumerate(target_sites_global):
                        cost[mi, ti] = abs(mr - tr) + abs(mc - tc)
                rows, cols = linear_sum_assignment(cost)
                # Build candidate output by moving modules to assigned sites
                result = inp.copy()
                used_sites = set()
                for mi, ti in zip(rows, cols):
                    if ti in used_sites:
                        continue
                    used_sites.add(ti)
                    module = modules[mi]
                    module_anchors_global = tuple(
                        sorted({(ar, ac) for i in module.component_indices
                                for ar, ac in shapes[i].anchors_global})
                    )
                    if not module_anchors_global:
                        all_ok = False
                        break
                    target_global = target_sites_global[ti]
                    best_anchor = None
                    best_dist = float('inf')
                    for ma in module_anchors_global:
                        dist = abs(ma[0] - target_global[0]) + abs(ma[1] - target_global[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_anchor = ma
                    if best_anchor is None:
                        all_ok = False
                        break
                    r0, c0 = module_anchor_origin(shapes, module)
                    chosen_source_anchor = (best_anchor[0] - r0, best_anchor[1] - c0)
                    dr = target_global[0] - best_anchor[0]
                    dc = target_global[1] - best_anchor[1]

                    # Erase module from old position
                    for obj_idx in module.component_indices:
                        obj = shapes[obj_idx]
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.patch[r, c] == obj.color:
                                    result[obj.row + r, obj.col + c] = facts.bg
                    # Place module at new position
                    for obj_idx in module.component_indices:
                        obj = shapes[obj_idx]
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.patch[r, c] == obj.color:
                                    nr = obj.row + r + dr
                                    nc = obj.col + c + dc
                                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                        result[nr, nc] = obj.color
                if not all_ok:
                    break
                if not np.array_equal(result, out):
                    all_ok = False
                    break

            if all_ok:
                prog = SearchProgram(
                    steps=[SearchStep('registration_anchor_transfer', {
                        'shape_color': int(shape_color),
                        'anchor_color': int(anchor_color),
                    })],
                    provenance='derive:registration_anchor_transfer',
                )
                if prog.verify(demos):
                    return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 2h: Grid-cell broadcast
# ---------------------------------------------------------------------------

def _derive_grid_broadcast(all_facts, demos):
    """Fill empty cells between repeated grid-cell content along rows/cols.

    Detects: a separator-defined grid. For each row/col, if two cells
    share the same non-bg content color (mode='color') or identical
    content pattern (mode='pattern'), all empty cells between them
    are filled with that content in the output.
    """
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
        cell_content_color,
    )

    # All demos must have a separator grid
    for di, (inp, out) in enumerate(demos):
        facts = all_facts[di]
        grid_info = detect_grid(facts)
        if grid_info is None:
            return []

    def _build_result(inp, facts, grid_info, mode):
        bg = facts.bg
        result = inp.copy()

        for axis in ('row', 'col'):
            n_lines = grid_info.n_rows if axis == 'row' else grid_info.n_cols
            n_cross = grid_info.n_cols if axis == 'row' else grid_info.n_rows

            for line in range(n_lines):
                keys_at = {}
                content_at = {}
                for cross in range(n_cross):
                    gr = line if axis == 'row' else cross
                    gc = cross if axis == 'row' else line
                    cell = grid_info.cell_at(gr, gc)
                    if cell and cell_has_content(inp, cell, bg):
                        content = cell_content(inp, cell, bg)
                        if mode == 'pattern':
                            key = (content.shape, content.tobytes())
                        else:
                            color = cell_content_color(inp, cell, bg)
                            if color is None:
                                continue
                            key = ('color', int(color))
                        keys_at[cross] = key
                        content_at[cross] = content

                key_positions = {}
                for pos, key in keys_at.items():
                    key_positions.setdefault(key, []).append(pos)

                for key, positions in key_positions.items():
                    if len(positions) < 2:
                        continue
                    positions.sort()
                    lo, hi = positions[0], positions[-1]
                    src = content_at[lo]
                    for cross in range(lo, hi + 1):
                        if cross not in keys_at:
                            gr = line if axis == 'row' else cross
                            gc = cross if axis == 'row' else line
                            cell = grid_info.cell_at(gr, gc)
                            if cell:
                                h, w = cell.height, cell.width
                                sh, sw = src.shape
                                fh, fw = min(h, sh), min(w, sw)
                                result[cell.r0:cell.r0 + fh,
                                       cell.c0:cell.c0 + fw] = src[:fh, :fw]

        return result

    for mode in ('color', 'pattern'):
        all_ok = True
        for di, (inp, out) in enumerate(demos):
            facts = all_facts[di]
            grid_info = detect_grid(facts)
            if grid_info is None:
                all_ok = False
                break
            result = _build_result(inp, facts, grid_info, mode)
            if not np.array_equal(result, out):
                all_ok = False
                break

        if all_ok:
            prog = SearchProgram(
                steps=[SearchStep('grid_fill_between', {'mode': mode})],
                provenance=f'derive:grid_fill_between_{mode}',
            )
            if prog.verify(demos):
                return [prog]

    def _build_fill_all(inp, facts, grid_info, mode):
        bg = facts.bg
        result = inp.copy()
        keys = []
        contents = []
        for r in range(grid_info.n_rows):
            for c in range(grid_info.n_cols):
                cell = grid_info.cell_at(r, c)
                if cell and cell_has_content(inp, cell, bg):
                    content = cell_content(inp, cell, bg)
                    if mode == 'pattern':
                        key = (content.shape, content.tobytes())
                    else:
                        color = cell_content_color(inp, cell, bg)
                        if color is None:
                            continue
                        key = ('color', int(color))
                    keys.append(key)
                    contents.append(content)

        if not keys:
            return result

        from collections import Counter
        key, _ = Counter(keys).most_common(1)[0]
        template = None
        if mode == 'pattern':
            for content in contents:
                if (content.shape, content.tobytes()) == key:
                    template = content
                    break
        else:
            template = key[1]

        for r in range(grid_info.n_rows):
            for c in range(grid_info.n_cols):
                cell = grid_info.cell_at(r, c)
                if cell and not cell_has_content(inp, cell, bg):
                    if mode == 'pattern' and template is not None:
                        h, w = cell.height, cell.width
                        sh, sw = template.shape
                        if sh > h or sw > w:
                            continue
                        result[cell.r0:cell.r0 + sh,
                               cell.c0:cell.c0 + sw] = template
                    elif mode == 'color':
                        result[cell.r0:cell.r0 + cell.height,
                               cell.c0:cell.c0 + cell.width] = int(template)

        return result

    for mode in ('color', 'pattern'):
        all_ok = True
        for di, (inp, out) in enumerate(demos):
            facts = all_facts[di]
            grid_info = detect_grid(facts)
            if grid_info is None:
                all_ok = False
                break
            result = _build_fill_all(inp, facts, grid_info, mode)
            if not np.array_equal(result, out):
                all_ok = False
                break

        if all_ok:
            prog = SearchProgram(
                steps=[SearchStep('grid_fill_between', {'mode': mode, 'fill_all': True})],
                provenance=f'derive:grid_fill_all_{mode}',
            )
            if prog.verify(demos):
                return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 2i: Grid-cell pack
# ---------------------------------------------------------------------------

def _derive_grid_cell_pack(all_facts, demos):
    """Pack non-empty grid cells into a fixed order (row/col/color)."""
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
    )

    orderings = ('row', 'col', 'color')

    for ordering in orderings:
        all_ok = True
        for di, (inp, out) in enumerate(demos):
            facts = all_facts[di]
            grid_info = detect_grid(facts)
            if grid_info is None:
                all_ok = False
                break

            items = []
            for r in range(grid_info.n_rows):
                for c in range(grid_info.n_cols):
                    cell = grid_info.cell_at(r, c)
                    if cell and cell_has_content(inp, cell, facts.bg):
                        items.append({
                            'row': r,
                            'col': c,
                            'content': cell_content(inp, cell, facts.bg),
                        })

            if ordering == 'col':
                items.sort(key=lambda x: (x['col'], x['row']))
            elif ordering == 'color':
                def _key(it):
                    content = it['content']
                    flat = content.ravel()
                    non_bg = flat[flat != facts.bg]
                    if len(non_bg) == 0:
                        return (999, it['row'], it['col'])
                    from collections import Counter
                    color = Counter(non_bg.tolist()).most_common(1)[0][0]
                    return (int(color), it['row'], it['col'])
                items.sort(key=_key)
            else:
                items.sort(key=lambda x: (x['row'], x['col']))

            result = np.full_like(inp, facts.bg)
            if facts.separators:
                for sep in facts.separators:
                    if sep.axis == 'row':
                        result[sep.index, :] = sep.color
                    else:
                        result[:, sep.index] = sep.color

            idx = 0
            for r in range(grid_info.n_rows):
                for c in range(grid_info.n_cols):
                    if idx >= len(items):
                        break
                    cell = grid_info.cell_at(r, c)
                    if cell is None:
                        continue
                    content = items[idx]['content']
                    h, w = content.shape
                    if h > cell.height or w > cell.width:
                        all_ok = False
                        break
                    result[cell.r0:cell.r0 + h,
                           cell.c0:cell.c0 + w] = content
                    idx += 1
                if not all_ok:
                    break

            if not all_ok or not np.array_equal(result, out):
                all_ok = False
                break

        if all_ok:
            prog = SearchProgram(
                steps=[SearchStep('grid_cell_pack', {'ordering': ordering})],
                provenance=f'derive:grid_cell_pack_{ordering}',
            )
            if prog.verify(demos):
                return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 2j: Grid-slot transfer
# ---------------------------------------------------------------------------

def _grid_cell_match_cost(src_mask, tgt_mask, src_cell, tgt_cell, bg):
    """Tiered cost for matching a source cell to a target cell.

    Tier 0 (cost 0): exact content match (same shape + pixels).
    Tier 1 (cost 1-10): same shape, near-identical (mask overlap >= 0.5).
    Tier 2 (cost 20+): same cell dims, spatial distance tiebreaker.
    Tier INF (cost 1e6): incompatible sizes.
    """
    sh, sw = src_mask.shape
    th, tw = tgt_mask.shape
    if sh != tgt_cell.height or sw != tgt_cell.width:
        # src content doesn't fit in target cell
        if sh > tgt_cell.height or sw > tgt_cell.width:
            return 1e6
    # Exact content match
    if src_mask.shape == tgt_mask.shape and np.array_equal(src_mask, tgt_mask):
        return 0.0
    # Same-shape near match (mask overlap)
    if src_mask.shape == tgt_mask.shape:
        src_nz = src_mask != bg
        tgt_nz = tgt_mask != bg
        overlap = int(np.sum(src_nz & tgt_nz))
        total = max(int(np.sum(src_nz)), int(np.sum(tgt_nz)), 1)
        ratio = overlap / total
        if ratio >= 0.5:
            return 1.0 + (1.0 - ratio) * 9.0
    # Same cell dims — spatial distance tiebreaker
    if sh <= tgt_cell.height and sw <= tgt_cell.width:
        dist = abs(src_cell.r0 - tgt_cell.r0) + abs(src_cell.c0 - tgt_cell.c0)
        return 20.0 + dist
    return 1e6


def _derive_grid_slot_transfer(all_facts, demos):
    """Move source cell contents into empty target cells via feature matching.

    Feature matching tiers: exact content > near-shape > spatial.
    Uses Hungarian assignment when multiple sources/targets exist.
    """
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
    )
    from scipy.optimize import linear_sum_assignment

    for di, (inp, out) in enumerate(demos):
        facts = all_facts[di]
        g = detect_grid(facts)
        if g is None:
            return []
        bg = facts.bg

        sources = []
        targets = []
        for cell in g.cells:
            in_has = cell_has_content(inp, cell, bg)
            out_has = cell_has_content(out, cell, bg)
            if in_has and not out_has:
                sources.append(cell)
            elif not in_has and out_has:
                targets.append(cell)

        if not sources or not targets or len(sources) != len(targets):
            return []

        # Feature-based matching
        src_masks = [cell_content(inp, c, bg) for c in sources]
        tgt_masks = [cell_content(out, c, bg) for c in targets]
        n = len(sources)
        cost = np.full((n, n), 1e6, dtype=float)
        for si in range(n):
            for ti in range(n):
                cost[si, ti] = _grid_cell_match_cost(
                    src_masks[si], tgt_masks[ti],
                    sources[si], targets[ti], bg,
                )
        rows, cols = linear_sum_assignment(cost)
        if any(cost[r, c] >= 1e6 for r, c in zip(rows, cols)):
            return []

    prog = SearchProgram(
        steps=[SearchStep('grid_slot_transfer', {})],
        provenance='derive:grid_slot_transfer',
    )
    if prog.verify(demos):
        return [prog]
    return []


# ---------------------------------------------------------------------------
# Strategy 2k: Grid-conditional transfer
# ---------------------------------------------------------------------------

def _derive_grid_conditional_transfer(all_facts, demos):
    """Fill empty grid cells using a row/col/mirror rule verified across demos.

    Tries four rules: nearest_row, nearest_col, mirror_h, mirror_v.
    Accepts the first rule that reproduces the output in all demos.
    """
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
    )

    grids = []
    for di, (inp, out) in enumerate(demos):
        g = detect_grid(all_facts[di])
        if g is None:
            return []
        grids.append(g)

    for rule in ('nearest_row', 'nearest_col', 'mirror_h', 'mirror_v'):
        all_ok = True
        for di, (inp, out) in enumerate(demos):
            bg = all_facts[di].bg
            g = grids[di]

            # Build content map from input
            content_map = {}
            for cell in g.cells:
                if cell_has_content(inp, cell, bg):
                    content_map[(cell.grid_row, cell.grid_col)] = cell_content(inp, cell, bg)

            # Check: every empty input cell that has content in output
            # must match the rule prediction
            for cell in g.cells:
                if cell_has_content(inp, cell, bg):
                    continue
                if not cell_has_content(out, cell, bg):
                    continue
                gr, gc = cell.grid_row, cell.grid_col
                expected = cell_content(out, cell, bg)
                predicted = None

                if rule == 'nearest_row':
                    best_dist = float('inf')
                    for (r, c), cnt in content_map.items():
                        if r == gr and abs(c - gc) < best_dist:
                            best_dist = abs(c - gc)
                            predicted = cnt
                elif rule == 'nearest_col':
                    best_dist = float('inf')
                    for (r, c), cnt in content_map.items():
                        if c == gc and abs(r - gr) < best_dist:
                            best_dist = abs(r - gr)
                            predicted = cnt
                elif rule == 'mirror_h':
                    predicted = content_map.get((gr, g.n_cols - 1 - gc))
                elif rule == 'mirror_v':
                    predicted = content_map.get((g.n_rows - 1 - gr, gc))

                if predicted is None or predicted.shape != expected.shape:
                    all_ok = False
                    break
                if not np.array_equal(predicted, expected):
                    all_ok = False
                    break
            if not all_ok:
                break

        if all_ok:
            prog = SearchProgram(
                steps=[SearchStep('grid_conditional_transfer', {'rule': rule})],
                provenance='derive:grid_conditional_transfer',
            )
            if prog.verify(demos):
                return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 2l: Object grid pack
# ---------------------------------------------------------------------------

def _derive_object_grid_pack_prescan(demos):
    """Prescan variant that works before all_facts is computed."""
    from aria.guided.perceive import perceive
    all_facts = [perceive(inp) for inp, out in demos]
    return _derive_object_grid_pack(all_facts, demos)


def _derive_object_grid_pack(all_facts, demos):
    """Pack input objects into output grid by size/color/position order.

    Detects when the output is a regular arrangement of input object patches
    in a new grid layout. Tries row_major, size_asc, size_desc, color_asc orderings.
    """
    from aria.guided.perceive import perceive

    for order in ('row_major', 'size_asc', 'size_desc', 'color_asc'):
        all_ok = True
        shared_params = None

        for di, (inp, out) in enumerate(demos):
            facts = all_facts[di]
            bg = facts.bg
            objs = [o for o in facts.objects if o.size > 0]
            if not objs:
                all_ok = False
                break

            # Sort
            if order == 'size_asc':
                objs.sort(key=lambda o: (o.size, o.color))
            elif order == 'size_desc':
                objs.sort(key=lambda o: (-o.size, o.color))
            elif order == 'color_asc':
                objs.sort(key=lambda o: (o.color, -o.size))
            else:
                objs.sort(key=lambda o: (o.row, o.col))

            patches = [inp[o.row:o.row+o.height, o.col:o.col+o.width].copy() for o in objs]
            cell_h = max(o.height for o in objs)
            cell_w = max(o.width for o in objs)

            # Infer grid dims from output shape
            out_h, out_w = out.shape
            # Try with sep=0 and sep=1
            found = False
            for sep in (0, 1):
                for out_cols in range(1, len(objs) + 1):
                    out_rows = (len(objs) + out_cols - 1) // out_cols
                    th = out_rows * cell_h + max(0, out_rows - 1) * sep
                    tw = out_cols * cell_w + max(0, out_cols - 1) * sep
                    if th != out_h or tw != out_w:
                        continue

                    # Build candidate output
                    candidate = np.full((th, tw), bg, dtype=inp.dtype)
                    for idx, patch in enumerate(patches):
                        gr = idx // out_cols
                        gc = idx % out_cols
                        if gr >= out_rows:
                            break
                        r0 = gr * (cell_h + sep)
                        c0 = gc * (cell_w + sep)
                        ph, pw = patch.shape
                        h = min(ph, cell_h)
                        w = min(pw, cell_w)
                        candidate[r0:r0+h, c0:c0+w] = patch[:h, :w]

                    if np.array_equal(candidate, out):
                        params = {
                            'order': order, 'cell_h': cell_h, 'cell_w': cell_w,
                            'out_rows': out_rows, 'out_cols': out_cols, 'sep': sep,
                        }
                        if shared_params is None:
                            shared_params = params
                        elif (params['order'] != shared_params['order'] or
                              params['sep'] != shared_params['sep']):
                            all_ok = False
                        found = True
                        break
                if found:
                    break
            if not found:
                all_ok = False
                break

        if all_ok and shared_params is not None:
            prog = SearchProgram(
                steps=[SearchStep('object_grid_pack', shared_params)],
                provenance='derive:object_grid_pack',
            )
            if prog.verify(demos):
                return [prog]

    return []


# ---------------------------------------------------------------------------
# Strategy 2m: Panel legend map
# ---------------------------------------------------------------------------

def _derive_panel_legend_map(all_facts, demos):
    """Detect a legend panel and derive a color mapping to apply to the target panel.

    Looks for a single-row or single-col separator splitting the grid into
    a small legend region and a larger target region. Derives color mapping
    from legend entries that generalizes across all demos.
    """
    from aria.guided.perceive import perceive

    for di, (inp, out) in enumerate(demos):
        facts = all_facts[di]
        if not facts.separators:
            return []

    # Try each separator as a panel boundary
    first_facts = all_facts[0]
    for sep in first_facts.separators:
        axis = sep.axis
        idx = sep.index
        rows, cols = demos[0][0].shape

        if axis == 'row':
            # Split top/bottom
            for legend_side in ('top', 'bottom'):
                mapping = _try_legend_mapping(
                    all_facts, demos, axis, idx, legend_side)
                if mapping is not None:
                    params = {
                        'axis': axis, 'sep_idx': idx,
                        'legend_side': legend_side, 'mapping': mapping,
                    }
                    prog = SearchProgram(
                        steps=[SearchStep('panel_legend_map', params)],
                        provenance='derive:panel_legend_map',
                    )
                    if prog.verify(demos):
                        return [prog]
        elif axis == 'col':
            for legend_side in ('left', 'right'):
                mapping = _try_legend_mapping(
                    all_facts, demos, axis, idx, legend_side)
                if mapping is not None:
                    params = {
                        'axis': axis, 'sep_idx': idx,
                        'legend_side': legend_side, 'mapping': mapping,
                    }
                    prog = SearchProgram(
                        steps=[SearchStep('panel_legend_map', params)],
                        provenance='derive:panel_legend_map',
                    )
                    if prog.verify(demos):
                        return [prog]

    return []


def _try_legend_mapping(all_facts, demos, axis, sep_idx, legend_side):
    """Try to derive a consistent color mapping from legend→target across demos."""
    mappings = []
    for di, (inp, out) in enumerate(demos):
        bg = all_facts[di].bg
        if axis == 'col':
            if legend_side == 'left':
                legend = inp[:, :sep_idx]
                target_in = inp[:, sep_idx + 1:]
                target_out = out[:, sep_idx + 1:]
            else:
                legend = inp[:, sep_idx + 1:]
                target_in = inp[:, :sep_idx]
                target_out = out[:, :sep_idx]
        else:
            if legend_side == 'top':
                legend = inp[:sep_idx, :]
                target_in = inp[sep_idx + 1:, :]
                target_out = out[sep_idx + 1:, :]
            else:
                legend = inp[sep_idx + 1:, :]
                target_in = inp[:sep_idx, :]
                target_out = out[:sep_idx, :]

        if legend.size == 0 or target_in.size == 0:
            return None
        if target_in.shape != target_out.shape:
            return None

        # Legend must be smaller than target
        if legend.size >= target_in.size:
            return None

        # Derive per-pixel color mapping from target_in → target_out
        dm = {}
        diff_mask = target_in != target_out
        if not np.any(diff_mask):
            continue  # No change — mapping is identity
        for r in range(target_in.shape[0]):
            for c in range(target_in.shape[1]):
                if target_in[r, c] != target_out[r, c]:
                    src_c = int(target_in[r, c])
                    tgt_c = int(target_out[r, c])
                    if src_c in dm and dm[src_c] != tgt_c:
                        return None  # Inconsistent mapping
                    dm[src_c] = tgt_c

        # Verify mapping colors appear in legend
        legend_colors = set(int(x) for x in legend.flat if x != bg)
        for tgt_c in dm.values():
            if tgt_c != bg and tgt_c not in legend_colors:
                return None  # Target color not in legend

        mappings.append(dm)

    if not mappings:
        return None

    # All demos must agree on mapping
    base = mappings[0]
    for m in mappings[1:]:
        for k, v in m.items():
            if k in base and base[k] != v:
                return None
            base[k] = v

    return {str(k): v for k, v in base.items()} if base else None


# ---------------------------------------------------------------------------
# Shared: cross-demo selector from OID sets
# ---------------------------------------------------------------------------

def _find_selector_for_oid_sets(target_oids_per_demo, all_facts):
    """Find a selector that selects the given OID sets in each demo.

    Tries simple selectors first, falls back to cross-demo rule induction.
    """
    # Fast path: try simple selector from demo 0
    sel = _find_selector(target_oids_per_demo[0], all_facts[0])
    if sel is not None:
        ok = True
        for di in range(1, len(all_facts)):
            expected = target_oids_per_demo[di]
            selected = _resolve_selector(sel, all_facts[di])
            if selected is None or selected != expected:
                ok = False
                break
        if ok:
            return sel

    # Slow path: cross-demo rule induction
    from aria.search.selection_facts import extract_object_facts, STRUCTURAL_FEATURES
    from aria.search.rules import induce_boolean_dnf

    all_rows = []
    all_labels = []
    for di in range(len(all_facts)):
        target_oids = target_oids_per_demo[di]
        fact_rows = extract_object_facts(all_facts[di])
        for obj, row in zip(all_facts[di].objects, fact_rows):
            all_rows.append(row)
            all_labels.append(obj.oid in target_oids)

    if not any(all_labels):
        return None
    if all(all_labels):
        return StepSelect('all')

    # Structural features (fast)
    candidate_fields = list(STRUCTURAL_FEATURES)
    for row in all_rows:
        for f in candidate_fields:
            if f not in row:
                row[f] = False

    rule = induce_boolean_dnf(
        all_rows, all_labels,
        candidate_fields=candidate_fields,
        max_clause_size=3,
        max_clauses=1,
    )
    if rule is not None:
        return StepSelect('by_rule', {'rule': rule.to_dict()})

    # Extended with color features
    all_features: set[str] = set()
    for row in all_rows:
        all_features.update(row.keys())
    color_fields = sorted(f for f in all_features if f.startswith('color_is_'))
    extended_fields = candidate_fields + color_fields
    for row in all_rows:
        for f in extended_fields:
            if f not in row:
                row[f] = False

    rule = induce_boolean_dnf(
        all_rows, all_labels,
        candidate_fields=extended_fields,
        max_clause_size=2,
        max_clauses=1,
    )
    if rule is not None:
        return StepSelect('by_rule', {'rule': rule.to_dict()})

    return None


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

    if sel.role == 'by_rule':
        from aria.search.selection_facts import select_by_rule
        return set(o.oid for o in select_by_rule(sel.params.get('rule', {}), facts))

    if sel.role == 'by_predicate':
        preds = sel.params.get('predicates', [])
        if preds:
            return set(o.oid for o in prim_select(facts, preds))

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
    """Find a StepSelect that selects all objects of a given match_type.

    Tries simple predicates on demo 0 first (fast path). If found,
    verifies it works across all demos. Falls back to cross-demo
    rule induction over rich boolean object facts.
    """
    oids = set(t.in_obj.oid for t in all_transitions[0]
               if t.match_type == match_type and t.in_obj)
    if not oids:
        return None

    # Fast path: simple selector from demo 0
    sel = _find_selector(oids, all_facts[0])
    if sel is not None:
        # Cross-demo verification
        ok = True
        for di in range(1, len(all_transitions)):
            expected = {t.in_obj.oid for t in all_transitions[di]
                        if t.match_type == match_type and t.in_obj}
            selected = _resolve_selector(sel, all_facts[di])
            if selected is None or selected != expected:
                ok = False
                break
        if ok:
            return sel

    # Slow path: cross-demo rule induction
    return _find_selector_cross_demo(match_type, all_transitions, all_facts)


def _find_selector_cross_demo(match_type, all_transitions, all_facts):
    """Induce a selection rule over rich boolean facts across all demos.

    Pools object facts from all demos, labels targets by match_type,
    and finds a bounded DNF rule that exactly separates them.
    """
    from aria.search.selection_facts import extract_object_facts
    from aria.search.rules import induce_boolean_dnf

    all_rows = []
    all_labels = []

    for di in range(len(all_transitions)):
        target_oids = {t.in_obj.oid for t in all_transitions[di]
                       if t.match_type == match_type and t.in_obj}
        fact_rows = extract_object_facts(all_facts[di])
        for obj, row in zip(all_facts[di].objects, fact_rows):
            all_rows.append(row)
            all_labels.append(obj.oid in target_oids)

    if not any(all_labels):
        return None
    if all(all_labels):
        return StepSelect('all')

    # Structural features only first (fast); add per-color only if needed
    from aria.search.selection_facts import STRUCTURAL_FEATURES
    candidate_fields = list(STRUCTURAL_FEATURES)

    # Fill missing with False
    for row in all_rows:
        for f in candidate_fields:
            if f not in row:
                row[f] = False

    # Try structural-only conjunction (fast: ~30 features)
    rule = induce_boolean_dnf(
        all_rows, all_labels,
        candidate_fields=candidate_fields,
        max_clause_size=3,
        max_clauses=1,
    )
    if rule is not None:
        return StepSelect('by_rule', {'rule': rule.to_dict()})

    # Add per-color features and try again with size-2 conjunctions
    all_features: set[str] = set()
    for row in all_rows:
        all_features.update(row.keys())
    color_fields = sorted(f for f in all_features if f.startswith('color_is_'))
    extended_fields = candidate_fields + color_fields
    for row in all_rows:
        for f in extended_fields:
            if f not in row:
                row[f] = False

    rule = induce_boolean_dnf(
        all_rows, all_labels,
        candidate_fields=extended_fields,
        max_clause_size=2,
        max_clauses=1,
    )
    if rule is None:
        return None

    return StepSelect('by_rule', {'rule': rule.to_dict()})


def _find_selector(oids, facts):
    """Find a StepSelect that selects exactly the given object IDs.

    Tries simple single-predicate selectors first (fast). Falls back
    to bounded rule induction over rich boolean object facts.
    """
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

    # Rule induction: bounded conjunction search over rich boolean facts
    sel = _induce_selector_rule(target_set, facts)
    if sel is not None:
        return sel

    return None


def _induce_selector_rule(target_set, facts):
    """Induce a DNF selection rule from a single demo's facts."""
    from aria.search.selection_facts import extract_object_facts, STRUCTURAL_FEATURES
    from aria.search.rules import induce_boolean_dnf

    fact_rows = extract_object_facts(facts)
    labels = [obj.oid in target_set for obj in facts.objects]

    if not any(labels) or all(labels):
        return None

    # Structural features only (bounded, fast)
    candidate_fields = list(STRUCTURAL_FEATURES)
    for row in fact_rows:
        for f in candidate_fields:
            if f not in row:
                row[f] = False

    rule = induce_boolean_dnf(
        fact_rows, labels,
        candidate_fields=candidate_fields,
        max_clause_size=3,
        max_clauses=1,
    )
    if rule is not None:
        return StepSelect('by_rule', {'rule': rule.to_dict()})

    # Add per-color features, try size-2 conjunctions
    all_features: set[str] = set()
    for row in fact_rows:
        all_features.update(row.keys())
    color_fields = sorted(f for f in all_features if f.startswith('color_is_'))
    extended_fields = candidate_fields + color_fields
    for row in fact_rows:
        for f in extended_fields:
            if f not in row:
                row[f] = False

    rule = induce_boolean_dnf(
        fact_rows, labels,
        candidate_fields=extended_fields,
        max_clause_size=2,
        max_clauses=1,
    )
    if rule is None:
        return None

    return StepSelect('by_rule', {'rule': rule.to_dict()})


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

    # Attribute each added pixel to its nearest marker
    all_markers = [m for mlist in color_markers.values() for m in mlist]
    added_positions = list(zip(*np.where(added)))
    pixel_owner: dict[tuple[int, int], int] = {}  # (r,c) → marker index
    for r, c in added_positions:
        best_dist = float('inf')
        best_idx = -1
        for mi, m in enumerate(all_markers):
            d = abs(r - round(m.center_row)) + abs(c - round(m.center_col))
            if d < best_dist:
                best_dist = d
                best_idx = mi
        pixel_owner[(r, c)] = best_idx

    # For each marker color, learn template from owned added pixels
    templates = {}  # color → {(dr, dc): pixel_color}
    for color, mlist in color_markers.items():
        per_marker_templates = []
        for m in mlist:
            cr = round(m.center_row)
            cc = round(m.center_col)
            mi = all_markers.index(m)
            template = {}
            for (r, c), owner in pixel_owner.items():
                if owner == mi:
                    template[(r - cr, c - cc)] = int(out0[r, c])
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
