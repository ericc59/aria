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

    progs = _derive_legend_frame_fill(demos)
    if progs:
        return progs

    # Remaining strategies require same-shape
    if any(inp.shape != out.shape for inp, out in demos):
        return []

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

    # Demo 0 has specific color pairs. Verify using color-agnostic execution on ALL demos.
    from aria.search.executor import _exec_cavity_transfer_auto

    for i, (inp, out) in enumerate(demos):
        result = _exec_cavity_transfer_auto(inp)
        if not np.array_equal(result, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('cavity_transfer', {'mode': 'auto'})],
        provenance='derive:cavity_transfer_auto',
    )
    return [prog]


def _derive_legend_frame_fill(demos):
    """Fill bg cells enclosed by colored boundaries.

    For each non-bg color, flood-fill bg from the grid border. Any bg cells
    NOT reached are enclosed by that color's boundary. Fill them with a
    derived color (from a legend, or the boundary color itself).
    """
    if any(inp.shape != out.shape for inp, out in demos):
        return []

    inp0, out0 = demos[0]

    changed = (inp0 != out0)
    if changed.sum() == 0:
        return []

    # Verify using the actual executor semantics only. Older derive-side
    # reconstruction peeked into demo outputs and could reject tasks that the
    # emitted enclosed-fill executor would solve.
    from aria.search.executor import _exec_legend_frame_fill
    params = {'strategy': 'enclosed_fill'}
    for inp, out in demos:
        cand = _exec_legend_frame_fill(inp, params)
        if not np.array_equal(cand, out):
            return []

    prog = SearchProgram(
        steps=[SearchStep('legend_frame_fill', params)],
        provenance='derive:legend_enclosed_fill',
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
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_find_frame, prim_crop_bbox

    results = []

    # Analyze demo 0
    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    bg = facts0.bg

    # Find all framed objects
    frame_infos = []
    for obj in facts0.objects:
        frame = prim_find_frame(obj, inp0)
        if frame:
            r0, c0, r1, c1 = frame
            bbox = prim_crop_bbox(inp0, obj)
            fh, fw = bbox.shape
            # Check if interior is bg or colored
            if fh > 2 and fw > 2:
                interior = bbox[1:-1, 1:-1]
                interior_is_bg = np.all(interior == bg)
            else:
                interior_is_bg = True
            frame_infos.append({
                'obj': obj, 'bbox': bbox, 'frame': frame,
                'color': obj.color, 'row': obj.row, 'col': obj.col,
                'interior_bg': interior_is_bg,
            })

    if len(frame_infos) < 2:
        return []

    # All bboxes must be same shape
    bbox_shapes = set(f['bbox'].shape for f in frame_infos)
    if len(bbox_shapes) != 1:
        return []
    bh, bw = list(bbox_shapes)[0]

    oh, ow = out0.shape
    n_frames = len(frame_infos)

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
            candidate = np.full((oh, ow), bg, dtype=out0.dtype)
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
                    facts = perceive(inp)
                    finfos = []
                    for obj in facts.objects:
                        fr = prim_find_frame(obj, inp)
                        if fr:
                            bb = prim_crop_bbox(inp, obj)
                            if bb.shape != (bh, bw):
                                continue
                            if bb.shape[0] > 2 and bb.shape[1] > 2:
                                interior_bg = np.all(bb[1:-1, 1:-1] == facts.bg)
                            else:
                                interior_bg = True
                            finfos.append({
                                'obj': obj, 'bbox': bb, 'frame': fr,
                                'color': obj.color, 'row': obj.row, 'col': obj.col,
                                'interior_bg': interior_bg,
                            })

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
                    cand = np.full((oh_i, ow_i), facts.bg, dtype=inp.dtype)
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
