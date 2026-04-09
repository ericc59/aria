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

    # Shape-independent strategies first
    progs = _derive_object_repack(demos)
    if progs:
        return progs

    progs = _derive_symmetry_repair(demos)
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
