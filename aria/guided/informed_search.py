"""Output-informed search: uses train output structure to constrain and guide.

During training, we HAVE the output. Use it to:
1. Compute the residual (what changed)
2. Extract what colors appear in the residual
3. Infer likely bindings from output structure
4. Score partial programs by diff reduction (reward shaping)

This is the bridge between blind search and ARC solving.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np
from scipy import ndimage

from aria.guided.grammar import (
    Program, Action, Act, Target, Rewrite, execute_program,
)
from aria.guided.search import SearchResult, _verify, predict_test
from aria.guided.workspace import build_workspace, _detect_bg, ObjectInfo
from aria.types import Grid


def informed_search(
    train: list[tuple[Grid, Grid]],
    max_candidates: int = 2000,
    max_steps: int = 3,
) -> SearchResult:
    """Search using output-informed candidate generation.

    Instead of enumerating ALL possible extensions, analyzes the
    input→output diff to generate ONLY plausible extensions.
    """
    if not train:
        return SearchResult(False, None, 0, 0)

    bgs = [_detect_bg(inp) for inp, _ in train]
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)
    candidates_tried = 0

    # Analyze the residual across all demos
    analysis = _analyze_residuals(train, bgs)

    # Generate informed extensions based on the analysis
    queue: deque[Program] = deque()
    queue.append(Program())

    while queue and candidates_tried < max_candidates:
        partial = queue.popleft()
        n_steps = sum(1 for a in partial.actions if a.act in (Act.NEXT, Act.STOP))
        if n_steps >= max_steps:
            continue

        # Current canvas state (apply partial program to get intermediate)
        canvas_diffs = []
        for (inp, out), bg in zip(train, bgs):
            try:
                current = execute_program(partial, inp, bg)
            except Exception:
                current = inp.copy()
            remaining = int(np.sum(current != out))
            canvas_diffs.append(remaining)

        remaining_diff = sum(canvas_diffs)
        if remaining_diff == 0:
            # Already solved by partial program
            ok, diff = _verify(partial, train, bgs)
            if ok:
                return SearchResult(True, partial, candidates_tried, 0)

        # Generate extensions informed by what STILL needs to change
        for ext in _informed_extensions(partial, train, bgs, analysis):
            candidates_tried += 1
            candidate = partial.copy()
            for action in ext:
                candidate.append(action)

            if ext[-1].act == Act.STOP:
                ok, diff = _verify(candidate, train, bgs)
                if ok:
                    return SearchResult(True, candidate, candidates_tried, 0)
                if diff < best_diff:
                    best_diff = diff
            else:
                queue.append(candidate)

            if candidates_tried >= max_candidates:
                break

    return SearchResult(False, None, candidates_tried, best_diff)


# ---------------------------------------------------------------------------
# Residual analysis: what needs to change?
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class ResidualAnalysis:
    """Cross-demo analysis of what changes."""
    # Change types present across demos
    has_additions: bool       # bg → non-bg
    has_deletions: bool       # non-bg → bg
    has_recolors: bool        # non-bg → different non-bg

    # Colors
    residual_output_colors: set[int]  # colors that appear in output but not input at changed positions
    deleted_colors: set[int]          # input colors at deleted positions
    recolored_from: set[int]          # input colors at recolored positions
    recolored_to: set[int]            # output colors at recolored positions

    # Structural
    all_additions_enclosed: bool      # are all added pixels enclosed by non-bg?
    additions_match_frame: bool       # do added pixel colors match their enclosing frame?
    has_periodicity: bool             # does the grid have periodic patterns?

    # Per-demo color mappings
    per_demo_color_maps: list[dict[int, int] | None]  # consistent per-pixel color map per demo


from dataclasses import dataclass


def _analyze_residuals(train, bgs) -> ResidualAnalysis:
    has_add = has_del = has_rec = False
    res_out_colors: set[int] = set()
    del_colors: set[int] = set()
    rec_from: set[int] = set()
    rec_to: set[int] = set()
    all_add_enclosed = True
    add_match_frame = True
    has_period = False
    cmaps = []

    for (inp, out), bg in zip(train, bgs):
        diff = inp != out
        if not np.any(diff):
            cmaps.append(None)
            continue

        inp_at_diff = inp[diff]
        out_at_diff = out[diff]

        additions = diff & (inp == bg) & (out != bg)
        deletions = diff & (inp != bg) & (out == bg)
        recolors = diff & (inp != bg) & (out != bg)

        if np.any(additions):
            has_add = True
            res_out_colors.update(int(v) for v in np.unique(out[additions]))
            # Check if all additions are enclosed
            enclosed = _get_enclosed_bg(inp, bg)
            if not np.all(additions <= enclosed):
                all_add_enclosed = False
            # Check if added colors match enclosing frame
            if np.any(additions):
                rows, cols = inp.shape
                labeled, n = ndimage.label(enclosed & additions, structure=np.ones((3, 3)))
                struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
                for lid in range(1, n + 1):
                    comp = labeled == lid
                    dilated = ndimage.binary_dilation(comp, structure=struct4)
                    border = dilated & ~comp
                    border_vals = inp[border]
                    non_bg_border = border_vals[border_vals != bg]
                    if len(non_bg_border) > 0:
                        frame_c = int(Counter(non_bg_border.tolist()).most_common(1)[0][0])
                        add_c = set(int(v) for v in np.unique(out[comp]))
                        if add_c != {frame_c}:
                            add_match_frame = False

        if np.any(deletions):
            has_del = True
            del_colors.update(int(v) for v in np.unique(inp[deletions]))

        if np.any(recolors):
            has_rec = True
            rec_from.update(int(v) for v in np.unique(inp[recolors]))
            rec_to.update(int(v) for v in np.unique(out[recolors]))

        # Color map
        cmap: dict[int, int] = {}
        consistent = True
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                a, b = int(inp[r, c]), int(out[r, c])
                if a != b:
                    if a in cmap and cmap[a] != b:
                        consistent = False
                        break
                    cmap[a] = b
            if not consistent:
                break
        cmaps.append(cmap if consistent else None)

        # Periodicity check
        for r in range(out.shape[0]):
            for p in range(1, out.shape[1] // 2 + 1):
                tile = out[r, :p]
                if all(np.array_equal(out[r, i:i + p], tile[:min(p, out.shape[1] - i)])
                       for i in range(p, out.shape[1], p)):
                    if not np.array_equal(inp[r, :], out[r, :]):
                        has_period = True
                        break
            if has_period:
                break

    return ResidualAnalysis(
        has_additions=has_add,
        has_deletions=has_del,
        has_recolors=has_rec,
        residual_output_colors=res_out_colors,
        deleted_colors=del_colors,
        recolored_from=rec_from,
        recolored_to=rec_to,
        all_additions_enclosed=all_add_enclosed,
        additions_match_frame=add_match_frame,
        has_periodicity=has_period,
        per_demo_color_maps=cmaps,
    )


def _get_enclosed_bg(grid, bg):
    rows, cols = grid.shape
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return (grid == bg) & ~reachable


# ---------------------------------------------------------------------------
# Informed extension generation
# ---------------------------------------------------------------------------

def _informed_extensions(partial, train, bgs, analysis):
    """Generate extensions informed by residual analysis."""
    extensions = []
    a = analysis

    # Strategy: fill enclosed bg
    if a.has_additions and a.all_additions_enclosed:
        if a.additions_match_frame:
            # Fill with frame color (cross-demo varying)
            ext = [
                Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
                Action(Act.REWRITE, Rewrite.FILL),
                Action(Act.BIND, param_name="color_source", param_value="from_context:frame"),
                Action(Act.STOP),
            ]
            extensions.append(ext)
        # Also try each residual color as literal fill
        for c in a.residual_output_colors:
            ext = [
                Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
                Action(Act.REWRITE, Rewrite.FILL),
                Action(Act.BIND, param_name="color", param_value=c),
                Action(Act.BIND, param_name="color_source", param_value="literal"),
                Action(Act.STOP),
            ]
            extensions.append(ext)

    # Strategy: recolor by color
    if a.has_recolors:
        for fc in a.recolored_from:
            for tc in a.recolored_to:
                if fc != tc:
                    ext = [
                        Action(Act.SELECT_TARGET, Target.BY_COLOR),
                        Action(Act.REWRITE, Rewrite.RECOLOR),
                        Action(Act.BIND, param_name="from_color", param_value=fc),
                        Action(Act.BIND, param_name="color", param_value=tc),
                        Action(Act.STOP),
                    ]
                    extensions.append(ext)

    # Strategy: recolor adjacent to singleton
    if a.has_recolors:
        ext = [
            Action(Act.SELECT_TARGET, Target.ADJACENT_TO),
            Action(Act.REWRITE, Rewrite.RECOLOR),
            Action(Act.BIND, param_name="color_source", param_value="from_context:adjacent_singleton"),
            Action(Act.STOP),
        ]
        extensions.append(ext)

    # Strategy: delete by color
    if a.has_deletions:
        for c in a.deleted_colors:
            ext = [
                Action(Act.SELECT_TARGET, Target.BY_COLOR),
                Action(Act.REWRITE, Rewrite.DELETE),
                Action(Act.BIND, param_name="from_color", param_value=c),
                Action(Act.STOP),
            ]
            extensions.append(ext)

    # Strategy: symmetry repair
    for axis in ["h", "v"]:
        ext = [
            Action(Act.SELECT_TARGET, Target.ASYMMETRIC),
            Action(Act.REWRITE, Rewrite.SYMMETRIZE),
            Action(Act.BIND, param_name="axis", param_value=axis),
            Action(Act.STOP),
        ]
        extensions.append(ext)

    # Strategy: periodic repair
    if a.has_periodicity:
        for axis in ["row", "col"]:
            ext = [
                Action(Act.SELECT_TARGET, Target.ANOMALY),
                Action(Act.REWRITE, Rewrite.PERIODIC_REPAIR),
                Action(Act.BIND, param_name="axis", param_value=axis),
                Action(Act.STOP),
            ]
            extensions.append(ext)

    # Strategy: fill enclosed with frame color (context-parameterized)
    if a.has_additions:
        ext = [
            Action(Act.SELECT_TARGET, Target.ENCLOSED_BG),
            Action(Act.REWRITE, Rewrite.FILL_WITH_FRAME),
            Action(Act.STOP),
        ]
        extensions.append(ext)

    # Strategy: color swap (for tasks with bidirectional recoloring)
    if a.has_recolors and len(a.recolored_from) >= 1 and len(a.recolored_to) >= 1:
        # Check if there's a swap pattern (A->B and B->A)
        for cmap in a.per_demo_color_maps:
            if cmap is None:
                continue
            for ca, cb in cmap.items():
                if cb in cmap and cmap[cb] == ca and ca != cb:
                    ext = [
                        Action(Act.REWRITE, Rewrite.SWAP_COLORS),
                        Action(Act.BIND, param_name="color_a", param_value=ca),
                        Action(Act.BIND, param_name="color_b", param_value=cb),
                        Action(Act.STOP),
                    ]
                    extensions.append(ext)
                    break

    # Strategy: per-object role-based selection (delete, recolor, move)
    role_programs = _induce_object_role_programs(train, bgs, analysis)
    extensions.extend(role_programs)

    # Multi-step: try each single-step with NEXT instead of STOP
    single_stops = [e for e in extensions if e[-1].act == Act.STOP]
    for ext in single_stops:
        multi = list(ext[:-1]) + [Action(Act.NEXT)]
        extensions.append(multi)

    return extensions


def _induce_object_role_programs(train, bgs, analysis):
    """Induce object selection predicates from cross-demo analysis.

    For each demo:
    1. Find which objects changed
    2. Extract their properties (color, singleton, size, contained_by)
    3. Find properties shared by ALL changed objects in ALL demos
       but NOT by unchanged objects
    4. Generate programs that select by those properties
    """
    from aria.guided.workspace import build_workspace

    # Collect changed/unchanged object properties per demo
    per_demo_changed: list[list[dict]] = []
    per_demo_unchanged: list[list[dict]] = []

    for (inp, out), bg in zip(train, bgs):
        ws = build_workspace(inp, out)
        diff = inp != out
        changed_props = []
        unchanged_props = []

        for obj in ws.objects:
            obj_mask = np.zeros(inp.shape, dtype=bool)
            obj_mask[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] |= obj.mask
            is_changed = np.any(obj_mask & diff)

            rels = [r for r in ws.relations if r.src == obj.oid or r.dst == obj.oid]
            is_contained = any(r.rel_type == "contains" and r.dst == obj.oid for r in rels)
            has_adjacent = any(r.rel_type == "adjacent" for r in rels)

            props = {
                "color": obj.color,
                "singleton": obj.is_singleton,
                "size": obj.size,
                "contained": is_contained,
                "has_adjacent": has_adjacent,
            }

            if is_changed:
                # What happens to it?
                out_vals = set(int(out[r, c]) for r, c in zip(*np.where(obj_mask & diff)))
                props["change_to"] = out_vals
                props["deleted"] = out_vals == {bg}
                changed_props.append(props)
            else:
                unchanged_props.append(props)

        per_demo_changed.append(changed_props)
        per_demo_unchanged.append(unchanged_props)

    if not per_demo_changed or any(len(c) == 0 for c in per_demo_changed):
        return []

    # Find predicates that separate changed from unchanged across ALL demos
    predicates = _find_separating_predicates(per_demo_changed, per_demo_unchanged)

    # Generate programs for each predicate
    programs = []
    for pred_name, pred_desc, applies_fn in predicates:
        # Determine what action to take on selected objects
        # Check: do all changed objects get deleted?
        all_deleted = all(
            all(cp.get("deleted", False) for cp in demo_changed)
            for demo_changed in per_demo_changed
        )

        if all_deleted:
            ext = [
                Action(Act.SELECT_TARGET, Target.BY_COLOR),
                Action(Act.REWRITE, Rewrite.DELETE),
                Action(Act.BIND, param_name="object_predicate", param_value=pred_desc),
                Action(Act.STOP),
            ]
            programs.append(ext)
            # Also try: move to enclosed instead of delete
            ext_move = [
                Action(Act.REWRITE, Rewrite.MOVE_TO_ENCLOSED),
                Action(Act.BIND, param_name="object_predicate", param_value=pred_desc),
                Action(Act.STOP),
            ]
            programs.append(ext_move)
        else:
            # Check: consistent recolor?
            all_recolor_same = True
            recolor_targets = set()
            for demo_changed in per_demo_changed:
                for cp in demo_changed:
                    ct = cp.get("change_to", set())
                    recolor_targets.update(ct)
                    if len(ct) != 1:
                        all_recolor_same = False

            if all_recolor_same and len(recolor_targets) == 1:
                tc = next(iter(recolor_targets))
                ext = [
                    Action(Act.SELECT_TARGET, Target.BY_COLOR),
                    Action(Act.REWRITE, Rewrite.RECOLOR),
                    Action(Act.BIND, param_name="object_predicate", param_value=pred_desc),
                    Action(Act.BIND, param_name="color", param_value=tc),
                    Action(Act.STOP),
                ]
                programs.append(ext)

            # Also try: recolor to adjacent singleton color (context-parameterized)
            ext_adj = [
                Action(Act.REWRITE, Rewrite.RECOLOR_TO_ADJ),
                Action(Act.BIND, param_name="object_predicate", param_value=pred_desc),
                Action(Act.STOP),
            ]
            programs.append(ext_adj)

    return programs


def _find_separating_predicates(per_demo_changed, per_demo_unchanged):
    """Find predicates that all changed objects satisfy but no unchanged objects do."""
    predicates = []

    # Predicate: specific color
    changed_colors_per_demo = [set(cp["color"] for cp in dc) for dc in per_demo_changed]
    unchanged_colors_per_demo = [set(up["color"] for up in du) for du in per_demo_unchanged]
    common_changed_colors = changed_colors_per_demo[0]
    for s in changed_colors_per_demo[1:]:
        common_changed_colors &= s
    all_unchanged_colors = set()
    for s in unchanged_colors_per_demo:
        all_unchanged_colors |= s
    exclusive_colors = common_changed_colors - all_unchanged_colors
    for c in exclusive_colors:
        predicates.append((f"color={c}", {"color": c}, lambda o, col=c: o.color == col))

    # Predicate: singleton
    all_changed_singleton = all(all(cp["singleton"] for cp in dc) for dc in per_demo_changed)
    any_unchanged_singleton = any(any(up["singleton"] for up in du) for du in per_demo_unchanged)
    if all_changed_singleton and not any_unchanged_singleton:
        predicates.append(("singleton", {"singleton": True}, lambda o: o.is_singleton))

    # Predicate: not singleton
    all_changed_not_singleton = all(all(not cp["singleton"] for cp in dc) for dc in per_demo_changed)
    any_unchanged_not_singleton = any(any(not up["singleton"] for up in du) for du in per_demo_unchanged)
    if all_changed_not_singleton and not any_unchanged_not_singleton:
        predicates.append(("not_singleton", {"singleton": False}, lambda o: not o.is_singleton))

    # Predicate: contained by another object
    all_changed_contained = all(all(cp["contained"] for cp in dc) for dc in per_demo_changed)
    any_unchanged_contained = any(any(up["contained"] for up in du) for du in per_demo_unchanged)
    if all_changed_contained and not any_unchanged_contained:
        predicates.append(("contained", {"contained": True}, lambda o: True))

    # Predicate: color AND singleton
    for c in range(10):
        all_match = all(
            all(cp["color"] == c and cp["singleton"] for cp in dc)
            for dc in per_demo_changed
        )
        any_unchanged_match = any(
            any(up["color"] == c and up["singleton"] for up in du)
            for du in per_demo_unchanged
        )
        if all_match and not any_unchanged_match:
            predicates.append((f"color={c}&singleton", {"color": c, "singleton": True}, None))

    # Predicate: color AND not singleton
    for c in range(10):
        all_match = all(
            all(cp["color"] == c and not cp["singleton"] for cp in dc)
            for dc in per_demo_changed
        )
        any_unchanged_match = any(
            any(up["color"] == c and not up["singleton"] for up in du)
            for du in per_demo_unchanged
        )
        if all_match and not any_unchanged_match:
            predicates.append((f"color={c}&not_singleton", {"color": c, "singleton": False}, None))

    return predicates
