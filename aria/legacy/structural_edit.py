"""Bounded structural edits over near-miss executable candidates.

Three edit tiers:
1. **Shallow syntax edits**: literal swaps, op replacement,
   overlay wrapping, step append. Applied to any program.
2. **Semantic observation edits**: edits over the observation/pipeline/
   shared-composition structure using ObjectRule metadata from observe.py.
3. **Correspondence-aware movement edits**: match objects between input/output
   by color+shape, infer per-subgroup movement deltas, rebuild pipeline programs.

Semantic edits include:
- grouping key swap (color <-> color_size, proximity near <-> far, directional sibling)
- move delta edit (±1 on dr/dc)
- recolor map edit (swap color values)
- add/remove rule in shared composition
- flat -> pipeline or pipeline -> shared-composition promotion

Correspondence edits include:
- per-color uniform movement (all objects of a color move by same delta)
- per-color-size subgroup movement (split by size within color)
- mixed grouping: color-level where consistent, color+size refinement where not
- multi-rule shared composition from mixed correspondence groups

Each edit is verified first against repaired targets, then against
original demos for final acceptance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from aria.runtime.ops import OpSignature, all_ops, has_op
from aria.runtime.program import program_to_text
from aria.types import (
    Bind, Call, DemoPair, Expr, Grid, Literal, Program, Ref, Type,
)
from aria.verify.verifier import verify

# Callback type for program reranking.  Accepts program text list,
# returns (indices, changed, policy_name).  Kept as a protocol-free
# callable so callers can pass a lambda or a LocalPolicy method.
from typing import Callable

_ProgramRanker = Callable[[list[str]], tuple[tuple[int, ...], bool, str]]


@dataclass(frozen=True)
class RerankingReport:
    """Compact report on whether reranking changed candidate order."""

    applied: bool = False
    policy_name: str = ""
    changed_order: bool = False
    programs_ranked: int = 0
    original_order: tuple[int, ...] = ()
    reranked_order: tuple[int, ...] = ()


@dataclass(frozen=True)
class EditResult:
    """Outcome of structural edit search."""

    solved: bool
    winning_program: Program | None
    winning_edit: str | None
    candidates_tried: int
    matched_repaired_target: bool
    solved_original_task: bool
    winning_family: str | None = None  # "shallow", "semantic", or None
    family_breakdown: dict[str, int] | None = None
    reranking: RerankingReport | None = None


def structural_edit_search(
    demos: tuple[DemoPair, ...],
    near_miss_programs: list[Program],
    repaired_targets: tuple[Grid, ...] | None = None,
    *,
    observation_rules: list | None = None,
    repair_error_class: str | None = None,
    max_edits_per_program: int = 40,
    max_programs: int = 5,
    max_semantic_edits: int = 60,
    program_ranker: _ProgramRanker | None = None,
) -> EditResult:
    """Try bounded structural edits on near-miss programs.

    When ``program_ranker`` is None (default), candidates are tried in
    generation order: semantic first, then correspondence, then shallow.

    When ``program_ranker`` is provided, all candidates from every phase
    are collected first, ranked as a single batch, then tried in ranked
    order.  Verification remains the sole correctness gate.
    """
    if not demos:
        return EditResult(
            solved=False, winning_program=None, winning_edit=None,
            candidates_tried=0, matched_repaired_target=False,
            solved_original_task=False,
        )

    target_demos = None
    if repaired_targets is not None and len(repaired_targets) == len(demos):
        target_demos = tuple(
            DemoPair(input=d.input, output=t)
            for d, t in zip(demos, repaired_targets)
        )

    # -- Collect candidates from all three generators ----------------------

    candidates: list[tuple[Program, str, str]] = []  # (program, desc, family)

    if observation_rules:
        for edited, desc in _generate_semantic_edits(
            observation_rules, demos,
            error_class=repair_error_class,
            max_edits=max_semantic_edits,
        ):
            candidates.append((edited, desc, "semantic"))

    same_dims = all(d.input.shape == d.output.shape for d in demos)
    if same_dims:
        for edited, desc in _correspondence_movement_edits(demos, max_edits=30):
            candidates.append((edited, desc, "correspondence"))

    for prog in (near_miss_programs or [])[:max_programs]:
        for edited, desc in _generate_shallow_edits(prog, demos, max_edits_per_program):
            candidates.append((edited, desc, "shallow"))

    # -- Optionally rerank the whole batch ---------------------------------

    reranking_report: RerankingReport | None = None

    if program_ranker is not None and candidates:
        from aria.runtime.program import program_to_text as _p2t
        texts = [_p2t(prog) for prog, _, _ in candidates]
        indices, changed, policy_name = program_ranker(texts)
        reranking_report = RerankingReport(
            applied=True,
            policy_name=policy_name,
            changed_order=changed,
            programs_ranked=len(candidates),
            original_order=tuple(range(len(candidates))),
            reranked_order=indices,
        )
        if changed:
            candidates = [candidates[i] for i in indices]

    # -- Try candidates in order -------------------------------------------

    tried = 0
    best_matched_target = False
    family_counts: dict[str, int] = {"semantic": 0, "correspondence": 0, "shallow": 0}

    for edited, desc, family in candidates:
        tried += 1
        family_counts[family] = family_counts.get(family, 0) + 1

        if target_demos is not None:
            vr = verify(edited, target_demos)
            if not vr.passed:
                continue
            best_matched_target = True

        vr_orig = verify(edited, demos)
        if vr_orig.passed:
            return EditResult(
                solved=True, winning_program=edited,
                winning_edit=desc,
                candidates_tried=tried,
                matched_repaired_target=best_matched_target,
                solved_original_task=True,
                winning_family=family,
                family_breakdown=dict(family_counts),
                reranking=reranking_report,
            )

    return EditResult(
        solved=False, winning_program=None, winning_edit=None,
        candidates_tried=tried,
        matched_repaired_target=best_matched_target,
        solved_original_task=False,
        family_breakdown=dict(family_counts),
        reranking=reranking_report,
    )


# ---------------------------------------------------------------------------
# Semantic edits: operate on ObjectRule metadata, rebuild programs
# ---------------------------------------------------------------------------


# Grouping key alternatives for swap edits
_GROUPING_SWAPS = {
    "color": ["color_size"],
    "color_size": ["color"],
    "proximity": [],  # near/far flip handled separately
    "size_rank": [],   # largest/smallest flip handled separately
}

# Direction sibling map
_DIRECTION_SIBLINGS = {
    "above": ["below", "left", "right"],
    "below": ["above", "left", "right"],
    "left": ["right", "above", "below"],
    "right": ["left", "above", "below"],
}


def _generate_semantic_edits(
    rules: list,
    demos: tuple[DemoPair, ...],
    *,
    error_class: str | None = None,
    max_edits: int = 60,
):
    """Yield (program, description) from semantic rule mutations.

    Priority order based on error_class signal:
    - color_swap -> recolor map edits first
    - missing_content -> add-rule / promotion edits first
    - default -> grouping swaps first, then delta edits
    """
    from aria.observe import (
        ObjectRule,
        build_pipeline,
        build_shared_composition,
        rules_to_programs,
    )

    yielded = 0

    # Determine edit priority order
    if error_class == "color_swap":
        edit_fns = [_recolor_edits, _grouping_edits, _delta_edits,
                    _near_far_edits, _direction_edits,
                    _composition_edits, _promotion_edits]
    elif error_class in ("missing_content", "mixed"):
        edit_fns = [_composition_edits, _promotion_edits,
                    _grouping_edits, _recolor_edits,
                    _delta_edits, _near_far_edits, _direction_edits]
    else:
        edit_fns = [_grouping_edits, _delta_edits, _near_far_edits,
                    _direction_edits, _recolor_edits,
                    _composition_edits, _promotion_edits]

    for edit_fn in edit_fns:
        for prog, desc in edit_fn(rules, demos):
            if yielded >= max_edits:
                return
            yield prog, desc
            yielded += 1


def _grouping_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Swap grouping key: color <-> color_size."""
    from aria.observe import ObjectRule, build_pipeline, rules_to_programs

    for i, rule in enumerate(rules):
        if not rule.expressible:
            continue
        alts = _GROUPING_SWAPS.get(rule.grouping_key, [])
        for new_gkey in alts:
            mutated = _mutate_rule_grouping(rule, new_gkey)
            if mutated is None:
                continue
            # Try as pipeline
            prog = build_pipeline(mutated)
            if prog is not None:
                yield prog, f"semantic:grouping_swap:{rule.grouping_key}->{new_gkey}:{rule.kind}:{rule.input_color}"
            # Try rebuilt via rules_to_programs with mutated rule list
            mutated_rules = list(rules)
            mutated_rules[i] = mutated
            for prog, _ in rules_to_programs(mutated_rules, demos):
                yield prog, f"semantic:grouping_swap_rebuild:{rule.grouping_key}->{new_gkey}:{rule.kind}:{rule.input_color}"


def _delta_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Edit move delta: try ±1 on dr and dc."""
    from aria.observe import ObjectRule, build_pipeline

    for rule in rules:
        if rule.kind != "move" or not rule.expressible:
            continue
        dr = rule.details.get("dr", 0)
        dc = rule.details.get("dc", 0)
        for new_dr, new_dc in [
            (dr + 1, dc), (dr - 1, dc), (dr, dc + 1), (dr, dc - 1),
            (dr + 1, dc + 1), (dr - 1, dc - 1),
        ]:
            if new_dr == dr and new_dc == dc:
                continue
            new_details = dict(rule.details)
            new_details["dr"] = new_dr
            new_details["dc"] = new_dc
            mutated = ObjectRule(
                kind=rule.kind, input_color=rule.input_color,
                output_color=rule.output_color, offsets=rule.offsets,
                details=new_details, expressible=True,
                grouping_key=rule.grouping_key, expression_path=rule.expression_path,
            )
            prog = build_pipeline(mutated)
            if prog is not None:
                yield prog, f"semantic:delta_edit:({dr},{dc})->({new_dr},{new_dc}):{rule.input_color}"


def _near_far_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Flip near/far in proximity grouping."""
    from aria.observe import ObjectRule, build_pipeline, rules_to_programs

    for i, rule in enumerate(rules):
        if rule.grouping_key != "proximity" or not rule.expressible:
            continue
        gkey = rule.details.get("group_key", ())
        if len(gkey) < 2:
            continue
        flipped_near = not gkey[1]
        new_gkey = (gkey[0], flipped_near)
        new_details = dict(rule.details)
        new_details["group_key"] = new_gkey
        mutated = ObjectRule(
            kind=rule.kind, input_color=rule.input_color,
            output_color=rule.output_color, offsets=rule.offsets,
            details=new_details, expressible=True,
            grouping_key="proximity", expression_path=rule.expression_path,
        )
        prog = build_pipeline(mutated)
        if prog is not None:
            yield prog, f"semantic:near_far_flip:{gkey[1]}->{flipped_near}:{rule.kind}:{rule.input_color}"


def _direction_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Swap directional subgroup: above <-> below, left <-> right, etc."""
    from aria.observe import ObjectRule, build_pipeline

    for rule in rules:
        if not rule.grouping_key.startswith("direction_") or not rule.expressible:
            continue
        direction = rule.details.get("direction")
        if direction is None:
            continue
        siblings = _DIRECTION_SIBLINGS.get(direction, [])
        for new_dir in siblings:
            new_details = dict(rule.details)
            new_details["direction"] = new_dir
            gkey = rule.details.get("group_key", ())
            # Keep same in_direction boolean, just change which direction
            mutated = ObjectRule(
                kind=rule.kind, input_color=rule.input_color,
                output_color=rule.output_color, offsets=rule.offsets,
                details=new_details, expressible=True,
                grouping_key=f"direction_{new_dir}",
                expression_path=rule.expression_path,
            )
            prog = build_pipeline(mutated)
            if prog is not None:
                yield prog, f"semantic:direction_swap:{direction}->{new_dir}:{rule.kind}:{rule.input_color}"

        # Also flip in_direction boolean
        gkey = rule.details.get("group_key", ())
        if len(gkey) >= 2:
            new_gkey = (gkey[0], not gkey[1])
            new_details = dict(rule.details)
            new_details["group_key"] = new_gkey
            mutated = ObjectRule(
                kind=rule.kind, input_color=rule.input_color,
                output_color=rule.output_color, offsets=rule.offsets,
                details=new_details, expressible=True,
                grouping_key=rule.grouping_key,
                expression_path=rule.expression_path,
            )
            prog = build_pipeline(mutated)
            if prog is not None:
                yield prog, f"semantic:direction_invert:{direction}:in={gkey[1]}->in={not gkey[1]}:{rule.kind}:{rule.input_color}"


def _recolor_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Edit recolor maps: swap target colors between rules, try output palette."""
    from aria.observe import ObjectRule, build_pipeline, rules_to_programs

    output_colors = set()
    for d in demos:
        output_colors.update(int(v) for v in np.unique(d.output))

    recolor_rules = [r for r in rules if r.kind == "recolor" and r.details.get("color_map")]
    for i, rule in enumerate(recolor_rules):
        cmap = rule.details["color_map"]
        for old_c, cur_target in cmap.items():
            for new_target in sorted(output_colors):
                if new_target == cur_target:
                    continue
                new_cmap = dict(cmap)
                new_cmap[old_c] = new_target
                new_details = dict(rule.details)
                new_details["color_map"] = new_cmap
                mutated = ObjectRule(
                    kind="recolor", input_color=rule.input_color,
                    output_color=new_target, offsets=rule.offsets,
                    details=new_details, expressible=True,
                    grouping_key=rule.grouping_key,
                    expression_path=rule.expression_path,
                )
                # Try pipeline rebuild
                prog = build_pipeline(mutated)
                if prog is not None:
                    yield prog, f"semantic:recolor_edit:{old_c}:{cur_target}->{new_target}"
                # Try in full rule set
                mutated_rules = list(rules)
                mutated_rules[rules.index(rule)] = mutated
                for prog, _ in rules_to_programs(mutated_rules, demos):
                    yield prog, f"semantic:recolor_edit_rebuild:{old_c}:{cur_target}->{new_target}"
                break  # one swap per source color to keep bounded


def _composition_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Add/remove one rule from shared composition candidates."""
    from aria.observe import ObjectRule, build_shared_composition

    composable = [r for r in rules if r.expressible and r.kind in ("remove", "move", "recolor", "rigid_transform")]
    if len(composable) < 2:
        return

    # Try removing one rule from composition
    for i in range(len(composable)):
        subset = composable[:i] + composable[i + 1:]
        if len(subset) < 2:
            continue
        result = build_shared_composition(subset, demos)
        if result is not None:
            prog, _ = result
            removed = composable[i]
            yield prog, f"semantic:composition_remove:{removed.kind}:{removed.input_color}"

    # Try adding each non-composable rule that is expressible
    other_rules = [r for r in rules if r.expressible and r not in composable]
    for extra in other_rules:
        candidate = composable + [extra]
        result = build_shared_composition(candidate, demos)
        if result is not None:
            prog, _ = result
            yield prog, f"semantic:composition_add:{extra.kind}:{extra.input_color}"


def _promotion_edits(rules: list, demos: tuple[DemoPair, ...]):
    """Promote flat->pipeline or pipeline->shared-composition."""
    from aria.observe import ObjectRule, build_pipeline, build_shared_composition

    # Flat rules that could become pipelines
    for rule in rules:
        if not rule.expressible:
            continue
        if rule.expression_path == "flat" and rule.kind in ("recolor", "remove", "move", "rigid_transform"):
            prog = build_pipeline(rule)
            if prog is not None:
                yield prog, f"semantic:promote_flat_to_pipeline:{rule.kind}:{rule.input_color}"

    # Pipeline rules that could join a shared composition
    pipeline_rules = [r for r in rules if r.expressible and r.expression_path == "pipeline"]
    if len(pipeline_rules) >= 2:
        result = build_shared_composition(pipeline_rules, demos)
        if result is not None:
            prog, _ = result
            yield prog, f"semantic:promote_pipeline_to_composition:{len(pipeline_rules)}_rules"


def _mutate_rule_grouping(rule, new_grouping_key: str):
    """Create a new rule with a different grouping key, adjusting group_key tuple."""
    from aria.observe import ObjectRule

    gkey = rule.details.get("group_key", (rule.input_color,))
    color = gkey[0] if gkey else rule.input_color

    if new_grouping_key == "color":
        new_gkey = (color,)
    elif new_grouping_key == "color_size":
        # Use the rule's known size if available, else guess common sizes
        size = gkey[1] if len(gkey) >= 2 else rule.details.get("size", 1)
        new_gkey = (color, size)
    else:
        return None

    new_details = dict(rule.details)
    new_details["group_key"] = new_gkey
    return ObjectRule(
        kind=rule.kind, input_color=rule.input_color,
        output_color=rule.output_color, offsets=rule.offsets,
        details=new_details, expressible=True,
        grouping_key=new_grouping_key,
        expression_path=rule.expression_path,
    )


# ---------------------------------------------------------------------------
# Correspondence-aware movement edits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ObjectMatch:
    """One-to-one correspondence between an input object and an output object."""

    input_color: int
    input_row: int
    input_col: int
    input_size: int
    output_row: int
    output_col: int
    dr: int  # output_row - input_row
    dc: int  # output_col - input_col
    mask_bytes: bytes  # for dedup and shape comparison


def _extract_foreground_objects(grid: Grid) -> list[dict]:
    """Extract connected-component objects from a grid.

    Returns lightweight dicts with: color, row, col, size, mask_bytes.
    Uses 4-connectivity, ignores color 0 (background).
    """
    from scipy import ndimage

    objects = []
    rows, cols = grid.shape
    for color in range(1, 10):
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labeled == label_id)
            min_r, max_r = int(ys.min()), int(ys.max())
            min_c, max_c = int(xs.min()), int(xs.max())
            mask = labeled[min_r:max_r + 1, min_c:max_c + 1] == label_id
            objects.append({
                "color": color,
                "row": min_r,
                "col": min_c,
                "size": int(mask.sum()),
                "mask": mask,
                "mask_bytes": mask.tobytes(),
                "mask_shape": mask.shape,
            })
    return objects


def _match_objects(
    inp_objs: list[dict],
    out_objs: list[dict],
) -> list[_ObjectMatch] | None:
    """Match input objects to output objects by color + exact mask shape.

    Returns None if any ambiguity (multiple possible matches for one object).
    Only returns matches; unmatched objects are ignored (they may have been
    added/removed, which is fine — we only care about moved objects).
    """
    matches: list[_ObjectMatch] = []

    # Group output objects by (color, mask_bytes, mask_shape) for fast lookup
    out_by_key: dict[tuple, list[dict]] = {}
    for obj in out_objs:
        key = (obj["color"], obj["mask_bytes"], obj["mask_shape"])
        out_by_key.setdefault(key, []).append(obj)

    # For each input object, find matching output objects
    used_out: set[int] = set()
    for inp in inp_objs:
        key = (inp["color"], inp["mask_bytes"], inp["mask_shape"])
        candidates = out_by_key.get(key, [])
        # Filter out already-used output objects
        available = [c for c in candidates if id(c) not in used_out]

        if len(available) == 0:
            # Object was removed or changed shape — skip
            continue
        if len(available) == 1:
            out = available[0]
            used_out.add(id(out))
            matches.append(_ObjectMatch(
                input_color=inp["color"],
                input_row=inp["row"], input_col=inp["col"],
                input_size=inp["size"],
                output_row=out["row"], output_col=out["col"],
                dr=out["row"] - inp["row"],
                dc=out["col"] - inp["col"],
                mask_bytes=inp["mask_bytes"],
            ))
        else:
            # Multiple candidates of same color+shape — use proximity to disambiguate
            # Sort by distance to input position; if two are equidistant, ambiguous → skip all
            by_dist = sorted(available, key=lambda o: abs(o["row"] - inp["row"]) + abs(o["col"] - inp["col"]))
            d0 = abs(by_dist[0]["row"] - inp["row"]) + abs(by_dist[0]["col"] - inp["col"])
            d1 = abs(by_dist[1]["row"] - inp["row"]) + abs(by_dist[1]["col"] - inp["col"])
            if d0 == d1:
                # Ambiguous — skip this input object
                continue
            out = by_dist[0]
            used_out.add(id(out))
            matches.append(_ObjectMatch(
                input_color=inp["color"],
                input_row=inp["row"], input_col=inp["col"],
                input_size=inp["size"],
                output_row=out["row"], output_col=out["col"],
                dr=out["row"] - inp["row"],
                dc=out["col"] - inp["col"],
                mask_bytes=inp["mask_bytes"],
            ))

    return matches if matches else None


def _correspondence_movement_edits(
    demos: tuple[DemoPair, ...],
    *,
    max_edits: int = 30,
):
    """Yield (program, description) from correspondence-aware object movement.

    Uses mixed grouping: color-level for colors that are consistent,
    color+size refinement for colors that aren't. Both kinds of rules
    can coexist in a single shared-composition candidate.

    Steps:
    1. Extract objects from input and output of each demo
    2. Match by color + exact mask shape
    3. Find consistent color-level groups, identify inconsistent colors
    4. Refine inconsistent colors to color+size where that becomes consistent
    5. Build executable candidates from the merged rule set
    """
    from aria.observe import ObjectRule, build_pipeline, build_shared_composition

    if not demos:
        return

    # Step 1-2: match objects across all demos
    per_demo_matches: list[list[_ObjectMatch]] = []
    for demo in demos:
        inp_objs = _extract_foreground_objects(demo.input)
        out_objs = _extract_foreground_objects(demo.output)
        matches = _match_objects(inp_objs, out_objs)
        if matches is None:
            return  # no valid correspondences in at least one demo
        per_demo_matches.append(matches)

    # Step 3: color-level consistency check
    color_consistent, color_inconsistent = _find_consistent_color_deltas(per_demo_matches)

    # Step 4: refine inconsistent colors to color+size
    cs_refined: dict[tuple[int, int], tuple[int, int]] = {}
    if color_inconsistent:
        cs_refined = _find_consistent_color_size_deltas(
            per_demo_matches, only_colors=color_inconsistent,
        )

    # Check we have at least one moving group from either source
    has_color_movement = any(d != (0, 0) for d in color_consistent.values())
    has_cs_movement = any(d != (0, 0) for d in cs_refined.values())
    if not has_color_movement and not has_cs_movement:
        return

    yielded = 0

    # Build single-group pipeline candidates (color-level)
    for color, (dr, dc) in sorted(color_consistent.items()):
        if dr == 0 and dc == 0:
            continue
        rule = ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color,)},
            expressible=True, grouping_key="color",
            expression_path="pipeline",
        )
        prog = build_pipeline(rule)
        if prog is not None:
            yield prog, f"correspondence:move:color={color}:dr={dr}:dc={dc}"
            yielded += 1
            if yielded >= max_edits:
                return

    # Build single-group pipeline candidates (color+size refined)
    for (color, size), (dr, dc) in sorted(cs_refined.items()):
        if dr == 0 and dc == 0:
            continue
        rule = ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color, size)},
            expressible=True, grouping_key="color_size",
            expression_path="pipeline",
        )
        prog = build_pipeline(rule)
        if prog is not None:
            yield prog, f"correspondence:move_cs:color={color}:size={size}:dr={dr}:dc={dc}"
            yielded += 1
            if yielded >= max_edits:
                return

    # Build combined shared-composition from all moving groups
    all_rules: list[ObjectRule] = []
    color_grouping_count = 0
    cs_grouping_count = 0

    for color, (dr, dc) in sorted(color_consistent.items()):
        if dr == 0 and dc == 0:
            continue
        all_rules.append(ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color,)},
            expressible=True, grouping_key="color",
            expression_path="pipeline",
        ))
        color_grouping_count += 1

    for (color, size), (dr, dc) in sorted(cs_refined.items()):
        if dr == 0 and dc == 0:
            continue
        all_rules.append(ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color, size)},
            expressible=True, grouping_key="color_size",
            expression_path="pipeline",
        ))
        cs_grouping_count += 1

    # Add remove rules for disappeared colors
    all_input_colors = set()
    all_output_colors = set()
    for demo in demos:
        all_input_colors.update(int(v) for v in np.unique(demo.input) if v != 0)
        all_output_colors.update(int(v) for v in np.unique(demo.output) if v != 0)
    disappeared = all_input_colors - all_output_colors
    for color in sorted(disappeared):
        all_rules.append(ObjectRule(
            kind="remove", input_color=color, output_color=None,
            offsets=None,
            details={"group_key": (color,)},
            expressible=True, grouping_key="color",
            expression_path="pipeline",
        ))

    if len(all_rules) >= 2 and yielded < max_edits:
        result = build_shared_composition(all_rules, demos)
        if result is not None:
            prog, _ = result
            is_mixed = color_grouping_count > 0 and cs_grouping_count > 0
            label = "shared_move_mixed" if is_mixed else "shared_move"
            yield prog, f"correspondence:{label}:color_groups={color_grouping_count}:cs_groups={cs_grouping_count}"
            yielded += 1


def _find_consistent_color_deltas(
    per_demo_matches: list[list[_ObjectMatch]],
) -> tuple[dict[int, tuple[int, int]], set[int]]:
    """Find per-color deltas that are consistent across all demos.

    Returns (consistent, inconsistent) where:
    - consistent: {color: (dr, dc)} for colors with uniform delta
    - inconsistent: set of colors that failed color-level consistency
    """
    # Collect all (color, dr, dc) observations per demo
    per_demo_color_deltas: list[dict[int, set[tuple[int, int]]]] = []
    for matches in per_demo_matches:
        color_deltas: dict[int, set[tuple[int, int]]] = {}
        for m in matches:
            color_deltas.setdefault(m.input_color, set()).add((m.dr, m.dc))
        per_demo_color_deltas.append(color_deltas)

    # A color is consistent if all its objects have the same delta in every demo
    all_colors = set()
    for cd in per_demo_color_deltas:
        all_colors.update(cd.keys())

    consistent: dict[int, tuple[int, int]] = {}
    inconsistent: set[int] = set()
    for color in all_colors:
        delta_set: set[tuple[int, int]] | None = None
        ok = True
        for cd in per_demo_color_deltas:
            if color not in cd:
                continue
            if len(cd[color]) != 1:
                ok = False
                break
            if delta_set is None:
                delta_set = cd[color]
            elif delta_set != cd[color]:
                ok = False
                break
        if ok and delta_set is not None and len(delta_set) == 1:
            consistent[color] = next(iter(delta_set))
        else:
            inconsistent.add(color)

    return consistent, inconsistent


def _find_consistent_color_size_deltas(
    per_demo_matches: list[list[_ObjectMatch]],
    *,
    only_colors: set[int] | None = None,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Find per-(color, size) deltas that are consistent across all demos.

    Args:
        only_colors: if given, only consider matches for these colors.

    Returns {(color, size): (dr, dc)} for consistent groups (may be empty).
    """
    per_demo_cs_deltas: list[dict[tuple[int, int], set[tuple[int, int]]]] = []
    for matches in per_demo_matches:
        cs_deltas: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for m in matches:
            if only_colors is not None and m.input_color not in only_colors:
                continue
            key = (m.input_color, m.input_size)
            cs_deltas.setdefault(key, set()).add((m.dr, m.dc))
        per_demo_cs_deltas.append(cs_deltas)

    all_keys = set()
    for csd in per_demo_cs_deltas:
        all_keys.update(csd.keys())

    result: dict[tuple[int, int], tuple[int, int]] = {}
    for key in all_keys:
        delta_set: set[tuple[int, int]] | None = None
        ok = True
        for csd in per_demo_cs_deltas:
            if key not in csd:
                continue
            if len(csd[key]) != 1:
                ok = False
                break
            if delta_set is None:
                delta_set = csd[key]
            elif delta_set != csd[key]:
                ok = False
                break
        if ok and delta_set is not None and len(delta_set) == 1:
            result[key] = next(iter(delta_set))

    return result


def _build_movement_programs_color_size(
    cs_deltas: dict[tuple[int, int], tuple[int, int]],
    demos: tuple[DemoPair, ...],
    max_edits: int,
):
    """Build movement programs from (color, size) grouped correspondences."""
    from aria.observe import ObjectRule, build_pipeline, build_shared_composition

    yielded = 0
    rules = []

    for (color, size), (dr, dc) in sorted(cs_deltas.items()):
        if dr == 0 and dc == 0:
            continue
        rule = ObjectRule(
            kind="move", input_color=color, output_color=None,
            offsets=None,
            details={"dr": dr, "dc": dc, "group_key": (color, size)},
            expressible=True, grouping_key="color_size",
            expression_path="pipeline",
        )
        prog = build_pipeline(rule)
        if prog is not None:
            yield prog, f"correspondence:move_cs:color={color}:size={size}:dr={dr}:dc={dc}"
            yielded += 1
            if yielded >= max_edits:
                return
        rules.append(rule)

    if len(rules) >= 2 and yielded < max_edits:
        result = build_shared_composition(rules, demos)
        if result is not None:
            prog, _ = result
            yield prog, f"correspondence:shared_move_cs:{len(rules)}_groups"
            yielded += 1


# ---------------------------------------------------------------------------
# Shallow syntax edits (original implementation)
# ---------------------------------------------------------------------------


def _generate_shallow_edits(
    prog: Program,
    demos: tuple[DemoPair, ...],
    max_edits: int,
):
    """Yield (edited_program, description) pairs from syntax-level edits."""
    yielded = 0

    # Edit 1: swap individual literals
    expected_colors = set()
    expected_ints = set()
    for demo in demos:
        expected_colors.update(int(v) for v in np.unique(demo.output))
        expected_ints.update(int(v) for v in np.unique(demo.output))
        expected_ints.update([int(demo.output.shape[0]), int(demo.output.shape[1])])

    for step_idx, step in enumerate(prog.steps):
        if not isinstance(step, Bind) or not isinstance(step.expr, Call):
            continue
        for arg_idx, arg in enumerate(step.expr.args):
            if not isinstance(arg, Literal) or not isinstance(arg.value, int):
                continue
            candidates = expected_colors if arg.typ == Type.COLOR else expected_ints
            for new_val in sorted(candidates):
                if new_val == arg.value or yielded >= max_edits:
                    continue
                new_args = list(step.expr.args)
                new_args[arg_idx] = Literal(new_val, arg.typ)
                new_step = Bind(step.name, step.typ, Call(step.expr.op, tuple(new_args)), step.declared)
                new_steps = list(prog.steps)
                new_steps[step_idx] = new_step
                yield Program(tuple(new_steps), prog.output), f"literal_swap:step{step_idx}:arg{arg_idx}:{arg.value}->{new_val}"
                yielded += 1

    # Edit 2: op replacement (same return type, same param count+types)
    ops = all_ops()
    for step_idx, step in enumerate(prog.steps):
        if not isinstance(step, Bind) or not isinstance(step.expr, Call):
            continue
        if not has_op(step.expr.op):
            continue
        orig_sig = ops.get(step.expr.op)
        if orig_sig is None:
            continue
        for alt_name, alt_sig in ops.items():
            if alt_name == step.expr.op or yielded >= max_edits:
                continue
            if alt_sig.return_type != orig_sig.return_type:
                continue
            if len(alt_sig.params) != len(orig_sig.params):
                continue
            if not all(a[1] == b[1] for a, b in zip(alt_sig.params, orig_sig.params)):
                continue
            new_step = Bind(step.name, step.typ, Call(alt_name, step.expr.args), step.declared)
            new_steps = list(prog.steps)
            new_steps[step_idx] = new_step
            yield Program(tuple(new_steps), prog.output), f"op_swap:step{step_idx}:{step.expr.op}->{alt_name}"
            yielded += 1

    # Edit 3: overlay wrapping
    if yielded < max_edits:
        idx = len(prog.steps)
        for wrap_desc, args in [
            ("overlay(input,prog)", (Ref("input"), Ref(prog.output))),
            ("overlay(prog,input)", (Ref(prog.output), Ref("input"))),
        ]:
            wrapped = Program(
                steps=prog.steps + (Bind(f"v{idx}", Type.GRID, Call("overlay", args)),),
                output=f"v{idx}",
            )
            yield wrapped, f"wrap:{wrap_desc}"
            yielded += 1

    # Edit 4: append one 1-param GRID→GRID op
    if yielded < max_edits:
        idx = len(prog.steps)
        for name, sig in sorted(ops.items()):
            if yielded >= max_edits:
                break
            if sig.return_type == Type.GRID and len(sig.params) == 1 and sig.params[0][1] == Type.GRID:
                appended = Program(
                    steps=prog.steps + (Bind(f"v{idx}", Type.GRID, Call(name, (Ref(prog.output),))),),
                    output=f"v{idx}",
                )
                yield appended, f"append:{name}"
                yielded += 1


# Keep old name for backward compatibility
_generate_edits = _generate_shallow_edits
