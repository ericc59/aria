"""Output-anchored backward solver prototype.

Core loop:
1. Decompose each train output into bounded substructures (regions)
2. Align each output region with corresponding input support
3. Induce local rules per aligned pair
4. Unify local rules across regions and demos into one shared rule
5. Apply the shared rule to test input
6. Verify on train demos

The hypothesis: ARC-2 tasks are better solved by explaining the output
piece-by-piece rather than forward-searching whole-grid transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.decomposition import detect_bg
from aria.prototype.local_rule_induction import LocalRule, induce_local_rules
from aria.prototype.output_regions import OutputRegion, decompose_output
from aria.prototype.region_alignment import RegionAlignment, align_regions
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BackwardSolveResult:
    """Result of the backward solver on one task."""
    task_id: str
    train_verified: bool
    test_outputs: list[Grid]
    shared_rule: LocalRule | None
    per_demo_rules: list[list[LocalRule]]  # rules per demo per region
    train_diff: int
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def backward_solve(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> BackwardSolveResult:
    """Run the output-anchored backward solver on one task.

    For same-shape tasks:
    1. Find changed regions in each demo
    2. Induce local rules per changed region
    3. Try to unify across demos into one shared rule
    4. Apply the shared rule to verify on all demos

    For diff-shape tasks:
    1. Decompose each output into substructures
    2. Find corresponding input support
    3. Induce per-region rules
    4. Unify and verify
    """
    if not demos:
        return _empty_result(task_id)

    same_shape = all(d.input.shape == d.output.shape for d in demos)

    if same_shape:
        return _solve_same_shape(demos, task_id)
    else:
        return _solve_diff_shape(demos, task_id)


# ---------------------------------------------------------------------------
# Same-shape solver
# ---------------------------------------------------------------------------


def _solve_same_shape(
    demos: tuple[DemoPair, ...],
    task_id: str,
) -> BackwardSolveResult:
    """Backward solve for same-shape tasks.

    Strategy: the transformation is local edits to the grid.
    Find the changed regions, induce rules, unify.
    """
    bg = detect_bg(demos[0].input)

    # Phase 1: Decompose and align each demo independently
    per_demo_alignments: list[list[RegionAlignment]] = []
    per_demo_rules: list[list[LocalRule]] = []

    for di, demo in enumerate(demos):
        regions = decompose_output(demo.input, demo.output, bg)
        changed = [r for r in regions if r.changed]

        if not changed:
            per_demo_alignments.append([])
            per_demo_rules.append([])
            continue

        alignments = align_regions(demo.input, demo.output, changed, bg)
        per_demo_alignments.append(alignments)

        # Induce rules per region
        demo_rules = []
        for alignment in alignments:
            rules = induce_local_rules(alignment, bg)
            if rules:
                demo_rules.append(rules[0])  # take best rule
        per_demo_rules.append(demo_rules)

    # Phase 2: Try whole-grid rules first (more generalizable)
    whole_grid_result = _try_whole_grid_rules(demos, bg, task_id)
    if whole_grid_result is not None and whole_grid_result.train_verified:
        return whole_grid_result

    # Phase 3: Unify local rules across demos
    unified = _unify_rules_across_demos(per_demo_rules, demos, bg)

    if unified is not None:
        # Verify the unified rule on all demos
        train_diff = 0
        all_match = True
        for demo in demos:
            predicted = unified.apply(demo.input, bg)
            if not np.array_equal(predicted, demo.output):
                all_match = False
                if predicted.shape == demo.output.shape:
                    train_diff += int(np.sum(predicted != demo.output))
                else:
                    train_diff += demo.output.size

        if all_match:
            return BackwardSolveResult(
                task_id=task_id,
                train_verified=True,
                test_outputs=[],
                shared_rule=unified,
                per_demo_rules=per_demo_rules,
                train_diff=0,
                details={"strategy": "unified_local_rule"},
            )

    # Phase 4: Try per-region mask-recolor as last resort
    # (the mask may be consistent across demos if changes are at the same positions)
    mask_result = _try_consistent_mask(demos, bg, task_id)
    if mask_result is not None:
        return mask_result

    # Fallback: return best partial result
    best_diff = _compute_best_diff(demos, per_demo_rules, bg)
    return BackwardSolveResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        shared_rule=unified,
        per_demo_rules=per_demo_rules,
        train_diff=best_diff,
        details={"strategy": "no_unified_rule_found"},
    )


def _try_whole_grid_rules(
    demos: tuple[DemoPair, ...],
    bg: int,
    task_id: str,
) -> BackwardSolveResult | None:
    """Try rules that operate on the whole grid at once."""
    from aria.prototype.local_rule_induction import induce_local_rules
    from aria.prototype.region_alignment import RegionAlignment
    from aria.prototype.output_regions import OutputRegion

    d0 = demos[0]
    rows, cols = d0.input.shape

    # Create a whole-grid alignment for demo 0
    full_region = OutputRegion(
        bbox=(0, 0, rows - 1, cols - 1),
        content=d0.output.copy(),
        input_content=d0.input.copy(),
        changed=True,
        diff_count=int(np.sum(d0.input != d0.output)),
        region_id="full",
    )
    alignment = RegionAlignment(
        output_region=full_region,
        input_bbox=(0, 0, rows - 1, cols - 1),
        input_content=d0.input.copy(),
        output_content=d0.output.copy(),
        alignment_type="same_bbox",
        confidence=1.0,
    )

    rules = induce_local_rules(alignment, bg)

    # Filter out mask_recolor and constant (not generalizable)
    generalizable = [r for r in rules if r.op not in ("mask_recolor", "constant")]

    for rule in generalizable:
        all_match = True
        for demo in demos:
            demo_bg = detect_bg(demo.input)
            predicted = rule.apply(demo.input, demo_bg)
            if not np.array_equal(predicted, demo.output):
                all_match = False
                break

        if all_match:
            return BackwardSolveResult(
                task_id=task_id,
                train_verified=True,
                test_outputs=[],
                shared_rule=rule,
                per_demo_rules=[[rule]],
                train_diff=0,
                details={"strategy": f"whole_grid_{rule.op}"},
            )

    return None


def _try_consistent_mask(
    demos: tuple[DemoPair, ...],
    bg: int,
    task_id: str,
) -> BackwardSolveResult | None:
    """Check if all demos have the same relative change positions.

    If the mask of changed positions is consistent across demos
    (relative to some structural anchor), the mask itself is the rule.
    """
    # Simple check: same absolute positions change across all demos
    d0 = demos[0]
    diff0 = d0.input != d0.output
    positions0 = set(zip(*np.where(diff0)))

    if not positions0:
        return None

    # Check if changes are at same positions in all demos
    for demo in demos[1:]:
        if demo.input.shape != d0.input.shape:
            return None
        diff_i = demo.input != demo.output
        positions_i = set(zip(*np.where(diff_i)))
        if positions_i != positions0:
            return None

    # Same positions change in every demo — but do they change the same way?
    # Check if there's a consistent color transformation at each position
    color_rules: dict[tuple[int, int], dict[int, int]] = {}
    for r, c in positions0:
        pos_map: dict[int, int] = {}
        consistent = True
        for demo in demos:
            from_c = int(demo.input[r, c])
            to_c = int(demo.output[r, c])
            if from_c in pos_map and pos_map[from_c] != to_c:
                consistent = False
                break
            pos_map[from_c] = to_c
        if not consistent:
            return None
        color_rules[(r, c)] = pos_map

    # Build a unified mask rule
    changes = []
    for (r, c), cmap in color_rules.items():
        for from_c, to_c in cmap.items():
            changes.append((r, c, from_c, to_c))

    rule = LocalRule(
        "mask_recolor",
        {"changes": changes},
        0.9,
        f"consistent mask: {len(changes)} position-color rules",
    )

    # Verify
    all_match = True
    for demo in demos:
        predicted = rule.apply(demo.input, bg)
        if not np.array_equal(predicted, demo.output):
            all_match = False
            break

    if all_match:
        return BackwardSolveResult(
            task_id=task_id,
            train_verified=True,
            test_outputs=[],
            shared_rule=rule,
            per_demo_rules=[[rule]],
            train_diff=0,
            details={"strategy": "consistent_position_mask"},
        )
    return None


# ---------------------------------------------------------------------------
# Rule unification
# ---------------------------------------------------------------------------


def _unify_rules_across_demos(
    per_demo_rules: list[list[LocalRule]],
    demos: tuple[DemoPair, ...],
    bg: int,
) -> LocalRule | None:
    """Find one shared rule that explains all demos.

    Strategy: if all demos' best rules have the same op, try to merge
    their parameters into a single abstract rule.
    """
    # Collect the best rule per demo (skip empty demos)
    best_per_demo = []
    for demo_rules in per_demo_rules:
        if demo_rules:
            # Prefer generalizable rules over masks
            generalizable = [r for r in demo_rules if r.op not in ("mask_recolor", "constant")]
            if generalizable:
                best_per_demo.append(generalizable[0])
            else:
                best_per_demo.append(demo_rules[0])

    if not best_per_demo:
        return None

    # Check if all demos agree on the operation type
    ops = set(r.op for r in best_per_demo)
    if len(ops) == 1:
        shared_op = next(iter(ops))

        if shared_op == "color_map":
            return _unify_color_maps(best_per_demo, demos, bg)
        if shared_op == "fill_enclosed":
            return _unify_fill_enclosed(best_per_demo, demos, bg)
        if shared_op == "periodic_complete":
            return _unify_periodic(best_per_demo, demos, bg)
        if shared_op == "gravity":
            return _unify_gravity(best_per_demo, demos, bg)
        if shared_op == "mirror_repair":
            return _unify_mirror(best_per_demo, demos, bg)

    return None


def _unify_color_maps(rules: list[LocalRule], demos, bg) -> LocalRule | None:
    """Merge color maps from multiple demos into one shared map."""
    merged: dict[int, int] = {}
    for rule in rules:
        cmap = rule.params.get("map", {})
        for from_c, to_c in cmap.items():
            if from_c in merged and merged[from_c] != to_c:
                return None  # conflict
            merged[from_c] = to_c

    unified = LocalRule("color_map", {"map": merged}, 0.95,
                        f"unified color map {merged}")

    # Verify on all demos
    for demo in demos:
        if not np.array_equal(unified.apply(demo.input, bg), demo.output):
            return None
    return unified


def _unify_fill_enclosed(rules: list[LocalRule], demos, bg) -> LocalRule | None:
    colors = set(r.params.get("fill_color") for r in rules)
    if len(colors) == 1:
        unified = rules[0]
        for demo in demos:
            if not np.array_equal(unified.apply(demo.input, bg), demo.output):
                return None
        return unified
    return None


def _unify_periodic(rules: list[LocalRule], demos, bg) -> LocalRule | None:
    axes = set(r.params.get("axis") for r in rules)
    if len(axes) == 1:
        unified = rules[0]
        for demo in demos:
            if not np.array_equal(unified.apply(demo.input, bg), demo.output):
                return None
        return unified
    return None


def _unify_gravity(rules: list[LocalRule], demos, bg) -> LocalRule | None:
    dirs = set(r.params.get("direction") for r in rules)
    if len(dirs) == 1:
        unified = rules[0]
        for demo in demos:
            if not np.array_equal(unified.apply(demo.input, bg), demo.output):
                return None
        return unified
    return None


def _unify_mirror(rules: list[LocalRule], demos, bg) -> LocalRule | None:
    axes = set(r.params.get("axis") for r in rules)
    if len(axes) == 1:
        unified = rules[0]
        for demo in demos:
            if not np.array_equal(unified.apply(demo.input, bg), demo.output):
                return None
        return unified
    return None


# ---------------------------------------------------------------------------
# Diff-shape solver
# ---------------------------------------------------------------------------


def _solve_diff_shape(
    demos: tuple[DemoPair, ...],
    task_id: str,
) -> BackwardSolveResult:
    """Backward solve for diff-shape tasks.

    Strategy: decompose the output, find corresponding input support,
    induce per-region rules.
    """
    bg = detect_bg(demos[0].input)

    per_demo_rules: list[list[LocalRule]] = []
    for demo in demos:
        regions = decompose_output(demo.input, demo.output, bg)
        alignments = align_regions(demo.input, demo.output, regions, bg)

        demo_rules = []
        for alignment in alignments:
            rules = induce_local_rules(alignment, bg)
            if rules:
                demo_rules.append(rules[0])
        per_demo_rules.append(demo_rules)

    return BackwardSolveResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        shared_rule=None,
        per_demo_rules=per_demo_rules,
        train_diff=sum(d.output.size for d in demos),  # worst case
        details={"strategy": "diff_shape_not_yet_implemented"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_best_diff(demos, per_demo_rules, bg) -> int:
    """Compute the best total diff achievable with the induced rules."""
    total = 0
    for di, demo in enumerate(demos):
        if di < len(per_demo_rules) and per_demo_rules[di]:
            # Apply best rule
            rule = per_demo_rules[di][0]
            predicted = rule.apply(demo.input, bg)
            if predicted.shape == demo.output.shape:
                total += int(np.sum(predicted != demo.output))
            else:
                total += demo.output.size
        else:
            total += int(np.sum(demo.input != demo.output))
    return total


def _empty_result(task_id: str) -> BackwardSolveResult:
    return BackwardSolveResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        shared_rule=None,
        per_demo_rules=[],
        train_diff=0,
        details={"strategy": "empty"},
    )
