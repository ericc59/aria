"""Verifier-guided program repair for near-miss ScenePrograms.

Given a SceneProgram that almost solves a task (correct dims, most pixels
correct), this module applies bounded local edits to SceneStep params
guided by residual diagnostics.

Architecture:
  diagnose_residual()  — characterize what's wrong
  propose_repairs()    — generate prioritized edit candidates
  apply_repair()       — clone program with one param change or step insert
  repair_search()      — bounded loop: diagnose → propose → verify
  repair_near_misses() — entry point: score candidates, repair top-k

No task-id logic. No benchmark hacks. Exact verification remains final arbiter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.core.scene_executor import (
    SceneExecutionError,
    _MAP_PROPERTY_EXTRACTORS,
    _SELECTOR_PREDICATES,
    execute_scene_program,
)
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairAction:
    """One local edit to a SceneProgram step param.

    Two modes:
    - param edit: step_index >= 0, param_name is set → change one param
    - step insert: step_index = insertion point, insert_step is set → insert a new step
    """

    step_index: int
    param_name: str
    old_value: Any
    new_value: Any
    priority: float  # 0-1, higher = try first
    reason: str
    insert_step: SceneStep | None = None  # if set, insert this step at step_index


@dataclass(frozen=True)
class RepairDiagnostic:
    """Characterization of a near-miss residual."""

    pixel_diff_total: int
    pixel_diff_per_demo: tuple[int, ...]
    pixel_accuracy: float  # fraction correct across all demos
    dims_match: tuple[bool, ...]
    all_dims_match: bool
    diff_colors_expected: frozenset[int]  # in expected but not result
    diff_colors_spurious: frozenset[int]  # in result but not expected
    transform_mismatch: bool  # content looks rotated/flipped
    selector_mismatch: bool  # diff covers entire selected entity
    color_map_mismatch: bool  # any color difference in diff region
    execution_errors: int  # demos that errored


@dataclass(frozen=True)
class RepairEditTrace:
    """Trace of one repair attempt."""

    action: RepairAction
    pixel_diff_after: int
    improved: bool
    exact: bool


@dataclass(frozen=True)
class RepairRound:
    """Trace of one repair round."""

    round_idx: int
    edits_tried: tuple[RepairEditTrace, ...]
    best_pixel_diff_before: int
    best_pixel_diff_after: int


@dataclass(frozen=True)
class RepairTrace:
    """Full trace of a repair attempt on one candidate."""

    original_program: SceneProgram
    original_diagnostic: RepairDiagnostic
    rounds: tuple[RepairRound, ...]
    solved: bool
    final_program: SceneProgram | None
    total_verify_calls: int


@dataclass(frozen=True)
class RepairResult:
    """Result of repair_near_misses() across multiple candidates."""

    solved: bool
    winning_program: SceneProgram | None
    candidates_scored: int
    near_misses_found: int
    candidates_repaired: int
    traces: tuple[RepairTrace, ...]
    total_verify_calls: int


# ---------------------------------------------------------------------------
# Scoring / diagnosis
# ---------------------------------------------------------------------------

# All selector predicates and transforms, used by repair generators.
ALL_PREDICATES = tuple(sorted(_SELECTOR_PREDICATES.keys()))
ALL_TRANSFORMS = (
    "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose",
)
ALL_ENTITY_KINDS = ("object", "panel", "interior_region", "boundary")
ALL_BOOLEAN_OPS = ("overlay", "and", "xor")
ALL_RECOLOR_SCOPES = (
    "global", "objects", "frame_interior", "partition_cells", "object_bboxes",
)
ALL_MAP_PROPERTIES = tuple(sorted(_MAP_PROPERTY_EXTRACTORS.keys()))
ALL_FOR_EACH_RULES = (
    "fill_bbox_holes", "fill_enclosed_bbox", "recolor",
    "fill_bg_with_dominant",
)


def score_scene_candidate(
    program: SceneProgram,
    demos: tuple[DemoPair, ...],
) -> tuple[list[Grid | None], RepairDiagnostic]:
    """Execute program on all demos and compute residual diagnostic.

    Returns (per_demo_results, diagnostic).
    """
    results: list[Grid | None] = []
    pixel_diffs: list[int] = []
    dims_match: list[bool] = []
    total_pixels = 0
    total_correct = 0
    exec_errors = 0
    all_expected_colors: set[int] = set()
    all_result_colors: set[int] = set()

    for demo in demos:
        try:
            result = execute_scene_program(program, demo.input)
        except Exception:
            results.append(None)
            pixel_diffs.append(demo.output.size)
            dims_match.append(False)
            total_pixels += demo.output.size
            exec_errors += 1
            continue

        results.append(result)
        dm = result.shape == demo.output.shape
        dims_match.append(dm)

        if dm:
            diff = int(np.sum(result != demo.output))
            pixel_diffs.append(diff)
            total_pixels += demo.output.size
            total_correct += demo.output.size - diff

            # Color analysis on diff regions
            diff_mask = result != demo.output
            if np.any(diff_mask):
                all_expected_colors.update(int(v) for v in demo.output[diff_mask])
                all_result_colors.update(int(v) for v in result[diff_mask])
        else:
            pixel_diffs.append(demo.output.size)
            total_pixels += demo.output.size

    accuracy = total_correct / max(total_pixels, 1)
    all_dims = all(dims_match)

    # Detect transform mismatch: same shape, same pixel histogram,
    # but different arrangement
    transform_mismatch = False
    if all_dims:
        transform_mismatch = _detect_transform_mismatch(results, demos)

    # Detect selector mismatch: high-fraction diff concentrated
    # in one entity-sized region
    selector_mismatch = False
    if all_dims and accuracy < 0.5:
        selector_mismatch = True

    # Detect color map mismatch — any diff in color values (even swaps)
    color_map_mismatch = bool(all_expected_colors or all_result_colors)

    diag = RepairDiagnostic(
        pixel_diff_total=sum(pixel_diffs),
        pixel_diff_per_demo=tuple(pixel_diffs),
        pixel_accuracy=accuracy,
        dims_match=tuple(dims_match),
        all_dims_match=all_dims,
        diff_colors_expected=frozenset(all_expected_colors - all_result_colors),
        diff_colors_spurious=frozenset(all_result_colors - all_expected_colors),
        transform_mismatch=transform_mismatch,
        selector_mismatch=selector_mismatch,
        color_map_mismatch=color_map_mismatch,
        execution_errors=exec_errors,
    )
    return results, diag


def _detect_transform_mismatch(
    results: list[Grid | None],
    demos: tuple[DemoPair, ...],
) -> bool:
    """Check if result looks like a rotated/flipped version of expected."""
    for result, demo in zip(results, demos):
        if result is None or result.shape != demo.output.shape:
            continue
        if np.array_equal(result, demo.output):
            continue
        # Same pixel histogram but different arrangement?
        r_hist = np.bincount(result.ravel(), minlength=10)
        e_hist = np.bincount(demo.output.ravel(), minlength=10)
        if np.array_equal(r_hist, e_hist):
            return True
    return False


# ---------------------------------------------------------------------------
# Repair generators
# ---------------------------------------------------------------------------


def _find_steps_by_op(
    program: SceneProgram, op: StepOp,
) -> list[tuple[int, SceneStep]]:
    """Find all steps with a given op."""
    return [
        (i, step) for i, step in enumerate(program.steps) if step.op is op
    ]


def _gen_insert_recolor_step(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Insert a RECOLOR_OBJECT step before RENDER_SCENE when color mismatch detected.

    Handles:
    - single from→to recolor (derived per-demo, checked for cross-demo consistency)
    - multi-pair recolor / color swaps (derived from per-demo color maps)
    - uses all demos to find consistent mappings, not just demo 0
    """
    if not diag.color_map_mismatch:
        return []

    actions = []
    # Find the RENDER_SCENE step to insert before
    render_indices = [
        i for i, step in enumerate(program.steps)
        if step.op is StepOp.RENDER_SCENE
    ]
    if not render_indices:
        return []

    insert_idx = render_indices[0]

    # Collect per-demo color maps
    per_demo_maps: list[dict[int, int] | None] = []
    per_demo_from_colors: list[set[int]] = []
    per_demo_to_colors: list[set[int]] = []

    for demo in demos:
        try:
            result = execute_scene_program(program, demo.input)
        except Exception:
            per_demo_maps.append(None)
            per_demo_from_colors.append(set())
            per_demo_to_colors.append(set())
            continue

        if result.shape != demo.output.shape:
            per_demo_maps.append(None)
            per_demo_from_colors.append(set())
            per_demo_to_colors.append(set())
            continue

        diff_mask = result != demo.output
        if not np.any(diff_mask):
            per_demo_maps.append({})
            per_demo_from_colors.append(set())
            per_demo_to_colors.append(set())
            continue

        cmap = _infer_color_map(result, demo.output, diff_mask)
        per_demo_maps.append(cmap)
        per_demo_from_colors.append(set(int(v) for v in result[diff_mask]))
        per_demo_to_colors.append(set(int(v) for v in demo.output[diff_mask]))

    # Strategy 1: Find a consistent color map across ALL demos
    consistent_map = _find_consistent_cross_demo_map(per_demo_maps)
    if consistent_map:
        pairs = [(fc, tc) for fc, tc in consistent_map.items() if fc != tc]
        if pairs and len(pairs) <= 6:
            recolor_step = SceneStep(
                op=StepOp.RECOLOR_OBJECT,
                params={"color_pairs": pairs, "scope": "global"},
            )
            actions.append(RepairAction(
                step_index=insert_idx,
                param_name="__insert__",
                old_value=None,
                new_value=None,
                priority=0.95,
                reason=f"insert consistent multi-recolor {pairs}",
                insert_step=recolor_step,
            ))

    # Strategy 2: Per-demo single-pair recolors from demo 0
    if per_demo_maps and per_demo_maps[0] is not None:
        demo0_map = per_demo_maps[0]
        if demo0_map:
            for fc, tc in sorted(demo0_map.items()):
                if fc == tc:
                    continue
                recolor_step = SceneStep(
                    op=StepOp.RECOLOR_OBJECT,
                    params={"from_color": fc, "to_color": tc},
                )
                actions.append(RepairAction(
                    step_index=insert_idx,
                    param_name="__insert__",
                    old_value=None,
                    new_value=None,
                    priority=0.85,
                    reason=f"insert recolor {fc} → {tc}",
                    insert_step=recolor_step,
                ))

    # Strategy 3: Try all from→to pairs across all demos
    all_from: set[int] = set()
    all_to: set[int] = set()
    for fc_set, tc_set in zip(per_demo_from_colors, per_demo_to_colors):
        all_from.update(fc_set)
        all_to.update(tc_set)

    if len(all_from) <= 3 and len(all_to) <= 3:
        for fc in sorted(all_from):
            for tc in sorted(all_to):
                if fc == tc:
                    continue
                # Skip if already covered by strategies above
                if any(a.reason == f"insert recolor {fc} → {tc}" for a in actions):
                    continue
                recolor_step = SceneStep(
                    op=StepOp.RECOLOR_OBJECT,
                    params={"from_color": fc, "to_color": tc},
                )
                actions.append(RepairAction(
                    step_index=insert_idx,
                    param_name="__insert__",
                    old_value=None,
                    new_value=None,
                    priority=0.7,
                    reason=f"insert recolor {fc} → {tc}",
                    insert_step=recolor_step,
                ))

    return actions


def _find_consistent_cross_demo_map(
    per_demo_maps: list[dict[int, int] | None],
) -> dict[int, int] | None:
    """Find a color map consistent across all demos that have diffs.

    Returns None if demos disagree on any mapping.
    """
    merged: dict[int, int] = {}
    for cmap in per_demo_maps:
        if cmap is None:
            continue
        if not cmap:
            continue  # demo was already correct
        for fc, tc in cmap.items():
            if fc in merged:
                if merged[fc] != tc:
                    return None  # inconsistent
            else:
                merged[fc] = tc
    return merged if merged else None


def _infer_color_map(
    result: Grid,
    expected: Grid,
    diff_mask: np.ndarray,
) -> dict[int, int] | None:
    """Infer a color map from result to expected in the diff region."""
    cmap: dict[int, int] = {}
    for r_val, e_val in zip(result[diff_mask].ravel(), expected[diff_mask].ravel()):
        fc, tc = int(r_val), int(e_val)
        if fc in cmap:
            if cmap[fc] != tc:
                return None  # inconsistent
        else:
            cmap[fc] = tc
    return cmap


def _gen_swap_selector_predicate(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative selector predicates."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.SELECT_ENTITY):
        current = step.params.get("predicate", "largest_bbox_area")
        priority = 0.8 if diag.selector_mismatch else 0.3
        for pred in ALL_PREDICATES:
            if pred == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="predicate",
                old_value=current,
                new_value=pred,
                priority=priority,
                reason=f"swap selector predicate {current} → {pred}",
            ))
    return actions


def _gen_change_selector_rank(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative selector ranks."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.SELECT_ENTITY):
        current = int(step.params.get("rank", 0))
        priority = 0.7 if diag.selector_mismatch else 0.3
        for rank in range(4):
            if rank == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="rank",
                old_value=current,
                new_value=rank,
                priority=priority,
                reason=f"change selector rank {current} → {rank}",
            ))
    return actions


def _gen_change_selector_kind(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative entity kinds."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.SELECT_ENTITY):
        current = step.params.get("kind")
        priority = 0.5 if diag.selector_mismatch else 0.2
        for kind in ALL_ENTITY_KINDS:
            if kind == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="kind",
                old_value=current,
                new_value=kind,
                priority=priority,
                reason=f"change selector kind {current} → {kind}",
            ))
    return actions


def _gen_swap_transform(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative transforms."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.CANONICALIZE_OBJECT):
        current = step.params.get("transform")
        priority = 0.9 if diag.transform_mismatch else 0.3
        for transform in ALL_TRANSFORMS:
            if transform == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="transform",
                old_value=current,
                new_value=transform,
                priority=priority,
                reason=f"swap transform {current} → {transform}",
            ))
    return actions


def _gen_swap_boolean_op(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative boolean combine operations."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.BOOLEAN_COMBINE_PANELS):
        current = step.params.get("operation", "overlay")
        for op in ALL_BOOLEAN_OPS:
            if op == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="operation",
                old_value=current,
                new_value=op,
                priority=0.6,
                reason=f"swap boolean op {current} → {op}",
            ))
    return actions


def _gen_change_fill_color(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative fill colors, prioritizing colors from expected output."""
    actions = []

    # Collect colors present in expected outputs
    expected_colors: set[int] = set()
    for demo in demos:
        expected_colors.update(int(v) for v in np.unique(demo.output))

    target_ops = [StepOp.FILL_ENCLOSED_REGIONS, StepOp.FOR_EACH_ENTITY]
    for op in target_ops:
        for idx, step in _find_steps_by_op(program, op):
            current = step.params.get("fill_color")
            if current is None:
                continue
            # Prioritize colors from expected output
            for color in sorted(expected_colors):
                if color == current:
                    continue
                in_expected = color in diag.diff_colors_expected
                actions.append(RepairAction(
                    step_index=idx,
                    param_name="fill_color",
                    old_value=current,
                    new_value=color,
                    priority=0.8 if in_expected else 0.4,
                    reason=f"change fill color {current} → {color}",
                ))
    return actions


def _gen_swap_recolor_scope(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative recolor scopes."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.RECOLOR_OBJECT):
        current = step.params.get("scope", "global")
        priority = 0.7 if diag.color_map_mismatch else 0.3
        for scope in ALL_RECOLOR_SCOPES:
            if scope == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="scope",
                old_value=current,
                new_value=scope,
                priority=priority,
                reason=f"swap recolor scope {current} → {scope}",
            ))
    return actions


def _gen_adjust_color_map(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try adjusting recolor from/to colors based on residual."""
    actions = []
    if not diag.color_map_mismatch:
        return actions

    expected_colors: set[int] = set()
    for demo in demos:
        expected_colors.update(int(v) for v in np.unique(demo.output))

    for idx, step in _find_steps_by_op(program, StepOp.RECOLOR_OBJECT):
        current_to = step.params.get("to_color")
        if current_to is None:
            continue
        for color in sorted(expected_colors):
            if color == current_to:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="to_color",
                old_value=current_to,
                new_value=color,
                priority=0.7,
                reason=f"adjust recolor to_color {current_to} → {color}",
            ))
        current_from = step.params.get("from_color")
        if current_from is not None:
            for color in range(10):
                if color == current_from:
                    continue
                actions.append(RepairAction(
                    step_index=idx,
                    param_name="from_color",
                    old_value=current_from,
                    new_value=color,
                    priority=0.5,
                    reason=f"adjust recolor from_color {current_from} → {color}",
                ))
    return actions


def _gen_swap_map_property(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative map-over-entities property extractors."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.MAP_OVER_ENTITIES):
        current = step.params.get("property")
        for prop in ALL_MAP_PROPERTIES:
            if prop == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="property",
                old_value=current,
                new_value=prop,
                priority=0.5,
                reason=f"swap map property {current} → {prop}",
            ))
    return actions


def _gen_change_connectivity(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Toggle connectivity 4 ↔ 8."""
    actions = []
    for idx, step in enumerate(program.steps):
        current = step.params.get("connectivity")
        if current is None:
            continue
        new_val = 8 if current == 4 else 4
        actions.append(RepairAction(
            step_index=idx,
            param_name="connectivity",
            old_value=current,
            new_value=new_val,
            priority=0.4,
            reason=f"change connectivity {current} → {new_val}",
        ))
    return actions


def _gen_swap_for_each_rule(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Try alternative for-each rules."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.FOR_EACH_ENTITY):
        current = step.params.get("rule")
        if current is None:
            continue
        for rule in ALL_FOR_EACH_RULES:
            if rule == current:
                continue
            actions.append(RepairAction(
                step_index=idx,
                param_name="rule",
                old_value=current,
                new_value=rule,
                priority=0.5,
                reason=f"swap for-each rule {current} → {rule}",
            ))
    return actions


def _gen_toggle_require_non_bg(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
) -> list[RepairAction]:
    """Toggle require_non_bg on SELECT_ENTITY steps."""
    actions = []
    for idx, step in _find_steps_by_op(program, StepOp.SELECT_ENTITY):
        current = bool(step.params.get("require_non_bg", False))
        actions.append(RepairAction(
            step_index=idx,
            param_name="require_non_bg",
            old_value=current,
            new_value=not current,
            priority=0.3,
            reason=f"toggle require_non_bg {current} → {not current}",
        ))
    return actions


# All generators, in priority order.
_REPAIR_GENERATORS = [
    _gen_insert_recolor_step,  # highest value: many near-misses need color fixup
    _gen_swap_transform,
    _gen_swap_selector_predicate,
    _gen_change_selector_rank,
    _gen_swap_boolean_op,
    _gen_change_fill_color,
    _gen_swap_recolor_scope,
    _gen_adjust_color_map,
    _gen_swap_map_property,
    _gen_change_selector_kind,
    _gen_change_connectivity,
    _gen_swap_for_each_rule,
    _gen_toggle_require_non_bg,
]


# ---------------------------------------------------------------------------
# Propose & apply
# ---------------------------------------------------------------------------


def propose_repairs(
    program: SceneProgram,
    diag: RepairDiagnostic,
    demos: tuple[DemoPair, ...],
    *,
    max_actions: int = 30,
) -> list[RepairAction]:
    """Generate and prioritize repair actions based on residual diagnostic."""
    all_actions: list[RepairAction] = []
    for gen in _REPAIR_GENERATORS:
        all_actions.extend(gen(program, diag, demos))

    # Sort by priority descending, then by reason for stability
    all_actions.sort(key=lambda a: (-a.priority, a.reason))
    return all_actions[:max_actions]


def apply_repair(
    program: SceneProgram,
    action: RepairAction,
) -> SceneProgram:
    """Clone a SceneProgram with one edit applied.

    Two modes:
    - param edit: change one param on an existing step
    - step insert: insert a new step at the given index
    """
    steps = list(program.steps)

    if action.insert_step is not None:
        # Insert mode: add a new step at the index
        steps.insert(action.step_index, action.insert_step)
        return SceneProgram(steps=tuple(steps))

    # Param edit mode
    old_step = steps[action.step_index]
    new_params = dict(old_step.params)
    new_params[action.param_name] = action.new_value

    steps[action.step_index] = SceneStep(
        op=old_step.op,
        inputs=old_step.inputs,
        params=new_params,
        output_id=old_step.output_id,
    )
    return SceneProgram(steps=tuple(steps))


# ---------------------------------------------------------------------------
# Bounded repair search
# ---------------------------------------------------------------------------


def repair_search(
    program: SceneProgram,
    demos: tuple[DemoPair, ...],
    *,
    max_rounds: int = 2,
    max_edits_per_round: int = 12,
    near_miss_threshold: float = 0.85,
) -> RepairTrace:
    """Bounded repair loop on one near-miss candidate.

    1. Diagnose residual
    2. Propose repairs sorted by priority
    3. Apply each, verify
    4. Keep best improvement, repeat (up to max_rounds)
    """
    _, diag = score_scene_candidate(program, demos)
    verify_calls = 0

    if not diag.all_dims_match or diag.pixel_accuracy < near_miss_threshold:
        return RepairTrace(
            original_program=program,
            original_diagnostic=diag,
            rounds=(),
            solved=False,
            final_program=None,
            total_verify_calls=0,
        )

    current_prog = program
    current_diff = diag.pixel_diff_total
    rounds: list[RepairRound] = []

    for round_idx in range(max_rounds):
        # Re-diagnose if this isn't the first round
        if round_idx > 0:
            _, diag = score_scene_candidate(current_prog, demos)
            current_diff = diag.pixel_diff_total

        actions = propose_repairs(current_prog, diag, demos, max_actions=max_edits_per_round)
        if not actions:
            break

        edit_traces: list[RepairEditTrace] = []
        best_diff = current_diff
        best_prog = current_prog
        round_start_diff = current_diff

        for action in actions:
            candidate = apply_repair(current_prog, action)
            verify_calls += 1

            # Quick verify
            cand_diff = _total_pixel_diff(candidate, demos)
            exact = cand_diff == 0
            improved = cand_diff < best_diff

            edit_traces.append(RepairEditTrace(
                action=action,
                pixel_diff_after=cand_diff,
                improved=improved,
                exact=exact,
            ))

            if exact:
                rounds.append(RepairRound(
                    round_idx=round_idx,
                    edits_tried=tuple(edit_traces),
                    best_pixel_diff_before=round_start_diff,
                    best_pixel_diff_after=0,
                ))
                return RepairTrace(
                    original_program=program,
                    original_diagnostic=diag,
                    rounds=tuple(rounds),
                    solved=True,
                    final_program=candidate,
                    total_verify_calls=verify_calls,
                )

            if improved:
                best_diff = cand_diff
                best_prog = candidate

        rounds.append(RepairRound(
            round_idx=round_idx,
            edits_tried=tuple(edit_traces),
            best_pixel_diff_before=round_start_diff,
            best_pixel_diff_after=best_diff,
        ))

        # Only continue if we improved
        if best_diff >= round_start_diff:
            break
        current_prog = best_prog
        current_diff = best_diff

    # Check if final best is exact (in case it wasn't caught above)
    solved = current_diff == 0 and current_prog is not program
    return RepairTrace(
        original_program=program,
        original_diagnostic=diag,
        rounds=tuple(rounds),
        solved=solved,
        final_program=current_prog if solved else None,
        total_verify_calls=verify_calls,
    )


def _total_pixel_diff(
    program: SceneProgram,
    demos: tuple[DemoPair, ...],
) -> int:
    """Compute total pixel diff across all demos. Returns large number on error."""
    total = 0
    for demo in demos:
        try:
            result = execute_scene_program(program, demo.input)
        except Exception:
            total += demo.output.size
            continue
        if result.shape != demo.output.shape:
            total += demo.output.size
        else:
            total += int(np.sum(result != demo.output))
    return total


# ---------------------------------------------------------------------------
# Entry point: repair near-misses from a candidate set
# ---------------------------------------------------------------------------


def repair_near_misses(
    candidates: list[SceneProgram] | tuple[SceneProgram, ...],
    demos: tuple[DemoPair, ...],
    *,
    max_candidates: int = 3,
    max_rounds: int = 2,
    max_edits_per_round: int = 12,
    near_miss_threshold: float = 0.85,
) -> RepairResult:
    """Score candidates, identify near-misses, attempt bounded repair.

    Returns RepairResult. Check .solved and .winning_program.
    """
    if not candidates or not demos:
        return RepairResult(
            solved=False,
            winning_program=None,
            candidates_scored=0,
            near_misses_found=0,
            candidates_repaired=0,
            traces=(),
            total_verify_calls=0,
        )

    # Score all candidates
    scored: list[tuple[SceneProgram, RepairDiagnostic]] = []
    for prog in candidates:
        _, diag = score_scene_candidate(prog, demos)
        scored.append((prog, diag))

    # Filter to near-misses: dims correct on all demos, accuracy above threshold
    near_misses = [
        (prog, diag) for prog, diag in scored
        if diag.all_dims_match
        and diag.pixel_accuracy >= near_miss_threshold
        and diag.execution_errors == 0
    ]

    # Sort by pixel accuracy descending (best first)
    near_misses.sort(key=lambda x: -x[1].pixel_accuracy)
    near_misses = near_misses[:max_candidates]

    traces: list[RepairTrace] = []
    total_verify = 0

    for prog, _ in near_misses:
        trace = repair_search(
            prog, demos,
            max_rounds=max_rounds,
            max_edits_per_round=max_edits_per_round,
            near_miss_threshold=near_miss_threshold,
        )
        traces.append(trace)
        total_verify += trace.total_verify_calls

        if trace.solved:
            return RepairResult(
                solved=True,
                winning_program=trace.final_program,
                candidates_scored=len(candidates),
                near_misses_found=len(near_misses),
                candidates_repaired=len(traces),
                traces=tuple(traces),
                total_verify_calls=total_verify,
            )

    return RepairResult(
        solved=False,
        winning_program=None,
        candidates_scored=len(candidates),
        near_misses_found=len(near_misses),
        candidates_repaired=len(traces),
        traces=tuple(traces),
        total_verify_calls=total_verify,
    )
