"""Bidirectional task decomposition from perception.

Analyzes what changed between input and output (via StateGraph + Delta)
to derive typed constraints on what a multi-step program must do. These
constraints narrow the search space by ruling out irrelevant ops at each
step position rather than enumerating blindly.

The decomposition produces a SubGoalPlan: an ordered list of sub-goals,
each with a type constraint (what type the intermediate must be) and an
op constraint (which ops could produce it). The search then only needs
to fill in the specific arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.graph.extract import extract, extract_with_delta
from aria.graph.signatures import compute_task_signatures
from aria.runtime.ops import OpSignature, all_ops
from aria.types import (
    DemoPair,
    Delta,
    Grid,
    GridContext,
    StateGraph,
    Type,
)


@dataclass(frozen=True)
class SubGoal:
    """A constraint on one step of the program."""

    name: str  # human-readable, e.g. "extract_objects", "change_dims"
    output_type: Type
    candidate_ops: tuple[str, ...]  # ops that could produce this sub-goal
    required: bool = True  # must appear vs. optional/heuristic


@dataclass(frozen=True)
class DecompPlan:
    """A decomposition of a task into constrained sub-goals."""

    sub_goals: tuple[SubGoal, ...]
    final_type: Type  # always GRID for ARC tasks
    evidence: tuple[str, ...]  # why this decomposition was chosen

    def candidate_op_names(self) -> frozenset[str]:
        """Union of all candidate ops across sub-goals."""
        return frozenset(
            op for sg in self.sub_goals for op in sg.candidate_ops
        )

    def ops_for_depth(self, depth: int, max_depth: int) -> frozenset[str]:
        """Preferred ops for a specific search depth.

        Maps the sub-goal sequence onto the search depth range. Early
        depths get early sub-goal ops, late depths get late sub-goal ops.
        If there are fewer sub-goals than depths, all ops are returned
        for un-mapped depths. This gives the search per-step guidance
        without being rigid.
        """
        if not self.sub_goals:
            return frozenset()

        n_goals = len(self.sub_goals)
        if n_goals == 1:
            return frozenset(self.sub_goals[0].candidate_ops)

        # Map depth (1-indexed) to sub-goal index
        # Early depths → early sub-goals, final depth → final sub-goal
        if depth >= max_depth:
            # Final depth: use the last sub-goal (should produce GRID)
            goal_idx = n_goals - 1
        else:
            # Distribute remaining sub-goals across earlier depths
            fraction = (depth - 1) / max(max_depth - 1, 1)
            goal_idx = min(int(fraction * n_goals), n_goals - 1)

        return frozenset(self.sub_goals[goal_idx].candidate_ops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_goals": [
                {
                    "name": sg.name,
                    "output_type": sg.output_type.name,
                    "candidate_ops": list(sg.candidate_ops),
                    "required": sg.required,
                }
                for sg in self.sub_goals
            ],
            "final_type": self.final_type.name,
            "evidence": list(self.evidence),
        }


def decompose_task(demos: tuple[DemoPair, ...]) -> DecompPlan:
    """Analyze demos to produce a sub-goal decomposition.

    Examines each demo's input→output delta and derives constraints that
    are consistent across all demos. Returns a plan that constrains
    what types of operations the program must contain.
    """
    all_ops_registry = all_ops()

    # Collect observations across all demos
    observations = [_observe_demo(demo) for demo in demos]

    # Merge observations: a property holds only if it's true for ALL demos
    merged = _merge_observations(observations)

    # Derive sub-goals from merged observations
    sub_goals: list[SubGoal] = []
    evidence: list[str] = []

    # 1. Dimension change → need a dim-producing step
    if merged["dims_change"]:
        dim_ops = _ops_producing(Type.GRID, all_ops_registry,
                                  requires_param_type=Type.DIMS)
        dim_ops += _ops_changing_dims(all_ops_registry)
        sub_goals.append(SubGoal(
            name="change_dims",
            output_type=Type.GRID,
            candidate_ops=tuple(sorted(set(dim_ops))),
        ))
        evidence.append("dims_change")

    # 2. Objects preserved but modified → need object extraction + transform
    if merged["objects_modified"] and not merged["objects_removed_all"]:
        extract_ops = _ops_producing(Type.OBJECT_SET, all_ops_registry)
        sub_goals.append(SubGoal(
            name="extract_objects",
            output_type=Type.OBJECT_SET,
            candidate_ops=tuple(sorted(extract_ops)),
        ))
        evidence.append("objects_modified")

    # 3. New objects added → need construction
    if merged["objects_added"]:
        construct_ops = _ops_producing(Type.REGION, all_ops_registry)
        construct_ops += _ops_producing(Type.OBJECT, all_ops_registry)
        sub_goals.append(SubGoal(
            name="construct_new",
            output_type=Type.OBJECT,
            candidate_ops=tuple(sorted(set(construct_ops))),
            required=False,
        ))
        evidence.append("objects_added")

    # 4. Color palette changed → need color mapping
    if merged["palette_changed"]:
        color_ops = _color_transform_ops(all_ops_registry)
        sub_goals.append(SubGoal(
            name="color_transform",
            output_type=Type.GRID,
            candidate_ops=tuple(sorted(color_ops)),
        ))
        evidence.append("palette_changed")

    # 5. Same dims, content changed → need grid composition
    #    Sharper: if mostly preserved + additive, the final step is overlay
    if merged["dims_same"] and merged["content_changed"]:
        if merged["additive_only"] and merged["mostly_preserved"]:
            # Output = input + new content overlaid. Tight constraint.
            overlay_ops = _overlay_ops(all_ops_registry)
            sub_goals.append(SubGoal(
                name="overlay_additive",
                output_type=Type.GRID,
                candidate_ops=tuple(sorted(overlay_ops)),
            ))
            evidence.append("additive_overlay")
        else:
            compose_ops = _grid_compose_ops(all_ops_registry)
            sub_goals.append(SubGoal(
                name="compose_grid",
                output_type=Type.GRID,
                candidate_ops=tuple(sorted(compose_ops)),
            ))
            evidence.append("same_dims_content_changed")

    # 6. Symmetry present in output but not input → symmetry operation
    if merged["symmetry_added"]:
        sym_ops = _symmetry_ops(all_ops_registry)
        sub_goals.append(SubGoal(
            name="apply_symmetry",
            output_type=Type.GRID,
            candidate_ops=tuple(sorted(sym_ops)),
            required=False,
        ))
        evidence.append("symmetry_added")

    # Always need a final GRID output
    if not sub_goals:
        # Fallback: no specific constraints derived → generic
        evidence.append("generic_fallback")

    return DecompPlan(
        sub_goals=tuple(sub_goals),
        final_type=Type.GRID,
        evidence=tuple(evidence),
    )


# ---------------------------------------------------------------------------
# Demo observation
# ---------------------------------------------------------------------------


@dataclass
class _DemoObs:
    dims_change: bool
    dims_same: bool
    objects_modified: bool
    objects_added: bool
    objects_removed_all: bool
    palette_changed: bool
    content_changed: bool
    symmetry_added: bool
    bg_preserved: bool
    input_palette: frozenset[int]
    output_palette: frozenset[int]
    mostly_preserved: bool  # >50% of pixels unchanged
    additive_only: bool  # objects added, none removed/modified


def _observe_demo(demo: DemoPair) -> _DemoObs:
    """Extract structural observations from one demo pair."""
    sg_in, sg_out, delta = extract_with_delta(demo.input, demo.output)

    dims_change = delta.dims_changed is not None
    objects_modified = len(delta.modified) > 0
    objects_added = len(delta.added) > 0
    objects_removed_all = (
        len(delta.removed) > 0
        and len(delta.removed) == len(sg_in.objects)
    )
    palette_changed = sg_in.context.palette != sg_out.context.palette
    content_changed = not np.array_equal(demo.input, demo.output)

    in_sym = sg_in.context.symmetry
    out_sym = sg_out.context.symmetry
    symmetry_added = bool(out_sym - in_sym)

    # Pixel-level preservation analysis
    if not dims_change:
        preserved_count = int(np.sum(demo.input == demo.output))
        total = demo.input.size
        mostly_preserved = preserved_count > total * 0.5
    else:
        mostly_preserved = False

    additive_only = (
        objects_added
        and not objects_modified
        and len(delta.removed) == 0
    )

    return _DemoObs(
        dims_change=dims_change,
        dims_same=not dims_change,
        objects_modified=objects_modified,
        objects_added=objects_added,
        objects_removed_all=objects_removed_all,
        palette_changed=palette_changed,
        content_changed=content_changed,
        symmetry_added=symmetry_added,
        bg_preserved=sg_in.context.bg_color == sg_out.context.bg_color,
        input_palette=sg_in.context.palette,
        output_palette=sg_out.context.palette,
        mostly_preserved=mostly_preserved,
        additive_only=additive_only,
    )


def _merge_observations(observations: list[_DemoObs]) -> dict[str, bool]:
    """Merge observations: a property holds only if true for ALL demos."""
    if not observations:
        return {
            "dims_change": False, "dims_same": True,
            "objects_modified": False, "objects_added": False,
            "objects_removed_all": False, "palette_changed": False,
            "content_changed": False, "symmetry_added": False,
            "mostly_preserved": False, "additive_only": False,
        }
    return {
        "dims_change": all(o.dims_change for o in observations),
        "dims_same": all(o.dims_same for o in observations),
        "objects_modified": all(o.objects_modified for o in observations),
        "objects_added": any(o.objects_added for o in observations),
        "objects_removed_all": all(o.objects_removed_all for o in observations),
        "palette_changed": any(o.palette_changed for o in observations),
        "content_changed": all(o.content_changed for o in observations),
        "symmetry_added": all(o.symmetry_added for o in observations),
        "mostly_preserved": all(o.mostly_preserved for o in observations),
        "additive_only": all(o.additive_only for o in observations),
    }


# ---------------------------------------------------------------------------
# Op classification by type constraints (all derived from registry)
# ---------------------------------------------------------------------------


def _ops_producing(
    return_type: Type,
    registry: dict[str, OpSignature],
) -> list[str]:
    """All ops that return the given type."""
    return [name for name, sig in registry.items() if sig.return_type == return_type]


def _ops_producing_from(
    return_type: Type,
    input_type: Type,
    registry: dict[str, OpSignature],
) -> list[str]:
    """Ops that take input_type and produce return_type."""
    result = []
    for name, sig in registry.items():
        if sig.return_type != return_type:
            continue
        if any(pt == input_type for _, pt in sig.params):
            result.append(name)
    return result


def _ops_producing(
    return_type: Type,
    registry: dict[str, OpSignature],
    requires_param_type: Type | None = None,
) -> list[str]:
    result = []
    for name, sig in registry.items():
        if sig.return_type != return_type:
            continue
        if requires_param_type is not None:
            if not any(pt == requires_param_type for _, pt in sig.params):
                continue
        result.append(name)
    return result


def _ops_changing_dims(registry: dict[str, OpSignature]) -> list[str]:
    """Ops known to change grid dimensions (GRID→GRID but output dims ≠ input dims)."""
    # These are ops where the output grid has different dimensions than input.
    # Derived from type: takes GRID, returns GRID, and also takes a
    # size/dims parameter or is a known tiling/stacking op.
    result = []
    for name, sig in registry.items():
        if sig.return_type != Type.GRID:
            continue
        param_types = {pt for _, pt in sig.params}
        # Takes dims/int params alongside grid → likely changes size
        if Type.GRID in param_types and (Type.DIMS in param_types or Type.INT in param_types):
            result.append(name)
        # Takes multiple grids → likely stacking/composition
        grid_params = sum(1 for _, pt in sig.params if pt == Type.GRID)
        if grid_params >= 2:
            result.append(name)
    return result


def _color_transform_ops(registry: dict[str, OpSignature]) -> list[str]:
    """Ops that transform colors."""
    result = []
    for name, sig in registry.items():
        if sig.return_type != Type.GRID:
            continue
        param_types = {pt for _, pt in sig.params}
        if Type.COLOR_MAP in param_types or Type.COLOR in param_types:
            result.append(name)
    return result


def _grid_compose_ops(registry: dict[str, OpSignature]) -> list[str]:
    """Ops that compose/combine grids or overlay objects."""
    result = []
    for name, sig in registry.items():
        if sig.return_type != Type.GRID:
            continue
        param_types = [pt for _, pt in sig.params]
        # Takes GRID + something else → composition
        if Type.GRID in param_types and len(param_types) >= 2:
            result.append(name)
        # Takes objects/regions → construction
        if any(pt in (Type.OBJECT, Type.OBJECT_SET, Type.REGION) for pt in param_types):
            result.append(name)
    return result


def _overlay_ops(registry: dict[str, OpSignature]) -> list[str]:
    """Ops that overlay/compose content onto a grid, preserving most of it.

    Tight constraint for additive tasks: the final step must be something
    that takes a GRID (the base) and adds content from another source
    (GRID, OBJECT, REGION, OBJECT_SET).
    """
    result = []
    for name, sig in registry.items():
        if sig.return_type != Type.GRID:
            continue
        param_types = [pt for _, pt in sig.params]
        # (GRID, GRID) → GRID composition ops
        if len(param_types) == 2 and param_types[0] == Type.GRID and param_types[1] == Type.GRID:
            result.append(name)
        # (GRID, OBJECT/REGION) → GRID or (OBJECT/REGION, GRID) → GRID
        if any(pt in (Type.OBJECT, Type.REGION, Type.OBJECT_SET) for pt in param_types):
            if Type.GRID in param_types:
                result.append(name)
        # (GRID, COLOR) → GRID (fill operations)
        if Type.GRID in param_types and Type.COLOR in param_types:
            result.append(name)
    return result


def _symmetry_ops(registry: dict[str, OpSignature]) -> list[str]:
    """Ops that create or enforce symmetry."""
    result = []
    for name, sig in registry.items():
        if sig.return_type != Type.GRID:
            continue
        param_types = {pt for _, pt in sig.params}
        if Type.AXIS in param_types:
            result.append(name)
    # Also include explicit symmetry ops by naming convention
    for name in registry:
        if "symmetry" in name or "reflect" in name or "rotate" in name:
            if name not in result and registry[name].return_type == Type.GRID:
                result.append(name)
    return result
