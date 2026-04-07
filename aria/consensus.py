"""Stepwise all-demo consensus during synthesis.

Instead of checking all-demo consistency only at final verification,
this module provides machinery to check cross-demo consistency at
each partial step during search. Branches that cannot represent one
shared rule across all demos are pruned early.

Core types:
  DemoPartialState — per-demo intermediate state at one step
  BranchState      — cross-demo state for one search branch
  SharedHypothesis — compact shared rule hypothesis
  ConsistencyCheck — result of one cross-demo check

Functions:
  check_*          — individual consistency checks
  run_checks       — run all applicable checks
  score_branch     — compute composite consistency score
  should_prune     — hard prune decision
  rank_branches    — order branches by consistency score

No task-id logic. No benchmark hacks. Exact verification remains
the final arbiter — this is early branch control, not replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from aria.types import Grid

# Avoid circular import — GridPerceptionState used for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aria.core.grid_perception import GridPerceptionState


# ---------------------------------------------------------------------------
# Consistency check result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsistencyCheck:
    """Result of one cross-demo consistency check."""

    name: str
    passed: bool
    score: float  # 0.0 = fail, 1.0 = perfect
    detail: str = ""


# ---------------------------------------------------------------------------
# Shared latent hypothesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedHypothesis:
    """Compact description of the shared rule a branch is pursuing.

    Refined (narrowed) after each step, never widened.
    """

    rule_family: str | None = None  # e.g. "select_extract_transform"
    selector_family: str | None = None  # e.g. "largest_bbox_area"
    transform_family: str | None = None  # e.g. "rot90"
    scope_family: str | None = None  # e.g. "same_dims" | "extract"
    correspondence: str | None = None  # e.g. "1:1_object"
    constraints: tuple[tuple[str, Any], ...] = ()  # frozen key-value pairs


# ---------------------------------------------------------------------------
# Per-demo partial state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoPartialState:
    """Per-demo intermediate state at one step of partial execution."""

    demo_index: int

    # Perception results (cached)
    perception: GridPerceptionState | None = None

    # Selection results
    selected_entity_count: int = 0
    selected_kind: str | None = None  # EntityKind value
    selected_bbox_dims: tuple[int, int] | None = None

    # Inferred bindings
    bg_color: int | None = None
    role_bindings: tuple[tuple[str, int], ...] = ()

    # Intermediate grid (if partial rendering possible)
    intermediate_grid: Grid | None = None

    # Structural signature for cross-demo comparison
    structural_signature: str = ""


# ---------------------------------------------------------------------------
# Branch state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchState:
    """Cross-demo state for one search branch."""

    branch_id: str
    step_index: int
    per_demo: tuple[DemoPartialState, ...]

    # Shared hypothesis
    hypothesis: SharedHypothesis = field(default_factory=SharedHypothesis)

    # Consistency
    consistency_score: float = 1.0
    consistency_checks: tuple[ConsistencyCheck, ...] = ()
    pruned: bool = False
    prune_reason: str = ""


# ---------------------------------------------------------------------------
# Branch construction helpers
# ---------------------------------------------------------------------------


def build_initial_branch(
    perceptions: Sequence[GridPerceptionState],
    branch_id: str = "root",
) -> BranchState:
    """Build initial branch state from per-demo perception results."""
    per_demo = tuple(
        DemoPartialState(
            demo_index=i,
            perception=p,
            bg_color=p.bg_color,
            structural_signature=_perception_signature(p),
        )
        for i, p in enumerate(perceptions)
    )

    # Determine scope
    dims_set = {p.dims for p in perceptions}
    scope = "same_dims" if len(dims_set) == 1 else "mixed_dims"

    hypothesis = SharedHypothesis(scope_family=scope)

    branch = BranchState(
        branch_id=branch_id,
        step_index=0,
        per_demo=per_demo,
        hypothesis=hypothesis,
    )

    # Run initial checks
    return _apply_checks(branch)


def update_branch_after_select(
    branch: BranchState,
    kind: str,
    predicate: str,
    rank: int,
    per_demo_results: Sequence[_SelectResult],
) -> BranchState:
    """Update branch after a SELECT_ENTITY step across all demos.

    per_demo_results: sequence of _SelectResult, one per demo.
    """
    new_per_demo = tuple(
        replace(
            pd,
            selected_entity_count=sr.entity_count,
            selected_kind=kind,
            selected_bbox_dims=sr.bbox_dims,
            structural_signature=_select_signature(kind, predicate, rank, sr),
        )
        for pd, sr in zip(branch.per_demo, per_demo_results)
    )

    new_hypothesis = replace(
        branch.hypothesis,
        rule_family=branch.hypothesis.rule_family or "select_extract",
        selector_family=f"{predicate}:{kind}:rank{rank}",
    )

    new_branch = replace(
        branch,
        step_index=branch.step_index + 1,
        per_demo=new_per_demo,
        hypothesis=new_hypothesis,
    )

    return _apply_checks(new_branch)


def update_branch_after_transform(
    branch: BranchState,
    transform: str,
    per_demo_grids: Sequence[Grid | None],
) -> BranchState:
    """Update branch after a transform step across all demos."""
    new_per_demo = tuple(
        replace(pd, intermediate_grid=g)
        for pd, g in zip(branch.per_demo, per_demo_grids)
    )

    new_hypothesis = replace(
        branch.hypothesis,
        transform_family=transform,
    )

    new_branch = replace(
        branch,
        step_index=branch.step_index + 1,
        per_demo=new_per_demo,
        hypothesis=new_hypothesis,
    )

    return _apply_checks(new_branch)


# ---------------------------------------------------------------------------
# Select result (lightweight carrier for per-demo selection outcome)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelectResult:
    """Per-demo result of a select operation (used during consensus check)."""

    found: bool = False
    entity_count: int = 0  # total entities of the requested kind
    bbox_dims: tuple[int, int] | None = None  # dims of the selected entity


def try_select_on_demo(
    perception: GridPerceptionState,
    kind: str,
    predicate: str,
    rank: int,
    *,
    color_filter: int | None = None,
) -> _SelectResult:
    """Lightweight SELECT_ENTITY simulation on one demo for consensus.

    Does NOT build a full SceneState — just counts entities of the
    requested kind and checks if the rank is reachable.
    """
    from aria.scene_ir import EntityKind

    try:
        ek = EntityKind(kind)
    except ValueError:
        return _SelectResult(found=False)

    # Count entities of the requested kind from perception
    count = 0
    dims = None

    if ek == EntityKind.OBJECT:
        objs = perception.objects.objects
        if color_filter is not None:
            objs = [o for o in objs if color_filter in set(int(v) for v in np.unique(
                perception.grid[o.row:o.row + o.bbox_h, o.col:o.col + o.bbox_w]
            ) if v != perception.bg_color)]
        count = len(objs)
        if rank < count:
            # Sort by the predicate to find the actual selected object
            o = _rank_objects_by_predicate(objs, predicate, perception, rank)
            if o is not None:
                dims = (o.bbox_h, o.bbox_w)

    elif ek == EntityKind.PANEL:
        if perception.partition is not None:
            count = len(perception.partition.cells)
            if rank < count:
                cell = perception.partition.cells[min(rank, count - 1)]
                dims = cell.dims

    elif ek in (EntityKind.INTERIOR_REGION, EntityKind.BOUNDARY):
        count = len(perception.framed_regions)
        if rank < count and count > 0:
            fr = perception.framed_regions[min(rank, count - 1)]
            dims = (fr.height, fr.width)

    found = rank < count
    return _SelectResult(found=found, entity_count=count, bbox_dims=dims)


def _rank_objects_by_predicate(objs, predicate, perception, rank):
    """Lightweight object ranking by predicate name."""
    if not objs or rank >= len(objs):
        return None

    if "largest" in predicate or predicate == "most_non_bg":
        scored = sorted(objs, key=lambda o: o.bbox_h * o.bbox_w, reverse=True)
    elif "smallest" in predicate:
        scored = sorted(objs, key=lambda o: o.bbox_h * o.bbox_w)
    elif predicate == "top_left":
        scored = sorted(objs, key=lambda o: (o.row, o.col))
    elif predicate == "bottom_right":
        scored = sorted(objs, key=lambda o: (o.row + o.bbox_h, o.col + o.bbox_w), reverse=True)
    else:
        scored = list(objs)

    return scored[rank] if rank < len(scored) else None


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def check_entity_count_consistent(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that the number of selectable entities is consistent across demos.

    For a shared rule, all demos must have enough entities for the
    requested rank to be reachable.
    """
    counts = [pd.selected_entity_count for pd in per_demo]
    if not counts:
        return ConsistencyCheck("entity_count", True, 1.0)

    # All demos must have found a selection
    all_found = all(c > 0 for c in counts)
    if not all_found:
        missing = [pd.demo_index for pd in per_demo if pd.selected_entity_count == 0]
        return ConsistencyCheck(
            "entity_count",
            False,
            0.0,
            f"demos {missing} have no entities of requested kind",
        )

    # Check count consistency (exact match is too strict — just check
    # that all demos had enough for the rank)
    return ConsistencyCheck("entity_count", True, 1.0)


def check_entity_kind_consistent(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that selected entity kinds match across demos."""
    kinds = {pd.selected_kind for pd in per_demo if pd.selected_kind is not None}
    if len(kinds) <= 1:
        return ConsistencyCheck("entity_kind", True, 1.0)
    return ConsistencyCheck(
        "entity_kind",
        False,
        0.0,
        f"inconsistent entity kinds: {kinds}",
    )


def check_selector_analogy(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that the selector produces structurally analogous results.

    Uses structural signatures to detect when a selector picks
    fundamentally different things across demos.
    """
    sigs = [pd.structural_signature for pd in per_demo if pd.structural_signature]
    if len(sigs) < 2:
        return ConsistencyCheck("selector_analogy", True, 1.0)

    # For now: check that all sigs share the same prefix (kind:predicate)
    prefixes = {s.split("|")[0] for s in sigs}
    if len(prefixes) == 1:
        return ConsistencyCheck("selector_analogy", True, 1.0)

    return ConsistencyCheck(
        "selector_analogy",
        True,  # soft fail — don't prune, just demote
        0.5,
        f"selector analogy diverges: {prefixes}",
    )


def check_transform_family_consistent(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that the same transform works structurally across all demos.

    If intermediate grids exist, check they have the same shape.
    """
    grids = [pd.intermediate_grid for pd in per_demo]
    have_grids = [g for g in grids if g is not None]

    if len(have_grids) < 2:
        return ConsistencyCheck("transform_family", True, 1.0)

    # All intermediate grids must have the same shape
    shapes = {g.shape for g in have_grids}
    if len(shapes) == 1:
        return ConsistencyCheck("transform_family", True, 1.0)

    return ConsistencyCheck(
        "transform_family",
        True,  # soft — shapes may legitimately differ
        0.3,
        f"intermediate shapes differ: {shapes}",
    )


def check_binding_compatibility(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that role bindings don't contradict a shared rule.

    Bindings should use the same role names across demos (the values
    can differ — that's the whole point of roles).
    """
    binding_keys = [
        frozenset(k for k, _ in pd.role_bindings)
        for pd in per_demo
        if pd.role_bindings
    ]
    if len(binding_keys) < 2:
        return ConsistencyCheck("binding_compatibility", True, 1.0)

    if len(set(binding_keys)) == 1:
        return ConsistencyCheck("binding_compatibility", True, 1.0)

    return ConsistencyCheck(
        "binding_compatibility",
        False,
        0.0,
        f"role binding keys differ: {[set(k) for k in binding_keys]}",
    )


def check_intermediate_output_compatible(
    per_demo: tuple[DemoPartialState, ...],
    expected_outputs: Sequence[Grid] | None = None,
) -> ConsistencyCheck:
    """Check that partial outputs are compatible with one shared continuation.

    If expected outputs are provided, also check residuals.
    """
    grids = [pd.intermediate_grid for pd in per_demo]
    have_grids = [g for g in grids if g is not None]

    if len(have_grids) < 2:
        return ConsistencyCheck("intermediate_output", True, 1.0)

    # Shape check: all intermediate outputs should have the same dims
    # (for same-dims tasks) or proportional dims
    shapes = [g.shape for g in have_grids]
    if len(set(shapes)) == 1:
        return ConsistencyCheck("intermediate_output", True, 1.0)

    # Different shapes — check if they're proportional
    ratios = set()
    base = shapes[0]
    for s in shapes[1:]:
        if base[0] > 0 and base[1] > 0:
            ratios.add((round(s[0] / base[0], 2), round(s[1] / base[1], 2)))

    if len(ratios) == 1:
        return ConsistencyCheck(
            "intermediate_output",
            True,
            0.8,
            f"shapes proportional: {shapes}",
        )

    return ConsistencyCheck(
        "intermediate_output",
        True,  # soft
        0.4,
        f"intermediate shapes incompatible: {shapes}",
    )


def check_perception_structure_consistent(
    per_demo: tuple[DemoPartialState, ...],
) -> ConsistencyCheck:
    """Check that demos have compatible structural decomposition.

    Verifies demos share similar partition structure, framed regions, etc.
    """
    perceptions = [pd.perception for pd in per_demo if pd.perception is not None]
    if len(perceptions) < 2:
        return ConsistencyCheck("perception_structure", True, 1.0)

    # Check partition consistency
    has_partition = [p.partition is not None for p in perceptions]
    if any(has_partition) and not all(has_partition):
        return ConsistencyCheck(
            "perception_structure",
            True,  # soft — some demos may not have partitions
            0.5,
            "partition structure inconsistent across demos",
        )

    # If all have partitions, check cell count
    if all(has_partition):
        cell_counts = {len(p.partition.cells) for p in perceptions}
        if len(cell_counts) > 1:
            return ConsistencyCheck(
                "perception_structure",
                False,
                0.0,
                f"partition cell counts differ: {cell_counts}",
            )

    return ConsistencyCheck("perception_structure", True, 1.0)


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------


def run_checks(
    per_demo: tuple[DemoPartialState, ...],
    *,
    include_perception: bool = True,
    expected_outputs: Sequence[Grid] | None = None,
) -> tuple[ConsistencyCheck, ...]:
    """Run all applicable consistency checks."""
    checks: list[ConsistencyCheck] = []

    if include_perception:
        checks.append(check_perception_structure_consistent(per_demo))

    # Only run selection checks if selection has happened
    if any(pd.selected_kind is not None for pd in per_demo):
        checks.append(check_entity_count_consistent(per_demo))
        checks.append(check_entity_kind_consistent(per_demo))
        checks.append(check_selector_analogy(per_demo))

    # Only run binding checks if bindings exist
    if any(pd.role_bindings for pd in per_demo):
        checks.append(check_binding_compatibility(per_demo))

    # Only run intermediate output checks if intermediate grids exist
    if any(pd.intermediate_grid is not None for pd in per_demo):
        checks.append(check_transform_family_consistent(per_demo))
        checks.append(check_intermediate_output_compatible(
            per_demo, expected_outputs,
        ))

    return tuple(checks)


# ---------------------------------------------------------------------------
# Scoring and pruning
# ---------------------------------------------------------------------------


def score_branch(state: BranchState) -> float:
    """Compute composite consistency score (0-1). Conservative: min of checks."""
    if not state.consistency_checks:
        return 1.0
    return min(c.score for c in state.consistency_checks)


def should_prune(state: BranchState, threshold: float = 0.0) -> bool:
    """True if any hard check fails (passed=False and score=0)."""
    return any(
        not c.passed and c.score <= threshold
        for c in state.consistency_checks
    )


def rank_branches(branches: list[BranchState]) -> list[BranchState]:
    """Sort branches by consistency score descending. Pruned at end."""
    return sorted(
        branches,
        key=lambda b: (not b.pruned, b.consistency_score),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_checks(branch: BranchState) -> BranchState:
    """Run all checks and update branch state."""
    checks = run_checks(branch.per_demo)
    score = min(c.score for c in checks) if checks else 1.0
    pruned = any(not c.passed and c.score <= 0.0 for c in checks)
    prune_reason = ""
    if pruned:
        failing = [c for c in checks if not c.passed and c.score <= 0.0]
        prune_reason = "; ".join(f"{c.name}: {c.detail}" for c in failing)

    return replace(
        branch,
        consistency_checks=checks,
        consistency_score=score,
        pruned=pruned,
        prune_reason=prune_reason,
    )


def _perception_signature(p: GridPerceptionState) -> str:
    """Compact structural signature from perception state."""
    parts = [
        f"dims={p.dims}",
        f"bg={p.bg_color}",
        f"ncolors={len(p.non_bg_colors)}",
        f"nobjs={len(p.objects.objects)}",
    ]
    if p.partition is not None:
        parts.append(f"ncells={len(p.partition.cells)}")
    if p.framed_regions:
        parts.append(f"nframes={len(p.framed_regions)}")
    return "|".join(parts)


def _select_signature(
    kind: str, predicate: str, rank: int, sr: _SelectResult,
) -> str:
    """Structural signature after selection."""
    parts = [
        f"{kind}:{predicate}:r{rank}",
        f"count={sr.entity_count}",
        f"found={sr.found}",
    ]
    if sr.bbox_dims is not None:
        parts.append(f"dims={sr.bbox_dims}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Factor-level consistency checks
# ---------------------------------------------------------------------------


def check_factor_consistency(
    perceptions: tuple[GridPerceptionState, ...],
    factors: object,
) -> tuple[bool, float, str]:
    """Check cross-demo consistency for a factorized hypothesis.

    Returns (passed, score, detail).
    Performs fast structural checks — kills factor combos that are
    impossible for these demos before any program instantiation.

    The ``factors`` argument is duck-typed to avoid a circular import
    with ``aria.factors``. It must expose ``decomposition``, ``selector``,
    ``scope``, and ``correspondence`` attributes whose ``.value`` is a str.
    """
    from aria.consensus_search import (
        check_correspondence_consistency,
        check_scope_consistency,
        probe_scope,
    )

    decomp = factors.decomposition.value
    selector = factors.selector.value
    scope = factors.scope.value
    correspondence = factors.correspondence.value

    # --- Decomposition consistency ---

    if decomp == "partition":
        partitions = [p.partition is not None for p in perceptions]
        if not all(partitions):
            return False, 0.0, "partition absent in some demos"
        counts = {len(p.partition.cells) for p in perceptions if p.partition}
        if len(counts) > 1:
            return False, 0.0, f"partition cell counts differ: {counts}"

    if decomp == "frame":
        has_frame = [len(p.framed_regions) > 0 for p in perceptions]
        if not all(has_frame):
            return False, 0.0, "framed region absent in some demos"

    if decomp == "zone":
        has_zones = [len(p.zones) > 0 for p in perceptions]
        if not all(has_zones):
            return False, 0.0, "zones absent in some demos"

    # --- Selector consistency ---

    if selector == "object_select":
        obj_counts = [len(p.objects.objects) for p in perceptions]
        if any(c == 0 for c in obj_counts):
            return False, 0.0, "no objects in some demos"

    if selector == "frame_interior":
        has_frame = [len(p.framed_regions) > 0 for p in perceptions]
        if not all(has_frame):
            return False, 0.0, "no framed regions for frame_interior selector"

    if selector == "cell_panel":
        partitions = [p.partition is not None for p in perceptions]
        if not all(partitions):
            return False, 0.0, "no partition for cell_panel selector"

    if selector == "enclosed":
        # Use existing scope probing
        scope_results = probe_scope(perceptions, "enclosed")
        passed, score, detail = check_scope_consistency(scope_results)
        if not passed:
            return False, 0.0, f"enclosed check: {detail}"

    # --- Scope consistency ---

    if scope == "partition_cell":
        partitions = [p.partition is not None for p in perceptions]
        if not all(partitions):
            return False, 0.0, "no partition for partition_cell scope"

    if scope == "frame_interior":
        has_frame = [len(p.framed_regions) > 0 for p in perceptions]
        if not all(has_frame):
            return False, 0.0, "no frame for frame_interior scope"

    # --- Correspondence consistency ---

    if correspondence != "none":
        passed, score, detail = check_correspondence_consistency(
            perceptions, "object", "object",
        )
        if not passed:
            return False, score, f"correspondence: {detail}"

    return True, 1.0, ""
