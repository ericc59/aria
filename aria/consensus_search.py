"""Consensus-controlled compositional search.

Applies stepwise all-demo consensus as the controller for 2-step
compositional search: select-scope → apply-operation.

Instead of enumerating all (scope, operation) pairs and verifying each,
this module:
1. Enumerates scope hypotheses
2. Executes step-1 (scope selection) on ALL demos
3. Checks cross-demo consistency of step-1 results
4. Only tries step-2 (operations) on consistent scopes
5. Ranks branches by consensus score

This dramatically reduces the search space for:
- object-scoped edits (select object → recolor/fill/transform)
- enclosed-region selection (detect regions → conditional fill)
- scope-aware recoloring (identify scope → apply color map)
- object correspondence (match objects → transfer properties)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from aria.consensus import (
    BranchState,
    ConsistencyCheck,
    DemoPartialState,
    SharedHypothesis,
    should_prune,
)
from aria.consensus_trace import ConsensusTrace
from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.core.scene_executor import SceneExecutionError, execute_scene_program
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Step-1 result: scope / selection across all demos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopeResult:
    """Result of executing a scope-selection step on one demo."""
    found: bool = False
    n_entities: int = 0
    scope_kind: str = ""  # "partition", "objects", "frame", "enclosed"
    scope_dims: tuple[int, int] | None = None
    # For object scopes: object count and shape signature
    object_count: int = 0
    shape_signature: str = ""
    # For enclosed regions: region count
    region_count: int = 0
    # Intermediate state (partial execution result)
    intermediate: Any = None


@dataclass(frozen=True)
class CompositionBranch:
    """A composition search branch with consensus state."""
    branch_id: str
    scope_hypothesis: str  # e.g. "objects", "frame_interior", "enclosed"
    operation_hypothesis: str | None = None  # e.g. "recolor", "fill", "transform"
    per_demo_scope: tuple[ScopeResult, ...] = ()
    consistency_score: float = 1.0
    pruned: bool = False
    prune_reason: str = ""
    program: SceneProgram | None = None


# ---------------------------------------------------------------------------
# Cross-demo scope consistency
# ---------------------------------------------------------------------------

def check_scope_consistency(
    per_demo: tuple[ScopeResult, ...],
) -> tuple[bool, float, str]:
    """Check cross-demo consistency of scope selection results.

    Returns (passed, score, detail).
    """
    if not per_demo:
        return True, 1.0, ""

    # All demos must have found the scope
    if not all(sr.found for sr in per_demo):
        missing = [i for i, sr in enumerate(per_demo) if not sr.found]
        return False, 0.0, f"scope not found in demos {missing}"

    # For object scopes: check object counts are compatible
    if per_demo[0].scope_kind == "objects":
        counts = [sr.object_count for sr in per_demo]
        if any(c == 0 for c in counts):
            return False, 0.0, f"zero objects in some demos: {counts}"
        # Counts don't need to match exactly, but should all be > 0

    # For partition scopes: check cell counts match
    if per_demo[0].scope_kind == "partition":
        counts = [sr.n_entities for sr in per_demo]
        if len(set(counts)) > 1:
            return False, 0.0, f"partition cell counts differ: {counts}"

    # For enclosed regions: check region counts are compatible
    if per_demo[0].scope_kind == "enclosed":
        counts = [sr.region_count for sr in per_demo]
        if any(c == 0 for c in counts):
            return False, 0.0, f"no enclosed regions in some demos: {counts}"

    # For frame scopes: check frame exists in all demos
    if per_demo[0].scope_kind == "frame":
        if not all(sr.found for sr in per_demo):
            return False, 0.0, "frame not found in all demos"

    return True, 1.0, ""


def check_correspondence_consistency(
    perceptions: tuple[GridPerceptionState, ...],
    source_kind: str,
    target_kind: str,
) -> tuple[bool, float, str]:
    """Check whether an object correspondence hypothesis is consistent.

    Verifies that source and target entity kinds exist in all demos
    with compatible counts.
    """
    for i, p in enumerate(perceptions):
        src_count = _count_entities(p, source_kind)
        tgt_count = _count_entities(p, target_kind)
        if src_count == 0:
            return False, 0.0, f"demo {i}: no {source_kind} entities"
        if tgt_count == 0:
            return False, 0.0, f"demo {i}: no {target_kind} entities"

    # Check count ratios are consistent across demos
    src_counts = [_count_entities(p, source_kind) for p in perceptions]
    tgt_counts = [_count_entities(p, target_kind) for p in perceptions]

    if len(set(src_counts)) > 1 and len(set(tgt_counts)) > 1:
        # Both vary — check if ratios are consistent
        ratios = set()
        for s, t in zip(src_counts, tgt_counts):
            if t > 0:
                ratios.add(round(s / t, 2))
        if len(ratios) > 1:
            return True, 0.5, f"correspondence ratio varies: src={src_counts}, tgt={tgt_counts}"

    return True, 1.0, ""


def _count_entities(p: GridPerceptionState, kind: str) -> int:
    if kind == "object":
        return len(p.objects.objects)
    if kind == "panel":
        return len(p.partition.cells) if p.partition else 0
    if kind == "frame":
        return len(p.framed_regions)
    return 0


# ---------------------------------------------------------------------------
# Scope probing — execute step-1 on all demos
# ---------------------------------------------------------------------------

def probe_scope(
    perceptions: tuple[GridPerceptionState, ...],
    scope_kind: str,
) -> tuple[ScopeResult, ...]:
    """Probe a scope hypothesis across all demos without building a program."""
    results = []
    for p in perceptions:
        sr = _probe_scope_single(p, scope_kind)
        results.append(sr)
    return tuple(results)


def _probe_scope_single(p: GridPerceptionState, scope_kind: str) -> ScopeResult:
    """Probe scope on one demo."""
    if scope_kind == "objects":
        objs = p.objects.objects
        non_bg = [o for o in objs if o.color != p.bg_color]
        return ScopeResult(
            found=len(non_bg) > 0,
            n_entities=len(non_bg),
            scope_kind="objects",
            object_count=len(non_bg),
        )

    if scope_kind == "partition":
        if p.partition is None:
            return ScopeResult(found=False, scope_kind="partition")
        return ScopeResult(
            found=True,
            n_entities=len(p.partition.cells),
            scope_kind="partition",
        )

    if scope_kind == "frame":
        if not p.framed_regions:
            return ScopeResult(found=False, scope_kind="frame")
        fr = p.framed_regions[0]
        return ScopeResult(
            found=True,
            n_entities=len(p.framed_regions),
            scope_kind="frame",
            scope_dims=(fr.height, fr.width),
        )

    if scope_kind == "enclosed":
        # Check for enclosed regions (bg cells surrounded by non-bg)
        has_enclosed = _has_enclosed_regions(p)
        return ScopeResult(
            found=has_enclosed,
            scope_kind="enclosed",
            region_count=1 if has_enclosed else 0,
        )

    return ScopeResult(found=False, scope_kind=scope_kind)


def _has_enclosed_regions(p: GridPerceptionState) -> bool:
    """Quick check: does the grid have enclosed bg regions?"""
    from scipy import ndimage
    bg_mask = (p.grid == p.bg_color)
    labeled, n = ndimage.label(bg_mask)
    if n <= 1:
        return False
    # More than 1 bg component means some bg is enclosed
    # The largest component is the outer bg; others are enclosed
    sizes = ndimage.sum(bg_mask, labeled, range(1, n + 1))
    return sum(1 for s in sizes if s < max(sizes)) > 0


# ---------------------------------------------------------------------------
# Consensus-controlled compositional search
# ---------------------------------------------------------------------------

def consensus_compose_search(
    demos: tuple[DemoPair, ...],
    *,
    trace: ConsensusTrace | None = None,
) -> list[tuple[str, SceneProgram, float]]:
    """Run consensus-controlled 2-step composition search.

    Returns list of (description, program, score) ordered by consensus score.
    Only returns verified programs.
    """
    from aria.core.scene_solve import get_consensus_enabled

    if not demos:
        return []
    if not all(d.input.shape == d.output.shape for d in demos):
        return []

    perceptions = tuple(perceive_grid(d.input) for d in demos)
    consensus_on = get_consensus_enabled()
    results: list[tuple[str, SceneProgram, float]] = []

    # --- Step 1: Probe all scope hypotheses ---
    scope_kinds = ["objects", "enclosed", "frame", "partition"]
    consistent_scopes: list[tuple[str, tuple[ScopeResult, ...], float]] = []

    for scope_kind in scope_kinds:
        scope_results = probe_scope(perceptions, scope_kind)
        passed, score, detail = check_scope_consistency(scope_results)

        if trace is not None:
            branch = CompositionBranch(
                branch_id=f"scope:{scope_kind}",
                scope_hypothesis=scope_kind,
                per_demo_scope=scope_results,
                consistency_score=score,
                pruned=not passed,
                prune_reason=detail if not passed else "",
            )
            _trace_branch(trace, branch, f"scope probe: {scope_kind}")

        if consensus_on and not passed:
            continue

        consistent_scopes.append((scope_kind, scope_results, score))

    # --- Step 2: For each consistent scope, try operations ---
    for scope_kind, scope_results, scope_score in consistent_scopes:
        ops = _operations_for_scope(scope_kind, demos, perceptions)

        for op_name, prog in ops:
            # Check object correspondence if relevant
            if "correspond" in op_name and consensus_on:
                passed, cscore, detail = check_correspondence_consistency(
                    perceptions, "object", "object",
                )
                if not passed:
                    if trace is not None:
                        branch = CompositionBranch(
                            branch_id=f"{scope_kind}:{op_name}",
                            scope_hypothesis=scope_kind,
                            operation_hypothesis=op_name,
                            per_demo_scope=scope_results,
                            consistency_score=0.0,
                            pruned=True,
                            prune_reason=detail,
                        )
                        _trace_branch(trace, branch, f"correspondence check: {op_name}")
                    continue

            # Verify the composed program
            verified = _verify_all_demos(prog, demos)
            if verified:
                results.append((f"{scope_kind}:{op_name}", prog, scope_score))
                if trace is not None:
                    branch = CompositionBranch(
                        branch_id=f"{scope_kind}:{op_name}",
                        scope_hypothesis=scope_kind,
                        operation_hypothesis=op_name,
                        per_demo_scope=scope_results,
                        consistency_score=scope_score,
                        program=prog,
                    )
                    _trace_branch(trace, branch, f"VERIFIED: {scope_kind}:{op_name}")

    # Sort by consensus score descending
    results.sort(key=lambda x: -x[2])
    return results


def _operations_for_scope(
    scope_kind: str,
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
) -> list[tuple[str, SceneProgram]]:
    """Generate step-2 operations appropriate for a given scope."""
    ops: list[tuple[str, SceneProgram]] = []
    d0 = demos[0]
    bg = perceptions[0].bg_color

    if scope_kind == "objects":
        # Object-scoped fill operations
        for color in range(10):
            for conn in [4, 8]:
                for rule in ("fill_bbox_holes", "fill_enclosed_bbox"):
                    prog = SceneProgram(steps=(
                        SceneStep(op=StepOp.PARSE_SCENE),
                        SceneStep(
                            op=StepOp.FOR_EACH_ENTITY,
                            params={"kind": "object", "rule": rule,
                                    "fill_color": color, "connectivity": conn},
                        ),
                        SceneStep(op=StepOp.RENDER_SCENE),
                    ))
                    ops.append((f"{rule}_c{color}_conn{conn}", prog))

        # Object-scoped color map
        cmap = _infer_color_map(demos)
        if cmap:
            pairs = sorted(cmap.items())
            for scope in ("objects", "object_bboxes"):
                prog = SceneProgram(steps=(
                    SceneStep(op=StepOp.PARSE_SCENE),
                    SceneStep(
                        op=StepOp.RECOLOR_OBJECT,
                        params={"color_pairs": pairs, "scope": scope},
                    ),
                    SceneStep(op=StepOp.RENDER_SCENE, params={"source": "recolored"}),
                ))
                ops.append((f"recolor_{scope}", prog))

    elif scope_kind == "enclosed":
        # Fill enclosed regions with various colors
        diff = d0.input != d0.output
        if np.any(diff):
            fill_colors = set(int(d0.output[r, c]) for r, c in zip(*np.where(diff)))
            for fc in sorted(fill_colors):
                prog = SceneProgram(steps=(
                    SceneStep(op=StepOp.PARSE_SCENE),
                    SceneStep(
                        op=StepOp.FILL_ENCLOSED_REGIONS,
                        params={"fill_color": fc},
                    ),
                    SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
                ))
                ops.append((f"fill_enclosed_c{fc}", prog))

            # Auto-fill (boundary color)
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(
                    op=StepOp.FILL_ENCLOSED_REGIONS,
                    params={"mode": "boundary_color"},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
            ))
            ops.append(("fill_enclosed_auto", prog))

    elif scope_kind == "frame":
        # Frame-interior operations
        cmap = _infer_color_map(demos)
        if cmap:
            pairs = sorted(cmap.items())
            prog = SceneProgram(steps=(
                SceneStep(op=StepOp.PARSE_SCENE),
                SceneStep(
                    op=StepOp.RECOLOR_OBJECT,
                    params={"color_pairs": pairs, "scope": "frame_interior"},
                ),
                SceneStep(op=StepOp.RENDER_SCENE, params={"source": "recolored"}),
            ))
            ops.append(("recolor_frame_interior", prog))

        # Fill enclosed inside frame
        diff = d0.input != d0.output
        if np.any(diff):
            fill_colors = set(int(d0.output[r, c]) for r, c in zip(*np.where(diff)))
            for fc in sorted(fill_colors):
                prog = SceneProgram(steps=(
                    SceneStep(op=StepOp.PARSE_SCENE),
                    SceneStep(
                        op=StepOp.FILL_ENCLOSED_REGIONS,
                        params={"fill_color": fc},
                    ),
                    SceneStep(op=StepOp.RENDER_SCENE, params={"source": "filled"}),
                ))
                ops.append((f"fill_enclosed_frame_c{fc}", prog))

    return ops


def _infer_color_map(demos: tuple[DemoPair, ...]) -> dict[int, int] | None:
    """Infer a consistent color map across all demos."""
    cmap: dict[int, int] = {}
    for demo in demos:
        if demo.input.shape != demo.output.shape:
            return None
        diff = demo.input != demo.output
        if not np.any(diff):
            continue
        for r, c in zip(*np.where(diff)):
            ic, oc = int(demo.input[r, c]), int(demo.output[r, c])
            if ic in cmap and cmap[ic] != oc:
                return None
            cmap[ic] = oc
    return cmap if cmap and len(cmap) <= 5 else None


def _verify_all_demos(prog: SceneProgram, demos: tuple[DemoPair, ...]) -> bool:
    """Verify a scene program on all demos."""
    for d in demos:
        try:
            result = execute_scene_program(prog, d.input)
            if result.shape != d.output.shape or not np.array_equal(result, d.output):
                return False
        except Exception:
            return False
    return True


def _trace_branch(trace: ConsensusTrace, branch: CompositionBranch, desc: str) -> None:
    """Record a composition branch in the consensus trace."""
    from aria.consensus import BranchState
    bs = BranchState(
        branch_id=branch.branch_id,
        step_index=0,
        per_demo=(),
        hypothesis=SharedHypothesis(
            rule_family=f"{branch.scope_hypothesis}->{branch.operation_hypothesis or '?'}",
            scope_family=branch.scope_hypothesis,
        ),
        consistency_score=branch.consistency_score,
        pruned=branch.pruned,
        prune_reason=branch.prune_reason,
    )
    trace.record(bs, desc)
