"""Learned proposal ranking for decomposition and sketch candidates.

Ranking hooks are optional and pluggable. When absent, candidates are
processed in their natural proposal order. When present, they reorder
candidates so the most promising are compiled/verified first, reducing
the number of executed candidates needed.

Ranking does NOT change semantics — exact verification remains the only
correctness gate. Ranking only changes efficiency.

Two ranking tasks:
- DECOMP_RANK: order decomposition views by predicted usefulness
- SKETCH_RANK: order fitted sketches by predicted compile/verify likelihood
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from aria.sketch import Sketch


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingReport:
    """Diagnostics for one ranking application."""

    task: str                     # "DECOMP_RANK" or "SKETCH_RANK"
    candidates_ranked: int
    order_changed: bool
    original_order: tuple[int, ...]
    ranked_order: tuple[int, ...]
    policy_name: str = "none"
    scores: tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# Sketch ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SketchRankInput:
    """Features available for ranking a set of sketches."""

    task_signatures: tuple[str, ...]
    sketch_families: tuple[str, ...]      # family name per sketch
    sketch_confidences: tuple[float, ...]  # sketch.confidence per sketch
    bg_rotates: bool                       # color roles rotate across demos
    same_dims: bool
    n_demos: int


# The ranking callable signature: takes SketchRankInput + list of sketches,
# returns indices in preferred order.
SketchRanker = Callable[[SketchRankInput, Sequence[Sketch]], tuple[int, ...]]


def heuristic_sketch_ranker(
    inp: SketchRankInput,
    sketches: Sequence[Sketch],
) -> tuple[int, ...]:
    """Deterministic heuristic ranker for sketch candidates.

    Priority (best first):
    1. Higher confidence
    2. Families that match ARC-AGI-2 pressures:
       - composite_role_alignment when bg_rotates (symbolic interpretation)
       - framed_periodic_repair when has_frame (compositional reasoning)
    3. Stable tie-break by original index
    """
    sigs = frozenset(inp.task_signatures)

    def _score(idx: int) -> tuple[float, int, int]:
        s = sketches[idx]
        family = s.metadata.get("family", "")
        conf = s.confidence

        # Family bonus
        bonus = 0.0
        if family == "composite_role_alignment" and inp.bg_rotates:
            bonus = 0.3
        elif family == "framed_periodic_repair" and "role:has_frame" in sigs:
            bonus = 0.2

        # Higher score = better, so negate for sorting
        return (-conf - bonus, idx, 0)

    ranked = sorted(range(len(sketches)), key=_score)
    return tuple(ranked)


def rank_sketches(
    sketches: list[Sketch],
    task_signatures: tuple[str, ...],
    demos_metadata: dict[str, Any],
    *,
    ranker: SketchRanker | None = None,
) -> tuple[list[Sketch], RankingReport]:
    """Apply ranking to a list of fitted sketches.

    Returns (reordered_sketches, report).
    If no ranker is provided, returns original order.
    """
    if not sketches:
        return sketches, RankingReport(
            task="SKETCH_RANK",
            candidates_ranked=0,
            order_changed=False,
            original_order=(),
            ranked_order=(),
        )

    original_order = tuple(range(len(sketches)))

    if ranker is None:
        return sketches, RankingReport(
            task="SKETCH_RANK",
            candidates_ranked=len(sketches),
            order_changed=False,
            original_order=original_order,
            ranked_order=original_order,
            policy_name="none",
        )

    inp = SketchRankInput(
        task_signatures=task_signatures,
        sketch_families=tuple(s.metadata.get("family", "") for s in sketches),
        sketch_confidences=tuple(s.confidence for s in sketches),
        bg_rotates=demos_metadata.get("bg_rotates", False),
        same_dims=demos_metadata.get("same_dims", True),
        n_demos=demos_metadata.get("n_demos", 0),
    )

    ranked_order = ranker(inp, sketches)
    reordered = [sketches[i] for i in ranked_order]
    changed = list(ranked_order) != list(original_order)

    return reordered, RankingReport(
        task="SKETCH_RANK",
        candidates_ranked=len(sketches),
        order_changed=changed,
        original_order=original_order,
        ranked_order=ranked_order,
        policy_name=getattr(ranker, "__name__", "custom"),
    )


# ---------------------------------------------------------------------------
# Decomposition ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompRankInput:
    """Features available for ranking decomposition views."""

    task_signatures: tuple[str, ...]
    view_names: tuple[str, ...]    # e.g. ("objects", "framed_regions", "composites", "markers")
    same_dims: bool
    bg_rotates: bool
    n_demos: int


DecompRanker = Callable[[DecompRankInput, Sequence[str]], tuple[int, ...]]


def heuristic_decomp_ranker(
    inp: DecompRankInput,
    view_names: Sequence[str],
) -> tuple[int, ...]:
    """Deterministic heuristic ranker for decomposition views.

    Priority:
    1. composites when bg_rotates (symbolic interpretation)
    2. framed_regions when has_frame (compositional reasoning)
    3. markers when has_marker (contextual rule application)
    4. objects (always useful)
    """
    sigs = frozenset(inp.task_signatures)

    def _score(idx: int) -> tuple[int, int]:
        name = view_names[idx]
        priority = 99
        if name == "composites" and inp.bg_rotates:
            priority = 0
        elif name == "framed_regions" and "role:has_frame" in sigs:
            priority = 1
        elif name == "markers" and "role:has_marker" in sigs:
            priority = 2
        elif name == "objects":
            priority = 3
        return (priority, idx)

    ranked = sorted(range(len(view_names)), key=_score)
    return tuple(ranked)


def rank_decompositions(
    view_names: list[str],
    task_signatures: tuple[str, ...],
    demos_metadata: dict[str, Any],
    *,
    ranker: DecompRanker | None = None,
) -> tuple[list[str], RankingReport]:
    """Apply ranking to decomposition view names.

    Returns (reordered_names, report).
    """
    if not view_names:
        return view_names, RankingReport(
            task="DECOMP_RANK",
            candidates_ranked=0,
            order_changed=False,
            original_order=(),
            ranked_order=(),
        )

    original_order = tuple(range(len(view_names)))

    if ranker is None:
        return view_names, RankingReport(
            task="DECOMP_RANK",
            candidates_ranked=len(view_names),
            order_changed=False,
            original_order=original_order,
            ranked_order=original_order,
            policy_name="none",
        )

    inp = DecompRankInput(
        task_signatures=task_signatures,
        view_names=tuple(view_names),
        same_dims=demos_metadata.get("same_dims", True),
        bg_rotates=demos_metadata.get("bg_rotates", False),
        n_demos=demos_metadata.get("n_demos", 0),
    )

    ranked_order = ranker(inp, view_names)
    reordered = [view_names[i] for i in ranked_order]
    changed = list(ranked_order) != list(original_order)

    return reordered, RankingReport(
        task="DECOMP_RANK",
        candidates_ranked=len(view_names),
        order_changed=changed,
        original_order=original_order,
        ranked_order=ranked_order,
        policy_name=getattr(ranker, "__name__", "custom"),
    )


# ---------------------------------------------------------------------------
# Training data export
# ---------------------------------------------------------------------------


def export_sketch_rank_example(
    task_id: str,
    task_signatures: tuple[str, ...],
    sketch_families: tuple[str, ...],
    winning_family: str | None,
    verified_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Export a SKETCH_RANK training example.

    The label is the index of the winning/verified sketch (or -1 if none).
    """
    label = -1
    if winning_family is not None:
        for i, fam in enumerate(sketch_families):
            if fam == winning_family:
                label = i
                break

    return {
        "task_type": "SKETCH_RANK",
        "task_id": task_id,
        "input": {
            "task_signatures": list(task_signatures),
            "sketch_families": list(sketch_families),
            "n_candidates": len(sketch_families),
        },
        "label": {
            "winning_index": label,
            "winning_family": winning_family,
            "verified_indices": list(verified_indices),
        },
    }


def export_decomp_rank_example(
    task_id: str,
    task_signatures: tuple[str, ...],
    view_names: tuple[str, ...],
    useful_views: tuple[str, ...],
) -> dict[str, Any]:
    """Export a DECOMP_RANK training example.

    The label records which decomposition views produced useful evidence.
    """
    useful_indices = tuple(
        i for i, name in enumerate(view_names) if name in useful_views
    )

    return {
        "task_type": "DECOMP_RANK",
        "task_id": task_id,
        "input": {
            "task_signatures": list(task_signatures),
            "view_names": list(view_names),
            "n_views": len(view_names),
        },
        "label": {
            "useful_indices": list(useful_indices),
            "useful_views": list(useful_views),
        },
    }
