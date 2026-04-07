"""Search-cost opportunity audit and safe pruning for output-size candidates.

Quantifies where search budget is actually spent and evaluates whether
learned guidance can safely skip expensive candidates under high confidence.

No task-id logic. No solver correctness changes. Exact verification is truth.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.core.guidance_datasets import (
    FEATURE_NAMES,
    SIZE_MODE_FAMILIES,
    SIZE_FAMILY_NAMES,
    extract_perception_features,
    features_to_array,
)
from aria.core.guidance_size_model import SizeModelArtifact
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Search-cost audit
# ---------------------------------------------------------------------------


@dataclass
class CandidateCost:
    """Cost profile for one output-size candidate type."""
    name: str
    family: str
    total_time_s: float
    mean_time_ms: float
    max_time_ms: float
    n_tasks: int
    n_produces: int           # how often it returns a non-None spec
    n_is_winner: int          # how often it is the first verified spec (rank 0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "total_time_s": round(self.total_time_s, 3),
            "mean_time_ms": round(self.mean_time_ms, 1),
            "max_time_ms": round(self.max_time_ms, 0),
            "n_tasks": self.n_tasks,
            "n_produces": self.n_produces,
            "n_is_winner": self.n_is_winner,
            "produce_rate": round(self.n_produces / max(self.n_tasks, 1), 3),
            "cost_per_win": round(self.total_time_s / max(self.n_is_winner, 1), 3),
        }


@dataclass
class SearchCostAudit:
    """Complete search-cost audit results."""
    n_tasks: int
    total_time_s: float
    candidate_costs: list[CandidateCost]
    # Summary
    top_cost_candidate: str
    top_cost_fraction: float
    prunable_time_s: float     # time saveable by skipping non-winning expensive candidates
    # Opportunity ranking
    opportunities: list[dict]  # sorted by expected savings

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "total_time_s": round(self.total_time_s, 2),
            "top_cost_candidate": self.top_cost_candidate,
            "top_cost_fraction": round(self.top_cost_fraction, 3),
            "prunable_time_s": round(self.prunable_time_s, 3),
            "candidate_costs": [c.to_dict() for c in self.candidate_costs],
            "opportunities": self.opportunities,
        }


def audit_search_costs(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> SearchCostAudit:
    """Audit actual time spent on each output-size candidate across tasks."""
    from aria.core.output_size import (
        infer_verified_output_size_specs,
        verify_output_size_spec,
    )
    from aria.core import output_size as osz

    # All candidate functions in order
    candidate_fns = _get_candidate_fns()

    times: dict[str, list[float]] = defaultdict(list)
    produces: dict[str, int] = Counter()
    winners: dict[str, int] = Counter()

    for task_id, demos in task_demo_pairs:
        # Run full enumeration to find the actual winner
        verified = infer_verified_output_size_specs(demos)
        winner_mode = verified[0].mode if verified else None

        for name, fn in candidate_fns:
            t0 = time.perf_counter()
            result = fn(demos)
            t1 = time.perf_counter()
            times[name].append(t1 - t0)
            if result is not None:
                produces[name] += 1
                if result.mode == winner_mode:
                    winners[name] += 1

    n = len(task_demo_pairs)
    costs = []
    for name, fn in candidate_fns:
        t = times[name]
        family = SIZE_MODE_FAMILIES.get(name, "other")
        costs.append(CandidateCost(
            name=name, family=family,
            total_time_s=sum(t), mean_time_ms=np.mean(t) * 1000,
            max_time_ms=max(t) * 1000 if t else 0,
            n_tasks=n, n_produces=produces[name], n_is_winner=winners[name],
        ))

    total_time = sum(c.total_time_s for c in costs)
    costs.sort(key=lambda c: -c.total_time_s)
    top = costs[0] if costs else None

    # Prunable = time of expensive candidates that never win
    prunable = sum(c.total_time_s for c in costs if c.n_is_winner == 0 and c.mean_time_ms > 1)

    # Opportunity ranking: savings = total_time * (1 - win_rate)
    opportunities = []
    for c in costs:
        if c.total_time_s < 0.01:
            continue
        win_rate = c.n_is_winner / max(c.n_tasks, 1)
        savings = c.total_time_s * (1 - win_rate)
        opportunities.append({
            "candidate": c.name,
            "family": c.family,
            "savings_s": round(savings, 3),
            "win_rate": round(win_rate, 3),
            "total_cost_s": round(c.total_time_s, 3),
        })
    opportunities.sort(key=lambda x: -x["savings_s"])

    return SearchCostAudit(
        n_tasks=n,
        total_time_s=total_time,
        candidate_costs=costs,
        top_cost_candidate=top.name if top else "",
        top_cost_fraction=top.total_time_s / max(total_time, 1e-9) if top else 0,
        prunable_time_s=prunable,
        opportunities=opportunities,
    )


def _get_candidate_fns():
    from aria.core import output_size as osz
    return [
        ("same_as_input", osz._candidate_same_as_input),
        ("transpose_input", osz._candidate_transpose_input),
        ("scale_input_by_palette_size", osz._candidate_scale_input_by_palette_size),
        ("square_canvas_scaled", osz._candidate_square_canvas_scaled),
        ("row_strip_square_from_cols", osz._candidate_row_strip_square_from_cols),
        ("row_strip_half_cols_by_cols", osz._candidate_row_strip_half_cols_by_cols),
        ("row_strip_square_by_non_bg_pixel_count", osz._candidate_row_strip_square_by_non_bg_pixel_count),
        ("preserve_input_rows_expand_cols_by_rows_minus_one", osz._candidate_preserve_input_rows_expand_cols_by_rows_minus_one),
        ("expand_rows_to_four_times_minus_three_preserve_cols", osz._candidate_expand_rows_to_four_times_minus_three_preserve_cols),
        ("selected_partition_cell_size", osz._candidate_selected_partition_cell_size),
        ("separator_cell_size", osz._candidate_separator_cell_size),
        ("separator_panel_size", osz._candidate_separator_panel_size),
        ("partition_grid_shape", osz._candidate_partition_grid_shape),
        ("frame_interior_size", osz._candidate_frame_interior_size),
        ("selected_boxed_region_interior", osz._candidate_selected_boxed_region_interior),
        ("selected_strip_block_size", osz._candidate_selected_strip_block_size),
        ("tight_bbox_of_non_bg", osz._candidate_tight_bbox_of_non_bg),
        ("bbox_of_selected_object", osz._candidate_bbox_of_selected_object),
        ("bbox_of_selected_color", osz._candidate_bbox_of_selected_color),
        ("object_position_grid_shape", osz._candidate_object_position_grid_shape),
        ("solid_rectangle_layout_shape", osz._candidate_solid_rectangle_layout_shape),
        ("dominant_shape_top_row_span", osz._candidate_dominant_shape_top_row_span),
        ("non_bg_color_count_strip", osz._candidate_non_bg_color_count_strip),
        ("scale_input", osz._candidate_scale_input),
        ("additive_input", osz._candidate_additive_input),
        ("preserve_input_rows_non_bg_cols", osz._candidate_preserve_input_rows_non_bg_cols),
        ("constant_rows_preserve_input_cols", osz._candidate_constant_rows_preserve_input_cols),
        ("fixed_output_dims", osz._candidate_fixed_output_dims),
        ("scaled_bbox_of_selected_object", osz._candidate_scaled_bbox_of_selected_object),
        ("marker_stacked_selected_object", osz._candidate_marker_stacked_selected_object),
    ]


# ---------------------------------------------------------------------------
# Safe pruning evaluation
# ---------------------------------------------------------------------------


@dataclass
class PruningShadowResult:
    """Result of simulating pruning on one task."""
    task_id: str
    n_candidates_default: int
    n_candidates_pruned: int
    candidates_skipped: int
    time_saved_ms: float
    winner_preserved: bool      # was the actual winner kept in the pruned set?
    pruned_families: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_candidates_default": self.n_candidates_default,
            "n_candidates_pruned": self.n_candidates_pruned,
            "candidates_skipped": self.candidates_skipped,
            "time_saved_ms": round(self.time_saved_ms, 1),
            "winner_preserved": self.winner_preserved,
            "pruned_families": list(self.pruned_families),
        }


@dataclass
class PruningReport:
    """Aggregate pruning simulation results."""
    n_tasks: int
    n_evaluated: int
    n_winner_preserved: int
    n_winner_lost: int
    safety_rate: float          # fraction where winner is preserved
    total_time_saved_ms: float
    avg_time_saved_ms: float
    avg_candidates_skipped: float
    pruned_family_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "n_evaluated": self.n_evaluated,
            "n_winner_preserved": self.n_winner_preserved,
            "n_winner_lost": self.n_winner_lost,
            "safety_rate": round(self.safety_rate, 4),
            "total_time_saved_ms": round(self.total_time_saved_ms, 1),
            "avg_time_saved_ms": round(self.avg_time_saved_ms, 1),
            "avg_candidates_skipped": round(self.avg_candidates_skipped, 1),
            "pruned_family_counts": self.pruned_family_counts,
        }


def evaluate_pruning(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
    model: SizeModelArtifact,
    confidence_threshold: float = 0.9,
    max_prune_families: int = 3,
    never_prune: frozenset[str] = frozenset({"same_dims"}),
) -> tuple[PruningReport, list[PruningShadowResult]]:
    """Simulate safe output-size family pruning.

    Rules:
    1. Only prune families the model assigns < (1 - confidence_threshold) probability
    2. Never prune the model's top-1 prediction
    3. Never prune families in `never_prune`
    4. Never prune all candidates — always keep at least 2 families
    5. Prune at most `max_prune_families` families per task
    """
    from aria.core.output_size import infer_verified_output_size_specs

    candidate_fns = _get_candidate_fns()
    candidate_name_to_family = {name: SIZE_MODE_FAMILIES.get(name, "other") for name, _ in candidate_fns}

    # Pre-measure per-candidate time on a sample
    candidate_times = _estimate_candidate_times(task_demo_pairs[:min(30, len(task_demo_pairs))])

    results: list[PruningShadowResult] = []
    pruned_family_counts: Counter = Counter()

    for task_id, demos in task_demo_pairs:
        # Get model prediction
        features = extract_perception_features(demos)
        probs = model.predict_proba(features_to_array(features))
        model_top = model.label_names[np.argmax(probs)]

        # Determine which families to prune
        families_to_prune: set[str] = set()
        for i, fam in enumerate(model.label_names):
            if fam in never_prune:
                continue
            if fam == model_top:
                continue
            if probs[i] < (1 - confidence_threshold):
                families_to_prune.add(fam)

        # Limit prune count
        if len(families_to_prune) > max_prune_families:
            # Prune the lowest-probability families
            scored = [(probs[i], fam) for i, fam in enumerate(model.label_names)
                      if fam in families_to_prune]
            scored.sort()
            families_to_prune = {fam for _, fam in scored[:max_prune_families]}

        # Ensure we don't prune everything
        all_families = set(model.label_names)
        remaining = all_families - families_to_prune
        if len(remaining) < 2:
            families_to_prune = set()  # safety: don't prune

        # Get actual winner
        verified = infer_verified_output_size_specs(demos)
        if not verified:
            continue

        winner_mode = verified[0].mode
        winner_family = SIZE_MODE_FAMILIES.get(winner_mode, "other")

        # Check if winner would be preserved
        winner_preserved = winner_family not in families_to_prune

        # Count candidates pruned and estimate time saved
        candidates_skipped = 0
        time_saved = 0.0
        for cand_name, cand_family in candidate_name_to_family.items():
            if cand_family in families_to_prune:
                candidates_skipped += 1
                time_saved += candidate_times.get(cand_name, 0)

        for fam in families_to_prune:
            pruned_family_counts[fam] += 1

        results.append(PruningShadowResult(
            task_id=task_id,
            n_candidates_default=len(verified),
            n_candidates_pruned=max(0, len(verified) - candidates_skipped),
            candidates_skipped=candidates_skipped,
            time_saved_ms=time_saved * 1000,
            winner_preserved=winner_preserved,
            pruned_families=tuple(sorted(families_to_prune)),
        ))

    n = len(results)
    n_preserved = sum(1 for r in results if r.winner_preserved)
    n_lost = n - n_preserved

    return PruningReport(
        n_tasks=len(task_demo_pairs),
        n_evaluated=n,
        n_winner_preserved=n_preserved,
        n_winner_lost=n_lost,
        safety_rate=n_preserved / max(n, 1),
        total_time_saved_ms=sum(r.time_saved_ms for r in results),
        avg_time_saved_ms=float(np.mean([r.time_saved_ms for r in results])) if results else 0,
        avg_candidates_skipped=float(np.mean([r.candidates_skipped for r in results])) if results else 0,
        pruned_family_counts=dict(pruned_family_counts),
    ), results


def _estimate_candidate_times(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> dict[str, float]:
    """Estimate mean time per candidate function."""
    candidate_fns = _get_candidate_fns()
    times: dict[str, list[float]] = defaultdict(list)

    for _, demos in task_demo_pairs:
        for name, fn in candidate_fns:
            t0 = time.perf_counter()
            fn(demos)
            t1 = time.perf_counter()
            times[name].append(t1 - t0)

    return {name: float(np.mean(t)) for name, t in times.items()}
