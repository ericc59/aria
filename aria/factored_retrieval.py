"""Factored retrieval — retrieve abstraction records to bias search.

Builds perception keys from demos, scores stored records against query
features, and aggregates guidance (preferred factors) for the search stack.
All scoring is deterministic. Retrieval biases search, never bypasses
symbolic verification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.factored_memory import (
    FactoredMemoryStore,
    FactoredRecord,
    PerceptionKey,
    RepairPath,
    _compute_record_id,
)
from aria.graph.signatures import compute_task_signatures
from aria.types import DemoPair, Program


# ---------------------------------------------------------------------------
# Perception key construction from demos
# ---------------------------------------------------------------------------


def perception_key_from_demos(demos: tuple[DemoPair, ...]) -> PerceptionKey:
    """Build a compact perception key from task demos.

    Reuses compute_task_signatures for structural tags, then compacts
    into the PerceptionKey format for efficient matching.
    """
    if not demos:
        return _empty_perception_key()

    sigs = compute_task_signatures(demos)

    # dims_relation
    if "dims:same" in sigs:
        dims_relation = "same"
    elif "size:grow" in sigs:
        dims_relation = "grow"
    elif "size:shrink" in sigs:
        dims_relation = "shrink"
    else:
        dims_relation = "reshape"

    # palette_relation
    if "color:palette_same" in sigs:
        palette_relation = "same"
    elif "color:palette_subset" in sigs:
        palette_relation = "subset"
    elif "color:new_in_output" in sigs:
        palette_relation = "new_colors"
    else:
        palette_relation = "other"

    # change_type
    if "change:additive" in sigs:
        change_type = "additive"
    elif "change:bg_preserved" in sigs:
        change_type = "bg_preserved"
    else:
        change_type = "dense"

    # object_count_bucket
    if "obj:none" in sigs:
        obj_bucket = "none"
    elif "obj:single" in sigs:
        obj_bucket = "single"
    elif "obj:few" in sigs:
        obj_bucket = "few"
    elif "obj:many" in sigs:
        obj_bucket = "many"
    else:
        obj_bucket = "few"  # default

    # structural booleans
    has_partition = any(s.startswith("partition:") for s in sigs)
    has_frame = "role:has_frame" in sigs
    has_marker = "role:has_marker" in sigs
    has_legend = "legend:present" in sigs

    # symmetry tags
    symmetry_tags = tuple(sorted(s for s in sigs if s.startswith("sym:")))

    # color_count_bucket
    if "color:few_colors" in sigs:
        color_bucket = "few"
    elif "color:many_colors" in sigs:
        color_bucket = "many"
    else:
        color_bucket = "medium"

    # partition_cell_count
    partition_cell_count: int | None = None
    for s in sigs:
        if s.startswith("partition:cell_rows_"):
            try:
                rows = int(s.split("_")[-1])
                cols_tag = next(
                    (t for t in sigs if t.startswith("partition:cell_cols_")),
                    None,
                )
                cols = int(cols_tag.split("_")[-1]) if cols_tag else rows
                partition_cell_count = rows * cols
            except (ValueError, AttributeError):
                pass

    return PerceptionKey(
        dims_relation=dims_relation,
        palette_relation=palette_relation,
        change_type=change_type,
        object_count_bucket=obj_bucket,
        has_partition=has_partition,
        has_frame=has_frame,
        has_marker=has_marker,
        has_legend=has_legend,
        symmetry_tags=symmetry_tags,
        color_count_bucket=color_bucket,
        partition_cell_count=partition_cell_count,
    )


def _empty_perception_key() -> PerceptionKey:
    return PerceptionKey(
        dims_relation="same",
        palette_relation="other",
        change_type="dense",
        object_count_bucket="none",
        has_partition=False,
        has_frame=False,
        has_marker=False,
        has_legend=False,
        symmetry_tags=(),
        color_count_bucket="few",
        partition_cell_count=None,
    )


# ---------------------------------------------------------------------------
# Retrieval match
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalMatch:
    """A single retrieved record with score and match reasons."""

    record: FactoredRecord
    score: float
    match_reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Retrieval guidance — aggregated from top-k matches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalGuidance:
    """Aggregated guidance from top-k retrieval matches."""

    preferred_decomposition_types: tuple[str, ...]
    preferred_selector_families: tuple[str, ...]
    preferred_scope_families: tuple[str, ...]
    preferred_op_families: tuple[str, ...]
    preferred_correspondences: tuple[str, ...]
    candidate_repair_paths: tuple[RepairPath, ...]
    source_matches: tuple[RetrievalMatch, ...]

    def preferred_ops_as_set(self) -> frozenset[str]:
        """Op family names as a frozenset for merging into preferred_ops."""
        return frozenset(self.preferred_op_families)

    def make_decomp_ranker(self):
        """Build a DecompRanker callback that prioritizes retrieved decomp types.

        Maps factored record decomposition_type names to sketch_fit view names
        and boosts their rank. Returns None if no decomposition preferences.
        """
        if not self.preferred_decomposition_types:
            return None

        # Map factored record decomp types → sketch_fit view names
        _DECOMP_TO_VIEW = {
            "framed": "framed_regions",
            "framed_periodic": "framed_regions",
            "object": "composites",
            "composite_role": "composites",
            "partition_cell": "composites",
            "canvas": "composites",
            "flat": "composites",
        }
        preferred_views: list[str] = []
        for dt in self.preferred_decomposition_types:
            view = _DECOMP_TO_VIEW.get(dt)
            if view and view not in preferred_views:
                preferred_views.append(view)

        if not preferred_views:
            return None

        preferred_set = set(preferred_views)

        def _retrieval_decomp_ranker(inp, view_names):
            # Boost views that retrieval prefers, keep others in original order
            boosted = [i for i, v in enumerate(view_names) if v in preferred_set]
            rest = [i for i, v in enumerate(view_names) if v not in preferred_set]
            return tuple(boosted + rest)

        _retrieval_decomp_ranker.__name__ = "retrieval_decomp_ranker"
        return _retrieval_decomp_ranker

    def suggested_mutation_priority(self) -> tuple[str, ...]:
        """Return mutation edit_kinds that retrieved repair paths suggest trying first.

        Maps repair_kind and mutations_applied to mutate_program edit_kind names.
        """
        if not self.candidate_repair_paths:
            return ()

        # Collect mutation kinds from repair paths, weighted by frequency
        kinds: Counter[str] = Counter()
        for rp in self.candidate_repair_paths:
            # Map repair mutations to edit_kind names used by mutate_program
            for mut in rp.mutations_applied:
                kinds[mut] += 1
            # Map repair_kind to broad edit strategies
            if rp.repair_kind == "beam_mutation":
                kinds["replace_op"] += 1
                kinds["replace_literal"] += 1
            elif rp.repair_kind == "edit_search":
                kinds["replace_op"] += 1
            elif rp.repair_kind == "sketch_recompile":
                kinds["wrap_output"] += 1

        return tuple(k for k, _ in kinds.most_common())

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_decomposition_types": list(self.preferred_decomposition_types),
            "preferred_selector_families": list(self.preferred_selector_families),
            "preferred_scope_families": list(self.preferred_scope_families),
            "preferred_op_families": list(self.preferred_op_families),
            "preferred_correspondences": list(self.preferred_correspondences),
            "candidate_repair_paths": len(self.candidate_repair_paths),
            "source_match_count": len(self.source_matches),
            "top_score": self.source_matches[0].score if self.source_matches else 0.0,
        }


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


def retrieve_factored(
    demos: tuple[DemoPair, ...],
    store: FactoredMemoryStore,
    *,
    max_results: int = 20,
) -> list[RetrievalMatch]:
    """Retrieve and rank factored records by relevance to a task.

    Scoring is deterministic: perception key overlap is primary,
    task signature Jaccard is secondary, with bonuses for transfer
    strength and composition simplicity.
    """
    if not store or len(store) == 0:
        return []

    query_key = perception_key_from_demos(demos)
    query_sigs = compute_task_signatures(demos)

    matches: list[RetrievalMatch] = []
    for record in store.all_records():
        score, reasons = _score_record(record, query_key, query_sigs)
        if score > 0:
            matches.append(RetrievalMatch(
                record=record,
                score=score,
                match_reasons=tuple(reasons),
            ))

    matches.sort(key=lambda m: (-m.score, m.record.record_id))
    return matches[:max_results]


def _score_record(
    record: FactoredRecord,
    query_key: PerceptionKey,
    query_sigs: frozenset[str],
) -> tuple[float, list[str]]:
    """Score a single record against query features."""
    score = 0.0
    reasons: list[str] = []

    # 1. Perception key overlap (primary signal, 0-12+)
    overlap = query_key.field_overlap(record.perception_key)
    if overlap > 0:
        score += 3.0 * overlap
        reasons.append(f"perception:{overlap}_fields")

    # 2. Task signature Jaccard (secondary)
    record_sigs = set(record.task_signatures)
    if record_sigs and query_sigs:
        intersection = len(record_sigs & query_sigs)
        union = len(record_sigs | query_sigs)
        jaccard = intersection / union if union > 0 else 0.0
        if intersection > 0:
            score += 15.0 * jaccard
            reasons.append(f"sig:{intersection}_overlap")

    # 3. Transfer strength (multi-task evidence)
    if record.distinct_task_count >= 2:
        bonus = min(record.distinct_task_count, 5) * 2.0
        score += bonus
        reasons.append(f"transfer:{record.distinct_task_count}_tasks")

    # 4. Composition simplicity (prefer lower depth)
    if record.composition_depth <= 2:
        score += max(0.0, 3.0 - record.composition_depth)
        reasons.append(f"depth:{record.composition_depth}")

    # 5. Verified bonus
    if record.verified:
        score += 2.0

    # 6. Has repair path bonus (useful for near-miss guidance)
    if record.repair_path is not None:
        score += 1.0
        reasons.append("has_repair_path")

    return score, reasons


# ---------------------------------------------------------------------------
# Guidance aggregation
# ---------------------------------------------------------------------------


def aggregate_guidance(matches: list[RetrievalMatch]) -> RetrievalGuidance:
    """Aggregate top-k matches into ranked factor preferences.

    Each match votes for its factor families, weighted by score.
    Higher-scoring matches contribute more to the ranking.
    """
    if not matches:
        return RetrievalGuidance(
            preferred_decomposition_types=(),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )

    decomp_votes: Counter[str] = Counter()
    selector_votes: Counter[str] = Counter()
    scope_votes: Counter[str] = Counter()
    op_votes: Counter[str] = Counter()
    corr_votes: Counter[str] = Counter()
    repair_paths: list[RepairPath] = []

    for m in matches:
        w = m.score
        r = m.record
        if r.decomposition_type != "unknown":
            decomp_votes[r.decomposition_type] += w
        if r.selector_family != "unknown":
            selector_votes[r.selector_family] += w
        if r.scope_family != "unknown":
            scope_votes[r.scope_family] += w
        if r.op_family != "unknown":
            op_votes[r.op_family] += w
        if r.correspondence != "none":
            corr_votes[r.correspondence] += w
        if r.repair_path is not None:
            repair_paths.append(r.repair_path)

    return RetrievalGuidance(
        preferred_decomposition_types=_ranked_keys(decomp_votes),
        preferred_selector_families=_ranked_keys(selector_votes),
        preferred_scope_families=_ranked_keys(scope_votes),
        preferred_op_families=_ranked_keys(op_votes),
        preferred_correspondences=_ranked_keys(corr_votes),
        candidate_repair_paths=tuple(repair_paths),
        source_matches=tuple(matches),
    )


def _ranked_keys(votes: Counter[str]) -> tuple[str, ...]:
    """Return keys ranked by vote weight, deterministic tie-break."""
    return tuple(k for k, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Factor extraction from solved programs
# ---------------------------------------------------------------------------

# Op name -> op family mapping
_OP_TO_FAMILY: dict[str, str] = {
    "recolor": "recolor",
    "replace_pattern": "recolor",
    "apply_color_map": "color_map",
    "infer_map": "color_map",
    "extract_map": "color_map",
    "fill_region": "fill",
    "fill_enclosed": "fill",
    "flood_fill": "fill",
    "overlay": "stamp",
    "embed": "stamp",
    "stamp": "stamp",
    "rot90": "transform",
    "rot180": "transform",
    "rot270": "transform",
    "flip_h": "transform",
    "flip_v": "transform",
    "transpose": "transform",
    "scale_grid": "scale",
    "upscale_grid": "scale",
    "tile_grid": "tile",
    "stack_h": "compose",
    "stack_v": "compose",
    "crop": "extract",
    "subgrid": "extract",
    "from_object": "extract",
}

# Op name -> selector family mapping
_OP_TO_SELECTOR: dict[str, str] = {
    "by_color": "by_color",
    "by_size_rank": "by_size_rank",
    "by_shape": "by_shape",
    "where": "where_predicate",
    "singleton": "singleton",
    "nth": "nth",
    "nearest_to": "nearest_to",
    "largest": "by_size_rank",
    "smallest": "by_size_rank",
}

# Op name -> scope family mapping
_OP_TO_SCOPE: dict[str, str] = {
    "find_objects": "objects",
    "connected_components": "objects",
    "extract_zones": "zones",
    "partition_cells": "partition_cell",
    "detect_framed": "frame_interior",
    "box_region": "frame_interior",
}


def extract_factors_from_program(
    program: Program,
    *,
    source: str = "search",
) -> dict[str, str]:
    """Heuristically extract factor families from a solved Program.

    Returns a dict with keys: decomposition_type, selector_family,
    scope_family, op_family, correspondence.
    """
    op_names: list[str] = []
    for step in program.steps:
        if hasattr(step, "expr") and hasattr(step.expr, "op"):
            op_names.append(step.expr.op)

    # Identify op family from the most significant op
    op_family = "unknown"
    for name in reversed(op_names):  # last op is usually the output-producing one
        if name in _OP_TO_FAMILY:
            op_family = _OP_TO_FAMILY[name]
            break

    # Identify selector family
    selector_family = "unknown"
    for name in op_names:
        if name in _OP_TO_SELECTOR:
            selector_family = _OP_TO_SELECTOR[name]
            break

    # Identify scope family
    scope_family = "unknown"
    for name in op_names:
        if name in _OP_TO_SCOPE:
            scope_family = _OP_TO_SCOPE[name]
            break

    # Decomposition type heuristic
    decomposition_type = "flat"
    if any(n in ("find_objects", "connected_components") for n in op_names):
        decomposition_type = "object"
    elif any(n.startswith("partition") for n in op_names):
        decomposition_type = "partition_cell"
    elif any(n in ("detect_framed", "box_region") for n in op_names):
        decomposition_type = "framed"
    elif any(n in ("tile_grid", "upscale_grid", "scale_grid") for n in op_names):
        decomposition_type = "canvas"

    # Correspondence heuristic
    correspondence = "none"
    if any(n in ("map_objects", "for_each") for n in op_names):
        correspondence = "1:1_object"
    elif any(n.startswith("partition") for n in op_names):
        correspondence = "cell_to_cell"

    return {
        "decomposition_type": decomposition_type,
        "selector_family": selector_family,
        "scope_family": scope_family,
        "op_family": op_family,
        "correspondence": correspondence,
    }


def ingest_solve_record(
    demos: tuple[DemoPair, ...],
    program: Program,
    *,
    task_id: str | None,
    source: str,
    verify_mode: str = "stateless",
    repair_path: RepairPath | None = None,
    factored_store: FactoredMemoryStore,
) -> FactoredRecord:
    """Extract factors from a solved program and add to the memory store."""
    factors = extract_factors_from_program(program, source=source)
    perception_key = perception_key_from_demos(demos)
    task_signatures = tuple(sorted(compute_task_signatures(demos)))

    record_id = _compute_record_id(
        decomposition_type=factors["decomposition_type"],
        selector_family=factors["selector_family"],
        scope_family=factors["scope_family"],
        op_family=factors["op_family"],
        correspondence=factors["correspondence"],
        composition_depth=len(program.steps),
        perception_key=perception_key,
    )

    record = FactoredRecord(
        record_id=record_id,
        task_ids=(task_id,) if task_id else (),
        source=source,
        decomposition_type=factors["decomposition_type"],
        selector_family=factors["selector_family"],
        scope_family=factors["scope_family"],
        op_family=factors["op_family"],
        correspondence=factors["correspondence"],
        composition_depth=len(program.steps),
        perception_key=perception_key,
        task_signatures=task_signatures,
        verified=True,
        verify_mode=verify_mode,
        repair_path=repair_path,
    )

    return factored_store.add_record(record)
