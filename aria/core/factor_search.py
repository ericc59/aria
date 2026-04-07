"""Bounded factor-composition search.

Given demos, predicts/enumerates factor combinations, instantiates
skeleton programs, prunes with consensus, and verifies exactly.

Integrates into scene_solve as Family 15 (additive).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aria.consensus import check_factor_consistency
from aria.consensus_trace import ConsensusTrace
from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.core.scene_executor import execute_scene_program
from aria.factor_instantiate import instantiate_factor_set
from aria.factors import FactorSet
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def factor_composition_search(
    demos: tuple[DemoPair, ...],
    *,
    proposer: object | None = None,
    max_candidates: int = 50,
    max_programs: int = 200,
    consensus_trace: ConsensusTrace | None = None,
    retrieval_guidance: object | None = None,
) -> list[tuple[FactorSet, object]]:
    """Run bounded factor-composition search.

    1. Extract features from demos
    2. Predict/enumerate top-K factor combos
    3. Filter by compatibility + consensus
    4. Instantiate skeleton programs
    5. Verify exactly against all demos

    Returns list of (FactorSet, SceneProgram) for verified programs.
    """
    if not demos:
        return []

    perceptions = tuple(perceive_grid(d.input) for d in demos)

    # --- Get factor candidates ---
    if proposer is not None and hasattr(proposer, 'top_k_factor_sets'):
        from aria.core.guidance_proposer import extract_cross_demo_features
        features = extract_cross_demo_features(demos)
        ranked = proposer.top_k_factor_sets(features, k=max_candidates)
        factor_candidates = [fs for fs, _ in ranked]
    else:
        from aria.core.factor_proposer import uniform_factor_ranking
        factor_candidates = uniform_factor_ranking(max_combos=max_candidates)

    # --- Consensus filter ---
    surviving: list[FactorSet] = []
    for fs in factor_candidates:
        passed, score, detail = check_factor_consistency(perceptions, fs)
        if not passed:
            if consensus_trace is not None:
                _trace_prune(consensus_trace, fs, detail)
            continue
        surviving.append(fs)

    # --- Retrieval bias: reorder surviving candidates ---
    if retrieval_guidance is not None and surviving:
        surviving = _reorder_by_retrieval(surviving, retrieval_guidance)

    # --- Instantiate + verify ---
    verified: list[tuple[FactorSet, object]] = []
    programs_tried = 0

    for fs in surviving:
        if programs_tried >= max_programs:
            break

        programs = instantiate_factor_set(fs, demos, perceptions)
        for prog in programs:
            if programs_tried >= max_programs:
                break
            programs_tried += 1

            if _verify_all_demos(prog, demos):
                verified.append((fs, prog))
                if consensus_trace is not None:
                    _trace_verified(consensus_trace, fs)
                # Found a verify for this factor set — move to next
                break

    # --- Predicate-parameterized dispatch (if no verify yet) ---
    if not verified and programs_tried < max_programs:
        try:
            from aria.predicate_dispatch import build_predicate_dispatch_programs
            pred_progs = build_predicate_dispatch_programs(
                demos, perceptions,
                max_predicates=10, max_actions=5,
            )
            for prog in pred_progs:
                if programs_tried >= max_programs:
                    break
                programs_tried += 1
                if _verify_all_demos(prog, demos):
                    # Use a synthetic FactorSet for predicate-dispatch solves
                    from aria.factors import (
                        Decomposition, Selector, Scope, Op,
                        Correspondence, Depth, FactorSet as FS,
                    )
                    pred_fs = FS(
                        Decomposition.OBJECT, Selector.OBJECT_SELECT,
                        Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.TWO,
                    )
                    verified.append((pred_fs, prog))
                    if consensus_trace is not None:
                        _trace_verified(consensus_trace, pred_fs)
                    break
        except Exception:
            pass

    # --- Correspondence search (if still no verify) ---
    if not verified:
        try:
            from aria.correspondence import correspondence_search
            corr_results = correspondence_search(demos, max_programs=50)
            if corr_results:
                from aria.factors import (
                    Decomposition, Selector, Scope, Op,
                    Correspondence as Corr, Depth, FactorSet as FS,
                )
                corr_fs = FS(
                    Decomposition.OBJECT, Selector.OBJECT_SELECT,
                    Scope.OBJECT, Op.RECOLOR, Corr.OBJECT_MATCH, Depth.TWO,
                )
                # correspondence_search returns (description, CorrespondenceProgram)
                desc, corr_prog = corr_results[0]
                verified.append((corr_fs, corr_prog))
        except Exception:
            pass

    return verified


# ---------------------------------------------------------------------------
# Retrieval-biased reordering
# ---------------------------------------------------------------------------


def _reorder_by_retrieval(
    candidates: list[FactorSet],
    guidance: object,
) -> list[FactorSet]:
    """Reorder factor candidates so retrieval-preferred factors come first.

    Scores each candidate by how many of its factor values appear in
    the guidance's preferred lists. Higher-scoring candidates are tried first.
    This is a bias, not a filter — all candidates are preserved.
    """
    pref_decomp = set(getattr(guidance, "preferred_decomposition_types", ()))
    pref_sel = set(getattr(guidance, "preferred_selector_families", ()))
    pref_op = set(getattr(guidance, "preferred_op_families", ()))
    pref_corr = set(getattr(guidance, "preferred_correspondences", ()))

    if not (pref_decomp or pref_sel or pref_op or pref_corr):
        return candidates

    def _boost(fs: FactorSet) -> int:
        score = 0
        if fs.decomposition.value in pref_decomp:
            score += 4
        if fs.selector.value in pref_sel:
            score += 2
        if fs.op.value in pref_op:
            score += 3
        if fs.correspondence.value in pref_corr:
            score += 1
        return score

    # Stable sort: boosted first, original order preserved within ties
    return sorted(candidates, key=lambda fs: -_boost(fs))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_all_demos(prog: object, demos: tuple[DemoPair, ...]) -> bool:
    """Verify a scene program on all demos."""
    for d in demos:
        try:
            result = execute_scene_program(prog, d.input)
            if result.shape != d.output.shape:
                return False
            if not np.array_equal(result, d.output):
                return False
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def _trace_prune(trace: ConsensusTrace, fs: FactorSet, detail: str) -> None:
    """Record a factor-level prune in the consensus trace."""
    from aria.consensus import BranchState, SharedHypothesis
    bs = BranchState(
        branch_id=f"factor:{fs}",
        step_index=0,
        per_demo=(),
        hypothesis=SharedHypothesis(
            rule_family=f"{fs.decomposition.value}->{fs.op.value}",
            scope_family=fs.scope.value,
            selector_family=fs.selector.value,
        ),
        consistency_score=0.0,
        pruned=True,
        prune_reason=detail,
    )
    trace.record(bs, f"factor prune: {fs}")


def _trace_verified(trace: ConsensusTrace, fs: FactorSet) -> None:
    """Record a factor-level verification in the consensus trace."""
    from aria.consensus import BranchState, SharedHypothesis
    bs = BranchState(
        branch_id=f"factor:{fs}",
        step_index=0,
        per_demo=(),
        hypothesis=SharedHypothesis(
            rule_family=f"{fs.decomposition.value}->{fs.op.value}",
            scope_family=fs.scope.value,
            selector_family=fs.selector.value,
        ),
        consistency_score=1.0,
    )
    trace.record(bs, f"FACTOR VERIFIED: {fs}")
