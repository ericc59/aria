"""Clause inducer: discover relational rules from demo pairs.

For each demo:
1. Perceive input and output
2. Map output objects to input sources
3. For each non-preserved output object, generate candidate clauses
   explaining its target selection, support, derivation, and action
4. Across demos: keep only clauses consistent with ALL demos

This is the core reasoning engine.
"""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import numpy as np

from aria.guided.perceive import perceive, GridFacts, ObjFact
from aria.guided.correspond import (
    ObjMapping, map_output_to_input, map_output_to_input_topk, find_removed_objects,
)
from aria.guided.clause import (
    Clause, ClauseProgram, Predicate, Pred, Agg, Act,
)
from aria.types import Grid


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def induce_program(
    demos: list[tuple[Grid, Grid]],
) -> ClauseProgram | None:
    """Induce a clause program from demo pairs.

    Tries multiple correspondence hypotheses (top-K). For each:
    1. Generate candidate clauses
    2. Score by structural correctness on ALL demos
    3. If any single clause is exact → return it
    4. Otherwise try bounded 2-clause composition
    Best verified program across all hypotheses wins.
    """
    if not demos:
        return None

    # Perceive all demos
    demo_facts = [(perceive(inp), perceive(out)) for inp, out in demos]
    bg = demo_facts[0][0].bg

    # Infer output size rule (None if same-size)
    size_rule = None
    if not all(inp.shape == out.shape for inp, out in demos):
        from aria.guided.output_size import infer_output_size
        size_rule = infer_output_size(demos)

    # Try multiple correspondence hypotheses
    corr_hypotheses = _generate_correspondence_hypotheses(demo_facts)

    best_partial_prog = None

    for demo_mappings in corr_hypotheses:
        result = _induce_under_correspondence(demo_facts, demo_mappings, demos, bg)
        if result is None:
            continue
        prog, is_exact = result
        if size_rule is not None:
            prog.size_rule = size_rule
        if is_exact:
            return prog  # exact solve — done
        if best_partial_prog is None:
            best_partial_prog = prog

    return best_partial_prog


def _generate_correspondence_hypotheses(demo_facts):
    """Generate multiple correspondence hypotheses across demos.

    Returns a list of demo_mappings, each a list[list[ObjMapping]].
    The first hypothesis is always the default (largest-first greedy).
    Generates alternatives for ALL demos, not just demo 0.
    """
    hypotheses = []
    seen = set()

    # Hypothesis 1: default correspondence for all demos
    default_mappings = []
    for in_facts, out_facts in demo_facts:
        default_mappings.append(map_output_to_input(out_facts, in_facts))
    hypotheses.append(default_mappings)

    # Generate alternatives for each demo independently
    for demo_idx, (in_facts, out_facts) in enumerate(demo_facts):
        alt_mappings = map_output_to_input_topk(out_facts, in_facts, k=3)
        for alt_m in alt_mappings[1:]:  # skip first (it's the default)
            # Build hypothesis: alt for this demo, default for others
            hyp = []
            for j, (inf, outf) in enumerate(demo_facts):
                if j == demo_idx:
                    hyp.append(alt_m)
                else:
                    hyp.append(default_mappings[j])
            # Deduplicate by mapping signature
            key = tuple(
                tuple((id(m.out_obj), id(m.in_obj) if m.in_obj else None) for m in demo_map)
                for demo_map in hyp
            )
            if key not in seen:
                seen.add(key)
                hypotheses.append(hyp)

    return hypotheses


def _induce_under_correspondence(demo_facts, demo_mappings, demos, bg):
    """Induce under a specific correspondence. Returns (prog, is_exact) or None."""
    candidate_clauses = _generate_candidates(demo_facts, demo_mappings, demos)
    candidate_clauses = _cap_candidates(candidate_clauses, 200)

    exact, ranked_partials = _score_candidates(
        candidate_clauses, demos, bg, demo_facts=demo_facts)

    # 1. Single clause exact solve — prefer strongest explanations
    if exact:
        in_facts_0 = demo_facts[0][0]
        # Sort by explanation quality: structural > coincidental
        exact.sort(key=_clause_quality, reverse=True)
        deduped = _dedup_exact_clauses(exact, in_facts_0)
        return ClauseProgram(clauses=deduped, output_bg=bg), True

    # 2. Bounded 2-clause composition
    if ranked_partials:
        composed = _compose_two_clauses(ranked_partials, demos, bg,
                                         demo_facts, demo_mappings)
        if composed:
            return composed, True

    # 2b. Bounded 3-clause: apply best 2-clause partial, induce 3rd on residual
    if ranked_partials:
        composed3 = _compose_three_clauses(ranked_partials, demos, bg,
                                            demo_facts, demo_mappings)
        if composed3:
            return composed3, True

    # 3. Best partial
    if ranked_partials:
        return ClauseProgram(clauses=[ranked_partials[0][1]], output_bg=bg), False

    return None



# ---------------------------------------------------------------------------
# 2-clause composition
# ---------------------------------------------------------------------------

_MAX_PARTIALS = 50  # max unique partial clauses to consider for pairing

def _compose_two_clauses(ranked_partials, demos, bg, demo_facts, demo_mappings):
    """Try to find a 2-clause program that exactly solves all demos.

    Two strategies:
    A. Complementary pairing: find clause pairs whose error masks are disjoint
       (each clause fixes cells the other doesn't). Uses precomputed per-demo
       error masks to avoid blind pairwise execution.
    B. Residual composition: apply clause A, induce clause B on what's left.
    """
    # Dedup partials: ensure diversity across action types.
    # Within each action type, prefer simpler predicates (fewer = more general).
    from collections import defaultdict
    by_action = defaultdict(list)
    seen_descs = set()
    for item in ranked_partials:
        diff, clause, errors = item
        if clause.description in seen_descs:
            continue
        seen_descs.add(clause.description)
        by_action[clause.action].append(item)

    # Round-robin across actions, simplest predicates first within each
    for items in by_action.values():
        items.sort(key=lambda x: len(x[1].target_preds) + len(x[1].support_preds))

    deduped = []
    idx = {a: 0 for a in by_action}
    while len(deduped) < _MAX_PARTIALS and by_action:
        for action in list(by_action.keys()):
            items = by_action[action]
            if idx[action] < len(items):
                deduped.append(items[idx[action]])
                idx[action] += 1
            else:
                del by_action[action]
            if len(deduped) >= _MAX_PARTIALS:
                break

    if not deduped:
        return None

    # Cache perception for verify calls
    cached_facts = [inf for inf, _ in demo_facts]

    # Strategy A: Complementary pairing using error masks
    # Two clauses are complementary if their error sets don't overlap much
    # and together they cover all errors.
    for i in range(len(deduped)):
        di, c1, errs1 = deduped[i]
        if not errs1 or any(e is None for e in errs1):
            continue
        for j in range(i + 1, len(deduped)):
            dj, c2, errs2 = deduped[j]
            if not errs2 or any(e is None for e in errs2):
                continue
            # Skip: same target predicates (redundant)
            if c1.target_preds == c2.target_preds:
                continue
            # Skip: both grid-level
            if not c1.target_preds and not c2.target_preds:
                continue

            # Quick check: could combining them fix everything?
            # If the union of their correct cells covers all cells, worth trying.
            could_help = True
            for k in range(len(demos)):
                if k >= len(errs1) or k >= len(errs2):
                    could_help = False
                    break
                # Both have errors in the same cells → combining won't help
                overlap = errs1[k] & errs2[k]
                if np.any(overlap):
                    # Some cells are wrong in BOTH clauses — can't fix by combining
                    # Unless one clause changes those cells correctly
                    # Only skip if overlap is large relative to total error
                    total_err = int(np.sum(errs1[k] | errs2[k]))
                    overlap_size = int(np.sum(overlap))
                    if overlap_size > total_err * 0.5:
                        could_help = False
                        break

            if not could_help:
                continue

            # Worth trying — verify exactly
            prog = ClauseProgram(clauses=[c1, c2], output_bg=bg)
            if _verify_program_cached(prog, demos, cached_facts):
                return prog
            prog_rev = ClauseProgram(clauses=[c2, c1], output_bg=bg)
            if _verify_program_cached(prog_rev, demos, cached_facts):
                return prog_rev

    # Strategy B: Residual composition
    for _, first_clause, _ in deduped:
        composed = _residual_compose(first_clause, demos, bg,
                                      demo_facts, demo_mappings)
        if composed:
            return composed

    return None


def _compose_three_clauses(ranked_partials, demos, bg, demo_facts, demo_mappings):
    """Try 3-clause programs: find the best 2-clause partial, add a 3rd clause.

    Only attempts a bounded number of 2-clause starting points.
    """
    from collections import defaultdict

    # Get diverse partials
    by_action = defaultdict(list)
    seen = set()
    for item in ranked_partials:
        diff, clause, errors = item
        if clause.description in seen:
            continue
        seen.add(clause.description)
        by_action[clause.action].append(item)

    deduped = []
    idx = {a: 0 for a in by_action}
    while len(deduped) < 15 and by_action:
        for action in list(by_action.keys()):
            items = by_action[action]
            if idx[action] < len(items):
                deduped.append(items[idx[action]])
                idx[action] += 1
            else:
                del by_action[action]
            if len(deduped) >= 15:
                break

    cached_facts = [inf for inf, _ in demo_facts]

    # Try pairs as 2-clause starting points
    tried = 0
    for i in range(len(deduped)):
        for j in range(i + 1, len(deduped)):
            if tried >= 20:
                return None
            c1, c2 = deduped[i][1], deduped[j][1]
            if c1.target_preds == c2.target_preds:
                continue

            # Try both orderings
            for clauses_2 in [[c1, c2], [c2, c1]]:
                prog2 = ClauseProgram(clauses=clauses_2, output_bg=bg)
                # Check: does 2-clause get close? (fewer errors than single best)
                total_err = 0
                intermediates = []
                for inp, out in demos:
                    try:
                        mid = prog2.execute(inp)
                        total_err += int(np.sum(mid != out))
                        intermediates.append(mid)
                    except Exception:
                        total_err = 999999
                        break

                if total_err == 0:
                    continue  # already exact — should have been caught earlier
                if total_err > 50:
                    continue  # too many errors — not worth trying 3rd clause

                tried += 1

                # Induce 3rd clause on residual
                residual_demos = list(zip(intermediates, [out for _, out in demos]))
                third = _residual_compose_single(residual_demos, bg)
                if third:
                    prog3 = ClauseProgram(clauses=clauses_2 + [third], output_bg=bg)
                    if _verify_program_cached(prog3, demos, cached_facts):
                        return prog3

    return None


def _residual_compose_single(residual_demos, bg):
    """Induce a single clause on residual demos (intermediate → expected output)."""
    from aria.guided.correspond import map_output_to_input

    demo_facts = []
    demo_mappings = []
    for mid, out in residual_demos:
        in_f = perceive(mid)
        out_f = perceive(out)
        demo_facts.append((in_f, out_f))
        demo_mappings.append(map_output_to_input(out_f, in_f))

    candidates = _generate_candidates(demo_facts, demo_mappings, residual_demos)
    candidates = _cap_candidates(candidates, 100)
    exact, _ = _score_candidates(candidates, residual_demos, bg, demo_facts=demo_facts)
    if exact:
        exact.sort(key=_clause_quality, reverse=True)
        return exact[0]
    return None


def _residual_compose(first_clause, demos, bg, demo_facts, demo_mappings):
    """Apply first_clause, induce a second clause on the residual."""
    first_prog = ClauseProgram(clauses=[first_clause], output_bg=bg)

    intermediates = []
    for inp, out in demos:
        try:
            mid = first_prog.execute(inp)
        except Exception:
            return None
        intermediates.append(mid)

    # Check there's still something to solve
    residual_demos = list(zip(intermediates, [out for _, out in demos]))
    if all(np.array_equal(m, o) for m, o in residual_demos):
        return ClauseProgram(clauses=[first_clause], output_bg=bg)

    # Re-perceive and generate candidates for the residual
    try:
        residual_facts = [(perceive(mid), perceive(out))
                          for mid, out in residual_demos]
        residual_mappings = [map_output_to_input(of, inf)
                             for inf, of in residual_facts]
        residual_candidates = _generate_candidates(
            residual_facts, residual_mappings, residual_demos)
    except Exception:
        return None

    if len(residual_candidates) > 100:
        residual_candidates = residual_candidates[:100]

    for second_clause in residual_candidates:
        prog = ClauseProgram(clauses=[first_clause, second_clause], output_bg=bg)
        if _verify_program(prog, demos):
            return prog

    return None


def _verify_program(prog, demos):
    """Exact verification: program must produce correct output on ALL demos."""
    for inp, out in demos:
        try:
            pred = prog.execute(inp)
        except Exception:
            return False
        if not np.array_equal(pred, out):
            return False
    return True


def _verify_program_cached(prog, demos, cached_facts):
    """Exact verification with cached perception facts."""
    for idx, (inp, out) in enumerate(demos):
        try:
            pred = prog.execute(inp, facts=cached_facts[idx])
        except Exception:
            return False
        if not np.array_equal(pred, out):
            return False
    return True


# ---------------------------------------------------------------------------
# Explanation quality: structural > coincidental
# ---------------------------------------------------------------------------

def _clause_quality(clause) -> float:
    """Rate a clause's explanation quality (higher = better).

    Prefers structural explanations over coincidental ones:
    - Structural actions (gravity, slide_toward) > constant offset
    - Derived color (from support) > constant color
    - General predicates > over-specified predicates
    - Fewer arbitrary constants > more
    """
    score = 0.0

    # --- Action quality ---
    action = clause.action
    if action in (Act.GRAVITY, Act.SLIDE):
        direction = getattr(clause, '_gravity_dir', '')
        if direction == 'toward_support':
            score += 30  # best: direction derived from structure
        else:
            score += 20  # good: gravity/slide but fixed direction
    elif action == Act.RECOLOR:
        if clause.aggregation == Agg.LEARNED_MAP:
            score += 25  # learned from demos — structural
        elif clause.support_preds:
            score += 20  # color from support object
        elif hasattr(clause, '_fill_color'):
            score += 5   # constant color — weakest
        else:
            score += 15
    elif action == Act.FILL_ENCLOSED:
        if clause.aggregation == Agg.FRAME_COLOR:
            score += 25  # derived from structure
        else:
            score += 10  # constant fill
    elif action == Act.FILL_INTERIOR:
        if clause.support_preds:
            score += 20
        else:
            score += 10
    elif action == Act.PLACE_AT:
        score += 5  # constant offset — weakest movement
    elif action == Act.STAMP:
        score += 15
    elif action == Act.REMOVE:
        score += 15
    elif action == Act.TRANSFORM:
        score += 15
    else:
        score += 10

    # --- Predicate quality ---
    for p in clause.target_preds:
        if p.pred == Pred.UNIQUE_COLOR:
            score += 5  # structural: only one with this color
        elif p.pred == Pred.COLOR_EQ:
            score += 1  # literal color — weaker
        elif p.pred in (Pred.IS_SMALLEST, Pred.IS_LARGEST):
            score += 4  # relative — good
        elif p.pred in (Pred.ADJACENT_TO, Pred.CONTAINED_BY, Pred.CONTAINS):
            score += 6  # relational — best
        elif p.pred == Pred.NOT:
            score += 3
        elif p.pred == Pred.SIZE_GT:
            score += 2
        else:
            score += 2

    # --- Penalty for arbitrary constants ---
    if hasattr(clause, '_offset'):
        score -= 5  # constant offset is fragile
    if hasattr(clause, '_fill_color'):
        score -= 3  # constant color is fragile

    # --- Penalty for over-specified predicates ---
    if len(clause.target_preds) > 2:
        score -= 2  # too many conditions = over-fit

    return score


# ---------------------------------------------------------------------------
# Candidate scoring — structural, not pixel-based
# ---------------------------------------------------------------------------

def _score_candidates(candidates, demos, bg, demo_facts=None):
    """Score candidates by structural correctness across demos.

    Returns:
      exact: list of clauses that individually solve all demos
      ranked_partials: list of (n_changed_wrong, clause, per_demo_errors)
        sorted by fewest wrong CHANGED objects first

    Only scores against CHANGED output objects (new, moved, recolored).
    Static/preserved objects (separators, legends, identical structures)
    are free — preserving them is the default, not an achievement.

    per_demo_errors is a list of boolean masks for complementary-pair pruning.
    """
    exact = []
    partials = []

    # Cache perception and correspondence per demo
    cached_in_facts = []
    cached_out_facts = []
    if demo_facts:
        cached_in_facts = [inf for inf, _ in demo_facts]
        cached_out_facts = [outf for _, outf in demo_facts]
    else:
        cached_in_facts = [perceive(inp) for inp, _ in demos]
        cached_out_facts = [perceive(out) for _, out in demos]

    # Precompute which output objects are CHANGED (not identical to input)
    # Only these count for scoring.
    changed_objs_per_demo = []
    for idx in range(len(demos)):
        mappings = map_output_to_input(cached_out_facts[idx], cached_in_facts[idx])
        changed = [m.out_obj for m in mappings if m.match_type != "identical"]
        changed_objs_per_demo.append(changed)

    # Baseline: total changed objects across all demos
    baseline_wrong = sum(len(ch) for ch in changed_objs_per_demo)
    if baseline_wrong == 0:
        return [], []  # nothing to explain

    for clause in candidates:
        prog = ClauseProgram(clauses=[clause], output_bg=bg)
        total_changed_wrong = 0
        all_ok = True
        per_demo_errors = []

        for idx, (inp, out) in enumerate(demos):
            try:
                pred = prog.execute(inp, facts=cached_in_facts[idx])
            except Exception:
                all_ok = False
                total_changed_wrong = baseline_wrong + 1
                per_demo_errors = []
                break

            if not np.array_equal(pred, out):
                all_ok = False
                if pred.shape == out.shape:
                    err = pred != out
                    per_demo_errors.append(err)
                    # Only count CHANGED objects that are wrong
                    for obj in changed_objs_per_demo[idx]:
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.mask[r, c]:
                                    if pred[obj.row + r, obj.col + c] != out[obj.row + r, obj.col + c]:
                                        total_changed_wrong += 1
                                        break
                            else:
                                continue
                            break
                else:
                    total_changed_wrong = baseline_wrong + 1
                    per_demo_errors.append(None)
            else:
                per_demo_errors.append(np.zeros(out.shape, dtype=bool))

        if all_ok:
            exact.append(clause)
        elif total_changed_wrong < baseline_wrong and per_demo_errors:
            partials.append((total_changed_wrong, clause, per_demo_errors))

    # Sort partials: fewer wrong objects first, quality as tiebreaker
    partials.sort(key=lambda x: (x[0], -_clause_quality(x[1])))
    return exact, partials


# ---------------------------------------------------------------------------
# Exact clause deduplication
# ---------------------------------------------------------------------------

def _dedup_exact_clauses(exact, in_facts):
    """Remove exact clauses whose targets are subsets of other exact clauses.

    For clauses with the same action and offset, prefer the one with the
    fewest predicates (broadest match). A clause selecting
    [IS_SMALLEST, UNIQUE_COLOR] is redundant if [IS_SMALLEST] also exact-solves.
    """
    if len(exact) <= 1:
        return exact

    # Group by action signature (action + offset/direction)
    from collections import defaultdict
    groups = defaultdict(list)
    for c in exact:
        key_parts = [c.action.name]
        if hasattr(c, '_offset'):
            key_parts.append(str(c._offset))
        if hasattr(c, '_gravity_dir'):
            key_parts.append(c._gravity_dir)
        key = tuple(key_parts)
        groups[key].append(c)

    result = []
    for key, clauses in groups.items():
        # Within each group, compute selected object sets
        clauses.sort(key=lambda c: len(c.target_preds))
        kept_oid_sets = []
        for c in clauses:
            targets = c.select_targets(in_facts)
            oids = frozenset(t.oid for t in targets)
            # Keep only if this oid set is not a subset of any already-kept set
            if any(oids <= kept for kept in kept_oid_sets):
                continue
            kept_oid_sets.append(oids)
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# Candidate capping with diversity
# ---------------------------------------------------------------------------

def _cap_candidates(candidates, cap):
    """Cap candidates while preserving diversity across action types.

    Ensures each action type gets a fair share of the budget rather than
    letting the most prolific generator drown out others.
    """
    if len(candidates) <= cap:
        return candidates

    from collections import defaultdict
    by_action = defaultdict(list)
    for c in candidates:
        by_action[c.action].append(c)

    # Round-robin across action types, single-pred clauses first
    result = []
    remaining = dict(by_action)
    idx = {a: 0 for a in remaining}

    # First pass: one clause per action type with fewest predicates
    for action, clauses in remaining.items():
        # Sort: fewer predicates first (simpler clauses preferred)
        clauses.sort(key=lambda c: len(c.target_preds) + len(c.support_preds))

    while len(result) < cap and remaining:
        for action in list(remaining.keys()):
            clauses = remaining[action]
            if idx[action] < len(clauses):
                result.append(clauses[idx[action]])
                idx[action] += 1
                if len(result) >= cap:
                    break
            else:
                del remaining[action]

    return result


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(demo_facts, demo_mappings, demos):
    """Generate candidate clauses from the first demo, to be filtered by others."""
    candidates = []
    in_facts, out_facts = demo_facts[0]
    mappings = demo_mappings[0]

    # Handle output objects (recolored, moved, transformed, etc.)
    for m in mappings:
        if m.match_type == "identical":
            continue

        if m.match_type == "recolored":
            clauses = _candidates_for_recolor(m, in_facts)
            candidates.extend(clauses)

        elif m.match_type in ("moved", "moved_recolored"):
            clauses = _candidates_for_move(m, in_facts)
            candidates.extend(clauses)

        elif m.match_type == "transformed" and m.transform:
            clauses = _candidates_for_transform(m, in_facts)
            candidates.extend(clauses)

    # Learned mappings: for recolor with novel output colors that vary
    # across demos, learn shape→color mapping from all demos.
    learned_clauses = _candidates_for_learned_recolor(demo_facts, demo_mappings, demos)
    candidates.extend(learned_clauses)

    # Conditional dispatch: group changed objects by action type and
    # generate per-group clauses with group-distinguishing predicates.
    # This finds rules like "color 1 moves down, color 2 moves right."
    dispatch_clauses = _candidates_for_conditional_dispatch(mappings, in_facts)
    candidates.extend(dispatch_clauses)

    # Handle region-based rules (separator → data/answer regions)
    if in_facts.separators and in_facts.regions:
        clauses = _candidates_for_region_rules(in_facts, out_facts, demos[0])
        candidates.extend(clauses)

    # Handle REMOVED objects: input objects not matched by any output object
    for in_obj in find_removed_objects(in_facts, mappings):
        clauses = _candidates_for_remove(in_obj, in_facts)
        candidates.extend(clauses)

    # Handle fill_enclosed: if input has enclosed bg regions and output fills them
    if in_facts.dim_candidates.get('n_enclosed_bg', 0) > 0:
        clauses = _candidates_for_fill_enclosed(in_facts, out_facts, demos[0])
        candidates.extend(clauses)

    # Handle per-object interior fill: input objects with bg holes that get filled
    clauses = _candidates_for_fill_interior(in_facts, demos[0])
    candidates.extend(clauses)

    # Handle NEW objects: output objects with no input source
    new_mappings = [m for m in mappings if m.match_type == "new"]
    if new_mappings:
        clauses = _candidates_for_new(new_mappings, in_facts, out_facts)
        candidates.extend(clauses)

    # Note: TILE, SELF_TILE, PERIODIC_EXTEND moved to synthesize.py
    # as grid constructors (they are whole-grid patterns, not per-object clauses)

    return candidates



def _candidates_for_recolor(m: ObjMapping, in_facts: GridFacts) -> list[Clause]:
    """Generate candidate clauses explaining why this object was recolored."""
    candidates = []
    target = m.out_obj
    source = m.in_obj
    new_color = m.color_to

    target_pred_sets = _generate_target_predicates(source, in_facts)

    # Support-derived color: another input object provides the new color
    color_providers = [o for o in in_facts.objects if o.color == new_color and o.oid != source.oid]

    for provider in color_providers:
        support_pred_sets = _generate_support_predicates(provider, in_facts)
        for t_preds in target_pred_sets:
            for s_preds in support_pred_sets:
                clause = Clause(
                    target_preds=t_preds,
                    support_preds=s_preds,
                    aggregation=Agg.COLOR_OF,
                    action=Act.RECOLOR,
                    description=f"recolor [{_desc_preds(t_preds)}] to color of [{_desc_preds(s_preds)}]",
                )
                candidates.append(clause)

    # Constant color: new color doesn't come from any input object
    if not color_providers:
        for t_preds in target_pred_sets:
            clause = Clause(
                target_preds=t_preds,
                support_preds=[],
                aggregation=Agg.COLOR_OF,
                action=Act.RECOLOR,
                description=f"recolor [{_desc_preds(t_preds)}] to constant {new_color}",
            )
            clause._fill_color = new_color
            candidates.append(clause)

    # Target is REMOVED (new_color == bg)
    if new_color == in_facts.bg:
        for t_preds in target_pred_sets:
            clause = Clause(
                target_preds=t_preds,
                support_preds=[],
                aggregation=Agg.COLOR_OF,
                action=Act.REMOVE,
                description=f"remove [{_desc_preds(t_preds)}]",
            )
            candidates.append(clause)

    return candidates


def _candidates_for_remove(in_obj: ObjFact, in_facts: GridFacts) -> list[Clause]:
    """Generate candidate clauses for an input object that disappears."""
    candidates = []
    target_pred_sets = _generate_target_predicates(in_obj, in_facts)

    for t_preds in target_pred_sets:
        clause = Clause(
            target_preds=t_preds,
            support_preds=[],
            aggregation=Agg.COLOR_OF,
            action=Act.REMOVE,
            description=f"remove [{_desc_preds(t_preds)}]",
        )
        candidates.append(clause)

    return candidates


def _candidates_for_transform(m, in_facts):
    """Generate candidate clauses for an object that was geometrically transformed in place."""
    candidates = []
    source = m.in_obj
    if source is None:
        return candidates

    target_pred_sets = _generate_target_predicates(source, in_facts)
    xform = m.transform  # flip_h, flip_v, rot90, etc.

    for t_preds in target_pred_sets:
        clause = Clause(
            target_preds=t_preds,
            support_preds=[],
            aggregation=Agg.COLOR_OF,
            action=Act.TRANSFORM,
            description=f"transform [{_desc_preds(t_preds)}] {xform}",
        )
        clause._transform = xform
        candidates.append(clause)

    return candidates


def _candidates_for_move(m: ObjMapping, in_facts: GridFacts) -> list[Clause]:
    """Generate candidate clauses for an object that moved (and possibly recolored)."""
    candidates = []
    source = m.in_obj
    if source is None:
        return candidates

    target_pred_sets = _generate_target_predicates(source, in_facts)
    out_obj = m.out_obj
    dr = out_obj.row - source.row
    dc = out_obj.col - source.col
    rows, cols = in_facts.rows, in_facts.cols

    if m.match_type == "moved":
        # Check structural movement patterns FIRST (preferred over constant offset)
        at_bottom = (out_obj.row + out_obj.height == rows)
        at_top = (out_obj.row == 0)
        at_right = (out_obj.col + out_obj.width == cols)
        at_left = (out_obj.col == 0)

        # Gravity: output position is at a grid border
        for direction, cond in [('down', at_bottom), ('up', at_top),
                                ('right', at_right), ('left', at_left)]:
            if not cond:
                continue
            for t_preds in target_pred_sets:
                clause = Clause(
                    target_preds=t_preds,
                    support_preds=[],
                    aggregation=Agg.COLOR_OF,
                    action=Act.GRAVITY,
                    description=f"gravity [{_desc_preds(t_preds)}] {direction}",
                )
                clause._gravity_dir = direction
                candidates.append(clause)

        # Slide with fixed direction
        if dr != 0 or dc != 0:
            if abs(dr) > abs(dc):
                direction = 'down' if dr > 0 else 'up'
            else:
                direction = 'right' if dc > 0 else 'left'
            for t_preds in target_pred_sets:
                clause = Clause(
                    target_preds=t_preds,
                    support_preds=[],
                    aggregation=Agg.COLOR_OF,
                    action=Act.SLIDE,
                    description=f"slide [{_desc_preds(t_preds)}] {direction}",
                )
                clause._gravity_dir = direction
                candidates.append(clause)

        # Slide toward support: direction derived from relative position
        # at execution time. Generalizes across demos with different layouts.
        # Supports cardinal AND diagonal directions.
        if dr != 0 or dc != 0:
            for other in in_facts.objects:
                if other.oid == source.oid:
                    continue
                # Check: is the movement direction TOWARD this object?
                to_other_r = other.center_row - source.center_row
                to_other_c = other.center_col - source.center_col
                # Determine direction (including diagonals)
                if abs(to_other_r) > abs(to_other_c) * 2:
                    toward_dir = 'down' if to_other_r > 0 else 'up'
                elif abs(to_other_c) > abs(to_other_r) * 2:
                    toward_dir = 'right' if to_other_c > 0 else 'left'
                else:
                    # Diagonal
                    toward_dir = ('down' if to_other_r > 0 else 'up') + ('right' if to_other_c > 0 else 'left')
                if abs(dr) > abs(dc) * 2:
                    actual_dir = 'down' if dr > 0 else 'up'
                elif abs(dc) > abs(dr) * 2:
                    actual_dir = 'right' if dc > 0 else 'left'
                else:
                    actual_dir = ('down' if dr > 0 else 'up') + ('right' if dc > 0 else 'left')
                if toward_dir != actual_dir:
                    continue
                s_pred_sets = _generate_support_predicates(other, in_facts)
                for t_preds in target_pred_sets:
                    for s_preds in s_pred_sets:
                        clause = Clause(
                            target_preds=t_preds,
                            support_preds=s_preds,
                            aggregation=Agg.COLOR_OF,
                            action=Act.SLIDE,
                            description=f"slide [{_desc_preds(t_preds)}] toward [{_desc_preds(s_preds)}]",
                        )
                        clause._gravity_dir = 'toward_support'
                        candidates.append(clause)

        # Constant offset (fallback — least preferred)
        for t_preds in target_pred_sets:
            clause = Clause(
                target_preds=t_preds,
                support_preds=[],
                aggregation=Agg.COLOR_OF,
                action=Act.PLACE_AT,
                description=f"move [{_desc_preds(t_preds)}] by ({dr},{dc})",
            )
            clause._offset = (dr, dc)
            candidates.append(clause)

    elif m.match_type == "moved_recolored":
        # Different color AND position — need support for color
        new_color = m.color_to
        color_providers = [o for o in in_facts.objects if o.color == new_color and o.oid != source.oid]

        for provider in color_providers:
            support_pred_sets = _generate_support_predicates(provider, in_facts)
            for t_preds in target_pred_sets:
                for s_preds in support_pred_sets:
                    clause = Clause(
                        target_preds=t_preds,
                        support_preds=s_preds,
                        aggregation=Agg.COLOR_OF,
                        action=Act.PLACE_AT,
                        description=f"move+recolor [{_desc_preds(t_preds)}] by ({dr},{dc}) to color of [{_desc_preds(s_preds)}]",
                    )
                    clause._offset = (dr, dc)
                    candidates.append(clause)

    return candidates


def _candidates_for_region_rules(in_facts, out_facts, demo):
    """Generate clauses for tasks with separator-defined regions."""
    candidates = []
    inp, out = demo

    answer_regions = [r for r in in_facts.regions if r.role == "answer"]
    data_regions = [r for r in in_facts.regions if r.role == "data"]

    if not answer_regions or not data_regions:
        return candidates

    ar = answer_regions[0]
    dr = data_regions[0]

    # Check: does the output have a single new pixel in the answer region?
    bg = in_facts.bg
    answer_sub_in = inp[ar.r0:ar.r1 + 1, ar.c0:ar.c1 + 1]
    answer_sub_out = out[ar.r0:ar.r1 + 1, ar.c0:ar.c1 + 1]
    diff = answer_sub_in != answer_sub_out
    n_changed = int(np.sum(diff))

    if 0 < n_changed <= 3:
        # Small number of pixels placed in answer region
        # Check if the placed color = most common color in data region
        sep_colors = {s.color for s in in_facts.separators}
        data_counts = {k: v for k, v in dr.pixel_color_counts.items() if k not in sep_colors}

        if data_counts:
            mc = max(data_counts, key=data_counts.get)
            placed_colors = set(int(out[ar.r0 + r, ar.c0 + c])
                                for r, c in zip(*np.where(diff))
                                if int(out[ar.r0 + r, ar.c0 + c]) != bg)

            if placed_colors == {mc}:
                # Rule: place most_common_color from data region into answer region
                clause = Clause(
                    target_preds=[],  # no object target — it's a region-based rule
                    support_preds=[],
                    aggregation=Agg.MOST_COMMON_COLOR,
                    action=Act.PLACE_PIXEL,
                    description="place most_common_color from data region at answer center",
                )
                candidates.append(clause)

    return candidates


def _candidates_for_fill_enclosed(in_facts, out_facts, demo):
    """Generate fill_enclosed clauses with various color derivations.

    Tries multiple color sources for filling enclosed bg regions:
    1. Frame color (adjacent non-bg color)
    2. Constant color from the output
    3. Color from a specific input object
    """
    from aria.guided.clause import _compute_frame_colors
    from collections import deque

    candidates = []
    inp, out = demo

    if inp.shape != out.shape:
        return candidates

    bg = in_facts.bg
    rows, cols = inp.shape

    # Find enclosed bg regions
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and inp[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and inp[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))

    enclosed = (inp == bg) & ~reachable
    if not np.any(enclosed):
        return candidates

    # What colors does the output place in the enclosed cells?
    fill_values = out[enclosed]
    actual_fill_colors = set(int(v) for v in np.unique(fill_values)) - {bg}
    if not actual_fill_colors:
        return candidates

    # Strategy 1: frame color (existing)
    frame_colors = _compute_frame_colors(inp, bg)
    if frame_colors:
        all_frame_match = all(np.all(out[mask] == color)
                              for mask, color in frame_colors)
        if all_frame_match:
            candidates.append(Clause(
                target_preds=[], support_preds=[],
                aggregation=Agg.FRAME_COLOR, action=Act.FILL_ENCLOSED,
                description="fill enclosed with frame color",
            ))

    # Strategy 2: single constant color fills ALL enclosed regions
    if len(actual_fill_colors) == 1:
        fill_c = next(iter(actual_fill_colors))
        if np.all(fill_values == fill_c):
            # The fill color might come from an input object
            providers = [o for o in in_facts.objects if o.color == fill_c]
            if providers:
                for provider in providers[:3]:
                    s_preds_list = _generate_support_predicates(provider, in_facts)
                    for s_preds in s_preds_list:
                        clause = Clause(
                            target_preds=[], support_preds=s_preds,
                            aggregation=Agg.COLOR_OF,
                            action=Act.FILL_ENCLOSED,
                            description=f"fill enclosed with color of [{_desc_preds(s_preds)}]",
                        )
                        candidates.append(clause)
            # Also try as literal constant (color not in input)
            clause = Clause(
                target_preds=[], support_preds=[],
                aggregation=Agg.COLOR_OF, action=Act.FILL_ENCLOSED,
                description=f"fill enclosed with constant color {fill_c}",
            )
            clause._fill_color = fill_c
            candidates.append(clause)

    return candidates


def _candidates_for_fill_interior(in_facts, demo):
    """Generate FILL_INTERIOR clauses for objects with bg holes in their mask.

    For each input object that has holes (bg cells within its bbox where
    mask is False), check if the output fills those holes with a consistent color.
    """
    candidates = []
    inp, out = demo

    if inp.shape != out.shape:
        return candidates

    bg = in_facts.bg

    for obj in in_facts.objects:
        if obj.is_rectangular:
            continue  # no holes in a solid rectangle

        # Find bg cells within this object's bbox
        hole_cells = []
        for r in range(obj.height):
            for c in range(obj.width):
                if not obj.mask[r, c]:
                    gr, gc = obj.row + r, obj.col + c
                    if gr < inp.shape[0] and gc < inp.shape[1] and inp[gr, gc] == bg:
                        hole_cells.append((gr, gc))

        if not hole_cells:
            continue

        # What color does the output place in these holes?
        fill_colors = set()
        all_filled = True
        for gr, gc in hole_cells:
            ov = int(out[gr, gc])
            if ov == bg:
                all_filled = False
                break
            fill_colors.add(ov)

        if not all_filled or not fill_colors:
            continue

        # Generate candidates for each fill color derivation
        target_preds = _generate_target_predicates(obj, in_facts)

        if len(fill_colors) == 1:
            fill_c = next(iter(fill_colors))

            # Color from a support object?
            providers = [o for o in in_facts.objects
                         if o.color == fill_c and o.oid != obj.oid]
            for provider in providers[:3]:
                s_preds_list = _generate_support_predicates(provider, in_facts)
                for t_preds in target_preds:
                    for s_preds in s_preds_list:
                        clause = Clause(
                            target_preds=t_preds, support_preds=s_preds,
                            aggregation=Agg.COLOR_OF,
                            action=Act.FILL_INTERIOR,
                            description=(
                                f"fill interior of [{_desc_preds(t_preds)}] "
                                f"with color of [{_desc_preds(s_preds)}]"),
                        )
                        candidates.append(clause)

            # Constant color
            for t_preds in target_preds:
                clause = Clause(
                    target_preds=t_preds, support_preds=[],
                    aggregation=Agg.COLOR_OF,
                    action=Act.FILL_INTERIOR,
                    description=(
                        f"fill interior of [{_desc_preds(t_preds)}] "
                        f"with constant color {fill_c}"),
                )
                clause._fill_color = fill_c
                candidates.append(clause)

    return candidates


def _candidates_for_new(new_mappings, in_facts, out_facts):
    """Generate candidate clauses for output objects with no input source.

    Covers three creation patterns:
    1. STAMP: new object's mask matches an input object → copy at offset
    2. STAMP singleton: new 1x1 pixel placed at position derived from input
    3. STAMP with recolor: mask from one input obj, color from another
    """
    candidates = []
    per_obj_cap = 30   # cap per-new-object
    global_cap = 200   # total cap across all new objects

    for m in new_mappings:
        if len(candidates) >= global_cap:
            break
        new_obj = m.out_obj
        per_obj = []

        # Pattern 1: copied mask — find input objects with the same mask
        for in_obj in in_facts.objects:
            if (in_obj.height == new_obj.height and
                in_obj.width == new_obj.width and
                np.array_equal(in_obj.mask, new_obj.mask)):
                # This input object is the template
                dr = new_obj.row - in_obj.row
                dc = new_obj.col - in_obj.col

                template_preds = _generate_target_predicates(in_obj, in_facts)

                if in_obj.color == new_obj.color:
                    # Same color: STAMP with own color
                    for t_preds in template_preds:
                        clause = Clause(
                            target_preds=t_preds,
                            support_preds=[],
                            aggregation=Agg.COLOR_OF,
                            action=Act.STAMP,
                            description=f"stamp [{_desc_preds(t_preds)}] at ({dr},{dc})",
                        )
                        clause._offset = (dr, dc)
                        per_obj.append(clause)
                else:
                    # Different color: find who provides the color
                    color_providers = [o for o in in_facts.objects
                                       if o.color == new_obj.color
                                       and o.oid != in_obj.oid]
                    for provider in color_providers:
                        support_preds = _generate_support_predicates(
                            provider, in_facts)
                        for t_preds in template_preds:
                            for s_preds in support_preds:
                                clause = Clause(
                                    target_preds=t_preds,
                                    support_preds=s_preds,
                                    aggregation=Agg.COLOR_OF,
                                    action=Act.STAMP,
                                    description=(
                                        f"stamp [{_desc_preds(t_preds)}] "
                                        f"at ({dr},{dc}) "
                                        f"color of [{_desc_preds(s_preds)}]"),
                                )
                                clause._offset = (dr, dc)
                                per_obj.append(clause)

        # Pattern 2: new singleton — 1x1 pixel at a specific position
        # The position might be relative to a specific input object
        if new_obj.size == 1 and not per_obj:
            for in_obj in in_facts.objects:
                dr = new_obj.row - in_obj.row
                dc = new_obj.col - in_obj.col
                # Only consider small offsets (within reasonable range)
                if abs(dr) > max(in_facts.rows, 15) or abs(dc) > max(in_facts.cols, 15):
                    continue
                template_preds = _generate_target_predicates(in_obj, in_facts)
                if in_obj.color == new_obj.color:
                    for t_preds in template_preds:
                        clause = Clause(
                            target_preds=t_preds,
                            support_preds=[],
                            aggregation=Agg.COLOR_OF,
                            action=Act.STAMP,
                            description=f"stamp [{_desc_preds(t_preds)}] at ({dr},{dc})",
                        )
                        clause._offset = (dr, dc)
                        per_obj.append(clause)
                else:
                    color_providers = [o for o in in_facts.objects
                                       if o.color == new_obj.color
                                       and o.oid != in_obj.oid]
                    for provider in color_providers[:2]:  # limit
                        s_pred_sets = _generate_support_predicates(
                            provider, in_facts)
                        for t_preds in template_preds:
                            for s_preds in s_pred_sets:
                                clause = Clause(
                                    target_preds=t_preds,
                                    support_preds=s_preds,
                                    aggregation=Agg.COLOR_OF,
                                    action=Act.STAMP,
                                    description=(
                                        f"stamp [{_desc_preds(t_preds)}] "
                                        f"at ({dr},{dc}) "
                                        f"color of [{_desc_preds(s_preds)}]"),
                                )
                                clause._offset = (dr, dc)
                                per_obj.append(clause)

        # Pattern 3: transformed mask — flipped/rotated version of input object
        if not per_obj:
            for in_obj in in_facts.objects:
                for xform, xfn in [
                    ('flip_h', lambda m: m[:, ::-1]),
                    ('flip_v', lambda m: m[::-1, :]),
                    ('rot180', lambda m: np.rot90(m, 2)),
                ]:
                    if in_obj.height == new_obj.height and in_obj.width == new_obj.width:
                        if np.array_equal(xfn(in_obj.mask), new_obj.mask):
                            template_preds = _generate_target_predicates(in_obj, in_facts)
                            for t_preds in template_preds:
                                clause = Clause(
                                    target_preds=t_preds, support_preds=[],
                                    aggregation=Agg.COLOR_OF, action=Act.TRANSFORM,
                                    description=f"transform [{_desc_preds(t_preds)}] {xform}",
                                )
                                clause._transform = xform
                                per_obj.append(clause)
                            break
                # rot90 / transpose swap height and width
                for xform, xfn in [
                    ('rot90', lambda m: np.rot90(m)),
                    ('transpose', lambda m: m.T),
                ]:
                    if in_obj.height == new_obj.width and in_obj.width == new_obj.height:
                        if np.array_equal(xfn(in_obj.mask), new_obj.mask):
                            template_preds = _generate_target_predicates(in_obj, in_facts)
                            for t_preds in template_preds:
                                clause = Clause(
                                    target_preds=t_preds, support_preds=[],
                                    aggregation=Agg.COLOR_OF, action=Act.TRANSFORM,
                                    description=f"transform [{_desc_preds(t_preds)}] {xform}",
                                )
                                clause._transform = xform
                                per_obj.append(clause)
                            break

        candidates.extend(per_obj[:per_obj_cap])

    return candidates


def _candidates_for_learned_recolor(demo_facts, demo_mappings, demos):
    """Generate recolor clauses where the output color is learned from demos.

    For tasks where the output color is novel (not in input) and varies
    across demos, learn a mapping from a support object's shape to the
    output color. The mapping is built from ALL demo pairs.
    """
    candidates = []

    # Check: do any demos have recolored objects with novel output colors?
    # And does the novel color vary across demos?
    novel_colors_per_demo = []
    for i, (inp, out) in enumerate(demos):
        in_colors = set(int(v) for v in np.unique(inp)) - {demo_facts[i][0].bg}
        out_colors = set(int(v) for v in np.unique(out)) - {demo_facts[i][0].bg}
        novel = out_colors - in_colors
        novel_colors_per_demo.append(novel)

    # Need novel colors in at least 2 demos, and they must vary
    all_novel = set()
    for nc in novel_colors_per_demo:
        all_novel |= nc
    if len(all_novel) < 2:
        return candidates

    # For each demo: find the recolored object and a "key" object whose shape varies
    # Try to build: key_shape → output_color mapping
    in_facts_0 = demo_facts[0][0]
    mappings_0 = demo_mappings[0]

    # Find recolored objects in demo 0
    recolored = [m for m in mappings_0 if m.match_type == 'recolored' and m.in_obj]
    if not recolored:
        return candidates

    # For each non-recolored, non-identical object: could it be the key?
    recolored_oids = {m.in_obj.oid for m in recolored}
    key_candidates = [o for o in in_facts_0.objects if o.oid not in recolored_oids]

    for key_obj_0 in key_candidates:
        key_preds = _generate_support_predicates(key_obj_0, in_facts_0)

        for k_preds in key_preds:
            # Build the shape→color mapping across all demos
            learned_map = {}
            consistent = True

            for i, (inp, out) in enumerate(demos):
                in_f = demo_facts[i][0]
                # Find the key object in this demo using the same predicates
                keys = [o for o in in_f.objects
                        if all(p.test(o, in_f.objects) for p in k_preds)]
                if not keys:
                    consistent = False
                    break

                key = keys[0]
                shape_key = (key.height, key.width, key.mask.tobytes())

                # What's the novel output color in this demo?
                novel = novel_colors_per_demo[i]
                if len(novel) != 1:
                    consistent = False
                    break

                color = next(iter(novel))

                # Check consistency: same shape must map to same color
                if shape_key in learned_map and learned_map[shape_key] != color:
                    consistent = False
                    break
                learned_map[shape_key] = color

            if not consistent or not learned_map:
                continue

            # Generate clause: recolor [target] to LEARNED_MAP(key_shape)
            for m in recolored:
                t_preds_list = _generate_target_predicates(m.in_obj, in_facts_0)
                for t_preds in t_preds_list:
                    clause = Clause(
                        target_preds=t_preds,
                        support_preds=k_preds,
                        aggregation=Agg.LEARNED_MAP,
                        action=Act.RECOLOR,
                        description=(
                            f"recolor [{_desc_preds(t_preds)}] "
                            f"by learned map from [{_desc_preds(k_preds)}]"),
                    )
                    clause._learned_map = learned_map
                    candidates.append(clause)

    # Strategy 2: color→color mapping (global color permutation)
    # Build mapping: for each input color, what output color does it become?
    # Must be consistent across all demos.
    color_map = {}
    color_consistent = True
    for i, (inp, out) in enumerate(demos):
        if inp.shape != out.shape:
            color_consistent = False
            break
        for c in range(10):
            in_mask = inp == c
            if not np.any(in_mask):
                continue
            out_vals = out[in_mask]
            uniq = set(int(v) for v in np.unique(out_vals))
            if len(uniq) != 1:
                continue  # mixed mapping for this color
            mapped = next(iter(uniq))
            if mapped == c:
                continue  # unchanged
            if c in color_map and color_map[c] != mapped:
                color_consistent = False
                break
            color_map[c] = mapped
        if not color_consistent:
            break

    if color_consistent and len(color_map) >= 2:
        # All recolored objects share this global permutation
        # Generate one clause that applies the learned color map to all objects
        for t_preds in [[Predicate(Pred.SIZE_GT, 0)]]:  # targets all objects
            clause = Clause(
                target_preds=t_preds,
                support_preds=[],
                aggregation=Agg.LEARNED_MAP,
                action=Act.RECOLOR,
                description=f"recolor all by color map {color_map}",
            )
            clause._learned_map = color_map
            clause._learned_key_type = 'color'
            candidates.append(clause)

    return candidates


def _candidates_for_conditional_dispatch(mappings, in_facts):
    """Generate clauses for tasks where different objects get different actions.

    Groups changed objects by their transformation signature (action type +
    key parameters like offset direction or new color), then finds predicates
    that distinguish the groups. Emits per-group clauses.

    Example: if objects of color 1 moved down and objects of color 2 moved
    right, emits:
      - gravity [COLOR_EQ(1)] down
      - gravity [COLOR_EQ(2)] right
    """
    from collections import defaultdict

    candidates = []

    # Build transformation signature for each changed object
    changed = []
    for m in mappings:
        if m.match_type == "identical" or m.in_obj is None:
            continue
        src = m.in_obj
        out = m.out_obj
        dr = out.row - src.row
        dc = out.col - src.col

        if m.match_type == "recolored":
            sig = ('recolor', m.color_to)
        elif m.match_type == "moved":
            # Direction, not exact offset
            if abs(dr) > abs(dc):
                direction = 'down' if dr > 0 else 'up'
            elif abs(dc) > abs(dr):
                direction = 'right' if dc > 0 else 'left'
            else:
                direction = f'{dr},{dc}'
            sig = ('move', direction)
        elif m.match_type == "moved_recolored":
            if abs(dr) > abs(dc):
                direction = 'down' if dr > 0 else 'up'
            else:
                direction = 'right' if dc > 0 else 'left'
            sig = ('move_recolor', direction, m.color_to)
        else:
            continue
        changed.append((sig, src, m))

    if len(changed) < 2:
        return candidates

    # Group by signature
    groups = defaultdict(list)
    for sig, src, m in changed:
        groups[sig].append((src, m))

    # Only interesting if there are 2+ distinct groups
    if len(groups) < 2:
        return candidates

    # For each group, find predicates that select THIS group but not others
    all_srcs = [src for _, src, _ in changed]

    for sig, members in groups.items():
        group_srcs = [src for src, _ in members]
        other_srcs = [s for s in all_srcs if s not in group_srcs]

        if not other_srcs:
            continue

        # Find predicates true for ALL group members but NOT for others
        group_preds = _find_distinguishing_predicates(
            group_srcs, other_srcs, in_facts)

        if not group_preds:
            continue

        # Generate clauses for this group's action
        action_type, *params = sig
        for preds in group_preds:
            if action_type == 'recolor':
                new_color = params[0]
                color_providers = [o for o in in_facts.objects if o.color == new_color]
                for provider in color_providers[:2]:
                    for s_preds in _generate_support_predicates(provider, in_facts):
                        clause = Clause(
                            target_preds=preds, support_preds=s_preds,
                            aggregation=Agg.COLOR_OF, action=Act.RECOLOR,
                            description=f"recolor [{_desc_preds(preds)}] to color of [{_desc_preds(s_preds)}]",
                        )
                        candidates.append(clause)

            elif action_type == 'move':
                direction = params[0]
                if direction in ('down', 'up', 'left', 'right'):
                    # Gravity
                    clause = Clause(
                        target_preds=preds, support_preds=[],
                        aggregation=Agg.COLOR_OF, action=Act.GRAVITY,
                        description=f"gravity [{_desc_preds(preds)}] {direction}",
                    )
                    clause._gravity_dir = direction
                    candidates.append(clause)
                    # Slide
                    clause = Clause(
                        target_preds=preds, support_preds=[],
                        aggregation=Agg.COLOR_OF, action=Act.SLIDE,
                        description=f"slide [{_desc_preds(preds)}] {direction}",
                    )
                    clause._gravity_dir = direction
                    candidates.append(clause)

            elif action_type == 'move_recolor':
                direction = params[0]
                new_color = params[1]
                color_providers = [o for o in in_facts.objects if o.color == new_color]
                if direction in ('down', 'up', 'left', 'right'):
                    for provider in color_providers[:2]:
                        for s_preds in _generate_support_predicates(provider, in_facts):
                            clause = Clause(
                                target_preds=preds, support_preds=s_preds,
                                aggregation=Agg.COLOR_OF, action=Act.GRAVITY,
                                description=f"gravity [{_desc_preds(preds)}] {direction} color of [{_desc_preds(s_preds)}]",
                            )
                            clause._gravity_dir = direction
                            candidates.append(clause)

    return candidates


def _find_distinguishing_predicates(group_objs, other_objs, facts):
    """Find predicate sets that select all group_objs but no other_objs."""
    results = []

    # Try single predicates first
    candidate_preds = [
        Predicate(Pred.COLOR_EQ, group_objs[0].color),
        Predicate(Pred.IS_SMALLEST),
        Predicate(Pred.IS_LARGEST),
        Predicate(Pred.UNIQUE_COLOR),
        Predicate(Pred.IS_SINGLETON),
        Predicate(Pred.IS_RECTANGULAR),
        Predicate(Pred.IS_LINE),
        Predicate(Pred.NOT_TOUCHES_BORDER),
        Predicate(Pred.SIZE_GT, 1),
        Predicate(Pred.NOT, Predicate(Pred.IS_LARGEST)),
        Predicate(Pred.NOT, Predicate(Pred.IS_SMALLEST)),
        Predicate(Pred.NOT, Predicate(Pred.IS_LINE)),
    ]

    all_objs = facts.objects
    for p in candidate_preds:
        # Must be true for ALL group members
        if not all(p.test(o, all_objs) for o in group_objs):
            continue
        # Must be false for ALL others
        if any(p.test(o, all_objs) for o in other_objs):
            continue
        results.append([p])

    # Try color-based if group shares a color
    group_colors = set(o.color for o in group_objs)
    if len(group_colors) == 1:
        c = next(iter(group_colors))
        other_colors = set(o.color for o in other_objs)
        if c not in other_colors:
            results.append([Predicate(Pred.COLOR_EQ, c)])

    return results[:5]  # cap


def _generate_target_predicates(obj: ObjFact, facts: GridFacts) -> list[list[Predicate]]:
    """Generate different ways to SELECT this object.

    Returns single predicates, 2-predicate conjunctions, and relational
    predicates for higher selectivity.
    """
    from aria.guided.clause import _are_adjacent, _is_contained

    # Build single predicates that are true for this object
    singles = []

    if obj.size == min(o.size for o in facts.objects):
        singles.append(Predicate(Pred.IS_SMALLEST))
    if obj.size == max(o.size for o in facts.objects):
        singles.append(Predicate(Pred.IS_LARGEST))
    singles.append(Predicate(Pred.COLOR_EQ, obj.color))
    if obj.n_same_color == 1:
        singles.append(Predicate(Pred.UNIQUE_COLOR))
    if obj.size == 1:
        singles.append(Predicate(Pred.IS_SINGLETON))
    if obj.is_rectangular and not obj.is_line:
        singles.append(Predicate(Pred.IS_RECTANGULAR))
    if obj.is_line:
        singles.append(Predicate(Pred.IS_LINE))
    if not (obj.touches_top or obj.touches_bottom or obj.touches_left or obj.touches_right):
        singles.append(Predicate(Pred.NOT_TOUCHES_BORDER))
    # Note: positional predicates (IS_TOPMOST etc.) are available in the Pred enum
    # and used by selector search / dispatch, but NOT generated here in clause induction
    # to avoid candidate proliferation that breaks bounded 2-clause composition.
    singles.append(Predicate(Pred.SIZE_GT, 1))

    # Relational predicates: check relationships to other objects
    # Only emit if the relation is discriminative (doesn't match all objects)
    _rel_selectors = []
    # "adjacent to a singleton"
    adj_singletons = [o for o in facts.objects
                      if o.oid != obj.oid and o.size == 1 and _are_adjacent(obj, o)]
    if adj_singletons:
        _rel_selectors.append(
            Predicate(Pred.ADJACENT_TO, Predicate(Pred.IS_SINGLETON)))
    # "adjacent to the largest"
    largest = max(facts.objects, key=lambda o: o.size)
    if largest.oid != obj.oid and _are_adjacent(obj, largest):
        _rel_selectors.append(
            Predicate(Pred.ADJACENT_TO, Predicate(Pred.IS_LARGEST)))
    # "contained by the largest"
    if largest.oid != obj.oid and _is_contained(obj, largest):
        _rel_selectors.append(
            Predicate(Pred.CONTAINED_BY, Predicate(Pred.IS_LARGEST)))
    # "same shape as another object"
    same_shape = [o for o in facts.objects
                  if o.oid != obj.oid and o.height == obj.height
                  and o.width == obj.width and np.array_equal(o.mask, obj.mask)]
    if same_shape and len(same_shape) < len(facts.objects) - 1:
        # Only useful if not ALL objects share the same shape
        singles.append(Predicate(Pred.SAME_SHAPE_AS,
                                  Predicate(Pred.COLOR_EQ, same_shape[0].color)))

    singles.extend(_rel_selectors)

    # NOT combinator: for predicates that are FALSE for this object but
    # TRUE for other objects (helps select "everything except X")
    not_preds = []
    if obj.size != max(o.size for o in facts.objects):
        not_preds.append(Predicate(Pred.NOT, Predicate(Pred.IS_LARGEST)))
    if obj.size != min(o.size for o in facts.objects):
        not_preds.append(Predicate(Pred.NOT, Predicate(Pred.IS_SMALLEST)))
    if obj.n_same_color != 1:
        not_preds.append(Predicate(Pred.NOT, Predicate(Pred.UNIQUE_COLOR)))
    if not obj.is_line:
        not_preds.append(Predicate(Pred.NOT, Predicate(Pred.IS_LINE)))
    singles.extend(not_preds)

    # Single-predicate sets
    pred_sets = [[p] for p in singles]

    # 2-predicate conjunctions from different categories
    _size_preds = {Pred.IS_SMALLEST, Pred.IS_LARGEST, Pred.SIZE_GT, Pred.IS_SINGLETON}
    _color_preds = {Pred.COLOR_EQ, Pred.UNIQUE_COLOR}
    _shape_preds = {Pred.IS_RECTANGULAR, Pred.IS_LINE, Pred.SAME_SHAPE_AS}
    _pos_preds = {Pred.NOT_TOUCHES_BORDER, Pred.TOUCHES_BORDER,
                   Pred.IS_TOPMOST, Pred.IS_BOTTOMMOST, Pred.IS_LEFTMOST, Pred.IS_RIGHTMOST}
    _rel_preds = {Pred.ADJACENT_TO, Pred.CONTAINED_BY, Pred.CONTAINS}
    _neg_preds = {Pred.NOT}

    def _category(p):
        if p.pred in _size_preds: return 'size'
        if p.pred in _color_preds: return 'color'
        if p.pred in _shape_preds: return 'shape'
        if p.pred in _pos_preds: return 'pos'
        if p.pred in _rel_preds: return 'rel'
        if p.pred in _neg_preds: return 'neg'
        return 'other'

    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            if _category(singles[i]) != _category(singles[j]):
                pred_sets.append([singles[i], singles[j]])
            if len(pred_sets) > 40:
                break
        if len(pred_sets) > 40:
            break

    return pred_sets


def _generate_support_predicates(obj: ObjFact, facts: GridFacts) -> list[list[Predicate]]:
    """Generate different ways to FIND this support object."""
    pred_sets = []

    if obj.size == min(o.size for o in facts.objects):
        pred_sets.append([Predicate(Pred.IS_SMALLEST)])
    if obj.size == max(o.size for o in facts.objects):
        pred_sets.append([Predicate(Pred.IS_LARGEST)])
    if obj.size == 1:
        pred_sets.append([Predicate(Pred.IS_SINGLETON)])
    if obj.n_same_color == 1:
        pred_sets.append([Predicate(Pred.UNIQUE_COLOR)])

    pred_sets.append([Predicate(Pred.COLOR_EQ, obj.color)])

    return pred_sets


def _desc_preds(preds):
    return " & ".join(f"{p.pred.name}({p.param})" if p.param is not None else p.pred.name
                       for p in preds)


