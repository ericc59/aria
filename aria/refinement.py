"""Verifier-driven refinement loop over the typed DSL search space."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aria.local_policy import LocalPolicy

from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.decompose import DecompPlan, decompose_task
from aria.hypotheses import SkeletonResult, check_skeleton_hypotheses
from aria.observe import ObservationSynthesisResult, observe_and_synthesize
from aria.synthesize import SynthesisResult, synthesize_from_observations
from aria.mutation import Mutation, mutate_program
from aria.offline_search import (
    SearchTraceEntry,
    build_literal_pool,
    excluded_ops_from_signatures,
    search_program,
)
from aria.program_store import ProgramStore
from aria.retrieval import AbstractionHint, preferred_ops_from_hints, retrieve_abstractions
from aria.runtime.program import program_to_text
from aria.scoring import CandidateScore, score_program
from aria.types import DemoPair, Literal, Program, Type


_MARKER_GEOMETRY_OPS = frozenset({
    "background_obj",
    "box_region",
    "by_color",
    "by_size_rank",
    "center_x",
    "center_y",
    "chebyshev_distance",
    "connected_components",
    "count",
    "fill_region",
    "find_objects",
    "from_object",
    "get_color",
    "manhattan_distance",
    "nearest_to",
    "nth",
    "overlay",
    "singleton",
    "square_region",
    "where",
})

_SIZE_OPS = frozenset({
    "cols_of",
    "dims_make",
    "dims_of",
    "embed",
    "new_grid",
    "predict_dims",
    "rows_of",
    "scale_dims",
    "stack_h",
    "stack_v",
    "tile_grid",
    "upscale_grid",
})

_COLOR_MAP_OPS = frozenset({
    "apply_color_map",
    "by_color",
    "extract_map",
    "find_objects",
    "infer_map",
    "recolor",
    "replace_pattern",
    "where",
})


@dataclass(frozen=True)
class RefinementFeedback:
    dominant_error_type: str | None
    dimension_mismatch_count: int
    pixel_mismatch_count: int
    execution_error_count: int
    suggested_focus: str
    task_signatures: tuple[str, ...]
    best_candidate_num: int | None = None
    best_candidate_score: float | None = None
    best_candidate_error_type: str | None = None
    best_candidate_dims_match: bool | None = None
    best_candidate_pixel_diff_count: int | None = None
    best_candidate_wrong_row_count: int | None = None
    best_candidate_wrong_col_count: int | None = None
    best_candidate_palette_expected_coverage: float | None = None
    best_candidate_palette_precision: float | None = None
    best_candidate_preserved_input_ratio: float | None = None
    best_candidate_changed_cells_ratio: float | None = None


@dataclass(frozen=True)
class RefinementPlan:
    name: str
    max_steps: int
    max_candidates: int
    allowed_ops: frozenset[str] | None = None


@dataclass(frozen=True)
class RefinementRoundResult:
    plan: RefinementPlan
    solved: bool
    winning_program: Program | None
    candidates_tried: int
    feedback: RefinementFeedback
    trace: tuple[SearchTraceEntry, ...]
    policy_source: str = "heuristic"  # "heuristic" | "local_policy"


@dataclass(frozen=True)
class BeamTransition:
    round_idx: int
    rank: int
    edit_kind: str
    detail: str
    parent_program_text: str
    child_program_text: str
    parent_score: dict[str, Any]
    child_score: dict[str, Any]
    improved: bool


@dataclass(frozen=True)
class BeamRoundSummary:
    round_idx: int
    candidates_scored: int
    best_score: dict[str, Any] | None
    best_program_text: str | None
    mutations_tried: int
    improvements: int


@dataclass(frozen=True)
class BeamRefinementResult:
    solved: bool
    winning_program: Program | None
    candidates_scored: int
    rounds_completed: int
    best_score: dict[str, Any] | None
    best_program_text: str | None
    round_summaries: tuple[BeamRoundSummary, ...]
    transitions: tuple[BeamTransition, ...]


@dataclass(frozen=True)
class RefinementResult:
    solved: bool
    winning_program: Program | None
    candidates_tried: int
    rounds: tuple[RefinementRoundResult, ...]
    beam: BeamRefinementResult | None = None
    abstraction_hints: tuple[AbstractionHint, ...] = ()
    skeleton_result: SkeletonResult | None = None
    decomposition: DecompPlan | None = None
    synthesis_result: SynthesisResult | None = None


class HeuristicRefinementPolicy:
    """Small hand-built policy that can later be replaced by a learned model."""

    def next_plan(
        self,
        *,
        round_index: int,
        base_max_steps: int,
        base_max_candidates: int,
        task_signatures: frozenset[str],
        prior_rounds: tuple[RefinementRoundResult, ...],
    ) -> RefinementPlan:
        if round_index == 0:
            return RefinementPlan(
                name="generic",
                max_steps=base_max_steps,
                max_candidates=base_max_candidates,
            )

        last_feedback = prior_rounds[-1].feedback if prior_rounds else summarize_trace_feedback(
            task_signatures,
            (),
        )

        if last_feedback.suggested_focus == "marker_geometry":
            return RefinementPlan(
                name="marker_geometry",
                max_steps=max(base_max_steps + 1, 4),
                max_candidates=base_max_candidates,
                allowed_ops=_MARKER_GEOMETRY_OPS,
            )

        if last_feedback.suggested_focus == "size":
            return RefinementPlan(
                name="size",
                max_steps=max(base_max_steps, 3),
                max_candidates=base_max_candidates,
                allowed_ops=_SIZE_OPS,
            )

        if last_feedback.suggested_focus == "color_map":
            return RefinementPlan(
                name="color_map",
                max_steps=max(base_max_steps + 1, 4),
                max_candidates=base_max_candidates,
                allowed_ops=_COLOR_MAP_OPS,
            )

        return RefinementPlan(
            name=f"generic_expand_{round_index}",
            max_steps=base_max_steps + 1,
            max_candidates=base_max_candidates,
        )


def build_policy_input(
    task_signatures: frozenset[str],
    round_index: int,
    prior_rounds: tuple[RefinementRoundResult, ...],
) -> "PolicyInput":
    """Build a PolicyInput from existing refinement state."""
    from aria.local_policy import PolicyInput

    prior_focuses = tuple(r.feedback.suggested_focus for r in prior_rounds)
    prior_error_types = tuple(r.feedback.dominant_error_type for r in prior_rounds)

    last_fb = prior_rounds[-1].feedback if prior_rounds else None

    return PolicyInput(
        task_signatures=tuple(sorted(task_signatures)),
        round_index=round_index,
        prior_focuses=prior_focuses,
        prior_error_types=prior_error_types,
        best_candidate_score=last_fb.best_candidate_score if last_fb else None,
        best_candidate_error_type=last_fb.best_candidate_error_type if last_fb else None,
        best_candidate_dims_match=last_fb.best_candidate_dims_match if last_fb else None,
        best_candidate_pixel_diff_count=last_fb.best_candidate_pixel_diff_count if last_fb else None,
        best_candidate_wrong_row_count=last_fb.best_candidate_wrong_row_count if last_fb else None,
        best_candidate_wrong_col_count=last_fb.best_candidate_wrong_col_count if last_fb else None,
        best_candidate_palette_expected_coverage=last_fb.best_candidate_palette_expected_coverage if last_fb else None,
        best_candidate_palette_precision=last_fb.best_candidate_palette_precision if last_fb else None,
        best_candidate_preserved_input_ratio=last_fb.best_candidate_preserved_input_ratio if last_fb else None,
        best_candidate_changed_cells_ratio=last_fb.best_candidate_changed_cells_ratio if last_fb else None,
    )


def focus_to_plan(
    focus: str,
    *,
    round_index: int,
    base_max_steps: int,
    base_max_candidates: int,
) -> RefinementPlan:
    """Map a focus string to a RefinementPlan using the canonical op subsets."""
    if focus == "marker_geometry":
        return RefinementPlan(
            name="marker_geometry",
            max_steps=max(base_max_steps + 1, 4),
            max_candidates=base_max_candidates,
            allowed_ops=_MARKER_GEOMETRY_OPS,
        )
    if focus == "size":
        return RefinementPlan(
            name="size",
            max_steps=max(base_max_steps, 3),
            max_candidates=base_max_candidates,
            allowed_ops=_SIZE_OPS,
        )
    if focus == "color_map":
        return RefinementPlan(
            name="color_map",
            max_steps=max(base_max_steps + 1, 4),
            max_candidates=base_max_candidates,
            allowed_ops=_COLOR_MAP_OPS,
        )
    return RefinementPlan(
        name=f"generic_expand_{round_index}" if round_index > 0 else "generic",
        max_steps=base_max_steps + 1 if round_index > 0 else base_max_steps,
        max_candidates=base_max_candidates,
    )


def run_refinement_loop(
    demos: tuple[DemoPair, ...],
    library: Library,
    *,
    program_store: ProgramStore | None = None,
    max_steps: int = 3,
    max_candidates: int = 5000,
    max_rounds: int = 2,
    include_core_ops: bool = True,
    policy: HeuristicRefinementPolicy | None = None,
    local_policy: LocalPolicy | None = None,
    beam_width: int = 0,
    beam_rounds: int = 3,
    beam_mutations_per_candidate: int = 30,
) -> RefinementResult:
    task_signatures = compute_task_signatures(demos)
    refinement_policy = policy or HeuristicRefinementPolicy()
    rounds: list[RefinementRoundResult] = []
    attempted_plans: set[tuple[str, int, int, tuple[str, ...]]] = set()
    total_candidates = 0

    # Perception: decompose the task from input/output analysis
    decomposition = decompose_task(demos)

    # Phase 0a: direct observation synthesis — read the answer from the data
    synthesis_result = synthesize_from_observations(demos)
    total_candidates += synthesis_result.candidates_tested
    if synthesis_result.solved:
        return RefinementResult(
            solved=True,
            winning_program=synthesis_result.winning_program,
            candidates_tried=total_candidates,
            rounds=(),
            decomposition=decomposition,
            synthesis_result=synthesis_result,
        )

    # Phase 0b: per-object observation — find rules from entity-level changes
    obs_result = observe_and_synthesize(demos)
    total_candidates += obs_result.candidates_tested
    if obs_result.solved:
        return RefinementResult(
            solved=True,
            winning_program=obs_result.winning_program,
            candidates_tried=total_candidates,
            rounds=(),
            decomposition=decomposition,
            synthesis_result=synthesis_result,
        )

    # Retrieve strong abstractions as search guidance
    hints = tuple(retrieve_abstractions(demos, library))
    preferred = preferred_ops_from_hints(hints)

    # Merge decomposition candidates into preferred ops
    if decomposition.sub_goals:
        decomp_ops = decomposition.candidate_op_names()
        preferred = preferred | decomp_ops

    # Derive hard op exclusions from task signatures
    excluded = excluded_ops_from_signatures(task_signatures)

    # Hypothesis testing: try skeleton programs before enumeration
    skeleton_result = check_skeleton_hypotheses(demos, hints, library)
    total_candidates += skeleton_result.skeletons_tested
    if skeleton_result.solved:
        return RefinementResult(
            solved=True,
            winning_program=skeleton_result.winning_program,
            candidates_tried=total_candidates,
            rounds=(),
            abstraction_hints=hints,
            skeleton_result=skeleton_result,
            decomposition=decomposition,
            synthesis_result=synthesis_result,
        )

    # Round 0: decomposition-constrained search
    # Use decomposition ops as hard allowed_ops to stay focused.
    # Falls through to generic rounds if this doesn't solve.
    decomp_ops = decomposition.candidate_op_names()
    if decomp_ops and len(decomp_ops) < 80:
        traces_d: list[SearchTraceEntry] = []
        decomp_result = search_program(
            demos,
            library,
            program_store=program_store,
            max_steps=max_steps,
            max_candidates=max_candidates // 3,
            include_core_ops=include_core_ops,
            allowed_ops=frozenset(decomp_ops),
            preferred_ops=preferred,
            excluded_ops=excluded,
            depth_preferred_fn=decomposition.ops_for_depth,
            observer=lambda e: traces_d.append(e),
        )
        total_candidates += decomp_result.candidates_tried
        scored = score_trace(task_signatures, tuple(traces_d))
        feedback = summarize_trace_feedback(task_signatures, scored)
        rounds.append(RefinementRoundResult(
            plan=RefinementPlan(
                name="decomposition",
                max_steps=max_steps,
                max_candidates=max_candidates // 3,
                allowed_ops=frozenset(decomp_ops),
            ),
            solved=decomp_result.solved,
            winning_program=decomp_result.winning_program,
            candidates_tried=decomp_result.candidates_tried,
            feedback=feedback,
            trace=scored,
            policy_source="decomposition",
        ))
        if decomp_result.solved:
            return RefinementResult(
                solved=True,
                winning_program=decomp_result.winning_program,
                candidates_tried=total_candidates,
                rounds=tuple(rounds),
                abstraction_hints=hints,
                skeleton_result=skeleton_result,
                decomposition=decomposition,
                synthesis_result=synthesis_result,
            )

    for round_index in range(max_rounds):
        policy_source = "heuristic"

        if local_policy is not None and round_index > 0:
            policy_input = build_policy_input(
                task_signatures, round_index, tuple(rounds),
            )
            prediction = local_policy.predict_next_focus(policy_input)
            plan = focus_to_plan(
                prediction.focus,
                round_index=round_index,
                base_max_steps=max_steps,
                base_max_candidates=max_candidates,
            )
            policy_source = "local_policy"
        else:
            plan = refinement_policy.next_plan(
                round_index=round_index,
                base_max_steps=max_steps,
                base_max_candidates=max_candidates,
                task_signatures=task_signatures,
                prior_rounds=tuple(rounds),
            )
        plan_key = (
            plan.name,
            plan.max_steps,
            plan.max_candidates,
            tuple(sorted(plan.allowed_ops or ())),
        )
        if plan_key in attempted_plans:
            break
        attempted_plans.add(plan_key)

        traces: list[SearchTraceEntry] = []

        def observer(entry: SearchTraceEntry) -> None:
            traces.append(entry)

        result = search_program(
            demos,
            library,
            program_store=program_store,
            max_steps=plan.max_steps,
            max_candidates=plan.max_candidates,
            include_core_ops=include_core_ops,
            allowed_ops=plan.allowed_ops,
            preferred_ops=preferred,
            excluded_ops=excluded,
            depth_preferred_fn=decomposition.ops_for_depth,
            observer=observer,
        )

        total_candidates += result.candidates_tried
        scored_trace = score_trace(task_signatures, tuple(traces))
        feedback = summarize_trace_feedback(task_signatures, scored_trace)
        round_result = RefinementRoundResult(
            plan=plan,
            solved=result.solved,
            winning_program=result.winning_program,
            candidates_tried=result.candidates_tried,
            feedback=feedback,
            trace=scored_trace,
            policy_source=policy_source,
        )
        rounds.append(round_result)

        if result.solved:
            return RefinementResult(
                solved=True,
                winning_program=result.winning_program,
                candidates_tried=total_candidates,
                rounds=tuple(rounds),
                abstraction_hints=hints,
                skeleton_result=skeleton_result,
                decomposition=decomposition,
                synthesis_result=synthesis_result,
            )

    # Beam refinement phase
    beam_result = None
    if beam_width > 0:
        seed_programs = _extract_seed_programs(rounds)
        if seed_programs:
            literal_pool = build_literal_pool(demos)
            beam_result = run_beam_refinement(
                demos,
                seed_programs,
                literal_pool,
                beam_width=beam_width,
                max_rounds=beam_rounds,
                max_mutations_per_candidate=beam_mutations_per_candidate,
            )
            total_candidates += beam_result.candidates_scored
            if beam_result.solved:
                return RefinementResult(
                    solved=True,
                    winning_program=beam_result.winning_program,
                    candidates_tried=total_candidates,
                    rounds=tuple(rounds),
                    beam=beam_result,
                    abstraction_hints=hints,
                    skeleton_result=skeleton_result,
                    decomposition=decomposition,
                    synthesis_result=synthesis_result,
                )

    return RefinementResult(
        solved=False,
        winning_program=None,
        candidates_tried=total_candidates,
        rounds=tuple(rounds),
        beam=beam_result,
        abstraction_hints=hints,
        skeleton_result=skeleton_result,
        decomposition=decomposition,
        synthesis_result=synthesis_result,
    )


def summarize_trace_feedback(
    task_signatures: frozenset[str],
    trace: tuple[SearchTraceEntry, ...],
) -> RefinementFeedback:
    error_counts = Counter(entry.error_type for entry in trace if entry.error_type)
    best_entry = _best_scored_entry(trace)
    dimension_mismatch_count = sum(1 for entry in trace if _is_dimension_mismatch(entry))
    execution_error_count = sum(1 for entry in trace if entry.error_type == "execution_error")
    pixel_mismatch_count = sum(
        1
        for entry in trace
        if entry.error_type == "wrong_output" and not _is_dimension_mismatch(entry)
    )

    same_size_task = "dims:same" in task_signatures
    marker_additive = (
        "change:additive" in task_signatures
        and "role:has_marker" in task_signatures
        and same_size_task
    )
    color_transform_task = (
        "color:new_in_output" in task_signatures
        or "color:palette_subset" in task_signatures
    )

    if marker_additive and (pixel_mismatch_count > 0 or color_transform_task):
        focus = "marker_geometry"
    elif same_size_task and color_transform_task and pixel_mismatch_count > 0:
        focus = "color_map"
    elif not same_size_task and dimension_mismatch_count > max(pixel_mismatch_count, execution_error_count):
        focus = "size"
    else:
        focus = "generic"

    return RefinementFeedback(
        dominant_error_type=(
            best_entry.error_type
            if best_entry is not None and best_entry.error_type is not None
            else error_counts.most_common(1)[0][0] if error_counts else None
        ),
        dimension_mismatch_count=dimension_mismatch_count,
        pixel_mismatch_count=pixel_mismatch_count,
        execution_error_count=execution_error_count,
        suggested_focus=focus,
        task_signatures=tuple(sorted(task_signatures)),
        best_candidate_num=best_entry.candidate_num if best_entry is not None else None,
        best_candidate_score=best_entry.score if best_entry is not None else None,
        best_candidate_error_type=best_entry.error_type if best_entry is not None else None,
        best_candidate_dims_match=(
            not _is_dimension_mismatch(best_entry)
            if best_entry is not None
            else None
        ),
        best_candidate_pixel_diff_count=_pixel_diff_count(best_entry) if best_entry is not None else None,
        best_candidate_wrong_row_count=_wrong_row_count(best_entry) if best_entry is not None else None,
        best_candidate_wrong_col_count=_wrong_col_count(best_entry) if best_entry is not None else None,
        best_candidate_palette_expected_coverage=(
            _float_diff(best_entry, "palette_expected_coverage")
            if best_entry is not None
            else None
        ),
        best_candidate_palette_precision=(
            _float_diff(best_entry, "palette_precision")
            if best_entry is not None
            else None
        ),
        best_candidate_preserved_input_ratio=(
            _float_diff(best_entry, "preserved_input_ratio")
            if best_entry is not None
            else None
        ),
        best_candidate_changed_cells_ratio=(
            _float_diff(best_entry, "changed_cells_ratio")
            if best_entry is not None
            else None
        ),
    )


def _is_dimension_mismatch(entry: SearchTraceEntry) -> bool:
    if entry.diff is None:
        return False
    summary = str(entry.diff.get("pixel_diff_summary", ""))
    if "dimension mismatch" in summary:
        return True
    expected_dims = entry.diff.get("expected_dims")
    actual_dims = entry.diff.get("actual_dims")
    return expected_dims is not None and actual_dims is not None and expected_dims != actual_dims


def score_trace(
    task_signatures: frozenset[str],
    trace: tuple[SearchTraceEntry, ...],
) -> tuple[SearchTraceEntry, ...]:
    return tuple(
        replace(entry, score=score, score_reasons=reasons)
        for entry, score, reasons in (
            (entry, *_score_entry(task_signatures, entry))
            for entry in trace
        )
    )


def _score_entry(
    task_signatures: frozenset[str],
    entry: SearchTraceEntry,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []

    if entry.passed:
        return 1_000_000.0, ("exact_pass",)

    if entry.error_type == "execution_error":
        return -10_000.0, ("execution_error",)

    if _is_dimension_mismatch(entry):
        dim_distance = _dimension_distance(entry)
        score = -500.0 - (25.0 * dim_distance)
        reasons.append("dimension_mismatch")
        reasons.append(f"dim_distance={dim_distance}")
        return score, tuple(reasons)

    score = 250.0
    reasons.append("dims_match")

    pixel_diff_count = _pixel_diff_count(entry)
    if pixel_diff_count is not None:
        score += max(0.0, 200.0 - min(float(pixel_diff_count), 200.0))
        reasons.append(f"pixel_diff={pixel_diff_count}")

    wrong_row_count = _wrong_row_count(entry)
    if wrong_row_count is not None:
        score -= 10.0 * wrong_row_count
        reasons.append(f"wrong_rows={wrong_row_count}")

    wrong_col_count = _wrong_col_count(entry)
    if wrong_col_count is not None:
        score -= 6.0 * wrong_col_count
        reasons.append(f"wrong_cols={wrong_col_count}")

    palette_expected_coverage = _float_diff(entry, "palette_expected_coverage")
    if palette_expected_coverage is not None:
        score += 60.0 * palette_expected_coverage
        reasons.append(f"palette_cov={palette_expected_coverage:.2f}")

    palette_precision = _float_diff(entry, "palette_precision")
    if palette_precision is not None:
        score += 40.0 * palette_precision
        reasons.append(f"palette_prec={palette_precision:.2f}")

    preserved_input_ratio = _float_diff(entry, "preserved_input_ratio")
    if preserved_input_ratio is not None:
        score += 40.0 * preserved_input_ratio
        reasons.append(f"preserve={preserved_input_ratio:.2f}")

    changed_cells_ratio = _float_diff(entry, "changed_cells_ratio")
    if changed_cells_ratio is not None:
        score += 80.0 * changed_cells_ratio
        reasons.append(f"changed={changed_cells_ratio:.2f}")

    if "change:additive" in task_signatures:
        score += 10.0
        reasons.append("additive_task")
    if "color:new_in_output" in task_signatures:
        score += 10.0
        reasons.append("new_color_task")

    return score, tuple(reasons)


def _best_scored_entry(trace: tuple[SearchTraceEntry, ...]) -> SearchTraceEntry | None:
    scored = [entry for entry in trace if entry.score is not None]
    if not scored:
        return None
    return max(
        scored,
        key=lambda entry: (
            entry.score,
            entry.passed,
            not _is_dimension_mismatch(entry),
            -(_pixel_diff_count(entry) or 10**9),
            -entry.depth,
            -entry.candidate_num,
        ),
    )


def _dimension_distance(entry: SearchTraceEntry) -> int:
    if entry.diff is None:
        return 0
    expected_dims = entry.diff.get("expected_dims")
    actual_dims = entry.diff.get("actual_dims")
    if expected_dims is None or actual_dims is None:
        return 0
    return sum(abs(int(a) - int(b)) for a, b in zip(expected_dims, actual_dims))


def _pixel_diff_count(entry: SearchTraceEntry) -> int | None:
    if entry.diff is None:
        return None
    value = entry.diff.get("pixel_diff_count")
    return int(value) if value is not None else None


def _wrong_row_count(entry: SearchTraceEntry) -> int | None:
    if entry.diff is None:
        return None
    rows = entry.diff.get("wrong_rows")
    if rows is None:
        return None
    return len(rows)


def _wrong_col_count(entry: SearchTraceEntry) -> int | None:
    if entry.diff is None:
        return None
    cols = entry.diff.get("wrong_cols")
    if cols is None:
        return None
    return len(cols)


def _float_diff(entry: SearchTraceEntry, key: str) -> float | None:
    if entry.diff is None:
        return None
    value = entry.diff.get(key)
    return float(value) if value is not None else None


# ---------------------------------------------------------------------------
# Beam refinement
# ---------------------------------------------------------------------------


def run_beam_refinement(
    demos: tuple[DemoPair, ...],
    seed_programs: list[Program],
    literal_pool: dict[Type, tuple[Literal, ...]],
    *,
    beam_width: int = 8,
    max_rounds: int = 3,
    max_mutations_per_candidate: int = 30,
) -> BeamRefinementResult:
    """Beam search: score candidates, mutate the best, keep a ranked beam."""
    beam: list[tuple[CandidateScore, Program]] = []
    for prog in seed_programs:
        sc = score_program(prog, demos)
        if sc.passed:
            return _beam_result_solved(prog, 1, [], [])
        beam.append((sc, prog))

    beam.sort(key=lambda x: x[0].rank_key)
    beam = beam[:beam_width]

    transitions: list[BeamTransition] = []
    round_summaries: list[BeamRoundSummary] = []
    total_scored = len(beam)
    seen_texts: set[str] = {program_to_text(p) for _, p in beam}

    for round_idx in range(max_rounds):
        mutations_tried = 0
        improvements = 0
        next_beam: list[tuple[CandidateScore, Program]] = list(beam)

        for rank, (parent_score, parent_prog) in enumerate(beam):
            muts = mutate_program(
                parent_prog, literal_pool,
                max_mutations=max_mutations_per_candidate,
            )
            parent_text = program_to_text(parent_prog)
            for mut in muts:
                child_text = program_to_text(mut.program)
                if child_text in seen_texts:
                    continue
                seen_texts.add(child_text)

                child_score = score_program(mut.program, demos)
                total_scored += 1
                mutations_tried += 1
                improved = child_score < parent_score

                if improved:
                    improvements += 1

                transitions.append(BeamTransition(
                    round_idx=round_idx,
                    rank=rank,
                    edit_kind=mut.edit_kind,
                    detail=mut.detail,
                    parent_program_text=parent_text,
                    child_program_text=child_text,
                    parent_score=parent_score.to_dict(),
                    child_score=child_score.to_dict(),
                    improved=improved,
                ))

                if child_score.passed:
                    round_summaries.append(BeamRoundSummary(
                        round_idx=round_idx,
                        candidates_scored=mutations_tried,
                        best_score=child_score.to_dict(),
                        best_program_text=child_text,
                        mutations_tried=mutations_tried,
                        improvements=improvements,
                    ))
                    return _beam_result_solved(
                        mut.program, total_scored,
                        round_summaries, transitions,
                    )

                next_beam.append((child_score, mut.program))

        next_beam.sort(key=lambda x: x[0].rank_key)
        beam = next_beam[:beam_width]

        best_sc, best_prog = beam[0] if beam else (None, None)
        round_summaries.append(BeamRoundSummary(
            round_idx=round_idx,
            candidates_scored=mutations_tried,
            best_score=best_sc.to_dict() if best_sc else None,
            best_program_text=program_to_text(best_prog) if best_prog else None,
            mutations_tried=mutations_tried,
            improvements=improvements,
        ))

    best_sc, best_prog = beam[0] if beam else (None, None)
    return BeamRefinementResult(
        solved=False,
        winning_program=None,
        candidates_scored=total_scored,
        rounds_completed=len(round_summaries),
        best_score=best_sc.to_dict() if best_sc else None,
        best_program_text=program_to_text(best_prog) if best_prog else None,
        round_summaries=tuple(round_summaries),
        transitions=tuple(transitions),
    )


def _beam_result_solved(
    program: Program,
    total_scored: int,
    round_summaries: list[BeamRoundSummary],
    transitions: list[BeamTransition],
) -> BeamRefinementResult:
    text = program_to_text(program)
    return BeamRefinementResult(
        solved=True,
        winning_program=program,
        candidates_scored=total_scored,
        rounds_completed=len(round_summaries),
        best_score={"passed": True},
        best_program_text=text,
        round_summaries=tuple(round_summaries),
        transitions=tuple(transitions),
    )


def _extract_seed_programs(
    rounds: list[RefinementRoundResult],
    max_seeds: int = 16,
) -> list[Program]:
    """Extract best-scored programs from search traces to seed beam refinement."""
    from aria.proposer.parser import ParseError, parse_program

    all_entries: list[SearchTraceEntry] = []
    for round_result in rounds:
        all_entries.extend(round_result.trace)

    candidates = [
        e for e in all_entries
        if e.score is not None and e.error_type != "execution_error" and not e.passed
    ]
    candidates.sort(key=lambda e: -(e.score or float("-inf")))

    programs: list[Program] = []
    seen: set[str] = set()
    for entry in candidates[: max_seeds * 3]:
        if entry.program_text in seen:
            continue
        seen.add(entry.program_text)
        try:
            prog = parse_program(entry.program_text)
            programs.append(prog)
            if len(programs) >= max_seeds:
                break
        except ParseError:
            continue

    return programs
