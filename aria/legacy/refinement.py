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
from aria.explanation import RepairResult, attempt_program_repair, attempt_repair
from aria.sketch_compile import (
    CompileFailure,
    CompilePerDemoPrograms,
    CompileResult,
    CompileTaskProgram,
    compile_sketch,
    compile_sketch_graph,
)
from aria.structural_edit import EditResult, structural_edit_search
from aria.hypotheses import SkeletonResult, check_skeleton_hypotheses
from aria.observe import (
    DimsReconstructionResult,
    ObservationSynthesisResult,
    dims_change_reconstruct,
    observe_and_synthesize,
)
from aria.synthesize import SynthesisResult, synthesize_from_observations
from aria.mutation import Mutation, mutate_program
from aria.offline_search import (
    SearchTraceEntry,
    build_literal_pool,
    excluded_ops_from_signatures,
    search_program,
)
from aria.factored_memory import FactoredMemoryStore
from aria.factored_retrieval import (
    RetrievalGuidance,
    aggregate_guidance,
    retrieve_factored,
)
from aria.factored_trace import FactoredRetrievalTrace
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
    # Value-algebra: derived selection and actions
    "argmax_by",
    "argmin_by",
    "stamp",
    "cover_obj",
    "paint_cells",
    "connect_paint",
    "boundary",
    "displacement",
    "negate",
    "obj_bbox_region",
    "subgrid_of",
    "union_bbox",
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
    # Value-algebra: derived extraction/crop
    "argmax_by",
    "argmin_by",
    "crop",
    "find_objects",
    "from_object",
    "obj_bbox_region",
    "subgrid_of",
    "union_bbox",
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
    # Value-algebra: derived color operations
    "dominant_color_in",
    "minority_color_in",
    "most_common_val",
    "subgrid_of",
    "stamp",
    "cover_obj",
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
class SketchRefinementResult:
    """Outcome of the sketch-oriented refinement branch."""

    sketches_proposed: int = 0
    sketch_families: tuple[str, ...] = ()
    sketch_compiled: int = 0
    sketch_compile_failures: int = 0
    sketch_verified: int = 0           # programs that passed verification
    sketch_candidates_executed: int = 0
    sketch_budget_used: int = 0        # total work units
    solved: bool = False
    winning_program: Program | None = None
    winning_family: str | None = None
    compile_results: tuple[CompileResult, ...] = ()
    ranking_applied: bool = False
    ranking_changed_order: bool = False
    ranking_policy: str = "none"
    decomp_ranking_applied: bool = False
    decomp_ranking_changed_order: bool = False
    decomp_ranking_policy: str = "none"
    # Graph-native compilation reporting
    graph_native_attempted: int = 0    # sketches tried via graph path
    graph_native_compiled: int = 0     # graph path produced a result (not fallback)
    graph_native_verified: int = 0     # graph-compiled programs that passed verification
    fallback_used: int = 0             # sketches that fell through to legacy compile_sketch
    solve_path: str = "none"           # "graph_native", "fallback", or "none"


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
    repair_result: RepairResult | None = None
    structural_edit_result: EditResult | None = None
    dims_reconstruction: DimsReconstructionResult | None = None
    sketch_result: SketchRefinementResult | None = None
    factored_retrieval_trace: FactoredRetrievalTrace | None = None


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
    factored_store: FactoredMemoryStore | None = None,
    max_steps: int = 3,
    max_candidates: int = 5000,
    max_rounds: int = 2,
    include_core_ops: bool = True,
    policy: HeuristicRefinementPolicy | None = None,
    local_policy: LocalPolicy | None = None,
    beam_width: int = 0,
    beam_rounds: int = 3,
    beam_mutations_per_candidate: int = 30,
    rerank_edits: bool = True,
) -> RefinementResult:
    task_signatures = compute_task_signatures(demos)
    refinement_policy = policy or HeuristicRefinementPolicy()
    rounds: list[RefinementRoundResult] = []
    attempted_plans: set[tuple[str, int, int, tuple[str, ...]]] = set()
    total_candidates = 0

    # Perception: decompose the task from input/output analysis
    decomposition = decompose_task(demos)

    # Factored retrieval: bias search with abstraction memory (cheap, run early)
    factored_trace: FactoredRetrievalTrace | None = None
    factored_guidance: RetrievalGuidance | None = None
    if factored_store is not None and len(factored_store) > 0:
        factored_trace = FactoredRetrievalTrace()
        factored_matches = retrieve_factored(demos, factored_store)
        factored_trace.record_matches(factored_matches)
        if factored_matches:
            factored_guidance = aggregate_guidance(factored_matches)
            factored_trace.record_guidance(factored_guidance)

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

    # Phase 0c: dims-change reconstruction — blank-canvas strategies
    dims_recon: DimsReconstructionResult | None = None
    is_dims_change = demos and any(
        d.input.shape != d.output.shape for d in demos
    )
    if is_dims_change:
        dims_recon = dims_change_reconstruct(demos)
        total_candidates += dims_recon.candidates_tested
        if dims_recon.solved:
            return RefinementResult(
                solved=True,
                winning_program=dims_recon.winning_program,
                candidates_tried=total_candidates,
                rounds=(),
                decomposition=decomposition,
                synthesis_result=synthesis_result,
                dims_reconstruction=dims_recon,
            )

    # Phase 0d: sketch-oriented refinement — propose, compile, verify
    retrieval_decomp_ranker = (
        factored_guidance.make_decomp_ranker()
        if factored_guidance is not None
        else None
    )
    sketch_result = _run_sketch_refinement(
        demos,
        decomp_ranker=retrieval_decomp_ranker,
    )
    total_candidates += sketch_result.sketch_budget_used
    # Track whether retrieval decomp bias actually changed the order
    if factored_trace is not None and sketch_result.decomp_ranking_changed_order:
        factored_trace.decomp_bias_changed_order = True
    if sketch_result.solved and sketch_result.winning_program is not None:
        return RefinementResult(
            solved=True,
            winning_program=sketch_result.winning_program,
            candidates_tried=total_candidates,
            rounds=(),
            decomposition=decomposition,
            synthesis_result=synthesis_result,
            dims_reconstruction=dims_recon,
            sketch_result=sketch_result,
        )

    # Retrieve strong abstractions as search guidance
    hints = tuple(retrieve_abstractions(demos, library))
    preferred = preferred_ops_from_hints(hints)

    # Merge factored retrieval op preferences into preferred ops
    if factored_guidance is not None:
        preferred = preferred | factored_guidance.preferred_ops_as_set()

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
            sketch_result=sketch_result,
            factored_retrieval_trace=factored_trace,
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
                sketch_result=sketch_result,
                factored_retrieval_trace=factored_trace,
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
                sketch_result=sketch_result,
                factored_retrieval_trace=factored_trace,
            )

    # Beam refinement phase
    beam_result = None
    if beam_width > 0:
        seed_programs = _extract_seed_programs(rounds)
        if seed_programs:
            literal_pool = build_literal_pool(demos)
            mutation_priority = (
                factored_guidance.suggested_mutation_priority()
                if factored_guidance is not None
                else ()
            )
            beam_result = run_beam_refinement(
                demos,
                seed_programs,
                literal_pool,
                beam_width=beam_width,
                max_rounds=beam_rounds,
                max_mutations_per_candidate=beam_mutations_per_candidate,
                mutation_priority=mutation_priority,
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
                    sketch_result=sketch_result,
                    factored_retrieval_trace=factored_trace,
                )

    # Explanation-driven repair phase: try to fix same-dims near-misses
    repair_result = None
    if demos and demos[0].input.shape == demos[0].output.shape:
        near_miss_grids, near_miss_progs = _collect_near_miss_grids(demos, rounds, synthesis_result)

        # Phase 1: try program-level repair (produces executable programs)
        prog_candidates = [p for p in near_miss_progs if p is not None]
        if prog_candidates:
            repair_result = attempt_program_repair(demos, prog_candidates, max_repairs=30)
            total_candidates += repair_result.repairs_tried
            if repair_result.solved and repair_result.winning_program is not None:
                return RefinementResult(
                    solved=True,
                    winning_program=repair_result.winning_program,
                    candidates_tried=total_candidates,
                    rounds=tuple(rounds),
                    beam=beam_result,
                    abstraction_hints=hints,
                    skeleton_result=skeleton_result,
                    decomposition=decomposition,
                    synthesis_result=synthesis_result,
                    repair_result=repair_result,
                    sketch_result=sketch_result,
                    factored_retrieval_trace=factored_trace,
                )

        # Phase 2: grid-level repair (diagnostic, may not produce programs)
        if near_miss_grids:
            grid_repair = attempt_repair(
                demos, near_miss_grids, near_miss_progs, max_repairs=20,
            )
            total_candidates += grid_repair.repairs_tried
            if repair_result is None or (grid_repair.solved and not (repair_result and repair_result.solved)):
                repair_result = grid_repair

            # Phase 3: target-directed search — search against repaired targets
            if (grid_repair.solved
                    and grid_repair.winning_program is None
                    and grid_repair.repaired_targets is not None
                    and len(grid_repair.repaired_targets) == len(demos)):
                # Build alternate demos with repaired outputs as targets
                target_demos = tuple(
                    DemoPair(input=demo.input, output=target)
                    for demo, target in zip(demos, grid_repair.repaired_targets)
                )
                target_result = search_program(
                    target_demos,
                    library,
                    program_store=program_store,
                    max_steps=max_steps + 1,
                    max_candidates=max(max_candidates // 2, 500),
                    include_core_ops=include_core_ops,
                    preferred_ops=preferred,
                    excluded_ops=excluded,
                    depth_preferred_fn=decomposition.ops_for_depth,
                )
                total_candidates += target_result.candidates_tried
                if target_result.solved and target_result.winning_program is not None:
                    # Validate against original demos as final acceptance
                    from aria.verify.verifier import verify as final_verify
                    final_vr = final_verify(target_result.winning_program, demos)
                    if final_vr.passed:
                        return RefinementResult(
                            solved=True,
                            winning_program=target_result.winning_program,
                            candidates_tried=total_candidates,
                            rounds=tuple(rounds),
                            beam=beam_result,
                            abstraction_hints=hints,
                            skeleton_result=skeleton_result,
                            decomposition=decomposition,
                            synthesis_result=synthesis_result,
                            repair_result=repair_result,
                            sketch_result=sketch_result,
                            factored_retrieval_trace=factored_trace,
                        )

            # Phase 3b: structural edit search over near-miss programs
            near_miss_executable = [p for p in near_miss_progs if p is not None]
            if near_miss_executable or (obs_result and obs_result.rules):
                edit_targets = (
                    grid_repair.repaired_targets
                    if grid_repair.repaired_targets is not None
                    and len(grid_repair.repaired_targets) == len(demos)
                    else None
                )

                # Build reranking callback from policy when enabled.
                edit_ranker = (
                    _build_program_ranker(
                        local_policy, task_signatures, tuple(rounds),
                    )
                    if rerank_edits
                    else None
                )

                edit_result = structural_edit_search(
                    demos, near_miss_executable or [],
                    repaired_targets=edit_targets,
                    observation_rules=list(obs_result.rules) if obs_result else None,
                    repair_error_class=(
                        grid_repair.primary_error_class
                        if hasattr(grid_repair, 'primary_error_class')
                        else None
                    ),
                    program_ranker=edit_ranker,
                )
                total_candidates += edit_result.candidates_tried
                if edit_result.solved and edit_result.winning_program is not None:
                    return RefinementResult(
                        solved=True,
                        winning_program=edit_result.winning_program,
                        candidates_tried=total_candidates,
                        rounds=tuple(rounds),
                        beam=beam_result,
                        abstraction_hints=hints,
                        skeleton_result=skeleton_result,
                        decomposition=decomposition,
                        synthesis_result=synthesis_result,
                        repair_result=repair_result,
                        structural_edit_result=edit_result,
                        sketch_result=sketch_result,
                        factored_retrieval_trace=factored_trace,
                    )

            if grid_repair.solved:
                return RefinementResult(
                    solved=True,
                    winning_program=grid_repair.winning_program,
                    candidates_tried=total_candidates,
                    rounds=tuple(rounds),
                    beam=beam_result,
                    abstraction_hints=hints,
                    skeleton_result=skeleton_result,
                    decomposition=decomposition,
                    synthesis_result=synthesis_result,
                    repair_result=repair_result,
                    sketch_result=sketch_result,
                    factored_retrieval_trace=factored_trace,
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
        repair_result=repair_result,
        dims_reconstruction=dims_recon,
        sketch_result=sketch_result,
        factored_retrieval_trace=factored_trace,
    )


def _run_sketch_refinement(
    demos: tuple[DemoPair, ...],
    *,
    sketch_ranker=None,
    decomp_ranker=None,
    task_signatures: tuple[str, ...] = (),
) -> SketchRefinementResult:
    """Propose sketches, compile via graph-native path, verify.

    Compilation order for each sketch:
    1. Graph-native: SketchGraph → specialize → compile_sketch_graph
    2. Fallback: compile_sketch (legacy linear path)

    Only marks solved=True if a CompileTaskProgram passes verification
    across all train demos. Per-demo specializations are recorded but
    not promoted.
    """
    from aria.sketch import SketchGraph
    from aria.sketch_fit import fit_sketches_with_report, specialize_sketch
    from aria.sketch_rank import RankingReport, rank_sketches
    from aria.verify.verifier import verify

    # Propose sketches with decomposition ranking
    fit_result = fit_sketches_with_report(
        demos, "",
        decomp_ranker=decomp_ranker,
        task_signatures=task_signatures,
    )
    sketches = fit_result.sketches
    if not sketches:
        return SketchRefinementResult(
            sketches_proposed=0,
            decomp_ranking_applied=fit_result.decomp_ranking_applied,
            decomp_ranking_changed_order=fit_result.decomp_ranking_changed_order,
            decomp_ranking_policy=fit_result.decomp_ranking_policy,
        )

    # Optionally rank sketches
    same_dims = all(d.input.shape == d.output.shape for d in demos)
    from aria.decomposition import detect_bg
    bg_colors = [detect_bg(d.input) for d in demos]
    bg_rotates = len(set(bg_colors)) > 1
    demos_meta = {
        "same_dims": same_dims,
        "bg_rotates": bg_rotates,
        "n_demos": len(demos),
    }

    sketches, rank_report = rank_sketches(
        sketches,
        task_signatures,
        demos_meta,
        ranker=sketch_ranker,
    )

    families = tuple(s.metadata.get("family", "unknown") for s in sketches)
    compile_results: list[CompileResult] = []
    compiled_count = 0
    failure_count = 0
    verified_count = 0
    budget = 0

    # Graph-native tracking
    graph_native_attempted = 0
    graph_native_compiled = 0
    graph_native_verified = 0
    fallback_used = 0
    solve_path = "none"

    winning_program: Program | None = None
    winning_family: str | None = None

    for sketch in sketches:
        budget += 1

        # --- Try graph-native path first ---
        result: CompileResult | None = None
        used_graph_native = False
        try:
            graph = SketchGraph.from_sketch(sketch)
            spec = specialize_sketch(graph, demos)
            graph_native_attempted += 1
            graph_result = compile_sketch_graph(graph, spec, demos)
            # Check if the graph compiler handled it natively (not fallback)
            if isinstance(graph_result, (CompileTaskProgram, CompilePerDemoPrograms)):
                desc = getattr(graph_result, "description", "")
                if "graph" in desc:
                    result = graph_result
                    used_graph_native = True
                    graph_native_compiled += 1
            elif isinstance(graph_result, CompileFailure):
                # Graph-native failure is still a real result for graph-native lanes
                desc = getattr(graph_result, "reason", "")
                if "not implemented" not in desc and "compile_sketch_graph" not in desc:
                    result = graph_result
                    used_graph_native = True
                    graph_native_compiled += 1
        except Exception:
            pass

        # --- Fallback to legacy compile_sketch if graph path didn't handle it ---
        if result is None:
            result = compile_sketch(sketch, demos)
            fallback_used += 1

        compile_results.append(result)

        if isinstance(result, CompileTaskProgram):
            compiled_count += 1
            budget += 1
            vr = verify(result.program, demos)
            if vr.passed:
                verified_count += 1
                if used_graph_native:
                    graph_native_verified += 1
                winning_program = result.program
                winning_family = result.family
                solve_path = "graph_native" if used_graph_native else "fallback"
                break
        elif isinstance(result, CompilePerDemoPrograms):
            compiled_count += 1
            for prog, demo in zip(result.programs, demos):
                budget += 1
                vr = verify(prog, (demo,))
                if vr.passed:
                    verified_count += 1
                    if used_graph_native:
                        graph_native_verified += 1
        elif isinstance(result, CompileFailure):
            failure_count += 1

    return SketchRefinementResult(
        sketches_proposed=len(sketches),
        sketch_families=families,
        sketch_compiled=compiled_count,
        sketch_compile_failures=failure_count,
        sketch_verified=verified_count,
        sketch_candidates_executed=budget,
        sketch_budget_used=budget,
        solved=winning_program is not None,
        winning_program=winning_program,
        winning_family=winning_family,
        compile_results=tuple(compile_results),
        ranking_applied=rank_report.order_changed or rank_report.policy_name != "none",
        ranking_changed_order=rank_report.order_changed,
        ranking_policy=rank_report.policy_name,
        decomp_ranking_applied=fit_result.decomp_ranking_applied,
        decomp_ranking_changed_order=fit_result.decomp_ranking_changed_order,
        decomp_ranking_policy=fit_result.decomp_ranking_policy,
        graph_native_attempted=graph_native_attempted,
        graph_native_compiled=graph_native_compiled,
        graph_native_verified=graph_native_verified,
        fallback_used=fallback_used,
        solve_path=solve_path,
    )


def _collect_near_miss_grids(
    demos: tuple[DemoPair, ...],
    rounds: list[RefinementRoundResult],
    synthesis_result: SynthesisResult | None,
) -> tuple[list[Grid], list[Program | None]]:
    """Collect same-dims candidate grids and their source programs.

    Returns parallel lists: (grids, programs). programs[i] is None when
    the grid has no known source program (e.g., the input itself).
    """
    import numpy as np
    from aria.runtime.executor import execute
    from aria.proposer.parser import parse_program, ParseError

    grids: list[Grid] = []
    programs: list[Program | None] = []
    seen: set[bytes] = set()
    expected_shape = demos[0].output.shape

    def _add(g: Grid, p: Program | None = None) -> None:
        if g.shape != expected_shape:
            return
        key = g.tobytes()
        if key not in seen:
            seen.add(key)
            grids.append(g)
            programs.append(p)

    # Add the input as a candidate (no program)
    _add(demos[0].input, None)

    # Add candidates from search traces (with programs)
    for rnd in rounds:
        for entry in rnd.trace:
            if entry.error_type == "wrong_output" and entry.diff:
                actual_dims = entry.diff.get("actual_dims")
                if actual_dims and actual_dims == tuple(expected_shape):
                    try:
                        prog = parse_program(entry.program_text)
                        result = execute(prog, demos[0].input, None)
                        if isinstance(result, np.ndarray):
                            _add(result, prog)
                    except Exception:
                        pass
            if len(grids) >= 8:
                break

    return grids, programs


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
    mutation_priority: tuple[str, ...] = (),
) -> BeamRefinementResult:
    """Beam search: score candidates, mutate the best, keep a ranked beam.

    When *mutation_priority* is provided (from factored retrieval repair paths),
    mutations with matching edit_kinds are tried first in each round.
    """
    beam: list[tuple[CandidateScore, Program]] = []
    for prog in seed_programs:
        sc = score_program(prog, demos)
        if sc.passed:
            return _beam_result_solved(prog, 1, [], [])
        beam.append((sc, prog))

    beam.sort(key=lambda x: x[0].rank_key)
    beam = beam[:beam_width]

    # Build priority set for mutation ordering
    priority_set = frozenset(mutation_priority)

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
            # Bias mutation order: try retrieval-suggested edit_kinds first
            if priority_set:
                muts.sort(key=lambda m: (0 if m.edit_kind in priority_set else 1))
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


def _build_program_ranker(
    local_policy: "LocalPolicy | None",
    task_signatures: frozenset[str],
    prior_rounds: tuple[RefinementRoundResult, ...],
):
    """Build a program_ranker callback for structural_edit_search.

    If local_policy is None, uses HeuristicBaselinePolicy as default.
    Returns a callable matching the _ProgramRanker protocol:
        (list[str]) -> (indices, changed, policy_name)
    """
    from aria.local_policy import HeuristicBaselinePolicy

    policy = local_policy if local_policy is not None else HeuristicBaselinePolicy()
    policy_input = build_policy_input(task_signatures, len(prior_rounds), prior_rounds)

    def _ranker(program_texts: list[str]) -> tuple[tuple[int, ...], bool, str]:
        ranking = policy.rank_programs(program_texts, policy_input)
        return ranking.indices, ranking.changed, ranking.policy_name

    return _ranker
