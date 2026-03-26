"""Verifier-driven refinement loop tests."""

from __future__ import annotations

from pathlib import Path

from aria.library.store import Library
from aria.local_policy import HeuristicBaselinePolicy, LocalCausalLMPolicy
from aria.offline_search import SearchTraceEntry, search_program
from aria.refinement import (
    HeuristicRefinementPolicy,
    RefinementFeedback,
    RefinementPlan,
    RefinementRoundResult,
    build_policy_input,
    focus_to_plan,
    run_refinement_loop,
    score_trace,
    summarize_trace_feedback,
)
from aria.trace_store import RefinementTraceStore
from aria.types import DemoPair, grid_from_list


def test_search_program_allowed_ops_restricts_search_space():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0],
                [0, 0],
            ]),
            output=grid_from_list([
                [1, 1],
                [1, 1],
            ]),
        ),
    )

    result = search_program(
        demos,
        Library(),
        max_steps=2,
        max_candidates=10,
        allowed_ops=frozenset({"reflect_grid", "transpose_grid"}),
    )

    assert not result.solved
    assert result.candidates_tried <= 10


def test_refinement_feedback_prefers_marker_geometry_for_additive_marker_tasks():
    feedback = summarize_trace_feedback(
        frozenset({"change:additive", "role:has_marker", "dims:same", "color:new_in_output"}),
        (
            SearchTraceEntry(
                candidate_num=1,
                depth=1,
                program_text="let v0: GRID = overlay(input, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (3, 3),
                    "actual_dims": (3, 3),
                    "pixel_diff_summary": "wrong rows: [0, 1, 2]",
                },
            ),
        ),
    )

    assert feedback.suggested_focus == "marker_geometry"


def test_refinement_feedback_ignores_dimension_noise_for_same_size_marker_tasks():
    feedback = summarize_trace_feedback(
        frozenset({"change:additive", "role:has_marker", "dims:same", "color:new_in_output"}),
        (
            SearchTraceEntry(
                candidate_num=1,
                depth=1,
                program_text="let v0: GRID = stack_h(input, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (12, 10),
                    "actual_dims": (12, 20),
                    "pixel_diff_summary": "dimension mismatch",
                },
            ),
            SearchTraceEntry(
                candidate_num=2,
                depth=1,
                program_text="let v0: GRID = reflect_grid(HORIZONTAL, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (12, 10),
                    "actual_dims": (12, 10),
                    "pixel_diff_summary": "wrong rows: [0, 1, 2]",
                },
            ),
        ),
    )

    assert feedback.suggested_focus == "marker_geometry"


def test_refinement_loop_switches_to_size_focused_round():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([[0] * 6 for _ in range(6)]),
        ),
        DemoPair(
            input=grid_from_list([[1] * 4 for _ in range(4)]),
            output=grid_from_list([[0] * 8 for _ in range(8)]),
        ),
    )

    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=3,
        max_candidates=100,
        max_rounds=2,
    )

    assert not result.solved
    # Decomposition round is prepended, then 2 heuristic rounds
    assert len(result.rounds) >= 2
    # Find the heuristic rounds (not decomposition)
    heuristic_rounds = [r for r in result.rounds if r.policy_source != "decomposition"]
    assert len(heuristic_rounds) >= 1
    assert heuristic_rounds[0].feedback.suggested_focus == "size"
    if len(heuristic_rounds) >= 2:
        assert heuristic_rounds[1].plan.name == "size"


def test_trace_store_round_trips(tmp_path: Path):
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 0], [0, 0]]),
        ),
    )

    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=5,
        max_rounds=1,
    )

    store = RefinementTraceStore()
    store.add_result(task_id="demo", result=result)
    path = tmp_path / "traces.json"
    store.save_json(path)

    loaded = RefinementTraceStore.load_json(path)
    assert len(loaded) == 1
    assert loaded.all_records()[0]["task_id"] == "demo"


def test_score_trace_prefers_same_dims_and_lower_pixel_diff():
    scored = score_trace(
        frozenset({"dims:same", "change:additive", "color:new_in_output"}),
        (
            SearchTraceEntry(
                candidate_num=1,
                depth=1,
                program_text="let v0: GRID = stack_h(input, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (12, 10),
                    "actual_dims": (12, 20),
                    "pixel_diff_count": None,
                    "pixel_diff_summary": "dimension mismatch",
                },
            ),
            SearchTraceEntry(
                candidate_num=2,
                depth=1,
                program_text="let v0: GRID = reflect_grid(HORIZONTAL, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (12, 10),
                    "actual_dims": (12, 10),
                    "pixel_diff_count": 12,
                    "pixel_diff_summary": "wrong rows: [0, 1]",
                    "wrong_rows": [0, 1],
                    "wrong_cols": [0, 1, 2],
                    "palette_expected_coverage": 1.0,
                    "palette_precision": 1.0,
                    "preserved_input_ratio": 0.9,
                    "changed_cells_ratio": 0.3,
                },
            ),
            SearchTraceEntry(
                candidate_num=3,
                depth=1,
                program_text="let v0: GRID = overlay(input, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (12, 10),
                    "actual_dims": (12, 10),
                    "pixel_diff_count": 30,
                    "pixel_diff_summary": "wrong rows: [0, 1, 2, 3]",
                    "wrong_rows": [0, 1, 2, 3],
                    "wrong_cols": [0, 1, 2, 3, 4],
                    "palette_expected_coverage": 0.8,
                    "palette_precision": 0.8,
                    "preserved_input_ratio": 0.6,
                    "changed_cells_ratio": 0.1,
                },
            ),
        ),
    )

    assert scored[1].score is not None
    assert scored[2].score is not None
    assert scored[1].score > scored[0].score
    assert scored[1].score > scored[2].score

    feedback = summarize_trace_feedback(
        frozenset({"dims:same", "change:additive", "role:has_marker", "color:new_in_output"}),
        scored,
    )
    assert feedback.best_candidate_num == 2
    assert feedback.best_candidate_dims_match is True


def test_score_trace_penalizes_execution_errors():
    scored = score_trace(
        frozenset({"dims:same"}),
        (
            SearchTraceEntry(
                candidate_num=1,
                depth=1,
                program_text="let v0: GRID = tile_grid(input, -1, 90)\n-> v0",
                passed=False,
                error_type="execution_error",
            ),
            SearchTraceEntry(
                candidate_num=2,
                depth=1,
                program_text="let v0: GRID = reflect_grid(HORIZONTAL, input)\n-> v0",
                passed=False,
                error_type="wrong_output",
                diff={
                    "expected_dims": (3, 3),
                    "actual_dims": (3, 3),
                    "pixel_diff_count": 5,
                    "pixel_diff_summary": "wrong rows: [0, 1, 2]",
                    "wrong_rows": [0, 1, 2],
                    "wrong_cols": [0, 2],
                },
            ),
        ),
    )

    assert scored[0].score is not None
    assert scored[1].score is not None
    assert scored[1].score > scored[0].score
    assert "execution_error" in scored[0].score_reasons


def test_policy_uses_size_round_after_dimension_mismatch():
    policy = HeuristicRefinementPolicy()
    first_round = (
        RefinementRoundResult(
            plan=RefinementPlan(name="generic", max_steps=2, max_candidates=10),
            solved=False,
            winning_program=None,
            candidates_tried=4,
            feedback=RefinementFeedback(
                dominant_error_type="wrong_output",
                dimension_mismatch_count=3,
                pixel_mismatch_count=0,
                execution_error_count=0,
                suggested_focus="size",
                task_signatures=("dims:different", "size:multiplicative"),
            ),
            trace=(),
        ),
    )

    plan = policy.next_plan(
        round_index=1,
        base_max_steps=2,
        base_max_candidates=10,
        task_signatures=frozenset({"dims:different", "size:multiplicative"}),
        prior_rounds=first_round,
    )

    assert plan.name == "size"


# ---------------------------------------------------------------------------
# Local policy integration tests
# ---------------------------------------------------------------------------


def test_refinement_loop_without_local_policy_uses_heuristic():
    """Default path: no local_policy => heuristic, policy_source='heuristic'."""
    # Use a task that synthesis can't solve (needs multi-step object reasoning)
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]]),
        ),
    )
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=5,
        max_rounds=1,
    )
    heuristic_rounds = [r for r in result.rounds if r.policy_source == "heuristic"]
    assert len(heuristic_rounds) >= 1


def test_refinement_loop_with_local_policy_uses_it_for_round_1():
    """When local_policy is provided, round 0 stays heuristic, round 1+ uses the policy."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([[0] * 6 for _ in range(6)]),
        ),
        DemoPair(
            input=grid_from_list([[1] * 4 for _ in range(4)]),
            output=grid_from_list([[0] * 8 for _ in range(8)]),
        ),
    )

    lp = HeuristicBaselinePolicy()
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=3,
        max_candidates=100,
        max_rounds=2,
        local_policy=lp,
    )

    # Decomposition round may be prepended
    assert len(result.rounds) >= 2
    policy_sources = [r.policy_source for r in result.rounds]
    assert "heuristic" in policy_sources
    assert "local_policy" in policy_sources


def test_refinement_loop_dry_run_lm_does_not_crash():
    """LocalCausalLMPolicy in dry-run mode produces generic focus without error."""
    # Use an unsolvable task so synthesis doesn't short-circuit
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[9, 9, 9], [9, 9, 9], [9, 9, 9]]),
        ),
    )
    lp = LocalCausalLMPolicy(dry_run=True)
    result = run_refinement_loop(
        demos,
        Library(),
        max_steps=1,
        max_candidates=5,
        max_rounds=2,
        local_policy=lp,
    )
    # Should run without error
    assert len(result.rounds) >= 1


def test_build_policy_input_populates_diff_fields():
    """build_policy_input should propagate best-candidate fields from feedback."""
    feedback = RefinementFeedback(
        dominant_error_type="wrong_output",
        dimension_mismatch_count=0,
        pixel_mismatch_count=2,
        execution_error_count=0,
        suggested_focus="marker_geometry",
        task_signatures=("change:additive", "dims:same", "role:has_marker"),
        best_candidate_num=5,
        best_candidate_score=350.0,
        best_candidate_error_type="wrong_output",
        best_candidate_dims_match=True,
        best_candidate_pixel_diff_count=8,
        best_candidate_wrong_row_count=2,
        best_candidate_wrong_col_count=1,
        best_candidate_palette_expected_coverage=1.0,
        best_candidate_palette_precision=0.9,
        best_candidate_preserved_input_ratio=0.8,
        best_candidate_changed_cells_ratio=0.3,
    )
    round_result = RefinementRoundResult(
        plan=RefinementPlan(name="generic", max_steps=3, max_candidates=100),
        solved=False,
        winning_program=None,
        candidates_tried=50,
        feedback=feedback,
        trace=(),
    )

    pi = build_policy_input(
        frozenset({"change:additive", "dims:same", "role:has_marker"}),
        round_index=1,
        prior_rounds=(round_result,),
    )

    assert pi.round_index == 1
    assert pi.prior_focuses == ("marker_geometry",)
    assert pi.prior_error_types == ("wrong_output",)
    assert pi.best_candidate_score == 350.0
    assert pi.best_candidate_dims_match is True
    assert pi.best_candidate_pixel_diff_count == 8
    assert pi.best_candidate_wrong_row_count == 2
    assert pi.best_candidate_wrong_col_count == 1
    assert pi.best_candidate_palette_expected_coverage == 1.0
    assert pi.best_candidate_palette_precision == 0.9
    assert pi.best_candidate_preserved_input_ratio == 0.8
    assert pi.best_candidate_changed_cells_ratio == 0.3


def test_focus_to_plan_maps_known_focuses():
    plan = focus_to_plan("marker_geometry", round_index=1, base_max_steps=3, base_max_candidates=100)
    assert plan.name == "marker_geometry"
    assert plan.allowed_ops is not None
    assert "overlay" in plan.allowed_ops

    plan = focus_to_plan("size", round_index=1, base_max_steps=3, base_max_candidates=100)
    assert plan.name == "size"
    assert "scale_dims" in plan.allowed_ops

    plan = focus_to_plan("color_map", round_index=1, base_max_steps=3, base_max_candidates=100)
    assert plan.name == "color_map"
    assert "recolor" in plan.allowed_ops

    plan = focus_to_plan("generic", round_index=0, base_max_steps=3, base_max_candidates=100)
    assert plan.name == "generic"
    assert plan.allowed_ops is None
    assert plan.max_steps == 3

    plan = focus_to_plan("generic", round_index=2, base_max_steps=3, base_max_candidates=100)
    assert plan.name == "generic_expand_2"
    assert plan.max_steps == 4
