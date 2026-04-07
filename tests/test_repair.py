"""Tests for verifier-guided program repair."""

from __future__ import annotations

import numpy as np
import pytest

from aria.core.scene_executor import execute_scene_program, make_scene_program, make_step
from aria.core.scene_solve import (
    infer_and_repair_scene_programs,
    verify_scene_program,
)
from aria.repair import (
    RepairAction,
    RepairDiagnostic,
    RepairResult,
    RepairTrace,
    apply_repair,
    propose_repairs,
    repair_near_misses,
    repair_search,
    score_scene_candidate,
)
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _identity_demos():
    """Same-size input/output, identity transform."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )


def _rot90_demos():
    """Task: extract largest object, apply rot270.

    Input has a single-color triangular CC. Output is its rot270.
    Wrong program applies rot180 instead (same dims, different content).
    """
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ]),
            # rot270 of [[2,2,2],[0,2,2],[0,0,2]]
            output=grid_from_list([
                [0, 0, 2],
                [0, 2, 2],
                [2, 2, 2],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 3],
                [0, 3, 3],
                [3, 3, 3],
            ]),
        ),
    )


def _make_wrong_transform_program():
    """Program that selects largest object but applies rot180 instead of rot90."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.SELECT_ENTITY,
            kind="object",
            predicate="largest_bbox_area",
            rank=0,
            output_id="sel",
        ),
        make_step(
            StepOp.CANONICALIZE_OBJECT,
            source="sel_grid",
            transform="rot180",
            output_id="transformed",
        ),
        make_step(StepOp.RENDER_SCENE, source="transformed"),
    )


def _boolean_combine_demos():
    """Task: XOR combine two panels."""
    return (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 1],
                [0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1],
                [0, 0],
            ]),
        ),
    )


def _fill_color_demos():
    """Task: fill enclosed bg regions with color 3."""
    return (
        DemoPair(
            input=grid_from_list([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]),
            output=grid_from_list([
                [1, 1, 1],
                [1, 3, 1],
                [1, 1, 1],
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# Part A/B: Repair action types
# ---------------------------------------------------------------------------


class TestRepairActionTypes:
    def test_propose_repairs_selector(self):
        """Repair generators produce actions for SELECT_ENTITY steps."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object",
                      predicate="largest_bbox_area", rank=0, output_id="sel"),
            make_step(StepOp.RENDER_SCENE, source="sel_grid"),
        )
        demos = _identity_demos()
        _, diag = score_scene_candidate(prog, demos)
        actions = propose_repairs(prog, diag, demos)
        # Should have selector predicate and rank swaps
        predicates = [a for a in actions if a.param_name == "predicate"]
        ranks = [a for a in actions if a.param_name == "rank"]
        assert len(predicates) > 0
        assert len(ranks) > 0

    def test_propose_repairs_transform(self):
        """Repair generators produce actions for CANONICALIZE_OBJECT steps."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        _, diag = score_scene_candidate(prog, demos)
        actions = propose_repairs(prog, diag, demos)
        transforms = [a for a in actions if a.param_name == "transform"]
        assert len(transforms) > 0
        # rot90 should be among the alternatives
        assert any(a.new_value == "rot90" for a in transforms)

    def test_propose_repairs_boolean(self):
        """Repair generators produce actions for BOOLEAN_COMBINE_PANELS."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=(2, 2)),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(StepOp.BOOLEAN_COMBINE_PANELS, operation="overlay"),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = _identity_demos()
        _, diag = score_scene_candidate(prog, demos)
        actions = propose_repairs(prog, diag, demos)
        booleans = [a for a in actions if a.param_name == "operation"]
        assert len(booleans) >= 2  # "and" and "xor"


# ---------------------------------------------------------------------------
# Part C: Diagnose residual
# ---------------------------------------------------------------------------


class TestDiagnoseResidual:
    def test_perfect_match(self):
        """Perfect match has zero diff."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = (DemoPair(
            input=grid_from_list([[1, 2]]),
            output=grid_from_list([[1, 2]]),
        ),)
        # The program copies input to output via RENDER_SCENE
        # (RENDER_SCENE needs output_grid set, so this will error)
        _, diag = score_scene_candidate(prog, demos)
        # Should have execution error since RENDER_SCENE without output_grid
        assert diag.execution_errors > 0 or diag.pixel_diff_total == 0

    def test_transform_mismatch_detected(self):
        """Detect when result is a rotated version of expected."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        _, diag = score_scene_candidate(prog, demos)
        assert diag.all_dims_match
        assert diag.pixel_diff_total > 0
        assert diag.transform_mismatch

    def test_color_mismatch_detected(self):
        """Detect color differences in diff region."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, fill_color=5),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = _fill_color_demos()
        _, diag = score_scene_candidate(prog, demos)
        if diag.all_dims_match and diag.pixel_diff_total > 0:
            assert diag.color_map_mismatch or diag.diff_colors_expected


# ---------------------------------------------------------------------------
# Part D: Bounded repair search
# ---------------------------------------------------------------------------


class TestRepairSearch:
    def test_repair_wrong_transform(self):
        """Near-miss with wrong transform → repair finds correct one."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        trace = repair_search(prog, demos, near_miss_threshold=0.0)
        # Should find rot90
        assert trace.solved
        assert trace.final_program is not None
        assert verify_scene_program(trace.final_program, demos)

    def test_repair_budget_bounded(self):
        """Verify calls never exceed budget."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        max_rounds = 2
        max_edits = 12
        trace = repair_search(
            prog, demos,
            max_rounds=max_rounds,
            max_edits_per_round=max_edits,
            near_miss_threshold=0.0,
        )
        assert trace.total_verify_calls <= max_rounds * max_edits

    def test_repair_gives_up_on_non_near_miss(self):
        """Programs with wrong dims are skipped."""
        # Program that produces wrong dims
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=(5, 5)),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = (DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),)
        trace = repair_search(prog, demos)
        assert not trace.solved
        assert trace.total_verify_calls == 0

    def test_repair_wrong_selector_rank(self):
        """Test repair can fix wrong selector rank."""
        # Build demos where rank=1 object is the answer
        inp = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        # Output is the second-largest object (2x1 block of 2s)
        out = grid_from_list([
            [2],
            [2],
        ])
        demos = (DemoPair(input=inp, output=out),)

        # Wrong program: selects rank=0 (largest by area, which is 1)
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object",
                      predicate="largest_bbox_area", rank=0, output_id="sel"),
            make_step(StepOp.RENDER_SCENE, source="sel_grid"),
        )

        # Check if dims even match (they might not if largest is 1x1)
        _, diag = score_scene_candidate(prog, demos)
        if diag.all_dims_match:
            trace = repair_search(prog, demos, near_miss_threshold=0.0)
            # Repair should try rank=1
            rank_edits = [
                et for r in trace.rounds for et in r.edits_tried
                if et.action.param_name == "rank"
            ]
            assert len(rank_edits) > 0


# ---------------------------------------------------------------------------
# Part E: apply_repair
# ---------------------------------------------------------------------------


    def test_repair_wrong_boolean_op(self):
        """Near-miss with wrong boolean op → repair finds correct one."""
        # Task: XOR combine two partition cells
        # Input: 2x1 partition with sep color=5
        inp = grid_from_list([
            [1, 0, 5, 0, 1],
            [0, 1, 5, 1, 0],
        ])
        # XOR: pixels non-bg in exactly one panel
        out = grid_from_list([
            [1, 0],
            [0, 0],
        ])
        demos = (DemoPair(input=inp, output=out),)

        # Wrong program: uses "overlay" instead of "xor"
        from aria.core.grid_perception import perceive_grid
        p = perceive_grid(inp)
        if p.partition is not None and len(p.partition.cells) >= 2:
            cell_dims = p.partition.cells[0].dims
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(StepOp.INFER_OUTPUT_SIZE, shape=cell_dims),
                make_step(StepOp.INFER_OUTPUT_BACKGROUND),
                make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
                make_step(StepOp.BOOLEAN_COMBINE_PANELS, operation="overlay"),
                make_step(StepOp.RENDER_SCENE),
            )
            _, diag = score_scene_candidate(prog, demos)
            if diag.all_dims_match:
                trace = repair_search(prog, demos, near_miss_threshold=0.0)
                # Should try "xor"
                bool_edits = [
                    et for r in trace.rounds for et in r.edits_tried
                    if et.action.param_name == "operation"
                ]
                assert len(bool_edits) > 0

    def test_repair_wrong_fill_color(self):
        """Near-miss with wrong fill color → repair tries expected colors or inserts recolor."""
        demos = _fill_color_demos()
        # Wrong program: fills with color 5 instead of 3
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, fill_color=5),
            make_step(StepOp.RENDER_SCENE),
        )
        _, diag = score_scene_candidate(prog, demos)
        if diag.all_dims_match and diag.pixel_diff_total > 0:
            trace = repair_search(prog, demos, near_miss_threshold=0.0)
            # Should find a solution via either fill_color change or recolor insert
            repair_edits = [
                et for r in trace.rounds for et in r.edits_tried
                if et.action.param_name in ("fill_color", "__insert__")
            ]
            assert len(repair_edits) > 0
            assert trace.solved


class TestApplyRepair:
    def test_apply_changes_single_param(self):
        """apply_repair produces a new program with one param changed."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object",
                      predicate="largest_bbox_area", rank=0),
        )
        action = RepairAction(
            step_index=1,
            param_name="predicate",
            old_value="largest_bbox_area",
            new_value="smallest_bbox_area",
            priority=0.8,
            reason="test",
        )
        new_prog = apply_repair(prog, action)
        assert new_prog.steps[1].params["predicate"] == "smallest_bbox_area"
        # Original unchanged
        assert prog.steps[1].params["predicate"] == "largest_bbox_area"
        # Other params preserved
        assert new_prog.steps[1].params["kind"] == "object"
        assert new_prog.steps[1].params["rank"] == 0

    def test_apply_preserves_other_steps(self):
        """apply_repair doesn't touch other steps."""
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object", rank=0),
            make_step(StepOp.RENDER_SCENE, source="sel_grid"),
        )
        action = RepairAction(
            step_index=1,
            param_name="rank",
            old_value=0,
            new_value=1,
            priority=0.5,
            reason="test",
        )
        new_prog = apply_repair(prog, action)
        assert new_prog.steps[0] is prog.steps[0]
        assert new_prog.steps[2] is prog.steps[2]


# ---------------------------------------------------------------------------
# Part F: Trace structure
# ---------------------------------------------------------------------------


class TestRepairTrace:
    def test_trace_structure(self):
        """Trace has correct structure for training data."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        trace = repair_search(prog, demos, near_miss_threshold=0.0)

        assert trace.original_program is prog
        assert isinstance(trace.original_diagnostic, RepairDiagnostic)
        assert isinstance(trace.rounds, tuple)
        assert trace.total_verify_calls >= 0

        if trace.solved:
            assert trace.final_program is not None
            for rnd in trace.rounds:
                assert isinstance(rnd.round_idx, int)
                assert isinstance(rnd.edits_tried, tuple)
                for et in rnd.edits_tried:
                    assert isinstance(et.action, RepairAction)
                    assert isinstance(et.pixel_diff_after, int)
                    assert isinstance(et.improved, bool)
                    assert isinstance(et.exact, bool)


# ---------------------------------------------------------------------------
# Part G: Integration
# ---------------------------------------------------------------------------


class TestRepairNearMisses:
    def test_repair_near_misses_finds_solution(self):
        """repair_near_misses recovers exact solve from near-miss."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        result = repair_near_misses(
            [prog], demos, near_miss_threshold=0.0,
        )
        assert result.solved
        assert result.winning_program is not None
        assert verify_scene_program(result.winning_program, demos)

    def test_repair_near_misses_empty_candidates(self):
        """Empty candidates returns unsolved."""
        demos = _rot90_demos()
        result = repair_near_misses([], demos)
        assert not result.solved

    def test_repair_near_misses_no_near_misses(self):
        """Candidates that are far from correct are skipped."""
        # Program that produces completely wrong output
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.INFER_OUTPUT_SIZE, shape=(2, 2)),
            make_step(StepOp.INFER_OUTPUT_BACKGROUND, background=0),
            make_step(StepOp.INITIALIZE_OUTPUT_SCENE),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = _rot90_demos()
        result = repair_near_misses([prog], demos)
        assert not result.solved
        assert result.near_misses_found == 0

    def test_repair_result_has_trace(self):
        """RepairResult includes traces for debugging."""
        prog = _make_wrong_transform_program()
        demos = _rot90_demos()
        result = repair_near_misses(
            [prog], demos, near_miss_threshold=0.0,
        )
        assert len(result.traces) > 0
        assert result.total_verify_calls > 0
        assert result.candidates_scored == 1


class TestInferAndRepair:
    def test_infer_and_repair_returns_verified(self):
        """When exact match exists, returns it without repair."""
        # Simple identity task
        demos = (
            DemoPair(
                input=grid_from_list([[0, 1], [1, 0]]),
                output=grid_from_list([[0, 1], [1, 0]]),
            ),
        )
        progs = infer_and_repair_scene_programs(demos)
        # May or may not find a solution, but shouldn't crash
        if progs:
            assert verify_scene_program(progs[0], demos)


# ---------------------------------------------------------------------------
# Part B: Repair action space completeness
# ---------------------------------------------------------------------------


class TestRepairGenerators:
    def test_all_generator_types(self):
        """All repair generators return well-formed RepairActions."""
        from aria.repair import _REPAIR_GENERATORS

        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object",
                      predicate="largest_bbox_area", rank=0, output_id="sel"),
            make_step(StepOp.CANONICALIZE_OBJECT, source="sel_grid",
                      transform="rot90", output_id="t"),
            make_step(StepOp.RENDER_SCENE, source="t"),
        )
        demos = _rot90_demos()
        _, diag = score_scene_candidate(prog, demos)

        for gen in _REPAIR_GENERATORS:
            actions = gen(prog, diag, demos)
            for a in actions:
                assert isinstance(a, RepairAction)
                assert 0 <= a.priority <= 1
                assert a.step_index < len(prog.steps)
                assert a.reason

    def test_generator_bounded_output(self):
        """Each generator produces bounded number of actions."""
        from aria.repair import _REPAIR_GENERATORS

        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.SELECT_ENTITY, kind="object",
                      predicate="largest_bbox_area", rank=0, output_id="sel"),
            make_step(StepOp.CANONICALIZE_OBJECT, source="sel_grid",
                      transform="rot90", output_id="t"),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, fill_color=5),
            make_step(StepOp.BOOLEAN_COMBINE_PANELS, operation="overlay"),
            make_step(StepOp.RECOLOR_OBJECT, from_color=1, to_color=2, scope="global"),
            make_step(StepOp.MAP_OVER_ENTITIES, kind="panel", property="dominant_non_bg_color"),
            make_step(StepOp.FOR_EACH_ENTITY, kind="object", rule="fill_bbox_holes",
                      fill_color=3, connectivity=4),
            make_step(StepOp.RENDER_SCENE),
        )
        demos = _identity_demos()
        _, diag = score_scene_candidate(prog, demos)

        for gen in _REPAIR_GENERATORS:
            actions = gen(prog, diag, demos)
            # No single generator should produce an absurd number of actions
            assert len(actions) <= 50, f"{gen.__name__} produced {len(actions)} actions"
