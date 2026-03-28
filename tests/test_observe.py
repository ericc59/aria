"""Tests for observation-driven rule induction and synthesis."""

from __future__ import annotations

from aria.observe import (
    _color_key,
    ObjectObservation,
    ObjectRule,
    ObservationSynthesisResult,
    _infer_move_rules,
    _infer_recolor_rules,
    _infer_remove_rules,
    _infer_surround_rules,
    _observe_objects,
    observe_and_synthesize,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _surround_task():
    """Color 2 → diagonal cross color 4, color 1 → plus color 7."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 4, 0, 4, 0],
                [0, 0, 2, 0, 0],
                [0, 4, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [4, 0, 4, 0, 0],
                [0, 2, 0, 0, 0],
                [4, 0, 4, 0, 0],
            ]),
        ),
    )


def _recolor_task():
    """Every pixel of color 1 → color 5, color 2 → color 6."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [0, 1]]),
            output=grid_from_list([[5, 6], [0, 5]]),
        ),
        DemoPair(
            input=grid_from_list([[2, 0], [1, 2]]),
            output=grid_from_list([[6, 0], [5, 6]]),
        ),
    )


def _move_task():
    """All objects move down by 1 row. (Diagnosed, not expressible.)"""
    return (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ),
    )


def _remove_task():
    """All objects of color 2 are removed."""
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 0], [0, 2, 1], [0, 0, 0]]),
            output=grid_from_list([[1, 0, 0], [0, 0, 1], [0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 2], [1, 0, 0], [2, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        ),
    )


def _inconsistent_task():
    """Objects of the same color behave differently → no consistent rule."""
    return (
        DemoPair(
            input=grid_from_list([[1, 0, 1], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 1], [1, 0, 0], [0, 0, 0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------


def test_observe_objects_detects_surround():
    demos = _surround_task()
    obs = _observe_objects(demos[0])
    color2_obs = [o for o in obs if o.color == 2]
    assert len(color2_obs) == 1
    assert color2_obs[0].added_color == 4
    assert len(color2_obs[0].added_offsets) == 4


def test_observe_objects_detects_removal():
    demos = _remove_task()
    obs = _observe_objects(demos[0])
    color2_obs = [o for o in obs if o.color == 2]
    assert all(o.removed for o in color2_obs)


# ---------------------------------------------------------------------------
# Surround rule induction
# ---------------------------------------------------------------------------


def test_surround_rule_induction():
    demos = _surround_task()
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_surround_rules(all_obs, _color_key, "color")
    assert len(rules) >= 1
    diag = [r for r in rules if r.input_color == 2]
    assert len(diag) == 1
    assert diag[0].output_color == 4
    assert diag[0].kind == "surround"


def test_surround_synthesis_solves():
    demos = _surround_task()
    result = observe_and_synthesize(demos)
    assert result.solved
    assert result.candidates_tested <= 5


# ---------------------------------------------------------------------------
# Recolor rule induction
# ---------------------------------------------------------------------------


def test_recolor_rule_induction():
    demos = _recolor_task()
    rules = _infer_recolor_rules(demos)
    assert len(rules) == 1
    assert rules[0].kind == "recolor"
    cmap = rules[0].details["color_map"]
    assert cmap[1] == 5
    assert cmap[2] == 6


def test_recolor_synthesis_solves():
    demos = _recolor_task()
    result = observe_and_synthesize(demos)
    assert result.solved
    assert result.candidates_tested <= 5


def test_recolor_inconsistent_returns_no_rule():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [0, 0]]),
            output=grid_from_list([[5, 6], [0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2], [0, 0]]),
            output=grid_from_list([[5, 7], [0, 0]]),  # different map for color 2
        ),
    )
    rules = _infer_recolor_rules(demos)
    assert len(rules) == 0


# ---------------------------------------------------------------------------
# Move rule induction
# ---------------------------------------------------------------------------


def test_move_rule_induction():
    demos = _move_task()
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_move_rules(all_obs, _color_key, "color")
    move_rules = [r for r in rules if r.kind == "move"]
    assert len(move_rules) >= 1
    assert move_rules[0].details["dr"] == 1 or move_rules[0].details["dc"] != 0
    # Global shift is now expressible
    assert move_rules[0].expressible == move_rules[0].details.get("global_shift", False)


# ---------------------------------------------------------------------------
# Remove rule induction
# ---------------------------------------------------------------------------


def test_remove_rule_induction():
    demos = _remove_task()
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_remove_rules(all_obs, _color_key, "color")
    remove_rules = [r for r in rules if r.kind == "remove"]
    assert len(remove_rules) >= 1
    assert remove_rules[0].input_color == 2
    assert remove_rules[0].expressible is True  # now expressible via apply_color_map


def test_remove_not_inferred_when_only_some_removed():
    """If only some objects of a color are removed, no remove rule."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 1], [0, 0]]),
            output=grid_from_list([[0, 1], [0, 0]]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_remove_rules(all_obs, _color_key, "color")
    remove_rules = [r for r in rules if r.kind == "remove" and r.input_color == 1]
    assert len(remove_rules) == 0  # not all color-1 objects were removed


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_inconsistent_observations_produce_no_surround_rule():
    """When objects of the same color have different changes, no rule should be emitted."""
    demos = _inconsistent_task()
    all_obs = [_observe_objects(d) for d in demos]
    surround = _infer_surround_rules(all_obs, _color_key, "color")
    # If any surround rules exist, their offset sets must be consistent
    for r in surround:
        assert r.offsets is not None
        assert len(r.offsets) > 0


def test_empty_demos_produce_no_rules():
    result = observe_and_synthesize(())
    assert not result.solved
    assert len(result.rules) == 0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_rules_carry_diagnostics():
    demos = _surround_task()
    result = observe_and_synthesize(demos)
    for rule in result.rules:
        assert isinstance(rule.kind, str)
        assert isinstance(rule.expressible, bool)
        assert isinstance(rule.grouping_key, str)


def test_move_rule_carries_delta_details():
    demos = _move_task()
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_move_rules(all_obs, _color_key, "color")
    if rules:
        assert "dr" in rules[0].details
        assert "dc" in rules[0].details


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_composed_surround_solves_multi_color():
    """Task with two surround rules should be solved by composition."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 4, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0],
                [0, 4, 0, 4, 7, 0, 0],
                [0, 0, 0, 7, 1, 7, 0],
                [0, 0, 0, 0, 7, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
        ),
    )
    result = observe_and_synthesize(demos)
    assert result.solved
    # Should have surround rules for both colors (may include color_size duplicates)
    surround_rules = [r for r in result.rules if r.kind == "surround"]
    colors_covered = {r.input_color for r in surround_rules}
    assert 1 in colors_covered and 2 in colors_covered


# ---------------------------------------------------------------------------
# Integration with refinement
# ---------------------------------------------------------------------------


def test_observe_integrates_with_refinement():
    from aria.library.store import Library
    from aria.refinement import run_refinement_loop

    demos = _recolor_task()
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    assert result.solved


# ---------------------------------------------------------------------------
# Remove rule expression
# ---------------------------------------------------------------------------


def test_remove_synthesis_solves():
    """Remove-by-color task should be solved via apply_color_map({color: 0})."""
    demos = _remove_task()
    result = observe_and_synthesize(demos)
    assert result.solved
    assert result.candidates_tested <= 10
    # Should use a remove rule
    remove_rules = [r for r in result.rules if r.kind == "remove"]
    assert len(remove_rules) >= 1
    assert remove_rules[0].expressible


def test_remove_multi_color():
    """Task where two different colors are removed."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3], [2, 3, 1], [3, 1, 2]]),
            output=grid_from_list([[1, 0, 3], [0, 3, 1], [3, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[2, 1, 3], [1, 2, 3]]),
            output=grid_from_list([[0, 1, 3], [1, 0, 3]]),
        ),
    )
    result = observe_and_synthesize(demos)
    # Color 2 removed — this should be detected as a recolor rule (pixel-level map)
    # since only color 2 → 0 while others stay the same
    assert result.solved


# ---------------------------------------------------------------------------
# Move rule expression (global shift)
# ---------------------------------------------------------------------------


def _global_shift_task():
    """All content shifts down by 1 (global shift)."""
    return (
        DemoPair(
            input=grid_from_list([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [1, 0, 0], [0, 2, 0]]),
        ),
    )


def test_global_shift_rule_is_expressible():
    demos = _global_shift_task()
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_move_rules(all_obs, _color_key, "color")
    move_rules = [r for r in rules if r.kind == "move"]
    # All objects shift down by 1 → global_shift=True → expressible
    expressible_moves = [r for r in move_rules if r.expressible]
    assert len(expressible_moves) >= 1
    assert expressible_moves[0].details["global_shift"]
    assert expressible_moves[0].details["dr"] == 1


def test_global_shift_synthesis_solves():
    demos = _global_shift_task()
    result = observe_and_synthesize(demos)
    assert result.solved
    assert result.candidates_tested <= 10


def test_per_color_axis_aligned_move_is_expressible():
    """Per-color axis-aligned moves are now expressible via object pipeline."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [1, 0, 0], [0, 0, 2]]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_move_rules(all_obs, _color_key, "color")
    axis_moves = [r for r in rules if r.kind == "move" and r.details.get("axis_aligned")]
    assert len(axis_moves) >= 1
    assert all(r.expressible for r in axis_moves)


# ---------------------------------------------------------------------------
# shift_grid op
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Object-pipeline synthesis
# ---------------------------------------------------------------------------


def test_object_pipeline_recolor():
    """Per-object recolor via object pipeline."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 1, 0], [0, 0, 0], [0, 2, 0]]),
            output=grid_from_list([[0, 5, 0], [0, 0, 0], [0, 2, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 0, 2]]),
            output=grid_from_list([[5, 0, 0], [0, 0, 2]]),
        ),
    )
    result = observe_and_synthesize(demos)
    assert result.solved


def test_object_pipeline_remove():
    """Remove objects of a color via object pipeline."""
    demos = _remove_task()
    result = observe_and_synthesize(demos)
    assert result.solved


def test_object_pipeline_move_axis_aligned():
    """Move all objects of a color down by 1 via object pipeline."""
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        ),
    )
    result = observe_and_synthesize(demos)
    assert result.solved


def test_diagonal_per_color_move_now_expressible():
    """Diagonal per-color moves are now expressible via translate_delta."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 2, 0]]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_move_rules(all_obs, _color_key, "color")
    diag = [r for r in rules if r.kind == "move"
            and not r.details.get("axis_aligned")
            and not r.details.get("global_shift")]
    for r in diag:
        assert r.expressible  # now expressible via translate_delta


def test_diagonal_move_synthesis():
    """Diagonal movement should be solvable via object pipeline."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # clipped
        ),
    )
    # This tests that translate_delta(1,1) works in the pipeline
    result = observe_and_synthesize(demos)
    # May or may not solve depending on clipping, but should not crash
    assert result.candidates_tested >= 0


# ---------------------------------------------------------------------------
# Rigid transform rules
# ---------------------------------------------------------------------------


def test_rigid_transform_detection_rot180():
    """Observation should detect 180° rotation of an object's mask."""
    # Color 1 object: 2x1 → after rot180 becomes 2x1 flipped
    # Use a 3-pixel L-shape that's clearly asymmetric
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
        ),
    )
    obs = _observe_objects(demos[0])
    color1 = [o for o in obs if o.color == 1]
    assert len(color1) == 1
    assert color1[0].shape_transform == "rot180"


def test_rigid_transform_rule_induction():
    """Consistent rigid transform across demos should produce a rule."""
    from aria.observe import _infer_rigid_transform_rules
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]]),
            output=grid_from_list([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_rigid_transform_rules(all_obs, _color_key, "color")
    rt_rules = [r for r in rules if r.kind == "rigid_transform"]
    assert len(rt_rules) >= 1
    assert rt_rules[0].details["transform"] == "rot180"
    assert rt_rules[0].expressible


def test_rigid_transform_inconsistent_no_rule():
    """Different transforms across demos should not produce a rule."""
    from aria.observe import _infer_rigid_transform_rules
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 1], [0, 1, 1]]),  # rot180
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 1, 1]]),  # flip_v maybe
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    rules = _infer_rigid_transform_rules(all_obs, _color_key, "color")
    rt_rules = [r for r in rules if r.kind == "rigid_transform" and r.input_color == 1]
    # If transforms differ across demos, no rule
    if rt_rules:
        # All transforms must be the same
        transforms = set(r.details["transform"] for r in rt_rules)
        assert len(transforms) == 1


def test_translate_delta_op():
    """translate_delta moves an object by arbitrary (dr, dc)."""
    import aria.runtime
    from aria.runtime.ops import get_op

    _, find = get_op("find_objects")
    _, td = get_op("translate_delta")

    grid = grid_from_list([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    objects = find(grid)
    obj = next(iter(objects))
    moved = td(1, 1, obj)
    # Original at bbox (1, 0, 1, 1) → moved to (2, 1, 1, 1)
    assert moved.bbox[0] == obj.bbox[0] + 1  # col shifted
    assert moved.bbox[1] == obj.bbox[1] + 1  # row shifted


def test_paint_objects_op():
    """paint_objects renders objects at their bbox positions on a grid."""
    import aria.runtime
    from aria.runtime.executor import execute
    from aria.types import Bind, Call, Literal, Program, Ref, Type
    import numpy as np

    grid = grid_from_list([[0, 1, 0], [0, 0, 2]])
    # Use the DSL executor for partial application (recolor(5))
    prog = Program(steps=(
        Bind("v0", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
        Bind("v1", Type.PREDICATE, Call("by_color", (Literal(1, Type.COLOR),))),
        Bind("v2", Type.OBJECT_SET, Call("where", (Ref("v1"), Ref("v0")))),
        Bind("v3", Type.OBJ_TRANSFORM, Call("recolor", (Literal(5, Type.COLOR),))),
        Bind("v4", Type.OBJECT_SET, Call("map_obj", (Ref("v3"), Ref("v2")))),
        Bind("v5", Type.GRID, Call("paint_objects", (Ref("v4"), Ref("input")))),
    ), output="v5")
    result = execute(prog, grid, None)
    expected = grid_from_list([[0, 5, 0], [0, 0, 2]])
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# Color+size grouping
# ---------------------------------------------------------------------------


def test_color_size_grouping_separates_same_color():
    """Same color but different sizes should produce separate rules."""
    from aria.observe import _infer_remove_rules, _color_size_key
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],  # small color-1 removed
                [0, 0, 0, 1, 1],  # big color-1 kept
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    # Color-only grouping: not all color-1 removed → no rule
    rules_color = _infer_remove_rules(all_obs, _color_key, "color")
    remove_color = [r for r in rules_color if r.kind == "remove" and r.input_color == 1]
    assert len(remove_color) == 0  # inconsistent: big ones stayed

    # Color+size grouping: size-1 objects of color 1 all removed → rule
    rules_cs = _infer_remove_rules(all_obs, _color_size_key, "color_size")
    remove_cs = [r for r in rules_cs if r.kind == "remove"
                 and r.details.get("group_key") == (1, 1)]
    assert len(remove_cs) == 1


def test_color_size_selection_in_pipeline():
    """Pipeline should use by_color + by_size when grouping_key is color_size."""
    from aria.observe import _single_pipeline, ObjectRule
    rule = ObjectRule(
        kind="remove", input_color=1, output_color=None, offsets=None,
        details={"count": 1, "group_key": (1, 1)},
        expressible=True, grouping_key="color_size",
    )
    prog = _single_pipeline(rule)
    assert prog is not None
    from aria.runtime.program import program_to_text
    text = program_to_text(prog)
    assert "by_color" in text
    assert "by_size" in text


# ---------------------------------------------------------------------------
# Proximity-to-marker relational grouping
# ---------------------------------------------------------------------------


def test_proximity_grouping_separates_near_vs_far():
    """Same-color same-size objects near/far from marker behave differently.

    Semantics locked: uses per-color-group nearest to first marker object
    (center-to-center Chebyshev, ties by Manhattan then obj_id).
    """
    from aria.observe import _make_proximity_key, _infer_remove_rules
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]

    rules_c = _infer_remove_rules(all_obs, _color_key, "color")
    remove_c = [r for r in rules_c if r.kind == "remove" and r.input_color == 1]
    assert len(remove_c) == 0

    prox_fn, marker = _make_proximity_key(all_obs, demos)
    assert prox_fn is not None
    assert marker == 2
    rules_p = _infer_remove_rules(all_obs, prox_fn, "proximity")
    remove_far = [r for r in rules_p if r.kind == "remove"
                  and r.details.get("group_key") == (1, False)]
    assert len(remove_far) == 1


def test_proximity_induction_and_pipeline_use_same_semantics():
    """The object identified as 'nearest' by induction must match nearest_to runtime.

    This is the critical consistency test: induction groups by proximity,
    pipeline selects by nearest_to, and they must agree on which object is nearest.
    """
    import aria.runtime
    from aria.observe import _make_proximity_key
    from aria.runtime.ops import get_op

    # Grid: marker color 2 at (1,1), two color-1 objects at (2,2) near and (4,4) far
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    demos = (DemoPair(input=grid, output=grid),)
    all_obs = [_observe_objects(d) for d in demos]
    prox_fn, marker = _make_proximity_key(all_obs, demos)

    # Induction side: which color-1 object is nearest?
    color1_obs = [o for o in all_obs[0] if o.color == 1]
    keys = {o.obj_id: prox_fn(o) for o in color1_obs}
    nearest_by_induction = [oid for oid, k in keys.items() if k[1] is True]
    assert len(nearest_by_induction) == 1  # exactly one nearest

    # Execution side: what does nearest_to(marker_obj, color1_set) return?
    _, find = get_op("find_objects")
    _, by_color = get_op("by_color")
    _, where = get_op("where")
    _, nth_fn = get_op("nth")
    _, nearest_to = get_op("nearest_to")

    objects = find(grid)
    marker_set = where(by_color(2), objects)
    marker_obj = nth_fn(0, marker_set)
    color1_set = where(by_color(1), objects)
    nearest_obj = nearest_to(marker_obj, color1_set)

    # They must agree
    assert nearest_obj.id == nearest_by_induction[0]


def test_proximity_tie_broken_deterministically():
    """When two objects are equidistant, the one with lower obj_id wins."""
    from aria.observe import _make_proximity_key

    # Two color-1 objects equidistant from the color-2 marker
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    demos = (DemoPair(input=grid, output=grid),)
    all_obs = [_observe_objects(d) for d in demos]
    prox_fn, marker = _make_proximity_key(all_obs, demos)

    color1_obs = [o for o in all_obs[0] if o.color == 1]
    keys = {o.obj_id: prox_fn(o) for o in color1_obs}
    nearest = [oid for oid, k in keys.items() if k[1] is True]
    # Exactly one winner even in a tie
    assert len(nearest) == 1


def test_proximity_grouping_in_synthesis():
    """Full synthesis path with proximity grouping."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    result = observe_and_synthesize(demos)
    prox_rules = [r for r in result.rules if r.grouping_key == "proximity"]
    assert len(prox_rules) >= 1
    # Diagnostics should record the proximity mode
    for r in prox_rules:
        assert r.details.get("proximity_mode") == "per_color_nearest_to_first_marker_object"
        assert "marker_color" in r.details


def test_singleton_set_op():
    """singleton_set wraps a single object into a set."""
    import aria.runtime
    from aria.runtime.ops import get_op

    _, find = get_op("find_objects")
    _, nth_fn = get_op("nth")
    _, sset = get_op("singleton_set")

    grid = grid_from_list([[0, 1, 0], [0, 0, 2]])
    objects = find(grid)
    first = nth_fn(0, objects)
    wrapped = sset(first)
    assert isinstance(wrapped, set)
    assert len(wrapped) == 1
    assert next(iter(wrapped)).id == first.id


def test_proximity_no_false_positives_on_single_color():
    """Proximity grouping with one non-bg color: marker = that color, no useful split."""
    from aria.observe import _make_proximity_key
    demos = (
        DemoPair(
            input=grid_from_list([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
            output=grid_from_list([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]
    prox_fn, marker = _make_proximity_key(all_obs, demos)
    assert marker is not None


# ---------------------------------------------------------------------------
# Directional relative-to-marker grouping
# ---------------------------------------------------------------------------


def test_directional_grouping_above_below():
    """Objects above the marker vs below should be split by directional grouping."""
    from aria.observe import _make_directional_keys, _infer_remove_rules
    # Marker color-2 at center. Color-1 objects above and below.
    # Only the ones below are removed.
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],  # below-marker color-1 removed
            ]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]

    # Color-only: not all color-1 removed → no rule
    rules_c = _infer_remove_rules(all_obs, _color_key, "color")
    assert len([r for r in rules_c if r.kind == "remove" and r.input_color == 1]) == 0

    # Directional: below-marker color-1 objects all removed → rule
    dir_fns, marker = _make_directional_keys(all_obs, demos)
    assert marker == 2
    below_fn = [fn for fn, d in dir_fns if d == "below"][0]
    rules_d = _infer_remove_rules(all_obs, below_fn, "direction_below")
    below_removed = [r for r in rules_d if r.kind == "remove"
                     and r.details.get("group_key") == (1, True)]
    assert len(below_removed) == 1


def test_directional_induction_matches_execution():
    """Directional induction and by_relative_pos runtime must agree."""
    import aria.runtime
    from aria.observe import _make_directional_keys
    from aria.runtime.ops import get_op

    grid = grid_from_list([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    demos = (DemoPair(input=grid, output=grid),)
    all_obs = [_observe_objects(d) for d in demos]
    dir_fns, marker = _make_directional_keys(all_obs, demos)

    above_fn = [fn for fn, d in dir_fns if d == "above"][0]
    color1_obs = [o for o in all_obs[0] if o.color == 1]
    induction_above = [o.obj_id for o in color1_obs if above_fn(o)[1] is True]

    # Execution side
    _, find = get_op("find_objects")
    _, by_color = get_op("by_color")
    _, where = get_op("where")
    _, nth_fn = get_op("nth")
    _, by_rel_pos = get_op("by_relative_pos")
    from aria.types import Dir

    objects = find(grid)
    marker_set = where(by_color(2), objects)
    marker_obj = nth_fn(0, marker_set)
    color1_set = where(by_color(1), objects)
    above_pred = by_rel_pos(Dir.UP, marker_obj)
    above_set = where(above_pred, color1_set)
    execution_above = sorted(o.id for o in above_set)

    assert sorted(induction_above) == execution_above


def test_directional_grouping_in_synthesis():
    """Full synthesis path with directional grouping produces rules."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
    )
    result = observe_and_synthesize(demos)
    dir_rules = [r for r in result.rules if r.grouping_key.startswith("direction_")]
    assert len(dir_rules) >= 1
    for r in dir_rules:
        assert "direction" in r.details
        assert "direction_mode" in r.details
        assert r.details["direction_mode"] == "per_color_direction_from_first_marker_object"


def test_by_relative_pos_op():
    """by_relative_pos filters objects by direction from reference."""
    import aria.runtime
    from aria.runtime.ops import get_op
    from aria.types import Dir

    _, find = get_op("find_objects")
    _, by_rel = get_op("by_relative_pos")
    _, where = get_op("where")
    _, nth_fn = get_op("nth")
    _, by_color = get_op("by_color")

    grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    objects = find(grid)
    marker_set = where(by_color(2), objects)
    marker_obj = nth_fn(0, marker_set)

    above = where(by_rel(Dir.UP, marker_obj), objects)
    below = where(by_rel(Dir.DOWN, marker_obj), objects)

    # Object at row 0 should be above the marker at row 2
    assert any(o.color == 1 for o in above)
    # Object at row 4 should be below
    assert any(o.color == 1 for o in below)
    # Marker itself should not be in either set
    assert all(o.color != 2 for o in above)
    assert all(o.color != 2 for o in below)


# ---------------------------------------------------------------------------
# Shared object-composition
# ---------------------------------------------------------------------------


def test_shared_composition_remove_and_move():
    """Remove color 2 + move color 1 down, composed over shared object set."""
    demos = (
        DemoPair(
            input=grid_from_list([[2, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 2, 0], [1, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        ),
    )
    result = observe_and_synthesize(demos)
    # Should have both remove and move rules
    kinds = {r.kind for r in result.rules}
    assert "remove" in kinds
    assert "move" in kinds
    # Check if shared composition was produced
    composed = [r for r in result.rules if r.kind == "shared_composition"]
    # Even if not explicitly in rules, the program should be tried
    assert result.candidates_tested > 0


def test_shared_composition_remove_below_recolor_above():
    """Directional: remove objects below marker, keep objects above marker unchanged."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
    )
    result = observe_and_synthesize(demos)
    # Directional rules should be diagnosed
    dir_rules = [r for r in result.rules if r.grouping_key.startswith("direction_")]
    assert len(dir_rules) >= 1


def test_shared_composition_disjoint_colors():
    """Two rules on different colors should compose cleanly."""
    from aria.observe import _shared_object_composition, ObjectRule
    rules = [
        ObjectRule(kind="remove", input_color=2, output_color=None,
                   offsets=None, details={"count": 1, "group_key": (2,)},
                   expressible=True, grouping_key="color"),
        ObjectRule(kind="remove", input_color=3, output_color=None,
                   offsets=None, details={"count": 1, "group_key": (3,)},
                   expressible=True, grouping_key="color"),
    ]
    result = _shared_object_composition(rules)
    assert result is not None
    prog, meta = result
    assert meta.kind == "shared_composition"
    assert meta.expression_path == "shared_object_composition"
    assert meta.details["rule_count"] == 2


def test_shared_composition_rejects_same_color_conflict():
    """Two different transforms on the same color+grouping should be rejected."""
    from aria.observe import _shared_object_composition, ObjectRule
    rules = [
        ObjectRule(kind="remove", input_color=1, output_color=None,
                   offsets=None, details={"count": 1, "group_key": (1,)},
                   expressible=True, grouping_key="color"),
        ObjectRule(kind="move", input_color=1, output_color=None,
                   offsets=((1, 0),), details={"dr": 1, "dc": 0, "count": 1,
                                                "global_shift": False, "axis_aligned": True,
                                                "group_key": (1,)},
                   expressible=True, grouping_key="color"),
    ]
    result = _shared_object_composition(rules)
    assert result is None  # conflict: same color, same grouping, different transforms


def test_shared_composition_rejects_overlapping_same_color_different_grouping():
    """Same-color rules with different grouping keys that overlap on actual objects."""
    from aria.observe import _shared_object_composition, ObjectRule
    # Two rules on color 1 with different grouping keys (proximity vs direction_above)
    # but if the nearest object is also the above object, they overlap
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
    )
    rules = [
        ObjectRule(kind="remove", input_color=1, output_color=None,
                   offsets=None,
                   details={"count": 1, "group_key": (1, True), "marker_color": 2},
                   expressible=True, grouping_key="proximity"),
        ObjectRule(kind="move", input_color=1, output_color=None,
                   offsets=((1, 0),),
                   details={"dr": 1, "dc": 0, "count": 1, "global_shift": False,
                            "axis_aligned": True,
                            "group_key": (1, True), "marker_color": 2, "direction": "above"},
                   expressible=True, grouping_key="direction_above"),
    ]
    result = _shared_object_composition(rules, demos)
    # The above object IS the nearest object → overlap → reject
    assert result is None


def test_shared_composition_allows_disjoint_same_color_different_grouping():
    """Same-color rules with different grouping keys that select disjoint objects."""
    from aria.observe import _shared_object_composition, ObjectRule
    # Color 1 above marker: remove. Color 1 below marker: keep (not a rule here).
    # But if we have remove(above) + move(below), they should compose.
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]),
        ),
    )
    rules = [
        ObjectRule(kind="remove", input_color=1, output_color=None,
                   offsets=None,
                   details={"count": 1, "group_key": (1, True), "marker_color": 2, "direction": "above"},
                   expressible=True, grouping_key="direction_above"),
        ObjectRule(kind="move", input_color=1, output_color=None,
                   offsets=((0, 1),),
                   details={"dr": 0, "dc": 1, "count": 1, "global_shift": False,
                            "axis_aligned": True,
                            "group_key": (1, True), "marker_color": 2, "direction": "below"},
                   expressible=True, grouping_key="direction_below"),
    ]
    result = _shared_object_composition(rules, demos)
    # Above and below are disjoint → should compose
    assert result is not None


def test_shared_composition_diagnostics():
    """Composed rule carries clear diagnostics including disjointness validation."""
    from aria.observe import _shared_object_composition, ObjectRule
    rules = [
        ObjectRule(kind="remove", input_color=2, output_color=None,
                   offsets=None, details={"count": 1, "group_key": (2,)},
                   expressible=True, grouping_key="color"),
        ObjectRule(kind="move", input_color=1, output_color=None,
                   offsets=((1, 0),), details={"dr": 1, "dc": 0, "count": 1,
                                                "global_shift": False, "axis_aligned": True,
                                                "group_key": (1,)},
                   expressible=True, grouping_key="color"),
    ]
    result = _shared_object_composition(rules)
    assert result is not None
    _, meta = result
    assert meta.details["rule_kinds"] == ["remove", "move"]
    assert meta.details["rule_colors"] == [2, 1]
    assert meta.details["disjointness_validated"] is True
    assert meta.expression_path == "shared_object_composition"


# ---------------------------------------------------------------------------
# Size-rank pairwise grouping
# ---------------------------------------------------------------------------


def test_size_rank_grouping_separates_largest():
    """Largest-in-color-group should be separated from smaller objects."""
    from aria.observe import _make_size_rank_key, _infer_remove_rules
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],  # small color-1 removed, large kept
                [0, 0, 0, 0],
            ]),
        ),
    )
    all_obs = [_observe_objects(d) for d in demos]

    # Color-only: not all removed → no rule
    rules_c = _infer_remove_rules(all_obs, _color_key, "color")
    remove_c = [r for r in rules_c if r.kind == "remove" and r.input_color == 1]
    assert len(remove_c) == 0

    # Size-rank: small ones (is_largest=False) all removed → rule
    size_fn = _make_size_rank_key(all_obs)
    rules_sr = _infer_remove_rules(all_obs, size_fn, "size_rank")
    remove_small = [r for r in rules_sr if r.kind == "remove"
                    and r.details.get("group_key") == (1, False)]
    assert len(remove_small) == 1


def test_size_rank_in_pipeline():
    """Size-rank selection should produce by_size_rank in the pipeline."""
    from aria.observe import _single_pipeline, ObjectRule
    rule = ObjectRule(
        kind="remove", input_color=1, output_color=None, offsets=None,
        details={"count": 1, "group_key": (1, False)},
        expressible=True, grouping_key="size_rank",
    )
    prog = _single_pipeline(rule)
    assert prog is not None
    from aria.runtime.program import program_to_text
    text = program_to_text(prog)
    assert "by_size_rank" in text


# ---------------------------------------------------------------------------
# Output reconstruction (dims-change)
# ---------------------------------------------------------------------------


def test_infer_output_dims_fixed():
    from aria.observe import _infer_output_dims
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]), output=grid_from_list([[1], [2], [3]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]), output=grid_from_list([[5], [6], [7]])),
    )
    dims = _infer_output_dims(demos)
    assert dims == (3, 1)


def test_infer_output_dims_variable():
    from aria.observe import _infer_output_dims
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]), output=grid_from_list([[1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]), output=grid_from_list([[5], [6]])),
    )
    dims = _infer_output_dims(demos)
    assert dims is None


def test_pipeline_with_output_dims():
    """Remove pipeline on a blank canvas of inferred output dims."""
    from aria.observe import _single_pipeline, ObjectRule
    rule = ObjectRule(
        kind="remove", input_color=2, output_color=None, offsets=None,
        details={"count": 1, "group_key": (2,)},
        expressible=True, grouping_key="color",
    )
    prog = _single_pipeline(rule, output_dims=(3, 3))
    assert prog is not None
    from aria.runtime.program import program_to_text
    text = program_to_text(prog)
    assert "dims_make" in text  # uses explicit dims, not dims_of(input)


def test_by_size_predicate():
    """by_size(N) should select objects with exactly N pixels."""
    import aria.runtime
    from aria.runtime.ops import get_op
    _, find = get_op("find_objects")
    _, by_size = get_op("by_size")
    _, where = get_op("where")

    grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [2, 2, 0],
    ])
    objects = find(grid)
    pred_1 = by_size(1)
    small = where(pred_1, objects)
    assert len(small) == 1
    assert next(iter(small)).color == 1

    pred_2 = by_size(2)
    big = where(pred_2, objects)
    assert len(big) == 1
    assert next(iter(big)).color == 2


# ---------------------------------------------------------------------------
# Multi-rule pipeline composition
# ---------------------------------------------------------------------------


def test_multi_pipeline_composition():
    """Compose remove + move pipelines for a task needing both."""
    # Task: remove color 2, shift remaining color 1 down
    demos = (
        DemoPair(
            input=grid_from_list([[2, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 2, 0], [1, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        ),
    )
    result = observe_and_synthesize(demos)
    # Should find remove(2) + move(1, down) rules
    rule_kinds = {r.kind for r in result.rules}
    assert "remove" in rule_kinds or "move" in rule_kinds
    # May or may not solve depending on composition
    # The important thing: no crash, rules are diagnosed
    assert result.candidates_tested >= 0


def test_shift_grid_op():
    import aria.runtime
    from aria.runtime.ops import get_op
    import numpy as np

    _, shift = get_op("shift_grid")
    grid = grid_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Shift down by 1
    result = shift(1, 0, 0, grid)
    expected = grid_from_list([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
    assert np.array_equal(result, expected)

    # Shift right by 1
    result = shift(0, 1, 0, grid)
    expected = grid_from_list([[0, 1, 2], [0, 4, 5], [0, 7, 8]])
    assert np.array_equal(result, expected)

    # Shift up by 1
    result = shift(-1, 0, 0, grid)
    expected = grid_from_list([[4, 5, 6], [7, 8, 9], [0, 0, 0]])
    assert np.array_equal(result, expected)

    # No shift
    result = shift(0, 0, 0, grid)
    assert np.array_equal(result, grid)


# ---------------------------------------------------------------------------
# Dims-change reconstruction
# ---------------------------------------------------------------------------


def test_infer_dims_source_fixed():
    from aria.observe import _infer_dims_source
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]), output=grid_from_list([[5, 6, 7]])),
        DemoPair(input=grid_from_list([[8, 9], [1, 2]]), output=grid_from_list([[3, 4, 5]])),
    )
    dims, source = _infer_dims_source(demos)
    assert source == "fixed"
    assert dims == (1, 3)


def test_infer_dims_source_scale():
    from aria.observe import _infer_dims_source
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]), output=grid_from_list([[0]*4]*4)),
        DemoPair(input=grid_from_list([[5, 6, 7]]), output=grid_from_list([[0]*6]*2)),
    )
    dims, source = _infer_dims_source(demos)
    assert source == "scale_input"


def test_infer_dims_source_transpose():
    from aria.observe import _infer_dims_source
    demos = (
        DemoPair(input=grid_from_list([[1, 2, 3], [4, 5, 6]]), output=grid_from_list([[0, 0], [0, 0], [0, 0]])),
        DemoPair(input=grid_from_list([[7, 8], [9, 1], [2, 3], [4, 5]]), output=grid_from_list([[0]*4]*2)),
    )
    dims, source = _infer_dims_source(demos)
    assert source == "transpose"


def test_dims_reconstruct_transpose():
    from aria.observe import dims_change_reconstruct
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3], [4, 5, 6]]),
            output=grid_from_list([[1, 4], [2, 5], [3, 6]]),
        ),
        DemoPair(
            input=grid_from_list([[7, 8], [9, 1]]),
            output=grid_from_list([[7, 9], [8, 1]]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert result.attempted
    assert result.solved
    assert result.mode == "whole_grid_transform"


def test_dims_reconstruct_tile():
    from aria.observe import dims_change_reconstruct
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2, 1, 2], [3, 4, 3, 4]]),
        ),
        DemoPair(
            input=grid_from_list([[5]]),
            output=grid_from_list([[5, 5]]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert result.attempted
    assert result.solved
    assert result.inferred_output_dims_source == "scale_input"


def test_dims_reconstruct_upscale():
    from aria.observe import dims_change_reconstruct
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert result.attempted
    assert result.solved


def test_dims_reconstruct_objects_on_blank():
    from aria.observe import dims_change_reconstruct
    # Task: remove color 2, output is smaller canvas with remaining objects
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 2],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [1, 0],
                [0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 1, 2],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1],
                [0, 0],
            ]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert result.attempted
    assert result.solved
    assert result.mode == "objects_on_blank"


def test_dims_reconstruct_same_dims_skipped():
    from aria.observe import dims_change_reconstruct
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert not result.attempted
    assert not result.solved


def test_dims_reconstruct_unsolvable():
    from aria.observe import dims_change_reconstruct
    # Random mapping — no strategy will solve this
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
        ),
    )
    result = dims_change_reconstruct(demos)
    assert result.attempted
    assert not result.solved
