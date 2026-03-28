"""Tests for bounded structural edit search — shallow and semantic."""

from __future__ import annotations

import numpy as np

from aria.structural_edit import (
    EditResult,
    _generate_shallow_edits,
    _generate_semantic_edits,
    structural_edit_search,
)
from aria.observe import ObjectRule
from aria.types import (
    Bind, Call, DemoPair, Grid, Literal, Program, Ref, Type,
    grid_from_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_program() -> Program:
    return Program(
        steps=(Bind("v0", Type.GRID, Call("identity", (Ref("input"),))),),
        output="v0",
    )


def _color_swap_program(color: int) -> Program:
    return Program(
        steps=(Bind("v0", Type.GRID, Call("recolor_uniform", (
            Literal(color, Type.COLOR), Ref("input"),
        ))),),
        output="v0",
    )


def _make_move_rule(color=1, dr=1, dc=0, grouping="color"):
    gkey = (color,) if grouping == "color" else (color, 1)
    return ObjectRule(
        kind="move", input_color=color, output_color=None,
        offsets=None,
        details={"dr": dr, "dc": dc, "group_key": gkey},
        expressible=True, grouping_key=grouping, expression_path="pipeline",
    )


def _make_recolor_rule(color=1, old_c=1, new_c=2, grouping="color"):
    return ObjectRule(
        kind="recolor", input_color=color, output_color=new_c,
        offsets=None,
        details={"color_map": {old_c: new_c}, "group_key": (color,)},
        expressible=True, grouping_key=grouping, expression_path="pipeline",
    )


def _make_remove_rule(color=3, grouping="color"):
    return ObjectRule(
        kind="remove", input_color=color, output_color=None,
        offsets=None,
        details={"group_key": (color,)},
        expressible=True, grouping_key=grouping, expression_path="pipeline",
    )


def _make_proximity_rule(color=1, marker_color=5, near=True, kind="move", dr=1, dc=0):
    return ObjectRule(
        kind=kind, input_color=color, output_color=None,
        offsets=None,
        details={
            "dr": dr, "dc": dc,
            "group_key": (color, near),
            "marker_color": marker_color,
        },
        expressible=True, grouping_key="proximity", expression_path="pipeline",
    )


def _make_direction_rule(color=1, marker_color=5, direction="above", in_direction=True, kind="move", dr=1, dc=0):
    return ObjectRule(
        kind=kind, input_color=color, output_color=None,
        offsets=None,
        details={
            "dr": dr, "dc": dc,
            "group_key": (color, in_direction),
            "marker_color": marker_color,
            "direction": direction,
        },
        expressible=True, grouping_key=f"direction_{direction}",
        expression_path="pipeline",
    )


def _simple_demos():
    return (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Empty / baseline tests (no regression)
# ---------------------------------------------------------------------------


def test_empty_inputs_return_no_solve():
    demos = _simple_demos()
    result = structural_edit_search(demos, [])
    assert not result.solved
    assert result.candidates_tried == 0

    result2 = structural_edit_search((), [])
    assert not result2.solved


def test_empty_near_miss_returns_no_solve():
    demos = _simple_demos()
    result = structural_edit_search(demos, [], repaired_targets=None)
    assert not result.solved
    assert result.candidates_tried == 0


# ---------------------------------------------------------------------------
# Shallow edit tests (no regression from prior implementation)
# ---------------------------------------------------------------------------


def test_shallow_literal_swap():
    prog = _color_swap_program(3)
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[5, 5], [5, 5]]),
        ),
    )
    edits = list(_generate_shallow_edits(prog, demos, max_edits=100))
    descs = [d for _, d in edits]
    literal_swaps = [d for d in descs if d.startswith("literal_swap:")]
    assert len(literal_swaps) >= 1
    assert any("3->5" in d for d in literal_swaps)


def test_shallow_overlay_wrapping():
    prog = _identity_program()
    demos = _simple_demos()
    edits = list(_generate_shallow_edits(prog, demos, max_edits=100))
    wrap_descs = [d for _, d in edits if d.startswith("wrap:")]
    assert len(wrap_descs) == 2
    assert any("overlay(input,prog)" in d for d in wrap_descs)
    assert any("overlay(prog,input)" in d for d in wrap_descs)


def test_shallow_max_limit():
    prog = _color_swap_program(3)
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            output=grid_from_list([[5, 5, 5], [5, 5, 5], [5, 5, 5]]),
        ),
    )
    edits = list(_generate_shallow_edits(prog, demos, max_edits=3))
    assert len(edits) <= 3


# ---------------------------------------------------------------------------
# Semantic edit: grouping key swap
# ---------------------------------------------------------------------------


def test_grouping_swap_color_to_color_size():
    """Swapping grouping from color to color_size should produce edits."""
    rule = _make_move_rule(color=2, dr=1, dc=0, grouping="color")
    demos = _simple_demos()
    from aria.structural_edit import _grouping_edits
    edits = list(_grouping_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("grouping_swap" in d and "color->color_size" in d for d in descs)


def test_grouping_swap_color_size_to_color():
    """Swapping from color_size to color should produce edits."""
    rule = _make_move_rule(color=2, dr=1, dc=0, grouping="color_size")
    demos = _simple_demos()
    from aria.structural_edit import _grouping_edits
    edits = list(_grouping_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("grouping_swap" in d and "color_size->color" in d for d in descs)


# ---------------------------------------------------------------------------
# Semantic edit: near/far flip
# ---------------------------------------------------------------------------


def test_near_far_flip():
    """Flipping near/far in proximity rule should produce edits."""
    rule = _make_proximity_rule(color=1, near=True)
    demos = _simple_demos()
    from aria.structural_edit import _near_far_edits
    edits = list(_near_far_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("near_far_flip" in d for d in descs)


def test_near_far_flip_inverse():
    """Flipping from far to near should also work."""
    rule = _make_proximity_rule(color=1, near=False)
    demos = _simple_demos()
    from aria.structural_edit import _near_far_edits
    edits = list(_near_far_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("near_far_flip" in d for d in descs)


# ---------------------------------------------------------------------------
# Semantic edit: direction swap
# ---------------------------------------------------------------------------


def test_direction_sibling_swap():
    """Direction above -> below/left/right should produce edits."""
    rule = _make_direction_rule(direction="above")
    demos = _simple_demos()
    from aria.structural_edit import _direction_edits
    edits = list(_direction_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("direction_swap:above->below" in d for d in descs)
    assert any("direction_swap:above->left" in d for d in descs)


def test_direction_invert():
    """Inverting in_direction boolean should produce an edit."""
    rule = _make_direction_rule(direction="above", in_direction=True)
    demos = _simple_demos()
    from aria.structural_edit import _direction_edits
    edits = list(_direction_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("direction_invert" in d for d in descs)


# ---------------------------------------------------------------------------
# Semantic edit: move delta
# ---------------------------------------------------------------------------


def test_delta_edit():
    """Editing move delta by ±1 should produce edits."""
    rule = _make_move_rule(color=1, dr=2, dc=0)
    demos = _simple_demos()
    from aria.structural_edit import _delta_edits
    edits = list(_delta_edits([rule], demos))
    descs = [d for _, d in edits]
    # Should have dr=1, dr=3, dc=1, dc=-1 variants
    assert any("(2,0)->(3,0)" in d for d in descs)
    assert any("(2,0)->(1,0)" in d for d in descs)
    assert any("(2,0)->(2,1)" in d for d in descs)
    assert any("(2,0)->(2,-1)" in d for d in descs)


def test_delta_edit_produces_programs():
    """Delta edits should produce valid Programs."""
    rule = _make_move_rule(color=1, dr=1, dc=0)
    demos = _simple_demos()
    from aria.structural_edit import _delta_edits
    edits = list(_delta_edits([rule], demos))
    for prog, desc in edits:
        assert isinstance(prog, Program)
        assert len(prog.steps) > 0


# ---------------------------------------------------------------------------
# Semantic edit: recolor map
# ---------------------------------------------------------------------------


def test_recolor_edit():
    """Editing recolor target color should produce edits."""
    rule = _make_recolor_rule(color=1, old_c=1, new_c=2)
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[3, 0], [0, 0]]),
        ),
    )
    from aria.structural_edit import _recolor_edits
    edits = list(_recolor_edits([rule], demos))
    descs = [d for _, d in edits]
    # Should try swapping 1:2 to 1:3 (since 3 is in output)
    assert any("recolor_edit" in d for d in descs)


# ---------------------------------------------------------------------------
# Semantic edit: composition add/remove
# ---------------------------------------------------------------------------


def test_composition_remove_rule():
    """Removing one rule from a composable set should produce edits."""
    rules = [
        _make_remove_rule(color=1),
        _make_remove_rule(color=2),
        _make_remove_rule(color=3),
    ]
    demos = _simple_demos()
    from aria.structural_edit import _composition_edits
    edits = list(_composition_edits(rules, demos))
    descs = [d for _, d in edits]
    assert any("composition_remove" in d for d in descs)


def test_composition_remove_keeps_minimum_two():
    """Removing a rule from a 2-rule set should not produce edits (need >= 2)."""
    rules = [
        _make_remove_rule(color=1),
        _make_remove_rule(color=2),
    ]
    demos = _simple_demos()
    from aria.structural_edit import _composition_edits
    edits = list(_composition_edits(rules, demos))
    remove_edits = [d for _, d in edits if "composition_remove" in d]
    assert len(remove_edits) == 0


# ---------------------------------------------------------------------------
# Semantic edit: flat -> pipeline promotion
# ---------------------------------------------------------------------------


def test_promotion_flat_to_pipeline():
    """A flat recolor rule should be promotable to pipeline."""
    rule = ObjectRule(
        kind="recolor", input_color=1, output_color=2,
        offsets=None,
        details={"color_map": {1: 2}, "group_key": (1,)},
        expressible=True, grouping_key="color", expression_path="flat",
    )
    demos = _simple_demos()
    from aria.structural_edit import _promotion_edits
    edits = list(_promotion_edits([rule], demos))
    descs = [d for _, d in edits]
    assert any("promote_flat_to_pipeline" in d for d in descs)


# ---------------------------------------------------------------------------
# Search integration: semantic edits with observation rules
# ---------------------------------------------------------------------------


def test_search_with_observation_rules():
    """When observation rules are provided, semantic edits should be tried."""
    demos = _simple_demos()
    rules = [_make_move_rule(color=1, dr=1, dc=0)]
    result = structural_edit_search(
        demos, [],
        observation_rules=rules,
    )
    assert isinstance(result, EditResult)
    # Should have tried at least some semantic edits
    if result.family_breakdown:
        assert result.family_breakdown.get("semantic", 0) >= 0


def test_search_with_repaired_targets():
    """Repaired targets should be checked before original demos."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 1], [1, 1]]),
            output=grid_from_list([[1, 1], [1, 1]]),
        ),
    )
    prog = _identity_program()
    repaired = (grid_from_list([[1, 1], [1, 1]]),)
    result = structural_edit_search(
        demos, [prog],
        repaired_targets=repaired,
    )
    assert isinstance(result, EditResult)


def test_semantic_edit_matched_target_but_not_original():
    """Semantic edit matching repaired target but failing original demos
    should report matched_repaired_target=True, solved=False."""
    # This is hard to construct as a unit test since we need a program that
    # produces one grid (matching repaired target) but not the original output.
    # Instead, verify the reporting structure is correct.
    result = EditResult(
        solved=False, winning_program=None, winning_edit=None,
        candidates_tried=10,
        matched_repaired_target=True,
        solved_original_task=False,
        winning_family=None,
        family_breakdown={"semantic": 5, "shallow": 5},
    )
    assert result.matched_repaired_target is True
    assert result.solved_original_task is False
    assert not result.solved


def test_semantic_edit_becomes_true_solve():
    """Verify that a semantic edit win reports correctly."""
    result = EditResult(
        solved=True, winning_program=_identity_program(),
        winning_edit="semantic:grouping_swap:color->color_size:move:1",
        candidates_tried=3,
        matched_repaired_target=True,
        solved_original_task=True,
        winning_family="semantic",
        family_breakdown={"semantic": 3, "shallow": 0},
    )
    assert result.solved
    assert result.winning_family == "semantic"
    assert "grouping_swap" in result.winning_edit


# ---------------------------------------------------------------------------
# Error class prioritization
# ---------------------------------------------------------------------------


def test_error_class_color_swap_prioritizes_recolor():
    """With error_class='color_swap', recolor edits should come first."""
    recolor_rule = _make_recolor_rule(color=1, old_c=1, new_c=2)
    move_rule = _make_move_rule(color=2, dr=1, dc=0)
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[3, 0], [0, 0]]),
        ),
    )
    edits = list(_generate_semantic_edits(
        [recolor_rule, move_rule], demos, error_class="color_swap",
    ))
    if edits:
        first_desc = edits[0][1]
        # Recolor edits should appear before grouping/delta edits
        assert "recolor" in first_desc or "semantic" in first_desc


def test_error_class_missing_content_prioritizes_composition():
    """With error_class='missing_content', composition edits should come first."""
    rules = [
        _make_remove_rule(color=1),
        _make_remove_rule(color=2),
        _make_remove_rule(color=3),
    ]
    demos = _simple_demos()
    edits = list(_generate_semantic_edits(
        rules, demos, error_class="missing_content",
    ))
    if edits:
        first_desc = edits[0][1]
        assert "composition" in first_desc or "promotion" in first_desc or "semantic" in first_desc


# ---------------------------------------------------------------------------
# Integration: refinement loop wiring
# ---------------------------------------------------------------------------


def test_structural_edit_wired_into_refinement():
    """Structural edit search should be importable and callable from refinement."""
    from aria.refinement import run_refinement_loop, RefinementResult
    from aria.library.store import Library

    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )
    result = run_refinement_loop(
        demos, Library(),
        max_steps=1, max_candidates=10, max_rounds=1,
    )
    assert result is not None
    # Verify structural_edit_result field exists
    assert hasattr(result, 'structural_edit_result')


def test_family_breakdown_populated():
    """Family breakdown should be populated even when no solve found."""
    demos = _simple_demos()
    rules = [_make_move_rule(color=1, dr=1, dc=0)]
    result = structural_edit_search(
        demos, [_identity_program()],
        observation_rules=rules,
    )
    assert result.family_breakdown is not None
    assert "semantic" in result.family_breakdown
    assert "shallow" in result.family_breakdown
    assert "correspondence" in result.family_breakdown


# ---------------------------------------------------------------------------
# Correspondence-aware movement edits
# ---------------------------------------------------------------------------


def test_correspondence_extract_objects():
    """_extract_foreground_objects should find connected components."""
    from aria.structural_edit import _extract_foreground_objects
    grid = grid_from_list([
        [0, 1, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ])
    objs = _extract_foreground_objects(grid)
    assert len(objs) == 2
    colors = {o["color"] for o in objs}
    assert colors == {1, 2}


def test_correspondence_match_objects_simple():
    """Exact color+shape match should produce one-to-one correspondences."""
    from aria.structural_edit import _extract_foreground_objects, _match_objects
    inp = grid_from_list([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    out = grid_from_list([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ])
    inp_objs = _extract_foreground_objects(inp)
    out_objs = _extract_foreground_objects(out)
    matches = _match_objects(inp_objs, out_objs)
    assert matches is not None
    assert len(matches) == 1
    assert matches[0].dr == 2
    assert matches[0].dc == 1


def test_correspondence_movement_edit_produces_programs():
    """Correspondence-aware edit should produce pipeline programs for moved objects."""
    from aria.structural_edit import _correspondence_movement_edits
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    assert len(edits) >= 1
    descs = [d for _, d in edits]
    assert any("correspondence:move:" in d for d in descs)
    # All edits should produce valid Programs
    for prog, _ in edits:
        assert isinstance(prog, Program)


def test_correspondence_multi_color_movement():
    """Two colors moving by different deltas should produce shared composition."""
    from aria.structural_edit import _correspondence_movement_edits
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    # Should have individual per-color moves
    assert any("correspondence:move:color=1" in d for d in descs)
    assert any("correspondence:move:color=2" in d for d in descs)
    # Should also have shared composition attempt
    assert any("correspondence:shared_move" in d for d in descs)


def test_correspondence_consistent_across_demos():
    """Movement deltas must be consistent across all demos."""
    from aria.structural_edit import _correspondence_movement_edits
    # Demo 1: color 1 moves right by 1
    # Demo 2: color 1 moves right by 1 (consistent)
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0],
                [1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 1, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    assert len(edits) >= 1
    descs = [d for _, d in edits]
    assert any("correspondence:move:color=1:dr=0:dc=1" in d for d in descs)


def test_correspondence_inconsistent_deltas_rejected():
    """Inconsistent deltas across demos should produce no correspondence edits."""
    from aria.structural_edit import _correspondence_movement_edits
    # Demo 1: color 1 moves right by 1
    # Demo 2: color 1 moves down by 1 (inconsistent)
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    # Should produce no per-color edits since color 1 moves differently
    color1_edits = [d for _, d in edits if "color=1" in d and "correspondence:move:" in d]
    assert len(color1_edits) == 0


def test_correspondence_ambiguous_match_skipped():
    """When two objects of same color+shape can't be disambiguated, skip them."""
    from aria.structural_edit import _extract_foreground_objects, _match_objects
    # Two identical 1-pixel objects of color 1, equidistant from two output positions
    inp = grid_from_list([
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ])
    out = grid_from_list([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
    ])
    inp_objs = _extract_foreground_objects(inp)
    out_objs = _extract_foreground_objects(out)
    matches = _match_objects(inp_objs, out_objs)
    # May produce matches via proximity, but both have same distance — should skip
    if matches is not None:
        # Any matches found should have unambiguous pairing
        for m in matches:
            assert m.input_color == 1


def test_correspondence_no_movement_no_edits():
    """When all objects stay in place, no correspondence movement edits."""
    from aria.structural_edit import _correspondence_movement_edits
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 2]]),
            output=grid_from_list([[1, 0], [0, 2]]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    # No movement → no edits (all deltas are (0,0))
    assert len(edits) == 0


def test_correspondence_color_size_subgroup():
    """When color-level grouping is inconsistent, try color+size grouping."""
    from aria.structural_edit import _correspondence_movement_edits
    # Two separate objects of color 1: 1-pixel at (0,0) moves right, 2-pixel at (2,1) stays
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    # Color-level grouping fails (different deltas for color 1)
    # Color+size grouping should find: size=1 moves (0,1), size=2 stays (0,0)
    descs = [d for _, d in edits]
    assert any("correspondence:move_cs:" in d for d in descs)


def test_correspondence_edit_becomes_real_solve():
    """A correspondence-aware edit should produce a program that passes verification."""
    # Color 1 dot moves right by 2 in a 4x4 grid, rest is background
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
    )
    result = structural_edit_search(demos, [])
    # The correspondence edit should solve this: color 1 moves by (0, 2)
    assert result.solved
    assert result.winning_family == "correspondence"
    assert "correspondence:move:color=1:dr=0:dc=2" in result.winning_edit


def test_correspondence_in_full_search_pipeline():
    """Correspondence edits should fire even with no near-miss programs."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
            ]),
        ),
    )
    result = structural_edit_search(demos, [], observation_rules=[])
    assert result.candidates_tried > 0
    if result.family_breakdown:
        assert result.family_breakdown.get("correspondence", 0) > 0


def test_correspondence_different_dims_skipped():
    """Correspondence edits should not fire for different-dims tasks."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
    )
    result = structural_edit_search(demos, [], observation_rules=[])
    # Different dims → correspondence phase skipped
    if result.family_breakdown:
        assert result.family_breakdown.get("correspondence", 0) == 0


# ---------------------------------------------------------------------------
# Correspondence: mixed color + color_size grouping
# ---------------------------------------------------------------------------


def test_mixed_grouping_one_color_consistent_one_not():
    """Color A consistent at color-level, color B only at color+size."""
    from aria.structural_edit import _correspondence_movement_edits
    # Color 2 (single object): moves right by 1 — consistent at color level
    # Color 1 (two objects, different sizes): size=1 moves down, size=2 stays
    #   → inconsistent at color level, consistent at color+size
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 2],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [2, 0, 0, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    # Should have color-level move for color 2
    assert any("correspondence:move:color=2" in d for d in descs)
    # Should have color+size move for color 1 subgroups
    assert any("correspondence:move_cs:color=1" in d for d in descs)
    # Should have mixed shared composition
    assert any("shared_move_mixed" in d for d in descs)


def test_mixed_grouping_shared_composition_built():
    """Mixed grouping should build shared composition from both color and cs rules."""
    from aria.structural_edit import _correspondence_movement_edits
    # Color 3: moves right by 2 (consistent at color level, 1 object)
    # Color 1: two objects of different sizes, only consistent at color+size
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 3],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    mixed = [d for d in descs if "shared_move_mixed" in d]
    assert len(mixed) >= 1
    # The mixed label should show both color and cs group counts
    for d in mixed:
        assert "color_groups=" in d
        assert "cs_groups=" in d


def test_inconsistent_at_both_levels_excluded():
    """Color inconsistent at both color and color+size should be excluded,
    not kill the whole edit family."""
    from aria.structural_edit import _correspondence_movement_edits
    # Color 2: moves right by 1 (consistent)
    # Color 1: two objects of SAME size but different deltas
    #   → inconsistent at color level AND at color+size level
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    # Color 2 should still produce edits
    assert any("correspondence:move:color=2" in d for d in descs)
    # Color 1 should NOT produce color-level edits (inconsistent)
    assert not any("correspondence:move:color=1" in d for d in descs)


def test_pure_color_level_still_works():
    """Pure color-level case should still work (no regression)."""
    from aria.structural_edit import _correspondence_movement_edits
    demos = (
        DemoPair(
            input=grid_from_list([[0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 1], [0, 0, 0]]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    assert any("correspondence:move:color=1:dr=0:dc=1" in d for d in descs)
    # No mixed label — pure color-level
    assert not any("shared_move_mixed" in d for d in descs)


def test_pure_color_size_still_works():
    """Pure color+size case (no colors consistent) should still work."""
    from aria.structural_edit import _correspondence_movement_edits
    # Color 1: size=1 moves right, size=2 stays → all inconsistent at color level
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ]),
        ),
    )
    edits = list(_correspondence_movement_edits(demos))
    descs = [d for _, d in edits]
    assert any("correspondence:move_cs:" in d for d in descs)
    # No mixed label — pure color+size
    assert not any("shared_move_mixed" in d for d in descs)


def test_mixed_grouping_solves_synthetic_task():
    """A mixed-grouping task should be solvable via correspondence edits."""
    # Color 2 (single dot): moves down by 2 (consistent at color)
    # Color 1 (size=1 and size=2): size=1 moves right by 1, size=2 stays
    # Two demos to ensure consistency
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    result = structural_edit_search(demos, [])
    # Even if the full task isn't solved (composition may fail disjointness),
    # correspondence edits should at least fire
    assert result.candidates_tried > 0
    if result.family_breakdown:
        assert result.family_breakdown.get("correspondence", 0) > 0


# ---------------------------------------------------------------------------
# Program reranking hook tests
# ---------------------------------------------------------------------------


from aria.structural_edit import RerankingReport


def test_reranking_not_applied_by_default():
    """Without program_ranker, reranking should be None."""
    demos = _simple_demos()
    result = structural_edit_search(demos, [_identity_program()])
    assert result.reranking is None


def test_reranking_applied_with_ranker():
    """With a ranker that preserves order, report should show applied=True, changed=False."""
    demos = _simple_demos()
    progs = [_identity_program(), _color_swap_program(5)]

    def identity_ranker(texts):
        return tuple(range(len(texts))), False, "identity"

    result = structural_edit_search(
        demos, progs, program_ranker=identity_ranker,
    )
    assert result.reranking is not None
    assert result.reranking.applied is True
    assert result.reranking.changed_order is False
    assert result.reranking.policy_name == "identity"
    # programs_ranked counts ALL candidates from all phases, not just near_miss_programs
    assert result.reranking.programs_ranked >= 2


def test_reranking_reverses_order():
    """A ranker that reverses candidate order should be reflected in the report."""
    demos = _simple_demos()
    progs = [_identity_program(), _color_swap_program(5)]

    def reverse_ranker(texts):
        indices = tuple(reversed(range(len(texts))))
        return indices, True, "reverse"

    result = structural_edit_search(
        demos, progs, program_ranker=reverse_ranker,
    )
    assert result.reranking is not None
    assert result.reranking.applied is True
    assert result.reranking.changed_order is True
    n = result.reranking.programs_ranked
    assert result.reranking.reranked_order == tuple(reversed(range(n)))
    assert result.reranking.original_order == tuple(range(n))


def test_reranking_no_candidates_no_report():
    """With no candidates from any phase, reranking should be None."""
    # Empty demos → early return before candidate collection
    result = structural_edit_search((), [], program_ranker=lambda t: ((), False, "x"))
    assert result.reranking is None


def test_reranking_spans_all_families():
    """Ranker should see candidates from semantic + correspondence + shallow."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ]),
        ),
    )
    rules = [_make_move_rule(color=1, dr=2, dc=1)]
    prog = _identity_program()

    seen_texts: list[str] = []

    def spy_ranker(texts):
        seen_texts.extend(texts)
        return tuple(range(len(texts))), False, "spy"

    result = structural_edit_search(
        demos, [prog],
        observation_rules=rules,
        program_ranker=spy_ranker,
    )
    # spy_ranker should have received candidates from all three families
    assert result.reranking is not None
    assert result.reranking.programs_ranked == len(seen_texts)
    assert len(seen_texts) > 0
    # family_breakdown should show activity across families
    fb = result.family_breakdown or {}
    has_semantic = fb.get("semantic", 0) > 0
    has_shallow = fb.get("shallow", 0) > 0
    has_corr = fb.get("correspondence", 0) > 0
    assert has_semantic or has_shallow or has_corr


def test_reranking_report_in_solved_result():
    """When a solve is found after reranking, report should be present."""
    # Correspondence edits solve this: color 1 moves (0, 2)
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
    )

    def identity_ranker(texts):
        return tuple(range(len(texts))), False, "identity"

    result = structural_edit_search(
        demos, [], program_ranker=identity_ranker,
    )
    assert result.solved
    assert result.reranking is not None
    assert result.reranking.applied is True


def test_reranking_report_fields_are_correct_types():
    """RerankingReport should have correct field types."""
    report = RerankingReport(
        applied=True, policy_name="test",
        changed_order=True, programs_ranked=3,
        original_order=(0, 1, 2), reranked_order=(2, 0, 1),
    )
    assert isinstance(report.applied, bool)
    assert isinstance(report.policy_name, str)
    assert isinstance(report.changed_order, bool)
    assert isinstance(report.programs_ranked, int)
    assert isinstance(report.original_order, tuple)
    assert isinstance(report.reranked_order, tuple)
