"""Tests for correspondence-driven object movement synthesis."""

from __future__ import annotations

import numpy as np

from aria.observe import (
    ObjectRule,
    observe_and_synthesize,
    _correspondence_move_analysis,
    _anchor_alignment_pipeline,
    _check_anchor_alignment_axis,
)
from aria.structural_edit import (
    _extract_foreground_objects,
    _match_objects,
    _find_consistent_color_deltas,
    _find_consistent_color_size_deltas,
)
from aria.types import DemoPair, grid_from_list
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _uniform_move_task():
    """All color-2 objects move right by 2. Standard observation fails
    because delta.py matching gets confused by multiple same-size objects.
    Correspondence should handle it because masks differ."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [2, 2, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 0, 0],
                [0, 0, 0, 2, 0, 0],
            ]),
        ),
    )


def _anchor_col_align_task():
    """Objects of color 2 align their center col to a stationary anchor (color 3).
    Each color-2 object starts at a different col, so the deltas differ per
    object but are all explained by anchor col alignment (anchor at col 3)."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 3, 0, 0],
                [0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
            ]),
            output=grid_from_list([
                [0, 0, 0, 3, 0, 0],
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 3, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 3, 0, 0],
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0],
            ]),
        ),
    )


def _anchor_row_align_task():
    """Objects of color 2 align their center row to a stationary anchor (color 3).
    The anchor sits at row 2, and all color-2 objects move vertically to align.
    Objects are placed so they don't merge into one CC in the output."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [3, 2, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [3, 0, 0, 2, 0, 2],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
    )


def _ambiguous_task():
    """Two objects of color 2 with identical shape, same distance from candidates.
    Should not produce incorrect matches."""
    return (
        DemoPair(
            input=grid_from_list([
                [0, 2, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 2, 0, 2],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# Tests: correspondence matching basics
# ---------------------------------------------------------------------------


def test_match_objects_unique_shapes():
    """Objects with different shapes match unambiguously."""
    inp = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
    ])
    out = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
    ])
    inp_objs = _extract_foreground_objects(inp)
    out_objs = _extract_foreground_objects(out)
    matches = _match_objects(inp_objs, out_objs)
    assert matches is not None
    assert len(matches) >= 2

    color2 = [m for m in matches if m.input_color == 2]
    assert len(color2) == 1
    assert color2[0].dr == 0
    assert color2[0].dc == 2


def test_match_objects_position_ordered():
    """Same-shape objects matched by proximity."""
    demos = _ambiguous_task()
    inp_objs = _extract_foreground_objects(demos[0].input)
    out_objs = _extract_foreground_objects(demos[0].output)
    matches = _match_objects(inp_objs, out_objs)
    # Should match (both are 1x1 color 2) — proximity-based
    assert matches is not None
    color2 = [m for m in matches if m.input_color == 2]
    assert len(color2) == 2
    # Both should show dc=+1
    for m in color2:
        assert m.dc == 1
        assert m.dr == 0


def test_consistent_color_deltas():
    """Uniform per-color deltas detected from correspondences."""
    demos = _uniform_move_task()
    per_demo = []
    for d in demos:
        inp_objs = _extract_foreground_objects(d.input)
        out_objs = _extract_foreground_objects(d.output)
        matches = _match_objects(inp_objs, out_objs)
        assert matches is not None
        per_demo.append(matches)

    consistent, inconsistent = _find_consistent_color_deltas(per_demo)
    assert 2 in consistent
    assert consistent[2] == (0, 2)


# ---------------------------------------------------------------------------
# Tests: correspondence-driven synthesis
# ---------------------------------------------------------------------------


def test_correspondence_uniform_move_synthesis():
    """Correspondence finds uniform delta and synthesizes a working program."""
    demos = _uniform_move_task()
    corr = _correspondence_move_analysis(demos)
    assert corr is not None
    assert len(corr.rules) >= 1

    move_rules = [r for r in corr.rules if r.kind == "move" and r.input_color == 2]
    assert len(move_rules) >= 1
    assert move_rules[0].details["dr"] == 0
    assert move_rules[0].details["dc"] == 2

    # At least one program should verify
    any_pass = False
    for prog, rule in corr.programs:
        vr = verify(prog, demos)
        if vr.passed:
            any_pass = True
            break
    assert any_pass


def test_correspondence_uniform_move_end_to_end():
    """observe_and_synthesize solves uniform move via correspondence fallback."""
    demos = _uniform_move_task()
    result = observe_and_synthesize(demos)
    assert result.solved


# ---------------------------------------------------------------------------
# Tests: anchor alignment
# ---------------------------------------------------------------------------


def test_anchor_col_alignment_detection():
    """Anchor alignment on col axis is detected from correspondence data."""
    demos = _anchor_col_align_task()

    per_demo_matches = []
    per_demo_inp_objs = []
    for d in demos:
        inp_objs = _extract_foreground_objects(d.input)
        out_objs = _extract_foreground_objects(d.output)
        matches = _match_objects(inp_objs, out_objs)
        assert matches is not None
        per_demo_matches.append(matches)
        per_demo_inp_objs.append(inp_objs)

    consistent, inconsistent = _find_consistent_color_deltas(per_demo_matches)
    # Color 3 (anchor) should be stationary
    assert 3 in consistent
    assert consistent[3] == (0, 0)
    # Color 2 should be inconsistent (different objects move differently)
    assert 2 in inconsistent

    ok = _check_anchor_alignment_axis(
        per_demo_matches, per_demo_inp_objs,
        anchor_color=3, moved_color=2, axis="col",
    )
    assert ok, "Column alignment should be detected"


def test_anchor_col_alignment_synthesis():
    """Anchor-aligned col movement synthesizes a correct program."""
    demos = _anchor_col_align_task()
    prog = _anchor_alignment_pipeline(moved_color=2, anchor_color=3, axis="col")
    assert prog is not None
    vr = verify(prog, demos)
    assert vr.passed, f"Anchor col alignment program should pass: {vr}"


def test_anchor_row_alignment_detection():
    """Anchor alignment on row axis is detected.

    This tests the merge-tolerant path: in the row-alignment task,
    moved objects can land adjacent and merge in the output grid.
    The checker uses output-grid verification for unmatched objects.
    """
    demos = _anchor_row_align_task()

    per_demo_matches = []
    per_demo_inp_objs = []
    for d in demos:
        inp_objs = _extract_foreground_objects(d.input)
        out_objs = _extract_foreground_objects(d.output)
        matches = _match_objects(inp_objs, out_objs)
        # Some demos may have partial matches (output merges)
        if matches is None:
            matches = []
        per_demo_matches.append(matches)
        per_demo_inp_objs.append(inp_objs)

    ok = _check_anchor_alignment_axis(
        per_demo_matches, per_demo_inp_objs,
        anchor_color=3, moved_color=2, axis="row",
        demos=demos,
    )
    assert ok, "Row alignment should be detected"


def test_anchor_row_alignment_synthesis():
    """Anchor-aligned row movement synthesizes a correct program."""
    demos = _anchor_row_align_task()
    prog = _anchor_alignment_pipeline(moved_color=2, anchor_color=3, axis="row")
    assert prog is not None
    vr = verify(prog, demos)
    assert vr.passed, f"Anchor row alignment program should pass: {vr}"


def test_anchor_alignment_end_to_end():
    """observe_and_synthesize solves anchor-aligned task via correspondence."""
    demos = _anchor_col_align_task()
    result = observe_and_synthesize(demos)
    assert result.solved
    # Check that correspondence source is in the rules
    corr_rules = [r for r in result.rules if r.details.get("anchor_alignment")]
    assert len(corr_rules) >= 1


# ---------------------------------------------------------------------------
# Tests: conservative rejection
# ---------------------------------------------------------------------------


def test_wrong_axis_rejected():
    """Checking wrong axis returns False."""
    demos = _anchor_col_align_task()
    per_demo_matches = []
    per_demo_inp_objs = []
    for d in demos:
        inp_objs = _extract_foreground_objects(d.input)
        out_objs = _extract_foreground_objects(d.output)
        matches = _match_objects(inp_objs, out_objs)
        per_demo_matches.append(matches)
        per_demo_inp_objs.append(inp_objs)

    # This task aligns on col, so row alignment should fail
    ok = _check_anchor_alignment_axis(
        per_demo_matches, per_demo_inp_objs,
        anchor_color=3, moved_color=2, axis="row",
    )
    assert not ok, "Wrong axis should be rejected"


def test_no_correspondence_for_dims_change():
    """Correspondence analysis returns None for dims-change tasks."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[1, 2, 1, 2], [3, 4, 3, 4]]),
        ),
    )
    assert _correspondence_move_analysis(demos) is None


def test_no_correspondence_when_objects_disappear():
    """If an object disappears (no match), matcher skips it gracefully."""
    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 2, 0],
                [0, 0, 0],
                [0, 3, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0],
                [0, 0, 0],
                [0, 3, 0],
            ]),
        ),
    )
    # Color 2 disappears — correspondence should still work but no movement
    corr = _correspondence_move_analysis(demos)
    # Should return None since no actual movement detected
    assert corr is None


# ---------------------------------------------------------------------------
# Tests: no regression for existing observation families
# ---------------------------------------------------------------------------


def test_surround_still_works():
    """Surround synthesis is unaffected by correspondence addition."""
    demos = (
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
    result = observe_and_synthesize(demos)
    assert result.solved


def test_recolor_still_works():
    """Recolor synthesis is unaffected by correspondence addition."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [0, 1]]),
            output=grid_from_list([[5, 6], [0, 5]]),
        ),
        DemoPair(
            input=grid_from_list([[2, 0], [1, 2]]),
            output=grid_from_list([[6, 0], [5, 6]]),
        ),
    )
    result = observe_and_synthesize(demos)
    assert result.solved


# ---------------------------------------------------------------------------
# Tests: align_center_to ops
# ---------------------------------------------------------------------------


def test_align_center_to_col_of_op():
    """The align_center_to_col_of op correctly translates object center."""
    from aria.runtime.ops import get_op
    from aria.types import ObjectNode, Shape

    _, align_fn = get_op("align_center_to_col_of")

    anchor = ObjectNode(
        id=0, color=3,
        mask=np.array([[True]], dtype=np.bool_),
        bbox=(5, 2, 1, 1),  # col=5, row=2
        shape=Shape.DOT, symmetry=frozenset(), size=1,
    )
    obj = ObjectNode(
        id=1, color=2,
        mask=np.array([[True]], dtype=np.bool_),
        bbox=(1, 3, 1, 1),  # col=1, row=3
        shape=Shape.DOT, symmetry=frozenset(), size=1,
    )
    transform = align_fn(anchor)
    moved = transform(obj)
    # Object center col should now be 5 (anchor's center col)
    assert moved.bbox[0] == 5  # new col = 5


def test_align_center_to_row_of_op():
    """The align_center_to_row_of op correctly translates object center."""
    from aria.runtime.ops import get_op
    from aria.types import ObjectNode, Shape

    _, align_fn = get_op("align_center_to_row_of")

    anchor = ObjectNode(
        id=0, color=3,
        mask=np.array([[True]], dtype=np.bool_),
        bbox=(2, 4, 1, 1),  # col=2, row=4
        shape=Shape.DOT, symmetry=frozenset(), size=1,
    )
    obj = ObjectNode(
        id=1, color=2,
        mask=np.array([[True]], dtype=np.bool_),
        bbox=(3, 1, 1, 1),  # col=3, row=1
        shape=Shape.DOT, symmetry=frozenset(), size=1,
    )
    transform = align_fn(anchor)
    moved = transform(obj)
    # Object center row should now be 4 (anchor's center row)
    assert moved.bbox[1] == 4  # new row = 4


# ---------------------------------------------------------------------------
# Tests: composite role-normalized correspondence
# ---------------------------------------------------------------------------


def _color_rotated_frame_center_task():
    """Two demos with identical structure but different color assignments.
    Each demo has:
    - bg color (most common)
    - frame color: forms shapes around center pixels
    - center color: singletons inside frames + one isolated anchor
    The anchor is at the bottom, objects align to its column.

    Demo 0: bg=0, frame=8, center=4, anchor at (5,3)
    Demo 1: bg=0, frame=3, center=1, anchor at (5,3)
    """
    # Demo 0: frame=8, center=4
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 8, 4, 8, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 8, 4, 8, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
        ),
        # Demo 1: same structure, frame=3, center=1
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 1, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 3, 1, 3, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
        ),
    )


def test_composite_assembly_frame_center():
    """Composite assembly groups singleton center with adjacent frame CCs."""
    from aria.observe import (
        _extract_objects_with_bg,
        _assemble_composites,
    )

    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 8, 4, 8, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ])
    objects = _extract_objects_with_bg(grid)
    composites, isolated = _assemble_composites(objects, grid)

    # One composite: 4-singleton enclosed by 8-frame
    assert len(composites) == 1
    assert composites[0].center["color"] == 4
    assert composites[0].center_row == 2
    assert composites[0].center_col == 2
    assert len(composites[0].frames) == 1
    assert composites[0].frames[0]["color"] == 8

    # One isolated: the anchor 4 at (5,3)
    assert len(isolated) == 1
    assert isolated[0]["color"] == 4
    assert isolated[0]["row"] == 5


def test_role_identification():
    """Role identification assigns center/frame/anchor correctly."""
    from aria.observe import (
        _extract_objects_with_bg,
        _identify_demo_roles,
    )

    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 8, 4, 8, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ])
    objects = _extract_objects_with_bg(grid)
    roles = _identify_demo_roles(objects, grid)

    assert roles is not None
    assert roles.bg_color == 0
    assert roles.center_color == 4
    assert roles.frame_color == 8
    assert roles.anchor is not None
    assert roles.anchor["row"] == 5
    assert roles.anchor["col"] == 3
    assert len(roles.composites) == 1


def test_composite_structural_signature_color_invariant():
    """Structural signatures match across demos despite color rotation."""
    from aria.observe import (
        _extract_objects_with_bg,
        _identify_demo_roles,
        _composite_structural_signature,
    )

    demos = _color_rotated_frame_center_task()
    sigs_per_demo = []
    for demo in demos:
        objs = _extract_objects_with_bg(demo.input)
        roles = _identify_demo_roles(objs, demo.input)
        assert roles is not None
        sigs = [_composite_structural_signature(c) for c in roles.composites]
        sigs_per_demo.append(sigs)

    # Signatures should be identical across the two demos
    assert sigs_per_demo[0] == sigs_per_demo[1]


def test_composite_correspondence_color_rotated():
    """Composite correspondence detects anchor alignment despite color rotation."""
    from aria.observe import _composite_correspondence_analysis

    demos = _color_rotated_frame_center_task()
    corr = _composite_correspondence_analysis(demos)

    assert corr is not None
    assert len(corr.rules) >= 1
    rule = corr.rules[0]
    assert rule.details.get("composite_correspondence") is True
    diag = rule.details.get("diagnostics", {})
    assert diag.get("alignment_verified") is True


def test_composite_correspondence_end_to_end():
    """observe_and_synthesize finds composite rules for color-rotated task."""
    demos = _color_rotated_frame_center_task()
    result = observe_and_synthesize(demos)
    # May not solve (no executable program yet) but should have composite rules
    composite_rules = [
        r for r in result.rules if r.details.get("composite_correspondence")
    ]
    assert len(composite_rules) >= 1
    assert composite_rules[0].details.get("diagnostics", {}).get("alignment_verified")


def test_composite_no_assembly_without_adjacency():
    """No composites formed when singletons have no adjacent frame CCs."""
    from aria.observe import _extract_objects_with_bg, _assemble_composites

    # Two isolated singletons of different colors, no adjacency
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 3],
    ])
    objects = _extract_objects_with_bg(grid)
    composites, isolated = _assemble_composites(objects, grid)
    assert len(composites) == 0
    assert len(isolated) == 2


def test_composite_rejects_same_color_adjacency():
    """Composites require different-color adjacency, not same-color."""
    from aria.observe import _extract_objects_with_bg, _assemble_composites

    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
    ])
    objects = _extract_objects_with_bg(grid)
    composites, isolated = _assemble_composites(objects, grid)
    # All CCs are the same color, so no composites from cross-color adjacency
    assert len(composites) == 0


def test_strict_correspondence_still_works():
    """Adding composite path doesn't break strict single-color correspondence."""
    demos = _uniform_move_task()
    result = observe_and_synthesize(demos)
    assert result.solved


def test_581f7754_enters_composite_hypothesis():
    """581f7754 produces a composite correspondence rule with verified alignment."""
    import json

    try:
        with open(
            "/Users/ericc59/dev/arcagi/arc-agi-benchmarking/data/"
            "public-v2/evaluation/581f7754.json"
        ) as f:
            d = json.load(f)
    except FileNotFoundError:
        import pytest
        pytest.skip("581f7754 data file not available")

    demos = tuple(
        DemoPair(
            input=grid_from_list(t["input"]),
            output=grid_from_list(t["output"]),
        )
        for t in d["train"]
    )

    from aria.observe import _composite_correspondence_analysis
    corr = _composite_correspondence_analysis(demos)
    assert corr is not None, "Should produce composite correspondence"
    assert len(corr.rules) >= 1

    rule = corr.rules[0]
    det = rule.details
    diag = det.get("diagnostics", {})

    # Verify role identification across all 3 demos
    role_map = diag.get("role_map", [])
    assert len(role_map) == 3
    # Colors rotate but roles are consistent
    centers = {rm["center_color"] for rm in role_map}
    frames = {rm["frame_color"] for rm in role_map}
    assert len(centers) == 3  # different center colors
    assert len(frames) == 3   # different frame colors

    # Alignment should be verified
    assert diag.get("alignment_verified") is True

    # Per-demo axes should be determined
    axes = det.get("per_demo_axis", diag.get("per_demo_axis"))
    assert axes is not None
    assert all(a in ("row", "col") for a in axes)
