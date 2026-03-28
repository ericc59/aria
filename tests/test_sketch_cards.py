"""Tests for per-task ARC-2 sketch card generation."""

from __future__ import annotations

import json

from aria.types import DemoPair, grid_from_list
from scripts.arc2_sketch_cards import (
    SketchCard,
    build_card,
    card_to_dict,
    card_to_markdown,
    _color_roles_per_demo,
    _pixel_diff_analysis,
    _object_structure,
    _spatial_regularity,
    _dominant_color,
)


def _identity_task():
    """Minimal same-dims task: output = input."""
    from aria.types import Task
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 2]]),
            output=grid_from_list([[1, 0], [0, 2]]),
        ),
    )
    return Task(train=demos, test=demos)


def _small_change_task():
    """Same-dims task with small pixel diff."""
    from aria.types import Task
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 1, 3], [0, 0, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 0], [0, 2, 3], [0, 0, 0]]),
        ),
    )
    return Task(train=demos, test=demos)


def _dims_change_task():
    """Dims-change task: output is smaller."""
    from aria.types import Task
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0]]),
            output=grid_from_list([[1, 2], [3, 4]]),
        ),
    )
    return Task(train=demos, test=demos)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_card_has_all_fields():
    """SketchCard has all required fields."""
    task = _identity_task()
    card = build_card("test_identity", task)
    assert card.task_id == "test_identity"
    assert isinstance(card.same_dims, bool)
    assert isinstance(card.decomposition, list)
    assert len(card.decomposition) >= 1
    assert isinstance(card.invariants, list)
    assert len(card.invariants) >= 1
    assert isinstance(card.construction, str)
    assert isinstance(card.system_failure, str)
    assert isinstance(card.sketch, str)
    assert isinstance(card.task_signatures, list)
    assert isinstance(card.evidence, dict)


def test_card_serializes_to_json():
    """Card converts to dict and round-trips through JSON."""
    task = _small_change_task()
    card = build_card("test_small", task)
    d = card_to_dict(card)
    assert isinstance(d, dict)
    assert d["task_id"] == "test_small"
    # Round-trip through JSON
    serialized = json.dumps(d)
    restored = json.loads(serialized)
    assert restored["task_id"] == "test_small"
    assert restored["same_dims"] is True
    assert len(restored["decomposition"]) >= 1


def test_card_renders_markdown():
    """Card produces non-empty markdown with expected sections."""
    task = _small_change_task()
    card = build_card("test_small", task)
    md = card_to_markdown(card)
    assert "### `test_small`" in md
    assert "**Decomposition**" in md
    assert "**Invariants**" in md
    assert "**Construction**" in md
    assert "**System failure**" in md
    assert "**Sketch**" in md
    assert "evidence" in md


def test_same_dims_card():
    """Same-dims task card has pixel diff analysis."""
    task = _small_change_task()
    card = build_card("test_same", task)
    assert card.same_dims is True
    assert "n/a" not in card.pixel_diff_summary


def test_dims_change_card():
    """Dims-change task card describes dimension relationship."""
    task = _dims_change_task()
    card = build_card("test_dims", task)
    assert card.same_dims is False
    assert "n/a" in card.pixel_diff_summary or "dims change" in card.pixel_diff_summary


def test_color_role_rotation_detected():
    """Color role rotation is detected when bg differs across demos."""
    from aria.types import Task
    demos = (
        DemoPair(
            input=grid_from_list([[3, 3, 3], [3, 1, 3], [3, 3, 3]]),
            output=grid_from_list([[3, 3, 3], [3, 1, 3], [3, 3, 3]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 5, 5], [5, 2, 5], [5, 5, 5]]),
            output=grid_from_list([[5, 5, 5], [5, 2, 5], [5, 5, 5]]),
        ),
    )
    task = Task(train=demos, test=demos)
    card = build_card("test_rotate", task)
    assert "roles_rotate=True" in card.color_roles_summary
    assert any("role" in inv.lower() or "rotate" in inv.lower() for inv in card.invariants)


# ---------------------------------------------------------------------------
# Analysis primitive tests
# ---------------------------------------------------------------------------


def test_dominant_color():
    grid = grid_from_list([[0, 0, 1], [0, 0, 0]])
    assert _dominant_color(grid) == 0


def test_color_roles_per_demo():
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 1], [0, 0, 0]]),
            output=grid_from_list([[0, 0, 1], [0, 3, 0]]),
        ),
    )
    roles = _color_roles_per_demo(demos)
    assert len(roles) == 1
    assert roles[0]["bg_in"] == 0
    assert roles[0]["fg_in"] == [1]
    assert 3 in roles[0]["colors_added"]


def test_pixel_diff_analysis():
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 1], [0, 0]]),
        ),
    )
    diff = _pixel_diff_analysis(demos)
    assert diff is not None
    assert diff["max_changed"] == 1
    assert diff["per_demo"][0]["changed_pixels"] == 1


def test_pixel_diff_returns_none_for_dims_change():
    demos = (
        DemoPair(
            input=grid_from_list([[0, 0, 0]]),
            output=grid_from_list([[0], [0]]),
        ),
    )
    assert _pixel_diff_analysis(demos) is None


def test_object_structure():
    demos = (
        DemoPair(
            input=grid_from_list([[0, 1, 0], [0, 0, 0], [2, 0, 3]]),
            output=grid_from_list([[0, 1, 0], [0, 0, 0], [2, 0, 3]]),
        ),
    )
    struct = _object_structure(demos)
    assert struct["per_demo"][0]["n_input_objects"] == 3
    assert struct["any_singletons"] is True


def test_spatial_regularity_frame():
    grid = grid_from_list([
        [5, 5, 5, 5],
        [5, 1, 2, 5],
        [5, 3, 4, 5],
        [5, 5, 5, 5],
    ])
    reg = _spatial_regularity(grid, bg=5)
    assert reg["has_frame"] is True
    assert reg["frame_color"] == 5


def test_each_card_is_task_specific():
    """Two different tasks should produce different cards."""
    task1 = _identity_task()
    task2 = _small_change_task()
    card1 = build_card("task_a", task1)
    card2 = build_card("task_b", task2)
    # They should have different decompositions or constructions
    assert card1.construction != card2.construction or card1.decomposition != card2.decomposition
