"""Tests for role-based color inference and per-object conditional dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from aria.core.scene_executor import (
    SceneState,
    _evaluate_guard,
    execute_scene_program,
    make_scene_program,
    make_step,
    resolve_color_role,
)
from aria.core.grid_perception import perceive_grid
from aria.core.scene_solve import _try_role_based_per_object, verify_scene_program
from aria.scene_ir import EntityKind, SceneEntity, StepOp
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Color role resolution
# ---------------------------------------------------------------------------


class TestColorRoleResolution:
    def test_singleton_color(self):
        grid = grid_from_list([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 3],  # 3 is singleton (1 pixel)
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        assert resolve_color_role("singleton_color", state) == 3

    def test_rarest_non_bg(self):
        grid = grid_from_list([
            [0, 1, 1],
            [1, 1, 2],
            [0, 0, 0],
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        assert resolve_color_role("rarest_non_bg", state) == 2

    def test_most_frequent_non_bg(self):
        grid = grid_from_list([
            [0, 1, 1],
            [1, 1, 2],
            [0, 0, 0],
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        assert resolve_color_role("most_frequent_non_bg", state) == 1

    def test_dominant_object_color(self):
        from aria.scene_ir import SceneEntity, EntityKind
        grid = grid_from_list([
            [0, 0, 0],
            [0, 5, 5],
            [0, 5, 3],
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        entity = SceneEntity(
            id="obj", kind=EntityKind.OBJECT,
            bbox=(1, 1, 2, 2),
        )
        assert resolve_color_role("dominant_object_color", state, entity) == 5

    def test_minority_object_color(self):
        from aria.scene_ir import SceneEntity, EntityKind
        grid = grid_from_list([
            [0, 0, 0],
            [0, 5, 5],
            [0, 5, 3],
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        entity = SceneEntity(
            id="obj", kind=EntityKind.OBJECT,
            bbox=(1, 1, 2, 2),
        )
        assert resolve_color_role("minority_object_color", state, entity) == 3


# ---------------------------------------------------------------------------
# Role-based FOR_EACH rules
# ---------------------------------------------------------------------------


class TestRoleBasedRules:
    def test_fill_bbox_holes_role(self):
        """Fill bg holes within objects using a color role."""
        # Object with color 2 forms a frame, bg=0 inside, singleton color 5
        inp = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 5, 0, 0],
        ])
        out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 5, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 5, 0, 0],
        ])
        demos = (DemoPair(input=inp, output=out),)

        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FOR_EACH_ENTITY, kind="object",
                      rule="fill_bbox_holes_role", color_role="singleton_color",
                      connectivity=4),
            make_step(StepOp.RENDER_SCENE),
        )
        result = execute_scene_program(prog, inp)
        assert np.array_equal(result, out)

    def test_recolor_dominant_to_minority(self):
        """Swap dominant and minority colors within each non-singleton object bbox."""
        # Single non-singleton 4-conn object (color 1, size=5) with
        # one pixel of color 2 nearby that's captured in its bbox.
        # Use a frame-like object with 2 colors in its bbox.
        inp = grid_from_list([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        # After swap within the object's bbox: dominant 1→2, minority 2→1
        out = grid_from_list([
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 2, 2, 1, 2, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        demos = (DemoPair(input=inp, output=out),)

        # The color-1 CC is a 4x4 block with the color-2 pixel inside its bbox
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FOR_EACH_ENTITY, kind="object",
                      rule="recolor_dominant_to_minority", connectivity=4),
            make_step(StepOp.RENDER_SCENE),
        )
        result = execute_scene_program(prog, inp)
        # The 4-conn object color=1 has bbox (1,1)-(4,4), and within that bbox
        # dominant=1 (15 pixels), minority=2 (1 pixel) → swap
        assert np.array_equal(result, out)

    def test_fill_enclosed_regions_role(self):
        """Fill enclosed regions using a color role."""
        # bg=0 (most frequent), 1 forms a frame, 3 is rarest non-bg
        inp = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 3, 0, 0],
        ])
        out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 3, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 3, 0, 0],
        ])
        demos = (DemoPair(input=inp, output=out),)

        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, color_role="rarest_non_bg"),
            make_step(StepOp.RENDER_SCENE),
        )
        result = execute_scene_program(prog, inp)
        assert np.array_equal(result, out)


# ---------------------------------------------------------------------------
# Cross-demo role consistency
# ---------------------------------------------------------------------------


class TestCrossDemoRoleConsistency:
    def test_role_varies_color_across_demos(self):
        """Same role resolves to different literal colors per demo."""
        # Demo 0: rarest non-bg is 5 (bg=0, frame=1)
        d0_in = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 5, 0, 0],
        ])
        d0_out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 5, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 5, 0, 0],
        ])
        # Demo 1: rarest non-bg is 3
        d1_in = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 3, 0, 0],
        ])
        d1_out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 3, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 3, 0, 0],
        ])
        demos = (
            DemoPair(input=d0_in, output=d0_out),
            DemoPair(input=d1_in, output=d1_out),
        )

        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FILL_ENCLOSED_REGIONS, color_role="rarest_non_bg"),
            make_step(StepOp.RENDER_SCENE),
        )
        assert verify_scene_program(prog, demos)


# ---------------------------------------------------------------------------
# Family 16 integration
# ---------------------------------------------------------------------------


class TestFamily16:
    def test_finds_role_based_solution(self):
        """_try_role_based_per_object finds a solution for cross-demo role tasks."""
        d0_in = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 5, 0, 0],
        ])
        d0_out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 5, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 5, 0, 0],
        ])
        d1_in = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 3, 0, 0],
        ])
        d1_out = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 3, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 3, 0, 0],
        ])
        demos = (
            DemoPair(input=d0_in, output=d0_out),
            DemoPair(input=d1_in, output=d1_out),
        )
        progs = _try_role_based_per_object(demos)
        assert len(progs) > 0
        assert verify_scene_program(progs[0], demos)

    def test_returns_empty_for_diff_dims(self):
        demos = (
            DemoPair(
                input=grid_from_list([[1, 2]]),
                output=grid_from_list([[1]]),
            ),
        )
        assert _try_role_based_per_object(demos) == []

    def test_v1_train_56ff96f3(self):
        """Real ARC task solved by fill_bbox_holes_role + dominant_object_color."""
        from aria.datasets import get_dataset, load_arc_task
        try:
            ds = get_dataset("v1-train")
            task = load_arc_task(ds, "56ff96f3")
        except (ValueError, FileNotFoundError):
            pytest.skip("v1-train dataset not available")
        demos = task.train
        progs = _try_role_based_per_object(demos)
        assert len(progs) > 0
        assert verify_scene_program(progs[0], demos)


# ---------------------------------------------------------------------------
# Guard predicates
# ---------------------------------------------------------------------------


class TestGuardPredicates:
    def _make_state(self, grid):
        from aria.core.scene_executor import _STEP_HANDLERS
        from aria.scene_ir import SceneStep
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        handler = _STEP_HANDLERS[StepOp.PARSE_SCENE]
        handler(state, SceneStep(op=StepOp.PARSE_SCENE))
        return state

    def test_guard_not_frame_like(self):
        """Guard skips entities whose bbox overlaps a BOUNDARY entity."""
        # Manually create entities + a BOUNDARY to test the guard directly
        from aria.scene_ir import SceneEntity as SE
        grid = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0],
            [0, 1, 0, 1, 0, 2, 2, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        # Add a BOUNDARY that covers the frame area only
        boundary = SE(id="frame_0", kind=EntityKind.BOUNDARY,
                       bbox=(0, 0, 4, 4), colors=(1,))
        state.add_entity(boundary)
        # Object inside frame bbox
        obj_inside = SE(id="obj_in", kind=EntityKind.OBJECT,
                        bbox=(1, 1, 3, 3), colors=(1,),
                        attrs={"connectivity": 4, "is_singleton": False, "size": 8})
        state.add_entity(obj_inside)
        # Object outside frame bbox
        obj_outside = SE(id="obj_out", kind=EntityKind.OBJECT,
                         bbox=(1, 5, 2, 6), colors=(2,),
                         attrs={"connectivity": 4, "is_singleton": False, "size": 4})
        state.add_entity(obj_outside)

        assert _evaluate_guard("is_frame_like", obj_inside, state) is True
        assert _evaluate_guard("not_frame_like", obj_inside, state) is False
        assert _evaluate_guard("is_frame_like", obj_outside, state) is False
        assert _evaluate_guard("not_frame_like", obj_outside, state) is True

    def test_guard_has_enclosed_bg(self):
        """Guard selects objects with enclosed bg (holes)."""
        # A ring (has enclosed bg) and a solid block (no enclosed bg)
        grid = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0],
            [0, 1, 0, 1, 0, 2, 2, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        state = self._make_state(grid)
        obj_entities = state.entities_of_kind(EntityKind.OBJECT)
        with_holes = [e for e in obj_entities if _evaluate_guard("has_enclosed_bg", e, state)]
        without_holes = [e for e in obj_entities if _evaluate_guard("no_enclosed_bg", e, state)]
        # The ring has enclosed bg; the solid block doesn't
        assert len(with_holes) > 0
        assert len(without_holes) > 0


# ---------------------------------------------------------------------------
# Object groups
# ---------------------------------------------------------------------------


class TestObjectGroups:
    def _make_state(self, grid):
        p = perceive_grid(grid)
        state = SceneState(input_grid=grid, perception=p)
        from aria.core.scene_executor import _STEP_HANDLERS
        from aria.scene_ir import SceneStep
        handler = _STEP_HANDLERS[StepOp.PARSE_SCENE]
        handler(state, SceneStep(op=StepOp.PARSE_SCENE))
        return state

    def test_object_group_from_frame(self):
        """OBJECT_GROUP entity created from a framed region."""
        grid = grid_from_list([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        state = self._make_state(grid)
        groups = state.entities_of_kind(EntityKind.OBJECT_GROUP)
        frame_groups = [g for g in groups if g.attrs.get("group_source") == "frame"]
        assert len(frame_groups) > 0
        fg = frame_groups[0]
        assert fg.attrs["has_frame"] is True
        assert fg.attrs["n_colors"] >= 1

    def test_object_group_proximity(self):
        """OBJECT_GROUP from adjacent objects via proximity clustering."""
        # Two adjacent motifs, separated by space — no full-grid frame.
        # Objects 1+2 are adjacent (should group), objects 3+4 are adjacent (should group).
        grid = grid_from_list([
            [1, 1, 2, 0, 0, 0, 3, 4],
            [1, 1, 2, 0, 0, 0, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        state = self._make_state(grid)
        groups = state.entities_of_kind(EntityKind.OBJECT_GROUP)
        prox_groups = [g for g in groups if g.attrs.get("group_source") == "proximity"]
        assert len(prox_groups) > 0
        pg = prox_groups[0]
        assert pg.attrs["has_frame"] is False
        assert len(pg.attrs["member_object_ids"]) >= 2


# ---------------------------------------------------------------------------
# Guarded end-to-end
# ---------------------------------------------------------------------------


class TestGuardedEndToEnd:
    def test_guarded_fill_only_objects_with_holes(self):
        """Guard has_enclosed_bg + fill_enclosed_role: only fills objects with holes."""
        # Ring (color 1) has enclosed bg → should get filled
        # Solid block (color 2) has no enclosed bg → should be skipped
        inp = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0],
            [0, 1, 0, 1, 0, 2, 2, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0],
        ])
        out = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 2, 2, 0],
            [0, 1, 5, 1, 0, 2, 2, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0],
        ])
        prog = make_scene_program(
            make_step(StepOp.PARSE_SCENE),
            make_step(StepOp.FOR_EACH_ENTITY, kind="object",
                      rule="fill_enclosed_role", color_role="singleton_color",
                      guard="has_enclosed_bg", connectivity=4),
            make_step(StepOp.RENDER_SCENE),
        )
        result = execute_scene_program(prog, inp)
        assert np.array_equal(result, out)

    def test_family16_guarded_solve(self):
        """Family 16 finds a guarded solution with has_enclosed_bg guard."""
        # Two demos: only the ring object has holes → only it gets filled
        d0_in = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 3, 3, 0],
            [0, 1, 0, 1, 0, 3, 3, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0],
        ])
        d0_out = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 3, 3, 0],
            [0, 1, 5, 1, 0, 3, 3, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0],
        ])
        d1_in = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 4, 4, 0],
            [0, 2, 0, 2, 0, 4, 4, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 0],
        ])
        d1_out = grid_from_list([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 4, 4, 0],
            [0, 2, 6, 2, 0, 4, 4, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 0],
        ])
        demos = (
            DemoPair(input=d0_in, output=d0_out),
            DemoPair(input=d1_in, output=d1_out),
        )
        progs = _try_role_based_per_object(demos)
        assert len(progs) > 0
        assert verify_scene_program(progs[0], demos)
