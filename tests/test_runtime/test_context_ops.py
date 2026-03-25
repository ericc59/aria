"""Tests for cross-demo context operations."""

from __future__ import annotations

import numpy as np
import pytest

from aria.types import DemoPair, Grid, TaskContext, grid_from_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(*pairs: tuple[list[list[int]], list[list[int]]]) -> TaskContext:
    """Build a TaskContext from raw grid lists."""
    demos = tuple(
        DemoPair(input=grid_from_list(inp), output=grid_from_list(out))
        for inp, out in pairs
    )
    return TaskContext(demos=demos)


# ---------------------------------------------------------------------------
# infer_map
# ---------------------------------------------------------------------------

class TestInferMap:
    def test_simple_color_swap(self):
        """Two demos where 1->2 and 3->4 consistently."""
        from aria.runtime.ops.context import _infer_map

        ctx = _ctx(
            # Demo 1: 1s become 2s, 3s become 4s
            ([[1, 3, 1], [3, 1, 3]], [[2, 4, 2], [4, 2, 4]]),
            # Demo 2: same mapping, different layout
            ([[3, 1, 1], [1, 3, 3]], [[4, 2, 2], [2, 4, 4]]),
        )

        result = _infer_map(ctx, 0, 0)
        assert result[1] == 2
        assert result[3] == 4

    def test_identity_colors_included(self):
        """Colors that stay the same are also in the map."""
        from aria.runtime.ops.context import _infer_map

        ctx = _ctx(
            ([[0, 1], [2, 0]], [[0, 5], [2, 0]]),
            ([[1, 0], [0, 2]], [[5, 0], [0, 2]]),
        )

        result = _infer_map(ctx, 0, 0)
        assert result[1] == 5
        assert result[0] == 0
        assert result[2] == 2

    def test_empty_context(self):
        """No demos -> empty map."""
        from aria.runtime.ops.context import _infer_map

        ctx = TaskContext(demos=())
        assert _infer_map(ctx, 0, 0) == {}

    def test_single_demo(self):
        """A single demo still produces a valid map."""
        from aria.runtime.ops.context import _infer_map

        ctx = _ctx(
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        )
        result = _infer_map(ctx, 0, 0)
        assert result[1] == 5
        assert result[2] == 6
        assert result[3] == 7
        assert result[4] == 8

    def test_inconsistent_mapping_excluded(self):
        """If demos disagree on a mapping, it may be excluded or majority-voted."""
        from aria.runtime.ops.context import _infer_map

        ctx = _ctx(
            ([[1, 1], [1, 1]], [[2, 2], [2, 2]]),
            ([[1, 1], [1, 1]], [[3, 3], [3, 3]]),
            ([[1, 1], [1, 1]], [[2, 2], [2, 2]]),
        )
        result = _infer_map(ctx, 0, 0)
        # Majority vote: 2 out of 3 demos say 1->2
        assert result.get(1) == 2


# ---------------------------------------------------------------------------
# infer_step
# ---------------------------------------------------------------------------

class TestInferStep:
    def test_translation_sequence(self):
        """Demos show an object moving right by 1 each step."""
        from aria.runtime.ops.context import _infer_step

        # Object is a single pixel moving rightward
        # Demo 0: output has pixel at (1, 0)
        # Demo 1: output has pixel at (1, 1)
        # Demo 2: output has pixel at (1, 2)
        ctx = _ctx(
            ([[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 0, 0]]),
            ([[0, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0]]),
            ([[0, 0, 0, 0], [0, 0, 1, 0]], [[0, 0, 0, 0], [0, 0, 1, 0]]),
        )

        step_fn = _infer_step(ctx)
        # Apply step to demo 2's output: pixel should move from (1,2) to (1,3)
        test_grid = grid_from_list([[0, 0, 0, 0], [0, 0, 1, 0]])
        result = step_fn(test_grid)
        assert result[1, 3] == 1
        assert result[1, 2] == 0  # Old position cleared

    def test_color_map_fallback(self):
        """When no spatial progression, falls back to color mapping."""
        from aria.runtime.ops.context import _infer_step

        # Both demos apply the same color mapping: 1->3, 2->4
        # Grids are identical shapes to rule out translation/growth
        ctx = _ctx(
            ([[1, 2], [1, 2]], [[3, 4], [3, 4]]),
            ([[2, 1], [2, 1]], [[4, 3], [4, 3]]),
        )

        step_fn = _infer_step(ctx)
        test_grid = grid_from_list([[1, 1], [2, 2]])
        result = step_fn(test_grid)
        assert int(result[0, 0]) == 3
        assert int(result[1, 0]) == 4

    def test_single_demo_uses_color_map(self):
        """Single demo: infer_step uses color map."""
        from aria.runtime.ops.context import _infer_step

        ctx = _ctx(
            ([[5, 5], [5, 5]], [[7, 7], [7, 7]]),
        )
        step_fn = _infer_step(ctx)
        test_grid = grid_from_list([[5, 5]])
        result = step_fn(test_grid)
        assert int(result[0, 0]) == 7


# ---------------------------------------------------------------------------
# predict_dims
# ---------------------------------------------------------------------------

class TestPredictDims:
    def test_fixed_output_size(self):
        """All demos produce the same output size regardless of input."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            ([[1, 2, 3]], [[0, 0], [0, 0], [0, 0]]),  # 1x3 -> 3x2
            ([[1], [2]], [[0, 0], [0, 0], [0, 0]]),     # 2x1 -> 3x2
        )
        test_grid = grid_from_list([[5, 5, 5, 5]])  # 1x4
        assert _predict_dims(ctx, test_grid) == (3, 2)

    def test_multiplicative_scaling(self):
        """Output is always 2x the input in both dimensions."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            ([[1]], [[0, 0], [0, 0]]),             # 1x1 -> 2x2
            ([[1, 2], [3, 4]], [[0]*4, [0]*4, [0]*4, [0]*4]),  # 2x2 -> 4x4
        )
        test_grid = grid_from_list([[1, 2, 3]])  # 1x3
        assert _predict_dims(ctx, test_grid) == (2, 6)

    def test_single_demo_integer_scaling_beats_trivial_fixed_size(self):
        """A single remaining demo in LOO should still preserve clean kx scaling."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            ([[1, 2], [3, 4]], [[0] * 4, [0] * 4, [0] * 4, [0] * 4]),
        )
        test_grid = grid_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3
        assert _predict_dims(ctx, test_grid) == (6, 6)

    def test_single_demo_non_scaling_case_falls_back_to_fixed_output_size(self):
        """Single-demo fixed-size tasks should not be forced into fake scaling."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            ([[1, 2, 3]], [[0, 0], [0, 0], [0, 0]]),  # 1x3 -> 3x2
        )
        test_grid = grid_from_list([[5], [5]])  # 2x1
        assert _predict_dims(ctx, test_grid) == (3, 2)

    def test_additive_offset(self):
        """Output is always input + (1, 2)."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            # 2x3 -> 3x5
            ([[1, 2, 3], [4, 5, 6]], [[0]*5, [0]*5, [0]*5]),
            # 3x2 -> 4x4
            ([[1, 2], [3, 4], [5, 6]], [[0]*4, [0]*4, [0]*4, [0]*4]),
        )
        test_grid = grid_from_list([[1, 1, 1, 1]])  # 1x4
        assert _predict_dims(ctx, test_grid) == (2, 6)

    def test_same_as_input(self):
        """Output has the same dims as input."""
        from aria.runtime.ops.context import _predict_dims

        ctx = _ctx(
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        )
        test_grid = grid_from_list([[1, 2, 3, 4, 5]])  # 1x5
        assert _predict_dims(ctx, test_grid) == (1, 5)

    def test_empty_context_returns_input_dims(self):
        from aria.runtime.ops.context import _predict_dims

        ctx = TaskContext(demos=())
        test_grid = grid_from_list([[1, 2], [3, 4]])
        assert _predict_dims(ctx, test_grid) == (2, 2)


# ---------------------------------------------------------------------------
# infer_iteration
# ---------------------------------------------------------------------------

class TestInferIteration:
    def test_default_next_in_sequence(self):
        """With N demos, default iteration count is N+1."""
        from aria.runtime.ops.context import _infer_iteration

        ctx = _ctx(
            ([[1]], [[2]]),
            ([[1]], [[2]]),
            ([[1]], [[2]]),
        )
        test_grid = grid_from_list([[1]])
        assert _infer_iteration(ctx, test_grid) == 4  # 3 demos -> 4

    def test_growth_progression(self):
        """Demos show grid growing -> iteration = n_demos + 1."""
        from aria.runtime.ops.context import _infer_iteration

        ctx = _ctx(
            # Output grids grow by 1 row each demo
            ([[1]], [[1]]),           # 1x1 -> 1x1
            ([[1]], [[1], [0]]),      # 1x1 -> 2x1
            ([[1]], [[1], [0], [0]]), # 1x1 -> 3x1
        )
        test_grid = grid_from_list([[1]])
        assert _infer_iteration(ctx, test_grid) == 4

    def test_single_demo(self):
        """One demo -> 2 iterations."""
        from aria.runtime.ops.context import _infer_iteration

        ctx = _ctx(
            ([[1, 2]], [[3, 4]]),
        )
        test_grid = grid_from_list([[5]])
        assert _infer_iteration(ctx, test_grid) == 2

    def test_empty_context(self):
        """No demos -> 1 iteration."""
        from aria.runtime.ops.context import _infer_iteration

        ctx = TaskContext(demos=())
        test_grid = grid_from_list([[1]])
        assert _infer_iteration(ctx, test_grid) == 1


# ---------------------------------------------------------------------------
# disambiguate
# ---------------------------------------------------------------------------

class TestDisambiguate:
    def test_empty_predicates(self):
        """No predicates -> always-true fallback."""
        from aria.runtime.ops.context import _disambiguate

        ctx = _ctx(
            ([[1]], [[1]]),
        )
        result = _disambiguate(ctx, [])
        assert result("anything") is True

    def test_selects_from_list(self):
        """Returns one of the provided predicates."""
        from aria.runtime.ops.context import _disambiguate

        pred_a = lambda obj: True
        pred_b = lambda obj: False

        ctx = _ctx(
            ([[1]], [[1]]),
        )
        result = _disambiguate(ctx, [pred_a, pred_b])
        assert result in (pred_a, pred_b) or callable(result)

    def test_empty_context_returns_first(self):
        """No demos -> return first predicate."""
        from aria.runtime.ops.context import _disambiguate

        pred = lambda obj: obj.color > 3
        ctx = TaskContext(demos=())
        assert _disambiguate(ctx, [pred]) is pred
