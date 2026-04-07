"""Tests for lane-local parameter priors."""

from __future__ import annotations

from aria.core.param_priors import (
    rank_relocation_params, rank_periodic_params, rank_replication_params,
)
from aria.core.arc import solve_arc_task
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_relocation_params_complete():
    params = rank_relocation_params(_simple())
    assert len(params) == 49  # 7 match * 7 align
    # All unique
    assert len(set(params)) == 49


def test_relocation_params_shape_nearest_first():
    """shape_nearest should be in the first few entries."""
    params = rank_relocation_params(_simple())
    first_rules = [mr for mr, al in params[:7]]
    assert 0 in first_rules  # MATCH_SHAPE_NEAREST = 0


def test_periodic_params_complete():
    params = rank_periodic_params(_simple())
    assert len(params) == 24  # 2 axes * 4 periods * 3 modes


def test_replication_params_complete():
    params = rank_replication_params(_simple())
    assert len(params) == 8  # 2 * 2 * 2


def test_replication_diff_color_erase_first():
    """The default (diff_color, erase, anchor_offset) should be first."""
    params = rank_replication_params(_simple())
    assert params[0] == (0, 0, 0)  # KEY_ADJACENT_DIFF_COLOR, SOURCE_ERASE, PLACE_ANCHOR_OFFSET


def test_no_task_id():
    import inspect
    for fn in [rank_relocation_params, rank_periodic_params, rank_replication_params]:
        src = inspect.getsource(fn)
        assert "1b59e163" not in src


def test_no_regressions():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved
