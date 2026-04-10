from __future__ import annotations

from aria.datasets import get_dataset, load_arc_task
from aria.search.geometry import (
    axis_ray,
    bbox,
    bridge_area,
    closest_boundary_anchor,
    diagonal_ray,
    orthogonal_bridge,
    orthogonal_path,
)
from aria.search.search import search_programs


def test_geometry_helpers_cover_basic_routing_and_anchor_cases():
    cells_a = {(2, 2), (2, 3)}
    cells_b = {(2, 7), (2, 8)}
    bridge = orthogonal_bridge(cells_a, cells_b)

    assert bbox(cells_a).left == 2
    assert bbox(cells_b).right == 8
    assert bridge == ("h", 2, 2, 4, 6)
    assert bridge_area(bridge) == 3
    assert closest_boundary_anchor(cells_a, "right", target_row=2.0, target_col=9.0) == 2
    assert axis_ray((4, 4), "down", (8, 8)) == [(5, 4), (6, 4), (7, 4)]
    assert diagonal_ray((2, 2), "down_right", (6, 6)) == [(3, 3), (4, 4), (5, 5)]
    assert orthogonal_path((1, 1), (3, 4), order="hv") == [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 4),
    ]


def test_search_programs_solves_3dc255db_with_cavity_transfer_geometry():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "3dc255db")
    demos = [(pair.input, pair.output) for pair in task.train]

    program = search_programs(demos, time_budget=5.0)

    assert program is not None
    assert "cavity_transfer" in program.description
    assert all((program.execute(inp) == out).all() for inp, out in demos)
    assert all((program.execute(pair.input) == pair.output).all() for pair in task.test)
