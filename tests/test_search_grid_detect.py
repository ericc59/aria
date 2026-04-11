"""Tests for grid detection and grid-fill-between."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive
from aria.search.grid_detect import (
    detect_separator_grid,
    cell_content,
    cell_has_content,
    cell_content_color,
)


def test_detect_separator_grid_3x3():
    """Detect a 3x3 grid with row/col separators."""
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, :] = 5  # row separator
    grid[4, :] = 5  # row separator
    grid[:, 2] = 5  # col separator
    grid[:, 4] = 5  # col separator

    facts = perceive(grid)
    g = detect_separator_grid(facts)

    assert g is not None
    assert g.n_rows == 3
    assert g.n_cols == 3
    assert g.sep_color == 5
    assert len(g.cells) == 9


def test_cell_content_detection():
    """cell_has_content and cell_content_color should work."""
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, :] = 5
    grid[4, :] = 5
    grid[:, 2] = 5
    grid[:, 4] = 5
    grid[0, 0] = 3  # content in cell (0,0)

    facts = perceive(grid)
    g = detect_separator_grid(facts)
    assert g is not None

    cell_00 = g.cell_at(0, 0)
    cell_01 = g.cell_at(0, 1)
    assert cell_00 is not None
    assert cell_has_content(grid, cell_00, bg=0)
    assert not cell_has_content(grid, cell_01, bg=0)
    assert cell_content_color(grid, cell_00, bg=0) == 3
    assert cell_content_color(grid, cell_01, bg=0) is None


def test_grid_fill_between_execution():
    """grid_fill_between should fill between same-color blocks."""
    from aria.search.sketch import SearchProgram, SearchStep

    # 3x3 grid with separators
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, :] = 5
    grid[4, :] = 5
    grid[:, 2] = 5
    grid[:, 4] = 5
    # Color 3 in cell (0,0) and (0,2) — should fill (0,1)
    grid[0:2, 0:2] = 3
    grid[0:2, 5:7] = 3

    prog = SearchProgram(
        steps=[SearchStep('grid_fill_between', {})],
        provenance='test',
    )
    result = prog.execute(grid)

    # Cell (0,1) should now be filled with 3
    assert result[0, 3] == 3
    assert result[1, 3] == 3
    # Cell (1,0) should still be empty
    assert result[3, 0] == 0


def test_06df4c85_solves():
    """Task 06df4c85 must solve via grid_fill_between."""
    from aria.datasets import get_dataset, load_arc_task
    from aria.search.derive import derive_programs

    ds = get_dataset('v1-train')
    task = load_arc_task(ds, '06df4c85')
    demos = [(p.input, p.output) for p in task.train]

    progs = derive_programs(demos)
    assert progs, "derive_programs returned no candidates"

    solved = False
    for p in progs:
        if all(np.array_equal(p.execute(pair.input), pair.output) for pair in task.train):
            solved = True
            if task.test:
                test_ok = all(np.array_equal(p.execute(pair.input), pair.output)
                              for pair in task.test)
                assert test_ok, f"train-verified but test-failed for {p.provenance}"
            break

    assert solved, "no derive program verified for 06df4c85"
