from __future__ import annotations

from aria.datasets import get_dataset, load_arc_task
from aria.search.search import search_programs


def test_search_programs_solves_16de56c4_with_line_arith_broadcast():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "16de56c4")
    demos = [(pair.input, pair.output) for pair in task.train]

    program = search_programs(demos, time_budget=5.0)

    assert program is not None
    assert "line_arith_broadcast" in program.description
    assert all((program.execute(inp) == out).all() for inp, out in demos)
    assert all((program.execute(pair.input) == pair.output).all() for pair in task.test)
