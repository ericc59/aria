from __future__ import annotations

from aria.datasets import get_dataset, load_arc_task
from aria.search.search import search_programs


def test_search_programs_solves_3e6067c3_with_legend_chain_connect():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "3e6067c3")
    demos = [(pair.input, pair.output) for pair in task.train]

    program = search_programs(demos, time_budget=5.0)

    assert program is not None
    assert "legend_chain_connect" in program.description
    assert all((program.execute(inp) == out).all() for inp, out in demos)
    assert all((program.execute(pair.input) == pair.output).all() for pair in task.test)
