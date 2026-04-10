from __future__ import annotations

from aria.datasets import get_dataset, load_arc_task
from aria.search.search import search_programs


def test_search_programs_solves_16b78196_with_barrier_port_transfer():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "16b78196")
    demos = [(pair.input, pair.output) for pair in task.train]

    program = search_programs(demos, time_budget=5.0)

    assert program is not None
    assert "barrier_port_transfer" in program.description
    assert all((program.execute(inp) == out).all() for inp, out in demos)
    assert all((program.execute(pair.input) == pair.output).all() for pair in task.test)
