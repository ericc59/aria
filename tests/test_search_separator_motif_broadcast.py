from __future__ import annotations

from aria.datasets import get_dataset, load_arc_task
from aria.search.search import search_programs


def test_search_programs_solves_1ae2feb7_with_separator_motif_broadcast():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "1ae2feb7")
    demos = [(pair.input, pair.output) for pair in task.train]

    program = search_programs(demos, time_budget=5.0)

    assert program is not None
    assert "separator_motif_broadcast" in program.description
    assert all((program.execute(inp) == out).all() for inp, out in demos)
    assert all((program.execute(pair.input) == pair.output).all() for pair in task.test)
