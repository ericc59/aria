"""Top-level ARC-AGI-2 solver.

Runs the guided expert first (fast, ~2s/task), then the search engine
on unsolved tasks with the remaining budget.
"""

from __future__ import annotations

import time
import numpy as np

from aria.types import Grid


def solve_task(
    demos: list[tuple[Grid, Grid]],
    time_budget: float = 30.0,
) -> dict:
    """Solve an ARC task given demo pairs.

    Returns dict with:
        'program': executable program (or None)
        'source': which engine found it ('guided', 'search', or None)
        'time': seconds taken
        'description': human-readable program description
    """
    t0 = time.time()

    # Phase 1: Guided expert (fast path)
    from aria.guided.dsl import synthesize_program, _verify
    try:
        prog = synthesize_program(demos)
        if prog and _verify(prog, demos):
            return {
                'program': prog,
                'source': 'guided',
                'time': time.time() - t0,
                'description': prog.description,
            }
    except Exception:
        pass

    elapsed = time.time() - t0
    remaining = time_budget - elapsed

    # Phase 2: Search engine
    if remaining > 1.0:
        from aria.search.search import search_programs
        try:
            result = search_programs(demos, time_budget=remaining)
            if result is not None:
                # Verify on test (the search already verified on demos)
                return {
                    'program': result,
                    'source': 'search',
                    'time': time.time() - t0,
                    'description': result.description,
                }
        except Exception:
            pass

    return {
        'program': None,
        'source': None,
        'time': time.time() - t0,
        'description': '',
    }
