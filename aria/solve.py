"""Top-level ARC solver for the canonical `aria/search` stack.

This entrypoint is intentionally search-only. Legacy guided and offline
solvers are retained elsewhere for ablation or reference, but they are not
part of the current default architecture.
"""

from __future__ import annotations

import time

from aria.types import Grid


def solve_task(
    demos: list[tuple[Grid, Grid]],
    time_budget: float = 30.0,
) -> dict:
    """Solve an ARC task given demo pairs.

    Returns dict with:
        'program': executable program (or None)
        'source': which engine found it ('search' or None)
        'time': seconds taken
        'description': human-readable program description
    """
    t0 = time.time()
    from aria.search.search import search_programs

    try:
        result = search_programs(demos, time_budget=time_budget)
        if result is not None:
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
