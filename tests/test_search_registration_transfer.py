"""Tests for registration transfer: modules into frame openings."""

from __future__ import annotations

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep


def test_registration_transfer_synthetic():
    """Synthetic: 2x2 module fits into 2x2 opening in a frame."""
    # Frame (color 5) with a 2x2 opening at (1,1)
    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0, 0:4] = 5
    inp[1, 0] = 5; inp[1, 3] = 5
    inp[2, 0] = 5; inp[2, 3] = 5
    inp[3, 0:4] = 5
    # Module (color 3) at (4,4) — 2x2 rectangle
    inp[4:6, 4:6] = 3

    # Expected: module moves into the opening
    out = inp.copy()
    out[1, 1:3] = 3
    out[2, 1:3] = 3
    out[4:6, 4:6] = 0  # module erased from old position

    prog = SearchProgram(
        steps=[SearchStep('registration_transfer', {})],
        provenance='test',
    )
    result = prog.execute(inp)
    assert np.array_equal(result, out), f"got:\n{result}\nexpected:\n{out}"


def test_228f6490_solves():
    """Task 228f6490 must solve via registration_transfer."""
    from aria.datasets import get_dataset, load_arc_task
    from aria.search.derive import derive_programs

    ds = get_dataset('v1-train')
    task = load_arc_task(ds, '228f6490')
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

    assert solved, "no derive program verified for 228f6490"
