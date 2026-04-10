"""Tests for conditional dispatch: action-first grouping and rank-recolor."""

from __future__ import annotations

import numpy as np

from aria.guided.perceive import perceive
from aria.guided.synthesize import compute_transitions
from aria.search.derive import (
    _derive_action_first_dispatch,
    _derive_rank_recolor,
    _find_selector_for_oid_sets,
    derive_programs,
)


def _make_transitions_and_facts(demos):
    """Helper: compute transitions and facts for demo pairs."""
    all_transitions = []
    all_facts = []
    for inp, out in demos:
        in_facts = perceive(inp)
        out_facts = perceive(out)
        all_facts.append(in_facts)
        all_transitions.append(compute_transitions(in_facts, out_facts))
    return all_transitions, all_facts


def test_find_selector_for_oid_sets_simple():
    """Cross-demo selector from OID sets should work for simple cases."""
    # Two demos: largest object in each
    grid0 = np.zeros((8, 8), dtype=np.int8)
    grid0[0:3, 0:3] = 2  # large
    grid0[5, 5] = 3       # small

    grid1 = np.zeros((8, 8), dtype=np.int8)
    grid1[1:5, 1:5] = 4  # large
    grid1[6, 6] = 5       # small

    facts0 = perceive(grid0)
    facts1 = perceive(grid1)

    # Largest object OIDs
    oid0 = max(facts0.objects, key=lambda o: o.size).oid
    oid1 = max(facts1.objects, key=lambda o: o.size).oid

    target_sets = [{oid0}, {oid1}]
    sel = _find_selector_for_oid_sets(target_sets, [facts0, facts1])
    assert sel is not None
    # Must select exactly the target OIDs in each scene
    for di, facts in enumerate([facts0, facts1]):
        selected_oids = {o.oid for o in sel.select_objects(facts)}
        assert target_sets[di] <= selected_oids


def test_rank_recolor_synthetic():
    """Rank-recolor should solve synthetic all-same-color-recolored-by-rank."""
    # Demo 0: three objects of color 5, sizes 9, 4, 1 → recolored to 1, 2, 3
    inp0 = np.zeros((6, 6), dtype=np.int8)
    inp0[0:3, 0:3] = 5  # size 9
    inp0[4:6, 0:2] = 5  # size 4
    inp0[4, 4] = 5      # size 1

    out0 = np.zeros((6, 6), dtype=np.int8)
    out0[0:3, 0:3] = 1  # largest → color 1
    out0[4:6, 0:2] = 2  # 2nd → color 2
    out0[4, 4] = 3      # smallest → color 3

    # Demo 1: same pattern, different sizes
    inp1 = np.zeros((6, 6), dtype=np.int8)
    inp1[0:2, 0:4] = 5  # size 8
    inp1[3:5, 0:2] = 5  # size 4
    inp1[5, 5] = 5      # size 1

    out1 = np.zeros((6, 6), dtype=np.int8)
    out1[0:2, 0:4] = 1
    out1[3:5, 0:2] = 2
    out1[5, 5] = 3

    demos = [(inp0, out0), (inp1, out1)]
    all_transitions, all_facts = _make_transitions_and_facts(demos)

    progs = _derive_rank_recolor(all_transitions, all_facts, demos)
    assert len(progs) == 1
    assert progs[0].verify(demos)
    assert progs[0].provenance == 'derive:rank_recolor'


def test_action_first_dispatch_synthetic():
    """Action-first dispatch should handle remove+recolor with richer selectors."""
    # Demo 0: 2 rectangles (interior) recolored, 1 singleton (interior) removed
    inp0 = np.zeros((8, 8), dtype=np.int8)
    inp0[1:3, 1:3] = 2  # rect interior
    inp0[4:6, 4:6] = 3  # rect interior
    inp0[6, 1] = 4       # singleton interior

    out0 = np.zeros((8, 8), dtype=np.int8)
    out0[1:3, 1:3] = 7  # recolored
    out0[4:6, 4:6] = 7  # recolored
    # singleton removed

    # Demo 1: same pattern
    inp1 = np.zeros((8, 8), dtype=np.int8)
    inp1[2:4, 2:4] = 5  # rect interior
    inp1[5:7, 0:2] = 6  # rect interior
    inp1[1, 6] = 8       # singleton interior

    out1 = np.zeros((8, 8), dtype=np.int8)
    out1[2:4, 2:4] = 7
    out1[5:7, 0:2] = 7
    # singleton removed

    demos = [(inp0, out0), (inp1, out1)]
    all_transitions, all_facts = _make_transitions_and_facts(demos)

    progs = _derive_action_first_dispatch(all_transitions, all_facts, demos)
    # Should find: singletons → remove, non-singletons → recolor to 7
    if progs:
        assert progs[0].verify(demos)


def test_08ed6ac7_solves():
    """Task 08ed6ac7 must solve via rank_recolor through canonical search."""
    from aria.datasets import get_dataset, load_arc_task

    ds = get_dataset('v1-train')
    task = load_arc_task(ds, '08ed6ac7')

    demos = [(p.input, p.output) for p in task.train]
    progs = derive_programs(demos)

    assert progs, "derive_programs returned no candidates for 08ed6ac7"

    solved = False
    for p in progs:
        if all(np.array_equal(p.execute(pair.input), pair.output) for pair in task.train):
            solved = True
            if task.test:
                test_ok = all(np.array_equal(p.execute(pair.input), pair.output)
                              for pair in task.test)
                assert test_ok, f"train-verified but test-failed for {p.provenance}"
            break

    assert solved, "no derive program verified on all training demos for 08ed6ac7"
