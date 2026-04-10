"""Tests for TEMPLATE_BROADCAST: mask-driven blockwise template placement."""

from __future__ import annotations

import numpy as np

from aria.search.ast import ASTNode, Op
from aria.search.executor import execute_ast
from aria.search.derive import derive_programs


def test_template_broadcast_direct_ast_execution():
    """Direct AST execution: out = kron(input != bg, input)."""
    inp = np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.int8)

    expected = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.int8)

    node = ASTNode(Op.TEMPLATE_BROADCAST, [ASTNode(Op.INPUT)], param={})
    result = execute_ast(node, inp)

    assert result is not None
    assert np.array_equal(result, expected), f"got:\n{result}\nexpected:\n{expected}"


def test_template_broadcast_with_explicit_bg():
    """Explicit bg param overrides auto-detection."""
    inp = np.array([
        [5, 0],
        [0, 5],
    ], dtype=np.int8)

    # bg=5: mask = [[0,1],[1,0]], template = inp
    expected = np.kron((inp != 5).astype(int), inp)

    node = ASTNode(Op.TEMPLATE_BROADCAST, [ASTNode(Op.INPUT)], param={'bg': 5})
    result = execute_ast(node, inp)

    assert result is not None
    assert np.array_equal(result, expected)


def test_007bbfb7_solves_via_search():
    """Task 007bbfb7 must solve through canonical aria/search."""
    from aria.datasets import get_dataset, load_arc_task

    ds = get_dataset('v1-train')
    task = load_arc_task(ds, '007bbfb7')

    demos = [(p.input, p.output) for p in task.train]
    progs = derive_programs(demos)

    assert progs, "derive_programs returned no candidates for 007bbfb7"

    # At least one program must verify on all training demos
    solved = False
    for p in progs:
        if all(np.array_equal(p.execute(pair.input), pair.output) for pair in task.train):
            solved = True
            # Also check test if available
            if task.test:
                test_ok = all(np.array_equal(p.execute(pair.input), pair.output)
                              for pair in task.test)
                assert test_ok, f"train-verified but test-failed for {p.provenance}"
            break

    assert solved, "no derive program verified on all training demos for 007bbfb7"
