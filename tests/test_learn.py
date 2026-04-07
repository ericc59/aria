"""Tests for the learn-propose-verify loop."""

from __future__ import annotations

from aria.core.graph import (
    CompileSuccess,
    CompileFailure,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    ResolvedBinding,
    Specialization,
)
from aria.core.library import GraphLibrary
from aria.core.proposer import propose_from_library, _compose_sequential
from aria.core.learn import learn_and_propose
from aria.core.arc import (
    ARCCompiler,
    ARCFitter,
    ARCSpecializer,
    ARCVerifier,
    solve_arc_task,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Library tests
# ---------------------------------------------------------------------------


def test_library_add_and_retrieve():
    lib = GraphLibrary()
    g = ComputationGraph(
        task_id="t1",
        nodes={"a": GraphNode(id="a", op="rotate", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t1", bindings=())
    lib.add(g, spec, source_task_id="t1")

    assert lib.size == 1
    assert "rotate" in lib.known_ops
    assert len(lib.templates_with_op("rotate")) == 1


def test_library_unique_structures():
    lib = GraphLibrary()
    for i, op in enumerate(["rotate", "rotate", "reflect"]):
        g = ComputationGraph(
            task_id=f"t{i}",
            nodes={"a": GraphNode(id="a", op=op, inputs=("input",))},
            output_id="a",
        )
        lib.add(g, Specialization(task_id=f"t{i}", bindings=()), f"t{i}")

    structs = lib.unique_graph_structures()
    assert len(structs) == 2  # rotate and reflect


# ---------------------------------------------------------------------------
# Proposer tests
# ---------------------------------------------------------------------------


def test_proposer_generates_proposals_from_library():
    lib = GraphLibrary()

    # Add a rotate template
    g = ComputationGraph(
        task_id="src",
        nodes={"a": GraphNode(
            id="a", op="APPLY_TRANSFORM", inputs=("input",),
            evidence={"transform": "rotate", "degrees": 180},
        )},
        output_id="a",
    )
    lib.add(g, Specialization(task_id="src", bindings=()), "src")

    # Propose for a new task
    proposals = propose_from_library(lib, [], task_id="new")
    assert len(proposals) >= 1
    # Should include the adapted template
    assert any(
        any(n.op == "APPLY_TRANSFORM" for n in p.nodes.values())
        for p in proposals
    )


def test_proposer_composes_two_templates():
    lib = GraphLibrary()

    for op_name, evidence in [
        ("APPLY_TRANSFORM", {"transform": "rotate", "degrees": 90}),
        ("APPLY_TRANSFORM", {"transform": "reflect", "axis": "row"}),
    ]:
        g = ComputationGraph(
            task_id=f"src_{op_name}",
            nodes={"a": GraphNode(
                id="a", op=op_name, inputs=("input",),
                evidence=evidence,
            )},
            output_id="a",
        )
        lib.add(g, Specialization(task_id=f"src_{op_name}", bindings=()), f"src_{op_name}")

    proposals = propose_from_library(lib, [], task_id="new")

    # Should include composed graphs (2-step)
    two_step = [p for p in proposals if len(p.nodes) == 2]
    assert len(two_step) >= 1


def test_compose_sequential_creates_valid_graph():
    from aria.core.library import GraphTemplate

    t1 = GraphTemplate(
        graph=ComputationGraph(
            task_id="a",
            nodes={"x": GraphNode(id="x", op="rotate", inputs=("input",),
                                   evidence={"transform": "rotate", "degrees": 90})},
            output_id="x",
        ),
        specialization=Specialization(task_id="a", bindings=()),
        source_task_id="a",
    )
    t2 = GraphTemplate(
        graph=ComputationGraph(
            task_id="b",
            nodes={"x": GraphNode(id="x", op="reflect", inputs=("input",),
                                   evidence={"transform": "reflect", "axis": "row"})},
            output_id="x",
        ),
        specialization=Specialization(task_id="b", bindings=()),
        source_task_id="b",
    )

    composed = _compose_sequential(t1, t2, "test")
    assert composed is not None
    assert len(composed.nodes) == 2
    assert composed.validate() == []
    # step2 should depend on step1
    assert "step1" in composed.nodes["step2"].inputs


# ---------------------------------------------------------------------------
# End-to-end learn loop with ARC tasks
# ---------------------------------------------------------------------------


def _tile_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 6, 5, 6], [7, 8, 7, 8],
                [5, 6, 5, 6], [7, 8, 7, 8],
            ]),
        ),
    )


def _rotate_180_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _unsolvable_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
    )


def test_learn_loop_solves_known_tasks():
    """Phase 1 should solve tasks that fitters handle."""
    tasks = [
        ("tile", _tile_task()),
        ("rotate", _rotate_180_task()),
        ("impossible", _unsolvable_task()),
    ]

    result = learn_and_propose(
        tasks,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
    )

    assert result.phase1_solved == 2
    assert result.total_tasks == 3
    assert "tile" in result.solved_task_ids
    assert "rotate" in result.solved_task_ids
    assert result.library_size == 2


def test_learn_loop_library_grows():
    """Library should contain templates from all solved tasks."""
    tasks = [
        ("tile", _tile_task()),
        ("rotate", _rotate_180_task()),
    ]
    lib = GraphLibrary()

    learn_and_propose(
        tasks,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        library=lib,
    )

    assert lib.size == 2
    # Library should know about the ops used
    assert len(lib.known_ops) >= 1


def test_proposer_adapts_template_for_new_task():
    """A rotate template should adapt to solve a different rotation task."""
    # First solve a rotate-180 task
    lib = GraphLibrary()
    tasks_phase1 = [("rot180", _rotate_180_task())]

    learn_and_propose(
        tasks_phase1,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        library=lib,
    )
    assert lib.size >= 1

    # Now propose for a rotate-90 task (different rotation)
    rotate_90_task = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 1], [4, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 5], [8, 6]]),
        ),
    )

    proposals = propose_from_library(lib, rotate_90_task, task_id="rot90")
    # Should propose parameterized variants including 90-degree
    assert len(proposals) >= 1
