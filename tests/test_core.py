"""Tests for the domain-general core framework.

Tests both the abstract framework and the ARC domain instantiation.
"""

from __future__ import annotations

import numpy as np

from aria.datasets import get_dataset, load_arc_task
from aria.core.graph import (
    CompileFailure,
    CompileSuccess,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    ResolvedBinding,
    RoleBinding,
    Specialization,
)
from aria.core.protocol import SolveResult, solve
from aria.core.arc import (
    ARCCompiler,
    ARCFitter,
    ARCSpecializer,
    ARCVerifier,
    sketch_graph_to_core,
    solve_arc_task,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Core graph IR tests (domain-independent)
# ---------------------------------------------------------------------------


def test_computation_graph_construction():
    g = ComputationGraph(
        task_id="test",
        nodes={
            "a": GraphNode(id="a", op="select", inputs=("input",)),
            "b": GraphNode(id="b", op="transform", inputs=("a",)),
        },
        output_id="b",
    )
    assert len(g.nodes) == 2
    assert g.validate() == []


def test_computation_graph_topo_order():
    g = ComputationGraph(
        task_id="test",
        nodes={
            "a": GraphNode(id="a", op="extract", inputs=("input",)),
            "b": GraphNode(id="b", op="analyze", inputs=("a",)),
            "c": GraphNode(id="c", op="construct", inputs=("a", "b")),
        },
        output_id="c",
    )
    order = g.topo_order()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_computation_graph_validates_missing_dep():
    g = ComputationGraph(
        task_id="test",
        nodes={"a": GraphNode(id="a", op="x", inputs=("nonexistent",))},
        output_id="a",
    )
    errors = g.validate()
    assert any("nonexistent" in e for e in errors)


def test_computation_graph_op_set():
    g = ComputationGraph(
        task_id="test",
        nodes={
            "a": GraphNode(id="a", op="select", inputs=("input",)),
            "b": GraphNode(id="b", op="transform", inputs=("a",)),
        },
        output_id="b",
    )
    assert g.op_set == frozenset({"select", "transform"})


def test_computation_graph_role_kinds():
    g = ComputationGraph(
        task_id="test",
        nodes={
            "a": GraphNode(
                id="a", op="bind",
                roles=(RoleBinding(name="bg", kind="BACKGROUND"),),
            ),
        },
        output_id="a",
    )
    assert "BACKGROUND" in g.role_kinds


def test_specialization_lookup():
    spec = Specialization(
        task_id="test",
        bindings=(
            ResolvedBinding(node_id="a", name="x", value=42, source="evidence"),
            ResolvedBinding(node_id="a", name="y", value="hello", source="inferred"),
        ),
    )
    assert spec.get("a", "x") == 42
    assert spec.get("a", "y") == "hello"
    assert spec.get("a", "z") is None
    assert spec.get("b", "x") is None


def test_specialization_bindings_for_node():
    spec = Specialization(
        task_id="test",
        bindings=(
            ResolvedBinding(node_id="a", name="x", value=1),
            ResolvedBinding(node_id="a", name="y", value=2),
            ResolvedBinding(node_id="b", name="z", value=3),
        ),
    )
    assert len(spec.bindings_for_node("a")) == 2
    assert len(spec.bindings_for_node("b")) == 1


# ---------------------------------------------------------------------------
# Core pipeline with a toy domain (proves domain-independence)
# ---------------------------------------------------------------------------


class _ToyExample:
    """Toy domain: input is a number, output is a number."""
    def __init__(self, inp: int, out: int):
        self._input = inp
        self._output = out

    @property
    def input(self) -> int:
        return self._input

    @property
    def output(self) -> int:
        return self._output


class _ToyFitter:
    """Always proposes 'multiply by N' graph."""
    def fit(self, examples, task_id=""):
        # Infer N from first example
        ex = examples[0]
        if ex.input == 0:
            return []
        n = ex.output // ex.input
        if n * ex.input != ex.output:
            return []
        return [ComputationGraph(
            task_id=task_id,
            nodes={"mul": GraphNode(
                id="mul", op="multiply",
                inputs=("input",),
                slots=(NodeSlot(name="factor", typ="INT", evidence=n),),
            )},
            output_id="mul",
        )]


class _ToySpecializer:
    """Extract the factor from node evidence."""
    def specialize(self, graph, examples):
        bindings = []
        for node in graph.nodes.values():
            for slot in node.slots:
                if slot.evidence is not None:
                    bindings.append(ResolvedBinding(
                        node_id=node.id, name=slot.name,
                        value=slot.evidence, source="evidence",
                    ))
        return Specialization(task_id=graph.task_id, bindings=tuple(bindings))


class _ToyCompiler:
    """Compile multiply graph into a lambda."""
    def compile(self, graph, specialization, examples):
        factor = specialization.get("mul", "factor")
        if factor is None:
            return CompileFailure(task_id=graph.task_id, reason="no factor")
        program = lambda x: x * factor  # noqa: E731
        return CompileSuccess(
            task_id=graph.task_id, program=program,
            description=f"multiply by {factor}",
        )


class _ToyVR:
    def __init__(self, passed):
        self.passed = passed


class _ToyVerifier:
    """Check all examples."""
    def verify(self, program, examples):
        for ex in examples:
            if program(ex.input) != ex.output:
                return _ToyVR(False)
        return _ToyVR(True)


def test_core_pipeline_toy_domain():
    """The core pipeline works with a non-ARC domain."""
    examples = [_ToyExample(2, 6), _ToyExample(3, 9), _ToyExample(5, 15)]
    result = solve(
        examples=examples,
        fitter=_ToyFitter(),
        specializer=_ToySpecializer(),
        compiler=_ToyCompiler(),
        verifier=_ToyVerifier(),
        task_id="multiply_by_3",
    )
    assert result.solved is True
    assert result.winning_program is not None
    assert result.winning_program(7) == 21
    assert result.graphs_proposed == 1
    assert result.graphs_compiled == 1
    assert result.graphs_verified == 1


def test_core_pipeline_toy_domain_failure():
    """Pipeline returns solved=False when no graph fits."""
    examples = [_ToyExample(0, 5)]
    result = solve(
        examples=examples,
        fitter=_ToyFitter(),
        specializer=_ToySpecializer(),
        compiler=_ToyCompiler(),
        verifier=_ToyVerifier(),
        task_id="impossible",
    )
    assert result.solved is False
    assert result.graphs_proposed == 0


# ---------------------------------------------------------------------------
# ARC domain instantiation tests
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


def _rotate_task():
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


def test_arc_fitter_proposes_graphs():
    demos = _tile_task()
    fitter = ARCFitter()
    graphs = fitter.fit(demos, task_id="test")
    assert len(graphs) >= 1
    assert all(isinstance(g, ComputationGraph) for g in graphs)
    assert all(g.validate() == [] for g in graphs)


def test_arc_solve_tile():
    demos = _tile_task()
    result = solve_arc_task(demos, task_id="test_tile")
    assert result.solved is True
    assert result.graphs_proposed >= 1
    assert result.graphs_verified >= 1


def test_arc_solve_rotate():
    demos = _rotate_task()
    result = solve_arc_task(demos, task_id="test_rotate")
    assert result.solved is True


def test_arc_solve_nosolution():
    """Task with contradictory demos returns solved=False."""
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[7, 9, 8]]),
        ),
    )
    result = solve_arc_task(demos, task_id="test_impossible")
    assert result.solved is False


def test_arc_solve_stops_when_stage1_size_fails():
    demos = (
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1, 0]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 0], [0, 0]]),
            output=grid_from_list([[1], [0]]),
        ),
    )
    result = solve_arc_task(demos, task_id="test_stage1_fail")
    assert result.solved is False
    assert result.graphs_proposed == 0
    assert result.graphs_compiled == 0
    assert result.graphs_verified == 0
    assert result.attempts == ()


def test_arc_solve_can_succeed_via_stage1_derivation():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "1a6449f1")
    result = solve_arc_task(task.train, task_id="1a6449f1", use_editor_search=False)
    assert result.solved is True
    assert result.winning_program is not None
    assert result.attempts == ()


def test_arc_solve_can_succeed_via_stage1_scaled_object_render():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "f25fbde4")
    result = solve_arc_task(task.train, task_id="f25fbde4", use_editor_search=False)
    assert result.solved is True
    assert result.winning_program is not None
    assert result.attempts == ()


def test_arc_solve_can_succeed_via_stage1_marker_stack_render():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "12997ef3")
    result = solve_arc_task(task.train, task_id="12997ef3", use_editor_search=False)
    assert result.solved is True
    assert result.winning_program is not None
    assert result.attempts == ()


def test_arc_solve_can_succeed_via_stage1_tiled_input_render():
    ds = get_dataset("v2-train")
    task = load_arc_task(ds, "00576224")
    result = solve_arc_task(task.train, task_id="00576224", use_editor_search=False)
    assert result.solved is True
    assert result.winning_program is not None
    assert result.attempts == ()


def test_arc_core_graph_roundtrip():
    """SketchGraph → core ComputationGraph preserves structure."""
    from aria.sketch import SketchGraph, SketchNode, Primitive, RoleVar, RoleKind, Slot, SlotType

    sg = SketchGraph(
        task_id="test",
        nodes={
            "a": SketchNode(
                id="a", primitive=Primitive.BIND_ROLE, inputs=("input",),
                roles=(RoleVar("bg", RoleKind.BG),),
            ),
            "b": SketchNode(
                id="b", primitive=Primitive.APPLY_TRANSFORM, inputs=("a",),
                slots=(Slot("x", SlotType.INT, evidence=42),),
            ),
        },
        output_id="b",
    )
    core = sketch_graph_to_core(sg)
    assert len(core.nodes) == 2
    assert core.nodes["a"].op == "BIND_ROLE"
    assert core.nodes["b"].slots[0].evidence == 42
    assert core.validate() == []


def test_solve_result_records_attempts():
    """SolveResult should record all compilation attempts."""
    demos = _tile_task()
    result = solve_arc_task(demos, task_id="test")
    assert len(result.attempts) >= 1
    # The winning attempt should be verified
    winning = [a for a in result.attempts if a.verified]
    assert len(winning) >= 1
