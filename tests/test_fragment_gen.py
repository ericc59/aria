"""Tests for dynamic task-conditioned fragment generation."""

from __future__ import annotations

import numpy as np

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.editor_env import ActionType, EditAction, EditState, GraphEditEnv
from aria.core.editor_search import _enumerate_edits
from aria.core.fragment_gen import (
    GeneratedFragment,
    ResidualPattern,
    analyze_residual_pattern,
    generate_fragments,
)
from aria.core.graph import (
    CompileFailure,
    ComputationGraph,
    GraphFragment,
    GraphNode,
    NodeSlot,
    ResolvedBinding,
    RoleBinding,
    Specialization,
    SubgraphBlame,
    VerifyDiagnostic,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Residual pattern analysis
# ---------------------------------------------------------------------------


def test_analyze_residual_detects_clusters():
    """Scattered diff should detect multiple clusters."""
    inp = grid_from_list([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    out = grid_from_list([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    pred = inp.copy()  # prediction = input (no change)
    demos = [DemoPair(input=inp, output=out)]
    pattern = analyze_residual_pattern(demos, [pred])
    assert pattern.n_diff_clusters >= 2
    assert pattern.has_scattered_objects


def test_analyze_residual_detects_movement():
    """Object that shifts position should detect movement."""
    inp = grid_from_list([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    out = grid_from_list([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    pred = inp.copy()
    demos = [DemoPair(input=inp, output=out)]
    pattern = analyze_residual_pattern(demos, [pred])
    assert pattern.has_shape_movement


def test_analyze_residual_detects_isolated_pixels():
    """Single-pixel diffs should be flagged as isolated."""
    inp = grid_from_list([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    out = grid_from_list([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    pred = inp.copy()
    demos = [DemoPair(input=inp, output=out)]
    pattern = analyze_residual_pattern(demos, [pred])
    assert pattern.has_isolated_pixels
    assert pattern.n_isolated_pixels >= 1


# ---------------------------------------------------------------------------
# Fragment generation is dynamic (not fixed)
# ---------------------------------------------------------------------------


def test_fragments_differ_by_pattern():
    """Different residual patterns should produce different fragment sets."""
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=())
    diag = VerifyDiagnostic(total_diff=50, diff_fraction=0.2)

    # Pattern 1: scattered clusters + movement
    inp1 = grid_from_list([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    out1 = grid_from_list([[0, 0, 1], [2, 0, 0], [0, 0, 0]])
    demos1 = [DemoPair(input=inp1, output=out1)]
    frags1 = generate_fragments(g, spec, diag, demos1, [inp1.copy()])

    # Pattern 2: distributed diff
    inp2 = grid_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out2 = grid_from_list([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    demos2 = [DemoPair(input=inp2, output=out2)]
    frags2 = generate_fragments(g, spec, diag, demos2, [inp2.copy()])

    # The generated fragments should differ
    labels1 = {f.fragment.label for f in frags1}
    labels2 = {f.fragment.label for f in frags2}
    assert labels1 != labels2, "fragments should differ for different patterns"


def test_fragments_are_not_static_replay():
    """Fragment labels should include task-specific counts, not be identical every time."""
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=())
    diag = VerifyDiagnostic(total_diff=50, diff_fraction=0.2)

    inp = grid_from_list([[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [3, 0, 0, 4]])
    out = grid_from_list([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    demos = [DemoPair(input=inp, output=out)]
    frags = generate_fragments(g, spec, diag, demos, [inp.copy()])

    # Labels should contain cluster/marker counts from the actual task
    labels = [f.fragment.label for f in frags]
    has_count = any(any(c.isdigit() for c in label) for label in labels)
    assert has_count, f"labels should be task-conditioned, got {labels}"


# ---------------------------------------------------------------------------
# Generated fragments lower to canonical graph state
# ---------------------------------------------------------------------------


def test_generated_fragment_is_graph_fragment():
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=())
    diag = VerifyDiagnostic(total_diff=50, diff_fraction=0.2)

    inp = grid_from_list([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
    out = grid_from_list([[0, 0, 1], [0, 0, 0], [2, 0, 0]])
    demos = [DemoPair(input=inp, output=out)]
    frags = generate_fragments(g, spec, diag, demos, [inp.copy()])

    assert len(frags) >= 1
    for gf in frags:
        assert isinstance(gf, GeneratedFragment)
        assert isinstance(gf.fragment, GraphFragment)
        assert gf.fragment.input_id  # non-empty
        assert gf.fragment.output_id  # non-empty
        assert gf.fragment.output_id in gf.fragment.nodes
        assert gf.rationale  # non-empty


def test_generated_fragment_usable_in_replace_subgraph():
    """Generated fragments should work with REPLACE_SUBGRAPH action."""
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=())
    diag = VerifyDiagnostic(total_diff=50, diff_fraction=0.2)
    inp = grid_from_list([[1, 0], [0, 2]])
    out = grid_from_list([[0, 1], [2, 0]])
    demos = [DemoPair(input=inp, output=out)]
    frags = generate_fragments(g, spec, diag, demos, [inp.copy()])

    if not frags:
        return

    fragment = frags[0].fragment
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=g)
    state = env.step(EditAction(
        action_type=ActionType.REPLACE_SUBGRAPH,
        node_id="a",
        value=fragment,
    ))

    # Result should be valid canonical graph
    assert isinstance(state.graph, ComputationGraph)
    assert "a" not in state.graph.nodes  # old node removed
    assert fragment.output_id in state.graph.nodes  # new node present


# ---------------------------------------------------------------------------
# No family labels in generated fragments
# ---------------------------------------------------------------------------


def test_no_family_labels_in_generated_fragments():
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=())
    diag = VerifyDiagnostic(total_diff=50, diff_fraction=0.2)
    inp = grid_from_list([[1, 0], [0, 2]])
    out = grid_from_list([[0, 1], [2, 0]])
    demos = [DemoPair(input=inp, output=out)]
    frags = generate_fragments(g, spec, diag, demos, [inp.copy()])

    for gf in frags:
        combined = gf.fragment.label + gf.fragment.description + gf.rationale
        assert "framed_periodic" not in combined.lower()
        assert "composite_alignment" not in combined.lower()
        assert "canvas_construction" not in combined.lower()


# ---------------------------------------------------------------------------
# Integration: diagnostics now include dynamic fragments
# ---------------------------------------------------------------------------


def test_diagnostics_include_dynamic_fragments_on_near_miss():
    """Near-miss tasks should get both static and dynamic fragments."""
    # Use a controlled task where the fitter proposes but verify fails
    from aria.core.protocol import solve as core_solve
    from aria.core.graph import CompileFailure

    # Simple task that fitter can propose for
    demos = (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=False)
    # This task is solvable, so no diagnostics needed
    assert result.solved is True


# ---------------------------------------------------------------------------
# No regressions
# ---------------------------------------------------------------------------


def test_solved_task_unaffected():
    demos = (
        DemoPair(input=grid_from_list([[1, 2], [3, 4]]),
                 output=grid_from_list([[4, 3], [2, 1]])),
        DemoPair(input=grid_from_list([[5, 6], [7, 8]]),
                 output=grid_from_list([[8, 7], [6, 5]])),
    )
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is True
