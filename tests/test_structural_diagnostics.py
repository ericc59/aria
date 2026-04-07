"""Tests for structural diagnostics: region residuals, subgraph blame, replacement actions."""

from __future__ import annotations

import numpy as np

from aria.core.arc import (
    ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier,
    _compute_subgraph_blame, _compatible_replacements,
    _get_replacement_fragments, solve_arc_task,
)
from aria.core.editor_env import ActionType, EditAction, EditState, GraphEditEnv
from aria.core.editor_search import _enumerate_edits
from aria.core.graph import (
    CompileFailure, ComputationGraph, GraphFragment, GraphNode,
    NodeSlot, RegionResidual, RepairHint, ResolvedBinding,
    RoleBinding, Specialization, SubgraphBlame, VerifyDiagnostic,
)
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Region residuals
# ---------------------------------------------------------------------------


def test_region_residual_fields():
    r = RegionResidual(
        region_label="interior",
        diff_pixels=10,
        total_pixels=100,
        diff_fraction=0.1,
    )
    assert r.region_label == "interior"
    assert r.diff_fraction == 0.1


# ---------------------------------------------------------------------------
# Subgraph blame
# ---------------------------------------------------------------------------


def test_subgraph_blame_fields():
    b = SubgraphBlame(
        node_ids=("repaired", "motif_repaired"),
        ops=("REPAIR_LINES", "REPAIR_2D_MOTIF"),
        residual_overlap=1.0,
        confidence=0.7,
        replacement_labels=("unary_transform", "motif_repair_only"),
        reason="parameter repair saturated",
    )
    assert len(b.node_ids) == 2
    assert "unary_transform" in b.replacement_labels


# ---------------------------------------------------------------------------
# Graph fragments
# ---------------------------------------------------------------------------


def test_replacement_fragment_library():
    frags = _get_replacement_fragments()
    assert len(frags) >= 4
    assert "unary_transform" in frags
    assert "motif_repair_only" in frags
    assert "line_repair_only" in frags
    assert "relation_alignment" in frags

    # Each fragment has valid structure
    for label, frag in frags.items():
        assert isinstance(frag, GraphFragment)
        assert frag.label == label
        assert frag.input_id  # non-empty
        assert frag.output_id  # non-empty
        assert frag.output_id in frag.nodes


def test_fragment_has_no_family_labels():
    """Fragments must be structurally described, not by family name."""
    frags = _get_replacement_fragments()
    for label, frag in frags.items():
        combined = label + frag.description
        assert "framed_periodic" not in combined.lower()
        assert "composite_alignment" not in combined.lower()
        assert "canvas_construction" not in combined.lower()


# ---------------------------------------------------------------------------
# Compatible replacements
# ---------------------------------------------------------------------------


def test_compatible_replacements_for_repair():
    labels = _compatible_replacements(["REPAIR_LINES", "REPAIR_2D_MOTIF"])
    assert "unary_transform" in labels
    assert "motif_repair_only" in labels
    assert "line_repair_only" in labels


def test_compatible_replacements_for_transform():
    labels = _compatible_replacements(["APPLY_TRANSFORM"])
    assert "relation_alignment" in labels
    assert "partition_local_rule" in labels


# ---------------------------------------------------------------------------
# REPLACE_SUBGRAPH action in editor env
# ---------------------------------------------------------------------------


def test_replace_subgraph_lowers_to_canonical_graph():
    """REPLACE_SUBGRAPH produces an ordinary ComputationGraph."""
    frags = _get_replacement_fragments()
    fragment = frags["unary_transform"]

    # Build a graph with repair nodes to replace
    graph = ComputationGraph(
        task_id="t",
        nodes={
            "roles": GraphNode(id="roles", op="BIND_ROLE", inputs=("input",),
                               roles=(RoleBinding(name="bg", kind="BG"),)),
            "repaired": GraphNode(id="repaired", op="REPAIR_LINES",
                                   inputs=("roles",)),
            "motif": GraphNode(id="motif", op="REPAIR_2D_MOTIF",
                               inputs=("repaired",)),
        },
        output_id="motif",
    )
    spec = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="repaired", name="axis", value="row"),
    ))

    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=graph)

    # Apply specialization
    for b in spec.bindings:
        state = env.step(EditAction(
            action_type=ActionType.BIND,
            node_id=b.node_id, key=b.name, value=b.value,
        ))

    # Replace the repair subgraph with unary_transform
    state = env.step(EditAction(
        action_type=ActionType.REPLACE_SUBGRAPH,
        node_id="repaired,motif",
        value=fragment,
    ))

    # Result should be a valid ComputationGraph
    assert isinstance(state.graph, ComputationGraph)
    assert state.graph.validate() == []

    # Repair nodes should be gone
    assert "repaired" not in state.graph.nodes
    assert "motif" not in state.graph.nodes

    # Fragment node should be present
    assert "_frag_t" in state.graph.nodes
    assert state.graph.nodes["_frag_t"].op == "APPLY_TRANSFORM"

    # Output should point to the fragment output
    assert state.graph.output_id == "_frag_t"

    # Bindings for removed nodes should be gone
    assert state.specialization.get("repaired", "axis") is None


def test_replace_subgraph_preserves_perception_nodes():
    """REPLACE_SUBGRAPH should not remove perception/setup nodes."""
    frags = _get_replacement_fragments()
    fragment = frags["unary_transform"]

    graph = ComputationGraph(
        task_id="t",
        nodes={
            "roles": GraphNode(id="roles", op="BIND_ROLE", inputs=("input",)),
            "interior": GraphNode(id="interior", op="PEEL_FRAME", inputs=("roles",)),
            "repaired": GraphNode(id="repaired", op="REPAIR_LINES",
                                   inputs=("interior",)),
        },
        output_id="repaired",
    )

    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="t",
    )
    state = env.reset(initial_graph=graph)

    state = env.step(EditAction(
        action_type=ActionType.REPLACE_SUBGRAPH,
        node_id="repaired",
        value=fragment,
    ))

    # Perception nodes preserved
    assert "roles" in state.graph.nodes
    assert "interior" in state.graph.nodes

    # Fragment connected to interior
    frag_node = state.graph.nodes["_frag_t"]
    assert "interior" in frag_node.inputs


# ---------------------------------------------------------------------------
# Enumerate edits includes replacement actions
# ---------------------------------------------------------------------------


def test_enumerate_edits_includes_subgraph_replacements():
    """When diagnostic has subgraph blame, enumeration should include REPLACE_SUBGRAPH."""
    frags = _get_replacement_fragments()
    diag = VerifyDiagnostic(
        total_diff=87,
        diff_fraction=0.13,
        subgraph_blames=(
            SubgraphBlame(
                node_ids=("repaired", "motif"),
                ops=("REPAIR_LINES", "REPAIR_2D_MOTIF"),
                residual_overlap=1.0,
                confidence=0.7,
                replacement_labels=("unary_transform", "motif_repair_only"),
            ),
        ),
        replacement_fragments=(
            frags["unary_transform"],
            frags["motif_repair_only"],
        ),
    )

    graph = ComputationGraph(
        task_id="t",
        nodes={
            "roles": GraphNode(id="roles", op="BIND_ROLE", inputs=("input",)),
            "repaired": GraphNode(id="repaired", op="REPAIR_LINES", inputs=("roles",)),
            "motif": GraphNode(id="motif", op="REPAIR_2D_MOTIF", inputs=("repaired",)),
        },
        output_id="motif",
    )
    state = EditState(
        graph=graph,
        specialization=Specialization(task_id="t", bindings=()),
        compile_result=None, verified=False, diff_pixels=87,
        step=0, score=5.0, diagnostic=diag,
    )

    edits = _enumerate_edits(state)
    replace_edits = [e for e in edits if e.action_type == ActionType.REPLACE_SUBGRAPH]
    assert len(replace_edits) >= 2  # unary_transform + motif_repair_only

    # Replacement edits should come first
    first_replace_idx = next(i for i, e in enumerate(edits)
                            if e.action_type == ActionType.REPLACE_SUBGRAPH)
    assert first_replace_idx == 0


# ---------------------------------------------------------------------------
# No regressions on solved tasks
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


def test_impossible_task_unaffected():
    demos = (DemoPair(input=grid_from_list([[1, 2, 3]]),
                      output=grid_from_list([[9, 8, 7]])),
             DemoPair(input=grid_from_list([[1, 2, 3]]),
                      output=grid_from_list([[7, 9, 8]])),)
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is False
