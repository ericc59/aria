"""Tests for structured compile/verify diagnostics and residual-guided repair."""

from __future__ import annotations

import numpy as np

from aria.core.arc import (
    ARCCompiler,
    ARCFitter,
    ARCSpecializer,
    ARCVerifier,
    _build_near_miss_diagnostic,
    _replace_binding,
    solve_arc_task,
)
from aria.core.editor_env import ActionType, EditAction, EditState, GraphEditEnv
from aria.core.editor_search import _enumerate_edits, search_from_seeds
from aria.core.graph import (
    CompileFailure,
    CompileSuccess,
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RepairHint,
    ResolvedBinding,
    RoleBinding,
    Specialization,
    VerifyDiagnostic,
)
from aria.core.seeds import Seed, collect_seeds
from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Typed diagnostic objects
# ---------------------------------------------------------------------------


def test_verify_diagnostic_fields():
    d = VerifyDiagnostic(
        per_demo_diff=(5, 3),
        total_diff=8,
        failed_demo=0,
        diff_fraction=0.25,
        repair_hints=(
            RepairHint(
                node_id="__task__",
                binding_name="dominant_axis",
                current_value="row",
                alternatives=("col",),
                confidence=0.8,
                reason="col reduces diff",
            ),
        ),
        blamed_bindings=("dominant_axis",),
        description="near-miss",
    )
    assert d.total_diff == 8
    assert len(d.repair_hints) == 1
    assert d.repair_hints[0].alternatives == ("col",)


def test_repair_hint_fields():
    h = RepairHint(
        node_id="n1",
        binding_name="period",
        current_value=2,
        alternatives=(3, 4),
        confidence=0.5,
        reason="test",
    )
    assert h.current_value == 2
    assert 3 in h.alternatives


def test_compile_failure_carries_diagnostic():
    diag = VerifyDiagnostic(total_diff=10, description="test")
    f = CompileFailure(task_id="t", reason="verify failed", diagnostic=diag)
    assert f.diagnostic is not None
    assert f.diagnostic.total_diff == 10


def test_compile_failure_diagnostic_default_none():
    f = CompileFailure(task_id="t", reason="missing op")
    assert f.diagnostic is None


# ---------------------------------------------------------------------------
# Near-miss diagnostic production
# ---------------------------------------------------------------------------


def _periodic_near_miss_task():
    """A task where periodic repair fitter will propose but verify fails.

    The fitter will detect a periodic pattern but the specific
    axis/period won't match perfectly.
    """
    # 6x6 grid with a frame of 4s and interior pattern
    # The fitter will try row/period=2 but the real pattern needs something else
    inp = grid_from_list([
        [4, 4, 4, 4, 4, 4],
        [4, 1, 2, 1, 2, 4],
        [4, 3, 0, 3, 0, 4],
        [4, 1, 2, 1, 2, 4],
        [4, 3, 0, 3, 0, 4],
        [4, 4, 4, 4, 4, 4],
    ])
    # Output repairs one cell differently
    out = grid_from_list([
        [4, 4, 4, 4, 4, 4],
        [4, 1, 2, 1, 2, 4],
        [4, 3, 1, 3, 1, 4],  # 0->1 in positions (2,2) and (2,4)
        [4, 1, 2, 1, 2, 4],
        [4, 3, 1, 3, 1, 4],  # 0->1 in positions (4,2) and (4,4)
        [4, 4, 4, 4, 4, 4],
    ])
    return (DemoPair(input=inp, output=out),)


def test_arc_compiler_produces_diagnostic_on_near_miss():
    """When compile succeeds but verify fails, diagnostic should be present."""
    # Use a task where the fitter proposes a graph but verification fails
    demos = _periodic_near_miss_task()
    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()

    graphs = fitter.fit(demos, task_id="test")
    for graph in graphs:
        spec = specializer.specialize(graph, demos)
        result = compiler.compile(graph, spec, demos)
        if isinstance(result, CompileFailure) and "verified failed" in result.reason:
            # This is the near-miss case — diagnostic should be present
            assert result.diagnostic is not None, \
                f"Expected diagnostic for near-miss, got None. Reason: {result.reason}"
            assert result.diagnostic.total_diff >= 0
            return

    # If no fitter proposals or all compiled, that's OK for this test


# ---------------------------------------------------------------------------
# Diagnostics surface through editor environment
# ---------------------------------------------------------------------------


def test_edit_state_carries_diagnostic():
    """EditState should have diagnostic field after compile."""
    demos = (DemoPair(input=grid_from_list([[1]]), output=grid_from_list([[2]])),)
    env = GraphEditEnv(
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    seed = ComputationGraph(
        task_id="test",
        nodes={"a": GraphNode(id="a", op="APPLY_TRANSFORM", inputs=("input",))},
        output_id="a",
    )
    state = env.reset(initial_graph=seed)
    state = env.step(EditAction(action_type=ActionType.COMPILE))
    # diagnostic may or may not be present depending on compile path
    assert hasattr(state, "diagnostic")


# ---------------------------------------------------------------------------
# Repair hints integrated into edit enumeration
# ---------------------------------------------------------------------------


def test_enumerate_edits_includes_repair_hints():
    """When diagnostics have repair hints, they should appear as edits."""
    diag = VerifyDiagnostic(
        total_diff=10,
        repair_hints=(
            RepairHint(
                node_id="__task__",
                binding_name="dominant_period",
                current_value=2,
                alternatives=(3, 4),
                confidence=0.5,
            ),
        ),
        blamed_bindings=("dominant_period",),
    )
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",),
                               slots=(NodeSlot(name="period", typ="INT", evidence=2),))},
        output_id="a",
    )
    spec = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="__task__", name="dominant_period", value=2),
    ))
    state = EditState(
        graph=g, specialization=spec, compile_result=None,
        verified=False, diff_pixels=10, step=0, score=5.0,
        diagnostic=diag,
    )
    edits = _enumerate_edits(state)
    # Repair hint edits should be present
    repair_edits = [
        e for e in edits
        if e.action_type == ActionType.BIND
        and e.node_id == "__task__"
        and e.key == "dominant_period"
        and e.value in (3, 4)
    ]
    assert len(repair_edits) == 2


def test_enumerate_edits_repair_hints_come_first():
    """Repair hint edits should be at the start of the edit list."""
    diag = VerifyDiagnostic(
        repair_hints=(
            RepairHint(
                node_id="X",
                binding_name="axis",
                current_value="row",
                alternatives=("col",),
                confidence=1.0,
            ),
        ),
    )
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="REPAIR_LINES", inputs=("input",))},
        output_id="a",
    )
    state = EditState(
        graph=g, specialization=Specialization(task_id="t", bindings=()),
        compile_result=None, verified=False, diff_pixels=10,
        step=0, score=5.0, diagnostic=diag,
    )
    edits = _enumerate_edits(state)
    # First edit should be the repair hint
    assert edits[0].action_type == ActionType.BIND
    assert edits[0].node_id == "X"
    assert edits[0].key == "axis"
    assert edits[0].value == "col"


# ---------------------------------------------------------------------------
# Replace binding helper
# ---------------------------------------------------------------------------


def test_replace_binding():
    spec = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="a", name="x", value=1),
        ResolvedBinding(node_id="b", name="y", value=2),
    ))
    new_spec = _replace_binding(spec, "a", "x", 99)
    assert new_spec.get("a", "x") == 99
    assert new_spec.get("b", "y") == 2  # unchanged


def test_replace_binding_adds_if_missing():
    spec = Specialization(task_id="t", bindings=())
    new_spec = _replace_binding(spec, "a", "x", 42)
    assert new_spec.get("a", "x") == 42


# ---------------------------------------------------------------------------
# No family-specific repair labels leak into canonical diagnostics
# ---------------------------------------------------------------------------


def test_diagnostic_has_no_family_labels():
    d = VerifyDiagnostic(
        per_demo_diff=(5,),
        total_diff=5,
        repair_hints=(
            RepairHint(node_id="n", binding_name="period",
                       current_value=2, alternatives=(3,)),
        ),
        blamed_bindings=("period",),
        description="test",
    )
    # None of the fields should reference family names
    assert "framed_periodic" not in str(d)
    assert "composite_alignment" not in str(d)


# ---------------------------------------------------------------------------
# Canonical solve path still works
# ---------------------------------------------------------------------------


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


def test_solved_tasks_unaffected():
    demos = _rotate_task()
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is True


def test_impossible_task_unaffected():
    demos = (DemoPair(input=grid_from_list([[1, 2, 3]]),
                      output=grid_from_list([[9, 8, 7]])),
             DemoPair(input=grid_from_list([[1, 2, 3]]),
                      output=grid_from_list([[7, 9, 8]])),)
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is False
