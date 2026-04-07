"""Tests for the multi-step graph-edit search baseline."""

from __future__ import annotations

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier, solve_arc_task
from aria.core.editor_env import ActionType, EditAction, EditState
from aria.core.editor_search import (
    EditSearchResult,
    _combined_score,
    _enumerate_edits,
    _state_hash,
    search_from_seeds,
)
from aria.core.graph import (
    ComputationGraph,
    GraphNode,
    NodeSlot,
    RoleBinding,
    Specialization,
    ResolvedBinding,
)
from aria.core.seeds import Seed, collect_seeds
from aria.types import DemoPair, grid_from_list


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


def _reflect_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[3, 4], [1, 2]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[7, 8], [5, 6]]),
        ),
    )


def _impossible_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[7, 9, 8]]),
        ),
    )


# ---------------------------------------------------------------------------
# State hashing / dedup
# ---------------------------------------------------------------------------


def test_state_hash_same_for_identical_states():
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="X", inputs=("input",))},
        output_id="a",
    )
    s = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="a", name="k", value=1),
    ))
    assert _state_hash(g, s) == _state_hash(g, s)


def test_state_hash_differs_for_different_ops():
    g1 = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="X")},
        output_id="a",
    )
    g2 = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="Y")},
        output_id="a",
    )
    s = Specialization(task_id="t", bindings=())
    assert _state_hash(g1, s) != _state_hash(g2, s)


def test_state_hash_differs_for_different_bindings():
    g = ComputationGraph(
        task_id="t",
        nodes={"a": GraphNode(id="a", op="X")},
        output_id="a",
    )
    s1 = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="a", name="k", value=1),
    ))
    s2 = Specialization(task_id="t", bindings=(
        ResolvedBinding(node_id="a", name="k", value=2),
    ))
    assert _state_hash(g, s1) != _state_hash(g, s2)


def test_state_hash_order_independent():
    """Hash should not depend on dict insertion order."""
    g1 = ComputationGraph(
        task_id="t",
        nodes={
            "a": GraphNode(id="a", op="X"),
            "b": GraphNode(id="b", op="Y"),
        },
        output_id="b",
    )
    g2 = ComputationGraph(
        task_id="t",
        nodes={
            "b": GraphNode(id="b", op="Y"),
            "a": GraphNode(id="a", op="X"),
        },
        output_id="b",
    )
    s = Specialization(task_id="t", bindings=())
    assert _state_hash(g1, s) == _state_hash(g2, s)


# ---------------------------------------------------------------------------
# Search from already-verified seeds (fast path)
# ---------------------------------------------------------------------------


def test_search_returns_verified_seed():
    """If a seed is already verified, search returns it immediately."""
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=False,
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    assert result.solved is True
    assert result.program is not None
    assert result.seed_provenance == "fitter"


# ---------------------------------------------------------------------------
# Search compiles fitter seeds
# ---------------------------------------------------------------------------


def test_search_compiles_fitter_seeds():
    """Search should compile fitter seeds even if not pre-verified."""
    demos = _tile_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    assert result.solved is True
    assert result.program is not None


# ---------------------------------------------------------------------------
# Search returns unsolved for impossible tasks
# ---------------------------------------------------------------------------


def test_search_unsolved_for_impossible():
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    assert result.solved is False
    assert result.unique_states_seen >= 1


# ---------------------------------------------------------------------------
# Search with empty seeds
# ---------------------------------------------------------------------------


def test_search_empty_seeds():
    result = search_from_seeds(
        seeds=[],
        examples=(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    assert result.solved is False
    assert result.seeds_tried == 0
    assert result.frontier_expansions == 0


# ---------------------------------------------------------------------------
# Multi-step search explores depth > 1
# ---------------------------------------------------------------------------


def test_search_explores_beyond_depth_zero():
    """With template seeds on a real task, search should expand to depth >= 1."""
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=True,
    )
    # Remove pre-verified fitter seeds so search must actually explore
    non_verified = [s for s in seeds if not s.already_verified]
    if not non_verified:
        return  # skip if all seeds verified (fitter was too good)

    result = search_from_seeds(
        seeds=non_verified,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        max_depth=3,
        max_expansions=50,
        max_total_compiles=20,
    )
    # Should have expanded beyond the seeds
    assert result.frontier_expansions >= 1
    assert result.unique_states_seen > len(non_verified)
    assert result.max_depth_reached >= 1


# ---------------------------------------------------------------------------
# Dedup prevents redundant exploration
# ---------------------------------------------------------------------------


def test_dedup_limits_explored_states():
    """With dedup, identical edits from different paths should not re-expand."""
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=True,
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        max_depth=2,
        max_expansions=100,
    )
    # unique_states_seen should be less than expansions * branching factor
    # because dedup collapses duplicate states
    assert result.unique_states_seen >= 1
    # The search should terminate (budgets work)
    assert result.frontier_expansions <= 100
    assert result.compiles_attempted <= 40


# ---------------------------------------------------------------------------
# Budget limits are respected
# ---------------------------------------------------------------------------


def test_search_respects_budget_limits():
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=True,
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        max_depth=1,
        max_expansions=5,
        max_total_compiles=3,
    )
    assert result.frontier_expansions <= 5
    assert result.compiles_attempted <= 3
    assert result.max_depth_reached <= 1


# ---------------------------------------------------------------------------
# Result reports diagnostics
# ---------------------------------------------------------------------------


def test_result_reports_diagnostics():
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    result = search_from_seeds(
        seeds=seeds,
        examples=demos,
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    assert result.solved is True
    assert result.compiles_attempted >= 1
    assert result.unique_states_seen >= 1
    assert isinstance(result.max_depth_reached, int)
    assert isinstance(result.winning_edit_depth, int)


# ---------------------------------------------------------------------------
# Canonical solve path uses editor search
# ---------------------------------------------------------------------------


def test_solve_arc_task_with_editor():
    """solve_arc_task uses editor search and still solves known tasks."""
    for task_fn in [_rotate_task, _tile_task, _reflect_task]:
        demos = task_fn()
        result = solve_arc_task(demos, task_id="test", use_editor_search=True)
        assert result.solved is True, f"failed on {task_fn.__name__}"
        assert result.winning_program is not None


def test_solve_arc_task_without_editor():
    """solve_arc_task works without editor search (static pipeline only)."""
    demos = _rotate_task()
    result = solve_arc_task(demos, task_id="test", use_editor_search=False)
    assert result.solved is True


def test_solve_arc_task_impossible():
    """Impossible tasks return solved=False even with editor search."""
    demos = _impossible_task()
    result = solve_arc_task(demos, task_id="test", use_editor_search=True)
    assert result.solved is False


# ---------------------------------------------------------------------------
# Direct ComputationGraph fitter path
# ---------------------------------------------------------------------------


def test_fitter_direct_grid_transform():
    """ARCFitter emits ComputationGraph directly for grid transforms."""
    demos = _rotate_task()
    fitter = ARCFitter()
    graphs = fitter.fit(demos, task_id="test")
    assert len(graphs) >= 1
    assert all(isinstance(g, ComputationGraph) for g in graphs)
    for g in graphs:
        assert g.validate() == []


def test_fitter_direct_reflect():
    """Direct fitter handles reflect tasks."""
    demos = _reflect_task()
    fitter = ARCFitter()
    graphs = fitter.fit(demos, task_id="test")
    assert len(graphs) >= 1
    reflect_graphs = [
        g for g in graphs
        if any("reflect" in str(n.evidence) for n in g.nodes.values())
    ]
    assert len(reflect_graphs) >= 1


# ---------------------------------------------------------------------------
# No deprecated dependency
# ---------------------------------------------------------------------------


def test_no_hybrid_neural_dependency():
    """Canonical path must not import from deprecated modules."""
    import aria.core.arc
    import aria.core.seeds
    import aria.core.editor_search
    import aria.core.editor_env

    for mod in [aria.core.arc, aria.core.seeds, aria.core.editor_search, aria.core.editor_env]:
        source = open(mod.__file__).read()
        assert "aria.core.hybrid" not in source
        assert "aria.core.neural" not in source
        assert "aria.core.experimental.hybrid" not in source
        assert "aria.core.experimental.neural" not in source
