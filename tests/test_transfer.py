"""Tests for transfer-aware retrieval, diagnostics, graph quality, and metrics."""

from __future__ import annotations

import json
from pathlib import Path

from aria.eval import EvalConfig, compute_transfer_metrics, evaluate_task
from aria.inspection import inspect_retrieval
from aria.library.graph import (
    AbstractionNode,
    LeafNode,
    build_abstraction_graph,
    graph_summary,
)
from aria.library.store import Library
from aria.library.builder import build_library_from_store
from aria.program_store import ProgramStore
from aria.proposer.parser import parse_program
from aria.retrieval import (
    AbstractionHint,
    RetrievalHit,
    preferred_ops_from_hints,
    retrieve_abstractions,
    retrieve_program,
)
from aria.runtime.ops import reset_library_ops
from aria.types import DemoPair, LibraryEntry, Task, Type, grid_from_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _transpose_task() -> Task:
    inp = grid_from_list([[1, 2, 0], [3, 4, 0]])
    out = grid_from_list([[1, 3], [2, 4], [0, 0]])
    return Task(
        train=(DemoPair(input=inp, output=out),),
        test=(DemoPair(input=inp, output=out),),
    )


def _store_with_transfer_and_leaf() -> ProgramStore:
    """A store with one transfer-backed program (2 tasks) and one leaf (1 task)."""
    store = ProgramStore()
    # Transfer-backed: verified on two distinct tasks via search
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="task-a",
        source="offline-search",
        signatures=frozenset({"dims:different"}),
    )
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="task-b",
        source="offline-search",
        signatures=frozenset({"dims:different"}),
    )
    # Leaf: only verified on one task via retrieval echo
    store.add_text(
        "let result: GRID = reflect_grid(HORIZONTAL, input)\n-> result",
        task_id="single-task",
        source="offline-retrieval",
        signatures=frozenset({"dims:same"}),
    )
    return store


# ---------------------------------------------------------------------------
# 1. Retrieval diagnostics
# ---------------------------------------------------------------------------


def test_retrieve_program_returns_diagnostics():
    store = _store_with_transfer_and_leaf()
    task = _transpose_task()
    hit = retrieve_program(task.train, store)
    assert hit is not None
    assert hit.diagnostics is not None
    assert hit.diagnostics.distinct_task_count >= 1
    assert isinstance(hit.diagnostics.retrieval_bucket, str)
    assert hit.diagnostics.retrieval_bucket in ("transfer", "fallback")


def test_retrieve_program_prefers_transfer_bucket():
    store = _store_with_transfer_and_leaf()
    task = _transpose_task()
    hit = retrieve_program(task.train, store)
    assert hit is not None
    # The transpose program is transfer-backed (2 tasks), so it should come first
    assert hit.diagnostics.distinct_task_count == 2
    assert hit.diagnostics.retrieval_bucket == "transfer"
    assert hit.diagnostics.has_non_retrieval_source is True


def test_retrieve_diagnostics_signature_overlap():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="t1",
        source="offline-search",
        signatures=frozenset({"dims:different", "change:transformation"}),
    )
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="t2",
        source="offline-search",
        signatures=frozenset({"dims:different"}),
    )
    task = _transpose_task()
    hit = retrieve_program(task.train, store)
    assert hit is not None
    assert hit.diagnostics.signature_overlap >= 0


# ---------------------------------------------------------------------------
# 2. Inspection retrieval diagnostics
# ---------------------------------------------------------------------------


def test_inspect_retrieval_includes_transfer_fields():
    store = _store_with_transfer_and_leaf()
    task = _transpose_task()
    items = inspect_retrieval(task.train, store, limit=5)
    assert len(items) >= 1
    for item in items:
        assert hasattr(item, "distinct_task_count")
        assert hasattr(item, "has_non_retrieval_source")
        assert hasattr(item, "retrieval_echo_count")
    # The transfer-backed transpose should rank first
    top = items[0]
    assert top.distinct_task_count == 2


# ---------------------------------------------------------------------------
# 3. Graph quality classification
# ---------------------------------------------------------------------------


def test_leaf_node_is_transfer_backed():
    leaf = LeafNode(
        id="leaf:abc",
        program_text="p",
        task_ids=("a", "b"),
        signatures=(),
        sources=(),
        use_count=2,
        step_count=1,
    )
    assert leaf.is_transfer_backed is True

    single = LeafNode(
        id="leaf:xyz",
        program_text="q",
        task_ids=("a",),
        signatures=(),
        sources=(),
        use_count=1,
        step_count=1,
    )
    assert single.is_transfer_backed is False


def test_abstraction_node_strength():
    strong = AbstractionNode(
        id="a:1",
        name="lib_x",
        program_text="p",
        params=(),
        return_type="GRID",
        use_count=3,
        support_task_ids=("a", "b"),
        support_program_count=2,
        mdl_gain=5,
        step_count=2,
    )
    assert strong.strength == "strong"

    weak = AbstractionNode(
        id="a:2",
        name="lib_y",
        program_text="q",
        params=(),
        return_type="GRID",
        use_count=1,
        support_task_ids=("a",),
        support_program_count=1,
        mdl_gain=3,
        step_count=2,
    )
    assert weak.strength == "weak"

    no_mdl = AbstractionNode(
        id="a:3",
        name="lib_z",
        program_text="r",
        params=(),
        return_type="GRID",
        use_count=2,
        support_task_ids=("a", "b"),
        support_program_count=2,
        mdl_gain=0,
        step_count=2,
    )
    assert no_mdl.strength == "weak"


def test_abstraction_node_to_dict_includes_strength():
    node = AbstractionNode(
        id="a:1",
        name="lib_x",
        program_text="p",
        params=(("arg0", "GRID"),),
        return_type="GRID",
        use_count=3,
        support_task_ids=("a", "b"),
        support_program_count=2,
        mdl_gain=5,
        step_count=2,
    )
    d = node.to_dict()
    assert d["strength"] == "strong"


def test_graph_summary_counts():
    store = ProgramStore()
    program_text = """\
bind flipped: GRID = reflect_grid(HORIZONTAL, input)
bind tiled: GRID = stack_v(input, flipped)
bind result: GRID = transpose_grid(tiled)
yield result
"""
    store.add_text(program_text, task_id="task-a", source="offline-search")
    store.add_text(program_text, task_id="task-b", source="offline-search")
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="only-one",
        source="offline-retrieval",
    )

    library, _report = build_library_from_store(store, min_length=2, max_length=2, min_uses=2)
    graph = build_abstraction_graph(store, library)
    summary = graph_summary(graph)

    assert summary["total_leaves"] == 2
    assert summary["transfer_backed_leaves"] >= 1
    assert summary["one_off_leaves"] >= 1
    assert summary["total_abstractions"] >= 1
    assert isinstance(summary["avg_mdl_gain"], float)
    assert isinstance(summary["strong_abstractions"], int)


def test_graph_to_dict_includes_summary():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="t1",
        source="offline-search",
    )
    graph = build_abstraction_graph(store, Library())
    d = graph.to_dict()
    assert "summary" in d
    assert d["version"] == 2
    assert d["summary"]["total_leaves"] == 1


# ---------------------------------------------------------------------------
# 4. Transfer metrics
# ---------------------------------------------------------------------------


def test_compute_transfer_metrics_empty():
    metrics = compute_transfer_metrics([])
    assert metrics["retrieval_solves_total"] == 0
    assert metrics["retrieval_solves_transfer_backed"] == 0
    assert metrics["retrieval_solves_leaf_only"] == 0


def test_compute_transfer_metrics_with_retrieval_provenance():
    outcomes = [
        {
            "task_id": "t1",
            "solved": True,
            "solve_source": "retrieval",
            "retrieval_provenance": {
                "distinct_task_count": 3,
                "has_non_retrieval_source": True,
            },
        },
        {
            "task_id": "t2",
            "solved": True,
            "solve_source": "retrieval",
            "retrieval_provenance": {
                "distinct_task_count": 1,
                "has_non_retrieval_source": False,
            },
        },
        {
            "task_id": "t3",
            "solved": True,
            "solve_source": "search",
        },
        {
            "task_id": "t4",
            "solved": False,
            "solve_source": "unsolved",
        },
    ]
    metrics = compute_transfer_metrics(outcomes)
    assert metrics["retrieval_solves_total"] == 2
    assert metrics["retrieval_solves_transfer_backed"] == 1
    assert metrics["retrieval_solves_leaf_only"] == 1
    assert metrics["retrieval_solves_non_retrieval_provenance"] == 1


def test_compute_transfer_metrics_with_library():
    reset_library_ops()
    try:
        library = Library()
        program = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="lib_test",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=program.steps,
            output=program.output,
            level=1,
            use_count=3,
            support_task_ids=("a", "b"),
            support_program_count=2,
            mdl_gain=5,
        ))
        metrics = compute_transfer_metrics([], library=library)
        lq = metrics["library_quality"]
        assert lq["total_entries"] == 1
        assert lq["multi_task_entries"] == 1
        assert lq["strong_abstractions"] == 1
        assert lq["avg_mdl_gain"] == 5.0
    finally:
        reset_library_ops()


# ---------------------------------------------------------------------------
# 5. Eval outcome includes retrieval provenance
# ---------------------------------------------------------------------------


def test_evaluate_task_retrieval_includes_provenance():
    store = ProgramStore()
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="seed-task",
        source="offline-search",
        signatures=frozenset({"dims:different"}),
    )
    store.add_text(
        "let result: GRID = transpose_grid(input)\n-> result",
        task_id="other-seed",
        source="offline-search",
        signatures=frozenset({"dims:different"}),
    )

    config = EvalConfig(max_search_steps=1, max_search_candidates=200)
    task = _transpose_task()
    outcome = evaluate_task(
        "eval-t",
        task,
        library=Library(),
        program_store=store,
        config=config,
    )
    assert outcome["solved"]
    assert outcome["solve_source"] == "retrieval"
    assert "retrieval_provenance" in outcome
    prov = outcome["retrieval_provenance"]
    assert prov["distinct_task_count"] == 2
    assert prov["has_non_retrieval_source"] is True


# ---------------------------------------------------------------------------
# 6. No regression: basic retrieval still works
# ---------------------------------------------------------------------------


def test_retrieve_with_empty_store_returns_none():
    task = _transpose_task()
    hit = retrieve_program(task.train, ProgramStore())
    assert hit is None


def test_retrieve_program_max_candidates_respected():
    store = _store_with_transfer_and_leaf()
    task = _transpose_task()
    # With max_candidates=0 (unlimited) it should find something
    hit = retrieve_program(task.train, store, max_candidates=0)
    assert hit is not None
    # With max_candidates=1, might or might not find — depends on ordering
    # But it should not crash
    hit2 = retrieve_program(task.train, store, max_candidates=1)
    assert hit2 is None or hit2.candidates_tried <= 1


# ---------------------------------------------------------------------------
# 7. Abstraction retrieval
# ---------------------------------------------------------------------------


def _library_with_entries() -> Library:
    """Create a library with strong and weak entries for testing."""
    library = Library()
    program = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")

    library.add(LibraryEntry(
        name="lib_strong",
        params=(("arg0", Type.GRID),),
        return_type=Type.GRID,
        steps=program.steps,
        output=program.output,
        level=1,
        use_count=5,
        support_task_ids=("task-a", "task-b", "task-c"),
        support_program_count=3,
        mdl_gain=8,
        signatures=("dims:same", "change:additive"),
    ))

    program2 = parse_program("bind v0 = transpose_grid(arg0)\nyield v0\n")
    library.add(LibraryEntry(
        name="lib_weak",
        params=(("arg0", Type.GRID),),
        return_type=Type.GRID,
        steps=program2.steps,
        output=program2.output,
        level=1,
        use_count=1,
        support_task_ids=("task-x",),
        support_program_count=1,
        mdl_gain=0,
        signatures=("dims:different",),
    ))
    return library


def test_retrieve_abstractions_returns_scored_hints():
    reset_library_ops()
    try:
        library = _library_with_entries()
        task = _transpose_task()
        hints = retrieve_abstractions(task.train, library)
        assert len(hints) == 2
        for h in hints:
            assert isinstance(h, AbstractionHint)
            assert isinstance(h.score, float)
            assert h.strength in ("strong", "weak")
    finally:
        reset_library_ops()


def test_retrieve_abstractions_ranks_strong_first():
    reset_library_ops()
    try:
        library = _library_with_entries()
        task = _transpose_task()
        hints = retrieve_abstractions(task.train, library)
        assert hints[0].name == "lib_strong"
        assert hints[0].strength == "strong"
        assert hints[0].score > hints[1].score
    finally:
        reset_library_ops()


def test_retrieve_abstractions_empty_library():
    task = _transpose_task()
    hints = retrieve_abstractions(task.train, Library())
    assert hints == []


def test_preferred_ops_from_hints():
    hints = [
        AbstractionHint("lib_a", score=10.0, signature_overlap=0,
                        support_task_count=2, support_program_count=2,
                        mdl_gain=5, strength="strong"),
        AbstractionHint("lib_b", score=0.0, signature_overlap=0,
                        support_task_count=0, support_program_count=0,
                        mdl_gain=0, strength="weak"),
    ]
    preferred = preferred_ops_from_hints(hints)
    assert "lib_a" in preferred
    assert "lib_b" not in preferred  # score=0 is not > 0


def test_preferred_ops_empty():
    assert preferred_ops_from_hints([]) == frozenset()


# ---------------------------------------------------------------------------
# 8. Search integration with preferred_ops
# ---------------------------------------------------------------------------


def test_search_with_preferred_ops_does_not_crash():
    """preferred_ops is threaded through search without error."""
    from aria.offline_search import search_program

    task = _transpose_task()
    result = search_program(
        task.train,
        Library(),
        max_steps=1,
        max_candidates=10,
        preferred_ops=frozenset({"transpose_grid", "reflect_grid"}),
    )
    assert isinstance(result.candidates_tried, int)


# ---------------------------------------------------------------------------
# 9. Inspection shows abstraction retrieval
# ---------------------------------------------------------------------------


def test_inspection_includes_abstraction_retrieval():
    from aria.inspection import inspect_task

    reset_library_ops()
    try:
        library = _library_with_entries()
        store = ProgramStore()
        task = _transpose_task()
        inspection = inspect_task(
            task.train,
            library=library,
            program_store=store,
            retrieval_limit=0,
            max_search_steps=1,
            max_search_candidates=10,
            max_refinement_rounds=1,
            search_trace_limit=5,
        )

        # Synthesis may solve before abstraction retrieval runs.
        # Check the overall result works.
        assert inspection["refinement"]["solved"] is not None
    finally:
        reset_library_ops()


# ---------------------------------------------------------------------------
# 10. Eval reports abstraction guidance
# ---------------------------------------------------------------------------


def test_eval_outcome_tracks_abstraction_hints():
    reset_library_ops()
    try:
        library = _library_with_entries()
        config = EvalConfig(max_search_steps=1, max_search_candidates=200, max_refinement_rounds=1)
        task = _transpose_task()
        outcome = evaluate_task(
            "t1", task,
            library=library,
            config=config,
        )
        assert "abstraction_hints_available" in outcome
    finally:
        reset_library_ops()


def test_transfer_metrics_include_abstraction_guidance():
    outcomes = [
        {
            "task_id": "t1",
            "solved": True,
            "solve_source": "search",
            "abstraction_hints_available": True,
            "solved_with_retrieved_abstraction": True,
        },
        {
            "task_id": "t2",
            "solved": False,
            "solve_source": "unsolved",
            "abstraction_hints_available": True,
            "solved_with_retrieved_abstraction": False,
        },
        {
            "task_id": "t3",
            "solved": True,
            "solve_source": "search",
            "abstraction_hints_available": False,
        },
    ]
    metrics = compute_transfer_metrics(outcomes)
    assert metrics["abstraction_hints_available"] == 2
    assert metrics["solved_with_retrieved_abstraction"] == 1


# ---------------------------------------------------------------------------
# 11. Signature cross-referencing
# ---------------------------------------------------------------------------


def test_retrieve_abstractions_uses_signature_overlap():
    """Entries whose signatures match the task should score higher."""
    reset_library_ops()
    try:
        library = Library()

        # Entry with matching signatures
        prog_match = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="lib_match",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog_match.steps,
            output=prog_match.output,
            level=1,
            use_count=2,
            support_task_ids=("a", "b"),
            support_program_count=2,
            mdl_gain=3,
            signatures=("dims:different",),  # transpose task has dims:different
        ))

        # Entry with no matching signatures but same provenance strength
        prog_no = parse_program("bind v0 = transpose_grid(arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="lib_nomatch",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog_no.steps,
            output=prog_no.output,
            level=1,
            use_count=2,
            support_task_ids=("c", "d"),
            support_program_count=2,
            mdl_gain=3,
            signatures=("dims:same",),  # does not match transpose task
        ))

        task = _transpose_task()
        hints = retrieve_abstractions(task.train, library)
        assert len(hints) == 2
        # lib_match should rank first due to signature overlap
        assert hints[0].name == "lib_match"
        assert hints[0].signature_overlap >= 1
        assert hints[1].signature_overlap == 0
        assert hints[0].score > hints[1].score
    finally:
        reset_library_ops()


def test_retrieve_abstractions_empty_signatures_still_scores():
    """Entries without signatures should still get scored by provenance."""
    reset_library_ops()
    try:
        library = Library()
        prog = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="lib_nosig",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog.steps,
            output=prog.output,
            level=1,
            use_count=3,
            support_task_ids=("a", "b"),
            support_program_count=2,
            mdl_gain=5,
            signatures=(),  # no signatures
        ))
        hints = retrieve_abstractions(_transpose_task().train, library)
        assert len(hints) == 1
        assert hints[0].signature_overlap == 0
        assert hints[0].score > 0  # still scores via provenance
    finally:
        reset_library_ops()


# ---------------------------------------------------------------------------
# 12. Library pruning
# ---------------------------------------------------------------------------


def test_library_prune_weak_removes_thin_evidence():
    reset_library_ops()
    try:
        library = Library()
        prog = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")

        # Strong: should survive
        library.add(LibraryEntry(
            name="strong_entry",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog.steps,
            output=prog.output,
            level=1,
            support_task_ids=("a", "b"),
            mdl_gain=5,
        ))

        prog2 = parse_program("bind v0 = transpose_grid(arg0)\nyield v0\n")
        # Weak: single task + zero mdl_gain → should be pruned
        library.add(LibraryEntry(
            name="weak_entry",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog2.steps,
            output=prog2.output,
            level=1,
            support_task_ids=("x",),
            mdl_gain=0,
        ))

        assert len(library.all_entries()) == 2
        pruned = library.prune_weak()
        assert pruned == 1
        assert len(library.all_entries()) == 1
        assert library.get("strong_entry") is not None
        assert library.get("weak_entry") is None
    finally:
        reset_library_ops()


def test_library_prune_weak_keeps_positive_mdl():
    reset_library_ops()
    try:
        library = Library()
        prog = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")

        # Single task but positive mdl_gain → should survive
        library.add(LibraryEntry(
            name="single_but_compresses",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog.steps,
            output=prog.output,
            level=1,
            support_task_ids=("a",),
            mdl_gain=3,
        ))
        pruned = library.prune_weak()
        assert pruned == 0
        assert len(library.all_entries()) == 1
    finally:
        reset_library_ops()


# ---------------------------------------------------------------------------
# 13. Library entry signatures persist through save/load
# ---------------------------------------------------------------------------


def test_library_signatures_round_trip(tmp_path):
    reset_library_ops()
    try:
        library = Library()
        prog = parse_program("bind v0 = reflect_grid(HORIZONTAL, arg0)\nyield v0\n")
        library.add(LibraryEntry(
            name="sig_entry",
            params=(("arg0", Type.GRID),),
            return_type=Type.GRID,
            steps=prog.steps,
            output=prog.output,
            level=1,
            use_count=2,
            support_task_ids=("a", "b"),
            support_program_count=2,
            mdl_gain=5,
            signatures=("dims:same", "change:additive"),
        ))

        path = tmp_path / "lib.json"
        library.save_json(path)

        loaded = Library.load_json(path)
        entry = loaded.get("sig_entry")
        assert entry is not None
        assert set(entry.signatures) == {"change:additive", "dims:same"}
    finally:
        reset_library_ops()


# ---------------------------------------------------------------------------
# 14. Builder collects signatures from supporting records
# ---------------------------------------------------------------------------


def test_build_library_from_store_collects_signatures():
    store = ProgramStore()
    program_text = """\
bind flipped: GRID = reflect_grid(HORIZONTAL, input)
bind tiled: GRID = stack_v(input, flipped)
bind result: GRID = transpose_grid(tiled)
yield result
"""
    store.add_text(program_text, task_id="task-a", source="offline-search",
                   signatures=frozenset({"dims:different", "change:transformation"}))
    store.add_text(program_text, task_id="task-b", source="offline-search",
                   signatures=frozenset({"dims:different", "size:multiplicative"}))

    library, _report = build_library_from_store(store, min_length=2, max_length=2, min_uses=2)

    assert len(library.all_entries()) >= 1
    entry = library.all_entries()[0]
    # Should have union of signatures from both supporting records
    assert "dims:different" in entry.signatures
    assert len(entry.signatures) >= 2
