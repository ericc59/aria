"""Tests for factored memory, retrieval, and trace modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aria.factored_memory import (
    FactoredMemoryStore,
    FactoredRecord,
    PerceptionKey,
    RepairPath,
    _compute_record_id,
)
from aria.factored_retrieval import (
    RetrievalGuidance,
    RetrievalMatch,
    aggregate_guidance,
    extract_factors_from_program,
    perception_key_from_demos,
    retrieve_factored,
)
from aria.factored_trace import (
    FactoredRetrievalTrace,
    format_factored_retrieval_trace,
)
from aria.types import Bind, Call, DemoPair, Expr, Literal, Program, Ref, Type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demo(in_rows: int = 3, in_cols: int = 3,
               out_rows: int = 3, out_cols: int = 3) -> DemoPair:
    """Create a simple demo pair with specified dimensions."""
    inp = np.zeros((in_rows, in_cols), dtype=np.uint8)
    out = np.zeros((out_rows, out_cols), dtype=np.uint8)
    # Add some non-trivial content
    inp[0, 0] = 1
    inp[1, 1] = 2
    out[0, 0] = 2
    out[1, 1] = 1
    return DemoPair(input=inp, output=out)


def _make_perception_key(**overrides) -> PerceptionKey:
    """Build a PerceptionKey with sensible defaults."""
    defaults = dict(
        dims_relation="same",
        palette_relation="same",
        change_type="dense",
        object_count_bucket="few",
        has_partition=False,
        has_frame=False,
        has_marker=False,
        has_legend=False,
        symmetry_tags=(),
        color_count_bucket="few",
        partition_cell_count=None,
    )
    defaults.update(overrides)
    return PerceptionKey(**defaults)


def _make_record(
    *,
    decomposition_type: str = "object",
    selector_family: str = "by_color",
    scope_family: str = "objects",
    op_family: str = "recolor",
    correspondence: str = "none",
    task_ids: tuple[str, ...] = ("task_1",),
    task_signatures: tuple[str, ...] = ("dims:same", "color:palette_same"),
    perception_key: PerceptionKey | None = None,
    repair_path: RepairPath | None = None,
) -> FactoredRecord:
    """Build a FactoredRecord with sensible defaults."""
    pk = perception_key or _make_perception_key()
    record_id = _compute_record_id(
        decomposition_type, selector_family, scope_family,
        op_family, correspondence, 2, pk,
    )
    return FactoredRecord(
        record_id=record_id,
        task_ids=task_ids,
        source="test",
        decomposition_type=decomposition_type,
        selector_family=selector_family,
        scope_family=scope_family,
        op_family=op_family,
        correspondence=correspondence,
        composition_depth=2,
        perception_key=pk,
        task_signatures=task_signatures,
        verified=True,
        verify_mode="stateless",
        repair_path=repair_path,
    )


# ---------------------------------------------------------------------------
# PerceptionKey tests
# ---------------------------------------------------------------------------


class TestPerceptionKey:
    def test_field_overlap_identical(self):
        k = _make_perception_key()
        assert k.field_overlap(k) == 10  # 10 scalar fields match

    def test_field_overlap_partial(self):
        k1 = _make_perception_key(dims_relation="same", change_type="additive")
        k2 = _make_perception_key(dims_relation="same", change_type="dense")
        overlap = k1.field_overlap(k2)
        # dims_relation matches, change_type doesn't
        assert overlap < k1.field_overlap(k1)

    def test_field_overlap_symmetry_tags(self):
        k1 = _make_perception_key(symmetry_tags=("sym:input_reflective", "sym:output_periodic"))
        k2 = _make_perception_key(symmetry_tags=("sym:input_reflective",))
        overlap_with = k1.field_overlap(k2)
        k3 = _make_perception_key(symmetry_tags=())
        overlap_without = k1.field_overlap(k3)
        assert overlap_with > overlap_without

    def test_round_trip_dict(self):
        k = _make_perception_key(
            has_partition=True,
            symmetry_tags=("sym:input_reflective",),
            partition_cell_count=4,
        )
        d = k.to_dict()
        k2 = PerceptionKey.from_dict(d)
        assert k == k2

    def test_from_demos(self):
        demos = (_make_demo(),)
        key = perception_key_from_demos(demos)
        assert isinstance(key, PerceptionKey)
        assert key.dims_relation == "same"

    def test_from_demos_different_dims(self):
        demos = (_make_demo(in_rows=3, in_cols=3, out_rows=6, out_cols=6),)
        key = perception_key_from_demos(demos)
        assert key.dims_relation in ("grow", "reshape")

    def test_from_empty_demos(self):
        key = perception_key_from_demos(())
        assert key.dims_relation == "same"


# ---------------------------------------------------------------------------
# RepairPath tests
# ---------------------------------------------------------------------------


class TestRepairPath:
    def test_round_trip(self):
        rp = RepairPath(
            initial_error_type="pixel_mismatch",
            repair_kind="beam_mutation",
            mutations_applied=("swap_color", "add_overlay"),
            rounds_to_solve=2,
            initial_pixel_diff=5,
        )
        d = rp.to_dict()
        rp2 = RepairPath.from_dict(d)
        assert rp == rp2


# ---------------------------------------------------------------------------
# FactoredRecord tests
# ---------------------------------------------------------------------------


class TestFactoredRecord:
    def test_factor_tuple(self):
        r = _make_record()
        ft = r.factor_tuple()
        assert ft == ("object", "by_color", "objects", "recolor", "none")

    def test_distinct_task_count(self):
        r = _make_record(task_ids=("a", "b", "c"))
        assert r.distinct_task_count == 3

    def test_round_trip_dict(self):
        rp = RepairPath("pixel_mismatch", "beam_mutation", ("swap",), 1, 3)
        r = _make_record(repair_path=rp)
        d = r.to_dict()
        r2 = FactoredRecord.from_dict(d)
        assert r2.record_id == r.record_id
        assert r2.repair_path == rp
        assert r2.factor_tuple() == r.factor_tuple()


# ---------------------------------------------------------------------------
# FactoredMemoryStore tests
# ---------------------------------------------------------------------------


class TestFactoredMemoryStore:
    def test_add_and_len(self):
        store = FactoredMemoryStore()
        r = _make_record()
        store.add_record(r)
        assert len(store) == 1

    def test_dedup_merges_task_ids(self):
        store = FactoredMemoryStore()
        r1 = _make_record(task_ids=("a",))
        r2 = FactoredRecord(
            record_id=r1.record_id,
            task_ids=("b",),
            source=r1.source,
            decomposition_type=r1.decomposition_type,
            selector_family=r1.selector_family,
            scope_family=r1.scope_family,
            op_family=r1.op_family,
            correspondence=r1.correspondence,
            composition_depth=r1.composition_depth,
            perception_key=r1.perception_key,
            task_signatures=r1.task_signatures,
            verified=True,
            verify_mode="stateless",
            repair_path=None,
        )
        store.add_record(r1)
        merged = store.add_record(r2)
        assert len(store) == 1
        assert set(merged.task_ids) == {"a", "b"}

    def test_save_load_round_trip(self):
        store = FactoredMemoryStore()
        store.add_record(_make_record(task_ids=("x",)))
        store.add_record(_make_record(
            op_family="fill",
            task_ids=("y",),
        ))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            store.save_json(path)

            loaded = FactoredMemoryStore.load_json(path)
            assert len(loaded) == 2
            for r in loaded.all_records():
                assert r.verified

    def test_load_nonexistent(self):
        store = FactoredMemoryStore.load_json("/nonexistent/path.json")
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Retrieval scoring tests
# ---------------------------------------------------------------------------


class TestRetrieveFactored:
    def test_empty_store_returns_empty(self):
        demos = (_make_demo(),)
        store = FactoredMemoryStore()
        matches = retrieve_factored(demos, store)
        assert matches == []

    def test_higher_overlap_ranks_first(self):
        demos = (_make_demo(),)  # same dims
        store = FactoredMemoryStore()

        # Record with matching dims_relation
        pk_match = _make_perception_key(dims_relation="same")
        r_good = _make_record(perception_key=pk_match, task_ids=("a",))

        # Record with non-matching dims_relation
        pk_mismatch = _make_perception_key(dims_relation="grow")
        r_bad = _make_record(
            perception_key=pk_mismatch,
            op_family="scale",
            task_ids=("b",),
        )

        store.add_record(r_good)
        store.add_record(r_bad)

        matches = retrieve_factored(demos, store)
        assert len(matches) >= 1
        # The matching record should score higher
        assert matches[0].record.record_id == r_good.record_id

    def test_transfer_strength_bonus(self):
        store = FactoredMemoryStore()
        pk = _make_perception_key()

        r_single = _make_record(task_ids=("a",), perception_key=pk)
        r_multi = _make_record(
            task_ids=("a", "b", "c"),
            perception_key=pk,
            op_family="fill",  # different so record_id differs
        )

        store.add_record(r_single)
        store.add_record(r_multi)

        demos = (_make_demo(),)
        matches = retrieve_factored(demos, store)
        assert len(matches) == 2
        # Multi-task should score higher due to transfer bonus
        assert matches[0].record.distinct_task_count > matches[1].record.distinct_task_count

    def test_max_results(self):
        store = FactoredMemoryStore()
        for i in range(30):
            store.add_record(_make_record(
                task_ids=(f"t{i}",),
                op_family=f"op_{i}",
            ))

        demos = (_make_demo(),)
        matches = retrieve_factored(demos, store, max_results=5)
        assert len(matches) <= 5


# ---------------------------------------------------------------------------
# Guidance aggregation tests
# ---------------------------------------------------------------------------


class TestAggregateGuidance:
    def test_empty(self):
        g = aggregate_guidance([])
        assert g.preferred_op_families == ()
        assert g.source_matches == ()

    def test_single_match(self):
        r = _make_record(op_family="recolor", selector_family="by_color")
        m = RetrievalMatch(record=r, score=10.0, match_reasons=("test",))
        g = aggregate_guidance([m])
        assert "recolor" in g.preferred_op_families
        assert "by_color" in g.preferred_selector_families

    def test_vote_weighting(self):
        r1 = _make_record(op_family="recolor", task_ids=("a",))
        r2 = _make_record(op_family="fill", task_ids=("b",))
        m1 = RetrievalMatch(record=r1, score=20.0, match_reasons=())
        m2 = RetrievalMatch(record=r2, score=5.0, match_reasons=())
        g = aggregate_guidance([m1, m2])
        # Higher-scored match should rank first
        assert g.preferred_op_families[0] == "recolor"

    def test_repair_paths_collected(self):
        rp = RepairPath("pixel_mismatch", "beam_mutation", ("swap",), 1, 3)
        r = _make_record(repair_path=rp)
        m = RetrievalMatch(record=r, score=10.0, match_reasons=())
        g = aggregate_guidance([m])
        assert len(g.candidate_repair_paths) == 1
        assert g.candidate_repair_paths[0].repair_kind == "beam_mutation"

    def test_preferred_ops_as_set(self):
        r = _make_record(op_family="recolor")
        m = RetrievalMatch(record=r, score=10.0, match_reasons=())
        g = aggregate_guidance([m])
        ops = g.preferred_ops_as_set()
        assert "recolor" in ops


# ---------------------------------------------------------------------------
# Factor extraction tests
# ---------------------------------------------------------------------------


class TestExtractFactors:
    def test_simple_recolor_program(self):
        program = Program(
            steps=(
                Bind("objs", Type.OBJECT_SET, Call("find_objects", (Ref("input"),))),
                Bind("obj", Type.OBJECT, Call("by_color", (Ref("objs"), Literal(1, Type.COLOR)))),
                Bind("result", Type.GRID, Call("recolor", (Ref("input"), Ref("obj"), Literal(2, Type.COLOR)))),
            ),
            output="result",
        )
        factors = extract_factors_from_program(program)
        assert factors["op_family"] == "recolor"
        assert factors["selector_family"] == "by_color"
        assert factors["scope_family"] == "objects"
        assert factors["decomposition_type"] == "object"

    def test_scale_program(self):
        program = Program(
            steps=(
                Bind("result", Type.GRID, Call("upscale_grid", (Ref("input"), Literal(2, Type.INT)))),
            ),
            output="result",
        )
        factors = extract_factors_from_program(program)
        assert factors["op_family"] == "scale"
        assert factors["decomposition_type"] == "canvas"

    def test_unknown_ops(self):
        program = Program(
            steps=(
                Bind("result", Type.GRID, Call("custom_op", (Ref("input"),))),
            ),
            output="result",
        )
        factors = extract_factors_from_program(program)
        assert factors["op_family"] == "unknown"


# ---------------------------------------------------------------------------
# Ingest solve record tests
# ---------------------------------------------------------------------------


class TestIngestSolveRecord:
    def test_basic_ingest(self):
        store = FactoredMemoryStore()
        demos = (_make_demo(),)
        program = Program(
            steps=(
                Bind("result", Type.GRID, Call("recolor", (Ref("input"), Literal(1, Type.COLOR), Literal(2, Type.COLOR)))),
            ),
            output="result",
        )
        from aria.factored_retrieval import ingest_solve_record
        record = ingest_solve_record(
            demos, program,
            task_id="task_001",
            source="test",
            factored_store=store,
        )
        assert len(store) == 1
        assert "task_001" in record.task_ids
        assert record.verified
        assert isinstance(record.perception_key, PerceptionKey)


# ---------------------------------------------------------------------------
# Trace tests
# ---------------------------------------------------------------------------


class TestFactoredRetrievalTrace:
    def test_record_matches(self):
        trace = FactoredRetrievalTrace()
        r = _make_record()
        m = RetrievalMatch(record=r, score=15.0, match_reasons=("perception:5_fields",))
        trace.record_matches([m])
        assert trace.matches_returned == 1
        assert trace.top_match_score == 15.0
        assert len(trace.entries) == 1

    def test_record_guidance(self):
        trace = FactoredRetrievalTrace()
        g = RetrievalGuidance(
            preferred_decomposition_types=("object",),
            preferred_selector_families=("by_color",),
            preferred_scope_families=("objects",),
            preferred_op_families=("recolor",),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )
        trace.record_guidance(g)
        assert "decomp:object" in trace.factors_borrowed
        assert "op:recolor" in trace.factors_borrowed

    def test_summary(self):
        trace = FactoredRetrievalTrace()
        r = _make_record()
        m = RetrievalMatch(record=r, score=10.0, match_reasons=())
        trace.record_matches([m])
        s = trace.summary()
        assert s["matches_returned"] == 1

    def test_format(self):
        trace = FactoredRetrievalTrace()
        r = _make_record()
        m = RetrievalMatch(record=r, score=10.0, match_reasons=("perception:5_fields",))
        trace.record_matches([m])
        text = format_factored_retrieval_trace(trace)
        assert "1 matches" in text
        assert "10.0" in text

    def test_per_mechanism_tracking(self):
        trace = FactoredRetrievalTrace()
        rp = RepairPath("pixel_mismatch", "beam_mutation", ("swap_color",), 1, 3)
        g = RetrievalGuidance(
            preferred_decomposition_types=("object",),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=("recolor", "fill"),
            preferred_correspondences=(),
            candidate_repair_paths=(rp,),
            source_matches=(),
        )
        trace.record_guidance(g)
        assert trace.decomp_bias_active
        assert trace.op_bias_count == 2
        assert trace.repair_bias_active
        assert trace.repair_mutations_suggested == 1
        s = trace.summary()
        assert s["decomp_bias_active"] is True
        assert s["repair_bias_active"] is True


# ---------------------------------------------------------------------------
# Decomp ranker integration tests
# ---------------------------------------------------------------------------


class TestDecompRanker:
    def test_make_decomp_ranker_none_when_empty(self):
        g = RetrievalGuidance(
            preferred_decomposition_types=(),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )
        assert g.make_decomp_ranker() is None

    def test_make_decomp_ranker_reorders(self):
        g = RetrievalGuidance(
            preferred_decomposition_types=("framed",),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )
        ranker = g.make_decomp_ranker()
        assert ranker is not None
        # framed maps to "framed_regions" view, should be boosted
        views = ["composites", "framed_regions"]
        order = ranker(None, views)
        assert order[0] == 1  # framed_regions index

    def test_make_decomp_ranker_object_maps_to_composites(self):
        g = RetrievalGuidance(
            preferred_decomposition_types=("object",),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )
        ranker = g.make_decomp_ranker()
        assert ranker is not None
        views = ["framed_regions", "composites"]
        order = ranker(None, views)
        assert order[0] == 1  # composites index


# ---------------------------------------------------------------------------
# Mutation priority tests
# ---------------------------------------------------------------------------


class TestMutationPriority:
    def test_empty_repair_paths(self):
        g = RetrievalGuidance(
            preferred_decomposition_types=(),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(),
            source_matches=(),
        )
        assert g.suggested_mutation_priority() == ()

    def test_beam_mutation_suggests_replace_ops(self):
        rp = RepairPath("pixel_mismatch", "beam_mutation", ("swap_color",), 1, 3)
        g = RetrievalGuidance(
            preferred_decomposition_types=(),
            preferred_selector_families=(),
            preferred_scope_families=(),
            preferred_op_families=(),
            preferred_correspondences=(),
            candidate_repair_paths=(rp,),
            source_matches=(),
        )
        priority = g.suggested_mutation_priority()
        assert len(priority) > 0
        # beam_mutation should suggest replace_op and replace_literal
        assert "replace_op" in priority or "swap_color" in priority
