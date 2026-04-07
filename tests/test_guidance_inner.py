"""Tests for inner-loop search traces, metrics, and baselines.

Covers:
- Graph-edit trace schema roundtrip
- Parameter-trial trace schema roundtrip
- Export integration with inner_trace
- Inner-loop metrics
- Lightweight baselines
- Failure category categorization
- No task-id logic
- No regressions
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


# ---------------------------------------------------------------------------
# Phase 1: Graph-edit trace schema tests
# ---------------------------------------------------------------------------


class TestGraphEditTraceSchema:
    def test_candidate_roundtrip(self):
        from aria.core.guidance_inner_traces import GraphEditCandidate, FailureCategory

        c = GraphEditCandidate(
            edit_id="edit_0_1", action_type="SET_SLOT", node_id="n0",
            key="transform", value_summary="rotate 90",
            parent_id="seed_0", depth=1, priority=-500.0,
            attempted=True, compiled=True, verified=False,
            failure_category=FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
            diff_pixels_before=100, diff_pixels_after=80,
            residual_fraction=0.8, score_before=5.0, score_after=4.5,
        )
        d = c.to_dict()
        c2 = GraphEditCandidate.from_dict(d)
        assert c2.edit_id == c.edit_id
        assert c2.failure_category == c.failure_category
        assert c2.depth == c.depth

    def test_episode_roundtrip(self):
        from aria.core.guidance_inner_traces import (
            GraphEditCandidate, GraphEditEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        c = GraphEditCandidate(
            "seed_0", "SEED", "", "", "fitter", None, 0, 0.0,
            True, True, True, FailureCategory.VERIFIED,
            50, 0, 0.0, 3.0, 1.0,
        )
        ep = GraphEditEpisode(
            INNER_TRACE_VERSION, "task_1", 1, 10, 5, 3, 1, 8, 2,
            True, "seed_0", 0, "grid_transform", (c,),
        )

        d = ep.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        d2 = json.loads(serialized)
        ep2 = GraphEditEpisode.from_dict(d2)

        assert ep2.task_id == "task_1"
        assert ep2.solved is True
        assert ep2.n_edits_generated == 10
        assert len(ep2.edits) == 1


class TestParamTrialTraceSchema:
    def test_candidate_roundtrip(self):
        from aria.core.guidance_inner_traces import ParamTrialCandidate, FailureCategory

        t = ParamTrialCandidate(
            trial_id="param_0", family="periodic_repair",
            stage="compile_attempt", param_name="axis",
            param_value="row", rank=0, gate_passed=True,
            compile_succeeded=True, verified=False,
            failure_category=FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
            residual_fraction=0.3,
        )
        d = t.to_dict()
        t2 = ParamTrialCandidate.from_dict(d)
        assert t2.trial_id == t.trial_id
        assert t2.family == t.family

    def test_episode_roundtrip(self):
        from aria.core.guidance_inner_traces import (
            ParamTrialCandidate, ParamTrialEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        t = ParamTrialCandidate(
            "param_0", "grid_transform", "compile_attempt",
            "transform", "rotate", 0, True, True, True,
            FailureCategory.VERIFIED, None,
        )
        ep = ParamTrialEpisode(INNER_TRACE_VERSION, "task_1", (t,), 1, 1, 1, 1, "param_0", "grid_transform")

        d = ep.to_dict()
        ep2 = ParamTrialEpisode.from_dict(json.loads(json.dumps(d)))
        assert ep2.winner_family == "grid_transform"

    def test_inner_loop_trace_composite(self):
        from aria.core.guidance_inner_traces import InnerLoopTrace

        trace = InnerLoopTrace(task_id="t1", edit_episode=None, param_episode=None)
        d = trace.to_dict()
        t2 = InnerLoopTrace.from_dict(json.loads(json.dumps(d)))
        assert t2.task_id == "t1"
        assert t2.edit_episode is None


# ---------------------------------------------------------------------------
# Phase 2: Export integration tests
# ---------------------------------------------------------------------------


class TestInnerExportIntegration:
    def test_export_with_inner_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_inner_trace=True)

        assert "inner_trace" in record
        assert record["inner_trace"] is not None
        it = record["inner_trace"]
        assert it["task_id"] == "t1"

    def test_export_without_inner_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_inner_trace=False)
        assert record["inner_trace"] is None

    def test_inner_trace_json_serializable(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_inner_trace=True)
        serialized = json.dumps(record, sort_keys=True)
        roundtrip = json.loads(serialized)
        assert roundtrip["inner_trace"]["task_id"] == "t1"

    def test_trace_edit_search_directly(self):
        from aria.core.guidance_inner_traces import trace_edit_search

        demos = _identity_demos()
        ep = trace_edit_search("t1", demos)
        assert ep.task_id == "t1"
        assert ep.n_seeds >= 0
        assert len(ep.edits) >= 0

    def test_trace_param_trials_directly(self):
        from aria.core.guidance_inner_traces import trace_param_trials

        demos = _identity_demos()
        ep = trace_param_trials("t1", demos)
        assert ep.task_id == "t1"
        assert len(ep.trials) >= 0


# ---------------------------------------------------------------------------
# Phase 4: Inner-loop metrics tests
# ---------------------------------------------------------------------------


class TestInnerLoopMetrics:
    def _make_edit_episode(self, solved: bool = True) -> "GraphEditEpisode":
        from aria.core.guidance_inner_traces import (
            GraphEditCandidate, GraphEditEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        edits = [
            GraphEditCandidate(
                "seed_0", "SEED", "", "", "fitter", None, 0, 0.0,
                True, True, solved, FailureCategory.VERIFIED if solved else FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
                100, 0 if solved else 80, 0.0 if solved else 0.8, 3.0, 1.0 if solved else 2.0,
            ),
            GraphEditCandidate(
                "edit_1_0", "SET_SLOT", "n0", "transform", "rotate", "seed_0", 1, -500.0,
                True, False, False, FailureCategory.COMPILE_MISSING_OP,
                100, None, None, 3.0, None,
            ),
        ]

        return GraphEditEpisode(
            INNER_TRACE_VERSION, "task_1", 1, 5, 2, 1 if solved else 0,
            1 if solved else 0, 4, 1,
            solved, "seed_0" if solved else None, 0 if solved else None,
            "grid_transform", tuple(edits),
        )

    def _make_param_episode(self) -> "ParamTrialEpisode":
        from aria.core.guidance_inner_traces import (
            ParamTrialCandidate, ParamTrialEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        trials = [
            ParamTrialCandidate("p0", "grid_transform", "compile_attempt",
                                "transform", "rotate", 0, True, True, True,
                                FailureCategory.VERIFIED, None),
            ParamTrialCandidate("p1", "periodic_repair", "compile_attempt",
                                "axis", "row", 1, True, True, False,
                                FailureCategory.EXECUTABLE_HIGH_RESIDUAL, 0.3),
        ]
        return ParamTrialEpisode(INNER_TRACE_VERSION, "task_1", tuple(trials), 2, 2, 2, 1, "p0", "grid_transform")

    def test_evaluate_inner_traces(self):
        from aria.core.guidance_inner_eval import evaluate_inner_traces
        from aria.core.guidance_inner_traces import InnerLoopTrace

        edit_ep = self._make_edit_episode(solved=True)
        param_ep = self._make_param_episode()
        traces = [InnerLoopTrace("t1", edit_ep, param_ep)]

        report = evaluate_inner_traces(traces)

        assert report.edit_metrics.n_tasks == 1
        assert report.edit_metrics.n_solved == 1
        assert report.param_metrics.n_tasks == 1
        assert report.param_metrics.n_tasks_with_winner == 1

    def test_edit_metrics_failure_counts(self):
        from aria.core.guidance_inner_eval import evaluate_inner_traces
        from aria.core.guidance_inner_traces import InnerLoopTrace

        ep = self._make_edit_episode(solved=False)
        traces = [InnerLoopTrace("t1", ep, None)]

        report = evaluate_inner_traces(traces)
        assert len(report.edit_metrics.failure_counts) > 0

    def test_param_metrics_family_stats(self):
        from aria.core.guidance_inner_eval import evaluate_inner_traces
        from aria.core.guidance_inner_traces import InnerLoopTrace

        param_ep = self._make_param_episode()
        traces = [InnerLoopTrace("t1", None, param_ep)]

        report = evaluate_inner_traces(traces)
        assert "grid_transform" in report.param_metrics.family_stats

    def test_report_to_dict(self):
        from aria.core.guidance_inner_eval import evaluate_inner_traces
        from aria.core.guidance_inner_traces import InnerLoopTrace

        edit_ep = self._make_edit_episode()
        traces = [InnerLoopTrace("t1", edit_ep, None)]

        report = evaluate_inner_traces(traces)
        d = report.to_dict()
        serialized = json.dumps(d)
        assert "edit_search" in json.loads(serialized)


# ---------------------------------------------------------------------------
# Phase 5: Lightweight baselines
# ---------------------------------------------------------------------------


class TestInnerLoopBaselines:
    def test_edit_type_reranker(self):
        from aria.core.guidance_inner_eval import EditTypeReranker
        from aria.core.guidance_inner_traces import (
            GraphEditCandidate, GraphEditEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        c_win = GraphEditCandidate(
            "e0", "SET_SLOT", "n", "k", "v", None, 0, 0.0,
            True, True, True, FailureCategory.VERIFIED,
            100, 0, 0.0, 3.0, 1.0,
        )
        c_fail = GraphEditCandidate(
            "e1", "ADD_NODE", "n", "", "", None, 0, 0.0,
            True, False, False, FailureCategory.COMPILE_MISSING_OP,
            100, None, None, 3.0, None,
        )

        ep = GraphEditEpisode(INNER_TRACE_VERSION, "t", 1, 2, 2, 1, 1, 3, 0,
                              True, "e0", 0, "", (c_win, c_fail))

        reranker = EditTypeReranker()
        reranker.fit([ep])

        # SET_SLOT should score higher
        assert reranker.score(c_win) > reranker.score(c_fail)

        reranked = reranker.rerank([c_fail, c_win])
        assert reranked[0].edit_id == "e0"

    def test_family_param_reranker(self):
        from aria.core.guidance_inner_eval import FamilyParamReranker
        from aria.core.guidance_inner_traces import (
            ParamTrialCandidate, ParamTrialEpisode, FailureCategory, INNER_TRACE_VERSION,
        )

        t_win = ParamTrialCandidate("p0", "gt", "compile", "transform", "rot", 0,
                                     True, True, True, FailureCategory.VERIFIED, None)
        t_fail = ParamTrialCandidate("p1", "gt", "compile", "axis", "row", 1,
                                      True, True, False, FailureCategory.EXECUTABLE_HIGH_RESIDUAL, 0.5)

        ep = ParamTrialEpisode(INNER_TRACE_VERSION, "t", (t_win, t_fail), 1, 2, 2, 1, "p0", "gt")

        reranker = FamilyParamReranker()
        reranker.fit([ep])

        assert reranker.score(t_win) > reranker.score(t_fail)


# ---------------------------------------------------------------------------
# Phase 6: Failure categorization
# ---------------------------------------------------------------------------


class TestFailureCategories:
    def test_categorize_verified(self):
        from aria.core.guidance_inner_traces import _categorize_failure, FailureCategory
        from aria.core.graph import CompileSuccess

        result = CompileSuccess(task_id="", program=None, bindings_used=None)
        assert _categorize_failure(result, True, 100, 0) == FailureCategory.VERIFIED

    def test_categorize_noop(self):
        from aria.core.guidance_inner_traces import _categorize_failure, FailureCategory
        from aria.core.graph import CompileSuccess

        result = CompileSuccess(task_id="", program=None, bindings_used=None)
        assert _categorize_failure(result, False, 50, 50) == FailureCategory.EXECUTABLE_NOOP

    def test_categorize_residual_reduced(self):
        from aria.core.guidance_inner_traces import _categorize_failure, FailureCategory
        from aria.core.graph import CompileSuccess

        result = CompileSuccess(task_id="", program=None, bindings_used=None)
        assert _categorize_failure(result, False, 100, 50) == FailureCategory.RESIDUAL_REDUCED

    def test_categorize_missing_op(self):
        from aria.core.guidance_inner_traces import _categorize_failure, FailureCategory
        from aria.core.graph import CompileFailure

        result = CompileFailure(task_id="", reason="missing", missing_ops=("foo",))
        assert _categorize_failure(result, False, 100, 100) == FailureCategory.COMPILE_MISSING_OP


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestInnerNoTaskIdLogic:
    def _inner_source_files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return sorted(core.glob("guidance_inner*.py"))

    def test_no_task_id_dispatch(self):
        dispatch_patterns = [r'if\s+.*task_id\s*==', r'if\s+.*task_id\s+in\b']

        for path in self._inner_source_files():
            content = path.read_text()
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in dispatch_patterns:
                    if "task_id" in stripped and re.search(pattern, stripped):
                        pytest.fail(f"{path.name}:{i} task-id dispatch: {stripped}")


class TestNoRegressions:
    def test_existing_export_still_works(self):
        from aria.core.guidance_export import export_task, validate_record

        demos = _identity_demos()
        record = export_task("t1", demos)
        errors = validate_record(record)
        assert not errors

    def test_existing_labels_still_work(self):
        from aria.core.guidance_labels import extract_labels

        demos = _identity_demos()
        labels = extract_labels("t1", demos)
        assert labels.task_id == "t1"

    def test_existing_traces_still_work(self):
        from aria.core.guidance_traces import trace_search_episode

        demos = _identity_demos()
        ep = trace_search_episode("t1", demos)
        assert ep.task_id == "t1"
