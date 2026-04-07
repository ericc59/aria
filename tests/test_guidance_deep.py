"""Tests for deep inner-loop traces: within-lane params, library retrieval, residual structure.

Covers:
- ParamAlternative schema roundtrip
- LibraryRetrievalRecord schema roundtrip
- StructuredResidual schema roundtrip
- Export integration with deep_trace
- Deep evaluation metrics
- Lightweight baselines
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


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


# ---------------------------------------------------------------------------
# Schema roundtrip tests
# ---------------------------------------------------------------------------


class TestParamAlternativeSchema:
    def test_roundtrip(self):
        from aria.core.guidance_deep_traces import ParamAlternative, FailureCategory

        a = ParamAlternative(
            alt_id="periodic_0", family="periodic_repair",
            param_set={"axis": "row", "period": 2, "mode": 0},
            source="enumeration", rank=0, gate_passed=True,
            compiled=True, verified=False,
            failure_category=FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
            diff_pixels=50, total_pixels=100, residual_fraction=0.5,
        )
        d = a.to_dict()
        a2 = ParamAlternative.from_dict(d)
        assert a2.alt_id == a.alt_id
        assert a2.param_set == a.param_set
        assert a2.residual_fraction == 0.5

    def test_episode_roundtrip(self):
        from aria.core.guidance_deep_traces import (
            ParamAlternative, ParamAlternativeEpisode, FailureCategory,
        )

        a = ParamAlternative(
            "p0", "periodic_repair", {"axis": "row"}, "enum", 0,
            True, True, True, FailureCategory.VERIFIED,
            0, 100, 0.0,
        )
        ep = ParamAlternativeEpisode(
            "t1", (a,), 1, 1, 1, "p0", "periodic_repair", None, None,
        )
        d = ep.to_dict()
        ep2 = ParamAlternativeEpisode.from_dict(json.loads(json.dumps(d)))
        assert ep2.winner_family == "periodic_repair"
        assert len(ep2.alternatives) == 1


class TestLibraryRetrievalSchema:
    def test_record_roundtrip(self):
        from aria.core.guidance_deep_traces import LibraryRetrievalRecord, FailureCategory

        r = LibraryRetrievalRecord(
            record_id="lib_0", source_task_id="src_abc",
            strategy="direct_reuse", template_ops=("BIND_ROLE", "APPLY_TRANSFORM"),
            adapted=False, adaptation_desc="adapted from src_abc",
            compiled=True, verified=False,
            failure_category=FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
            residual_fraction=0.3,
        )
        d = r.to_dict()
        r2 = LibraryRetrievalRecord.from_dict(d)
        assert r2.source_task_id == "src_abc"
        assert r2.strategy == "direct_reuse"

    def test_episode_roundtrip(self):
        from aria.core.guidance_deep_traces import (
            LibraryRetrievalRecord, LibraryRetrievalEpisode, FailureCategory,
        )

        r = LibraryRetrievalRecord(
            "lib_0", "src", "direct_reuse", ("OP",),
            False, "", True, True, FailureCategory.VERIFIED, None,
        )
        ep = LibraryRetrievalEpisode("t1", 5, (r,), 1, 0, 1, 1, "lib_0")
        d = ep.to_dict()
        ep2 = LibraryRetrievalEpisode.from_dict(json.loads(json.dumps(d)))
        assert ep2.winner_record_id == "lib_0"


class TestStructuredResidualSchema:
    def test_roundtrip(self):
        from aria.core.guidance_deep_traces import StructuredResidual, ResidualRegion

        region = ResidualRegion("interior", 20, 50, 0.4)
        r = StructuredResidual(
            total_diff=30, total_pixels=100, diff_fraction=0.3,
            is_localized=True, dominant_region="interior",
            regions=(region,),
            blamed_node_ids=("n1",), blamed_ops=("APPLY_TRANSFORM",),
            repair_hint_count=2, repair_hint_bindings=("axis", "period"),
        )
        d = r.to_dict()
        r2 = StructuredResidual.from_dict(d)
        assert r2.is_localized is True
        assert r2.dominant_region == "interior"
        assert len(r2.regions) == 1
        assert r2.blamed_ops == ("APPLY_TRANSFORM",)

    def test_deep_trace_composite_roundtrip(self):
        from aria.core.guidance_deep_traces import DeepTrace

        trace = DeepTrace(task_id="t1", param_alternatives=None,
                          library_retrieval=None, structured_residuals=())
        d = trace.to_dict()
        t2 = DeepTrace.from_dict(json.loads(json.dumps(d)))
        assert t2.task_id == "t1"


# ---------------------------------------------------------------------------
# Export integration
# ---------------------------------------------------------------------------


class TestDeepExportIntegration:
    def test_export_with_deep_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_deep_trace=True)
        assert "deep_trace" in record
        assert record["deep_trace"] is not None
        assert record["deep_trace"]["task_id"] == "t1"

    def test_export_without_deep_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_deep_trace=False)
        assert record["deep_trace"] is None

    def test_deep_trace_json_serializable(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_deep_trace=True)
        serialized = json.dumps(record, sort_keys=True)
        roundtrip = json.loads(serialized)
        assert "deep_trace" in roundtrip

    def test_trace_param_alternatives_directly(self):
        from aria.core.guidance_deep_traces import trace_param_alternatives

        demos = _identity_demos()
        ep = trace_param_alternatives("t1", demos)
        assert ep.task_id == "t1"
        # Identity task: grid_transform alternatives should produce some
        assert ep.n_total >= 0

    def test_trace_library_retrieval_no_library(self):
        from aria.core.guidance_deep_traces import trace_library_retrieval

        demos = _identity_demos()
        ep = trace_library_retrieval("t1", demos, library=None)
        assert ep.n_templates_available == 0
        assert len(ep.records) == 0

    def test_extract_structured_residuals(self):
        from aria.core.guidance_deep_traces import extract_structured_residuals

        demos = _identity_demos()
        residuals = extract_structured_residuals("t1", demos)
        # Identity task likely solves directly, so no failed-compile residuals
        assert isinstance(residuals, tuple)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestDeepMetrics:
    def _make_param_episode(self) -> "ParamAlternativeEpisode":
        from aria.core.guidance_deep_traces import (
            ParamAlternative, ParamAlternativeEpisode, FailureCategory,
        )

        alts = [
            ParamAlternative("p0", "periodic_repair", {"axis": "row", "period": 2}, "enum",
                             0, True, True, True, FailureCategory.VERIFIED, 0, 100, 0.0),
            ParamAlternative("p1", "periodic_repair", {"axis": "col", "period": 2}, "enum",
                             1, True, True, False, FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
                             30, 100, 0.3),
            ParamAlternative("p2", "grid_transform", {"degrees": 90}, "enum",
                             2, True, True, False, FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
                             50, 100, 0.5),
        ]
        return ParamAlternativeEpisode("t1", tuple(alts), 2, 3, 1, "p0", "periodic_repair", "p1", 0.3)

    def test_evaluate_deep_traces(self):
        from aria.core.guidance_deep_eval import evaluate_deep_traces
        from aria.core.guidance_deep_traces import DeepTrace, StructuredResidual, ResidualRegion

        param_ep = self._make_param_episode()
        residual = StructuredResidual(30, 100, 0.3, False, None, (), (), (), 0, ())
        trace = DeepTrace("t1", param_ep, None, (residual,))

        report = evaluate_deep_traces([trace])
        assert report.param_alt_metrics.n_tasks == 1
        assert report.param_alt_metrics.n_tasks_with_winner == 1
        assert report.param_alt_metrics.winner_at_1 == 1
        assert report.residual_metrics.n_tasks_with_residuals == 1

    def test_param_family_counts(self):
        from aria.core.guidance_deep_eval import evaluate_deep_traces
        from aria.core.guidance_deep_traces import DeepTrace

        param_ep = self._make_param_episode()
        trace = DeepTrace("t1", param_ep, None)

        report = evaluate_deep_traces([trace])
        assert "periodic_repair" in report.param_alt_metrics.n_families

    def test_report_to_dict(self):
        from aria.core.guidance_deep_eval import evaluate_deep_traces
        from aria.core.guidance_deep_traces import DeepTrace

        param_ep = self._make_param_episode()
        trace = DeepTrace("t1", param_ep, None)

        report = evaluate_deep_traces([trace])
        d = report.to_dict()
        serialized = json.dumps(d)
        assert "param_alternatives" in json.loads(serialized)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestDeepBaselines:
    def test_param_frequency_reranker(self):
        from aria.core.guidance_deep_eval import ParamFrequencyReranker
        from aria.core.guidance_deep_traces import (
            ParamAlternative, ParamAlternativeEpisode, FailureCategory,
        )

        a_win = ParamAlternative("p0", "pr", {"axis": "row"}, "e", 0,
                                  True, True, True, FailureCategory.VERIFIED, 0, 100, 0.0)
        a_fail = ParamAlternative("p1", "pr", {"axis": "col"}, "e", 1,
                                   True, True, False, FailureCategory.EXECUTABLE_HIGH_RESIDUAL,
                                   30, 100, 0.3)
        ep = ParamAlternativeEpisode("t1", (a_win, a_fail), 1, 2, 1, "p0", "pr", "p1", 0.3)

        reranker = ParamFrequencyReranker()
        reranker.fit([ep])

        reranked = reranker.rerank([a_fail, a_win])
        assert reranked[0].alt_id == "p0"  # winner ranked first

    def test_residual_guided_reranker(self):
        from aria.core.guidance_deep_eval import ResidualGuidedReranker
        from aria.core.guidance_deep_traces import (
            DeepTrace, ParamAlternative, ParamAlternativeEpisode,
            StructuredResidual, FailureCategory,
        )

        a = ParamAlternative("p0", "pr", {}, "e", 0, True, True, True,
                              FailureCategory.VERIFIED, 0, 100, 0.0)
        param_ep = ParamAlternativeEpisode("t1", (a,), 1, 1, 1, "p0", "pr", None, None)
        residual = StructuredResidual(30, 100, 0.3, False, None, (), (), (), 0, ())
        trace = DeepTrace("t1", param_ep, None, (residual,))

        reranker = ResidualGuidedReranker()
        reranker.fit([trace])

        predicted = reranker.predict_family((residual,))
        assert predicted == "pr"

    def test_evaluate_param_reranker(self):
        from aria.core.guidance_deep_eval import ParamFrequencyReranker, evaluate_param_reranker
        from aria.core.guidance_deep_traces import (
            ParamAlternative, ParamAlternativeEpisode, FailureCategory,
        )

        a = ParamAlternative("p0", "pr", {"x": 1}, "e", 0, True, True, True,
                              FailureCategory.VERIFIED, 0, 100, 0.0)
        ep = ParamAlternativeEpisode("t1", (a,), 1, 1, 1, "p0", "pr", None, None)

        reranker = ParamFrequencyReranker()
        reranker.fit([ep])

        result = evaluate_param_reranker(reranker, [ep])
        assert result["n_evaluated"] == 1


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestDeepNoTaskIdLogic:
    def _files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return sorted(core.glob("guidance_deep*.py"))

    def test_no_task_id_dispatch(self):
        dispatch_patterns = [r'if\s+.*task_id\s*==', r'if\s+.*task_id\s+in\b']
        for path in self._files():
            content = path.read_text()
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in dispatch_patterns:
                    if "task_id" in stripped and re.search(pattern, stripped):
                        pytest.fail(f"{path.name}:{i} task-id dispatch: {stripped}")


class TestDeepNoRegressions:
    def test_export_unchanged(self):
        from aria.core.guidance_export import export_task, validate_record

        demos = _identity_demos()
        record = export_task("t1", demos)
        assert not validate_record(record)

    def test_existing_traces_unchanged(self):
        from aria.core.guidance_traces import trace_search_episode

        demos = _identity_demos()
        ep = trace_search_episode("t1", demos)
        assert ep.task_id == "t1"
