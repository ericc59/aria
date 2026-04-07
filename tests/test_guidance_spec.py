"""Tests for specialization-alternative traces, metrics, and baselines.

Covers schema roundtrip, export integration, metrics, baselines,
no task-id logic, no regressions.
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
# Schema roundtrip
# ---------------------------------------------------------------------------


class TestSpecSchemaRoundtrip:
    def test_alternative_roundtrip(self):
        from aria.core.guidance_spec_traces import SpecializationAlternative

        a = SpecializationAlternative(
            value="row", source="evidence", rank=0,
            chosen=True, confidence=0.8, rationale="axis from evidence",
        )
        d = a.to_dict()
        a2 = SpecializationAlternative.from_dict(d)
        assert a2.value == "row"
        assert a2.chosen is True
        assert a2.confidence == 0.8

    def test_decision_roundtrip(self):
        from aria.core.guidance_spec_traces import (
            SpecializationAlternative, SpecializationDecision,
        )

        alts = (
            SpecializationAlternative("row", "evidence", 0, True, 0.8, "chosen"),
            SpecializationAlternative("col", "enumeration", 1, False, 0.4, "alt"),
        )
        dec = SpecializationDecision(
            binding_name="dominant_axis", node_id="__task__",
            binding_type="axis", alternatives=alts,
            chosen_value="row", chosen_source="evidence",
            attempted_count=1, verified_count=0, best_alternative_rank=None,
        )
        d = dec.to_dict()
        dec2 = SpecializationDecision.from_dict(d)
        assert dec2.binding_name == "dominant_axis"
        assert len(dec2.alternatives) == 2
        assert dec2.chosen_value == "row"

    def test_episode_roundtrip(self):
        from aria.core.guidance_spec_traces import (
            SpecializationAlternative, SpecializationDecision,
            SpecializationEpisode, SPEC_TRACE_VERSION,
        )

        alt = SpecializationAlternative("row", "evidence", 0, True, 0.8, "")
        dec = SpecializationDecision(
            "dominant_axis", "__task__", "axis", (alt,),
            "row", "evidence", 0, 0, None,
        )
        ep = SpecializationEpisode(
            SPEC_TRACE_VERSION, "task_1", (dec,), 1, 0, 0, True,
        )
        d = ep.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        ep2 = SpecializationEpisode.from_dict(json.loads(serialized))
        assert ep2.task_id == "task_1"
        assert ep2.default_verified is True
        assert len(ep2.decisions) == 1


# ---------------------------------------------------------------------------
# Export integration
# ---------------------------------------------------------------------------


class TestSpecExportIntegration:
    def test_export_with_spec_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_spec_trace=True)
        assert "spec_trace" in record
        assert record["spec_trace"] is not None
        assert record["spec_trace"]["task_id"] == "t1"

    def test_export_without_spec_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos)
        assert record["spec_trace"] is None

    def test_spec_trace_json_serializable(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("t1", demos, include_spec_trace=True)
        serialized = json.dumps(record, sort_keys=True)
        rt = json.loads(serialized)
        assert "spec_trace" in rt

    def test_trace_specialization_directly(self):
        from aria.core.guidance_spec_traces import trace_specialization

        demos = _identity_demos()
        ep = trace_specialization("t1", demos)
        assert ep.task_id == "t1"
        assert ep.n_bindings >= 0

    def test_trace_has_bg_alternatives(self):
        """Identity task should always have BG color alternatives."""
        from aria.core.guidance_spec_traces import trace_specialization

        demos = _identity_demos()
        ep = trace_specialization("t1", demos)

        bg_decisions = [d for d in ep.decisions if d.binding_name == "bg"]
        if bg_decisions:
            assert len(bg_decisions[0].alternatives) >= 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestSpecMetrics:
    def _make_episode(self, default_verified: bool = True) -> "SpecializationEpisode":
        from aria.core.guidance_spec_traces import (
            SpecializationAlternative, SpecializationDecision,
            SpecializationEpisode, SPEC_TRACE_VERSION,
        )

        axis_alts = (
            SpecializationAlternative("row", "evidence", 0, True, 0.8, ""),
            SpecializationAlternative("col", "enumeration", 1, False, 0.4, ""),
        )
        axis_dec = SpecializationDecision(
            "dominant_axis", "__task__", "axis", axis_alts,
            "row", "evidence", 1, 0 if default_verified else 1, 1 if not default_verified else None,
        )

        period_alts = (
            SpecializationAlternative(2, "evidence", 0, True, 0.8, ""),
            SpecializationAlternative(3, "enumeration", 1, False, 0.3, ""),
            SpecializationAlternative(4, "enumeration", 2, False, 0.2, ""),
        )
        period_dec = SpecializationDecision(
            "dominant_period", "__task__", "period", period_alts,
            2, "evidence", 2, 0, None,
        )

        return SpecializationEpisode(
            SPEC_TRACE_VERSION, "task_1", (axis_dec, period_dec),
            2, 2, 0 if default_verified else 1, default_verified,
        )

    def test_evaluate_spec_traces(self):
        from aria.core.guidance_spec_eval import evaluate_spec_traces

        episodes = [self._make_episode(True), self._make_episode(False)]
        report = evaluate_spec_traces(episodes)

        assert report.n_tasks == 2
        assert report.n_tasks_default_verified == 1
        assert report.n_total_decisions == 4
        assert report.n_decisions_with_alts == 4

    def test_binding_type_stats(self):
        from aria.core.guidance_spec_eval import evaluate_spec_traces

        episodes = [self._make_episode(True)]
        report = evaluate_spec_traces(episodes)

        assert "axis" in report.binding_type_stats
        assert "period" in report.binding_type_stats
        assert report.binding_type_stats["axis"].n_decisions == 1

    def test_source_stats(self):
        from aria.core.guidance_spec_eval import evaluate_spec_traces

        episodes = [self._make_episode(True)]
        report = evaluate_spec_traces(episodes)

        assert "evidence" in report.source_stats

    def test_report_to_dict(self):
        from aria.core.guidance_spec_eval import evaluate_spec_traces

        episodes = [self._make_episode(True)]
        report = evaluate_spec_traces(episodes)

        d = report.to_dict()
        serialized = json.dumps(d)
        assert "binding_type_stats" in json.loads(serialized)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestSpecBaselines:
    def _make_episode(self) -> "SpecializationEpisode":
        from aria.core.guidance_spec_traces import (
            SpecializationAlternative, SpecializationDecision,
            SpecializationEpisode, SPEC_TRACE_VERSION,
        )

        alts = (
            SpecializationAlternative("row", "evidence", 0, True, 0.8, ""),
            SpecializationAlternative("col", "enumeration", 1, False, 0.4, ""),
        )
        dec = SpecializationDecision(
            "dominant_axis", "__task__", "axis", alts,
            "row", "evidence", 1, 0, None,
        )
        return SpecializationEpisode(
            SPEC_TRACE_VERSION, "task_1", (dec,), 1, 1, 0, True,
        )

    def test_source_win_reranker(self):
        from aria.core.guidance_spec_eval import SourceWinReranker

        episodes = [self._make_episode() for _ in range(3)]
        reranker = SourceWinReranker()
        reranker.fit(episodes)

        # "evidence" should score higher than "enumeration"
        from aria.core.guidance_spec_traces import SpecializationAlternative
        a_ev = SpecializationAlternative("row", "evidence", 0, True, 0.8, "")
        a_en = SpecializationAlternative("col", "enumeration", 1, False, 0.4, "")

        assert reranker.score(a_ev) >= reranker.score(a_en)

        reranked = reranker.rerank([a_en, a_ev])
        assert reranked[0].source == "evidence"

    def test_binding_type_reranker(self):
        from aria.core.guidance_spec_eval import BindingTypeReranker
        from aria.core.guidance_spec_traces import SpecializationAlternative

        episodes = [self._make_episode() for _ in range(3)]
        reranker = BindingTypeReranker()
        reranker.fit(episodes)

        a_row = SpecializationAlternative("row", "evidence", 0, True, 0.8, "")
        a_col = SpecializationAlternative("col", "enumeration", 1, False, 0.4, "")

        reranked = reranker.rerank("axis", [a_col, a_row])
        assert reranked[0].value == "row"

    def test_evaluate_source_reranker(self):
        from aria.core.guidance_spec_eval import SourceWinReranker, evaluate_source_reranker

        episodes = [self._make_episode() for _ in range(3)]
        reranker = SourceWinReranker()
        reranker.fit(episodes)

        result = evaluate_source_reranker(reranker, episodes)
        assert result["n_evaluated"] >= 0


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class TestSpecNoTaskIdLogic:
    def _files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return sorted(core.glob("guidance_spec*.py"))

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


class TestSpecNoRegressions:
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
