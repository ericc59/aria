"""Tests for search-decision traces, decision-level evaluation, and trace baselines.

Covers:
- Decision-trace schema roundtrip
- Export format including candidate traces
- Decision-level evaluation metrics
- Trace baselines (frequency, retrieval, feature-conditioned)
- No task-id logic
- No regressions in existing guidance export
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list


# ---------------------------------------------------------------------------
# Synthetic task helpers
# ---------------------------------------------------------------------------


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


def _scale_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 2], [3, 4]])
    o1 = grid_from_list([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    g2 = grid_from_list([[5, 6], [7, 8]])
    o2 = grid_from_list([[5, 5, 6, 6], [5, 5, 6, 6], [7, 7, 8, 8], [7, 7, 8, 8]])
    return (DemoPair(input=g1, output=o1), DemoPair(input=g2, output=o2))


# ---------------------------------------------------------------------------
# Phase 1: Decision-trace schema tests
# ---------------------------------------------------------------------------


class TestDecisionTraceSchema:
    def test_candidate_record_roundtrip(self):
        from aria.core.guidance_traces import CandidateRecord

        c = CandidateRecord(
            candidate_id="size_0",
            stage="output_size",
            label="same_as_input",
            rank=0,
            attempted=True,
            verified=True,
            skipped_reason=None,
            params={"mode": "same_as_input"},
            residual=None,
            score=1.0,
        )
        d = c.to_dict()
        c2 = CandidateRecord.from_dict(d)
        assert c2.candidate_id == c.candidate_id
        assert c2.label == c.label
        assert c2.verified == c.verified
        assert c2.score == c.score

    def test_decision_step_roundtrip(self):
        from aria.core.guidance_traces import CandidateRecord, DecisionStep

        c1 = CandidateRecord("c0", "lane", "replication", 0, True, False, None, {}, None, 0.5)
        c2 = CandidateRecord("c1", "lane", "relocation", 1, True, True, None, {}, None, 0.8)

        step = DecisionStep(
            stage="lane",
            n_candidates=2,
            n_attempted=2,
            n_verified=1,
            winner_id="c1",
            best_failed_id="c0",
            candidates=(c1, c2),
        )
        d = step.to_dict()
        step2 = DecisionStep.from_dict(d)
        assert step2.stage == "lane"
        assert step2.n_candidates == 2
        assert step2.winner_id == "c1"
        assert len(step2.candidates) == 2

    def test_search_episode_roundtrip(self):
        from aria.core.guidance_traces import (
            CandidateRecord, DecisionStep, SearchEpisode, TRACE_SCHEMA_VERSION,
        )

        c = CandidateRecord("size_0", "output_size", "same_as_input", 0, True, True, None, {}, None, 1.0)
        step = DecisionStep("output_size", 1, 1, 1, "size_0", None, (c,))

        episode = SearchEpisode(
            schema_version=TRACE_SCHEMA_VERSION,
            task_id="test_1",
            steps=(step,),
            final_winner="same_as_input",
            final_winner_stage="output_size",
            best_failed=None,
            best_failed_stage=None,
            best_failed_residual=None,
            solved=True,
        )
        d = episode.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        d2 = json.loads(serialized)
        ep2 = SearchEpisode.from_dict(d2)

        assert ep2.task_id == "test_1"
        assert ep2.solved is True
        assert ep2.total_candidates == 1
        assert ep2.steps[0].winner_id == "size_0"

    def test_episode_helpers(self):
        from aria.core.guidance_traces import (
            CandidateRecord, DecisionStep, SearchEpisode, TRACE_SCHEMA_VERSION,
        )

        c1 = CandidateRecord("s0", "output_size", "same", 0, True, True, None, {}, None, 1.0)
        c2 = CandidateRecord("l0", "lane", "repair", 0, True, False, None, {}, None, 0.3)
        step1 = DecisionStep("output_size", 1, 1, 1, "s0", None, (c1,))
        step2 = DecisionStep("lane", 1, 1, 0, None, "l0", (c2,))

        ep = SearchEpisode(TRACE_SCHEMA_VERSION, "t", (step1, step2),
                           "same", "output_size", "repair", "lane", None, True)

        assert ep.total_candidates == 2
        assert ep.total_attempted == 2
        assert ep.step_for_stage("output_size") is step1
        assert ep.step_for_stage("lane") is step2
        assert ep.step_for_stage("nonexistent") is None


# ---------------------------------------------------------------------------
# Phase 2: Export integration tests
# ---------------------------------------------------------------------------


class TestExportIntegration:
    def test_export_with_search_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_1", demos, include_search_trace=True)

        assert "search_trace" in record
        st = record["search_trace"]
        assert st is not None
        assert st["task_id"] == "test_1"
        assert "steps" in st
        assert isinstance(st["steps"], list)

    def test_export_without_search_trace(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_1", demos, include_search_trace=False)

        assert "search_trace" in record
        assert record["search_trace"] is None

    def test_export_search_trace_is_json_serializable(self):
        from aria.core.guidance_export import export_task

        demos = _identity_demos()
        record = export_task("test_1", demos, include_search_trace=True)

        serialized = json.dumps(record, sort_keys=True)
        roundtrip = json.loads(serialized)
        assert roundtrip["search_trace"]["task_id"] == "test_1"

    def test_trace_search_episode_directly(self):
        from aria.core.guidance_traces import trace_search_episode

        demos = _identity_demos()
        episode = trace_search_episode("test_1", demos)

        assert episode.task_id == "test_1"
        assert len(episode.steps) > 0

        # Should have at least output_size and lane stages
        stages = {s.stage for s in episode.steps}
        assert "output_size" in stages
        assert "lane" in stages

    def test_trace_has_candidates(self):
        from aria.core.guidance_traces import trace_search_episode

        demos = _identity_demos()
        episode = trace_search_episode("test_1", demos)

        # Identity task should have at least one size candidate (same_as_input)
        size_step = episode.step_for_stage("output_size")
        assert size_step is not None
        assert size_step.n_candidates > 0

    def test_trace_batch_export(self):
        from aria.core.guidance_traces import export_search_traces, load_search_traces

        demos_map = {"t1": _identity_demos(), "t2": _identity_demos()}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            counts = export_search_traces(
                list(demos_map.keys()),
                lambda tid: demos_map[tid],
                path,
            )
            assert counts["exported"] == 2

            loaded = load_search_traces(path)
            assert len(loaded) == 2
            assert {e.task_id for e in loaded} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Phase 3: Decision-level evaluation metrics tests
# ---------------------------------------------------------------------------


class TestDecisionEvalMetrics:
    def _make_episode(self, solved: bool = True) -> "SearchEpisode":
        from aria.core.guidance_traces import (
            CandidateRecord, DecisionStep, SearchEpisode, TRACE_SCHEMA_VERSION,
        )

        size_c0 = CandidateRecord("s0", "output_size", "same_as_input", 0, True, True, None, {}, None, 1.0)
        size_c1 = CandidateRecord("s1", "output_size", "scale_input", 1, True, False, None, {},
                                   {"total_diff": 10, "total_pixels": 100, "diff_fraction": 0.1}, 0.9)

        lane_c0 = CandidateRecord("l0", "lane", "periodic_repair", 0, True, False, None,
                                   {"class_score": 0.7, "final_score": 0.5}, None, 0.5)
        lane_c1 = CandidateRecord("l1", "lane", "grid_transform", 1, True, False, None,
                                   {"class_score": 0.3, "final_score": 0.2}, None, 0.2)

        graph_c0 = CandidateRecord("g0", "graph", "grid_transform", 0, True, solved, None,
                                    {"family": "grid_transform"}, None, 1.0 if solved else 0.5)

        steps = (
            DecisionStep("output_size", 2, 2, 1, "s0", "s1", (size_c0, size_c1)),
            DecisionStep("lane", 2, 2, 0, "l0", None, (lane_c0, lane_c1)),
            DecisionStep("graph", 1, 1, 1 if solved else 0,
                         "g0" if solved else None, None if solved else "g0", (graph_c0,)),
        )

        return SearchEpisode(
            TRACE_SCHEMA_VERSION, "test_task", steps,
            "grid_transform" if solved else None,
            "graph" if solved else None,
            None if solved else "grid_transform",
            None if solved else "graph",
            None,
            solved,
        )

    def test_evaluate_search_traces(self):
        from aria.core.guidance_trace_eval import evaluate_search_traces

        episodes = [self._make_episode(solved=True), self._make_episode(solved=False)]
        report = evaluate_search_traces(episodes)

        assert report.n_episodes == 2
        assert report.n_solved == 1
        assert report.solve_rate == 0.5

        # Stage metrics
        assert "output_size" in report.stage_metrics
        assert "lane" in report.stage_metrics
        assert "graph" in report.stage_metrics

        size_m = report.stage_metrics["output_size"]
        assert size_m.n_tasks_with_candidates == 2
        assert size_m.total_candidates == 4
        assert size_m.winner_at_1 == 2  # winner is at rank 0 in both

    def test_winner_in_top_k(self):
        from aria.core.guidance_trace_eval import winner_in_top_k

        ep = self._make_episode(solved=True)

        assert winner_in_top_k(ep, "output_size", 1) is True
        assert winner_in_top_k(ep, "graph", 1) is True
        assert winner_in_top_k(ep, "nonexistent", 1) is None

    def test_winner_rank(self):
        from aria.core.guidance_trace_eval import winner_rank

        ep = self._make_episode(solved=True)

        assert winner_rank(ep, "output_size") == 0
        assert winner_rank(ep, "graph") == 0
        assert winner_rank(ep, "lane") == 0

    def test_family_confusion(self):
        from aria.core.guidance_trace_eval import evaluate_search_traces

        episodes = [self._make_episode(solved=True)]
        report = evaluate_search_traces(episodes)

        assert len(report.family_confusion) > 0
        # grid_transform should appear in confusion stats
        assert "grid_transform" in report.family_confusion

    def test_calibration_buckets(self):
        from aria.core.guidance_trace_eval import evaluate_search_traces

        episodes = [self._make_episode(solved=True)]
        report = evaluate_search_traces(episodes)

        assert len(report.calibration) == 5
        for bucket in report.calibration:
            assert "score_range" in bucket.to_dict()
            assert "verification_rate" in bucket.to_dict()

    def test_report_to_dict(self):
        from aria.core.guidance_trace_eval import evaluate_search_traces

        episodes = [self._make_episode(solved=True)]
        report = evaluate_search_traces(episodes)

        d = report.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        assert "stage_metrics" in json.loads(serialized)


# ---------------------------------------------------------------------------
# Phase 4: Trace baselines tests
# ---------------------------------------------------------------------------


class TestTraceBaselines:
    def _make_episode(self, winner_label: str = "same_as_input") -> "SearchEpisode":
        from aria.core.guidance_traces import (
            CandidateRecord, DecisionStep, SearchEpisode, TRACE_SCHEMA_VERSION,
        )

        c0 = CandidateRecord("s0", "output_size", winner_label, 0, True, True, None, {}, None, 1.0)
        c1 = CandidateRecord("s1", "output_size", "scale_input", 1, True, False, None, {}, None, 0.5)

        step = DecisionStep("output_size", 2, 2, 1, "s0", "s1", (c0, c1))

        return SearchEpisode(
            TRACE_SCHEMA_VERSION, "test", (step,),
            winner_label, "output_size", None, None, None, True,
        )

    def _make_record(self) -> dict:
        return {
            "task_id": "test",
            "train_demos": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "perception_summaries": [
                {"n_objects_4": 2, "palette": [0, 1, 2, 3, 4],
                 "partition": None, "n_framed_regions": 0, "has_legend": False}
            ],
            "slot_grid": None,
            "size_spec": {"mode": "same_as_input"},
            "verify_result": "solved",
        }

    def test_frequency_reranker(self):
        from aria.core.guidance_trace_baselines import FrequencyReranker

        episodes = [self._make_episode("same_as_input") for _ in range(5)]
        reranker = FrequencyReranker()
        reranker.fit(episodes)

        # same_as_input should be ranked first
        step = episodes[0].steps[0]
        reranked = reranker.rerank(step)
        assert reranked[0].label == "same_as_input"

    def test_frequency_predict_top_k(self):
        from aria.core.guidance_trace_baselines import FrequencyReranker

        episodes = [self._make_episode("same_as_input") for _ in range(3)]
        reranker = FrequencyReranker()
        reranker.fit(episodes)

        step = episodes[0].steps[0]
        top = reranker.predict_top_k(step, k=1)
        assert "same_as_input" in top

    def test_retrieval_reranker(self):
        from aria.core.guidance_trace_baselines import RetrievalReranker

        episodes = [self._make_episode("same_as_input")]
        records = [self._make_record()]

        reranker = RetrievalReranker()
        reranker.fit(episodes, records)

        assert len(reranker.entries) == 1

        step = episodes[0].steps[0]
        reranked = reranker.rerank(step, records[0])
        assert len(reranked) == 2

    def test_feature_conditioned_reranker(self):
        from aria.core.guidance_trace_baselines import FeatureConditionedReranker

        episodes = [self._make_episode("same_as_input")]
        records = [self._make_record()]

        reranker = FeatureConditionedReranker()
        reranker.fit(episodes, records)

        step = episodes[0].steps[0]
        reranked = reranker.rerank(step, records[0])
        assert len(reranked) == 2
        # same_as_input should rank first since it was the winner
        assert reranked[0].label == "same_as_input"

    def test_evaluate_reranker(self):
        from aria.core.guidance_trace_baselines import (
            FrequencyReranker, evaluate_reranker,
        )

        episodes = [self._make_episode("same_as_input") for _ in range(3)]
        reranker = FrequencyReranker()
        reranker.fit(episodes)

        result = evaluate_reranker(
            "frequency",
            reranker.rerank,
            episodes,
            "output_size",
        )
        assert result.n_evaluated == 3
        assert result.reranked_winner_at_1 == 3
        d = result.to_dict()
        assert "reranker_name" in d
        assert "winner_at_1_lift" in d


# ---------------------------------------------------------------------------
# Integrity tests
# ---------------------------------------------------------------------------


class TestTraceNoTaskIdLogic:
    def _trace_source_files(self) -> list[Path]:
        core = Path(__file__).parent.parent / "aria" / "core"
        return sorted(core.glob("guidance_trace*.py"))

    def test_no_task_id_dispatch(self):
        dispatch_patterns = [
            r'if\s+.*task_id\s*==',
            r'if\s+.*task_id\s+in\b',
            r'task_id\s*\[',
        ]

        for path in self._trace_source_files():
            content = path.read_text()
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in dispatch_patterns:
                    if "task_id" in stripped and re.search(pattern, stripped):
                        if "record_by_id" in stripped or "record.get" in stripped:
                            continue
                        pytest.fail(
                            f"{path.name}:{i} looks like task-id dispatch: {stripped}"
                        )


class TestExistingGuidanceNotBroken:
    """Verify existing guidance export/labels/eval still work after trace integration."""

    def test_export_without_trace_unchanged(self):
        from aria.core.guidance_export import export_task, REQUIRED_KEYS, validate_record

        demos = _identity_demos()
        record = export_task("test_1", demos)

        # All required keys present
        missing = REQUIRED_KEYS - set(record.keys())
        assert not missing

        # Validation passes
        errors = validate_record(record)
        assert not errors

    def test_labels_still_work(self):
        from aria.core.guidance_labels import extract_labels

        demos = _identity_demos()
        labels = extract_labels("test_1", demos)
        assert labels.task_id == "test_1"

    def test_eval_still_works(self):
        from aria.core.guidance_eval import evaluate_proposals
        from aria.core.guidance_export import export_task
        from aria.core.guidance_labels import extract_labels

        demos = _identity_demos()
        record = export_task("t1", demos)
        labels = extract_labels("t1", demos)

        report = evaluate_proposals([record], [labels])
        assert report.n_tasks == 1
