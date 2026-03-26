"""Tests for the training corpus audit script."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from scripts.audit_training_corpus import (
    AuditReport,
    CorpusStats,
    LibraryStats,
    SolveStats,
    TraceStats,
    audit_library,
    audit_program_store,
    audit_solve_report,
    audit_trace_store,
    detect_weaknesses,
    format_report,
    generate_recommendations,
    run_audit,
)


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


def _write_program_store(path: Path, programs: list[dict]) -> None:
    path.write_text(json.dumps({"version": 2, "programs": programs}))


def _write_solve_report(path: Path, tasks: list[dict]) -> None:
    total = len(tasks)
    solved = sum(1 for t in tasks if t.get("solved"))
    path.write_text(json.dumps({
        "total": total,
        "solved": solved,
        "solve_rate": f"{100*solved/max(total,1):.1f}%",
        "tasks": tasks,
    }))


def _write_trace_store(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "records": records}))


def _write_library(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "entries": entries}))


# ---------------------------------------------------------------------------
# audit_program_store
# ---------------------------------------------------------------------------


class TestAuditProgramStore:
    def test_empty_file(self, results_dir: Path) -> None:
        _write_program_store(results_dir / "program_store.json", [])
        stats = audit_program_store(results_dir / "program_store.json")
        assert stats.total_programs == 0
        assert stats.unique_task_ids == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        stats = audit_program_store(tmp_path / "nope.json")
        assert stats.total_programs == 0

    def test_basic_counts(self, results_dir: Path) -> None:
        programs = [
            {
                "program": "let v0: GRID = fill_enclosed(input, 4)\n-> v0",
                "task_ids": ["abc"],
                "sources": ["solve-report:r1.json"],
                "use_count": 1,
                "signatures": ["dims:same", "change:additive"],
            },
            {
                "program": "let v0: OBJECT_SET = find_objects(input)\nlet v1: GRID = overlay(input, v0)\n-> v1",
                "task_ids": ["abc", "def"],
                "sources": ["corpus-report:c1.json"],
                "use_count": 2,
                "signatures": ["dims:same"],
            },
        ]
        _write_program_store(results_dir / "program_store.json", programs)
        stats = audit_program_store(results_dir / "program_store.json")

        assert stats.total_programs == 2
        assert stats.unique_task_ids == 2  # abc, def
        assert stats.programs_with_signatures == 2
        assert "dims:same" in stats.signature_set
        assert "change:additive" in stats.signature_set
        assert stats.step_count_distribution == {1: 1, 2: 1}
        assert stats.op_usage["find_objects"] == 1
        assert stats.op_usage["fill_enclosed"] == 1
        assert stats.op_usage["overlay"] == 1

    def test_task_with_multiple_programs(self, results_dir: Path) -> None:
        programs = [
            {"program": "let v0: GRID = foo(input)\n-> v0", "task_ids": ["t1"], "sources": [], "use_count": 1, "signatures": []},
            {"program": "let v0: GRID = bar(input)\n-> v0", "task_ids": ["t1"], "sources": [], "use_count": 1, "signatures": []},
            {"program": "let v0: GRID = baz(input)\n-> v0", "task_ids": ["t2"], "sources": [], "use_count": 1, "signatures": []},
        ]
        _write_program_store(results_dir / "program_store.json", programs)
        stats = audit_program_store(results_dir / "program_store.json")

        assert stats.task_ids_with_multiple_programs == 1


# ---------------------------------------------------------------------------
# audit_solve_report
# ---------------------------------------------------------------------------


class TestAuditSolveReport:
    def test_basic(self, results_dir: Path) -> None:
        tasks = [
            {"task_id": "a", "solved": True, "rounds": 1, "solve_source": "search"},
            {"task_id": "b", "solved": False, "rounds": 2},
            {"task_id": "c", "solved": True, "rounds": 1, "solve_source": "retrieval",
             "task_signatures": ["dims:same"]},
        ]
        _write_solve_report(results_dir / "report.json", tasks)
        stats = audit_solve_report(results_dir / "report.json")

        assert stats.total_tasks == 3
        assert stats.solved_tasks == 2
        assert abs(stats.solve_rate - 66.7) < 0.1
        assert stats.rounds_distribution == {1: 2, 2: 1}
        assert stats.solve_sources == {"search": 1, "retrieval": 1}
        assert stats.task_signatures_available == 1
        assert "dims:same" in stats.unique_signatures

    def test_missing_file(self, tmp_path: Path) -> None:
        stats = audit_solve_report(tmp_path / "nope.json")
        assert stats.total_tasks == 0


# ---------------------------------------------------------------------------
# audit_trace_store
# ---------------------------------------------------------------------------


class TestAuditTraceStore:
    def test_missing(self, tmp_path: Path) -> None:
        stats = audit_trace_store(tmp_path / "nope.json")
        assert stats.total_records == 0

    def test_basic(self, results_dir: Path) -> None:
        records = [
            {
                "task_id": "t1",
                "solved": True,
                "candidates_tried": 50,
                "rounds": [
                    {
                        "plan": {"name": "generic", "max_steps": 3, "max_candidates": 100, "allowed_ops": []},
                        "solved": True,
                        "candidates_tried": 50,
                        "feedback": {"suggested_focus": "generic"},
                        "trace": [
                            {"candidate_num": 1, "depth": 2, "passed": False, "score": 400.0},
                            {"candidate_num": 2, "depth": 3, "passed": True, "score": 1000000.0},
                        ],
                    }
                ],
            },
            {
                "task_id": "t2",
                "solved": False,
                "candidates_tried": 200,
                "rounds": [
                    {
                        "plan": {"name": "generic", "max_steps": 3, "max_candidates": 100, "allowed_ops": []},
                        "solved": False,
                        "candidates_tried": 100,
                        "feedback": {"suggested_focus": "marker_geometry"},
                        "trace": [
                            {"candidate_num": 1, "depth": 1, "passed": False, "score": -500.0},
                        ],
                    },
                    {
                        "plan": {"name": "marker_geometry", "max_steps": 4, "max_candidates": 100, "allowed_ops": []},
                        "solved": False,
                        "candidates_tried": 100,
                        "feedback": {"suggested_focus": "marker_geometry"},
                        "trace": [
                            {"candidate_num": 1, "depth": 2, "passed": False, "score": 300.0},
                        ],
                    },
                ],
            },
        ]
        _write_trace_store(results_dir / "refinement_traces.json", records)
        stats = audit_trace_store(results_dir / "refinement_traces.json")

        assert stats.total_records == 2
        assert stats.solved_records == 1
        assert stats.failed_records == 1
        assert stats.total_rounds == 3
        assert stats.total_candidates == 250
        assert abs(stats.avg_candidates_per_record - 125.0) < 0.1
        assert stats.max_depth == 3
        assert stats.focus_distribution == {"generic": 1, "marker_geometry": 2}
        assert stats.records_with_near_miss == 1  # t1 has score=400 > 350

    def test_near_miss_threshold(self, results_dir: Path) -> None:
        records = [
            {
                "task_id": "t1",
                "solved": False,
                "candidates_tried": 10,
                "rounds": [{
                    "plan": {"name": "generic", "max_steps": 3, "max_candidates": 100, "allowed_ops": []},
                    "solved": False,
                    "candidates_tried": 10,
                    "feedback": {"suggested_focus": "size"},
                    "trace": [
                        {"candidate_num": 1, "depth": 1, "passed": False, "score": 349.0},
                    ],
                }],
            },
        ]
        _write_trace_store(results_dir / "refinement_traces.json", records)
        stats = audit_trace_store(results_dir / "refinement_traces.json")
        assert stats.records_with_near_miss == 0


# ---------------------------------------------------------------------------
# audit_library
# ---------------------------------------------------------------------------


class TestAuditLibrary:
    def test_basic(self, results_dir: Path) -> None:
        entries = [
            {"name": "lib_a", "use_count": 4, "level": 1, "return_type": "GRID"},
            {"name": "lib_b", "use_count": 2, "level": 1, "return_type": "OBJECT"},
            {"name": "lib_c", "use_count": 3, "level": 2, "return_type": "GRID"},
        ]
        _write_library(results_dir / "library.json", entries)
        stats = audit_library(results_dir / "library.json")

        assert stats.total_entries == 3
        assert stats.total_use_count == 9
        assert stats.max_level == 2
        assert stats.return_type_distribution == {"GRID": 2, "OBJECT": 1}


# ---------------------------------------------------------------------------
# Weakness detection
# ---------------------------------------------------------------------------


class TestWeaknesses:
    def test_low_corpus(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 30
        report.corpus.unique_task_ids = 10
        report.corpus.step_count_distribution = {1: 20, 2: 10}
        weaknesses = detect_weaknesses(report)
        assert any("CRITICAL" in w for w in weaknesses)
        assert any("skew" in w.lower() for w in weaknesses)

    def test_no_signatures(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 300
        report.corpus.unique_task_ids = 100
        report.corpus.step_count_distribution = {3: 100, 5: 100, 7: 100}
        weaknesses = detect_weaknesses(report)
        assert any("signatures" in w.lower() for w in weaknesses)

    def test_no_traces(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 300
        report.corpus.unique_task_ids = 100
        report.corpus.programs_with_signatures = 100
        report.corpus.step_count_distribution = {4: 300}
        weaknesses = detect_weaknesses(report)
        assert any("traces" in w.lower() for w in weaknesses)

    def test_healthy_corpus_no_critical(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 500
        report.corpus.unique_task_ids = 200
        report.corpus.programs_with_signatures = 500
        report.corpus.step_count_distribution = {3: 100, 5: 200, 8: 150, 10: 50}
        report.traces.total_records = 300
        report.traces.records_with_near_miss = 50
        report.traces.focus_distribution = {
            "generic": 100, "marker_geometry": 80, "color_map": 60, "size": 60,
        }
        report.library.total_entries = 40
        report.eval_solve.total_tasks = 100
        report.eval_solve.solved_tasks = 30
        weaknesses = detect_weaknesses(report)
        assert not any("CRITICAL" in w for w in weaknesses)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_not_ready(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 47
        report.corpus.programs_with_signatures = 0
        report.corpus.step_count_distribution = {1: 14, 2: 8, 3: 4, 4: 8, 5: 3, 6: 1}
        report.traces.total_records = 0
        recs = generate_recommendations(report)
        assert any("PRIORITY 1" in r for r in recs)
        assert any("NOT READY" in r and "SKETCH" in r for r in recs)
        assert any("NOT READY" in r and "NEXT_EDIT" in r for r in recs)

    def test_ready_for_sketch(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 300
        report.corpus.programs_with_signatures = 200
        report.corpus.step_count_distribution = {3: 100, 5: 100, 8: 100}
        report.traces.total_records = 200
        report.traces.records_with_near_miss = 50
        report.traces.focus_distribution = {"generic": 100, "marker_geometry": 50, "size": 50}
        report.library.total_entries = 50
        recs = generate_recommendations(report)
        assert any("feasible" in r.lower() and "SKETCH" in r for r in recs)


# ---------------------------------------------------------------------------
# Format and end-to-end
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_format_has_sections(self) -> None:
        report = AuditReport()
        report.corpus.total_programs = 10
        report.corpus.step_count_distribution = {1: 5, 2: 5}
        report.weaknesses = ["test weakness"]
        report.recommendations = ["test rec"]
        md = format_report(report)
        assert "## 1. Verified Program Corpus" in md
        assert "## 5. Weaknesses" in md
        assert "## 6. Recommendations" in md
        assert "## 7. Verdict" in md
        assert "test weakness" in md
        assert "test rec" in md


class TestRunAudit:
    def test_end_to_end(self, results_dir: Path) -> None:
        _write_program_store(results_dir / "program_store.json", [
            {
                "program": "let v0: GRID = fill_enclosed(input, 4)\n-> v0",
                "task_ids": ["t1"],
                "sources": ["solve-report:r.json"],
                "use_count": 1,
                "signatures": [],
            },
        ])
        _write_solve_report(results_dir / "v1-train.json", [
            {"task_id": "t1", "solved": True, "rounds": 1},
        ])
        _write_solve_report(results_dir / "v1-eval.json", [])
        _write_library(results_dir / "library.json", [])

        report = run_audit(results_dir)
        assert report.corpus.total_programs == 1
        assert report.train_solve.solved_tasks == 1
        assert report.eval_solve.total_tasks == 0
        assert len(report.weaknesses) > 0
        assert len(report.recommendations) > 0

        md = format_report(report)
        assert "Do not train yet" in md
