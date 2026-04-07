"""Tests for the structural gates evaluation harness.

Tests schema parsing, scorer logic, and report generation.
Does NOT require ARC data files — uses synthetic fixtures.
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from aria.eval.structural_gates_schema import (
    DecompLabel,
    EntityKind,
    GoldEntity,
    GoldRelation,
    GoldTask,
    RelationKind,
    TemplateFamily,
    load_gold_tasks,
    load_gold_tasks_map,
)
from aria.eval.structural_gates_scorer import (
    GATE_ORDER,
    GateResult,
    TaskGateResults,
    score_all_gates,
    score_decomposition_gate,
    score_entity_gate,
    score_executor_gate,
    score_relation_gate,
    score_slot_gate,
    score_template_gate,
)
from aria.eval.structural_gates_trace import (
    InducedEntity,
    InducedRelation,
    StageArtifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gold(
    task_id: str = "test_task",
    decomposition: str = "object",
    template: str = "match_recolor",
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
    critical_slots: dict | None = None,
) -> GoldTask:
    ents = []
    if entities:
        for e in entities:
            ents.append(GoldEntity(
                name=e["name"],
                kind=EntityKind(e["kind"]),
                selector_note=e.get("selector_note", ""),
            ))
    rels = []
    if relations:
        for r in relations:
            rels.append(GoldRelation(
                kind=RelationKind(r["kind"]),
                source=r["source"],
                target=r["target"],
            ))
    return GoldTask(
        task_id=task_id,
        decomposition=DecompLabel(decomposition),
        entities=tuple(ents),
        relations=tuple(rels),
        template=TemplateFamily(template),
        critical_slots=critical_slots or {},
    )


def _make_artifacts(
    task_id: str = "test_task",
    decomp_hyps: list[str] | None = None,
    entities: list[InducedEntity] | None = None,
    relations: list[InducedRelation] | None = None,
    template_hyps: list[str] | None = None,
    slot_candidates: dict | None = None,
    executor_ran: bool = False,
    executor_attempted: bool = False,
) -> StageArtifacts:
    a = StageArtifacts(task_id=task_id)
    a.decomposition_hypotheses = decomp_hyps or []
    a.entities = entities or []
    a.relations = relations or []
    a.template_hypotheses = template_hyps or []
    a.slot_candidates = slot_candidates or {}
    a.executor_ran = executor_ran
    a.executor_attempted = executor_attempted
    return a


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_enum_values(self):
        assert DecompLabel.OBJECT.value == "object"
        assert EntityKind.MARKER.value == "marker"
        assert RelationKind.CONTAINS.value == "contains"
        assert TemplateFamily.MATCH_RECOLOR.value == "match_recolor"

    def test_gold_entity(self):
        e = GoldEntity(name="obj1", kind=EntityKind.OBJECT)
        assert e.matches_kind("object")
        assert not e.matches_kind("marker")

    def test_gold_task_entity_names(self):
        g = _make_gold(entities=[
            {"name": "a", "kind": "object"},
            {"name": "b", "kind": "marker"},
        ])
        assert g.entity_names == frozenset({"a", "b"})

    def test_yaml_load(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            tasks:
              - task_id: test_001
                decomposition: object
                entities:
                  - name: obj_a
                    kind: object
                    selector_note: "test obj"
                  - name: marker_a
                    kind: marker
                relations:
                  - kind: paired_with
                    source: marker_a
                    target: obj_a
                template: match_recolor
                critical_slots:
                  marker_set: "test"
        """)
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        tasks = load_gold_tasks(p)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == "test_001"
        assert t.decomposition == DecompLabel.OBJECT
        assert len(t.entities) == 2
        assert t.entities[0].kind == EntityKind.OBJECT
        assert len(t.relations) == 1
        assert t.relations[0].kind == RelationKind.PAIRED_WITH
        assert t.template == TemplateFamily.MATCH_RECOLOR
        assert "marker_set" in t.critical_slots

    def test_yaml_load_map(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            tasks:
              - task_id: a
                decomposition: frame
                entities: []
                relations: []
                template: region_fill
              - task_id: b
                decomposition: panel
                entities: []
                relations: []
                template: panel_combine_rewrite
        """)
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        m = load_gold_tasks_map(p)
        assert set(m.keys()) == {"a", "b"}
        assert m["a"].template == TemplateFamily.REGION_FILL


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestDecompositionGate:
    def test_pass_exact_match(self):
        gold = _make_gold(decomposition="object")
        art = _make_artifacts(decomp_hyps=["object", "frame"])
        r = score_decomposition_gate(gold, art)
        assert r.passed
        assert r.score == 1.0
        assert r.details["rank"] == 1

    def test_pass_alias(self):
        gold = _make_gold(decomposition="host_slot")
        art = _make_artifacts(decomp_hyps=["composites", "object"])
        r = score_decomposition_gate(gold, art)
        assert r.passed

    def test_fail_not_present(self):
        gold = _make_gold(decomposition="partition")
        art = _make_artifacts(decomp_hyps=["object", "frame"])
        r = score_decomposition_gate(gold, art)
        assert not r.passed
        assert r.score == 0.0

    def test_empty_hypotheses(self):
        gold = _make_gold(decomposition="object")
        art = _make_artifacts(decomp_hyps=[])
        r = score_decomposition_gate(gold, art)
        assert not r.passed


class TestEntityGate:
    def test_pass_all_matched(self):
        gold = _make_gold(entities=[
            {"name": "obj", "kind": "object"},
            {"name": "mrk", "kind": "marker"},
        ])
        art = _make_artifacts(entities=[
            InducedEntity(id="e0", kind="object", bbox=(0, 0, 3, 3), size=9),
            InducedEntity(id="e1", kind="marker", bbox=(5, 5, 1, 1), size=1),
        ])
        r = score_entity_gate(gold, art)
        assert r.passed
        assert r.score == 1.0

    def test_partial_match(self):
        gold = _make_gold(entities=[
            {"name": "obj", "kind": "object"},
            {"name": "host", "kind": "host"},
        ])
        art = _make_artifacts(entities=[
            InducedEntity(id="e0", kind="object", bbox=(0, 0, 3, 3), size=9),
        ])
        r = score_entity_gate(gold, art)
        # host matches object via alias, so both match
        assert r.passed

    def test_fail_no_entities(self):
        gold = _make_gold(entities=[
            {"name": "obj", "kind": "panel"},
        ])
        art = _make_artifacts(entities=[])
        r = score_entity_gate(gold, art)
        assert not r.passed

    def test_no_gold_entities(self):
        gold = _make_gold(entities=[])
        art = _make_artifacts()
        r = score_entity_gate(gold, art)
        assert r.passed


class TestRelationGate:
    def test_pass(self):
        gold = _make_gold(relations=[
            {"kind": "adjacent_to", "source": "a", "target": "b"},
        ])
        art = _make_artifacts(relations=[
            InducedRelation(kind="adjacent_to", source_id="e0", target_id="e1"),
        ])
        r = score_relation_gate(gold, art)
        assert r.passed

    def test_fail(self):
        gold = _make_gold(relations=[
            {"kind": "host_of", "source": "a", "target": "b"},
        ])
        art = _make_artifacts(relations=[
            InducedRelation(kind="aligned_row", source_id="e0", target_id="e1"),
        ])
        r = score_relation_gate(gold, art)
        assert not r.passed

    def test_alias(self):
        gold = _make_gold(relations=[
            {"kind": "paired_with", "source": "a", "target": "b"},
        ])
        art = _make_artifacts(relations=[
            InducedRelation(kind="adjacent_to", source_id="e0", target_id="e1"),
        ])
        r = score_relation_gate(gold, art)
        assert r.passed  # adjacent_to is alias for paired_with


class TestTemplateGate:
    def test_pass_exact(self):
        gold = _make_gold(template="match_recolor")
        art = _make_artifacts(template_hyps=["match_recolor", "swap"])
        r = score_template_gate(gold, art)
        assert r.passed
        assert r.details["rank"] == 1

    def test_pass_alias(self):
        gold = _make_gold(template="region_fill")
        art = _make_artifacts(template_hyps=["periodic_repair"])
        r = score_template_gate(gold, art)
        assert r.passed

    def test_fail(self):
        gold = _make_gold(template="swap")
        art = _make_artifacts(template_hyps=["match_recolor", "extract_modify"])
        r = score_template_gate(gold, art)
        assert not r.passed


class TestSlotGate:
    def test_pass_direct(self):
        gold = _make_gold(critical_slots={"marker_set": "x"})
        art = _make_artifacts(
            entities=[InducedEntity(id="e0", kind="marker", bbox=(0, 0, 1, 1))],
        )
        r = score_slot_gate(gold, art)
        assert r.passed

    def test_pass_host_heuristic(self):
        gold = _make_gold(critical_slots={"host_set": "x"})
        art = _make_artifacts(
            entities=[InducedEntity(id="e0", kind="object", bbox=(0, 0, 5, 5), size=10)],
        )
        r = score_slot_gate(gold, art)
        assert r.passed

    def test_fail_missing(self):
        gold = _make_gold(critical_slots={"axis": "row", "color_role": "x"})
        art = _make_artifacts()
        r = score_slot_gate(gold, art)
        assert not r.passed

    def test_no_slots(self):
        gold = _make_gold(critical_slots={})
        art = _make_artifacts()
        r = score_slot_gate(gold, art)
        assert r.passed


class TestExecutorGate:
    def test_pass(self):
        gold = _make_gold(template="match_recolor")
        art = _make_artifacts(
            template_hyps=["match_recolor"],
            executor_attempted=True,
            executor_ran=True,
        )
        r = score_executor_gate(gold, art)
        assert r.passed

    def test_fail_not_attempted(self):
        gold = _make_gold(template="match_recolor")
        art = _make_artifacts()
        r = score_executor_gate(gold, art)
        assert not r.passed

    def test_executor_gap(self):
        gold = _make_gold(template="match_recolor")
        art = _make_artifacts(
            template_hyps=["match_recolor"],
            executor_attempted=True,
            executor_ran=False,
        )
        r = score_executor_gate(gold, art)
        assert not r.passed
        assert r.details["diagnosis"] == "executor_gap"


class TestAggregateScoring:
    def test_all_gates_scored(self):
        gold = _make_gold(
            entities=[{"name": "a", "kind": "object"}],
            relations=[{"kind": "adjacent_to", "source": "a", "target": "b"}],
            critical_slots={"marker_set": "x"},
        )
        art = _make_artifacts(
            decomp_hyps=["object"],
            entities=[InducedEntity(id="e0", kind="object", bbox=(0, 0, 3, 3))],
            relations=[InducedRelation(kind="adjacent_to", source_id="e0", target_id="e1")],
            template_hyps=["match_recolor"],
            executor_attempted=True,
            executor_ran=True,
        )
        results = score_all_gates(gold, art)
        assert len(results.gates) == 6
        gate_names = [g.gate_name for g in results.gates]
        assert gate_names == GATE_ORDER

    def test_first_failing_gate(self):
        gold = _make_gold(
            entities=[{"name": "a", "kind": "panel"}],
        )
        art = _make_artifacts(
            decomp_hyps=["object"],
            entities=[],  # will fail entity gate
        )
        results = score_all_gates(gold, art)
        assert results.first_failing_gate == "entity"


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReport:
    def test_text_report(self):
        from aria.eval.structural_gates_report import format_text_report
        from aria.eval.structural_gates_runner import RunReport, TaskRunResult

        gold = _make_gold()
        art = _make_artifacts(decomp_hyps=["object"], template_hyps=["match_recolor"])
        gate_results = score_all_gates(gold, art)

        report = RunReport(
            results=[TaskRunResult(
                task_id="test_task",
                gate_results=gate_results,
                artifacts=art,
                exact_solve=False,
                elapsed_sec=1.0,
            )],
            dataset="test",
            top_k=5,
            elapsed_total_sec=1.0,
        )
        text = format_text_report(report)
        assert "STRUCTURAL GATES" in text
        assert "test_task" in text

    def test_json_report(self):
        from aria.eval.structural_gates_report import format_json_report
        from aria.eval.structural_gates_runner import RunReport, TaskRunResult

        gold = _make_gold()
        art = _make_artifacts(decomp_hyps=["object"], template_hyps=["match_recolor"])
        gate_results = score_all_gates(gold, art)

        report = RunReport(
            results=[TaskRunResult(
                task_id="test_task",
                gate_results=gate_results,
                artifacts=art,
                exact_solve=False,
                elapsed_sec=1.0,
            )],
            dataset="test",
            top_k=5,
            elapsed_total_sec=1.0,
        )
        j = format_json_report(report)
        assert "aggregate" in j
        assert "per_task" in j
        assert j["meta"]["n_tasks"] == 1
        assert "decomposition_pass_rate" in j["aggregate"]
