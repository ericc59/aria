"""Six structural gate scorers.

Each gate answers a specific question about the solver's structural
induction quality. Gates are scored independently and aggregated.

Gate 1: Decomposition — did the gold decomposition appear in top-K?
Gate 2: Entity — did required gold entities appear in induced set?
Gate 3: Relation — did required gold relations appear between matched entities?
Gate 4: Template — did the gold template family appear in top-K?
Gate 5: Slot — did critical slot values appear in candidate proposals?
Gate 6: Executor — can the current executor express/run the gold template?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aria.eval.structural_gates_schema import (
    DecompLabel,
    EntityKind,
    GoldEntity,
    GoldRelation,
    GoldTask,
    RelationKind,
    TemplateFamily,
)
from aria.eval.structural_gates_trace import InducedEntity, InducedRelation, StageArtifacts


# ---------------------------------------------------------------------------
# Gate result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    passed: bool
    score: float  # 0.0-1.0
    details: dict[str, Any]


@dataclass(frozen=True)
class TaskGateResults:
    task_id: str
    gates: tuple[GateResult, ...]
    exact_solve: bool = False

    @property
    def first_failing_gate(self) -> str | None:
        for g in self.gates:
            if not g.passed:
                return g.gate_name
        return None

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def gate_by_name(self, name: str) -> GateResult | None:
        for g in self.gates:
            if g.gate_name == name:
                return g
        return None


# ---------------------------------------------------------------------------
# Decomposition label mapping
# ---------------------------------------------------------------------------

# Maps system decomposition hypothesis strings to our gold labels
_DECOMP_ALIAS: dict[str, str] = {
    "object": "object",
    "frame": "frame",
    "panel": "panel",
    "partition": "partition",
    "region": "region",
    "host_slot": "host_slot",
    # System-specific names
    "framed_regions": "frame",
    "composites": "host_slot",
    "composite": "host_slot",
}


def _normalize_decomp(hyp: str) -> str:
    return _DECOMP_ALIAS.get(hyp.lower().strip(), hyp.lower().strip())


# ---------------------------------------------------------------------------
# Gate 1: Decomposition
# ---------------------------------------------------------------------------


def score_decomposition_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
    top_k: int = 5,
) -> GateResult:
    """Did the gold decomposition label appear in the top-K hypotheses?"""
    gold_label = gold.decomposition.value
    hypotheses = artifacts.decomposition_hypotheses[:top_k]
    normalized = [_normalize_decomp(h) for h in hypotheses]

    found = gold_label in normalized
    rank = normalized.index(gold_label) + 1 if found else -1

    return GateResult(
        gate_name="decomposition",
        passed=found,
        score=1.0 if found else 0.0,
        details={
            "gold": gold_label,
            "hypotheses": hypotheses,
            "normalized": normalized,
            "rank": rank,
            "top_k": top_k,
        },
    )


# ---------------------------------------------------------------------------
# Gate 2: Entity
# ---------------------------------------------------------------------------


def _entity_kind_match(gold_kind: EntityKind, induced_kind: str) -> bool:
    """Coarse kind matching with aliases."""
    gk = gold_kind.value
    ik = induced_kind.lower().strip()

    if gk == ik:
        return True

    # Aliases
    aliases: dict[str, set[str]] = {
        "object": {"object", "non_singleton"},
        "marker": {"marker", "singleton"},
        "host": {"host", "object"},  # hosts are large objects
        "gap": {"gap", "marker", "singleton"},  # gaps are often singletons
        "panel": {"panel", "region", "frame"},
        "region": {"region", "frame"},
    }
    return ik in aliases.get(gk, set())


def _entity_overlap(
    gold: GoldEntity,
    induced: InducedEntity,
) -> bool:
    """Check if a gold entity matches an induced entity.

    Uses kind matching. BBox overlap would require gold bboxes,
    which we don't annotate — so we use kind + color heuristics.
    """
    if not _entity_kind_match(gold.kind, induced.kind):
        return False
    return True


def score_entity_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
) -> GateResult:
    """Did the required gold entities appear in the induced entity set?"""
    if not gold.entities:
        return GateResult(
            gate_name="entity",
            passed=True,
            score=1.0,
            details={"gold_count": 0, "note": "no gold entities to match"},
        )

    matched: list[str] = []
    unmatched: list[str] = []

    for ge in gold.entities:
        found = any(_entity_overlap(ge, ie) for ie in artifacts.entities)
        if found:
            matched.append(ge.name)
        else:
            unmatched.append(ge.name)

    recall = len(matched) / len(gold.entities) if gold.entities else 1.0
    passed = recall >= 0.5  # require at least half the gold entities

    return GateResult(
        gate_name="entity",
        passed=passed,
        score=recall,
        details={
            "gold_count": len(gold.entities),
            "matched": matched,
            "unmatched": unmatched,
            "induced_count": len(artifacts.entities),
            "recall": recall,
        },
    )


# ---------------------------------------------------------------------------
# Gate 3: Relation
# ---------------------------------------------------------------------------


def _relation_kind_match(gold_kind: RelationKind, induced_kind: str) -> bool:
    gk = gold_kind.value
    ik = induced_kind.lower().strip()
    if gk == ik:
        return True
    aliases: dict[str, set[str]] = {
        "contains": {"contains", "encloses"},
        "paired_with": {"paired_with", "adjacent_to", "corresponds_to"},
        "aligned_row": {"aligned_row", "aligned_with"},
        "aligned_col": {"aligned_col", "aligned_with"},
        "host_of": {"host_of", "contains", "encloses"},
        "gap_of": {"gap_of", "contains"},
        "adjacent_to": {"adjacent_to", "paired_with"},
    }
    return ik in aliases.get(gk, set())


def score_relation_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
) -> GateResult:
    """Did required gold relations appear between matched entities?"""
    if not gold.relations:
        return GateResult(
            gate_name="relation",
            passed=True,
            score=1.0,
            details={"gold_count": 0, "note": "no gold relations to match"},
        )

    matched: list[str] = []
    unmatched: list[str] = []

    for gr in gold.relations:
        found = any(
            _relation_kind_match(gr.kind, ir.kind)
            for ir in artifacts.relations
        )
        label = f"{gr.source}-{gr.kind.value}->{gr.target}"
        if found:
            matched.append(label)
        else:
            unmatched.append(label)

    recall = len(matched) / len(gold.relations)
    passed = recall >= 0.5

    return GateResult(
        gate_name="relation",
        passed=passed,
        score=recall,
        details={
            "gold_count": len(gold.relations),
            "matched": matched,
            "unmatched": unmatched,
            "induced_count": len(artifacts.relations),
            "recall": recall,
        },
    )


# ---------------------------------------------------------------------------
# Gate 4: Template
# ---------------------------------------------------------------------------

_TEMPLATE_ALIAS: dict[str, str] = {
    "match_recolor": "match_recolor",
    "host_slot_place": "host_slot_place",
    "extract_modify": "extract_modify",
    "panel_combine_rewrite": "panel_combine_rewrite",
    "region_fill": "region_fill",
    "swap": "swap",
    # System-specific
    "recolor": "match_recolor",
    "correspondence": "match_recolor",
    "periodic_repair": "region_fill",
    "repair": "region_fill",
    "tile": "extract_modify",
    "canvas": "extract_modify",
    "crop": "extract_modify",
    "movement": "swap",
    "composite_role_alignment": "match_recolor",
    "framed_periodic_repair": "region_fill",
}


def _normalize_template(hyp: str) -> str:
    return _TEMPLATE_ALIAS.get(hyp.lower().strip(), hyp.lower().strip())


def score_template_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
    top_k: int = 5,
) -> GateResult:
    """Did the gold template family appear in top-K template hypotheses?"""
    gold_label = gold.template.value
    hypotheses = artifacts.template_hypotheses[:top_k]
    normalized = [_normalize_template(h) for h in hypotheses]

    found = gold_label in normalized
    rank = normalized.index(gold_label) + 1 if found else -1

    return GateResult(
        gate_name="template",
        passed=found,
        score=1.0 if found else 0.0,
        details={
            "gold": gold_label,
            "hypotheses": hypotheses,
            "normalized": normalized,
            "rank": rank,
            "top_k": top_k,
        },
    )


# ---------------------------------------------------------------------------
# Gate 5: Slot
# ---------------------------------------------------------------------------


def score_slot_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
) -> GateResult:
    """Did critical slot values/types appear in candidate proposals?

    Coarse check: for each critical slot name in the gold annotation,
    check if the system produced any candidates for that slot family.
    """
    if not gold.critical_slots:
        return GateResult(
            gate_name="slot",
            passed=True,
            score=1.0,
            details={"gold_count": 0, "note": "no critical slots annotated"},
        )

    matched: list[str] = []
    unmatched: list[str] = []

    system_slot_names = set(artifacts.slot_candidates.keys())
    # Also check entity kinds as proxy for slot presence
    entity_kinds = {e.kind for e in artifacts.entities}
    entity_colors = {e.color for e in artifacts.entities if e.color >= 0}

    for slot_name, slot_value in gold.critical_slots.items():
        found = False
        # Direct slot match
        if slot_name in system_slot_names:
            found = True
        # Heuristic: marker_set present if markers found
        elif slot_name == "marker_set" and "marker" in entity_kinds:
            found = True
        elif slot_name == "host_set" and ("host" in entity_kinds or any(
            e.size > 4 for e in artifacts.entities
        )):
            found = True
        elif slot_name == "gap_set" and ("gap" in entity_kinds or "marker" in entity_kinds):
            found = True
        elif slot_name == "color_role" and "color_role" in system_slot_names:
            found = True
        elif slot_name == "axis":
            # Check if any decomposition or relation implies axis awareness
            has_axis = any(
                r.kind in ("aligned_row", "aligned_col")
                for r in artifacts.relations
            )
            found = has_axis
        elif slot_name == "assignment_rule":
            # Coarse: check if correspondence relations exist
            found = any(
                r.kind in ("paired_with", "adjacent_to", "contains")
                for r in artifacts.relations
            )

        if found:
            matched.append(slot_name)
        else:
            unmatched.append(slot_name)

    recall = len(matched) / len(gold.critical_slots)
    passed = recall >= 0.5

    return GateResult(
        gate_name="slot",
        passed=passed,
        score=recall,
        details={
            "gold_count": len(gold.critical_slots),
            "matched": matched,
            "unmatched": unmatched,
            "system_slots": list(system_slot_names),
            "recall": recall,
        },
    )


# ---------------------------------------------------------------------------
# Gate 6: Executor
# ---------------------------------------------------------------------------


def score_executor_gate(
    gold: GoldTask,
    artifacts: StageArtifacts,
) -> GateResult:
    """Can the executor produce a runnable candidate for this task?

    Gate 6 semantics: "a runnable executable candidate exists."
    Passed = at least one active solver path produced a candidate that
    executed without error. This is distinct from exact solve — a
    candidate that runs but gives wrong output still passes Gate 6.

    Path attribution is recorded for diagnosis:
    - executor_paths_produced: paths that generated candidates
    - executor_paths_ran: paths whose candidates executed without error
    - executor_paths_verified: paths whose candidates verified on all demos
    """
    template_found = gold.template.value in [
        _normalize_template(h) for h in artifacts.template_hypotheses
    ]

    if not artifacts.executor_attempted:
        return GateResult(
            gate_name="executor",
            passed=False,
            score=0.0,
            details={
                "template_found": template_found,
                "executor_attempted": False,
                "executor_ran": False,
                "note": "no executor attempt",
            },
        )

    passed = artifacts.executor_ran
    score = 1.0 if passed else 0.0

    # Diagnosis based on path attribution
    if artifacts.executor_paths_verified:
        diagnosis = "verified"
    elif artifacts.executor_paths_ran:
        diagnosis = "ran_not_verified"
    elif artifacts.executor_paths_produced:
        diagnosis = "produced_not_ran"
    elif template_found:
        diagnosis = "executor_gap"
    else:
        diagnosis = "induction_failure"

    return GateResult(
        gate_name="executor",
        passed=passed,
        score=score,
        details={
            "template_found": template_found,
            "executor_attempted": True,
            "executor_ran": artifacts.executor_ran,
            "executor_path": artifacts.executor_path,
            "executor_paths_tried": artifacts.executor_paths_tried,
            "executor_paths_produced": artifacts.executor_paths_produced,
            "executor_paths_ran": artifacts.executor_paths_ran,
            "executor_paths_verified": artifacts.executor_paths_verified,
            "executor_error": artifacts.executor_error,
            "diagnosis": diagnosis,
        },
    )


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


GATE_ORDER = [
    "decomposition",
    "entity",
    "relation",
    "template",
    "slot",
    "executor",
]


def score_all_gates(
    gold: GoldTask,
    artifacts: StageArtifacts,
    top_k: int = 5,
) -> TaskGateResults:
    """Score all six structural gates for one task."""
    gates = (
        score_decomposition_gate(gold, artifacts, top_k=top_k),
        score_entity_gate(gold, artifacts),
        score_relation_gate(gold, artifacts),
        score_template_gate(gold, artifacts, top_k=top_k),
        score_slot_gate(gold, artifacts),
        score_executor_gate(gold, artifacts),
    )
    return TaskGateResults(
        task_id=gold.task_id,
        gates=gates,
    )
