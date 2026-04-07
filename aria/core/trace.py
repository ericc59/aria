"""Single-task trace capture for canonical pipeline debugging.

Captures a JSON-serializable trace of everything the canonical system
does on one task: seeds proposed, graph states, edits applied,
compile/verify results, diff stats, and scoring over time.

Usage:
    trace = TaskTrace(task_id="abc", demos=demos)
    trace.add_seed(seed)
    trace.add_event("compile", {...})
    trace.to_dict()   # JSON-serializable dict
    trace.to_json()   # JSON string
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.core.graph import (
    CompileSuccess,
    CompileFailure,
    CompileResult,
    ComputationGraph,
    Specialization,
)
from aria.core.editor_env import EditAction, EditState


# ---------------------------------------------------------------------------
# Grid serialization
# ---------------------------------------------------------------------------


def _grid_to_list(grid: Any) -> list[list[int]]:
    """Convert a numpy grid to a JSON-safe nested list."""
    if hasattr(grid, "tolist"):
        return grid.tolist()
    return [[int(c) for c in row] for row in grid]


# ---------------------------------------------------------------------------
# Graph / Spec serialization
# ---------------------------------------------------------------------------


def _graph_to_dict(graph: ComputationGraph) -> dict:
    nodes = {}
    for nid, node in graph.nodes.items():
        nodes[nid] = {
            "op": node.op,
            "inputs": list(node.inputs),
            "slots": [
                {"name": s.name, "typ": s.typ, "evidence": _safe_val(s.evidence)}
                for s in node.slots
            ],
            "roles": [
                {"name": r.name, "kind": r.kind}
                for r in node.roles
            ],
            "evidence": {k: _safe_val(v) for k, v in node.evidence.items()},
        }
    return {
        "task_id": graph.task_id,
        "nodes": nodes,
        "output_id": graph.output_id,
        "description": graph.description,
    }


def _spec_to_dict(spec: Specialization | None) -> dict | None:
    if spec is None:
        return None
    return {
        "task_id": spec.task_id,
        "bindings": [
            {"node_id": b.node_id, "name": b.name,
             "value": _safe_val(b.value), "source": b.source}
            for b in spec.bindings
        ],
    }


def _action_to_dict(action: EditAction) -> dict:
    return {
        "type": action.action_type.name,
        "node_id": action.node_id,
        "target_id": action.target_id,
        "key": action.key,
        "value": _safe_val(action.value),
    }


def _diagnostic_to_dict(diag: Any) -> dict | None:
    """Serialize a VerifyDiagnostic to a JSON-safe dict."""
    if diag is None:
        return None
    d: dict = {
        "per_demo_diff": list(diag.per_demo_diff) if diag.per_demo_diff else [],
        "total_diff": diag.total_diff,
        "failed_demo": diag.failed_demo,
        "diff_fraction": round(diag.diff_fraction, 4),
        "repair_hints": [
            {
                "node_id": h.node_id,
                "binding_name": h.binding_name,
                "current_value": _safe_val(h.current_value),
                "alternatives": [_safe_val(a) for a in h.alternatives],
                "confidence": round(h.confidence, 3),
                "reason": h.reason,
            }
            for h in (diag.repair_hints or ())
        ],
        "blamed_bindings": list(diag.blamed_bindings or ()),
        "description": diag.description,
    }
    # Region residuals
    regions = getattr(diag, 'region_residuals', ())
    if regions:
        d["region_residuals"] = [
            {"label": r.region_label, "diff_pixels": r.diff_pixels,
             "total_pixels": r.total_pixels,
             "diff_fraction": round(r.diff_fraction, 4)}
            for r in regions
        ]
    # Subgraph blame
    blames = getattr(diag, 'subgraph_blames', ())
    if blames:
        d["subgraph_blames"] = [
            {"node_ids": list(b.node_ids), "ops": list(b.ops),
             "residual_overlap": round(b.residual_overlap, 3),
             "confidence": round(b.confidence, 3),
             "replacement_labels": list(b.replacement_labels),
             "reason": b.reason}
            for b in blames
        ]
    # Replacement fragments
    frags = getattr(diag, 'replacement_fragments', ())
    if frags:
        d["replacement_fragments"] = [
            {"label": f.label, "description": f.description,
             "nodes": list(f.nodes.keys())}
            for f in frags
        ]
    return d


def _compile_result_to_dict(result: CompileResult | None) -> dict | None:
    if result is None:
        return None
    if isinstance(result, CompileSuccess):
        return {"status": "success", "description": result.description, "scope": result.scope}
    if isinstance(result, CompileFailure):
        d: dict = {"status": "failure", "reason": result.reason,
                    "missing_ops": list(result.missing_ops)}
        if result.diagnostic is not None:
            d["diagnostic"] = _diagnostic_to_dict(result.diagnostic)
        return d
    return {"status": "unknown"}


def _safe_val(v: Any) -> Any:
    """Make a value JSON-safe."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_safe_val(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_val(val) for k, val in v.items()}
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Trace model
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """One event in the trace timeline."""
    timestamp: float
    phase: str          # "seed", "deterministic", "learned", "compile", "verify"
    event_type: str     # "seed_added", "edit", "compile", "verify", "score", etc.
    data: dict = field(default_factory=dict)


@dataclass
class TaskTrace:
    """Full trace of a single task through the canonical pipeline."""
    task_id: str
    demos: list[dict] = field(default_factory=list)  # [{input, output}]
    seeds: list[dict] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    # Final results
    static_result: dict = field(default_factory=dict)
    deterministic_result: dict = field(default_factory=dict)
    learned_result: dict = field(default_factory=dict)
    solved: bool = False
    solver: str = ""

    def set_demos(self, demo_pairs: Sequence[Any]) -> None:
        self.demos = []
        for d in demo_pairs:
            self.demos.append({
                "input": _grid_to_list(d.input),
                "output": _grid_to_list(d.output),
            })

    def add_seed(self, provenance: str, graph: ComputationGraph,
                 spec: Specialization | None, verified: bool) -> None:
        self.seeds.append({
            "provenance": provenance,
            "graph": _graph_to_dict(graph),
            "specialization": _spec_to_dict(spec),
            "verified": verified,
        })

    def add_event(self, phase: str, event_type: str, **data: Any) -> None:
        self.events.append(TraceEvent(
            timestamp=time.time(),
            phase=phase,
            event_type=event_type,
            data={k: _safe_val(v) for k, v in data.items()},
        ))

    def add_state_snapshot(self, phase: str, state: EditState,
                           label: str = "") -> None:
        self.add_event(phase, "state_snapshot",
            label=label,
            graph=_graph_to_dict(state.graph),
            specialization=_spec_to_dict(state.specialization),
            compile_result=_compile_result_to_dict(state.compile_result),
            verified=state.verified,
            diff_pixels=state.diff_pixels,
            step=state.step,
            score=state.score,
            n_actions=len(state.history),
        )

    def add_action(self, phase: str, action: EditAction,
                   depth: int = 0) -> None:
        self.add_event(phase, "action",
            action=_action_to_dict(action),
            depth=depth,
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "demos": self.demos,
            "seeds": self.seeds,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "phase": e.phase,
                    "event_type": e.event_type,
                    "data": e.data,
                }
                for e in self.events
            ],
            "static_result": self.static_result,
            "deterministic_result": self.deterministic_result,
            "learned_result": self.learned_result,
            "solved": self.solved,
            "solver": self.solver,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
