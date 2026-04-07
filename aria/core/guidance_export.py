"""Guidance data export — per-task JSONL records for future guided search.

Runs the canonical perception/stage-1/evidence/solve pipeline on a task
and emits a JSON-serializable record containing all intermediate structures
that a future learned guidance model would need.

Does not change solver semantics. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from aria.types import DemoPair


SCHEMA_VERSION = 1

# Required top-level keys in every exported record.
REQUIRED_KEYS = frozenset({
    "schema_version",
    "task_id",
    "train_demos",
    "size_spec",
    "derivation_spec",
    "render_spec",
    "perception_summaries",
    "roles",
    "legend",
    "zones",
    "slot_grid",
    "correspondences",
    "program_family",
    "program_skeleton",
    "verify_result",
    "residuals",
    "mechanism_evidence",
    "lane_ranking",
    "search_trace",
    "inner_trace",
    "deep_trace",
    "spec_trace",
})


# ---------------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------------


def _safe(v: Any) -> Any:
    """Make a value JSON-safe."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, frozenset):
        return sorted(_safe(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


def _grid_to_list(grid: Any) -> list[list[int]]:
    if hasattr(grid, "tolist"):
        return grid.tolist()
    return [[int(c) for c in row] for row in grid]


# ---------------------------------------------------------------------------
# Perception summary
# ---------------------------------------------------------------------------


def _perception_summary(state: Any) -> dict:
    """Extract a compact perception summary from a GridPerceptionState."""
    partition_info = None
    if state.partition is not None:
        p = state.partition
        partition_info = {
            "n_rows": p.n_rows,
            "n_cols": p.n_cols,
            "separator_color": int(p.separator_color),
            "is_uniform": p.is_uniform_partition,
        }

    return {
        "dims": list(state.dims),
        "bg_color": int(state.bg_color),
        "palette": sorted(int(c) for c in state.palette),
        "non_bg_colors": sorted(int(c) for c in state.non_bg_colors),
        "n_objects_4": len(state.objects.objects),
        "n_objects_8": len(state.objects8.objects),
        "n_framed_regions": len(state.framed_regions),
        "n_boxed_regions": len(state.boxed_regions),
        "n_zones": len(state.zones),
        "partition": partition_info,
        "has_legend": state.legend is not None,
    }


# ---------------------------------------------------------------------------
# Relation layer extraction
# ---------------------------------------------------------------------------


def _extract_legend(state: Any) -> dict | None:
    from aria.core.relations import build_legend_mapping

    lm = build_legend_mapping(state)
    if lm is None:
        return None
    return {
        "edge": lm.edge,
        "key_to_value": {int(k): int(v) for k, v in lm.key_to_value.items()},
        "bbox": list(lm.legend_bbox),
        "n_entries": len(lm.key_to_value),
    }


def _extract_slot_grid(state: Any) -> dict | None:
    from aria.core.relations import detect_slot_grid

    sg = detect_slot_grid(state)
    if sg is None:
        return None
    return {
        "n_rows": sg.n_rows,
        "n_cols": sg.n_cols,
        "slot_height": sg.slot_height,
        "slot_width": sg.slot_width,
        "row_stride": sg.row_stride,
        "col_stride": sg.col_stride,
    }


def _extract_roles(state: Any) -> list[dict]:
    return [
        {"role": r.role.name, "color": r.color}
        for r in state.roles
    ]


def _extract_zones(state: Any) -> list[dict]:
    return [
        {"x": z.x, "y": z.y, "w": z.w, "h": z.h}
        for z in state.zones
    ]


def _extract_correspondences(state: Any, demos: Sequence[DemoPair]) -> list[dict] | None:
    """Try to extract zone/partition correspondences across demos."""
    from aria.core.relations import verify_zone_summary_grid

    zsm = verify_zone_summary_grid(demos)
    if zsm is None:
        return None
    return [
        {
            "mapping_kind": zsm.mapping_kind,
            "zone_order": list(zsm.zone_order),
            "params": _safe(dict(zsm.params)),
        }
    ]


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------


def export_task(
    task_id: str,
    demos: tuple[DemoPair, ...],
    *,
    include_search_trace: bool = False,
    include_inner_trace: bool = False,
    include_deep_trace: bool = False,
    include_spec_trace: bool = False,
) -> dict[str, Any]:
    """Export a single task's guidance record.

    Runs perception, stage-1, mechanism evidence, and the canonical solve
    pipeline. Returns a JSON-serializable dict.
    """
    from aria.core.grid_perception import perceive_grid
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.output_stage1 import infer_output_stage1_spec
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.core.graph import CompileSuccess
    from aria.core.trace import _graph_to_dict

    # Train demo grids
    train_grids = [
        {"input": _grid_to_list(d.input), "output": _grid_to_list(d.output)}
        for d in demos
    ]

    # Perception on each input demo
    perception_summaries = []
    first_state = None
    for d in demos:
        state = perceive_grid(d.input)
        if first_state is None:
            first_state = state
        perception_summaries.append(_perception_summary(state))

    # Roles, legend, zones, slot grid from first demo
    roles: list[dict] = []
    legend: dict | None = None
    zones: list[dict] = []
    slot_grid: dict | None = None
    correspondences: list[dict] | None = None

    if first_state is not None:
        roles = _extract_roles(first_state)
        legend = _extract_legend(first_state)
        zones = _extract_zones(first_state)
        slot_grid = _extract_slot_grid(first_state)
        correspondences = _extract_correspondences(first_state, demos)

    # Stage-1
    stage1 = infer_output_stage1_spec(demos)
    size_spec = None
    derivation_spec = None
    render_spec = None
    if stage1 is not None:
        size_spec = _safe(asdict(stage1.size_spec))
        if stage1.derivation_spec is not None:
            derivation_spec = _safe(asdict(stage1.derivation_spec))
        if stage1.render_spec is not None:
            render_spec = _safe(dict(stage1.render_spec))

    # Mechanism evidence
    evidence, ranking = compute_evidence_and_rank(demos)
    evidence_dict = _safe(asdict(evidence))
    lane_list = [
        {
            "name": c.name,
            "class_score": round(c.class_score, 3),
            "exec_hint": round(c.exec_hint, 3),
            "anti_evidence": c.anti_evidence,
            "final_score": round(c.final_score, 3),
            "gate_pass": c.gate_pass,
        }
        for c in ranking.lanes
    ]

    # Program family = top lane
    program_family = ranking.lanes[0].name if ranking.lanes else ""

    # Solve attempt
    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()
    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)

    verify_result = "no_hypothesis"
    program_skeleton = None
    residuals = None

    if result.solved:
        verify_result = "solved"
    elif result.graphs_compiled > 0:
        verify_result = "compiled_not_verified"
    elif result.graphs_proposed > 0:
        verify_result = "compile_failed"

    # Capture first graph as skeleton
    if result.attempts:
        first_attempt = result.attempts[0]
        program_skeleton = _graph_to_dict(first_attempt.graph)

        # Residuals from failed attempts
        if not result.solved and isinstance(first_attempt.compile_result, CompileSuccess):
            try:
                vr = verifier.verify(first_attempt.compile_result.program, demos)
                if hasattr(vr, "diff") and vr.diff:
                    residuals = _safe(vr.diff)
            except Exception:
                pass

    # Search-decision trace (optional, adds candidate-level detail)
    search_trace = None
    if include_search_trace:
        from aria.core.guidance_traces import trace_search_episode
        episode = trace_search_episode(task_id, demos)
        search_trace = episode.to_dict()

    # Inner-loop trace (optional, adds graph-edit and param-trial detail)
    inner_trace = None
    if include_inner_trace:
        from aria.core.guidance_inner_traces import trace_inner_loop
        ilt = trace_inner_loop(task_id, demos)
        inner_trace = ilt.to_dict()

    # Deep trace (optional, adds within-lane params, library, residual structure)
    deep_trace = None
    if include_deep_trace:
        from aria.core.guidance_deep_traces import trace_deep
        dt = trace_deep(task_id, demos)
        deep_trace = dt.to_dict()

    # Specialization trace (optional, adds binding alternative analysis)
    spec_trace = None
    if include_spec_trace:
        from aria.core.guidance_spec_traces import trace_specialization
        st = trace_specialization(task_id, demos)
        spec_trace = st.to_dict()

    record = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "train_demos": train_grids,
        "size_spec": size_spec,
        "derivation_spec": derivation_spec,
        "render_spec": render_spec,
        "perception_summaries": perception_summaries,
        "roles": roles,
        "legend": legend,
        "zones": zones,
        "slot_grid": slot_grid,
        "correspondences": correspondences,
        "program_family": program_family,
        "program_skeleton": program_skeleton,
        "verify_result": verify_result,
        "residuals": residuals,
        "mechanism_evidence": evidence_dict,
        "lane_ranking": lane_list,
        "search_trace": search_trace,
        "inner_trace": inner_trace,
        "deep_trace": deep_trace,
        "spec_trace": spec_trace,
    }
    return record


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_batch(
    task_ids: list[str],
    demos_fn: Callable[[str], tuple[DemoPair, ...]],
    output_path: str | Path,
    *,
    on_error: str = "skip",
) -> dict[str, int]:
    """Export multiple tasks to a JSONL file.

    Args:
        task_ids: task identifiers to export.
        demos_fn: callable that returns demos for a task_id.
        output_path: path to write JSONL output.
        on_error: "skip" to silently skip failures, "raise" to propagate.

    Returns:
        {"exported": n, "skipped": m}
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    with open(out, "w") as f:
        for tid in task_ids:
            try:
                demos = demos_fn(tid)
                record = export_task(tid, demos)
                f.write(json.dumps(record, sort_keys=True) + "\n")
                exported += 1
            except Exception:
                if on_error == "raise":
                    raise
                skipped += 1

    return {"exported": exported, "skipped": skipped}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load exported JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_record(record: dict) -> list[str]:
    """Validate a single exported record. Returns list of errors (empty = valid)."""
    errors = []
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"wrong schema_version: {record.get('schema_version')}")
    if not isinstance(record.get("train_demos"), list):
        errors.append("train_demos must be a list")
    if not isinstance(record.get("perception_summaries"), list):
        errors.append("perception_summaries must be a list")
    if record.get("verify_result") not in (
        "solved", "compiled_not_verified", "compile_failed", "no_hypothesis"
    ):
        errors.append(f"invalid verify_result: {record.get('verify_result')}")
    return errors
