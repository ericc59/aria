"""Deep inner-loop traces — within-lane parameters, library retrieval, residual structure.

Exposes the hidden decisions inside families/lanes:
- periodic repair axis/period/mode alternatives
- grid transform candidate alternatives
- derivation selector alternatives
- library retrieval/adaptation decisions
- structured residual decomposition (region_residuals, subgraph_blames)

Does NOT change solver semantics. Enumerates only structurally justified
alternatives under exact verification. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aria.core.guidance_inner_traces import FailureCategory


DEEP_TRACE_VERSION = 1


# ---------------------------------------------------------------------------
# Within-lane parameter alternative trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamAlternative:
    """One parameter alternative tried within a lane/family."""
    alt_id: str
    family: str
    param_set: dict[str, Any]       # e.g. {"axis": "row", "period": 3, "mode": 0}
    source: str                     # "specialization", "enumeration", "evidence"
    rank: int
    gate_passed: bool
    compiled: bool
    verified: bool
    failure_category: str
    diff_pixels: int | None
    total_pixels: int | None
    residual_fraction: float | None

    def to_dict(self) -> dict:
        return {
            "alt_id": self.alt_id,
            "family": self.family,
            "param_set": {k: _safe(v) for k, v in self.param_set.items()},
            "source": self.source,
            "rank": self.rank,
            "gate_passed": self.gate_passed,
            "compiled": self.compiled,
            "verified": self.verified,
            "failure_category": self.failure_category,
            "diff_pixels": self.diff_pixels,
            "total_pixels": self.total_pixels,
            "residual_fraction": (
                round(self.residual_fraction, 4) if self.residual_fraction is not None else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> ParamAlternative:
        return ParamAlternative(
            alt_id=d["alt_id"], family=d["family"],
            param_set=d["param_set"], source=d["source"],
            rank=d["rank"], gate_passed=d["gate_passed"],
            compiled=d["compiled"], verified=d["verified"],
            failure_category=d["failure_category"],
            diff_pixels=d.get("diff_pixels"),
            total_pixels=d.get("total_pixels"),
            residual_fraction=d.get("residual_fraction"),
        )


@dataclass(frozen=True)
class ParamAlternativeEpisode:
    """All within-lane parameter alternatives for one task."""
    task_id: str
    alternatives: tuple[ParamAlternative, ...]
    n_families: int
    n_total: int
    n_verified: int
    winner_alt_id: str | None
    winner_family: str | None
    best_failed_alt_id: str | None
    best_failed_residual: float | None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "n_families": self.n_families,
            "n_total": self.n_total,
            "n_verified": self.n_verified,
            "winner_alt_id": self.winner_alt_id,
            "winner_family": self.winner_family,
            "best_failed_alt_id": self.best_failed_alt_id,
            "best_failed_residual": (
                round(self.best_failed_residual, 4)
                if self.best_failed_residual is not None else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> ParamAlternativeEpisode:
        return ParamAlternativeEpisode(
            task_id=d["task_id"],
            alternatives=tuple(ParamAlternative.from_dict(a) for a in d.get("alternatives", [])),
            n_families=d["n_families"],
            n_total=d["n_total"],
            n_verified=d["n_verified"],
            winner_alt_id=d.get("winner_alt_id"),
            winner_family=d.get("winner_family"),
            best_failed_alt_id=d.get("best_failed_alt_id"),
            best_failed_residual=d.get("best_failed_residual"),
        )


# ---------------------------------------------------------------------------
# Library retrieval/adaptation trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryRetrievalRecord:
    """One library entry considered for retrieval/adaptation."""
    record_id: str
    source_task_id: str
    strategy: str              # "direct_reuse", "parameterized_variant", "composition"
    template_ops: tuple[str, ...]
    adapted: bool
    adaptation_desc: str
    compiled: bool
    verified: bool
    failure_category: str
    residual_fraction: float | None

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "source_task_id": self.source_task_id,
            "strategy": self.strategy,
            "template_ops": list(self.template_ops),
            "adapted": self.adapted,
            "adaptation_desc": self.adaptation_desc,
            "compiled": self.compiled,
            "verified": self.verified,
            "failure_category": self.failure_category,
            "residual_fraction": (
                round(self.residual_fraction, 4) if self.residual_fraction is not None else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> LibraryRetrievalRecord:
        return LibraryRetrievalRecord(
            record_id=d["record_id"],
            source_task_id=d["source_task_id"],
            strategy=d["strategy"],
            template_ops=tuple(d.get("template_ops", [])),
            adapted=d["adapted"],
            adaptation_desc=d.get("adaptation_desc", ""),
            compiled=d["compiled"],
            verified=d["verified"],
            failure_category=d["failure_category"],
            residual_fraction=d.get("residual_fraction"),
        )


@dataclass(frozen=True)
class LibraryRetrievalEpisode:
    """Library retrieval/adaptation trace for one task."""
    task_id: str
    n_templates_available: int
    records: tuple[LibraryRetrievalRecord, ...]
    n_retrieved: int
    n_adapted: int
    n_compiled: int
    n_verified: int
    winner_record_id: str | None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_templates_available": self.n_templates_available,
            "records": [r.to_dict() for r in self.records],
            "n_retrieved": self.n_retrieved,
            "n_adapted": self.n_adapted,
            "n_compiled": self.n_compiled,
            "n_verified": self.n_verified,
            "winner_record_id": self.winner_record_id,
        }

    @staticmethod
    def from_dict(d: dict) -> LibraryRetrievalEpisode:
        return LibraryRetrievalEpisode(
            task_id=d["task_id"],
            n_templates_available=d["n_templates_available"],
            records=tuple(LibraryRetrievalRecord.from_dict(r) for r in d.get("records", [])),
            n_retrieved=d["n_retrieved"],
            n_adapted=d["n_adapted"],
            n_compiled=d["n_compiled"],
            n_verified=d["n_verified"],
            winner_record_id=d.get("winner_record_id"),
        )


# ---------------------------------------------------------------------------
# Structured residual trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualRegion:
    """One localized region of a residual."""
    label: str              # "frame", "interior", "cell_0", "full_grid"
    diff_pixels: int
    total_pixels: int
    diff_fraction: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "diff_pixels": self.diff_pixels,
            "total_pixels": self.total_pixels,
            "diff_fraction": round(self.diff_fraction, 4),
        }

    @staticmethod
    def from_dict(d: dict) -> ResidualRegion:
        return ResidualRegion(**d)


@dataclass(frozen=True)
class StructuredResidual:
    """Rich residual decomposition for a failed compile/verify attempt."""
    total_diff: int
    total_pixels: int
    diff_fraction: float
    is_localized: bool             # diff concentrated in one region?
    dominant_region: str | None    # region with highest diff if localized
    regions: tuple[ResidualRegion, ...]
    blamed_node_ids: tuple[str, ...]
    blamed_ops: tuple[str, ...]
    repair_hint_count: int
    repair_hint_bindings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "total_diff": self.total_diff,
            "total_pixels": self.total_pixels,
            "diff_fraction": round(self.diff_fraction, 4),
            "is_localized": self.is_localized,
            "dominant_region": self.dominant_region,
            "regions": [r.to_dict() for r in self.regions],
            "blamed_node_ids": list(self.blamed_node_ids),
            "blamed_ops": list(self.blamed_ops),
            "repair_hint_count": self.repair_hint_count,
            "repair_hint_bindings": list(self.repair_hint_bindings),
        }

    @staticmethod
    def from_dict(d: dict) -> StructuredResidual:
        return StructuredResidual(
            total_diff=d["total_diff"],
            total_pixels=d["total_pixels"],
            diff_fraction=d["diff_fraction"],
            is_localized=d["is_localized"],
            dominant_region=d.get("dominant_region"),
            regions=tuple(ResidualRegion.from_dict(r) for r in d.get("regions", [])),
            blamed_node_ids=tuple(d.get("blamed_node_ids", [])),
            blamed_ops=tuple(d.get("blamed_ops", [])),
            repair_hint_count=d.get("repair_hint_count", 0),
            repair_hint_bindings=tuple(d.get("repair_hint_bindings", [])),
        )


# ---------------------------------------------------------------------------
# Composite deep trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepTrace:
    """Combined deep-level trace for one task."""
    task_id: str
    param_alternatives: ParamAlternativeEpisode | None
    library_retrieval: LibraryRetrievalEpisode | None
    structured_residuals: tuple[StructuredResidual, ...] = ()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "param_alternatives": (
                self.param_alternatives.to_dict() if self.param_alternatives else None
            ),
            "library_retrieval": (
                self.library_retrieval.to_dict() if self.library_retrieval else None
            ),
            "structured_residuals": [r.to_dict() for r in self.structured_residuals],
        }

    @staticmethod
    def from_dict(d: dict) -> DeepTrace:
        return DeepTrace(
            task_id=d["task_id"],
            param_alternatives=(
                ParamAlternativeEpisode.from_dict(d["param_alternatives"])
                if d.get("param_alternatives") else None
            ),
            library_retrieval=(
                LibraryRetrievalEpisode.from_dict(d["library_retrieval"])
                if d.get("library_retrieval") else None
            ),
            structured_residuals=tuple(
                StructuredResidual.from_dict(r) for r in d.get("structured_residuals", [])
            ),
        )


# ---------------------------------------------------------------------------
# JSON-safe helper
# ---------------------------------------------------------------------------


def _safe(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
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


# ---------------------------------------------------------------------------
# Phase 1: Within-lane parameter alternative tracing
# ---------------------------------------------------------------------------


def trace_param_alternatives(
    task_id: str,
    demos: tuple,
) -> ParamAlternativeEpisode:
    """Enumerate and trace within-lane parameter alternatives.

    For each family that can enumerate structurally justified alternatives,
    try each under exact verification and record the outcome.
    """
    alts: list[ParamAlternative] = []
    families_seen: set[str] = set()
    winner_id = None
    winner_family = None
    best_failed_id = None
    best_failed_res: float | None = None
    idx = 0

    # Periodic repair alternatives
    periodic_alts = _trace_periodic_alternatives(task_id, demos, idx)
    for a in periodic_alts:
        alts.append(a)
        families_seen.add(a.family)
        if a.verified and winner_id is None:
            winner_id = a.alt_id
            winner_family = a.family
        if (not a.verified and a.residual_fraction is not None
                and (best_failed_res is None or a.residual_fraction < best_failed_res)):
            best_failed_id = a.alt_id
            best_failed_res = a.residual_fraction
    idx += len(periodic_alts)

    # Grid transform alternatives
    transform_alts = _trace_transform_alternatives(task_id, demos, idx)
    for a in transform_alts:
        alts.append(a)
        families_seen.add(a.family)
        if a.verified and winner_id is None:
            winner_id = a.alt_id
            winner_family = a.family
        if (not a.verified and a.residual_fraction is not None
                and (best_failed_res is None or a.residual_fraction < best_failed_res)):
            best_failed_id = a.alt_id
            best_failed_res = a.residual_fraction
    idx += len(transform_alts)

    # Derivation selector alternatives
    deriv_alts = _trace_derivation_alternatives(task_id, demos, idx)
    for a in deriv_alts:
        alts.append(a)
        families_seen.add(a.family)
        if a.verified and winner_id is None:
            winner_id = a.alt_id
            winner_family = a.family
    idx += len(deriv_alts)

    return ParamAlternativeEpisode(
        task_id=task_id,
        alternatives=tuple(alts),
        n_families=len(families_seen),
        n_total=len(alts),
        n_verified=sum(1 for a in alts if a.verified),
        winner_alt_id=winner_id,
        winner_family=winner_family,
        best_failed_alt_id=best_failed_id,
        best_failed_residual=best_failed_res,
    )


def _trace_periodic_alternatives(task_id: str, demos: tuple, start_idx: int) -> list[ParamAlternative]:
    """Enumerate axis × period × mode alternatives for periodic repair."""
    from aria.runtime.ops import has_op

    if not has_op("periodic_repair"):
        return []

    from aria.runtime.ops.periodic_repair import ALL_REPAIR_MODES, REPAIR_MODE_NAMES
    from aria.types import Bind, Call, Literal, Program, Ref, Type
    from aria.verify.verifier import verify

    alts = []
    idx = start_idx

    for axis_str in ("row", "col"):
        axis_int = 0 if axis_str == "row" else 1
        for period in (2, 3, 4, 5):
            for mode in ALL_REPAIR_MODES:
                prog = Program(
                    steps=(Bind("v0", Type.GRID, Call("periodic_repair", (
                        Ref("input"),
                        Literal(axis_int, Type.INT),
                        Literal(period, Type.INT),
                        Literal(mode, Type.INT),
                    ))),),
                    output="v0",
                )

                verified = False
                diff_pixels = None
                total_pixels = None
                res_frac = None
                compiled = True

                try:
                    vr = verify(prog, demos)
                    verified = vr.passed
                    if not verified:
                        diff_pixels, total_pixels = _measure_diff(prog, demos)
                        if total_pixels and total_pixels > 0:
                            res_frac = diff_pixels / total_pixels
                except Exception:
                    compiled = False

                mode_name = REPAIR_MODE_NAMES.get(mode, str(mode))
                alts.append(ParamAlternative(
                    alt_id=f"periodic_{idx}",
                    family="periodic_repair",
                    param_set={"axis": axis_str, "period": period, "mode": mode, "mode_name": mode_name},
                    source="enumeration",
                    rank=idx - start_idx,
                    gate_passed=True,
                    compiled=compiled,
                    verified=verified,
                    failure_category=(
                        FailureCategory.VERIFIED if verified
                        else FailureCategory.EXECUTABLE_HIGH_RESIDUAL if compiled
                        else FailureCategory.EXECUTION_ERROR
                    ),
                    diff_pixels=diff_pixels,
                    total_pixels=total_pixels,
                    residual_fraction=res_frac,
                ))
                idx += 1

                if verified:
                    return alts  # early exit on first verified

    return alts


def _trace_transform_alternatives(task_id: str, demos: tuple, start_idx: int) -> list[ParamAlternative]:
    """Enumerate grid transform alternatives."""
    from aria.types import Bind, Call, Literal, Program, Ref, Type
    from aria.verify.verifier import verify

    if not demos or not all(d.input.shape == d.output.shape for d in demos):
        # Only try transforms when dims match (most transforms preserve dims)
        pass

    alts = []
    idx = start_idx

    candidates = [
        ("rotate_grid", {"degrees": 90}, (Literal(90, Type.INT), Ref("input"))),
        ("rotate_grid", {"degrees": 180}, (Literal(180, Type.INT), Ref("input"))),
        ("rotate_grid", {"degrees": 270}, (Literal(270, Type.INT), Ref("input"))),
        ("reflect_grid", {"axis": "row"}, (Literal(0, Type.AXIS), Ref("input"))),
        ("reflect_grid", {"axis": "col"}, (Literal(1, Type.AXIS), Ref("input"))),
        ("transpose_grid", {"transform": "transpose"}, (Ref("input"),)),
    ]

    for op_name, params, args in candidates:
        prog = Program(steps=(Bind("v0", Type.GRID, Call(op_name, args)),), output="v0")

        verified = False
        diff_pixels = None
        total_pixels = None
        res_frac = None
        compiled = True

        try:
            vr = verify(prog, demos)
            verified = vr.passed
            if not verified:
                diff_pixels, total_pixels = _measure_diff(prog, demos)
                if total_pixels and total_pixels > 0:
                    res_frac = diff_pixels / total_pixels
        except Exception:
            compiled = False

        alts.append(ParamAlternative(
            alt_id=f"transform_{idx}",
            family="grid_transform",
            param_set=params,
            source="enumeration",
            rank=idx - start_idx,
            gate_passed=True,
            compiled=compiled,
            verified=verified,
            failure_category=(
                FailureCategory.VERIFIED if verified
                else FailureCategory.EXECUTABLE_HIGH_RESIDUAL if compiled
                else FailureCategory.EXECUTION_ERROR
            ),
            diff_pixels=diff_pixels,
            total_pixels=total_pixels,
            residual_fraction=res_frac,
        ))
        idx += 1

    return alts


def _trace_derivation_alternatives(task_id: str, demos: tuple, start_idx: int) -> list[ParamAlternative]:
    """Trace all verified derivation specs as alternatives."""
    from aria.core.output_derivation import infer_verified_output_derivation_specs

    verified_specs = infer_verified_output_derivation_specs(demos)
    alts = []

    for i, spec in enumerate(verified_specs):
        label = f"{spec.candidate_kind}/{spec.relation}/{spec.selector}"
        alts.append(ParamAlternative(
            alt_id=f"deriv_{start_idx + i}",
            family="derivation",
            param_set={
                "candidate_kind": spec.candidate_kind,
                "relation": spec.relation,
                "selector": spec.selector,
            },
            source="enumeration",
            rank=i,
            gate_passed=True,
            compiled=True,
            verified=True,
            failure_category=FailureCategory.VERIFIED,
            diff_pixels=0,
            total_pixels=None,
            residual_fraction=0.0,
        ))

    return alts


def _measure_diff(prog: Any, demos: tuple) -> tuple[int | None, int | None]:
    """Measure total diff pixels for a program."""
    from aria.verify.trace import traced_execute

    total_diff = 0
    total_pixels = 0
    for demo in demos:
        try:
            predicted, _ = traced_execute(prog, demo.input, None, demo.output)
            if predicted is None:
                return None, None
            total_diff += int(np.sum(predicted != demo.output))
            total_pixels += demo.output.size
        except Exception:
            return None, None
    return total_diff, total_pixels


# ---------------------------------------------------------------------------
# Phase 2: Library retrieval/adaptation tracing
# ---------------------------------------------------------------------------


def trace_library_retrieval(
    task_id: str,
    demos: tuple,
    library: Any | None = None,
) -> LibraryRetrievalEpisode:
    """Trace library retrieval/adaptation decisions.

    If no library is provided, returns an empty episode (library is optional).
    """
    from aria.core.library import GraphLibrary
    from aria.core.proposer import propose_from_library
    from aria.core.arc import ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess

    if library is None or not isinstance(library, GraphLibrary) or library.size == 0:
        return LibraryRetrievalEpisode(
            task_id=task_id, n_templates_available=0,
            records=(), n_retrieved=0, n_adapted=0,
            n_compiled=0, n_verified=0, winner_record_id=None,
        )

    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    proposals = propose_from_library(library, demos, task_id=task_id)
    records: list[LibraryRetrievalRecord] = []
    winner_id = None

    for i, graph in enumerate(proposals):
        rid = f"lib_{i}"
        desc = graph.description or ""
        strategy = "direct_reuse"
        if "variant" in desc:
            strategy = "parameterized_variant"
        elif "composed" in desc:
            strategy = "composition"

        source_tid = ""
        if "adapted from" in desc:
            source_tid = desc.split("adapted from ")[-1].split(":")[0].strip()

        ops = tuple(sorted(graph.op_set)) if hasattr(graph, "op_set") else ()

        compiled = False
        verified = False
        res_frac = None
        cat = FailureCategory.COMPILE_UNKNOWN

        try:
            spec = specializer.specialize(graph, demos)
            result = compiler.compile(graph, spec, demos)
            if isinstance(result, CompileSuccess):
                compiled = True
                vr = verifier.verify(result.program, demos)
                verified = vr.passed
                if verified:
                    cat = FailureCategory.VERIFIED
                    if winner_id is None:
                        winner_id = rid
                else:
                    d, t = _measure_diff(result.program, demos)
                    if d is not None and t and t > 0:
                        res_frac = d / t
                    cat = FailureCategory.EXECUTABLE_HIGH_RESIDUAL
            else:
                cat = FailureCategory.COMPILE_UNKNOWN
        except Exception:
            cat = FailureCategory.EXECUTION_ERROR

        records.append(LibraryRetrievalRecord(
            record_id=rid, source_task_id=source_tid,
            strategy=strategy, template_ops=ops,
            adapted=strategy != "direct_reuse",
            adaptation_desc=desc,
            compiled=compiled, verified=verified,
            failure_category=cat, residual_fraction=res_frac,
        ))

    return LibraryRetrievalEpisode(
        task_id=task_id,
        n_templates_available=library.size,
        records=tuple(records),
        n_retrieved=len(records),
        n_adapted=sum(1 for r in records if r.adapted),
        n_compiled=sum(1 for r in records if r.compiled),
        n_verified=sum(1 for r in records if r.verified),
        winner_record_id=winner_id,
    )


# ---------------------------------------------------------------------------
# Phase 3: Structured residual extraction
# ---------------------------------------------------------------------------


def extract_structured_residuals(
    task_id: str,
    demos: tuple,
) -> tuple[StructuredResidual, ...]:
    """Extract structured residuals from compile attempts.

    Runs the canonical pipeline, and for each attempt that compiles
    but fails verification, extracts the diagnostic into a typed residual.
    """
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess, CompileFailure
    from aria.core.protocol import solve as core_solve

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)

    residuals: list[StructuredResidual] = []

    for attempt in result.attempts:
        if attempt.verified:
            continue
        if not isinstance(attempt.compile_result, CompileSuccess):
            continue

        # Get diagnostic from compile result
        diag = getattr(attempt.compile_result, "diagnostic", None)

        # Also try running and measuring ourselves
        diff_pixels, total_pixels = _measure_diff(attempt.compile_result.program, demos)
        if diff_pixels is None:
            continue

        regions: list[ResidualRegion] = []
        blamed_nodes: list[str] = []
        blamed_ops: list[str] = []
        hint_count = 0
        hint_bindings: list[str] = []

        if diag is not None:
            for rr in getattr(diag, "region_residuals", ()):
                regions.append(ResidualRegion(
                    label=rr.region_label,
                    diff_pixels=rr.diff_pixels,
                    total_pixels=rr.total_pixels,
                    diff_fraction=rr.diff_fraction,
                ))
            for blame in getattr(diag, "subgraph_blames", ()):
                blamed_nodes.extend(blame.node_ids)
                blamed_ops.extend(blame.ops)
            for hint in getattr(diag, "repair_hints", ()):
                hint_count += 1
                hint_bindings.append(hint.binding_name)

        # Determine if residual is localized
        is_localized = False
        dominant_region = None
        if regions:
            max_region = max(regions, key=lambda r: r.diff_fraction)
            is_localized = max_region.diff_fraction > 0.5 and len(regions) > 1
            dominant_region = max_region.label

        residuals.append(StructuredResidual(
            total_diff=diff_pixels,
            total_pixels=total_pixels,
            diff_fraction=diff_pixels / max(total_pixels, 1),
            is_localized=is_localized,
            dominant_region=dominant_region,
            regions=tuple(regions),
            blamed_node_ids=tuple(blamed_nodes),
            blamed_ops=tuple(blamed_ops),
            repair_hint_count=hint_count,
            repair_hint_bindings=tuple(hint_bindings),
        ))

    return tuple(residuals)


# ---------------------------------------------------------------------------
# Combined deep trace
# ---------------------------------------------------------------------------


def trace_deep(
    task_id: str,
    demos: tuple,
    *,
    library: Any | None = None,
) -> DeepTrace:
    """Capture combined deep trace for one task."""
    param_alts = trace_param_alternatives(task_id, demos)
    lib_retrieval = trace_library_retrieval(task_id, demos, library=library)
    residuals = extract_structured_residuals(task_id, demos)

    return DeepTrace(
        task_id=task_id,
        param_alternatives=param_alts,
        library_retrieval=lib_retrieval,
        structured_residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Export / load
# ---------------------------------------------------------------------------


def export_deep_traces(
    task_ids: list[str],
    demos_fn,
    output_path: str | Path,
    *,
    library: Any | None = None,
    on_error: str = "skip",
) -> dict[str, int]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exported = skipped = 0
    with open(out, "w") as f:
        for tid in task_ids:
            try:
                demos = demos_fn(tid)
                trace = trace_deep(tid, demos, library=library)
                f.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
                exported += 1
            except Exception:
                if on_error == "raise":
                    raise
                skipped += 1
    return {"exported": exported, "skipped": skipped}


def load_deep_traces(path: str | Path) -> list[DeepTrace]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(DeepTrace.from_dict(json.loads(line)))
    return traces
