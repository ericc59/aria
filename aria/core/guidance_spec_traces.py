"""Specialization-alternative traces — expose top-k binding candidates.

Instruments specialize_sketch and related binding logic to capture
which alternative bindings were plausible, why one was chosen, and
whether an alternative would have led to a better compile/verify outcome.

Does NOT change solver semantics. Enumerates only structurally plausible
alternatives. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aria.core.guidance_inner_traces import FailureCategory


SPEC_TRACE_VERSION = 1


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecializationAlternative:
    """One candidate value for a specialization binding."""
    value: Any
    source: str                 # "evidence", "consensus", "enumeration", "metadata_fallback", "demo_decomposition"
    rank: int                   # 0 = chosen by default, 1+ = alternatives
    chosen: bool                # was this the default pick?
    confidence: float           # 0-1, how strongly supported
    rationale: str              # brief reason
    # Residual from verification attempt (filled by _verify_alternatives)
    compiled: bool | None = None
    verified: bool | None = None
    residual_fraction: float | None = None
    residual_localized: bool | None = None
    residual_dominant_region: str | None = None
    blamed_ops: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "value": _safe(self.value),
            "source": self.source,
            "rank": self.rank,
            "chosen": self.chosen,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
        if self.compiled is not None:
            d["compiled"] = self.compiled
        if self.verified is not None:
            d["verified"] = self.verified
        if self.residual_fraction is not None:
            d["residual_fraction"] = round(self.residual_fraction, 4)
        if self.residual_localized is not None:
            d["residual_localized"] = self.residual_localized
        if self.residual_dominant_region is not None:
            d["residual_dominant_region"] = self.residual_dominant_region
        if self.blamed_ops:
            d["blamed_ops"] = list(self.blamed_ops)
        return d

    @staticmethod
    def from_dict(d: dict) -> SpecializationAlternative:
        return SpecializationAlternative(
            value=d["value"], source=d["source"], rank=d["rank"],
            chosen=d["chosen"], confidence=d["confidence"],
            rationale=d.get("rationale", ""),
            compiled=d.get("compiled"),
            verified=d.get("verified"),
            residual_fraction=d.get("residual_fraction"),
            residual_localized=d.get("residual_localized"),
            residual_dominant_region=d.get("residual_dominant_region"),
            blamed_ops=tuple(d.get("blamed_ops", ())),
        )


@dataclass(frozen=True)
class SpecializationDecision:
    """One binding decision with its alternatives."""
    binding_name: str           # e.g. "dominant_axis", "frame_colors", "strategy"
    node_id: str                # "__task__", "__canvas__", etc.
    binding_type: str           # "axis", "period", "color", "strategy", "transform", etc.
    alternatives: tuple[SpecializationAlternative, ...]
    chosen_value: Any
    chosen_source: str
    # Verification of alternatives (filled by compile/verify pass)
    attempted_count: int = 0
    verified_count: int = 0
    best_alternative_rank: int | None = None  # rank of best-performing alt

    def to_dict(self) -> dict:
        return {
            "binding_name": self.binding_name,
            "node_id": self.node_id,
            "binding_type": self.binding_type,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "chosen_value": _safe(self.chosen_value),
            "chosen_source": self.chosen_source,
            "attempted_count": self.attempted_count,
            "verified_count": self.verified_count,
            "best_alternative_rank": self.best_alternative_rank,
        }

    @staticmethod
    def from_dict(d: dict) -> SpecializationDecision:
        return SpecializationDecision(
            binding_name=d["binding_name"],
            node_id=d["node_id"],
            binding_type=d["binding_type"],
            alternatives=tuple(SpecializationAlternative.from_dict(a) for a in d.get("alternatives", [])),
            chosen_value=d["chosen_value"],
            chosen_source=d["chosen_source"],
            attempted_count=d.get("attempted_count", 0),
            verified_count=d.get("verified_count", 0),
            best_alternative_rank=d.get("best_alternative_rank"),
        )


@dataclass(frozen=True)
class SpecializationEpisode:
    """All specialization decisions for one task."""
    schema_version: int
    task_id: str
    decisions: tuple[SpecializationDecision, ...]
    n_bindings: int
    n_with_alternatives: int    # bindings that have >1 candidate
    n_default_wrong: int        # bindings where default didn't verify but an alt did
    default_verified: bool      # did the full default specialization verify?

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "n_bindings": self.n_bindings,
            "n_with_alternatives": self.n_with_alternatives,
            "n_default_wrong": self.n_default_wrong,
            "default_verified": self.default_verified,
        }

    @staticmethod
    def from_dict(d: dict) -> SpecializationEpisode:
        return SpecializationEpisode(
            schema_version=d.get("schema_version", SPEC_TRACE_VERSION),
            task_id=d["task_id"],
            decisions=tuple(SpecializationDecision.from_dict(dec) for dec in d.get("decisions", [])),
            n_bindings=d["n_bindings"],
            n_with_alternatives=d["n_with_alternatives"],
            n_default_wrong=d.get("n_default_wrong", 0),
            default_verified=d.get("default_verified", False),
        )


# ---------------------------------------------------------------------------
# JSON helpers
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
# Instrumentation: extract alternatives from specialize_sketch paths
# ---------------------------------------------------------------------------


def trace_specialization(
    task_id: str,
    demos: tuple,
) -> SpecializationEpisode:
    """Trace specialization decisions with top-k alternatives.

    Mirrors the binding logic in specialize_sketch but records
    which alternatives were structurally plausible at each decision point.
    Then verifies the default specialization and any alternatives where
    the binding is substitutable.
    """
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess
    from aria.sketch import SketchGraph
    from aria.sketch_fit import specialize_sketch
    from aria.decomposition import detect_bg, decompose_composites

    decisions: list[SpecializationDecision] = []

    # Get a graph to specialize
    fitter = ARCFitter()
    graphs = fitter.fit(demos, task_id=task_id)
    if not graphs:
        return SpecializationEpisode(
            SPEC_TRACE_VERSION, task_id, (), 0, 0, 0, False,
        )

    graph = graphs[0]
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    # Run default specialization
    default_spec = specializer.specialize(graph, demos)
    default_result = compiler.compile(graph, default_spec, demos)
    default_verified = False
    if isinstance(default_result, CompileSuccess):
        vr = verifier.verify(default_result.program, demos)
        default_verified = vr.passed

    # Extract alternatives for each binding category

    # 1. Axis alternatives
    _trace_axis_alternatives(demos, graph, default_spec, decisions)

    # 2. Period alternatives
    _trace_period_alternatives(demos, graph, default_spec, decisions)

    # 3. BG color alternatives
    _trace_bg_alternatives(demos, decisions)

    # 4. Frame color alternatives
    _trace_frame_color_alternatives(demos, graph, decisions)

    # 5. Transform alternatives
    _trace_transform_alternatives(demos, graph, default_spec, decisions)

    # 6. Movement strategy alternatives
    _trace_movement_alternatives(demos, graph, default_spec, decisions)

    # 7. Canvas strategy alternatives
    _trace_canvas_alternatives(demos, graph, default_spec, decisions)

    # Verify alternatives by substituting into the default specialization
    n_default_wrong = 0
    for decision in decisions:
        _verify_alternatives(
            decision, graph, default_spec, demos,
            specializer, compiler, verifier,
        )
        if (not default_verified and decision.verified_count > 0
                and not any(a.chosen for a in decision.alternatives if decision.verified_count > 0)):
            n_default_wrong += 1

    # Rebuild decisions with updated verification counts
    updated_decisions = []
    n_default_wrong = 0
    for d in decisions:
        # Check if default was wrong: no alt with chosen=True verified, but some other did
        chosen_verified = False
        any_alt_verified = False
        for a in d.alternatives:
            if a.chosen and d.verified_count > 0:
                # Need to check actual verification per-alt; approximate: if default verified, chosen is good
                if default_verified:
                    chosen_verified = True
        if d.verified_count > 0 and not default_verified:
            n_default_wrong += 1
        updated_decisions.append(d)

    n_with_alts = sum(1 for d in updated_decisions if len(d.alternatives) > 1)

    return SpecializationEpisode(
        schema_version=SPEC_TRACE_VERSION,
        task_id=task_id,
        decisions=tuple(updated_decisions),
        n_bindings=len(updated_decisions),
        n_with_alternatives=n_with_alts,
        n_default_wrong=n_default_wrong,
        default_verified=default_verified,
    )


def _trace_axis_alternatives(
    demos: tuple, graph: Any, default_spec: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Axis binding: always {row, col} as alternatives."""
    # Find current axis from specialization
    current_axis = default_spec.get("__task__", "dominant_axis")
    if current_axis is None:
        # Check node evidence
        for node in graph.nodes.values():
            for slot in getattr(node, "slots", ()):
                if hasattr(slot, "name") and slot.name == "axis" and hasattr(slot, "evidence") and slot.evidence:
                    current_axis = slot.evidence
                    break
            if current_axis:
                break

    if current_axis is None:
        return

    alts = [
        SpecializationAlternative(
            value="row", source="enumeration", rank=0 if current_axis == "row" else 1,
            chosen=current_axis == "row", confidence=0.6 if current_axis == "row" else 0.4,
            rationale="row axis" + (" (chosen)" if current_axis == "row" else ""),
        ),
        SpecializationAlternative(
            value="col", source="enumeration", rank=0 if current_axis == "col" else 1,
            chosen=current_axis == "col", confidence=0.6 if current_axis == "col" else 0.4,
            rationale="col axis" + (" (chosen)" if current_axis == "col" else ""),
        ),
    ]
    alts.sort(key=lambda a: a.rank)

    decisions.append(SpecializationDecision(
        binding_name="dominant_axis",
        node_id="__task__",
        binding_type="axis",
        alternatives=tuple(alts),
        chosen_value=current_axis,
        chosen_source="evidence",
    ))


def _trace_period_alternatives(
    demos: tuple, graph: Any, default_spec: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Period binding: enumerate small integers {2,3,4,5}."""
    current_period = default_spec.get("__task__", "dominant_period")
    if current_period is None:
        return

    alts = []
    for i, p in enumerate([2, 3, 4, 5]):
        is_chosen = (p == current_period)
        alts.append(SpecializationAlternative(
            value=p, source="enumeration",
            rank=0 if is_chosen else i + 1,
            chosen=is_chosen,
            confidence=0.8 if is_chosen else 0.2,
            rationale=f"period={p}" + (" (chosen)" if is_chosen else ""),
        ))
    alts.sort(key=lambda a: a.rank)

    decisions.append(SpecializationDecision(
        binding_name="dominant_period",
        node_id="__task__",
        binding_type="period",
        alternatives=tuple(alts),
        chosen_value=current_period,
        chosen_source="evidence",
    ))


def _trace_bg_alternatives(
    demos: tuple,
    decisions: list[SpecializationDecision],
) -> None:
    """BG color alternatives: per-demo majority colors."""
    from aria.decomposition import detect_bg

    if not demos:
        return

    bg_colors = [detect_bg(d.input) for d in demos]
    consensus = bg_colors[0] if len(set(bg_colors)) == 1 else None

    # Alternative: most common color across all demos
    all_colors: Counter = Counter()
    for d in demos:
        vals, counts = np.unique(d.input, return_counts=True)
        for v, c in zip(vals, counts):
            all_colors[int(v)] += int(c)

    top_colors = [c for c, _ in all_colors.most_common(3)]

    alts = []
    for i, c in enumerate(top_colors):
        is_chosen = (c == consensus) if consensus is not None else (i == 0)
        alts.append(SpecializationAlternative(
            value=c, source="consensus" if is_chosen else "enumeration",
            rank=i, chosen=is_chosen,
            confidence=0.9 if is_chosen else 0.3,
            rationale=f"bg={c}, frequency rank {i}",
        ))

    if not alts:
        return

    decisions.append(SpecializationDecision(
        binding_name="bg",
        node_id="__task__",
        binding_type="color",
        alternatives=tuple(alts),
        chosen_value=consensus if consensus is not None else top_colors[0],
        chosen_source="consensus",
    ))


def _trace_frame_color_alternatives(
    demos: tuple, graph: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Frame color alternatives from evidence vs metadata."""
    from aria.decomposition import detect_bg, detect_framed_regions

    if not demos:
        return

    # Collect frame colors from demo evidence
    frame_colors_per_demo: list[list[int]] = []
    for d in demos:
        bg = detect_bg(d.input)
        try:
            regions = detect_framed_regions(d.input, bg=bg)
            colors = list({r.frame_color for r in regions})
            frame_colors_per_demo.append(colors)
        except Exception:
            frame_colors_per_demo.append([])

    all_frame_colors: Counter = Counter()
    for colors in frame_colors_per_demo:
        for c in colors:
            all_frame_colors[c] += 1

    if not all_frame_colors:
        return

    top_colors = [c for c, _ in all_frame_colors.most_common(3)]

    # Check what graph evidence says
    graph_frame = None
    for node in graph.nodes.values():
        fc = node.evidence.get("frame_colors_observed")
        if fc:
            graph_frame = fc[0] if isinstance(fc, list) and fc else fc
            break

    chosen = graph_frame if graph_frame is not None else top_colors[0]

    alts = []
    for i, c in enumerate(top_colors):
        is_chosen = (c == chosen)
        source = "node_evidence" if (c == graph_frame) else "demo_decomposition"
        alts.append(SpecializationAlternative(
            value=c, source=source, rank=i,
            chosen=is_chosen,
            confidence=0.8 if is_chosen else 0.3,
            rationale=f"frame_color={c} from {source}",
        ))
    alts.sort(key=lambda a: (not a.chosen, a.rank))
    for i, a in enumerate(alts):
        alts[i] = SpecializationAlternative(
            value=a.value, source=a.source, rank=i,
            chosen=a.chosen, confidence=a.confidence, rationale=a.rationale,
        )

    decisions.append(SpecializationDecision(
        binding_name="frame_colors",
        node_id="__task__",
        binding_type="color",
        alternatives=tuple(alts),
        chosen_value=chosen,
        chosen_source="node_evidence" if graph_frame else "demo_decomposition",
    ))


def _trace_transform_alternatives(
    demos: tuple, graph: Any, default_spec: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Grid transform type alternatives."""
    current = default_spec.get("__grid_transform__", "transform")
    if current is None:
        return

    transforms = ["rotate", "reflect", "transpose", "fill_enclosed"]
    alts = []
    for i, t in enumerate(transforms):
        is_chosen = (t == current)
        alts.append(SpecializationAlternative(
            value=t, source="node_evidence" if is_chosen else "enumeration",
            rank=0 if is_chosen else i + 1,
            chosen=is_chosen, confidence=0.9 if is_chosen else 0.2,
            rationale=f"transform={t}",
        ))
    alts.sort(key=lambda a: a.rank)

    decisions.append(SpecializationDecision(
        binding_name="transform",
        node_id="__grid_transform__",
        binding_type="transform",
        alternatives=tuple(alts),
        chosen_value=current,
        chosen_source="node_evidence",
    ))


def _trace_movement_alternatives(
    demos: tuple, graph: Any, default_spec: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Movement strategy alternatives."""
    current = default_spec.get("__movement__", "strategy")
    if current is None:
        return

    strategies = ["uniform_translate", "gravity"]
    alts = []
    for i, s in enumerate(strategies):
        is_chosen = (s == current)
        alts.append(SpecializationAlternative(
            value=s, source="node_evidence" if is_chosen else "enumeration",
            rank=0 if is_chosen else i + 1,
            chosen=is_chosen, confidence=0.9 if is_chosen else 0.3,
            rationale=f"strategy={s}",
        ))
    alts.sort(key=lambda a: a.rank)

    decisions.append(SpecializationDecision(
        binding_name="strategy",
        node_id="__movement__",
        binding_type="strategy",
        alternatives=tuple(alts),
        chosen_value=current,
        chosen_source="node_evidence",
    ))


def _trace_canvas_alternatives(
    demos: tuple, graph: Any, default_spec: Any,
    decisions: list[SpecializationDecision],
) -> None:
    """Canvas strategy alternatives."""
    current = default_spec.get("__canvas__", "strategy")
    if current is None:
        return

    strategies = ["tile", "upscale", "crop"]
    alts = []
    for i, s in enumerate(strategies):
        is_chosen = (s == current)
        alts.append(SpecializationAlternative(
            value=s, source="demo_decomposition" if is_chosen else "enumeration",
            rank=0 if is_chosen else i + 1,
            chosen=is_chosen, confidence=0.9 if is_chosen else 0.2,
            rationale=f"canvas_strategy={s}",
        ))
    alts.sort(key=lambda a: a.rank)

    decisions.append(SpecializationDecision(
        binding_name="strategy",
        node_id="__canvas__",
        binding_type="strategy",
        alternatives=tuple(alts),
        chosen_value=current,
        chosen_source="demo_decomposition",
    ))


# ---------------------------------------------------------------------------
# Alternative verification
# ---------------------------------------------------------------------------


def _verify_alternatives(
    decision: SpecializationDecision,
    graph: Any,
    default_spec: Any,
    demos: tuple,
    specializer: Any,
    compiler: Any,
    verifier: Any,
) -> None:
    """Try compiling/verifying each alternative binding value.

    Enriches each alternative with compile/verify results and structured
    residual information. Mutates decision counts.
    """
    from aria.core.graph import CompileSuccess, CompileFailure, ResolvedBinding, Specialization
    from aria.verify.trace import traced_execute

    attempted = 0
    verified_count = 0
    best_rank = None
    enriched_alts: list[SpecializationAlternative] = []

    for alt in decision.alternatives:
        if alt.chosen:
            enriched_alts.append(alt)
            continue

        new_bindings = _substitute_binding(
            default_spec.bindings, decision.node_id, decision.binding_name, alt.value, alt.rank,
        )
        alt_spec = Specialization(
            task_id=default_spec.task_id,
            bindings=tuple(new_bindings),
            metadata=dict(default_spec.metadata),
        )

        alt_compiled = False
        alt_verified = False
        res_frac = None
        res_localized = None
        res_dominant = None
        blamed = ()

        try:
            result = compiler.compile(graph, alt_spec, demos)
            attempted += 1
            if isinstance(result, CompileSuccess):
                alt_compiled = True
                vr = verifier.verify(result.program, demos)
                alt_verified = vr.passed
                if alt_verified:
                    verified_count += 1
                    if best_rank is None or alt.rank < best_rank:
                        best_rank = alt.rank
                else:
                    # Measure residual
                    total_diff = 0
                    total_px = 0
                    for demo in demos:
                        try:
                            pred, _ = traced_execute(result.program, demo.input, None, demo.output)
                            if pred is not None:
                                total_diff += int(np.sum(pred != demo.output))
                                total_px += demo.output.size
                        except Exception:
                            pass
                    if total_px > 0:
                        res_frac = total_diff / total_px
                    # Extract diagnostic if available
                    diag = getattr(result, "diagnostic", None)
                    if diag is not None:
                        rr = getattr(diag, "region_residuals", ())
                        if rr:
                            max_r = max(rr, key=lambda r: r.diff_fraction)
                            res_localized = max_r.diff_fraction > 0.5 and len(rr) > 1
                            res_dominant = max_r.region_label
                        for blame in getattr(diag, "subgraph_blames", ()):
                            blamed = tuple(blame.ops)
            else:
                # Compile failure — extract blame from diagnostic
                diag = getattr(result, "diagnostic", None)
                if diag is not None:
                    for blame in getattr(diag, "subgraph_blames", ()):
                        blamed = tuple(blame.ops)
        except Exception:
            attempted += 1

        enriched_alts.append(SpecializationAlternative(
            value=alt.value, source=alt.source, rank=alt.rank,
            chosen=alt.chosen, confidence=alt.confidence, rationale=alt.rationale,
            compiled=alt_compiled, verified=alt_verified,
            residual_fraction=res_frac, residual_localized=res_localized,
            residual_dominant_region=res_dominant, blamed_ops=blamed,
        ))

    # Rebuild decision with enriched alternatives
    object.__setattr__(decision, "alternatives", tuple(enriched_alts))
    object.__setattr__(decision, "attempted_count", attempted)
    object.__setattr__(decision, "verified_count", verified_count)
    object.__setattr__(decision, "best_alternative_rank", best_rank)


def _substitute_binding(bindings, node_id, name, value, rank):
    from aria.core.graph import ResolvedBinding
    new = []
    substituted = False
    for b in bindings:
        if b.node_id == node_id and b.name == name:
            new.append(ResolvedBinding(node_id=b.node_id, name=b.name, value=value, source=f"alternative_{rank}"))
            substituted = True
        else:
            new.append(b)
    if not substituted:
        new.append(ResolvedBinding(node_id=node_id, name=name, value=value, source=f"alternative_{rank}"))
    return new


# ---------------------------------------------------------------------------
# Export / load
# ---------------------------------------------------------------------------


def export_spec_traces(
    task_ids: list[str],
    demos_fn,
    output_path: str | Path,
    *,
    on_error: str = "skip",
) -> dict[str, int]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exported = skipped = 0
    with open(out, "w") as f:
        for tid in task_ids:
            try:
                demos = demos_fn(tid)
                ep = trace_specialization(tid, demos)
                f.write(json.dumps(ep.to_dict(), sort_keys=True) + "\n")
                exported += 1
            except Exception:
                if on_error == "raise":
                    raise
                skipped += 1
    return {"exported": exported, "skipped": skipped}


def load_spec_traces(path: str | Path) -> list[SpecializationEpisode]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(SpecializationEpisode.from_dict(json.loads(line)))
    return episodes
