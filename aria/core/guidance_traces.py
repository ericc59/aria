"""Search-decision trace schema — structured records of candidate/decision processes.

Captures what was proposed, ranked, tried, failed, and verified during the
canonical solve pipeline. This is the real supervision for guided search:
not just final labels, but the full decision trace.

Does not change solver semantics. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Core trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRecord:
    """A single candidate considered during search.

    Represents one proposal at any stage (size, derivation, family, graph).
    """
    candidate_id: str           # unique within the episode, e.g. "size_0", "deriv_2"
    stage: str                  # "output_size", "derivation", "render", "lane", "graph"
    label: str                  # human name, e.g. mode name, family name
    rank: int                   # 0-indexed position in proposal order
    attempted: bool             # was this candidate compiled/verified?
    verified: bool              # did it pass exact verification?
    skipped_reason: str | None  # why it was not attempted (None if attempted)
    params: dict                # candidate-specific parameters
    residual: dict | None       # diff stats if attempted and failed
    score: float | None         # scoring metric if available

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "label": self.label,
            "rank": self.rank,
            "attempted": self.attempted,
            "verified": self.verified,
            "skipped_reason": self.skipped_reason,
            "params": self.params,
            "residual": self.residual,
            "score": self.score,
        }

    @staticmethod
    def from_dict(d: dict) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=d["candidate_id"],
            stage=d["stage"],
            label=d["label"],
            rank=d["rank"],
            attempted=d["attempted"],
            verified=d["verified"],
            skipped_reason=d.get("skipped_reason"),
            params=d.get("params", {}),
            residual=d.get("residual"),
            score=d.get("score"),
        )


@dataclass(frozen=True)
class DecisionStep:
    """One decision point in the search process.

    Groups candidates at a single stage (e.g. "all output-size proposals").
    """
    stage: str                          # same as CandidateRecord.stage
    n_candidates: int
    n_attempted: int
    n_verified: int
    winner_id: str | None               # candidate_id of winner, if any
    best_failed_id: str | None          # best failed candidate (lowest residual)
    candidates: tuple[CandidateRecord, ...]

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_candidates": self.n_candidates,
            "n_attempted": self.n_attempted,
            "n_verified": self.n_verified,
            "winner_id": self.winner_id,
            "best_failed_id": self.best_failed_id,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @staticmethod
    def from_dict(d: dict) -> DecisionStep:
        return DecisionStep(
            stage=d["stage"],
            n_candidates=d["n_candidates"],
            n_attempted=d["n_attempted"],
            n_verified=d["n_verified"],
            winner_id=d.get("winner_id"),
            best_failed_id=d.get("best_failed_id"),
            candidates=tuple(CandidateRecord.from_dict(c) for c in d.get("candidates", [])),
        )


@dataclass(frozen=True)
class SearchEpisode:
    """Complete search-decision trace for one task.

    Contains the full sequence of decision steps, from stage-1 through
    graph proposal and verification.
    """
    schema_version: int
    task_id: str
    steps: tuple[DecisionStep, ...]
    final_winner: str | None            # winning candidate label across all stages
    final_winner_stage: str | None      # which stage produced the winner
    best_failed: str | None             # best failed candidate label overall
    best_failed_stage: str | None
    best_failed_residual: dict | None
    solved: bool

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_winner": self.final_winner,
            "final_winner_stage": self.final_winner_stage,
            "best_failed": self.best_failed,
            "best_failed_stage": self.best_failed_stage,
            "best_failed_residual": self.best_failed_residual,
            "solved": self.solved,
        }

    @staticmethod
    def from_dict(d: dict) -> SearchEpisode:
        return SearchEpisode(
            schema_version=d.get("schema_version", TRACE_SCHEMA_VERSION),
            task_id=d["task_id"],
            steps=tuple(DecisionStep.from_dict(s) for s in d.get("steps", [])),
            final_winner=d.get("final_winner"),
            final_winner_stage=d.get("final_winner_stage"),
            best_failed=d.get("best_failed"),
            best_failed_stage=d.get("best_failed_stage"),
            best_failed_residual=d.get("best_failed_residual"),
            solved=d.get("solved", False),
        )

    @property
    def total_candidates(self) -> int:
        return sum(s.n_candidates for s in self.steps)

    @property
    def total_attempted(self) -> int:
        return sum(s.n_attempted for s in self.steps)

    def step_for_stage(self, stage: str) -> DecisionStep | None:
        for s in self.steps:
            if s.stage == stage:
                return s
        return None


# ---------------------------------------------------------------------------
# Trace construction from the canonical pipeline
# ---------------------------------------------------------------------------


def _safe(v: Any) -> Any:
    """JSON-safe conversion."""
    import numpy as np
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


def _spec_to_params(spec: Any) -> dict:
    """Convert a dataclass spec to a JSON-safe params dict."""
    try:
        return _safe(asdict(spec))
    except Exception:
        return {}


def trace_search_episode(
    task_id: str,
    demos: tuple,
) -> SearchEpisode:
    """Run the canonical pipeline on one task and capture the full decision trace.

    Instruments stage-1 (size/derivation/render), mechanism evidence (lanes),
    and graph proposal/compile/verify, recording every candidate considered.
    """
    from aria.core.grid_perception import perceive_grid
    from aria.core.output_size import infer_verified_output_size_specs, verify_output_size_spec
    from aria.core.output_derivation import infer_verified_output_derivation_specs
    from aria.core.output_stage1 import infer_output_stage1_spec
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess
    from aria.core.protocol import solve as core_solve

    decision_steps: list[DecisionStep] = []
    overall_winner = None
    overall_winner_stage = None
    best_failed_label = None
    best_failed_stage = None
    best_failed_residual = None
    solved = False

    # ---- Stage 1: Output size candidates ----
    size_candidates = _trace_size_candidates(demos)
    decision_steps.append(size_candidates)

    size_winner = size_candidates.winner_id
    if size_winner:
        # ---- Stage 1: Derivation candidates ----
        deriv_candidates = _trace_derivation_candidates(demos)
        decision_steps.append(deriv_candidates)

        if deriv_candidates.winner_id:
            overall_winner = deriv_candidates.winner_id
            overall_winner_stage = "derivation"

        # ---- Stage 1: Render candidates ----
        render_candidates = _trace_render_candidates(demos)
        if render_candidates.n_candidates > 0:
            decision_steps.append(render_candidates)
            if render_candidates.winner_id and not overall_winner:
                overall_winner = render_candidates.winner_id
                overall_winner_stage = "render"

    # ---- Mechanism evidence / lane ranking ----
    lane_candidates = _trace_lane_candidates(demos)
    decision_steps.append(lane_candidates)

    # ---- Graph proposal / compile / verify ----
    graph_candidates = _trace_graph_candidates(task_id, demos)
    decision_steps.append(graph_candidates)

    if graph_candidates.winner_id:
        overall_winner = graph_candidates.winner_id
        overall_winner_stage = "graph"
        solved = True

    # Check if stage-1 direct program solves it
    stage1 = infer_output_stage1_spec(demos)
    if stage1 is not None and not solved:
        from aria.core.output_stage1 import compile_stage1_program
        prog = compile_stage1_program(stage1)
        if prog is not None:
            verifier = ARCVerifier()
            vr = verifier.verify(prog, demos)
            if vr.passed:
                overall_winner = "stage1_direct"
                overall_winner_stage = "stage1"
                solved = True

    # Find best failed across all steps
    for step in decision_steps:
        if step.best_failed_id:
            for c in step.candidates:
                if c.candidate_id == step.best_failed_id and c.residual:
                    if best_failed_residual is None:
                        best_failed_label = c.label
                        best_failed_stage = step.stage
                        best_failed_residual = c.residual

    return SearchEpisode(
        schema_version=TRACE_SCHEMA_VERSION,
        task_id=task_id,
        steps=tuple(decision_steps),
        final_winner=overall_winner,
        final_winner_stage=overall_winner_stage,
        best_failed=best_failed_label,
        best_failed_stage=best_failed_stage,
        best_failed_residual=best_failed_residual,
        solved=solved,
    )


# ---------------------------------------------------------------------------
# Per-stage trace helpers
# ---------------------------------------------------------------------------


def _trace_size_candidates(demos: tuple) -> DecisionStep:
    """Trace all output-size candidates: proposed, verified, skipped."""
    from aria.core.output_size import (
        infer_verified_output_size_specs,
        OutputSizeSpec,
        MODE_SAME_AS_INPUT, MODE_TRANSPOSE_INPUT, MODE_SCALE_INPUT,
        MODE_FIXED_OUTPUT_DIMS,
    )

    # Get all verified specs (the pipeline tries all candidates internally)
    verified_specs = infer_verified_output_size_specs(demos)

    candidates = []
    winner_id = None

    for i, spec in enumerate(verified_specs):
        cid = f"size_{i}"
        candidates.append(CandidateRecord(
            candidate_id=cid,
            stage="output_size",
            label=spec.mode,
            rank=i,
            attempted=True,
            verified=True,
            skipped_reason=None,
            params=_spec_to_params(spec),
            residual=None,
            score=1.0,
        ))
        if i == 0:
            winner_id = cid

    return DecisionStep(
        stage="output_size",
        n_candidates=len(candidates),
        n_attempted=len(candidates),
        n_verified=len(candidates),
        winner_id=winner_id,
        best_failed_id=None,
        candidates=tuple(candidates),
    )


def _trace_derivation_candidates(demos: tuple) -> DecisionStep:
    """Trace all output-derivation candidates."""
    from aria.core.output_derivation import infer_verified_output_derivation_specs

    verified_specs = infer_verified_output_derivation_specs(demos)

    candidates = []
    winner_id = None

    for i, spec in enumerate(verified_specs):
        cid = f"deriv_{i}"
        label = f"{spec.candidate_kind}/{spec.relation}/{spec.selector}"
        candidates.append(CandidateRecord(
            candidate_id=cid,
            stage="derivation",
            label=label,
            rank=i,
            attempted=True,
            verified=True,
            skipped_reason=None,
            params=_spec_to_params(spec),
            residual=None,
            score=1.0,
        ))
        if i == 0:
            winner_id = cid

    return DecisionStep(
        stage="derivation",
        n_candidates=len(candidates),
        n_attempted=len(candidates),
        n_verified=len(candidates),
        winner_id=winner_id,
        best_failed_id=None,
        candidates=tuple(candidates),
    )


def _trace_render_candidates(demos: tuple) -> DecisionStep:
    """Trace render spec candidates from stage-1."""
    from aria.core.output_stage1 import infer_output_stage1_spec

    stage1 = infer_output_stage1_spec(demos)

    candidates = []
    winner_id = None

    if stage1 is not None and stage1.render_spec is not None:
        rs = stage1.render_spec
        cid = "render_0"
        candidates.append(CandidateRecord(
            candidate_id=cid,
            stage="render",
            label=str(rs.get("kind", "tiled_input")),
            rank=0,
            attempted=True,
            verified=True,
            skipped_reason=None,
            params=_safe(dict(rs)),
            residual=None,
            score=1.0,
        ))
        winner_id = cid

    return DecisionStep(
        stage="render",
        n_candidates=len(candidates),
        n_attempted=len(candidates),
        n_verified=len(candidates),
        winner_id=winner_id,
        best_failed_id=None,
        candidates=tuple(candidates),
    )


def _trace_lane_candidates(demos: tuple) -> DecisionStep:
    """Trace mechanism-evidence lane ranking."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank

    evidence, ranking = compute_evidence_and_rank(demos)

    candidates = []
    winner_id = None

    for i, lane in enumerate(ranking.lanes):
        cid = f"lane_{i}"
        candidates.append(CandidateRecord(
            candidate_id=cid,
            stage="lane",
            label=lane.name,
            rank=i,
            attempted=lane.gate_pass,
            verified=False,  # lanes are not directly verified
            skipped_reason=None if lane.gate_pass else f"gate_failed: {lane.anti_evidence}",
            params={
                "class_score": round(lane.class_score, 3),
                "exec_hint": round(lane.exec_hint, 3),
                "final_score": round(lane.final_score, 3),
                "anti_evidence": lane.anti_evidence,
            },
            residual=None,
            score=round(lane.final_score, 3),
        ))
        if i == 0 and lane.gate_pass:
            winner_id = cid

    return DecisionStep(
        stage="lane",
        n_candidates=len(candidates),
        n_attempted=sum(1 for c in candidates if c.attempted),
        n_verified=0,
        winner_id=winner_id,
        best_failed_id=None,
        candidates=tuple(candidates),
    )


def _trace_graph_candidates(task_id: str, demos: tuple) -> DecisionStep:
    """Trace graph proposal / compile / verify pipeline."""
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess, CompileFailure
    from aria.core.trace import _graph_to_dict

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    graphs = fitter.fit(demos, task_id=task_id)

    candidates = []
    winner_id = None
    best_failed_id = None
    best_failed_diff = float("inf")

    for i, graph in enumerate(graphs):
        cid = f"graph_{i}"
        family = graph.metadata.get("family", graph.description[:40])
        attempted = True
        verified = False
        residual = None
        score = None
        skipped = None

        try:
            spec = specializer.specialize(graph, demos)
            result = compiler.compile(graph, spec, demos)

            if isinstance(result, CompileSuccess):
                vr = verifier.verify(result.program, demos)
                verified = vr.passed
                if vr.passed:
                    score = 1.0
                else:
                    # Measure residual
                    residual = _measure_graph_residual(result.program, demos)
                    if residual:
                        diff = residual.get("total_diff", float("inf"))
                        score = max(0, 1.0 - diff / max(residual.get("total_pixels", 1), 1))
                        if diff < best_failed_diff:
                            best_failed_diff = diff
                            best_failed_id = cid
            else:
                # Compile failed
                skipped = getattr(result, "reason", "compile_failed")
                attempted = True
                score = 0.0
        except Exception as e:
            skipped = f"exception: {type(e).__name__}"
            score = 0.0

        candidates.append(CandidateRecord(
            candidate_id=cid,
            stage="graph",
            label=family,
            rank=i,
            attempted=attempted,
            verified=verified,
            skipped_reason=skipped,
            params={"description": graph.description, "family": graph.metadata.get("family", "")},
            residual=residual,
            score=score,
        ))

        if verified and winner_id is None:
            winner_id = cid

    return DecisionStep(
        stage="graph",
        n_candidates=len(candidates),
        n_attempted=sum(1 for c in candidates if c.attempted),
        n_verified=sum(1 for c in candidates if c.verified),
        winner_id=winner_id,
        best_failed_id=best_failed_id,
        candidates=tuple(candidates),
    )


def _measure_graph_residual(program: Any, demos: tuple) -> dict | None:
    """Measure pixel-level residual of a compiled program against demos."""
    import numpy as np
    from aria.verify.trace import traced_execute
    from aria.types import grid_eq

    total_diff = 0
    total_pixels = 0
    per_demo = []

    for demo in demos:
        try:
            predicted, _ = traced_execute(program, demo.input, None, demo.output)
            if predicted is None:
                return None
            diff = int(np.sum(predicted != demo.output))
            pixels = demo.output.size
            total_diff += diff
            total_pixels += pixels
            per_demo.append({"diff": diff, "pixels": pixels})
        except Exception:
            return None

    return {
        "total_diff": total_diff,
        "total_pixels": total_pixels,
        "diff_fraction": round(total_diff / max(total_pixels, 1), 4),
        "per_demo": per_demo,
    }


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_search_traces(
    task_ids: list[str],
    demos_fn,
    output_path: str | Path,
    *,
    on_error: str = "skip",
) -> dict[str, int]:
    """Export search-decision traces for multiple tasks to JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    with open(out, "w") as f:
        for tid in task_ids:
            try:
                demos = demos_fn(tid)
                episode = trace_search_episode(tid, demos)
                f.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")
                exported += 1
            except Exception:
                if on_error == "raise":
                    raise
                skipped += 1

    return {"exported": exported, "skipped": skipped}


def load_search_traces(path: str | Path) -> list[SearchEpisode]:
    """Load search-decision traces from JSONL."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(SearchEpisode.from_dict(json.loads(line)))
    return episodes
