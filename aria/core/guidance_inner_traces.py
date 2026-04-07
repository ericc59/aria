"""Inner-loop search traces — graph-edit and parameter/specialization decisions.

Instruments the graph-edit search (editor_search) and the compile/specialize
parameter selection paths to capture structured decision records.

Does NOT modify solver semantics. Uses read-only observation of the existing
search process. No task-id logic. No benchmark hacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Sequence

import numpy as np


INNER_TRACE_VERSION = 1


# ---------------------------------------------------------------------------
# Failure categories (Phase 6)
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """Structured failure categories for compile/verify attempts."""
    VERIFIED = "verified"
    RESIDUAL_REDUCED = "residual_reduced"
    EXECUTABLE_HIGH_RESIDUAL = "executable_high_residual"
    EXECUTABLE_NOOP = "executable_noop"
    COMPILE_TYPE_ERROR = "compile_type_error"
    COMPILE_MISSING_OP = "compile_missing_op"
    COMPILE_GATE_FAILED = "compile_gate_failed"
    COMPILE_UNKNOWN = "compile_unknown"
    EXECUTION_ERROR = "execution_error"
    SKIPPED_DEDUP = "skipped_dedup"
    SKIPPED_BUDGET = "skipped_budget"
    SKIPPED_DEPTH = "skipped_depth"


# ---------------------------------------------------------------------------
# Graph-edit trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphEditCandidate:
    """One graph-edit candidate in the search."""
    edit_id: str
    action_type: str           # ActionType name
    node_id: str
    key: str
    value_summary: str         # short description of value
    parent_id: str | None      # edit_id of parent state (None for seeds)
    depth: int
    priority: float            # frontier priority score
    attempted: bool            # was compile triggered?
    compiled: bool             # did compile succeed?
    verified: bool
    failure_category: str      # FailureCategory value
    diff_pixels_before: int
    diff_pixels_after: int | None
    residual_fraction: float | None
    score_before: float
    score_after: float | None

    def to_dict(self) -> dict:
        return {
            "edit_id": self.edit_id,
            "action_type": self.action_type,
            "node_id": self.node_id,
            "key": self.key,
            "value_summary": self.value_summary,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "priority": round(self.priority, 3),
            "attempted": self.attempted,
            "compiled": self.compiled,
            "verified": self.verified,
            "failure_category": self.failure_category,
            "diff_pixels_before": self.diff_pixels_before,
            "diff_pixels_after": self.diff_pixels_after,
            "residual_fraction": (
                round(self.residual_fraction, 4)
                if self.residual_fraction is not None else None
            ),
            "score_before": round(self.score_before, 3),
            "score_after": round(self.score_after, 3) if self.score_after is not None else None,
        }

    @staticmethod
    def from_dict(d: dict) -> GraphEditCandidate:
        return GraphEditCandidate(**d)


@dataclass(frozen=True)
class GraphEditEpisode:
    """Complete graph-edit search trace for one task."""
    schema_version: int
    task_id: str
    n_seeds: int
    n_edits_generated: int
    n_edits_attempted: int
    n_edits_compiled: int
    n_edits_verified: int
    n_unique_states: int
    max_depth: int
    solved: bool
    winner_edit_id: str | None
    winner_depth: int | None
    top_lane: str
    edits: tuple[GraphEditCandidate, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "n_seeds": self.n_seeds,
            "n_edits_generated": self.n_edits_generated,
            "n_edits_attempted": self.n_edits_attempted,
            "n_edits_compiled": self.n_edits_compiled,
            "n_edits_verified": self.n_edits_verified,
            "n_unique_states": self.n_unique_states,
            "max_depth": self.max_depth,
            "solved": self.solved,
            "winner_edit_id": self.winner_edit_id,
            "winner_depth": self.winner_depth,
            "top_lane": self.top_lane,
            "edits": [e.to_dict() for e in self.edits],
        }

    @staticmethod
    def from_dict(d: dict) -> GraphEditEpisode:
        return GraphEditEpisode(
            schema_version=d.get("schema_version", INNER_TRACE_VERSION),
            task_id=d["task_id"],
            n_seeds=d["n_seeds"],
            n_edits_generated=d["n_edits_generated"],
            n_edits_attempted=d["n_edits_attempted"],
            n_edits_compiled=d["n_edits_compiled"],
            n_edits_verified=d["n_edits_verified"],
            n_unique_states=d["n_unique_states"],
            max_depth=d["max_depth"],
            solved=d["solved"],
            winner_edit_id=d.get("winner_edit_id"),
            winner_depth=d.get("winner_depth"),
            top_lane=d.get("top_lane", ""),
            edits=tuple(GraphEditCandidate.from_dict(e) for e in d.get("edits", [])),
        )


# ---------------------------------------------------------------------------
# Parameter trial trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamTrialCandidate:
    """One parameter/specialization candidate tried during compilation."""
    trial_id: str
    family: str                # compilation lane / family name
    stage: str                 # "compile_gate", "compile_attempt", "specialize"
    param_name: str            # e.g. "axis", "period", "transform"
    param_value: str           # string representation
    rank: int                  # order in which it was tried
    gate_passed: bool          # did the gate check pass?
    compile_succeeded: bool
    verified: bool
    failure_category: str
    residual_fraction: float | None

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "family": self.family,
            "stage": self.stage,
            "param_name": self.param_name,
            "param_value": self.param_value,
            "rank": self.rank,
            "gate_passed": self.gate_passed,
            "compile_succeeded": self.compile_succeeded,
            "verified": self.verified,
            "failure_category": self.failure_category,
            "residual_fraction": (
                round(self.residual_fraction, 4) if self.residual_fraction is not None else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> ParamTrialCandidate:
        return ParamTrialCandidate(**d)


@dataclass(frozen=True)
class ParamTrialEpisode:
    """Parameter/specialization trial trace for one task."""
    schema_version: int
    task_id: str
    trials: tuple[ParamTrialCandidate, ...]
    n_families_tried: int
    n_gate_passed: int
    n_compiled: int
    n_verified: int
    winner_trial_id: str | None
    winner_family: str | None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "trials": [t.to_dict() for t in self.trials],
            "n_families_tried": self.n_families_tried,
            "n_gate_passed": self.n_gate_passed,
            "n_compiled": self.n_compiled,
            "n_verified": self.n_verified,
            "winner_trial_id": self.winner_trial_id,
            "winner_family": self.winner_family,
        }

    @staticmethod
    def from_dict(d: dict) -> ParamTrialEpisode:
        return ParamTrialEpisode(
            schema_version=d.get("schema_version", INNER_TRACE_VERSION),
            task_id=d["task_id"],
            trials=tuple(ParamTrialCandidate.from_dict(t) for t in d.get("trials", [])),
            n_families_tried=d["n_families_tried"],
            n_gate_passed=d["n_gate_passed"],
            n_compiled=d["n_compiled"],
            n_verified=d["n_verified"],
            winner_trial_id=d.get("winner_trial_id"),
            winner_family=d.get("winner_family"),
        )


# ---------------------------------------------------------------------------
# Composite inner-loop trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InnerLoopTrace:
    """Combined inner-loop trace for one task."""
    task_id: str
    edit_episode: GraphEditEpisode | None
    param_episode: ParamTrialEpisode | None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "edit_episode": self.edit_episode.to_dict() if self.edit_episode else None,
            "param_episode": self.param_episode.to_dict() if self.param_episode else None,
        }

    @staticmethod
    def from_dict(d: dict) -> InnerLoopTrace:
        return InnerLoopTrace(
            task_id=d["task_id"],
            edit_episode=(
                GraphEditEpisode.from_dict(d["edit_episode"])
                if d.get("edit_episode") else None
            ),
            param_episode=(
                ParamTrialEpisode.from_dict(d["param_episode"])
                if d.get("param_episode") else None
            ),
        )


# ---------------------------------------------------------------------------
# Graph-edit search instrumentation (read-only wrapper)
# ---------------------------------------------------------------------------


def _categorize_failure(compile_result: Any, verified: bool, diff_before: int, diff_after: int) -> str:
    """Categorize a compile/verify outcome into a structured failure type."""
    from aria.core.graph import CompileSuccess, CompileFailure

    if verified:
        return FailureCategory.VERIFIED

    if compile_result is None:
        return FailureCategory.COMPILE_UNKNOWN

    if isinstance(compile_result, CompileSuccess):
        if diff_after == diff_before:
            return FailureCategory.EXECUTABLE_NOOP
        elif diff_after < diff_before:
            return FailureCategory.RESIDUAL_REDUCED
        else:
            return FailureCategory.EXECUTABLE_HIGH_RESIDUAL

    # CompileFailure
    reason = getattr(compile_result, "reason", "")
    missing = getattr(compile_result, "missing_ops", ())

    if missing:
        return FailureCategory.COMPILE_MISSING_OP
    if "type" in reason.lower() or "Type" in reason:
        return FailureCategory.COMPILE_TYPE_ERROR
    if "gate" in reason.lower() or "can_compile" in reason.lower():
        return FailureCategory.COMPILE_GATE_FAILED
    return FailureCategory.COMPILE_UNKNOWN


def _action_summary(action: Any) -> str:
    """Short summary of an EditAction value."""
    v = getattr(action, "value", None)
    if v is None:
        return ""
    if hasattr(v, "label"):
        return str(v.label)
    return str(v)[:60]


def trace_edit_search(
    task_id: str,
    demos: tuple,
    *,
    max_depth: int = 4,
    max_frontier: int = 200,
    max_total_compiles: int = 40,
    max_expansions: int = 300,
) -> GraphEditEpisode:
    """Run graph-edit search with full instrumentation.

    Mirrors search_from_seeds but records every edit candidate.
    Does NOT change search behavior — same algorithm, same results.
    """
    import heapq
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.editor_env import ActionType, EditAction, EditState, GraphEditEnv, score_graph
    from aria.core.editor_search import (
        _state_hash, _enumerate_edits, _combined_score, _extract_program,
    )
    from aria.core.graph import CompileSuccess, Specialization
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.seeds import collect_seeds

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    seeds = collect_seeds(demos, fitter, specializer, compiler, verifier, task_id=task_id)
    evidence, ranking = compute_evidence_and_rank(demos)
    top_lane = ranking.lanes[0].name if ranking.lanes else ""

    env = GraphEditEnv(
        examples=demos, specializer=specializer,
        compiler=compiler, verifier=verifier, task_id=task_id,
    )

    edit_records: list[GraphEditCandidate] = []
    visited: set[int] = set()
    frontier: list[tuple[float, int, int, EditState, str, str | None]] = []
    counter = 0
    total_compiles = 0
    total_expansions = 0
    max_depth_seen = 0
    total_generated = 0
    solved = False
    winner_id = None
    winner_depth = None

    # Seed the frontier
    for si, seed in enumerate(seeds):
        state = env.reset(initial_graph=seed.graph)
        if seed.specialization is not None:
            for b in seed.specialization.bindings:
                state = env.step(EditAction(
                    action_type=ActionType.BIND,
                    node_id=b.node_id, key=b.name, value=b.value,
                ))

        h = _state_hash(state.graph, state.specialization)
        if h in visited:
            continue
        visited.add(h)

        eid = f"seed_{si}"
        diff_before = state.diff_pixels

        # Fast verify check
        if seed.already_verified:
            env._state = state
            state = env.step(EditAction(action_type=ActionType.COMPILE))
            total_compiles += 1
            cat = _categorize_failure(state.compile_result, state.verified, diff_before, state.diff_pixels)
            edit_records.append(GraphEditCandidate(
                edit_id=eid, action_type="SEED", node_id="", key="",
                value_summary=seed.provenance, parent_id=None, depth=0,
                priority=_combined_score(state), attempted=True,
                compiled=isinstance(state.compile_result, CompileSuccess),
                verified=state.verified, failure_category=cat,
                diff_pixels_before=diff_before, diff_pixels_after=state.diff_pixels,
                residual_fraction=state.diff_pixels / max(diff_before, 1) if diff_before > 0 else None,
                score_before=score_graph(seed.graph, seed.specialization or Specialization(task_id=task_id, bindings=())),
                score_after=state.score,
            ))
            if state.verified:
                solved = True
                winner_id = eid
                winner_depth = 0
                break
        else:
            edit_records.append(GraphEditCandidate(
                edit_id=eid, action_type="SEED", node_id="", key="",
                value_summary=seed.provenance, parent_id=None, depth=0,
                priority=_combined_score(state), attempted=False,
                compiled=False, verified=False,
                failure_category=FailureCategory.SKIPPED_BUDGET,
                diff_pixels_before=diff_before, diff_pixels_after=None,
                residual_fraction=None,
                score_before=state.score, score_after=None,
            ))

        counter += 1
        heapq.heappush(frontier, (_combined_score(state), counter, 0, state, seed.provenance, eid))

    # Main search loop
    if not solved:
        while frontier and total_expansions < max_expansions:
            priority, _, depth, state, prov, parent_eid = heapq.heappop(frontier)
            max_depth_seen = max(max_depth_seen, depth)

            # Compile if not yet compiled
            diff_before = state.diff_pixels
            if state.compile_result is None and total_compiles < max_total_compiles:
                env._state = state
                state = env.step(EditAction(action_type=ActionType.COMPILE))
                total_compiles += 1

                if state.verified:
                    solved = True
                    winner_id = parent_eid
                    winner_depth = depth
                    break

            if depth >= max_depth:
                continue

            total_expansions += 1
            child_edits = _enumerate_edits(state, top_lane=top_lane)
            total_generated += len(child_edits)

            for ei, edit in enumerate(child_edits):
                if len(frontier) >= max_frontier:
                    break

                env._state = state
                child_state = env.step(edit)

                child_hash = _state_hash(child_state.graph, child_state.specialization)
                eid = f"edit_{total_expansions}_{ei}"

                if child_hash in visited:
                    edit_records.append(GraphEditCandidate(
                        edit_id=eid, action_type=edit.action_type.name,
                        node_id=edit.node_id, key=edit.key,
                        value_summary=_action_summary(edit),
                        parent_id=parent_eid, depth=depth + 1,
                        priority=_combined_score(child_state),
                        attempted=False, compiled=False, verified=False,
                        failure_category=FailureCategory.SKIPPED_DEDUP,
                        diff_pixels_before=state.diff_pixels,
                        diff_pixels_after=None, residual_fraction=None,
                        score_before=state.score, score_after=None,
                    ))
                    continue

                visited.add(child_hash)

                # Eager compile for subgraph replacements
                child_compiled = False
                child_verified = False
                child_cat = FailureCategory.SKIPPED_BUDGET
                if (edit.action_type == ActionType.REPLACE_SUBGRAPH
                        and total_compiles < max_total_compiles):
                    env._state = child_state
                    child_state = env.step(EditAction(action_type=ActionType.COMPILE))
                    total_compiles += 1
                    child_compiled = isinstance(child_state.compile_result, CompileSuccess)
                    child_verified = child_state.verified
                    child_cat = _categorize_failure(
                        child_state.compile_result, child_state.verified,
                        state.diff_pixels, child_state.diff_pixels,
                    )
                    if child_verified:
                        solved = True
                        winner_id = eid
                        winner_depth = depth + 1

                edit_records.append(GraphEditCandidate(
                    edit_id=eid, action_type=edit.action_type.name,
                    node_id=edit.node_id, key=edit.key,
                    value_summary=_action_summary(edit),
                    parent_id=parent_eid, depth=depth + 1,
                    priority=_combined_score(child_state),
                    attempted=child_compiled or (edit.action_type == ActionType.REPLACE_SUBGRAPH),
                    compiled=child_compiled, verified=child_verified,
                    failure_category=child_cat,
                    diff_pixels_before=state.diff_pixels,
                    diff_pixels_after=child_state.diff_pixels if child_compiled else None,
                    residual_fraction=(
                        child_state.diff_pixels / max(state.diff_pixels, 1)
                        if child_compiled and state.diff_pixels > 0 else None
                    ),
                    score_before=state.score,
                    score_after=child_state.score if child_compiled else None,
                ))

                if child_verified:
                    break

                counter += 1
                heapq.heappush(frontier, (
                    _combined_score(child_state), counter, depth + 1,
                    child_state, prov, eid,
                ))

            if solved:
                break

    return GraphEditEpisode(
        schema_version=INNER_TRACE_VERSION,
        task_id=task_id,
        n_seeds=len(seeds),
        n_edits_generated=total_generated,
        n_edits_attempted=sum(1 for e in edit_records if e.attempted),
        n_edits_compiled=sum(1 for e in edit_records if e.compiled),
        n_edits_verified=sum(1 for e in edit_records if e.verified),
        n_unique_states=len(visited),
        max_depth=max_depth_seen,
        solved=solved,
        winner_edit_id=winner_id,
        winner_depth=winner_depth,
        top_lane=top_lane,
        edits=tuple(edit_records),
    )


# ---------------------------------------------------------------------------
# Parameter/specialization trial instrumentation
# ---------------------------------------------------------------------------


def trace_param_trials(
    task_id: str,
    demos: tuple,
) -> ParamTrialEpisode:
    """Trace parameter/specialization trials during compilation.

    Runs the compilation dispatch logic and records which families were
    tried, which gates passed, and which compiled/verified.
    """
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.graph import CompileSuccess
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.sketch_compile import (
        CompileTaskProgram, CompilePerDemoPrograms, CompileFailure,
    )

    fitter = ARCFitter()
    specializer_inst = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    # Get graphs from fitter
    graphs = fitter.fit(demos, task_id=task_id)

    trials: list[ParamTrialCandidate] = []
    trial_count = 0
    families_tried = set()
    winner_id = None
    winner_family = None

    # Also trace mechanism evidence lanes
    evidence, ranking = compute_evidence_and_rank(demos)
    for i, lane in enumerate(ranking.lanes):
        tid = f"lane_{trial_count}"
        trials.append(ParamTrialCandidate(
            trial_id=tid, family=lane.name, stage="lane_ranking",
            param_name="lane_score", param_value=str(round(lane.final_score, 3)),
            rank=i, gate_passed=lane.gate_pass,
            compile_succeeded=False, verified=False,
            failure_category=(
                FailureCategory.COMPILE_GATE_FAILED if not lane.gate_pass
                else FailureCategory.SKIPPED_BUDGET
            ),
            residual_fraction=None,
        ))
        trial_count += 1

    # For each graph, trace the compile dispatch
    for gi, graph in enumerate(graphs):
        spec_obj = specializer_inst.specialize(graph, demos)
        result = compiler.compile(graph, spec_obj, demos)

        family = graph.metadata.get("family", graph.description[:30])
        families_tried.add(family)

        # Extract parameter info from specialization
        param_pairs = []
        for b in spec_obj.bindings:
            param_pairs.append((b.name, str(b.value)[:40]))

        is_success = isinstance(result, CompileSuccess)
        is_verified = False
        residual = None

        if is_success:
            vr = verifier.verify(result.program, demos)
            is_verified = vr.passed
            if not vr.passed:
                residual = _measure_residual_fraction(result.program, demos)

        if is_verified and winner_id is None:
            winner_family = family

        for pi, (pname, pval) in enumerate(param_pairs):
            tid = f"param_{trial_count}"
            cat = _categorize_compile_result(result, is_verified)
            trials.append(ParamTrialCandidate(
                trial_id=tid, family=family, stage="compile_attempt",
                param_name=pname, param_value=pval,
                rank=pi, gate_passed=True,
                compile_succeeded=is_success, verified=is_verified,
                failure_category=cat, residual_fraction=residual,
            ))
            if is_verified and winner_id is None:
                winner_id = tid
            trial_count += 1

        # Record the overall graph compile as a single trial too
        if not param_pairs:
            tid = f"param_{trial_count}"
            cat = _categorize_compile_result(result, is_verified)
            trials.append(ParamTrialCandidate(
                trial_id=tid, family=family, stage="compile_attempt",
                param_name="graph", param_value=f"graph_{gi}",
                rank=0, gate_passed=True,
                compile_succeeded=is_success, verified=is_verified,
                failure_category=cat, residual_fraction=residual,
            ))
            if is_verified and winner_id is None:
                winner_id = tid
            trial_count += 1

    return ParamTrialEpisode(
        schema_version=INNER_TRACE_VERSION,
        task_id=task_id,
        trials=tuple(trials),
        n_families_tried=len(families_tried),
        n_gate_passed=sum(1 for t in trials if t.gate_passed),
        n_compiled=sum(1 for t in trials if t.compile_succeeded),
        n_verified=sum(1 for t in trials if t.verified),
        winner_trial_id=winner_id,
        winner_family=winner_family,
    )


def _categorize_compile_result(result: Any, verified: bool) -> str:
    from aria.core.graph import CompileSuccess, CompileFailure

    if verified:
        return FailureCategory.VERIFIED
    if isinstance(result, CompileSuccess):
        return FailureCategory.EXECUTABLE_HIGH_RESIDUAL
    reason = getattr(result, "reason", "")
    missing = getattr(result, "missing_ops", ())
    if missing:
        return FailureCategory.COMPILE_MISSING_OP
    if "type" in reason.lower():
        return FailureCategory.COMPILE_TYPE_ERROR
    return FailureCategory.COMPILE_UNKNOWN


def _measure_residual_fraction(program: Any, demos: tuple) -> float | None:
    """Measure residual as fraction of total output pixels."""
    from aria.verify.trace import traced_execute

    total_diff = 0
    total_pixels = 0
    for demo in demos:
        try:
            predicted, _ = traced_execute(program, demo.input, None, demo.output)
            if predicted is None:
                return None
            total_diff += int(np.sum(predicted != demo.output))
            total_pixels += demo.output.size
        except Exception:
            return None
    return total_diff / max(total_pixels, 1)


# ---------------------------------------------------------------------------
# Combined trace
# ---------------------------------------------------------------------------


def trace_inner_loop(
    task_id: str,
    demos: tuple,
    *,
    include_edit_search: bool = True,
    include_param_trials: bool = True,
) -> InnerLoopTrace:
    """Capture combined inner-loop trace for one task."""
    edit_ep = None
    param_ep = None

    if include_param_trials:
        param_ep = trace_param_trials(task_id, demos)

    if include_edit_search:
        edit_ep = trace_edit_search(task_id, demos)

    return InnerLoopTrace(
        task_id=task_id,
        edit_episode=edit_ep,
        param_episode=param_ep,
    )


# ---------------------------------------------------------------------------
# Batch export / load
# ---------------------------------------------------------------------------


def export_inner_traces(
    task_ids: list[str],
    demos_fn,
    output_path: str | Path,
    *,
    on_error: str = "skip",
) -> dict[str, int]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = 0
    with open(out, "w") as f:
        for tid in task_ids:
            try:
                demos = demos_fn(tid)
                trace = trace_inner_loop(tid, demos)
                f.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
                exported += 1
            except Exception:
                if on_error == "raise":
                    raise
                skipped += 1
    return {"exported": exported, "skipped": skipped}


def load_inner_traces(path: str | Path) -> list[InnerLoopTrace]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(InnerLoopTrace.from_dict(json.loads(line)))
    return traces
