"""Evaluation harness for running the current search-only solver.

This module is the bridge from dataset iteration/reporting to the newer
``aria.solve.solve_task`` entrypoint. It keeps the persisted report shape
roughly stable while routing evaluation through the canonical
``aria/search`` path instead of the older guided/offline solver stacks.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from aria.datasets import DatasetInfo, iter_tasks
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.reporting import build_solve_report, extract_library_ops_used
from aria.solve import solve_task
from aria.trace_store import RefinementTraceStore
from aria.types import LibraryEntry, grid_eq
from aria.graph.signatures import compute_task_signatures


@dataclass(frozen=True)
class EvalConfig:
    """Solver parameters for an evaluation run.

    Most historical refinement/retrieval fields are kept for CLI/report
    compatibility, but the active eval path now uses ``time_budget_sec`` and
    the canonical search-only ``aria.solve`` entrypoint.
    """

    retrieval_limit: int = 0
    max_search_steps: int = 3
    max_search_candidates: int = 5000
    max_refinement_rounds: int = 2
    include_core_ops: bool = True
    beam_width: int = 0
    beam_rounds: int = 3
    beam_mutations_per_candidate: int = 30
    rerank_edits: bool = True
    time_budget_sec: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_limit": self.retrieval_limit,
            "max_search_steps": self.max_search_steps,
            "max_search_candidates": self.max_search_candidates,
            "max_refinement_rounds": self.max_refinement_rounds,
            "include_core_ops": self.include_core_ops,
            "beam_width": self.beam_width,
            "beam_rounds": self.beam_rounds,
            "beam_mutations_per_candidate": self.beam_mutations_per_candidate,
            "rerank_edits": self.rerank_edits,
            "time_budget_sec": self.time_budget_sec,
        }


def _execute_on_test(task, program) -> list[Any]:
    """Execute a guided or AST-backed program on the task test inputs."""
    return [program.execute(test_pair.input) for test_pair in task.test]


def _program_text(program: Any) -> str:
    """Best-effort text form for reporting and submissions."""
    if hasattr(program, "description") and isinstance(program.description, str):
        return program.description
    return repr(program)


def evaluate_task(
    task_id: str,
    task,
    *,
    library: Library,
    program_store: ProgramStore | None = None,
    config: EvalConfig,
    trace_store: RefinementTraceStore | None = None,
    freeze_stores: bool = True,
) -> dict[str, Any]:
    """Evaluate a single task and return a result dict."""
    t0 = time.time()
    task_signatures = tuple(sorted(compute_task_signatures(task.train)))
    result = solve_task(
        list((pair.input, pair.output) for pair in task.train),
        time_budget=config.time_budget_sec,
    )
    elapsed = time.time() - t0
    del freeze_stores, library, program_store

    outcome: dict[str, Any] = {
        "task_id": task_id,
        "solved": False,
        "program_found": result["program"] is not None,
        "time_sec": round(elapsed, 2),
        "retrieved": False,
        "searched": result["source"] in ("search", None),
        "retrieval_candidates_tried": 0,
        "search_candidates_tried": 0,
        "refinement_rounds": 0,
        "total_candidates": 0,
        "solve_source": result["source"] or "unsolved",
        "abstractions_mined": 0,
        "task_signatures": list(task_signatures),
        "solve_phase": result["source"] or "unsolved",
        "abstraction_hints_available": False,
    }

    if outcome["program_found"]:
        program = result["program"]
        test_outputs = _execute_on_test(task, program)
        program_text = _program_text(program)
        outcome["program"] = program_text
        outcome["description"] = result.get("description", "")
        outcome["test_outputs"] = [g.tolist() for g in test_outputs]
        test_results = []
        known_test_correct = []
        for idx, (test_pair, output) in enumerate(zip(task.test, test_outputs)):
            correct = (
                grid_eq(output, test_pair.output)
                if test_pair.output is not None else None
            )
            test_results.append({"test_idx": idx, "correct": correct})
            if correct is not None:
                known_test_correct.append(correct)
        outcome["test_results"] = test_results
        outcome["solved"] = all(known_test_correct) if known_test_correct else True
        outcome["library_ops_used"] = extract_library_ops_used(program_text, [])
        if outcome["solved"]:
            # Capture structured solve trace if SearchProgram is available
            search_prog = getattr(program, 'search_program', None)
            if search_prog is not None:
                from aria.search.trace_capture import capture_solve_trace
                trace = capture_solve_trace(
                    task_id=task_id,
                    task_signatures=task_signatures,
                    program=search_prog,
                    n_demos=len(task.train),
                    test_correct=all(known_test_correct) if known_test_correct else None,
                )
                outcome["solve_trace"] = trace.to_dict()
            if trace_store is not None:
                trace_store.add_search_result(
                    task_id=task_id,
                    solved=True,
                    winning_program_text=program_text,
                    task_signatures=task_signatures,
                )
        else:
            outcome["failure_bucket"] = "test_mismatch"
            cluster = classify_failure_cluster(outcome)
            outcome["failure_cluster"] = cluster["primary"]
            if cluster["secondary"]:
                outcome["failure_cluster_hints"] = cluster["secondary"]
            if trace_store is not None:
                trace_store.add_search_result(
                    task_id=task_id,
                    solved=False,
                    task_signatures=task_signatures,
                )
    else:
        outcome["failure_bucket"] = "search_budget_exhausted"
        cluster = classify_failure_cluster(outcome)
        outcome["failure_cluster"] = cluster["primary"]
        if cluster["secondary"]:
            outcome["failure_cluster_hints"] = cluster["secondary"]
        if trace_store is not None:
            trace_store.add_search_result(
                task_id=task_id,
                solved=False,
                task_signatures=task_signatures,
            )

    return outcome


def run_evaluation(
    ds: DatasetInfo,
    *,
    library: Library,
    program_store: ProgramStore | None = None,
    config: EvalConfig,
    trace_store: RefinementTraceStore | None = None,
    freeze_stores: bool = True,
    limit: int = 0,
    task_ids: list[str] | None = None,
    on_task_done: Any = None,
) -> dict[str, Any]:
    """Run the solver across a dataset and return a solve report.

    *on_task_done*, if given, is called with (task_id, outcome_dict) after
    each task so callers can print progress or save incrementally.
    """
    task_outcomes: list[dict[str, Any]] = []

    for task_id, task in iter_tasks(ds, limit=limit, task_ids=task_ids):
        outcome = evaluate_task(
            task_id,
            task,
            library=library,
            program_store=program_store,
            config=config,
            trace_store=trace_store,
            freeze_stores=freeze_stores,
        )
        task_outcomes.append(outcome)
        if on_task_done is not None:
            on_task_done(task_id, outcome)

    report_config = {
        "engine": "search",
        "dataset": ds.name,
        "dataset_version": ds.version,
        "dataset_split": ds.split,
        **config.to_dict(),
        "freeze_stores": freeze_stores,
    }
    report = build_solve_report(task_outcomes, report_config)
    report["transfer_metrics"] = compute_transfer_metrics(
        task_outcomes, library=library,
    )
    report["failure_clusters"] = failure_cluster_report(task_outcomes)
    return report


# ---------------------------------------------------------------------------
# Transfer-quality metrics
# ---------------------------------------------------------------------------


def compute_transfer_metrics(
    task_outcomes: list[dict[str, Any]],
    *,
    library: Library | None = None,
) -> dict[str, Any]:
    """Compute metrics that quantify transfer quality, not just solve count.

    These answer "are we actually learning transferable structure?"
    rather than "how many tasks did we solve?"
    """
    retrieval_solves = [t for t in task_outcomes if t.get("solve_source") == "retrieval"]
    prov_solves = [t for t in retrieval_solves if "retrieval_provenance" in t]

    transfer_backed = sum(
        1 for t in prov_solves
        if t["retrieval_provenance"].get("distinct_task_count", 0) >= 2
    )
    non_retrieval_provenance = sum(
        1 for t in prov_solves
        if t["retrieval_provenance"].get("has_non_retrieval_source", False)
    )
    leaf_only = len(retrieval_solves) - transfer_backed

    # Abstraction-guided search metrics
    hints_available = sum(
        1 for t in task_outcomes if t.get("abstraction_hints_available")
    )
    solved_with_abstraction = sum(
        1 for t in task_outcomes if t.get("solved_with_retrieved_abstraction")
    )
    skeleton_tested = sum(
        1 for t in task_outcomes if t.get("skeleton_hypotheses_tested", 0) > 0
    )
    solved_by_skeleton = sum(
        1 for t in task_outcomes if t.get("solved_by_skeleton")
    )

    # Solve-phase breakdown
    phase_counts = Counter(t.get("solve_phase", "unknown") for t in task_outcomes)

    # Failure-bucket breakdown for unsolved tasks
    failure_counts = Counter(
        t.get("failure_bucket", "unknown")
        for t in task_outcomes if not t.get("solved")
    )

    # Failure-cluster breakdown for unsolved tasks
    cluster_counts = Counter(
        t.get("failure_cluster", "unknown")
        for t in task_outcomes if not t.get("solved")
    )

    metrics: dict[str, Any] = {
        "retrieval_solves_total": len(retrieval_solves),
        "retrieval_solves_transfer_backed": transfer_backed,
        "retrieval_solves_leaf_only": leaf_only,
        "retrieval_solves_non_retrieval_provenance": non_retrieval_provenance,
        "abstraction_hints_available": hints_available,
        "solved_with_retrieved_abstraction": solved_with_abstraction,
        "skeleton_hypotheses_tested": skeleton_tested,
        "solved_by_skeleton": solved_by_skeleton,
        "solve_phase_breakdown": dict(phase_counts),
        "failure_bucket_breakdown": dict(failure_counts),
        "failure_cluster_breakdown": dict(cluster_counts),
    }

    # Reranking ablation metrics
    metrics["reranking"] = _reranking_summary(task_outcomes)

    if library is not None:
        entries = library.all_entries()
        metrics["library_quality"] = _library_quality_metrics(entries)

    return metrics


def _reranking_summary(task_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate reranking statistics across all tasks."""
    structural_edit_tasks = [
        t for t in task_outcomes if "structural_edit_tried" in t
    ]
    applied = [t for t in structural_edit_tasks if t.get("reranking_applied")]
    changed = [t for t in applied if t.get("reranking_changed_order")]
    solved_with_edit = [t for t in structural_edit_tasks if t.get("structural_edit_solved")]
    solved_with_reranking = [t for t in solved_with_edit if t.get("reranking_applied")]
    solved_with_change = [t for t in solved_with_edit if t.get("reranking_changed_order")]

    # Family breakdown of structural edit solves
    family_counts = Counter(
        t.get("structural_edit_winning_family", "unknown")
        for t in solved_with_edit
    )

    return {
        "structural_edit_tasks": len(structural_edit_tasks),
        "reranking_applied": len(applied),
        "reranking_changed_order": len(changed),
        "structural_edit_solved": len(solved_with_edit),
        "solved_with_reranking_applied": len(solved_with_reranking),
        "solved_with_reranking_changed": len(solved_with_change),
        "winning_family_breakdown": dict(family_counts) if family_counts else None,
    }


def _library_quality_metrics(entries: list[LibraryEntry]) -> dict[str, Any]:
    """Summarize library health from the perspective of transfer learning."""
    if not entries:
        return {
            "total_entries": 0,
            "multi_task_entries": 0,
            "single_task_entries": 0,
            "avg_support_program_count": 0.0,
            "avg_mdl_gain": 0.0,
            "strong_abstractions": 0,
            "weak_abstractions": 0,
        }

    multi_task = sum(1 for e in entries if len(e.support_task_ids) >= 2)
    mdl_gains = [e.mdl_gain for e in entries]
    support_counts = [e.support_program_count for e in entries]
    strong = sum(1 for e in entries if len(e.support_task_ids) >= 2 and e.mdl_gain > 0)

    def _mean(vals: list[int | float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total_entries": len(entries),
        "multi_task_entries": multi_task,
        "single_task_entries": len(entries) - multi_task,
        "avg_support_program_count": _mean(support_counts),
        "avg_mdl_gain": _mean(mdl_gains),
        "strong_abstractions": strong,
        "weak_abstractions": len(entries) - strong,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persist_trace(
    trace_store: RefinementTraceStore | None,
    result: SolveResult,
) -> None:
    if trace_store is None:
        return
    if result.refinement_result is not None:
        trace_store.add_result(
            task_id=result.task_id,
            result=result.refinement_result,
            task_signatures=result.task_signatures,
        )
        return
    if result.retrieved and result.winning_program is not None:
        trace_store.add_retrieval_result(
            task_id=result.task_id,
            winning_program=result.winning_program,
            candidates_tried=result.retrieval_candidates_tried,
            task_signatures=result.task_signatures,
        )


def _lookup_provenance(
    program_store: ProgramStore,
    winning_program,
) -> dict[str, Any] | None:
    """Look up the StoredProgram record for the winning program to get provenance."""
    program_text = program_to_text(winning_program)
    for record in program_store.all_records():
        if record.program_text == program_text:
            return retrieval_provenance(record)
    return None


def _best_near_miss(hypotheses):
    """Pick the skeleton hypothesis closest to passing.

    Priority: wrong_output (got dims right) > execution_error > others.
    Among wrong_output, prefer those that were tried (error_type is set).
    """
    from aria.hypotheses import SkeletonHypothesis

    candidates = [h for h in hypotheses if not h.passed and h.error_type == "wrong_output"]
    if candidates:
        return candidates[0]
    candidates = [h for h in hypotheses if not h.passed and h.error_type]
    if candidates:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Failure clustering — actionable families for unsolved tasks
# ---------------------------------------------------------------------------

# Cluster names, ordered by priority for deterministic assignment.
FAILURE_CLUSTERS = (
    "same_dims_near_miss",
    "dims_change",
    "selection",
    "composition",
    "transform",
    "same_dims_exhausted",
    "no_signal",
)

# Signature prefixes that indicate each structural family.
_SELECTION_SIGS = frozenset({
    "role:has_marker", "rel:contains_pair", "change:additive",
})
_COMPOSITION_SIGS = frozenset({
    "partition:has_separator_grid", "legend:present",
    "partition:cell_summary_task",
})
_TRANSFORM_SIGS = frozenset({
    "sym:input_reflective", "sym:input_rotational", "sym:input_periodic",
    "sym:output_reflective", "sym:output_rotational", "sym:output_periodic",
    "size:multiplicative",
})


def classify_failure_cluster(
    outcome: dict[str, Any],
) -> dict[str, Any]:
    """Assign a primary failure cluster and optional secondary hints.

    Maps the low-level ``failure_bucket`` (runtime reason) plus
    ``task_signatures`` (structural tags) into a higher-level planning
    cluster that answers "what kind of task is this miss?".

    Deterministic: depends only on fields already present in *outcome*.
    Returns ``{"primary": str, "secondary": list[str]}``.

    Cluster vocabulary (see ``FAILURE_CLUSTERS``):
      same_dims_near_miss  – same-dims task, skeleton got close
      dims_change          – output dims differ from input
      selection            – marker / containment / additive-change signals
      composition          – partition / legend signals
      transform            – symmetry / multiplicative-size signals
      same_dims_exhausted  – same-dims, search budget used up, no near miss
      no_signal            – nothing actionable from signatures or bucket
    """
    sigs = frozenset(outcome.get("task_signatures", ()))
    bucket = outcome.get("failure_bucket", "unknown")
    same_dims = "dims:same" in sigs
    diff_dims = "dims:different" in sigs

    secondary: list[str] = []

    # --- primary assignment (priority order) ---

    # 1. Near miss: skeleton got dims right, content wrong.
    #    Route here even without an explicit dims:same signature — the
    #    near_miss_wrong_output bucket already implies dims matched.
    if bucket == "near_miss_wrong_output" and not diff_dims:
        primary = "same_dims_near_miss"
    # 2. Dims-change tasks
    elif diff_dims or bucket in ("dims_change_unsupported", "dims_change_reconstruction_miss"):
        primary = "dims_change"
    # 3. Selection/subgroup: marker or containment signals
    elif sigs & _SELECTION_SIGS:
        primary = "selection"
    # 4. Composition: partition/legend signals
    elif sigs & _COMPOSITION_SIGS:
        primary = "composition"
    # 5. Transform: symmetry or multiplicative size
    elif sigs & _TRANSFORM_SIGS:
        primary = "transform"
    # 6. Same-dims but search exhausted (no near miss)
    elif same_dims and bucket == "search_budget_exhausted":
        primary = "same_dims_exhausted"
    # 7. Fallback
    else:
        primary = "no_signal"

    # --- secondary hints from signature overlap ---
    if sigs & _SELECTION_SIGS and primary != "selection":
        secondary.append("selection")
    if sigs & _COMPOSITION_SIGS and primary != "composition":
        secondary.append("composition")
    if sigs & _TRANSFORM_SIGS and primary != "transform":
        secondary.append("transform")

    # Repair-error hint
    repair_err = outcome.get("repair_primary_error")
    if repair_err:
        secondary.append(f"repair:{repair_err}")

    return {"primary": primary, "secondary": secondary}


def failure_cluster_report(
    task_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact, machine-readable cluster report from eval outcomes.

    Returns a dict with:
      - ``clusters``: {cluster_name: {"count": int, "task_ids": [str]}}
      - ``total_unsolved``: int
    Sorted by count descending so the biggest family is first.
    """
    unsolved = [t for t in task_outcomes if not t.get("solved")]
    buckets: dict[str, list[str]] = {}
    for t in unsolved:
        cluster = t.get("failure_cluster", "unknown")
        buckets.setdefault(cluster, []).append(t.get("task_id", "?"))

    clusters = {
        name: {"count": len(ids), "task_ids": sorted(ids)}
        for name, ids in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    }
    return {"clusters": clusters, "total_unsolved": len(unsolved)}


def _classify_failure(result: SolveResult) -> str:
    """Assign a single low-level failure bucket to an unsolved task.

    These buckets describe the *runtime reason* the solver failed (e.g.
    dims-change blocked reconstruction, search budget ran out).  They are
    consumed by :func:`classify_failure_cluster` which maps them — together
    with task signatures — into higher-level *planning clusters* that
    drive roadmap decisions.

    Bucket vocabulary (closed set):
      no_candidate                   – no refinement result or zero search candidates
      dims_change_unsupported        – dims change detected, no reconstruction path
      dims_change_reconstruction_miss – reconstruction attempted but failed
      near_miss_wrong_output         – skeleton got dims right, content wrong
      search_budget_exhausted        – search ran but budget expired
    """
    rr = result.refinement_result

    # No refinement result at all
    if rr is None:
        return "no_candidate"

    # Check if dims change prevented reconstruction
    if rr.decomposition is not None:
        evidence = rr.decomposition.evidence
        if "dims_change" in evidence:
            # Distinguish: reconstruction was attempted but failed vs never tried
            if rr.dims_reconstruction is not None and rr.dims_reconstruction.attempted:
                return "dims_change_reconstruction_miss"
            return "dims_change_unsupported"

    # Check if observation diagnosed rules but none were expressible
    if hasattr(rr, 'synthesis_result') and rr.synthesis_result is not None:
        pass  # synthesis ran

    # Check near-miss from skeleton
    if rr.skeleton_result is not None and rr.skeleton_result.hypotheses:
        near_misses = [h for h in rr.skeleton_result.hypotheses if h.error_type == "wrong_output"]
        if near_misses:
            return "near_miss_wrong_output"

    # Search exhausted budget
    if result.search_candidates_tried > 0:
        return "search_budget_exhausted"

    return "no_candidate"
