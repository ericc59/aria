"""Evaluation harness for running the offline solver on a dataset.

Produces a machine-readable results dict compatible with the existing
solve-report format, plus dataset-level metadata and transfer-quality
metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aria.datasets import DatasetInfo, iter_tasks
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.reporting import build_solve_report, extract_library_ops_used
from aria.reporting import extract_op_names
from aria.retrieval import retrieval_provenance
from aria.runtime.program import program_to_text
from aria.solver import SolveResult, solve_task
from aria.trace_store import RefinementTraceStore
from aria.types import LibraryEntry, grid_eq


@dataclass(frozen=True)
class EvalConfig:
    """Solver parameters for an evaluation run."""

    retrieval_limit: int = 0
    max_search_steps: int = 3
    max_search_candidates: int = 5000
    max_refinement_rounds: int = 2
    include_core_ops: bool = True
    beam_width: int = 0
    beam_rounds: int = 3
    beam_mutations_per_candidate: int = 30

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
        }


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
    task_lib = library.clone() if freeze_stores else library
    task_ps = program_store.clone() if (program_store is not None and freeze_stores) else program_store

    t0 = time.time()
    result = solve_task(
        task,
        library=task_lib,
        program_store=task_ps,
        task_id=task_id,
        retrieval_limit=config.retrieval_limit,
        max_search_steps=config.max_search_steps,
        max_search_candidates=config.max_search_candidates,
        max_refinement_rounds=config.max_refinement_rounds,
        include_core_ops=config.include_core_ops,
        beam_width=config.beam_width,
        beam_rounds=config.beam_rounds,
        beam_mutations_per_candidate=config.beam_mutations_per_candidate,
    )
    elapsed = time.time() - t0

    _persist_trace(trace_store, result)

    outcome: dict[str, Any] = {
        "task_id": task_id,
        "solved": result.solved,
        "time_sec": round(elapsed, 2),
        "retrieved": result.retrieved,
        "searched": result.searched,
        "retrieval_candidates_tried": result.retrieval_candidates_tried,
        "search_candidates_tried": result.search_candidates_tried,
        "refinement_rounds": result.refinement_rounds,
        "total_candidates": (
            result.retrieval_candidates_tried
            if result.retrieved
            else result.search_candidates_tried
        ),
        "solve_source": (
            "retrieval" if result.retrieved else "search" if result.solved else "unsolved"
        ),
        "abstractions_mined": result.abstractions_mined,
        "task_signatures": list(result.task_signatures),
    }

    if result.solved and result.winning_program is not None:
        program_text = program_to_text(result.winning_program)
        outcome["program"] = program_text
        outcome["library_ops_used"] = extract_library_ops_used(
            program_text, (library.names() if library else []),
        )
        outcome["test_outputs"] = [g.tolist() for g in result.test_outputs]
        test_results = []
        for idx, (test_pair, output) in enumerate(zip(task.test, result.test_outputs)):
            correct = grid_eq(output, test_pair.output)
            test_results.append({"test_idx": idx, "correct": correct})
        outcome["test_results"] = test_results

    # Attach retrieval provenance when solved via retrieval
    if result.retrieved and result.winning_program is not None and program_store is not None:
        prov = _lookup_provenance(program_store, result.winning_program)
        if prov is not None:
            outcome["retrieval_provenance"] = prov

    # Track abstraction-guided search
    rr = result.refinement_result
    if rr is not None and rr.abstraction_hints:
        hint_names = frozenset(h.name for h in rr.abstraction_hints)
        outcome["abstraction_hints_count"] = len(rr.abstraction_hints)
        outcome["abstraction_hints_available"] = True
        if result.solved and result.winning_program is not None:
            winning_text = program_to_text(result.winning_program)
            winning_ops = extract_op_names(winning_text)
            used = sorted(hint_names & winning_ops)
            outcome["solved_with_retrieved_abstraction"] = bool(used)
            outcome["retrieved_abstractions_used"] = used
        else:
            outcome["solved_with_retrieved_abstraction"] = False
    else:
        outcome["abstraction_hints_available"] = bool(
            rr is not None and rr.abstraction_hints
        )

    # Track skeleton hypothesis testing
    if rr is not None and rr.skeleton_result is not None:
        sr = rr.skeleton_result
        outcome["skeleton_hypotheses_tested"] = sr.skeletons_tested
        outcome["solved_by_skeleton"] = sr.solved

        # Capture best near-miss for diagnosis
        if not sr.solved and sr.hypotheses:
            best = _best_near_miss(sr.hypotheses)
            if best is not None:
                outcome["skeleton_near_miss"] = {
                    "source": best.source,
                    "error_type": best.error_type,
                    "program_text": best.program_text,
                }

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
        "engine": "offline",
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

    metrics: dict[str, Any] = {
        "retrieval_solves_total": len(retrieval_solves),
        "retrieval_solves_transfer_backed": transfer_backed,
        "retrieval_solves_leaf_only": leaf_only,
        "retrieval_solves_non_retrieval_provenance": non_retrieval_provenance,
        "abstraction_hints_available": hints_available,
        "solved_with_retrieved_abstraction": solved_with_abstraction,
        "skeleton_hypotheses_tested": skeleton_tested,
        "solved_by_skeleton": solved_by_skeleton,
    }

    if library is not None:
        entries = library.all_entries()
        metrics["library_quality"] = _library_quality_metrics(entries)

    return metrics


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
