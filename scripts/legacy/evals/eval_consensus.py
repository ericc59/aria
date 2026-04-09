#!/usr/bin/env python3
"""Evaluate stepwise all-demo consensus on ARC-2.

Compares baseline (no consensus) vs consensus-gated search.
Reports: solve count, candidate counts, prune rates, false-prune detection.

Usage:
    # Quick smoke test (5 tasks):
    python scripts/eval_consensus.py --limit 5

    # Full v2-eval (120 tasks):
    python scripts/eval_consensus.py

    # Single task with full trace:
    python scripts/eval_consensus.py --task 0934a4d8 --verbose

    # Specific dataset:
    python scripts/eval_consensus.py --dataset v2-train --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from aria.consensus_trace import ConsensusTrace, format_consensus_trace
from aria.core.scene_solve import (
    get_consensus_counters,
    reset_consensus_counters,
    set_consensus_enabled,
)
from aria.datasets import DatasetInfo, dataset_names, get_dataset, list_task_ids, load_arc_task
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.solver import SolveResult, solve_task
from aria.types import Task

RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_PROGRAM_STORE = RESULTS_DIR / "program_store.json"
DEFAULT_LIBRARY_STORE = RESULTS_DIR / "library.json"


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------

@dataclass
class TaskConsensusResult:
    task_id: str
    # Baseline (no consensus)
    baseline_solved: bool = False
    baseline_candidates: int = 0
    baseline_time: float = 0.0
    baseline_verify_calls: int = 0
    # Consensus-gated
    consensus_solved: bool = False
    consensus_candidates: int = 0
    consensus_time: float = 0.0
    consensus_verify_calls: int = 0
    consensus_prune_count: int = 0
    # Shadow diagnostics
    branches_pruned: int = 0
    branches_survived: int = 0
    prune_checks: dict[str, int] = field(default_factory=dict)
    # False-prune detection
    false_prune: bool = False  # True if baseline solved but consensus didn't
    consensus_trace_text: str = ""

    def candidate_savings(self) -> int:
        return self.baseline_candidates - self.consensus_candidates

    def verify_savings(self) -> int:
        return self.baseline_verify_calls - self.consensus_verify_calls

    def time_savings(self) -> float:
        return self.baseline_time - self.consensus_time


# ---------------------------------------------------------------------------
# Run one task in both modes
# ---------------------------------------------------------------------------

def eval_task_consensus(
    task_id: str,
    task: Task,
    library: Library,
    program_store: ProgramStore | None,
) -> TaskConsensusResult:
    """Run a task in baseline and consensus mode, return comparison."""
    result = TaskConsensusResult(task_id=task_id)

    # --- Baseline: consensus OFF ---
    set_consensus_enabled(False)
    reset_consensus_counters()
    t0 = time.time()
    try:
        baseline = solve_task(
            task,
            library=library.clone(),
            program_store=program_store.clone() if program_store else None,
            task_id=task_id,
        )
        result.baseline_solved = baseline.solved
        result.baseline_candidates = baseline.search_candidates_tried
    except Exception:
        pass
    result.baseline_time = round(time.time() - t0, 3)
    baseline_counters = get_consensus_counters()
    result.baseline_verify_calls = baseline_counters["verified"]

    # --- Consensus: consensus ON ---
    set_consensus_enabled(True)
    reset_consensus_counters()
    t0 = time.time()
    try:
        consensus = solve_task(
            task,
            library=library.clone(),
            program_store=program_store.clone() if program_store else None,
            task_id=task_id,
        )
        result.consensus_solved = consensus.solved
        result.consensus_candidates = consensus.search_candidates_tried
    except Exception:
        pass
    result.consensus_time = round(time.time() - t0, 3)
    consensus_counters = get_consensus_counters()
    result.consensus_verify_calls = consensus_counters["verified"]
    result.consensus_prune_count = consensus_counters["pruned"]

    # --- Shadow: run scene_solve with trace to capture prune details ---
    set_consensus_enabled(True)
    try:
        trace = ConsensusTrace()
        from aria.core.scene_solve import infer_scene_programs
        infer_scene_programs(task.train, consensus_trace=trace)
        result.branches_pruned = trace.branches_pruned
        result.branches_survived = trace.branches_survived
        # Count prunes by check name
        for entry in trace.entries:
            if entry.pruned:
                for check in entry.checks:
                    if not check.passed:
                        result.prune_checks[check.name] = (
                            result.prune_checks.get(check.name, 0) + 1
                        )
        result.consensus_trace_text = format_consensus_trace(trace)
    except Exception:
        pass

    # False-prune detection
    result.false_prune = result.baseline_solved and not result.consensus_solved

    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class ConsensusReport:
    dataset: str = ""
    n_tasks: int = 0
    baseline_solves: int = 0
    consensus_solves: int = 0
    tasks_improved: list[str] = field(default_factory=list)
    tasks_worsened: list[str] = field(default_factory=list)
    total_baseline_candidates: int = 0
    total_consensus_candidates: int = 0
    total_baseline_verify_calls: int = 0
    total_consensus_verify_calls: int = 0
    total_consensus_prune_count: int = 0
    total_baseline_time: float = 0.0
    total_consensus_time: float = 0.0
    total_branches_pruned: int = 0
    total_branches_survived: int = 0
    prune_by_check: dict[str, int] = field(default_factory=dict)
    false_prune_tasks: list[str] = field(default_factory=list)
    per_task: list[dict] = field(default_factory=list)


def aggregate(results: list[TaskConsensusResult], dataset: str) -> ConsensusReport:
    report = ConsensusReport(dataset=dataset, n_tasks=len(results))

    for r in results:
        if r.baseline_solved:
            report.baseline_solves += 1
        if r.consensus_solved:
            report.consensus_solves += 1
        if r.consensus_solved and not r.baseline_solved:
            report.tasks_improved.append(r.task_id)
        if r.baseline_solved and not r.consensus_solved:
            report.tasks_worsened.append(r.task_id)
            report.false_prune_tasks.append(r.task_id)

        report.total_baseline_candidates += r.baseline_candidates
        report.total_consensus_candidates += r.consensus_candidates
        report.total_baseline_verify_calls += r.baseline_verify_calls
        report.total_consensus_verify_calls += r.consensus_verify_calls
        report.total_consensus_prune_count += r.consensus_prune_count
        report.total_baseline_time += r.baseline_time
        report.total_consensus_time += r.consensus_time
        report.total_branches_pruned += r.branches_pruned
        report.total_branches_survived += r.branches_survived

        for check_name, count in r.prune_checks.items():
            report.prune_by_check[check_name] = (
                report.prune_by_check.get(check_name, 0) + count
            )

        report.per_task.append({
            "task_id": r.task_id,
            "baseline_solved": r.baseline_solved,
            "consensus_solved": r.consensus_solved,
            "baseline_candidates": r.baseline_candidates,
            "consensus_candidates": r.consensus_candidates,
            "candidate_savings": r.candidate_savings(),
            "baseline_verify_calls": r.baseline_verify_calls,
            "consensus_verify_calls": r.consensus_verify_calls,
            "verify_savings": r.verify_savings(),
            "consensus_prune_count": r.consensus_prune_count,
            "baseline_time": r.baseline_time,
            "consensus_time": r.consensus_time,
            "time_savings": r.time_savings(),
            "branches_pruned": r.branches_pruned,
            "branches_survived": r.branches_survived,
            "prune_checks": r.prune_checks,
            "false_prune": r.false_prune,
        })

    return report


def format_report(report: ConsensusReport) -> str:
    lines = [
        "=" * 60,
        f"Consensus Evaluation: {report.dataset}",
        f"Tasks: {report.n_tasks}",
        "=" * 60,
        "",
        "--- Solve Count ---",
        f"  Baseline:  {report.baseline_solves}/{report.n_tasks}",
        f"  Consensus: {report.consensus_solves}/{report.n_tasks}",
        f"  Delta:     {report.consensus_solves - report.baseline_solves:+d}",
        f"  Improved:  {report.tasks_improved or 'none'}",
        f"  Worsened:  {report.tasks_worsened or 'none'}",
        "",
        "--- Candidate Counts (DSL search) ---",
        f"  Baseline total:  {report.total_baseline_candidates}",
        f"  Consensus total: {report.total_consensus_candidates}",
        f"  Savings:         {report.total_baseline_candidates - report.total_consensus_candidates:+d}",
        "",
        "--- Scene Verify Calls (where consensus prunes) ---",
        f"  Baseline total:  {report.total_baseline_verify_calls}",
        f"  Consensus total: {report.total_consensus_verify_calls}",
        f"  Savings:         {report.total_baseline_verify_calls - report.total_consensus_verify_calls:+d}",
        f"  Consensus prunes:{report.total_consensus_prune_count}",
        "",
        "--- Timing ---",
        f"  Baseline total:  {report.total_baseline_time:.1f}s",
        f"  Consensus total: {report.total_consensus_time:.1f}s",
        f"  Savings:         {report.total_baseline_time - report.total_consensus_time:+.1f}s",
        "",
        "--- Prune Rates ---",
        f"  Total branches pruned:   {report.total_branches_pruned}",
        f"  Total branches survived: {report.total_branches_survived}",
    ]

    if report.total_branches_pruned + report.total_branches_survived > 0:
        rate = report.total_branches_pruned / (
            report.total_branches_pruned + report.total_branches_survived
        )
        lines.append(f"  Prune rate: {rate:.1%}")

    lines.append("")
    lines.append("--- Prune by Check ---")
    for check_name, count in sorted(
        report.prune_by_check.items(), key=lambda x: -x[1],
    ):
        lines.append(f"  {check_name}: {count}")

    if report.false_prune_tasks:
        lines.append("")
        lines.append("--- FALSE PRUNES (baseline solved, consensus didn't) ---")
        for tid in report.false_prune_tasks:
            lines.append(f"  {tid}")
    else:
        lines.append("")
        lines.append("No false prunes detected.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stepwise all-demo consensus on ARC-2",
    )
    parser.add_argument(
        "--dataset", default="v2-eval", choices=dataset_names(),
    )
    parser.add_argument("--task", help="Single task ID")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--program-store", default=str(DEFAULT_PROGRAM_STORE))
    parser.add_argument("--library-store", default=str(DEFAULT_LIBRARY_STORE))
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    ps_path = Path(args.program_store)
    lib_path = Path(args.library_store)

    program_store = ProgramStore.load_json(ps_path)
    library = Library.load_json(lib_path)

    if args.task:
        task_ids = [args.task]
    else:
        task_ids = list_task_ids(ds)
        if args.limit:
            task_ids = task_ids[:args.limit]

    print(f"Dataset: {ds.name} ({len(task_ids)} tasks)")
    print(f"Program store: {ps_path} ({len(program_store)} programs)")
    print(f"Library: {lib_path} ({len(library.all_entries())} entries)")
    print("-" * 60)

    results: list[TaskConsensusResult] = []

    for i, tid in enumerate(task_ids):
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            print(f"  [{i+1}] {tid}: SKIP ({e})")
            continue

        r = eval_task_consensus(tid, task, library, program_store)
        results.append(r)

        base_status = "SOLVED" if r.baseline_solved else "      "
        cons_status = "SOLVED" if r.consensus_solved else "      "
        verify_delta = r.verify_savings()
        prune_flag = " FALSE-PRUNE" if r.false_prune else ""
        improved = " +NEW" if r.consensus_solved and not r.baseline_solved else ""
        print(
            f"  [{i+1}/{len(task_ids)}] {tid}: "
            f"base={base_status} cons={cons_status} "
            f"verify={r.baseline_verify_calls}->{r.consensus_verify_calls} "
            f"({verify_delta:+d}) "
            f"pruned={r.consensus_prune_count} "
            f"t={r.baseline_time:.1f}s->{r.consensus_time:.1f}s"
            f"{prune_flag}{improved}"
        )

        if args.verbose and r.consensus_trace_text:
            print(r.consensus_trace_text)

    # Ensure consensus is re-enabled
    set_consensus_enabled(True)

    report = aggregate(results, ds.name)
    print()
    print(format_report(report))

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / f"consensus_{ds.name}_{ts}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "dataset": report.dataset,
            "n_tasks": report.n_tasks,
            "baseline_solves": report.baseline_solves,
            "consensus_solves": report.consensus_solves,
            "tasks_improved": report.tasks_improved,
            "tasks_worsened": report.tasks_worsened,
            "total_baseline_candidates": report.total_baseline_candidates,
            "total_consensus_candidates": report.total_consensus_candidates,
            "total_baseline_verify_calls": report.total_baseline_verify_calls,
            "total_consensus_verify_calls": report.total_consensus_verify_calls,
            "total_consensus_prune_count": report.total_consensus_prune_count,
            "total_baseline_time": report.total_baseline_time,
            "total_consensus_time": report.total_consensus_time,
            "total_branches_pruned": report.total_branches_pruned,
            "total_branches_survived": report.total_branches_survived,
            "prune_by_check": report.prune_by_check,
            "false_prune_tasks": report.false_prune_tasks,
            "per_task": report.per_task,
        }, f, indent=2)
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
