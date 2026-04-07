#!/usr/bin/env python3
"""A/B evaluation: factored retrieval vs baseline on ARC-2.

Pass 1: Baseline solve (no factored retrieval)
Pass 2: Populate FactoredMemoryStore from pass-1 solves
Pass 3: Re-run with factored retrieval enabled
Report: per-mechanism breakdown of where retrieval helped/hurt
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from aria.datasets import get_dataset, iter_tasks
from aria.factored_memory import FactoredMemoryStore
from aria.factored_retrieval import ingest_solve_record
from aria.factored_trace import format_factored_retrieval_trace
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.solver import SolveResult, solve_task


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "v2-train"
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Loading {dataset_name}...")
    ds = get_dataset(dataset_name)

    # Collect tasks
    tasks: dict[str, object] = {}
    for tid, task in iter_tasks(ds, limit=max_tasks):
        tasks[tid] = task
    print(f"Loaded {len(tasks)} tasks")

    # ---- Pass 1: Baseline ----
    print("\n=== Pass 1: Baseline (no factored retrieval) ===")
    baseline_results: dict[str, SolveResult] = {}
    library = Library()
    program_store = ProgramStore()
    t0 = time.time()

    for i, (task_id, task) in enumerate(sorted(tasks.items())):
        result = solve_task(
            task, library, program_store,
            task_id=task_id,
            max_search_steps=3,
            max_search_candidates=5000,
            max_refinement_rounds=2,
        )
        baseline_results[task_id] = result
        if (i + 1) % 50 == 0:
            solved = sum(1 for r in baseline_results.values() if r.solved)
            print(f"  [{i+1}/{len(tasks)}] solved={solved}")

    baseline_solved = {tid for tid, r in baseline_results.items() if r.solved}
    baseline_time = time.time() - t0
    print(f"  Baseline: {len(baseline_solved)}/{len(tasks)} solved in {baseline_time:.1f}s")

    # ---- Pass 2: Populate factored memory ----
    print("\n=== Pass 2: Populating factored memory store ===")
    factored_store = FactoredMemoryStore()
    for task_id, result in baseline_results.items():
        if result.solved and result.winning_program is not None:
            task = tasks[task_id]
            ingest_solve_record(
                task.train,
                result.winning_program,
                task_id=task_id,
                source="baseline",
                factored_store=factored_store,
            )
    print(f"  Ingested {len(factored_store)} factored records from {len(baseline_solved)} solves")

    # ---- Pass 3: Retrieval-biased ----
    print("\n=== Pass 3: Factored retrieval enabled ===")
    retrieval_results: dict[str, SolveResult] = {}
    library2 = Library()
    program_store2 = ProgramStore()
    t0 = time.time()

    for i, (task_id, task) in enumerate(sorted(tasks.items())):
        result = solve_task(
            task, library2, program_store2,
            task_id=task_id,
            max_search_steps=3,
            max_search_candidates=5000,
            max_refinement_rounds=2,
            factored_store=factored_store,
        )
        retrieval_results[task_id] = result
        if (i + 1) % 50 == 0:
            solved = sum(1 for r in retrieval_results.values() if r.solved)
            print(f"  [{i+1}/{len(tasks)}] solved={solved}")

    retrieval_solved = {tid for tid, r in retrieval_results.items() if r.solved}
    retrieval_time = time.time() - t0
    print(f"  Retrieval: {len(retrieval_solved)}/{len(tasks)} solved in {retrieval_time:.1f}s")

    # ---- Report ----
    print("\n" + "=" * 70)
    print("FACTORED RETRIEVAL A/B COMPARISON")
    print("=" * 70)
    print(f"Dataset: {dataset_name} ({len(tasks)} tasks)")
    print(f"Baseline solves: {len(baseline_solved)}")
    print(f"Retrieval solves: {len(retrieval_solved)}")

    gained = retrieval_solved - baseline_solved
    lost = baseline_solved - retrieval_solved
    print(f"Gained: {len(gained)} tasks: {sorted(gained)[:10]}")
    print(f"Lost: {len(lost)} tasks: {sorted(lost)[:10]}")
    print(f"Net: {len(retrieval_solved) - len(baseline_solved):+d}")

    # Candidate count comparison
    both_solved = baseline_solved & retrieval_solved
    if both_solved:
        baseline_cands = sum(
            baseline_results[tid].search_candidates_tried for tid in both_solved
        )
        retrieval_cands = sum(
            retrieval_results[tid].search_candidates_tried for tid in both_solved
        )
        print(f"\nCandidate counts (tasks solved by both, n={len(both_solved)}):")
        print(f"  Baseline total:   {baseline_cands}")
        print(f"  Retrieval total:  {retrieval_cands}")
        if baseline_cands > 0:
            print(f"  Reduction:        {(1 - retrieval_cands / baseline_cands) * 100:.1f}%")

    # ---- Per-mechanism breakdown ----
    print("\n" + "-" * 70)
    print("PER-MECHANISM BREAKDOWN")
    print("-" * 70)

    decomp_bias_count = 0
    decomp_bias_changed = 0
    op_bias_count = 0
    repair_bias_count = 0
    tasks_with_retrieval = 0

    for tid, result in retrieval_results.items():
        rr = result.refinement_result
        if rr is None or rr.factored_retrieval_trace is None:
            continue
        trace = rr.factored_retrieval_trace
        if trace.matches_returned == 0:
            continue
        tasks_with_retrieval += 1
        if trace.decomp_bias_active:
            decomp_bias_count += 1
        if trace.decomp_bias_changed_order:
            decomp_bias_changed += 1
        if trace.op_bias_count > 0:
            op_bias_count += 1
        if trace.repair_bias_active:
            repair_bias_count += 1

    print(f"Tasks with retrieval matches: {tasks_with_retrieval}/{len(tasks)}")
    print(f"  Decomposition bias active: {decomp_bias_count}")
    print(f"    Changed decomp order:    {decomp_bias_changed}")
    print(f"  Op family bias active:     {op_bias_count}")
    print(f"  Repair path bias active:   {repair_bias_count}")

    # Per-mechanism solve improvement
    _print_mechanism_impact(
        "Decomposition bias",
        retrieval_results, baseline_results,
        lambda trace: trace.decomp_bias_active,
    )
    _print_mechanism_impact(
        "Op family bias",
        retrieval_results, baseline_results,
        lambda trace: trace.op_bias_count > 0,
    )
    _print_mechanism_impact(
        "Repair path bias",
        retrieval_results, baseline_results,
        lambda trace: trace.repair_bias_active,
    )

    # ---- Gained task traces ----
    if gained:
        print(f"\n--- Retrieval traces for gained tasks ---")
        for tid in sorted(gained)[:5]:
            result = retrieval_results[tid]
            if (result.refinement_result
                    and result.refinement_result.factored_retrieval_trace):
                trace = result.refinement_result.factored_retrieval_trace
                print(f"\n  {tid}:")
                for line in format_factored_retrieval_trace(trace).splitlines():
                    print(f"    {line}")

    # ---- Bad retrieval analysis ----
    if lost:
        print(f"\n--- Analysis of lost tasks ---")
        for tid in sorted(lost)[:5]:
            result = retrieval_results[tid]
            baseline_r = baseline_results[tid]
            print(f"\n  {tid}:")
            print(f"    Baseline candidates: {baseline_r.search_candidates_tried}")
            print(f"    Retrieval candidates: {result.search_candidates_tried}")
            if (result.refinement_result
                    and result.refinement_result.factored_retrieval_trace):
                trace = result.refinement_result.factored_retrieval_trace
                print(f"    Matches: {trace.matches_returned}, top score: {trace.top_match_score:.1f}")
                if trace.factors_borrowed:
                    print(f"    Borrowed: {', '.join(trace.factors_borrowed)}")

    # Neutral task analysis
    neutral = set(tasks.keys()) - gained - lost - (baseline_solved ^ retrieval_solved)
    neutral_with_retrieval = 0
    for tid in neutral:
        result = retrieval_results[tid]
        if (result.refinement_result
                and result.refinement_result.factored_retrieval_trace
                and result.refinement_result.factored_retrieval_trace.matches_returned > 0):
            neutral_with_retrieval += 1
    print(f"\nNeutral tasks with retrieval matches: {neutral_with_retrieval}")

    # Save store
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    store_path = output_dir / f"factored_memory_{dataset_name}.json"
    factored_store.save_json(store_path)
    print(f"\nFactored memory store saved to {store_path}")


def _print_mechanism_impact(
    name: str,
    retrieval_results: dict[str, SolveResult],
    baseline_results: dict[str, SolveResult],
    predicate,
) -> None:
    """Print per-mechanism solve impact."""
    active_tasks: set[str] = set()
    for tid, result in retrieval_results.items():
        rr = result.refinement_result
        if rr is None or rr.factored_retrieval_trace is None:
            continue
        trace = rr.factored_retrieval_trace
        if predicate(trace):
            active_tasks.add(tid)

    if not active_tasks:
        return

    gained_here = sum(
        1 for tid in active_tasks
        if retrieval_results[tid].solved and not baseline_results[tid].solved
    )
    lost_here = sum(
        1 for tid in active_tasks
        if not retrieval_results[tid].solved and baseline_results[tid].solved
    )
    both = sum(
        1 for tid in active_tasks
        if retrieval_results[tid].solved and baseline_results[tid].solved
    )

    # Candidate reduction for tasks solved by both
    if both > 0:
        b_cands = sum(
            baseline_results[tid].search_candidates_tried
            for tid in active_tasks
            if retrieval_results[tid].solved and baseline_results[tid].solved
        )
        r_cands = sum(
            retrieval_results[tid].search_candidates_tried
            for tid in active_tasks
            if retrieval_results[tid].solved and baseline_results[tid].solved
        )
        reduction = (1 - r_cands / b_cands) * 100 if b_cands > 0 else 0.0
    else:
        reduction = 0.0

    print(f"\n  {name} (active on {len(active_tasks)} tasks):")
    print(f"    Gained: {gained_here}, Lost: {lost_here}, Both: {both}")
    if both > 0:
        print(f"    Candidate reduction (both-solved): {reduction:.1f}%")


if __name__ == "__main__":
    main()
