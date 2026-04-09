"""Evaluate verifier-guided repair on real ARC-2 tasks.

Reports:
- near-miss coverage
- exact solve recoveries
- per-generator effectiveness
- cost vs gain
- failure analysis
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from aria.core.scene_executor import execute_scene_program
from aria.core.scene_solve import (
    _NEAR_MISS_SINK,
    infer_and_repair_scene_programs,
    infer_scene_programs,
    verify_scene_program,
)
from aria.datasets import get_dataset, iter_tasks, list_task_ids
from aria.repair import (
    RepairResult,
    RepairTrace,
    repair_near_misses,
    score_scene_candidate,
)
from aria.scene_ir import SceneProgram
from aria.types import DemoPair, Task


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------


@dataclass
class TaskRepairResult:
    task_id: str
    baseline_solved: bool  # solved by infer_scene_programs (no repair)
    repair_solved: bool
    near_misses_collected: int
    near_misses_scored: int  # those that passed threshold
    repair_result: RepairResult | None = None
    baseline_candidates: int = 0
    repair_verify_calls: int = 0
    best_near_miss_accuracy: float = 0.0
    failure_reason: str = ""
    winning_generators: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_task(task_id: str, task: Task) -> TaskRepairResult:
    """Evaluate repair on one task."""
    demos = task.train

    # --- Baseline: infer without repair ---
    baseline_progs = infer_scene_programs(demos)
    baseline_solved = any(verify_scene_program(p, demos) for p in baseline_progs)

    if baseline_solved:
        return TaskRepairResult(
            task_id=task_id,
            baseline_solved=True,
            repair_solved=True,  # already solved
            near_misses_collected=0,
            near_misses_scored=0,
            baseline_candidates=len(baseline_progs),
        )

    # --- Collect near-misses via infer_and_repair path ---
    repair_progs = infer_and_repair_scene_programs(demos)
    repair_solved = len(repair_progs) > 0 and verify_scene_program(repair_progs[0], demos)

    # Also do manual near-miss analysis for reporting
    # Re-collect near-misses by running the sink manually
    import aria.core.scene_solve as ss
    ss._NEAR_MISS_SINK = []
    _ = infer_scene_programs(demos)
    collected = list(ss._NEAR_MISS_SINK)
    ss._NEAR_MISS_SINK = None

    # Score near-misses
    scored_near_misses = []
    best_accuracy = 0.0
    for prog in collected:
        _, diag = score_scene_candidate(prog, demos)
        if diag.all_dims_match and diag.execution_errors == 0:
            scored_near_misses.append((prog, diag))
            best_accuracy = max(best_accuracy, diag.pixel_accuracy)

    # Run repair with detailed trace for generator analysis
    repair_result = None
    winning_gens: list[str] = []
    if scored_near_misses:
        progs_to_repair = [p for p, _ in scored_near_misses]
        repair_result = repair_near_misses(
            progs_to_repair, demos,
            near_miss_threshold=0.0,  # try all near-misses with correct dims
        )
        if repair_result.solved:
            # Find which generator(s) produced the winning edit
            for trace in repair_result.traces:
                if trace.solved:
                    for rnd in trace.rounds:
                        for et in rnd.edits_tried:
                            if et.exact:
                                winning_gens.append(et.action.reason)

    # Failure classification
    failure_reason = ""
    if not repair_solved:
        if not collected:
            failure_reason = "no_candidates_reached_verify"
        elif not scored_near_misses:
            failure_reason = "no_near_miss_with_correct_dims"
        elif best_accuracy < 0.5:
            failure_reason = "residual_too_global"
        elif best_accuracy >= 0.85:
            failure_reason = "near_miss_but_repair_failed"
        else:
            failure_reason = "moderate_diff_repair_insufficient"

    return TaskRepairResult(
        task_id=task_id,
        baseline_solved=baseline_solved,
        repair_solved=repair_solved,
        near_misses_collected=len(collected),
        near_misses_scored=len(scored_near_misses),
        repair_result=repair_result,
        baseline_candidates=len(baseline_progs),
        repair_verify_calls=repair_result.total_verify_calls if repair_result else 0,
        best_near_miss_accuracy=best_accuracy,
        failure_reason=failure_reason,
        winning_generators=winning_gens,
    )


def run_evaluation(dataset_name: str = "v2-eval", limit: int = 0):
    """Run full evaluation and print report."""
    ds = get_dataset(dataset_name)
    task_ids = list_task_ids(ds)
    if limit > 0:
        task_ids = task_ids[:limit]

    print(f"Evaluating repair on {dataset_name}: {len(task_ids)} tasks")
    print("=" * 70)

    results: list[TaskRepairResult] = []
    generator_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"proposed": 0, "improved": 0, "exact": 0, "residual_reduction": 0}
    )
    total_time = 0.0

    for i, (task_id, task) in enumerate(iter_tasks(ds, task_ids=task_ids)):
        t0 = time.time()
        try:
            r = evaluate_task(task_id, task)
        except Exception as e:
            print(f"  [{i+1}/{len(task_ids)}] {task_id}: ERROR {e}")
            results.append(TaskRepairResult(
                task_id=task_id,
                baseline_solved=False,
                repair_solved=False,
                near_misses_collected=0,
                near_misses_scored=0,
                failure_reason=f"error: {e}",
            ))
            continue
        elapsed = time.time() - t0
        total_time += elapsed

        results.append(r)

        # Collect generator stats from traces
        if r.repair_result:
            for trace in r.repair_result.traces:
                for rnd in trace.rounds:
                    for et in rnd.edits_tried:
                        gen_name = _generator_name(et.action)
                        generator_stats[gen_name]["proposed"] += 1
                        if et.improved:
                            generator_stats[gen_name]["improved"] += 1
                        if et.exact:
                            generator_stats[gen_name]["exact"] += 1

        # Progress
        status = "BASELINE" if r.baseline_solved else ("REPAIRED" if r.repair_solved else "FAILED")
        nm_info = f"nm={r.near_misses_collected}" if r.near_misses_collected else ""
        acc_info = f"best={r.best_near_miss_accuracy:.1%}" if r.best_near_miss_accuracy > 0 else ""
        cost_info = f"vc={r.repair_verify_calls}" if r.repair_verify_calls > 0 else ""
        detail = " ".join(filter(None, [nm_info, acc_info, cost_info]))
        print(f"  [{i+1}/{len(task_ids)}] {task_id}: {status} {detail} ({elapsed:.1f}s)")

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("REPAIR EVALUATION REPORT")
    print("=" * 70)

    n = len(results)
    baseline_solved = sum(1 for r in results if r.baseline_solved)
    repair_solved = sum(1 for r in results if r.repair_solved)
    repair_only = sum(1 for r in results if r.repair_solved and not r.baseline_solved)
    had_near_misses = sum(1 for r in results if r.near_misses_collected > 0)
    had_scored_near_misses = sum(1 for r in results if r.near_misses_scored > 0)
    total_near_misses = sum(r.near_misses_collected for r in results)
    total_scored = sum(r.near_misses_scored for r in results)
    total_verify = sum(r.repair_verify_calls for r in results)

    print(f"\nDataset: {dataset_name} ({n} tasks)")
    print(f"Baseline solves (scene programs): {baseline_solved}/{n}")
    print(f"After repair: {repair_solved}/{n} (+{repair_only} from repair)")
    print(f"Tasks with near-miss candidates: {had_near_misses}/{n}")
    print(f"Tasks with scored near-misses (dims ok): {had_scored_near_misses}/{n}")
    print(f"Total near-miss candidates collected: {total_near_misses}")
    print(f"Total scored near-misses: {total_scored}")
    print(f"Total repair verify calls: {total_verify}")
    print(f"Total evaluation time: {total_time:.1f}s")

    if repair_only > 0:
        print(f"\nSolve gain per verify call: {repair_only / max(total_verify, 1):.4f}")

    # Generator effectiveness
    print(f"\n{'Generator':<30} {'Proposed':>8} {'Improved':>8} {'Exact':>8} {'Hit%':>8}")
    print("-" * 70)
    for gen_name in sorted(generator_stats, key=lambda g: -generator_stats[g]["exact"]):
        gs = generator_stats[gen_name]
        hit_pct = gs["improved"] / max(gs["proposed"], 1) * 100
        print(f"  {gen_name:<28} {gs['proposed']:>8} {gs['improved']:>8} {gs['exact']:>8} {hit_pct:>7.1f}%")

    # Failure analysis
    failure_counts = Counter(r.failure_reason for r in results if not r.repair_solved)
    print(f"\nFailure analysis ({n - repair_solved} unsolved):")
    for reason, count in failure_counts.most_common():
        if reason:
            print(f"  {reason}: {count}")

    # Near-miss accuracy distribution
    accs = [r.best_near_miss_accuracy for r in results if r.best_near_miss_accuracy > 0]
    if accs:
        print(f"\nNear-miss accuracy distribution (n={len(accs)}):")
        for threshold in [0.5, 0.7, 0.85, 0.9, 0.95, 0.99]:
            above = sum(1 for a in accs if a >= threshold)
            print(f"  >= {threshold:.0%}: {above}")

    # Detailed repair wins
    if repair_only > 0:
        print(f"\nRepair wins ({repair_only} tasks):")
        for r in results:
            if r.repair_solved and not r.baseline_solved:
                gens = ", ".join(r.winning_generators) if r.winning_generators else "unknown"
                print(f"  {r.task_id}: acc={r.best_near_miss_accuracy:.1%} vc={r.repair_verify_calls} gen={gens}")


def _generator_name(action) -> str:
    """Classify a repair action into its generator name."""
    if action.insert_step is not None:
        return "insert_recolor_step"
    if action.param_name == "predicate":
        return "swap_selector_predicate"
    if action.param_name == "rank":
        return "change_selector_rank"
    if action.param_name == "kind":
        return "change_selector_kind"
    if action.param_name == "transform":
        return "swap_transform"
    if action.param_name == "operation":
        return "swap_boolean_op"
    if action.param_name == "fill_color":
        return "change_fill_color"
    if action.param_name == "scope":
        return "swap_recolor_scope"
    if action.param_name in ("from_color", "to_color"):
        return "adjust_color_map"
    if action.param_name == "property":
        return "swap_map_property"
    if action.param_name == "connectivity":
        return "change_connectivity"
    if action.param_name == "rule":
        return "swap_for_each_rule"
    if action.param_name == "require_non_bg":
        return "toggle_require_non_bg"
    return f"other:{action.param_name}"


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "v2-eval"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_evaluation(dataset, limit)
