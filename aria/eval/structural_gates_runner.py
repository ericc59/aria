"""Structural gates evaluation runner.

Executes the solver/trace pipeline for annotated tasks, computes all
six gates, and produces a compact report for branch comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aria.datasets import DatasetInfo, get_dataset, load_arc_task
from aria.eval.structural_gates_schema import GoldTask, load_gold_tasks
from aria.eval.structural_gates_scorer import (
    GATE_ORDER,
    TaskGateResults,
    score_all_gates,
)
from aria.eval.structural_gates_trace import StageArtifacts, extract_stage_artifacts
from aria.types import grid_eq


# ---------------------------------------------------------------------------
# Runner result types
# ---------------------------------------------------------------------------


@dataclass
class TaskRunResult:
    task_id: str
    gate_results: TaskGateResults
    artifacts: StageArtifacts
    exact_solve: bool
    elapsed_sec: float
    solve_path: str | None = None  # which path solved, if exact_solve


@dataclass
class RunReport:
    results: list[TaskRunResult]
    dataset: str
    top_k: int
    elapsed_total_sec: float

    @property
    def n_tasks(self) -> int:
        return len(self.results)

    @property
    def exact_solve_count(self) -> int:
        return sum(1 for r in self.results if r.exact_solve)

    def gate_pass_rate(self, gate_name: str) -> float:
        relevant = [
            r for r in self.results
            if r.gate_results.gate_by_name(gate_name) is not None
        ]
        if not relevant:
            return 0.0
        passed = sum(
            1 for r in relevant
            if r.gate_results.gate_by_name(gate_name).passed  # type: ignore
        )
        return passed / len(relevant)

    def gate_recall(self, gate_name: str) -> float:
        relevant = [
            r for r in self.results
            if r.gate_results.gate_by_name(gate_name) is not None
        ]
        if not relevant:
            return 0.0
        scores = [
            r.gate_results.gate_by_name(gate_name).score  # type: ignore
            for r in relevant
        ]
        return sum(scores) / len(scores)

    def most_common_failing_gate(self) -> str | None:
        from collections import Counter
        failures: list[str] = []
        for r in self.results:
            first_fail = r.gate_results.first_failing_gate
            if first_fail:
                failures.append(first_fail)
        if not failures:
            return None
        return Counter(failures).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _try_exact_solve(
    task_id: str,
    ds: DatasetInfo,
) -> tuple[bool, str | None]:
    """Run the canonical active solver and check for exact match on test.

    Uses solve_arc_task (the real active stack including stage-1 gating,
    scene_solve, correspondence, and editor search) rather than the old
    refinement-only pipeline.

    Returns (solved, solve_path).
    """
    import numpy as np

    task = load_arc_task(ds, task_id)

    # Canonical active solver
    try:
        from aria.core.arc import solve_arc_task
        result = solve_arc_task(task.train, task_id=task_id)
        if result.solved and result.winning_program is not None:
            if _verify_on_test(result.winning_program, task):
                return True, "solve_arc_task"
    except Exception:
        pass

    # Scene-solve programs verified on train may generalize to test.
    # solve_arc_task wraps CorrespondencePrograms into Program stubs
    # that can't be executed by the runtime, so we also try the raw
    # scene_solve path which keeps the original executable objects.
    try:
        from aria.core.scene_solve import infer_scene_programs, verify_scene_program
        candidates = infer_scene_programs(task.train)
        for prog in candidates:
            if verify_scene_program(prog, task.train):
                if _verify_scene_prog_on_test(prog, task):
                    return True, "scene_solve"
    except Exception:
        pass

    # Fallback: old refinement pipeline
    try:
        from aria.library.store import Library
        from aria.solver import solve_task

        lib = Library()
        old_result = solve_task(
            task,
            lib,
            task_id=task_id,
            max_search_steps=3,
            max_search_candidates=10000,
            max_refinement_rounds=2,
        )
        if old_result.solved:
            for test_pair, predicted in zip(task.test, old_result.test_outputs):
                if not grid_eq(test_pair.output, predicted):
                    return False, None
            return True, "refinement_loop"
    except Exception:
        pass

    return False, None


def _verify_on_test(prog, task) -> bool:
    """Verify a program on test pairs."""
    for test_pair in task.test:
        try:
            if hasattr(prog, 'execute'):
                predicted = prog.execute(test_pair.input)
            else:
                from aria.runtime.executor import execute
                from aria.verify.mode import detect_mode
                from aria.types import TaskContext, VerifyMode
                mode = detect_mode(prog)
                ctx = None if mode == VerifyMode.STATELESS else TaskContext(demos=task.train)
                predicted = execute(prog, test_pair.input, ctx)
            if not grid_eq(test_pair.output, predicted):
                return False
        except Exception:
            return False
    return True


def _verify_scene_prog_on_test(prog, task) -> bool:
    """Verify a scene/correspondence program on test pairs."""
    import numpy as np

    for test_pair in task.test:
        try:
            if hasattr(prog, 'execute'):
                predicted = prog.execute(test_pair.input)
            elif hasattr(prog, 'verify_on_demo'):
                # CorrespondenceProgram: verify_on_demo checks exact match
                if not prog.verify_on_demo(test_pair.input, test_pair.output):
                    return False
                continue
            else:
                from aria.core.scene_executor import execute_scene_program
                predicted = execute_scene_program(prog, test_pair.input)
            if not np.array_equal(predicted, test_pair.output):
                return False
        except Exception:
            return False
    return True


def run_structural_gates(
    gold_path: str | Path,
    dataset_name: str = "v2-train",
    top_k: int = 5,
    run_exact_solve: bool = True,
    task_ids: list[str] | None = None,
) -> RunReport:
    """Run the structural gates evaluation on annotated tasks.

    Args:
        gold_path: Path to the gold annotations YAML file.
        dataset_name: Dataset to load tasks from.
        top_k: K for recall@K metrics.
        run_exact_solve: Whether to also run the full solver for exact solve.
        task_ids: If given, only evaluate these task IDs (must be in gold).
    """
    gold_tasks = load_gold_tasks(gold_path)
    ds = get_dataset(dataset_name)

    if task_ids:
        gold_tasks = [g for g in gold_tasks if g.task_id in set(task_ids)]

    results: list[TaskRunResult] = []
    t0 = time.time()

    for gold in gold_tasks:
        t_task = time.time()
        try:
            task = load_arc_task(ds, gold.task_id)
        except FileNotFoundError:
            # Try other datasets
            for alt_ds_name in ["v2-eval", "v1-train", "v1-eval"]:
                try:
                    ds_alt = get_dataset(alt_ds_name)
                    task = load_arc_task(ds_alt, gold.task_id)
                    break
                except (FileNotFoundError, ValueError):
                    continue
            else:
                print(f"  SKIP {gold.task_id}: task not found in any dataset")
                continue

        # Extract stage artifacts
        artifacts = extract_stage_artifacts(gold.task_id, task.train)

        # Score gates
        gate_results = score_all_gates(gold, artifacts, top_k=top_k)

        # Exact solve (optional, expensive)
        exact_solve = False
        solve_path = None
        if run_exact_solve:
            try:
                exact_solve, solve_path = _try_exact_solve(gold.task_id, ds)
            except Exception:
                pass

        gate_results = TaskGateResults(
            task_id=gold.task_id,
            gates=gate_results.gates,
            exact_solve=exact_solve,
        )

        elapsed = time.time() - t_task
        results.append(TaskRunResult(
            task_id=gold.task_id,
            gate_results=gate_results,
            artifacts=artifacts,
            exact_solve=exact_solve,
            elapsed_sec=round(elapsed, 2),
            solve_path=solve_path,
        ))

    return RunReport(
        results=results,
        dataset=dataset_name,
        top_k=top_k,
        elapsed_total_sec=round(time.time() - t0, 2),
    )
