#!/usr/bin/env python3
"""Evaluate the next-generation solver prototype.

Reports:
- train_verified_rate: fraction of tasks where unified rule matches all train demos
- exact_solve_rate: fraction of tasks where rule solves test
- per_demo_coverage: fraction of demos with at least one explanation
- graph statistics
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task, list_task_ids
from aria.decomposition import detect_bg
from aria.ngs.execute import ngs_solve, ngs_predict
from aria.ngs.output_units import _whole_unit
from aria.ngs.backward_explain import explain_unit


def main():
    ds = get_dataset("v2-eval")
    task_ids = list_task_ids(ds)[:120]

    results = []
    total_demos = 0
    explained_demos = 0
    t0 = time.time()

    for tid in task_ids:
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            print(f"{tid}: LOAD_ERROR {e}")
            continue

        t1 = time.time()
        result = ngs_solve(task.train, task_id=tid)
        elapsed = time.time() - t1

        # Count per-demo coverage
        demo_coverage = 0
        for d in task.train:
            total_demos += 1
            bg = detect_bg(d.input)
            if d.input.shape != d.output.shape:
                continue
            unit = _whole_unit(d.input, d.output, bg)
            exps = explain_unit(d.input, d.output, unit, bg)
            if exps:
                explained_demos += 1
                demo_coverage += 1

        # If train-verified, try test
        test_solved = False
        test_diff = None
        if result.train_verified and result.unified_rule is not None:
            total_test_diff = 0
            all_test_ok = True
            for tp in task.test:
                predicted = ngs_predict(result.unified_rule, tp.input)
                if predicted is not None:
                    if predicted.shape == tp.output.shape:
                        d = int(np.sum(predicted != tp.output))
                        total_test_diff += d
                        if d > 0:
                            all_test_ok = False
                    else:
                        all_test_ok = False
                        total_test_diff += tp.output.size
                else:
                    all_test_ok = False
                    total_test_diff += tp.output.size
            test_solved = all_test_ok
            test_diff = total_test_diff

        strategy = result.details.get("strategy", "?")
        desc = result.details.get("description", "none")

        results.append({
            "tid": tid,
            "train_verified": result.train_verified,
            "train_diff": result.train_diff,
            "test_solved": test_solved,
            "strategy": strategy,
            "description": desc,
            "elapsed": round(elapsed, 2),
            "demo_coverage": demo_coverage,
            "total_demos": len(task.train),
        })

        # Only print tasks with partial coverage
        if demo_coverage > 0 or result.train_verified:
            mark = "TRAIN_VERIFIED" if result.train_verified else f"diff={result.train_diff}"
            test_mark = " TEST_PASS" if test_solved else ""
            print(f"{tid}: {mark:>20} cover={demo_coverage}/{len(task.train)} "
                  f"desc={desc:<25}{test_mark} ({elapsed:.2f}s)")

    total_time = time.time() - t0
    n_verified = sum(1 for r in results if r["train_verified"])
    n_test = sum(1 for r in results if r["test_solved"])
    n_tasks = len(results)
    n_any_cover = sum(1 for r in results if r["demo_coverage"] > 0)
    n_full_cover = sum(1 for r in results if r["demo_coverage"] == r["total_demos"])

    print(f"\n{'='*80}")
    print(f"NGS PROTOTYPE RESULTS — {n_tasks} tasks, {total_demos} demos")
    print(f"{'='*80}")
    print(f"Train verified:           {n_verified}/{n_tasks} ({n_verified/n_tasks*100:.0f}%)")
    print(f"Exact solve (test):       {n_test}/{n_tasks}")
    print(f"Per-demo coverage:        {explained_demos}/{total_demos} ({explained_demos/total_demos*100:.1f}%)")
    print(f"Tasks with any coverage:  {n_any_cover}/{n_tasks}")
    print(f"Tasks with full coverage: {n_full_cover}/{n_tasks}")
    print(f"Time:                     {total_time:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
