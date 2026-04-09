#!/usr/bin/env python3
"""Evaluate the Layered Workspace Solver."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task, list_task_ids
from aria.decomposition import detect_bg
from aria.lws.solver import lws_solve, lws_predict


def main():
    ds = get_dataset("v2-eval")
    task_ids = list_task_ids(ds)[:120]

    results = []
    t0 = time.time()

    for tid in task_ids:
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            continue

        t1 = time.time()
        result = lws_solve(task.train, task_id=tid)
        elapsed = time.time() - t1

        test_solved = False
        if result.train_verified and result.unified_rule is not None:
            all_ok = True
            for tp in task.test:
                predicted = lws_predict(result.unified_rule, tp.input)
                if predicted is None or not np.array_equal(predicted, tp.output):
                    all_ok = False
                    break
            test_solved = all_ok

        results.append({
            "tid": tid,
            "train_verified": result.train_verified,
            "test_solved": test_solved,
            "strategy": result.details.get("strategy", "none"),
            "elapsed": round(elapsed, 2),
        })

        if result.train_verified:
            mark = " TEST_PASS" if test_solved else " test_FAIL"
            print(f"{tid}: TRAIN_VERIFIED  strategy={result.details.get('strategy','?')}{mark} ({elapsed:.2f}s)")

    total_time = time.time() - t0
    n = len(results)
    n_verified = sum(1 for r in results if r["train_verified"])
    n_test = sum(1 for r in results if r["test_solved"])

    print(f"\n{'='*70}")
    print(f"LAYERED WORKSPACE SOLVER — {n} tasks")
    print(f"{'='*70}")
    print(f"Train verified:  {n_verified}/{n} ({n_verified/n*100:.1f}%)")
    print(f"Exact solve:     {n_test}/{n}")
    print(f"Time:            {total_time:.1f}s")
    print(f"{'='*70}")

    # Strategy breakdown
    strats = [r["strategy"] for r in results if r["train_verified"]]
    from collections import Counter
    for s, cnt in Counter(strats).most_common():
        test_cnt = sum(1 for r in results if r["strategy"] == s and r["test_solved"])
        print(f"  {s}: {cnt} train_verified, {test_cnt} test_solved")


if __name__ == "__main__":
    main()
