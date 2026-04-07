#!/usr/bin/env python3
"""Evaluate the output-anchored backward solver prototype."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task
from aria.prototype.backward_solver import backward_solve


SLICE_TASKS = [
    "409aa875",
    "dd6b8c4b",
    "4c416de3",
    "dbff022c",
    "3dc255db",
    "135a2760",
    "7b80bb43",
    "58490d8a",
]


def main():
    ds = get_dataset("v2-eval")

    results = []
    t0 = time.time()

    for tid in SLICE_TASKS:
        task = load_arc_task(ds, tid)
        t1 = time.time()

        result = backward_solve(task.train, task_id=tid)
        elapsed = time.time() - t1

        # If train-verified, try on test
        test_solved = False
        test_diff = None
        if result.train_verified and result.shared_rule is not None:
            from aria.decomposition import detect_bg
            bg = detect_bg(task.train[0].input)
            test_outputs = []
            total_test_diff = 0
            all_test_ok = True
            for tp in task.test:
                predicted = result.shared_rule.apply(tp.input, bg)
                test_outputs.append(predicted)
                if predicted.shape == tp.output.shape:
                    d = int(np.sum(predicted != tp.output))
                    total_test_diff += d
                    if d > 0:
                        all_test_ok = False
                else:
                    all_test_ok = False
                    total_test_diff += tp.output.size
            test_solved = all_test_ok
            test_diff = total_test_diff

        mark = "TRAIN_VERIFIED" if result.train_verified else f"diff={result.train_diff}"
        test_mark = f" TEST_PASS" if test_solved else (f" test_diff={test_diff}" if test_diff is not None else "")
        strategy = result.details.get("strategy", "?")
        rule_op = result.shared_rule.op if result.shared_rule else "none"

        print(
            f"{tid}: {mark:>20} rule={rule_op:<20} "
            f"strategy={strategy:<30}{test_mark} ({elapsed:.1f}s)"
        )

        results.append({
            "tid": tid,
            "train_verified": result.train_verified,
            "train_diff": result.train_diff,
            "test_solved": test_solved,
            "test_diff": test_diff,
            "rule_op": rule_op,
            "strategy": strategy,
            "elapsed": round(elapsed, 1),
        })

    total_time = time.time() - t0

    n_verified = sum(1 for r in results if r["train_verified"])
    n_test = sum(1 for r in results if r["test_solved"])
    n_tasks = len(results)

    print(f"\n{'='*70}")
    print(f"BACKWARD SOLVER PROTOTYPE RESULTS")
    print(f"{'='*70}")
    print(f"Tasks:               {n_tasks}")
    print(f"Train verified:      {n_verified}/{n_tasks} ({n_verified/n_tasks*100:.0f}%)")
    print(f"Exact solve (test):  {n_test}/{n_tasks}")
    print(f"Time:                {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
