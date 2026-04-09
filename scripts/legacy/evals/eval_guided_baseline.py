#!/usr/bin/env python3
"""Evaluate multi-step search on the synthetic benchmark."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.guided.synthetic import generate_benchmark, split_benchmark
from aria.guided.search import search, predict_test


def main():
    print("Generating synthetic benchmark (10 types including multi-step)...")
    all_tasks = generate_benchmark(n_tasks=200, seed=42)
    train_tasks, test_tasks = split_benchmark(all_tasks, 0.7)

    types = defaultdict(int)
    for t in all_tasks:
        types[t.rule_type] += 1
    print(f"  Total: {len(all_tasks)}, Train: {len(train_tasks)}, Test: {len(test_tasks)}")
    print(f"  Types: {dict(types)}")

    multi_step = [t for t in test_tasks if t.rule_type in ('fill_then_recolor', 'delete_then_symmetrize', 'periodic_then_fill')]
    single_step = [t for t in test_tasks if t not in multi_step]
    print(f"  Test single-step: {len(single_step)}, multi-step: {len(multi_step)}")

    print(f"\nRunning multi-step search on {len(test_tasks)} held-out tasks...")
    t0 = time.time()

    per_type = defaultdict(lambda: {"train": 0, "test": 0, "cands": [], "total": 0})

    for task in test_tasks:
        g = per_type[task.rule_type]
        g["total"] += 1

        result = search(task.train, max_candidates=3000, max_steps=3)
        g["cands"].append(result.candidates_tried)

        if result.solved:
            g["train"] += 1
            # Verify on test
            all_ok = True
            for test_inp, test_out in task.test:
                pred = predict_test(result.program, test_inp)
                if pred is None or not np.array_equal(pred, test_out):
                    all_ok = False
                    break
            if all_ok:
                g["test"] += 1

    total_time = time.time() - t0

    n = len(test_tasks)
    total_train = sum(g["train"] for g in per_type.values())
    total_test = sum(g["test"] for g in per_type.values())
    all_cands = [c for g in per_type.values() for c in g["cands"]]

    print(f"\n{'='*70}")
    print(f"MULTI-STEP SEARCH BASELINE — {n} held-out tasks")
    print(f"{'='*70}")
    print(f"Train verified:  {total_train}/{n} ({total_train/n*100:.0f}%)")
    print(f"Test solved:     {total_test}/{n} ({total_test/n*100:.0f}%)")
    print(f"Avg candidates:  {np.mean(all_cands):.0f}")
    print(f"Time:            {total_time:.1f}s")
    print(f"{'='*70}")

    print(f"\n{'Type':30s} {'Train':>8s} {'Test':>8s} {'Cands':>8s} {'Steps':>6s}")
    for rtype in sorted(per_type.keys()):
        g = per_type[rtype]
        t = g["total"]
        avg_c = np.mean(g["cands"]) if g["cands"] else 0
        is_multi = "2" if rtype in ('fill_then_recolor', 'delete_then_symmetrize', 'periodic_then_fill') else "1"
        print(f"  {rtype:28s} {g['train']:>5d}/{t:<2d} {g['test']:>5d}/{t:<2d} {avg_c:>7.0f} {is_multi:>5s}")


if __name__ == "__main__":
    main()
