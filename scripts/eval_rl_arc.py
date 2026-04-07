#!/usr/bin/env python3
"""Train RL policy on ARC-2, then evaluate."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task, list_task_ids
from aria.guided.workspace import _detect_bg
from aria.guided.rl_search import PolicyNet, train_rl, rl_search
from aria.guided.search import predict_test


def main():
    ds = get_dataset("v2-eval")
    tids = list_task_ids(ds)[:120]

    # Load all same-shape ARC tasks
    arc_tasks = []
    for tid in tids:
        try:
            task = load_arc_task(ds, tid)
        except:
            continue
        if not all(d.input.shape == d.output.shape for d in task.train):
            continue
        train_pairs = [(d.input, d.output) for d in task.train]
        arc_tasks.append((tid, train_pairs, task.test))

    print(f"Loaded {len(arc_tasks)} same-shape ARC-2 tasks")

    # Train RL policy
    print("\nTraining RL policy on ARC-2 train pairs...")
    policy = PolicyNet(input_dim=30, seed=42)
    train_data = [(tid, pairs) for tid, pairs, _ in arc_tasks]

    t0 = time.time()
    history = train_rl(
        policy, train_data,
        n_epochs=100,
        episodes_per_task=3,
        max_candidates=100,
        lr=0.002,
        epsilon_start=0.8,
        epsilon_end=0.1,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate with trained policy
    print(f"\nEvaluating RL-guided search...")
    n_train = 0
    n_test = 0

    for tid, train_pairs, test_pairs in arc_tasks:
        result = rl_search(train_pairs, policy, max_candidates=500, max_steps=3, top_k=15)
        if result.solved:
            n_train += 1
            tok = all(
                np.array_equal(predict_test(result.program, tp.input), tp.output)
                for tp in test_pairs
            )
            if tok:
                n_test += 1
            print(f"  {tid}: TRAIN_VERIFIED{' TEST_PASS' if tok else ''} "
                  f"program={result.program} cands={result.candidates_tried}")

    n = len(arc_tasks)
    print(f"\n{'='*70}")
    print(f"RL-GUIDED SEARCH — {n} ARC-2 tasks")
    print(f"{'='*70}")
    print(f"Train verified: {n_train}/{n}")
    print(f"Test solved:    {n_test}/{n}")
    print(f"{'='*70}")

    # Compare: unguided baseline
    print(f"\nUnguided baseline comparison...")
    from aria.guided.search import search
    ug_train = 0
    ug_test = 0
    for tid, train_pairs, test_pairs in arc_tasks:
        result = search(train_pairs, max_candidates=500, max_steps=3)
        if result.solved:
            ug_train += 1
            tok = all(
                np.array_equal(predict_test(result.program, tp.input), tp.output)
                for tp in test_pairs
            )
            if tok:
                ug_test += 1
    print(f"  Unguided: {ug_train}/{n} train, {ug_test}/{n} test")
    print(f"  RL-guided: {n_train}/{n} train, {n_test}/{n} test")


if __name__ == "__main__":
    main()
