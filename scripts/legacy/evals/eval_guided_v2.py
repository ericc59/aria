#!/usr/bin/env python3
"""Evaluate unguided vs guided multi-step search."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.guided.synthetic import generate_benchmark, split_benchmark
from aria.guided.training_data import extract_step_examples
from aria.guided.selector_model import StepSelector, train_selector
from aria.guided.search import search as unguided_search, predict_test
from aria.guided.guided_search import guided_search


def main():
    print("Generating benchmark...")
    all_tasks = generate_benchmark(n_tasks=500, seed=42)
    train_tasks, test_tasks = split_benchmark(all_tasks, 0.8)
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Train selector
    print("Extracting step examples...")
    step_examples = extract_step_examples(train_tasks)
    print(f"  {len(step_examples)} step examples")

    print("Training selector...")
    model = StepSelector(input_dim=30, seed=42)
    history = train_selector(model, step_examples, epochs=150, lr=0.003)
    print(f"  Final: loss={history['loss'][-1]:.3f} target_acc={history['target_acc'][-1]:.3f} "
          f"rewrite_acc={history['rewrite_acc'][-1]:.3f}")

    # Evaluate
    print(f"\nEvaluating on {len(test_tasks)} held-out tasks...")

    per_type = defaultdict(lambda: {
        "ug_train": 0, "ug_test": 0, "ug_cands": [],
        "g_train": 0, "g_test": 0, "g_cands": [],
        "total": 0,
    })

    for task in test_tasks:
        g = per_type[task.rule_type]
        g["total"] += 1

        # Unguided
        ug = unguided_search(task.train, max_candidates=3000, max_steps=3)
        g["ug_cands"].append(ug.candidates_tried)
        if ug.solved:
            g["ug_train"] += 1
            if _test_ok(ug.program, task.test):
                g["ug_test"] += 1

        # Guided (with fallback to unguided)
        gg = guided_search(task.train, model, max_candidates=2000, max_steps=3, beam_width=50)
        total_g_cands = gg.candidates_tried
        if gg.solved:
            g["g_train"] += 1
            if _test_ok(gg.program, task.test):
                g["g_test"] += 1
        elif ug.solved:
            # Fallback: use unguided result
            g["g_train"] += 1
            total_g_cands += ug.candidates_tried
            if _test_ok(ug.program, task.test):
                g["g_test"] += 1
        g["g_cands"].append(total_g_cands)

    n = len(test_tasks)
    total_ug_t = sum(g["ug_train"] for g in per_type.values())
    total_ug_s = sum(g["ug_test"] for g in per_type.values())
    total_g_t = sum(g["g_train"] for g in per_type.values())
    total_g_s = sum(g["g_test"] for g in per_type.values())
    avg_ug_c = np.mean([c for g in per_type.values() for c in g["ug_cands"]])
    avg_g_c = np.mean([c for g in per_type.values() for c in g["g_cands"]])

    print(f"\n{'='*80}")
    print(f"UNGUIDED vs GUIDED MULTI-STEP SEARCH — {n} tasks")
    print(f"{'='*80}")
    print(f"{'':30s} {'Unguided':>20s} {'Guided':>20s}")
    print(f"  {'Train verified':28s} {total_ug_t:>10d}/{n} {total_g_t:>10d}/{n}")
    print(f"  {'Test solved':28s} {total_ug_s:>10d}/{n} {total_g_s:>10d}/{n}")
    print(f"  {'Avg candidates':28s} {avg_ug_c:>15.0f} {avg_g_c:>15.0f}")
    cost_red = (1 - avg_g_c / max(1, avg_ug_c)) * 100
    print(f"  {'Cost reduction':28s} {'':>15s} {cost_red:>14.0f}%")

    print(f"\nPer-type:")
    print(f"{'Type':30s} {'UG Train':>9s} {'G Train':>9s} {'UG Cands':>9s} {'G Cands':>9s} {'Steps':>6s}")
    for rtype in sorted(per_type.keys()):
        g = per_type[rtype]
        t = g["total"]
        uc = np.mean(g["ug_cands"])
        gc = np.mean(g["g_cands"])
        is_multi = "2" if rtype in ('fill_then_recolor', 'delete_then_symmetrize') else "1"
        print(f"  {rtype:28s} {g['ug_train']:>5d}/{t:<3d} {g['g_train']:>5d}/{t:<3d} {uc:>8.0f} {gc:>8.0f} {is_multi:>5s}")


def _test_ok(prog, test_pairs):
    if prog is None:
        return False
    for inp, out in test_pairs:
        pred = predict_test(prog, inp)
        if pred is None or not np.array_equal(pred, out):
            return False
    return True


if __name__ == "__main__":
    main()
