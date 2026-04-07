#!/usr/bin/env python3
"""Evaluate guided expansion search vs unguided baseline."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.guided.synthetic import generate_benchmark, split_benchmark
from aria.guided.training_data import extract_training_data, featurize_example
from aria.guided.selector_model import SelectorMLP, train_selector
from aria.guided.search import unguided_search
from aria.guided.expansion import guided_expansion_search
from aria.guided.workspace import _detect_bg


def main():
    print("Generating benchmark...")
    all_tasks = generate_benchmark(n_tasks=200, seed=42)
    train_tasks, test_tasks = split_benchmark(all_tasks, 0.7)

    # Train selector
    print("Training selector...")
    task_type_map = {t.task_id: t.rule_type for t in all_tasks}
    train_ex = extract_training_data(train_tasks)
    for ex in train_ex:
        ex.features["_task_type"] = task_type_map.get(ex.task_id, "")
    train_feats = [featurize_example(ex) for ex in train_ex]

    model = SelectorMLP(seed=42)
    train_selector(model, train_feats, epochs=80, lr=0.005)

    # Evaluate
    print(f"\nEvaluating on {len(test_tasks)} held-out tasks...")

    per_type = defaultdict(lambda: {
        "ug_train": 0, "ug_test": 0, "ug_cands": [],
        "g_train": 0, "g_test": 0, "g_cands": [],
        "total": 0,
    })

    for task in test_tasks:
        rtype = task.rule_type
        g = per_type[rtype]
        g["total"] += 1

        # Unguided
        ug = unguided_search(task.train, max_candidates=500)
        g["ug_cands"].append(ug.candidates_tried)
        if ug.solved:
            g["ug_train"] += 1
            if _verify_test(ug.op_fn, task.test):
                g["ug_test"] += 1

        # Guided
        solved, op_fn, cands, diff = guided_expansion_search(task.train, model, max_candidates=300)
        g["g_cands"].append(cands)
        if solved:
            g["g_train"] += 1
            if _verify_test(op_fn, task.test):
                g["g_test"] += 1

    # Report
    print(f"\n{'='*80}")
    print(f"UNGUIDED vs GUIDED EXPANSION — {len(test_tasks)} held-out tasks")
    print(f"{'='*80}")
    print(f"{'Type':25s} | {'Unguided':^25s} | {'Guided':^25s} | {'Cost Reduction':>14s}")
    print(f"{'':25s} | {'Train':>7s} {'Test':>7s} {'Cands':>7s} | {'Train':>7s} {'Test':>7s} {'Cands':>7s} |")
    print(f"{'-'*25}-+-{'-'*25}-+-{'-'*25}-+-{'-'*14}")

    total_ug_train = total_ug_test = total_g_train = total_g_test = 0
    total_ug_cands = total_g_cands = 0
    n_total = 0

    for rtype in sorted(per_type.keys()):
        g = per_type[rtype]
        t = g["total"]
        ug_c = np.mean(g["ug_cands"]) if g["ug_cands"] else 0
        g_c = np.mean(g["g_cands"]) if g["g_cands"] else 0
        reduction = f"{(1-g_c/max(1,ug_c))*100:.0f}%" if ug_c > 0 else "N/A"

        print(f"  {rtype:23s} | {g['ug_train']:>5d}/{t:<2d} {g['ug_test']:>5d}/{t:<2d} {ug_c:>6.0f} "
              f"| {g['g_train']:>5d}/{t:<2d} {g['g_test']:>5d}/{t:<2d} {g_c:>6.0f} | {reduction:>13s}")

        total_ug_train += g["ug_train"]
        total_ug_test += g["ug_test"]
        total_g_train += g["g_train"]
        total_g_test += g["g_test"]
        total_ug_cands += sum(g["ug_cands"])
        total_g_cands += sum(g["g_cands"])
        n_total += t

    avg_ug_c = total_ug_cands / max(1, n_total)
    avg_g_c = total_g_cands / max(1, n_total)
    total_reduction = f"{(1-avg_g_c/max(1,avg_ug_c))*100:.0f}%"

    print(f"{'-'*25}-+-{'-'*25}-+-{'-'*25}-+-{'-'*14}")
    print(f"  {'TOTAL':23s} | {total_ug_train:>5d}/{n_total:<2d} {total_ug_test:>5d}/{n_total:<2d} {avg_ug_c:>6.0f} "
          f"| {total_g_train:>5d}/{n_total:<2d} {total_g_test:>5d}/{n_total:<2d} {avg_g_c:>6.0f} | {total_reduction:>13s}")

    print(f"\n  Train-verified improvement: {total_g_train - total_ug_train:+d} "
          f"({total_ug_train}/{n_total} → {total_g_train}/{n_total})")
    print(f"  Test-solved improvement:    {total_g_test - total_ug_test:+d} "
          f"({total_ug_test}/{n_total} → {total_g_test}/{n_total})")
    print(f"  Search cost reduction:      {total_reduction}")


def _verify_test(op_fn, test_pairs):
    if op_fn is None:
        return False
    for inp, out in test_pairs:
        bg = _detect_bg(inp)
        try:
            pred = op_fn(inp, bg)
        except Exception:
            return False
        if pred is None or not np.array_equal(pred, out):
            return False
    return True


if __name__ == "__main__":
    main()
