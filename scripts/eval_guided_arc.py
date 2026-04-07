#!/usr/bin/env python3
"""Evaluate the guided engine on an ARC-2 slice."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task
from aria.guided.workspace import build_workspace, _detect_bg
from aria.guided.synthetic import generate_benchmark, split_benchmark
from aria.guided.training_data import extract_training_data, featurize_example
from aria.guided.selector_model import SelectorMLP, train_selector
from aria.guided.search import unguided_search
from aria.guided.expansion import guided_expansion_search


# Top 20 same-shape ARC-2 tasks by preservation ratio
ARC_SLICE = [
    '8e5c0c38', '88e364bc', '135a2760', '9bbf930d', '409aa875',
    '7b80bb43', 'b99e7126', 'e376de54', '97d7923e', '3e6067c3',
    '332f06d7', '4c416de3', '8b7bacbf', 'dbff022c', 'c4d067a0',
    '53fb4810', '7ed72f31', 'c7f57c3e', 'b10624e5', 'a25697e4',
]


def main():
    ds = get_dataset("v2-eval")

    # Train selector on synthetic data
    print("Training selector on synthetic benchmark...")
    syn_tasks = generate_benchmark(n_tasks=200, seed=42)
    train_syn, _ = split_benchmark(syn_tasks, 0.7)
    task_type_map = {t.task_id: t.rule_type for t in syn_tasks}
    train_ex = extract_training_data(train_syn)
    for ex in train_ex:
        ex.features["_task_type"] = task_type_map.get(ex.task_id, "")
    train_feats = [featurize_example(ex) for ex in train_ex]
    model = SelectorMLP(seed=42)
    train_selector(model, train_feats, epochs=80, lr=0.005)

    # Evaluate on ARC slice
    print(f"\nEvaluating on {len(ARC_SLICE)} ARC-2 tasks...")
    print(f"{'='*80}")

    ug_train = 0
    ug_test = 0
    g_train = 0
    g_test = 0
    ug_cands_list = []
    g_cands_list = []

    for tid in ARC_SLICE:
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            print(f"{tid}: LOAD_ERROR {e}")
            continue

        train_pairs = [(d.input, d.output) for d in task.train]

        # Unguided
        ug = unguided_search(train_pairs, max_candidates=500)
        ug_cands_list.append(ug.candidates_tried)
        ug_solved_train = ug.solved
        ug_solved_test = False
        if ug_solved_train:
            ug_train += 1
            ug_solved_test = _verify_test(ug.op_fn, task.test)
            if ug_solved_test:
                ug_test += 1

        # Guided
        solved, op_fn, cands, diff = guided_expansion_search(
            train_pairs, model, max_candidates=500)
        g_cands_list.append(cands)
        g_solved_train = solved
        g_solved_test = False
        if g_solved_train:
            g_train += 1
            g_solved_test = _verify_test(op_fn, task.test)
            if g_solved_test:
                g_test += 1

        # Report
        pres = np.mean([np.sum(d.input == d.output) / d.input.size for d in task.train])
        ug_mark = "TRAIN" if ug_solved_train else f"diff={ug.train_diff}"
        g_mark = "TRAIN" if g_solved_train else f"diff={diff}"
        if ug_solved_test:
            ug_mark += "+TEST"
        if g_solved_test:
            g_mark += "+TEST"

        improved = g_solved_train and not ug_solved_train
        marker = " <<<" if improved else (" !!!" if g_solved_test else "")

        print(f"  {tid}: pres={pres:.0%} ug={ug_mark:>15} g={g_mark:>15} "
              f"ug_c={ug.candidates_tried:>3} g_c={cands:>3}{marker}")

    n = len(ARC_SLICE)
    print(f"\n{'='*80}")
    print(f"ARC-2 SLICE RESULTS — {n} tasks")
    print(f"{'='*80}")
    print(f"{'Method':20s} {'Train Verified':>15s} {'Test Solved':>12s} {'Avg Cands':>10s}")
    print(f"  {'Unguided':20s} {ug_train:>10d}/{n:<4d} {ug_test:>8d}/{n:<3d} {np.mean(ug_cands_list):>9.0f}")
    print(f"  {'Guided':20s} {g_train:>10d}/{n:<4d} {g_test:>8d}/{n:<3d} {np.mean(g_cands_list):>9.0f}")
    print(f"{'='*80}")

    if g_train > ug_train:
        print(f"\n  Guided improves train-verified by +{g_train - ug_train}")
    elif g_train == ug_train:
        print(f"\n  No improvement in train-verified")
    else:
        print(f"\n  Guided is WORSE by {ug_train - g_train}")

    if g_test > 0:
        print(f"  EXACT SOLVES: {g_test}")


def _verify_test(op_fn, test_pairs):
    if op_fn is None:
        return False
    for tp in test_pairs:
        bg = _detect_bg(tp.input)
        try:
            pred = op_fn(tp.input, bg)
        except Exception:
            return False
        if pred is None or not np.array_equal(pred, tp.output):
            return False
    return True


if __name__ == "__main__":
    main()
