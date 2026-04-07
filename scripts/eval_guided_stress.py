#!/usr/bin/env python3
"""Stress-test the guided engine on harder synthetic variants + ablations."""

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


# ===================================================================
# Harder synthetic generators
# ===================================================================

def generate_harder_benchmark(n_tasks=100, seed=99):
    """Generate harder synthetic tasks: larger grids, more distractors."""
    from aria.guided.synthetic import (
        _gen_fill_residual, _gen_recolor_by_property,
        _gen_periodic_anomaly, _gen_reflect_repair, _gen_copy_stamp,
    )
    rng = np.random.RandomState(seed)
    tasks = []
    generators = [
        _gen_fill_residual,
        _gen_recolor_by_property,
        _gen_periodic_anomaly,
        _gen_reflect_repair,
        _gen_copy_stamp,
    ]

    for i in range(n_tasks):
        gen_fn = generators[i % len(generators)]
        task = gen_fn(rng, 1000 + i, 3, 1)

        # Make harder: add distractors to input
        for j in range(len(task.train)):
            inp, out = task.train[j]
            inp_h = _add_distractors(inp, rng, n_distractors=2)
            # Distractors should not change in output (preserved)
            out_h = out.copy()
            for r in range(inp_h.shape[0]):
                for c in range(inp_h.shape[1]):
                    if r < out.shape[0] and c < out.shape[1]:
                        if inp_h[r, c] != inp[r, c] and inp[r, c] == out[r, c]:
                            out_h[r, c] = inp_h[r, c]
            task.train[j] = (inp_h, out_h)

        for j in range(len(task.test)):
            inp, out = task.test[j]
            inp_h = _add_distractors(inp, rng, n_distractors=2)
            out_h = out.copy()
            for r in range(inp_h.shape[0]):
                for c in range(inp_h.shape[1]):
                    if r < out.shape[0] and c < out.shape[1]:
                        if inp_h[r, c] != inp[r, c] and inp[r, c] == out[r, c]:
                            out_h[r, c] = inp_h[r, c]
            task.test[j] = (inp_h, out_h)

        tasks.append(task)
    return tasks


def _add_distractors(grid, rng, n_distractors=2):
    """Add small random objects as distractors."""
    out = grid.copy()
    rows, cols = grid.shape
    bg_val = int(np.bincount(grid.flatten()).argmax())

    for _ in range(n_distractors):
        color = rng.randint(1, 10)
        r = rng.randint(0, max(1, rows - 2))
        c = rng.randint(0, max(1, cols - 2))
        h = rng.randint(1, 2)
        w = rng.randint(1, 2)
        # Only place on bg cells
        for dr in range(h):
            for dc in range(w):
                tr, tc = r + dr, c + dc
                if 0 <= tr < rows and 0 <= tc < cols and out[tr, tc] == bg_val:
                    out[tr, tc] = color
    return out


# ===================================================================
# Evaluation helpers
# ===================================================================

def evaluate_method(method_name, solve_fn, tasks):
    """Evaluate a solve function on a set of tasks."""
    results = defaultdict(lambda: {"train": 0, "test": 0, "cands": [], "total": 0})

    for task in tasks:
        rtype = task.rule_type
        g = results[rtype]
        g["total"] += 1

        solved, op_fn, cands, diff = solve_fn(task)
        g["cands"].append(cands)

        if solved:
            g["train"] += 1
            if _verify_test(op_fn, task.test):
                g["test"] += 1

    return results


def _solve_unguided(task):
    result = unguided_search(task.train, max_candidates=500)
    return result.solved, result.op_fn, result.candidates_tried, result.train_diff


def _solve_guided(model):
    def _fn(task):
        return guided_expansion_search(task.train, model, max_candidates=300)
    return _fn


def _solve_selector_only(model):
    """Use selector to pick support+op, but no structural candidates."""
    from aria.guided.expansion import _quick_featurize, _guided_candidates, _verify_on_train
    from aria.guided.workspace import build_workspace
    from aria.guided.training_data import SUPPORT_TYPES, REWRITE_OPS

    def _fn(task):
        ws = build_workspace(task.train[0][0], task.train[0][1])
        feat = _quick_featurize(ws)
        topk_s, topk_o = model.predict_topk(feat["x_global"], feat["x_objects"], k=3)
        cands = 0
        best_diff = sum(int(np.sum(inp != out)) for inp, out in task.train)
        for op_fn in _guided_candidates(ws, topk_s, topk_o):
            cands += 1
            ok, diff = _verify_on_train(op_fn, task.train)
            if ok:
                return True, op_fn, cands, 0
            if diff < best_diff:
                best_diff = diff
            if cands >= 300:
                break
        return False, None, cands, best_diff
    return _fn


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


def print_results(name, results, n_total):
    total_train = sum(g["train"] for g in results.values())
    total_test = sum(g["test"] for g in results.values())
    all_cands = [c for g in results.values() for c in g["cands"]]
    avg_c = np.mean(all_cands) if all_cands else 0
    print(f"  {name:25s}: train={total_train:>3d}/{n_total} test={total_test:>3d}/{n_total} avg_cands={avg_c:>5.0f}")
    for rtype in sorted(results.keys()):
        g = results[rtype]
        t = g["total"]
        rc = np.mean(g["cands"]) if g["cands"] else 0
        print(f"    {rtype:23s}: train={g['train']:>2d}/{t} test={g['test']:>2d}/{t} cands={rc:.0f}")


# ===================================================================
# Main
# ===================================================================

def main():
    # --- Train model on standard benchmark ---
    print("Training on standard benchmark...")
    std_tasks = generate_benchmark(n_tasks=200, seed=42)
    train_tasks, _ = split_benchmark(std_tasks, 0.7)

    task_type_map = {t.task_id: t.rule_type for t in std_tasks}
    train_ex = extract_training_data(train_tasks)
    for ex in train_ex:
        ex.features["_task_type"] = task_type_map.get(ex.task_id, "")
    train_feats = [featurize_example(ex) for ex in train_ex]

    model = SelectorMLP(seed=42)
    train_selector(model, train_feats, epochs=80, lr=0.005)

    # --- Part A: Standard held-out ---
    print("\n=== STANDARD HELD-OUT (seed=42, 60 tasks) ===")
    _, std_test = split_benchmark(std_tasks, 0.7)
    n = len(std_test)

    r_ug = evaluate_method("unguided", _solve_unguided, std_test)
    r_g = evaluate_method("guided", _solve_guided(model), std_test)
    print_results("Unguided", r_ug, n)
    print_results("Guided (full)", r_g, n)

    # --- Part A: New seed held-out ---
    print("\n=== NEW SEED HELD-OUT (seed=99, 60 tasks) ===")
    new_tasks = generate_benchmark(n_tasks=100, seed=99)
    _, new_test = split_benchmark(new_tasks, 0.5)
    n2 = len(new_test)

    r_ug2 = evaluate_method("unguided", _solve_unguided, new_test)
    r_g2 = evaluate_method("guided", _solve_guided(model), new_test)
    print_results("Unguided", r_ug2, n2)
    print_results("Guided (full)", r_g2, n2)

    # --- Part A: Harder (distractors) ---
    print("\n=== HARDER (distractors, seed=99, 100 tasks) ===")
    hard_tasks = generate_harder_benchmark(n_tasks=100, seed=99)
    n3 = len(hard_tasks)

    r_ug3 = evaluate_method("unguided", _solve_unguided, hard_tasks)
    r_g3 = evaluate_method("guided", _solve_guided(model), hard_tasks)
    print_results("Unguided", r_ug3, n3)
    print_results("Guided (full)", r_g3, n3)

    # --- Part B: Ablations on standard held-out ---
    print("\n=== ABLATIONS (standard held-out) ===")
    r_sel = evaluate_method("selector_only", _solve_selector_only(model), std_test)
    print_results("Unguided", r_ug, n)
    print_results("Selector only", r_sel, n)
    print_results("Guided (full)", r_g, n)

    # --- Part C: Summary ---
    total_g = sum(g["train"] for g in r_g.values())
    total_ug = sum(g["train"] for g in r_ug.values())
    total_g2 = sum(g["train"] for g in r_g2.values())
    total_ug2 = sum(g["train"] for g in r_ug2.values())
    total_g3 = sum(g["train"] for g in r_g3.values())
    total_ug3 = sum(g["train"] for g in r_ug3.values())

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Setting':30s} {'Unguided':>10s} {'Guided':>10s} {'Delta':>8s}")
    print(f"  {'Standard held-out':28s} {total_ug:>7d}/{n} {total_g:>7d}/{n} {total_g-total_ug:>+6d}")
    print(f"  {'New seed':28s} {total_ug2:>7d}/{n2} {total_g2:>7d}/{n2} {total_g2-total_ug2:>+6d}")
    print(f"  {'Harder (distractors)':28s} {total_ug3:>7d}/{n3} {total_g3:>7d}/{n3} {total_g3-total_ug3:>+6d}")

    all_deltas = [total_g - total_ug, total_g2 - total_ug2, total_g3 - total_ug3]
    robust = all(d >= 0 for d in all_deltas) and sum(d > 0 for d in all_deltas) >= 2
    print(f"\n  Robust improvement across settings: {'YES' if robust else 'NO'}")
    print(f"  Ready for ARC transfer: {'YES' if robust else 'NO'}")


if __name__ == "__main__":
    main()
