#!/usr/bin/env python3
"""Evaluate the learned target/support selector.

Reports:
- Aggregate: support accuracy, op accuracy, recall@K
- Per-generator: support alignment accuracy on copy_stamp,
  target selection accuracy on reflect_repair, etc.
- Downstream: train-verified rate with guided vs unguided search
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.guided.synthetic import generate_benchmark, split_benchmark
from aria.guided.training_data import (
    extract_training_data, featurize_example, SUPPORT_TYPES, REWRITE_OPS,
    SUPPORT_TO_IDX, OP_TO_IDX,
)
from aria.guided.selector_model import SelectorMLP, train_selector
from aria.guided.search import unguided_search
from aria.guided.workspace import _detect_bg


def main():
    print("Generating benchmark...")
    all_tasks = generate_benchmark(n_tasks=200, seed=42)
    train_tasks, test_tasks = split_benchmark(all_tasks, 0.7)

    # Extract training data
    print("Extracting training data...")
    train_examples = extract_training_data(train_tasks)
    test_examples = extract_training_data(test_tasks)

    # Tag each example with its task type for per-generator metrics
    task_type_map = {t.task_id: t.rule_type for t in all_tasks}
    for ex in train_examples:
        ex.features["_task_type"] = task_type_map.get(ex.task_id, "")
    for ex in test_examples:
        ex.features["_task_type"] = task_type_map.get(ex.task_id, "")

    # Featurize
    print("Featurizing...")
    train_feats = [featurize_example(ex) for ex in train_examples]
    test_feats = [featurize_example(ex) for ex in test_examples]

    # Train the model
    print(f"Training selector on {len(train_feats)} examples...")
    model = SelectorMLP(seed=42)
    history = train_selector(model, train_feats, epochs=80, lr=0.005)

    print(f"  Final train loss: {history['loss'][-1]:.3f}")
    print(f"  Final train support_acc: {history['support_acc'][-1]:.3f}")
    print(f"  Final train op_acc: {history['op_acc'][-1]:.3f}")

    # Evaluate on held-out test
    print(f"\nEvaluating on {len(test_feats)} held-out examples...")

    # --- Aggregate metrics ---
    support_correct = 0
    op_correct = 0
    support_recall_at_3 = 0
    op_recall_at_3 = 0

    # --- Per-generator metrics ---
    per_gen = defaultdict(lambda: {
        "support_correct": 0, "op_correct": 0,
        "support_recall3": 0, "op_recall3": 0,
        "total": 0,
        # copy_stamp specific: did we identify the stamp source object?
        "support_obj_correct": 0, "support_obj_total": 0,
    })

    for ex, feat in zip(test_examples, test_feats):
        pred_s, pred_o = model.predict(feat["x_global"], feat["x_objects"])
        topk_s, topk_o = model.predict_topk(feat["x_global"], feat["x_objects"], k=3)

        y_s = feat["y_support"]
        y_o = feat["y_op"]
        task_type = ex.features.get("_task_type", "unknown")

        s_ok = pred_s == y_s
        o_ok = pred_o == y_o
        s_r3 = y_s in topk_s
        o_r3 = y_o in topk_o

        support_correct += s_ok
        op_correct += o_ok
        support_recall_at_3 += s_r3
        op_recall_at_3 += o_r3

        g = per_gen[task_type]
        g["total"] += 1
        g["support_correct"] += s_ok
        g["op_correct"] += o_ok
        g["support_recall3"] += s_r3
        g["op_recall3"] += o_r3

    n = len(test_feats)
    print(f"\n{'='*70}")
    print(f"SELECTOR METRICS — {n} held-out examples")
    print(f"{'='*70}")
    print(f"Support type accuracy:     {support_correct}/{n} ({support_correct/n*100:.1f}%)")
    print(f"Support type recall@3:     {support_recall_at_3}/{n} ({support_recall_at_3/n*100:.1f}%)")
    print(f"Rewrite op accuracy:       {op_correct}/{n} ({op_correct/n*100:.1f}%)")
    print(f"Rewrite op recall@3:       {op_recall_at_3}/{n} ({op_recall_at_3/n*100:.1f}%)")
    print(f"{'='*70}")

    print(f"\nPer-generator breakdown:")
    print(f"{'Generator':25s} {'Support':>8s} {'Op':>8s} {'Sup@3':>8s} {'Op@3':>8s} {'N':>5s}")
    for gen_type in sorted(per_gen.keys()):
        g = per_gen[gen_type]
        t = g["total"]
        s_acc = g["support_correct"] / t if t > 0 else 0
        o_acc = g["op_correct"] / t if t > 0 else 0
        s_r3 = g["support_recall3"] / t if t > 0 else 0
        o_r3 = g["op_recall3"] / t if t > 0 else 0
        print(f"  {gen_type:23s} {s_acc:>7.1%} {o_acc:>7.1%} {s_r3:>7.1%} {o_r3:>7.1%} {t:>5d}")

    # --- Downstream: guided search vs unguided ---
    print(f"\n{'='*70}")
    print(f"DOWNSTREAM: guided selection → search")
    print(f"{'='*70}")

    guided_train_solved = 0
    guided_test_solved = 0
    unguided_train_solved = 0
    unguided_test_solved = 0

    for task in test_tasks:
        # Unguided baseline
        ug_result = unguided_search(task.train, max_candidates=500)
        if ug_result.solved:
            unguided_train_solved += 1
            ok = _verify_test(ug_result.op_fn, task.test)
            if ok:
                unguided_test_solved += 1

        # Guided: use selector to pick support type + op, then search
        # only matching candidates
        ws = build_workspace_quick(task.train[0][0], task.train[0][1])
        feat = featurize_workspace(ws)
        topk_s, topk_o = model.predict_topk(feat["x_global"], feat["x_objects"], k=3)

        # Try guided first (selector-filtered), fall back to unguided
        g_result = guided_search(task.train, topk_s, topk_o, max_candidates=200)
        if g_result.solved:
            guided_train_solved += 1
            ok = _verify_test(g_result.op_fn, task.test)
            if ok:
                guided_test_solved += 1
        elif ug_result.solved:
            # Fallback: use unguided result
            guided_train_solved += 1
            ok = _verify_test(ug_result.op_fn, task.test)
            if ok:
                guided_test_solved += 1

    n_tasks = len(test_tasks)
    print(f"{'Method':20s} {'Train Verified':>15s} {'Test Solved':>12s}")
    print(f"  {'Unguided':20s} {unguided_train_solved:>10d}/{n_tasks:<4d} {unguided_test_solved:>8d}/{n_tasks:<3d}")
    print(f"  {'Guided (selector)':20s} {guided_train_solved:>10d}/{n_tasks:<4d} {guided_test_solved:>8d}/{n_tasks:<3d}")

    # Compare
    improvement = guided_train_solved - unguided_train_solved
    print(f"\n  Improvement: {improvement:+d} train-verified ({improvement/max(1,n_tasks)*100:+.1f}%)")

    # Random baseline for support selection
    rng = np.random.RandomState(99)
    random_correct = sum(1 for feat in test_feats
                        if rng.randint(0, len(SUPPORT_TYPES)) == feat["y_support"])
    print(f"\n  Random support accuracy: {random_correct}/{n} ({random_correct/n*100:.1f}%)")
    print(f"  Learned support accuracy: {support_correct}/{n} ({support_correct/n*100:.1f}%)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_workspace_quick(inp, out):
    from aria.guided.workspace import build_workspace
    return build_workspace(inp, out)


def featurize_workspace(ws):
    from aria.guided.training_data import featurize_example, SelectionExample
    ex = SelectionExample(
        task_id="", demo_idx=0, features=ws.serialize(),
        support_type="", target_desc="", rewrite_op="", key_params={},
    )
    return featurize_example(ex)


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


def guided_search(train, topk_support, topk_ops, max_candidates=200):
    """Search restricted to selector's top-K support types and ops."""
    from aria.guided.search import (
        SearchResult, _op_fill_enclosed, _op_fill_enclosed_majority, _detect_bg,
    )
    from aria.guided.grammar import _repair_periodic_region
    from aria.guided.training_data import SUPPORT_TYPES, REWRITE_OPS
    from collections import deque

    bg = _detect_bg(train[0][0])
    candidates_tried = 0
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)

    # Map support/op indices to operations
    support_names = [SUPPORT_TYPES[i] for i in topk_support]
    op_names = [REWRITE_OPS[i] for i in topk_ops]

    ops = []

    for sn in support_names:
        for on in op_names:
            if sn == "enclosed_bg" and on == "FILL":
                for color in range(10):
                    if color == bg:
                        continue
                    def _f(inp, dbg, c=color):
                        return _op_fill_enclosed(inp, dbg, c)
                    ops.append(_f)

            if sn == "all_objects" and on == "RECOLOR":
                for fc in range(10):
                    for tc in range(10):
                        if fc == tc:
                            continue
                        def _r(inp, dbg, f=fc, t=tc):
                            out = inp.copy()
                            out[inp == f] = t
                            return out
                        ops.append(_r)

            if sn == "full_grid" and on == "PERIODIC_REPAIR":
                for axis in ["row", "col"]:
                    def _p(inp, dbg, ax=axis):
                        return _repair_periodic_region(inp, ax)
                    ops.append(_p)

            if sn == "all_objects" and on == "DELETE":
                for color in range(10):
                    def _d(inp, dbg, c=color):
                        out = inp.copy()
                        out[inp == c] = dbg
                        return out
                    ops.append(_d)

            if sn == "enclosed_bg" and on == "FILL":
                def _fm(inp, dbg):
                    return _op_fill_enclosed_majority(inp, dbg)
                ops.append(_fm)

    for op_fn in ops:
        candidates_tried += 1
        all_match = True
        total_diff = 0
        for inp, out in train:
            demo_bg = _detect_bg(inp)
            try:
                pred = op_fn(inp, demo_bg)
            except Exception:
                all_match = False
                total_diff += out.size
                continue
            if pred is None or not np.array_equal(pred, out):
                all_match = False
                if pred is not None and pred.shape == out.shape:
                    total_diff += int(np.sum(pred != out))
                else:
                    total_diff += out.size

        if all_match:
            return SearchResult(True, op_fn, candidates_tried, 0)
        if total_diff < best_diff:
            best_diff = total_diff
        if candidates_tried >= max_candidates:
            break

    return SearchResult(False, None, candidates_tried, best_diff)


if __name__ == "__main__":
    main()
