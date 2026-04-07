#!/usr/bin/env python3
"""Evaluate NGS selection induction on the 12-task preservation slice."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task
from aria.decomposition import detect_bg
from aria.ngs.preservation import factor_preservation
from aria.ngs.residual_explain import explain_residual
from aria.ngs.selection import (
    induce_selection, apply_selection_and_rewrite,
)

SLICE = [
    '8e5c0c38', '88e364bc', '135a2760', '9bbf930d', '409aa875',
    '97d7923e', '332f06d7', 'dbff022c', '4c416de3', '8b7bacbf',
    'c4d067a0', 'e376de54',
]

# Rewrite types to try for each selected mask
REWRITE_TYPES = ["delete", "recolor_to_adjacent", "fill_with_adjacent"]


def main():
    ds = get_dataset("v2-eval")
    results = []
    t0 = time.time()

    for tid in SLICE:
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            print(f"{tid}: LOAD_ERROR {e}")
            continue

        demos = task.train
        t1 = time.time()

        # Factor preservation
        factors = []
        for d in demos:
            bg = detect_bg(d.input)
            pf = factor_preservation(d.input, d.output, bg)
            factors.append(pf)

        # Induce selection
        sel_result = induce_selection(factors, demos)

        # If selection found, try combining with each rewrite
        train_verified = False
        test_solved = False
        best_rewrite = None
        best_diff = float('inf')

        if sel_result.rule is not None:
            for rtype in REWRITE_TYPES:
                # Verify on train
                all_match = True
                total_diff = 0
                for d in demos:
                    bg = detect_bg(d.input)
                    predicted = apply_selection_and_rewrite(
                        sel_result.rule, rtype, d.input, bg)
                    if predicted is None or not np.array_equal(predicted, d.output):
                        all_match = False
                        if predicted is not None and predicted.shape == d.output.shape:
                            total_diff += int(np.sum(predicted != d.output))
                        else:
                            total_diff += d.output.size

                if all_match:
                    train_verified = True
                    best_rewrite = rtype
                    best_diff = 0

                    # Try test
                    all_test_ok = True
                    for tp in task.test:
                        bg = detect_bg(tp.input)
                        pred = apply_selection_and_rewrite(
                            sel_result.rule, rtype, tp.input, bg)
                        if pred is None or not np.array_equal(pred, tp.output):
                            all_test_ok = False
                            break
                    test_solved = all_test_ok
                    break

                if total_diff < best_diff:
                    best_diff = total_diff
                    best_rewrite = rtype

        elapsed = time.time() - t1

        # Report
        avg_pres = np.mean([f.preservation_ratio for f in factors])
        sel_desc = sel_result.description if sel_result.rule else "no_rule"
        mark = "TRAIN_VERIFIED" if train_verified else f"diff={int(best_diff) if best_diff < 1e9 else '?'}"
        test_mark = " TEST_PASS" if test_solved else ""

        print(
            f"{tid}: {mark:>20} sel={sel_desc:<45} "
            f"recall={sel_result.train_mask_recall:.2f} "
            f"prec={sel_result.train_mask_precision:.2f} "
            f"rewrite={best_rewrite or 'none'}"
            f"{test_mark} ({elapsed:.2f}s)"
        )

        results.append({
            "tid": tid,
            "train_verified": train_verified,
            "test_solved": test_solved,
            "selection": sel_desc,
            "recall": sel_result.train_mask_recall,
            "precision": sel_result.train_mask_precision,
            "rewrite": best_rewrite,
            "diff": int(best_diff) if best_diff < float('inf') else -1,
        })

    total_time = time.time() - t0
    n = len(results)
    n_verified = sum(1 for r in results if r["train_verified"])
    n_test = sum(1 for r in results if r["test_solved"])
    n_sel = sum(1 for r in results if r["selection"] != "no_rule")
    n_good_sel = sum(1 for r in results if r["recall"] > 0.5 and r["precision"] > 0.5)

    print(f"\n{'='*80}")
    print(f"NGS SELECTION INDUCTION — {n} tasks")
    print(f"{'='*80}")
    print(f"Train verified:           {n_verified}/{n}")
    print(f"Exact solve (test):       {n_test}/{n}")
    print(f"Selection rules found:    {n_sel}/{n}")
    print(f"Good selection (R>0.5,P>0.5): {n_good_sel}/{n}")
    print(f"Time:                     {total_time:.1f}s")
    print(f"{'='*80}")

    print(f"\nBEFORE selection: 0/12 train_verified")
    print(f"AFTER selection:  {n_verified}/12 train_verified, {n_test}/12 exact solve")


if __name__ == "__main__":
    main()
