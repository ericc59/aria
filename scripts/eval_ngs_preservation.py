#!/usr/bin/env python3
"""Evaluate NGS with preservation factoring.

Reports preservation metrics and residual coverage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task
from aria.decomposition import detect_bg
from aria.ngs.preservation import factor_preservation
from aria.ngs.residual_explain import (
    explain_residual, unify_residual_rules, _execute_residual_rule,
)

# Slice: 12 high-preservation same-shape tasks
SLICE = [
    '8e5c0c38',   # 98.7%, delete anomalies
    '88e364bc',    # 98.3%, object swap
    '135a2760',    # 98.1%, add 1px
    '9bbf930d',    # 97.5%, swap
    '409aa875',    # 97.4%, add singletons
    '97d7923e',    # 96.0%, recolor to adjacent
    '332f06d7',    # 94.8%, recolor swap
    'dbff022c',    # 94.4%, fill enclosed (legend-based)
    '4c416de3',    # 95.0%, mixed per-region
    '8b7bacbf',    # 95.1%, add enclosed
    'c4d067a0',    # 94.7%, add scattered
    'e376de54',    # 96.5%, add+delete
]


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

        # Phase 1: Factor
        factors = []
        per_demo_rules = []
        for d in demos:
            bg = detect_bg(d.input)
            pf = factor_preservation(d.input, d.output, bg)
            factors.append(pf)
            rules = explain_residual(pf, d.input, d.output, bg)
            per_demo_rules.append(rules)

        # Phase 2: Unify
        unified = unify_residual_rules(per_demo_rules, factors, demos)

        # Phase 3: Test
        test_solved = False
        if unified is not None and unified.train_verified:
            all_ok = True
            for tp in task.test:
                bg = detect_bg(tp.input)
                pred = _execute_residual_rule(unified.rule.description, None, tp.input, bg)
                if pred is None or not np.array_equal(pred, tp.output):
                    all_ok = False
                    break
            test_solved = all_ok

        elapsed = time.time() - t1

        # Metrics
        avg_pres = np.mean([f.preservation_ratio for f in factors])
        avg_res = np.mean([f.n_residual for f in factors])
        total_px = np.mean([f.n_preserved + f.n_residual for f in factors])
        n_demo_rules = [len(r) for r in per_demo_rules]
        all_demos_have_rules = all(n > 0 for n in n_demo_rules)
        any_demos_have_rules = any(n > 0 for n in n_demo_rules)

        rule_types = set()
        for rules in per_demo_rules:
            for r in rules:
                rule_types.add(r.rule_type)

        train_verified = unified is not None and unified.train_verified

        # Cross-demo residual consistency
        change_type_per_demo = []
        adj_match_per_demo = []
        for pf in factors:
            types = tuple(sorted(r.change_type for r in pf.residual_regions))
            change_type_per_demo.append(types)
            adj_match = all(
                len(r.output_colors) == 1 and
                next(iter(r.output_colors)) in r.adjacent_colors
                for r in pf.residual_regions
            ) if pf.residual_regions else False
            adj_match_per_demo.append(adj_match)

        consistent_types = len(set(change_type_per_demo)) == 1
        all_adj_match = all(adj_match_per_demo) and any(adj_match_per_demo)

        mark = "TRAIN_VERIFIED" if train_verified else ("unified" if unified else "no_rule")
        test_mark = " TEST_PASS" if test_solved else ""

        print(
            f"{tid}: {mark:>15} pres={avg_pres:.1%} res={avg_res:>4.0f}/{total_px:.0f}px "
            f"demos_w_rules={sum(1 for n in n_demo_rules if n > 0)}/{len(demos)} "
            f"types={str(sorted(rule_types) if rule_types else 'none'):<35} "
            f"consistent_types={consistent_types} adj_match={all_adj_match}"
            f"{test_mark} ({elapsed:.2f}s)"
        )

        results.append({
            "tid": tid,
            "train_verified": train_verified,
            "test_solved": test_solved,
            "avg_preservation": avg_pres,
            "avg_residual": avg_res,
            "demos_with_rules": sum(1 for n in n_demo_rules if n > 0),
            "total_demos": len(demos),
            "rule_types": sorted(rule_types),
            "consistent_types": consistent_types,
            "all_adj_match": all_adj_match,
        })

    total_time = time.time() - t0
    n = len(results)
    n_verified = sum(1 for r in results if r["train_verified"])
    n_test = sum(1 for r in results if r["test_solved"])
    n_all_rules = sum(1 for r in results if r["demos_with_rules"] == r["total_demos"])
    n_any_rules = sum(1 for r in results if r["demos_with_rules"] > 0)
    n_consistent = sum(1 for r in results if r["consistent_types"])
    n_adj_match = sum(1 for r in results if r["all_adj_match"])
    avg_pres_all = np.mean([r["avg_preservation"] for r in results])
    avg_res_all = np.mean([r["avg_residual"] for r in results])

    print(f"\n{'='*80}")
    print(f"NGS + PRESERVATION FACTORING — {n} tasks")
    print(f"{'='*80}")
    print(f"Train verified:               {n_verified}/{n}")
    print(f"Exact solve (test):           {n_test}/{n}")
    print(f"Tasks w/ ALL demo rules:      {n_all_rules}/{n}")
    print(f"Tasks w/ any demo rules:      {n_any_rules}/{n}")
    print(f"Consistent residual types:    {n_consistent}/{n}")
    print(f"All residual adj-color match: {n_adj_match}/{n}")
    print(f"Avg preservation ratio:       {avg_pres_all:.1%}")
    print(f"Avg residual size:            {avg_res_all:.0f}px")
    print(f"Time:                         {total_time:.1f}s")
    print(f"{'='*80}")

    # Comparison with baseline (no preservation)
    print(f"\nBEFORE preservation: 0/120 train_verified, 6/359 demos explained (1.7%)")
    print(f"AFTER preservation:  {n_verified}/{n} train_verified, "
          f"{sum(r['demos_with_rules'] for r in results)}/{sum(r['total_demos'] for r in results)} "
          f"demos with rules, {n_adj_match}/{n} tasks with cross-demo adj-color pattern")
    print(f"\nPreservation factoring reduces explanation target from ~100% to ~{(1-avg_pres_all)*100:.0f}% of grid")
    print(f"Residual is {avg_res_all:.0f}px average vs full grid")


if __name__ == "__main__":
    main()
