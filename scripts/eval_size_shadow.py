#!/usr/bin/env python3
"""Train output-size model on train split, evaluate shadow mode on held-out split.

Usage:
    python scripts/eval_size_shadow.py --dataset v1-train --train-frac 0.7
    python scripts/eval_size_shadow.py --dataset v1-train --output /tmp/shadow_eval.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v1-train")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from aria.datasets import get_dataset, list_task_ids, load_arc_task
    from aria.core.guidance_size_model import (
        train_size_model, shadow_evaluate_batch, ShadowEvalReport,
    )

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)

    print(f"Loading {len(task_ids)} tasks from {args.dataset}...")
    pairs = []
    for tid in task_ids:
        try:
            task = load_arc_task(ds, tid)
            pairs.append((tid, task.train))
        except Exception:
            pass

    print(f"Loaded {len(pairs)} tasks")

    # Task-aware split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(pairs))
    split = int(len(pairs) * args.train_frac)
    train_pairs = [pairs[i] for i in indices[:split]]
    eval_pairs = [pairs[i] for i in indices[split:]]

    print(f"Train: {len(train_pairs)} tasks, Eval: {len(eval_pairs)} tasks")

    # Train
    print("\nTraining output-size model...")
    t0 = time.time()
    model = train_size_model(train_pairs)
    print(f"Trained in {time.time()-t0:.1f}s")

    # Shadow eval on train
    print("\nShadow evaluation on TRAIN split...")
    train_report, _ = shadow_evaluate_batch(train_pairs, model)
    _print_report("TRAIN", train_report)

    # Shadow eval on held-out
    print("\nShadow evaluation on HELD-OUT split...")
    eval_report, eval_results = shadow_evaluate_batch(eval_pairs, model)
    _print_report("HELD-OUT", eval_report)

    # Detailed per-task results on held-out (first 20)
    print("\n\nSample held-out task results:")
    print(f"{'Task':<20} {'Default':>8} {'Model':>8} {'Δ':>6} {'Winner Family':<20}")
    print("─" * 65)
    for r in eval_results[:20]:
        delta = r.rank_improvement
        marker = "✓" if delta > 0 else ("=" if delta == 0 else "✗")
        print(f"{r.task_id:<20} {r.default_winner_rank:>8} {r.model_winner_rank:>8} {delta:>+5d}{marker} {r.default_winner_family:<20}")

    # Readiness decision
    print("\n" + "=" * 70)
    print("INTEGRATION READINESS DECISION")
    print("=" * 70)

    ready = (
        eval_report.pct_worse < 0.05
        and eval_report.avg_rank_improvement >= 0
        and eval_report.n_evaluated >= 50
    )

    if ready:
        print("✓ READY for live reordering")
        print(f"  - {eval_report.pct_improved:.1%} of tasks improved")
        print(f"  - {eval_report.pct_worse:.1%} of tasks worsened (< 5% threshold)")
        print(f"  - {eval_report.avg_rank_improvement:+.2f} avg rank improvement")
        print(f"  - {eval_report.total_attempts_saved} total attempts saved across {eval_report.n_evaluated} tasks")
    else:
        print("✗ NOT ready for live reordering")
        if eval_report.pct_worse >= 0.05:
            print(f"  - {eval_report.pct_worse:.1%} tasks worsened (≥ 5% threshold)")
        if eval_report.avg_rank_improvement < 0:
            print(f"  - avg rank improvement is negative ({eval_report.avg_rank_improvement:+.2f})")
        if eval_report.n_evaluated < 50:
            print(f"  - only {eval_report.n_evaluated} tasks evaluated (< 50 minimum)")

    if args.output:
        full = {
            "train_report": train_report.to_dict(),
            "eval_report": eval_report.to_dict(),
            "n_train": len(train_pairs),
            "n_eval": len(eval_pairs),
            "ready": ready,
        }
        with open(args.output, "w") as f:
            json.dump(full, f, indent=2)
        print(f"\nFull report saved to {args.output}")


def _print_report(label: str, report: ShadowEvalReport) -> None:
    print(f"\n  {label} Shadow Results ({report.n_evaluated} tasks evaluated):")
    print(f"  Avg candidates per task:  {report.avg_candidates:.1f}")
    print(f"  Avg default winner rank:  {report.avg_default_rank:.2f}")
    print(f"  Avg model winner rank:    {report.avg_model_rank:.2f}")
    print(f"  Avg rank improvement:     {report.avg_rank_improvement:+.2f}")
    print(f"  Tasks improved:           {report.pct_improved:.1%}")
    print(f"  Tasks same:               {report.pct_same:.1%}")
    print(f"  Tasks worsened:           {report.pct_worse:.1%}")
    print(f"  Total attempts saved:     {report.total_attempts_saved}")

    if report.family_results:
        print(f"\n  Per-family breakdown:")
        for fam, stats in sorted(report.family_results.items()):
            print(f"    {fam:<20} n={stats['count']:>3}  avg_imp={stats['avg_improvement']:+.2f}  improved={stats['pct_improved']:.0%}")


if __name__ == "__main__":
    main()
