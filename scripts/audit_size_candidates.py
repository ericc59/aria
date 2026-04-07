#!/usr/bin/env python3
"""Audit output-size candidate cost and win rates.

Prevents silent stage-1 candidate bloat by reporting per-candidate
invocation count, cost, and win rate. Run periodically to ensure
new candidates justify their existence.

Usage:
    python scripts/audit_size_candidates.py --dataset v1-train --limit 50
    python scripts/audit_size_candidates.py --dataset v1-train --output /tmp/size_audit.json
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit output-size candidate costs")
    parser.add_argument("--dataset", default="v1-train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from aria.datasets import get_dataset, list_task_ids, load_arc_task
    from aria.core.guidance_search_cost import audit_search_costs

    ds = get_dataset(args.dataset)
    task_ids = list_task_ids(ds)
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    print(f"Auditing {len(task_ids)} tasks from {args.dataset}...")
    pairs = []
    for tid in task_ids:
        try:
            task = load_arc_task(ds, tid)
            pairs.append((tid, task.train))
        except Exception:
            pass

    t0 = time.time()
    audit = audit_search_costs(pairs)
    elapsed = time.time() - t0
    print(f"Audit complete in {elapsed:.1f}s\n")

    # Report
    print(f"{'Candidate':<45} {'Mean ms':>8} {'Max ms':>8} {'Total s':>8} {'Wins':>5} {'Rate':>6} {'$/Win':>8}")
    print("─" * 95)
    for c in audit.candidate_costs:
        rate = c.n_produces / max(c.n_tasks, 1)
        cpw = c.total_time_s / max(c.n_is_winner, 1) if c.n_is_winner > 0 else float("inf")
        cpw_str = f"{cpw:.2f}s" if cpw < 1000 else "∞"
        print(f"{c.name:<45} {c.mean_time_ms:>8.1f} {c.max_time_ms:>8.0f} {c.total_time_s:>8.2f} {c.n_is_winner:>5} {rate:>6.1%} {cpw_str:>8}")

    print(f"\nTotal enumeration: {audit.total_time_s:.1f}s")
    print(f"Top cost: {audit.top_cost_candidate} ({audit.top_cost_fraction:.0%})")
    print(f"Prunable (never-winning, >1ms): {audit.prunable_time_s:.1f}s")

    # Flag candidates that might be bloat
    print("\n⚠ Candidates to review (high cost, low/zero wins):")
    for c in audit.candidate_costs:
        if c.mean_time_ms > 5 and c.n_is_winner == 0:
            print(f"  {c.name}: {c.mean_time_ms:.0f}ms mean, 0 wins")
        elif c.mean_time_ms > 10 and c.n_is_winner <= 1:
            print(f"  {c.name}: {c.mean_time_ms:.0f}ms mean, {c.n_is_winner} win(s)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(audit.to_dict(), f, indent=2)
        print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
