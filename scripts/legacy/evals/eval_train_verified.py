#!/usr/bin/env python3
"""Evaluate train-verified candidate rate and best-diff metrics.

The primary metrics for the structural program synthesis branch:
1. train_verified_candidate_rate — tasks with at least one exact train match
2. best_train_total_diff@K — closest candidate's total diff across train demos
3. family_coverage — which synthesis families produce train-verified candidates
4. generalization_gap — among train-verified, how many fail on test

Usage:
    python scripts/eval_train_verified.py [--dataset v2-eval] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, iter_tasks, load_arc_task
from aria.types import grid_eq


def evaluate_task(tid, task):
    """Evaluate a single task. Returns metrics dict."""
    demos = task.train
    total_train_pixels = sum(d.output.size for d in demos)

    train_verified = False
    best_diff = total_train_pixels  # worst case = all pixels wrong
    families_produced = set()
    families_verified = set()
    n_candidates = 0
    best_candidate = None

    # Path 1: scene_solve candidates
    try:
        from aria.core.scene_solve import infer_scene_programs, verify_scene_program
        from aria.core.scene_executor import execute_scene_program
        candidates = infer_scene_programs(demos)
        n_candidates += len(candidates)
        for prog in candidates:
            diff = _compute_train_diff(prog, demos)
            if diff is not None:
                ptype = type(prog).__name__
                families_produced.add(ptype)
                if diff < best_diff:
                    best_diff = diff
                    best_candidate = prog
                if diff == 0:
                    train_verified = True
                    families_verified.add(ptype)
    except Exception:
        pass

    # Path 2: synthesize_from_observations
    try:
        from aria.synthesize import synthesize_from_observations
        sr = synthesize_from_observations(demos)
        families_produced.add("synthesize")
        n_candidates += sr.candidates_tested
        if sr.solved:
            train_verified = True
            families_verified.add("synthesize")
            best_diff = 0
            best_candidate = sr.winning_program
    except Exception:
        pass

    # Test generalization (only if train-verified)
    test_solved = False
    test_diff = None
    if train_verified and best_candidate is not None:
        test_solved, test_diff = _check_test(best_candidate, task)

    return {
        "tid": tid,
        "train_verified": train_verified,
        "best_diff": int(best_diff),
        "total_pixels": total_train_pixels,
        "diff_frac": round(best_diff / max(total_train_pixels, 1), 4),
        "n_candidates": n_candidates,
        "families_produced": sorted(families_produced),
        "families_verified": sorted(families_verified),
        "test_solved": test_solved,
        "test_diff": test_diff,
    }


def _compute_train_diff(prog, demos):
    """Total pixel diff across all train demos. None if execution fails."""
    total = 0
    for d in demos:
        try:
            if hasattr(prog, "verify_on_demo"):
                ok = prog.verify_on_demo(d.input, d.output)
                total += 0 if ok else d.output.size
            elif hasattr(prog, "execute"):
                out = prog.execute(d.input)
                if out.shape == d.output.shape:
                    total += int(np.sum(out != d.output))
                else:
                    total += d.output.size
            else:
                from aria.core.scene_executor import execute_scene_program
                out = execute_scene_program(prog, d.input)
                if out.shape == d.output.shape:
                    total += int(np.sum(out != d.output))
                else:
                    total += d.output.size
        except Exception:
            return None
    return total


def _check_test(prog, task):
    """Check if a train-verified program generalizes to test."""
    total_diff = 0
    for tp in task.test:
        try:
            if hasattr(prog, "execute"):
                out = prog.execute(tp.input)
            elif hasattr(prog, "verify_on_demo"):
                ok = prog.verify_on_demo(tp.input, tp.output)
                total_diff += 0 if ok else tp.output.size
                continue
            else:
                from aria.runtime.executor import execute
                out = execute(prog, tp.input, None)
            if np.array_equal(out, tp.output):
                pass
            else:
                total_diff += int(np.sum(out != tp.output)) if out.shape == tp.output.shape else tp.output.size
        except Exception:
            return False, None
    return total_diff == 0, total_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v2-eval")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    tasks_iter = list(iter_tasks(ds))
    if args.tasks:
        task_set = set(args.tasks)
        tasks_iter = [(tid, t) for tid, t in tasks_iter if tid in task_set]

    results = []
    t0 = time.time()

    for i, (tid, task) in enumerate(tasks_iter):
        t1 = time.time()
        r = evaluate_task(tid, task)
        r["time"] = round(time.time() - t1, 1)
        results.append(r)

        mark = "VERIFIED" if r["train_verified"] else f"diff={r['best_diff']}"
        test_mark = " TEST_PASS" if r["test_solved"] else ""
        print(
            f"[{i+1:3d}/{len(tasks_iter)}] {tid} {mark:>20} "
            f"cands={r['n_candidates']:>3}{test_mark} ({r['time']:.1f}s)",
            file=sys.stderr, flush=True,
        )

    total_time = time.time() - t0

    # Aggregate metrics
    n = len(results)
    n_verified = sum(1 for r in results if r["train_verified"])
    n_test_solved = sum(1 for r in results if r["test_solved"])
    non_verified_diffs = [r["best_diff"] for r in results if not r["train_verified"]]

    all_fam_produced = set()
    all_fam_verified = set()
    for r in results:
        all_fam_produced.update(r["families_produced"])
        all_fam_verified.update(r["families_verified"])

    # Diff buckets
    buckets = [(0, 1), (1, 50), (50, 200), (200, 500), (500, 1000), (1000, 99999)]
    diff_dist = {f"[{lo},{hi})": sum(1 for r in results if lo <= r["best_diff"] < hi) for lo, hi in buckets}

    report = {
        "dataset": args.dataset,
        "n_tasks": n,
        "train_verified_rate": round(n_verified / max(n, 1), 4),
        "train_verified_count": n_verified,
        "exact_solve_count": n_test_solved,
        "exact_solve_rate": round(n_test_solved / max(n, 1), 4),
        "mean_best_diff_non_verified": round(np.mean(non_verified_diffs), 1) if non_verified_diffs else 0,
        "median_best_diff_non_verified": round(float(np.median(non_verified_diffs)), 1) if non_verified_diffs else 0,
        "diff_distribution": diff_dist,
        "families_produced": sorted(all_fam_produced),
        "families_verified": sorted(all_fam_verified),
        "generalization_gap": n_verified - n_test_solved,
        "total_time_sec": round(total_time, 1),
    }

    if args.json:
        print(json.dumps({"report": report, "per_task": results}, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"TRAIN-VERIFIED CANDIDATE RATE REPORT")
        print(f"{'='*60}")
        print(f"Dataset:                    {report['dataset']}")
        print(f"Tasks:                      {report['n_tasks']}")
        print(f"Time:                       {report['total_time_sec']}s")
        print(f"")
        print(f"train_verified_rate:        {report['train_verified_count']}/{report['n_tasks']} ({report['train_verified_rate']*100:.1f}%)")
        print(f"exact_solve_rate:           {report['exact_solve_count']}/{report['n_tasks']} ({report['exact_solve_rate']*100:.1f}%)")
        print(f"generalization_gap:         {report['generalization_gap']}")
        print(f"")
        print(f"mean best_diff (non-verif): {report['mean_best_diff_non_verified']}")
        print(f"med  best_diff (non-verif): {report['median_best_diff_non_verified']}")
        print(f"")
        print(f"best_diff distribution:")
        for bucket, count in report["diff_distribution"].items():
            print(f"  {bucket:>14}: {count:>3}")
        print(f"")
        print(f"Families produced:   {report['families_produced']}")
        print(f"Families verified:   {report['families_verified']}")

        # Near-miss tasks (diff < 100)
        near = [r for r in results if 0 < r["best_diff"] < 100]
        if near:
            print(f"\nNear-miss tasks (diff < 100):")
            for r in sorted(near, key=lambda x: x["best_diff"]):
                print(f"  {r['tid']}: diff={r['best_diff']} ({r['diff_frac']*100:.1f}%)")

        print(f"{'='*60}")


if __name__ == "__main__":
    main()
