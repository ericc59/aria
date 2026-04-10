#!/usr/bin/env python3
"""Legacy multi-mode eval runner kept for ablation/debug comparisons.

Usage:
    python scripts/legacy/evals/solve_eval.py                        # v2-eval (default)
    python scripts/legacy/evals/solve_eval.py --dataset v1-eval
    python scripts/legacy/evals/solve_eval.py --dataset v2-eval --budget 30
    python scripts/legacy/evals/solve_eval.py --dataset v1-train --search-only
    python scripts/legacy/evals/solve_eval.py --task d8e07eb2        # single task
    python scripts/legacy/evals/solve_eval.py --task d8e07eb2 --task 3dc255db  # multiple
    python scripts/legacy/evals/solve_eval.py --search-only          # skip guided engine
    python scripts/legacy/evals/solve_eval.py --guided-only          # skip search engine
    python scripts/legacy/evals/solve_eval.py --verbose              # show per-task details
"""
import argparse
import os
import sys
import time

import numpy as np

from aria.datasets import get_dataset, load_arc_task


def run_search_only(demos, budget):
    from aria.search.search import search_programs
    prog = search_programs(demos, time_budget=budget)
    if prog:
        return {'program': prog, 'source': 'search', 'description': prog.description}
    return None


def run_guided_only(demos, budget):
    from aria.guided.dsl import synthesize_program, _verify
    prog = synthesize_program(demos, time_budget=budget)
    if prog and _verify(prog, demos):
        return {'program': prog, 'source': 'guided', 'description': str(prog)}
    return None


def run_full(demos, budget):
    from aria.solve import solve_task
    result = solve_task(demos, time_budget=budget)
    if result and result.get('program'):
        return result
    return None


def main():
    parser = argparse.ArgumentParser(description='Run aria solver on ARC datasets')
    parser.add_argument('--dataset', default='v2-eval',
                        choices=['v1-train', 'v1-eval', 'v2-train', 'v2-eval'],
                        help='Dataset split (default: v2-eval)')
    parser.add_argument('--task', action='append', default=None,
                        help='Specific task ID(s) to run (can repeat)')
    parser.add_argument('--budget', type=float, default=15.0,
                        help='Time budget per task in seconds (default: 15)')
    parser.add_argument('--search-only', action='store_true',
                        help='Only run the search engine (skip guided)')
    parser.add_argument('--guided-only', action='store_true',
                        help='Only run the guided engine (skip search)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-task status')
    args = parser.parse_args()

    ds = get_dataset(args.dataset)

    # Collect task IDs
    if args.task:
        task_ids = args.task
    else:
        task_ids = sorted(f.replace('.json', '') for f in os.listdir(ds.root)
                          if f.endswith('.json'))

    # Pick solver
    if args.search_only:
        solver = run_search_only
        solver_name = 'search'
    elif args.guided_only:
        solver = run_guided_only
        solver_name = 'guided'
    else:
        solver = run_full
        solver_name = 'full'

    print(f"Dataset: {args.dataset} ({len(task_ids)} tasks)")
    print(f"Solver: {solver_name}, budget: {args.budget}s/task")
    print()

    solved = []
    train_only = []
    failed = []
    errors = []
    t_total = time.time()

    for i, tid in enumerate(task_ids):
        t0 = time.time()
        try:
            t = load_arc_task(ds, tid)
            demos = [(d.input, d.output) for d in t.train]

            result = solver(demos, args.budget)
            elapsed = time.time() - t0

            if result and result.get('program'):
                prog = result['program']
                src = result.get('source', solver_name)
                desc = result.get('description', '')[:60]

                # Verify on test
                test_ok = all(
                    np.array_equal(prog.execute(d.input), d.output)
                    for d in t.test
                )

                if test_ok:
                    solved.append(tid)
                    print(f"  PASS  {tid} [{src}] {desc} ({elapsed:.1f}s)")
                else:
                    train_only.append(tid)
                    diffs = [np.sum(prog.execute(d.input) != d.output) for d in t.test]
                    print(f"  TRAIN {tid} [{src}] test_diffs={diffs} ({elapsed:.1f}s)")
            else:
                failed.append(tid)
                print(f"  --    {tid} ({elapsed:.1f}s)")
        except Exception as e:
            errors.append((tid, str(e)))
            if args.verbose:
                print(f"  ERROR {tid}: {e}")

        # Flush after each task for real-time output
        sys.stdout.flush()

    elapsed_total = time.time() - t_total

    # Report
    print()
    print(f"{'='*60}")
    print(f"Results: {args.dataset} ({solver_name})")
    print(f"{'='*60}")
    print(f"  Test-passing: {len(solved)}/{len(task_ids)} ({100*len(solved)/max(len(task_ids),1):.1f}%)")
    if train_only:
        print(f"  Train-only:   {len(train_only)}")
    if errors:
        print(f"  Errors:       {len(errors)}")
    print(f"  Time:         {elapsed_total:.1f}s ({elapsed_total/max(len(task_ids),1):.1f}s/task)")
    print()

    if solved:
        print(f"Solved tasks:")
        for tid in solved:
            print(f"  {tid}")

    if train_only and args.verbose:
        print(f"\nTrain-only tasks:")
        for tid in train_only:
            print(f"  {tid}")

    if errors and args.verbose:
        print(f"\nErrors:")
        for tid, msg in errors[:10]:
            print(f"  {tid}: {msg}")


if __name__ == '__main__':
    main()
