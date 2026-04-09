#!/usr/bin/env python3
"""Apply multi-step guided search to ARC-2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task, list_task_ids
from aria.guided.workspace import _detect_bg
from aria.guided.search import search, predict_test


# Top 20 same-shape ARC-2 by preservation
ARC_SLICE = [
    '8e5c0c38', '88e364bc', '135a2760', '9bbf930d', '409aa875',
    '7b80bb43', 'b99e7126', 'e376de54', '97d7923e', '3e6067c3',
    '332f06d7', '4c416de3', '8b7bacbf', 'dbff022c', 'c4d067a0',
    '53fb4810', '7ed72f31', 'c7f57c3e', 'b10624e5', 'a25697e4',
]


def main():
    ds = get_dataset("v2-eval")

    print(f"Multi-step search on {len(ARC_SLICE)} ARC-2 tasks...")
    print(f"(max_candidates=5000, max_steps=3)")
    print(f"{'='*80}")

    n_train = 0
    n_test = 0
    total = 0

    for tid in ARC_SLICE:
        try:
            task = load_arc_task(ds, tid)
        except Exception as e:
            print(f"  {tid}: LOAD_ERROR")
            continue

        total += 1
        train_pairs = [(d.input, d.output) for d in task.train]

        t1 = time.time()
        result = search(train_pairs, max_candidates=5000, max_steps=3)
        elapsed = time.time() - t1

        if result.solved:
            n_train += 1
            tok = all(
                np.array_equal(predict_test(result.program, tp.input), tp.output)
                for tp in task.test
            )
            if tok:
                n_test += 1
            mark = "TRAIN_VERIFIED" + (" TEST_PASS" if tok else " test_fail")
            print(f"  {tid}: {mark}  program={result.program}  cands={result.candidates_tried} ({elapsed:.1f}s)")
        else:
            pres = np.mean([np.sum(d.input == d.output) / d.input.size for d in task.train])
            print(f"  {tid}: diff={result.train_diff:>4d}  pres={pres:.0%}  cands={result.candidates_tried} ({elapsed:.1f}s)")

    print(f"\n{'='*80}")
    print(f"Train verified: {n_train}/{total}")
    print(f"Test solved:    {n_test}/{total}")
    print(f"{'='*80}")

    # Also try full 120
    print(f"\nFull 120-task scan...")
    tids = list_task_ids(ds)[:120]
    full_train = 0
    full_test = 0
    full_total = 0

    for tid in tids:
        try:
            task = load_arc_task(ds, tid)
        except:
            continue
        if not all(d.input.shape == d.output.shape for d in task.train):
            continue
        full_total += 1
        train_pairs = [(d.input, d.output) for d in task.train]
        result = search(train_pairs, max_candidates=5000, max_steps=3)
        if result.solved:
            full_train += 1
            tok = all(
                np.array_equal(predict_test(result.program, tp.input), tp.output)
                for tp in task.test
            )
            if tok:
                full_test += 1
            print(f"  {tid}: TRAIN_VERIFIED{' TEST_PASS' if tok else ''} program={result.program}")

    print(f"\nFull scan: {full_train}/{full_total} train-verified, {full_test}/{full_total} test-solved")


if __name__ == "__main__":
    main()
