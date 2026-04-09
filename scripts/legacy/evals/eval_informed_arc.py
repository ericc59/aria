#!/usr/bin/env python3
"""Evaluate output-informed multi-step search on ARC-2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.datasets import get_dataset, load_arc_task, list_task_ids
from aria.guided.informed_search import informed_search
from aria.guided.search import predict_test


def main():
    ds = get_dataset("v2-eval")
    tids = list_task_ids(ds)[:120]

    print("Output-informed multi-step search on ARC-2...")
    n_train = 0
    n_test = 0
    n_total = 0

    for tid in tids:
        try:
            task = load_arc_task(ds, tid)
        except:
            continue
        if not all(d.input.shape == d.output.shape for d in task.train):
            continue

        n_total += 1
        train_pairs = [(d.input, d.output) for d in task.train]

        result = informed_search(train_pairs, max_candidates=2000, max_steps=3)

        if result.solved:
            n_train += 1
            tok = all(
                np.array_equal(predict_test(result.program, tp.input), tp.output)
                for tp in task.test
            )
            if tok:
                n_test += 1
            print(f"  {tid}: TRAIN_VERIFIED{' TEST_PASS' if tok else ''} "
                  f"prog={result.program} cands={result.candidates_tried}")

    print(f"\n{'='*70}")
    print(f"INFORMED SEARCH — {n_total} same-shape ARC-2 tasks")
    print(f"{'='*70}")
    print(f"Train verified: {n_train}/{n_total}")
    print(f"Test solved:    {n_test}/{n_total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
