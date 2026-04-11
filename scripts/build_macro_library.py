#!/usr/bin/env python3
"""Build a macro library from solved search traces.

Reads search traces (JSONL or eval reports), mines exact repeated
compositions, and writes a macro library JSON file.

Usage:
    python scripts/build_macro_library.py \
        --traces data/search_traces.jsonl \
        --output results/search_macro_library.json

    python scripts/build_macro_library.py \
        --reports results/eval_v1-train_*.json \
        --output results/search_macro_library.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aria.search.macro_miner import (
    mine_macros,
    load_traces_jsonl,
    load_traces_from_eval_reports,
)


def main():
    parser = argparse.ArgumentParser(description='Build macro library from search traces')
    parser.add_argument('--traces', type=Path, default=None,
                        help='JSONL file with search traces')
    parser.add_argument('--reports', nargs='*', type=Path, default=None,
                        help='Eval report JSON files')
    parser.add_argument('--output', type=Path,
                        default=Path('results/search_macro_library.json'),
                        help='Output macro library JSON')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum group frequency to produce a macro')
    parser.add_argument('--min-steps', type=int, default=1,
                        help='Minimum steps per macro')
    parser.add_argument('--require-test-correct', action='store_true',
                        default=True,
                        help='Only include test-correct traces (default: True)')
    parser.add_argument('--no-require-test-correct', dest='require_test_correct',
                        action='store_false',
                        help='Include all traces regardless of test correctness')
    args = parser.parse_args()

    # Load traces
    traces = []
    if args.traces and args.traces.exists():
        traces.extend(load_traces_jsonl(str(args.traces)))
    if args.reports:
        traces.extend(load_traces_from_eval_reports(
            [str(p) for p in args.reports]))

    if not traces:
        print('No traces found. Run eval with trace capture first.')
        return

    print(f'Loaded {len(traces)} traces')

    # Mine
    lib = mine_macros(
        traces,
        min_frequency=args.min_frequency,
        min_steps=args.min_steps,
        require_test_correct=args.require_test_correct,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lib.save_json(str(args.output))

    # Report
    print(f'\nMined {len(lib.macros)} macros -> {args.output}')
    for m in lib.macros:
        test_pct = f'{m.solve_rate:.0%}' if m.solve_rate > 0 else '?'
        print(f'  {m.name}: freq={m.frequency} tasks={m.source_task_count} '
              f'test={test_pct} [{m.action_signature}]')


if __name__ == '__main__':
    main()
