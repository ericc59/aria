#!/usr/bin/env python3
"""Export structured solve traces from eval reports to JSONL.

Reads eval result JSON files, extracts solve_trace records from
solved tasks, and writes them as one-per-line JSONL for offline
macro mining and pattern analysis.

Usage:
    python scripts/build_search_traces.py \
        --reports results/eval_v1-train_*.json \
        --output data/search_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_traces(report_paths: list[Path]) -> list[dict]:
    """Extract solve traces from eval report files."""
    traces: list[dict] = []
    seen_tasks: set[str] = set()

    for path in sorted(report_paths, reverse=True):  # newest first
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        tasks = data.get('tasks', [])
        for task_result in tasks:
            if not task_result.get('solved'):
                continue
            trace = task_result.get('solve_trace')
            if trace is None:
                continue

            tid = trace.get('task_id', '')
            if tid in seen_tasks:
                continue
            seen_tasks.add(tid)

            trace['_source_report'] = str(path)
            traces.append(trace)

    return traces


def main():
    parser = argparse.ArgumentParser(description='Export search solve traces')
    parser.add_argument('--reports', nargs='+', type=Path, required=True,
                        help='Eval report JSON files')
    parser.add_argument('--output', type=Path, default=Path('data/search_traces.jsonl'),
                        help='Output JSONL file')
    args = parser.parse_args()

    traces = extract_traces(args.reports)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for trace in traces:
            f.write(json.dumps(trace) + '\n')

    # Summary
    provenance_counts: dict[str, int] = {}
    sig_counts: dict[str, int] = {}
    for t in traces:
        prov = t.get('provenance', 'unknown')
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
        sig = ' -> '.join(t.get('step_actions', []))
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    print(f'Exported {len(traces)} traces to {args.output}')
    print(f'\nBy provenance:')
    for prov, count in sorted(provenance_counts.items(), key=lambda x: -x[1]):
        print(f'  {prov}: {count}')
    print(f'\nTop action signatures:')
    for sig, count in sorted(sig_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'  {sig}: {count}')


if __name__ == '__main__':
    main()
