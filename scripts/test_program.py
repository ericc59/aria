#!/usr/bin/env python3
"""Test an ARIA program against a task's demo pairs.

Usage:
    python scripts/test_program.py <task_id> '<program_text>'

Example:
    python scripts/test_program.py 0d3d703e 'bind mapping = infer_map(ctx, 0, 0)
    bind result = apply_color_map(mapping, input)
    yield result'

Outputs: PASS/FAIL per demo, actual vs expected grids on failure.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aria.proposer.parser import parse_program, ParseError
from aria.runtime.executor import execute, ExecutionError
from aria.solver import load_task
from aria.types import TaskContext, grid_eq
from aria.verify.mode import detect_mode, VerifyMode

DATA_DIRS = [
    os.path.expanduser("~/dev/arcagi/arc-agi-benchmarking/data/public-v1/training"),
    os.path.expanduser("~/dev/arcagi/arc-agi-benchmarking/data/public-v1/evaluation"),
    os.path.expanduser("~/dev/arcagi/arc-agi-benchmarking/data/public-v2/training"),
    os.path.expanduser("~/dev/arcagi/arc-agi-benchmarking/data/public-v2/evaluation"),
]


def find_task(task_id: str):
    for d in DATA_DIRS:
        path = os.path.join(d, f"{task_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return load_task(json.load(f))
    return None


def grid_str(grid) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_program.py <task_id> '<program>'")
        sys.exit(1)

    task_id = sys.argv[1]
    program_text = sys.argv[2]

    task = find_task(task_id)
    if task is None:
        print(f"ERROR: Task '{task_id}' not found")
        sys.exit(1)

    try:
        program = parse_program(program_text)
    except ParseError as e:
        print(f"PARSE ERROR: {e}")
        sys.exit(1)

    mode = detect_mode(program)
    all_pass = True

    for i, demo in enumerate(task.train):
        ctx = None if mode == VerifyMode.STATELESS else TaskContext(demos=task.train)
        try:
            result = execute(program, demo.input, ctx)
        except (ExecutionError, Exception) as e:
            print(f"Demo {i}: EXEC ERROR: {e}")
            all_pass = False
            continue

        if grid_eq(result, demo.output):
            print(f"Demo {i}: PASS")
        else:
            all_pass = False
            print(f"Demo {i}: FAIL")
            print(f"  Expected ({demo.output.shape[0]}x{demo.output.shape[1]}):")
            for row in demo.output:
                print(f"    {' '.join(str(int(c)) for c in row)}")
            print(f"  Got ({result.shape[0]}x{result.shape[1]}):")
            for row in result:
                print(f"    {' '.join(str(int(c)) for c in row)}")

    if all_pass:
        print(f"\nALL {len(task.train)} DEMOS PASS")
    else:
        print(f"\nFAILED")


if __name__ == "__main__":
    main()
