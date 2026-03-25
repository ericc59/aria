"""Step-by-step execution tracing for error feedback.

Runs a program and records the value at each step, producing a trace
that can be included in re-proposal context to help the proposer
understand where computation diverged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import (
    Assert,
    Bind,
    Grid,
    Program,
    StepTraceEntry,
    TaskContext,
    grid_eq,
)


def _summarize_value(val: Any) -> str:
    """Produce a short string summary of a runtime value for the trace."""
    if isinstance(val, np.ndarray):
        if val.ndim == 2:
            return f"Grid({val.shape[0]}x{val.shape[1]})"
        return f"array(shape={val.shape})"
    if isinstance(val, (set, frozenset)):
        return f"set(len={len(val)})"
    if isinstance(val, (list, tuple)):
        return f"list(len={len(val)})"
    if isinstance(val, dict):
        return f"map(len={len(val)})"
    return repr(val)


def traced_execute(
    program: Program,
    input_grid: Grid,
    ctx: TaskContext | None = None,
    expected_output: Grid | None = None,
) -> tuple[Grid | None, tuple[StepTraceEntry, ...]]:
    """Execute a program with step-by-step tracing.

    Returns (result_grid_or_None, trace_entries).
    If execution fails at any step, returns None for the grid.
    """
    from aria.runtime.executor import eval_expr

    env: dict[str, Any] = {"input": input_grid}
    if ctx is not None:
        env["ctx"] = ctx

    entries: list[StepTraceEntry] = []

    for step in program.steps:
        match step:
            case Bind(name=name, typ=_, expr=expr):
                try:
                    val = eval_expr(expr, env)
                    env[name] = val
                    entries.append(StepTraceEntry(
                        step_name=f"bind {name}",
                        value=_summarize_value(val),
                        ok=True,
                    ))
                except Exception as e:
                    entries.append(StepTraceEntry(
                        step_name=f"bind {name}",
                        value=str(e),
                        ok=False,
                        suspect=f"execution error: {e}",
                    ))
                    return None, tuple(entries)

            case Assert(pred=pred):
                try:
                    val = eval_expr(pred, env)
                    if not val:
                        entries.append(StepTraceEntry(
                            step_name="assert",
                            value="False",
                            ok=False,
                            suspect="assertion failed",
                        ))
                        return None, tuple(entries)
                    entries.append(StepTraceEntry(
                        step_name="assert",
                        value="True",
                        ok=True,
                    ))
                except Exception as e:
                    entries.append(StepTraceEntry(
                        step_name="assert",
                        value=str(e),
                        ok=False,
                        suspect=f"assertion error: {e}",
                    ))
                    return None, tuple(entries)

    result = env.get(program.output)
    if result is None:
        return None, tuple(entries)

    if not isinstance(result, np.ndarray):
        entries.append(StepTraceEntry(
            step_name="yield",
            value=_summarize_value(result),
            ok=False,
            suspect=f"output is not a Grid, got {type(result).__name__}",
        ))
        return None, tuple(entries)

    # If we have expected output, mark dimension/content mismatches
    if expected_output is not None and not grid_eq(result, expected_output):
        suspect_parts = []
        if result.shape != expected_output.shape:
            suspect_parts.append(
                f"dims mismatch: got {result.shape} vs expected {expected_output.shape}"
            )
        else:
            diff_count = int(np.sum(result != expected_output))
            suspect_parts.append(f"{diff_count} pixels differ")
        entries.append(StepTraceEntry(
            step_name="yield",
            value=_summarize_value(result),
            ok=False,
            suspect="; ".join(suspect_parts),
        ))

    return result, tuple(entries)


def compute_diff(actual: Grid, expected: Grid) -> dict[str, Any]:
    """Compute a structured diff between actual and expected grids."""
    diff: dict[str, Any] = {
        "expected_dims": (expected.shape[0], expected.shape[1]),
        "actual_dims": (actual.shape[0], actual.shape[1]),
    }

    if actual.shape != expected.shape:
        diff["pixel_diff_count"] = None
        diff["pixel_diff_summary"] = "dimension mismatch"
        return diff

    mismatch = actual != expected
    diff["pixel_diff_count"] = int(np.sum(mismatch))

    # Summarize which rows/cols are wrong
    wrong_rows = [int(r) for r in range(mismatch.shape[0]) if mismatch[r].any()]
    if wrong_rows:
        diff["pixel_diff_summary"] = f"wrong rows: {wrong_rows}"
    else:
        diff["pixel_diff_summary"] = "identical"

    return diff
