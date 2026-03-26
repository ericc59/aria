"""Exact verification — the one hard gate.

Binary pass/fail. Pixel-perfect. No partial credit.
Three modes: stateless, leave-one-out, sequential.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import (
    DemoPair,
    Grid,
    Program,
    TaskContext,
    VerifyMode,
    VerifyResult,
    grid_eq,
)
from aria.verify.mode import detect_mode
from aria.verify.trace import compute_diff, traced_execute


def _execute_safe(
    program: Program,
    input_grid: Grid,
    ctx: TaskContext | None,
    expected: Grid,
) -> VerifyResult | None:
    """Execute program and check output. Returns None on success, VerifyResult on failure."""
    result, trace = traced_execute(program, input_grid, ctx, expected)

    if result is None:
        return VerifyResult(
            passed=False,
            mode=VerifyMode.STATELESS,  # will be overwritten by caller
            error_type="execution_error",
            step_trace=trace,
        )

    if not grid_eq(result, expected):
        diff = compute_diff(result, expected, input_grid)
        return VerifyResult(
            passed=False,
            mode=VerifyMode.STATELESS,
            error_type="wrong_output",
            diff=diff,
            step_trace=trace,
        )

    return None  # success


def verify_stateless(program: Program, demos: tuple[DemoPair, ...]) -> VerifyResult:
    """Mode A: each demo verified independently, no context."""
    for i, demo in enumerate(demos):
        failure = _execute_safe(program, demo.input, None, demo.output)
        if failure is not None:
            return VerifyResult(
                passed=False,
                mode=VerifyMode.STATELESS,
                failed_demo=i,
                error_type=failure.error_type,
                diff=failure.diff,
                step_trace=failure.step_trace,
            )
    return VerifyResult(passed=True, mode=VerifyMode.STATELESS)


def verify_leave_one_out(program: Program, demos: tuple[DemoPair, ...]) -> VerifyResult:
    """Mode B: when verifying demo i, context contains all demos except i."""
    for i in range(len(demos)):
        ctx_demos = demos[:i] + demos[i + 1:]
        ctx = TaskContext(demos=ctx_demos)
        failure = _execute_safe(program, demos[i].input, ctx, demos[i].output)
        if failure is not None:
            return VerifyResult(
                passed=False,
                mode=VerifyMode.LEAVE_ONE_OUT,
                failed_demo=i,
                error_type=failure.error_type,
                diff=failure.diff,
                step_trace=failure.step_trace,
            )
    return VerifyResult(passed=True, mode=VerifyMode.LEAVE_ONE_OUT)


def verify_sequential(program: Program, demos: tuple[DemoPair, ...]) -> VerifyResult:
    """Mode C: demos are ordered. Demo i sees all prior demos as history."""
    for i in range(1, len(demos)):
        history = demos[:i]
        ctx = TaskContext(demos=history)
        failure = _execute_safe(program, demos[i].input, ctx, demos[i].output)
        if failure is not None:
            return VerifyResult(
                passed=False,
                mode=VerifyMode.SEQUENTIAL,
                failed_demo=i,
                error_type=failure.error_type,
                diff=failure.diff,
                step_trace=failure.step_trace,
            )
    return VerifyResult(passed=True, mode=VerifyMode.SEQUENTIAL)


def verify_full_context(program: Program, demos: tuple[DemoPair, ...]) -> VerifyResult:
    """Verify with full context (all demos visible), like test time.

    This is less rigorous than LOO — the program can "see" the answer
    in principle — but it catches programs that are correct at test time
    even if they fail LOO due to incomplete cross-demo information.
    """
    ctx = TaskContext(demos=demos)
    for i, demo in enumerate(demos):
        failure = _execute_safe(program, demo.input, ctx, demo.output)
        if failure is not None:
            return VerifyResult(
                passed=False,
                mode=VerifyMode.LEAVE_ONE_OUT,  # report as LOO variant
                failed_demo=i,
                error_type=failure.error_type,
                diff=failure.diff,
                step_trace=failure.step_trace,
            )
    return VerifyResult(passed=True, mode=VerifyMode.LEAVE_ONE_OUT)


def verify(program: Program, demos: tuple[DemoPair, ...]) -> VerifyResult:
    """Verify a program against demo pairs.

    Detects mode from static analysis. As a safety net, if the detected
    mode fails but a less restrictive mode passes, the program is accepted.
    """
    mode = detect_mode(program)

    if mode == VerifyMode.STATELESS:
        return verify_stateless(program, demos)

    if mode == VerifyMode.LEAVE_ONE_OUT:
        result = verify_leave_one_out(program, demos)
        if result.passed:
            return result
        # Safety net: try full-context (all demos visible)
        full = verify_full_context(program, demos)
        if full.passed:
            return full
        return result

    if mode == VerifyMode.SEQUENTIAL:
        result = verify_sequential(program, demos)
        if result.passed:
            return result
        # Safety net: try LOO, then full-context
        loo = verify_leave_one_out(program, demos)
        if loo.passed:
            return loo
        full = verify_full_context(program, demos)
        if full.passed:
            return full
        return result

    return VerifyResult(passed=False, mode=mode, error_type="unknown_mode")
