"""Generic candidate scoring based on verifier diffs.

Compares verifier outcomes across candidates to reward generic progress
signals: correct dimensions, fewer pixel diffs, better palette overlap,
fewer execution errors. No task-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.types import DemoPair, Grid, Program, TaskContext, VerifyMode
from aria.verify.mode import detect_mode
from aria.verify.trace import traced_execute


@dataclass(frozen=True)
class DemoScore:
    """Per-demo verifier features."""

    demo_idx: int
    passed: bool
    dims_correct: bool
    pixel_diff_count: int | None  # None if dims wrong or exec error
    execution_error: bool
    palette_overlap: float  # 0.0–1.0
    total_pixels: int


@dataclass(frozen=True)
class CandidateScore:
    """Aggregate score for a candidate program across all demos."""

    passed: bool
    demos_passed: int
    total_demos: int
    dims_correct: int
    pixel_diff_total: int  # lower better; -1 when some demos lack pixel data
    execution_errors: int
    palette_overlap_avg: float
    demo_scores: tuple[DemoScore, ...]

    @property
    def rank_key(self) -> tuple:
        """Sorting key — lower is better."""
        return (
            not self.passed,
            -self.demos_passed,
            -self.dims_correct,
            self.execution_errors,
            self.pixel_diff_total if self.pixel_diff_total >= 0 else 10**9,
            -self.palette_overlap_avg,
        )

    def __lt__(self, other: CandidateScore) -> bool:
        return self.rank_key < other.rank_key

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "demos_passed": self.demos_passed,
            "total_demos": self.total_demos,
            "dims_correct": self.dims_correct,
            "pixel_diff_total": self.pixel_diff_total,
            "execution_errors": self.execution_errors,
            "palette_overlap_avg": round(self.palette_overlap_avg, 4),
        }


def score_program(program: Program, demos: tuple[DemoPair, ...]) -> CandidateScore:
    """Score a program against all demos, collecting per-demo features."""
    mode = detect_mode(program)
    demo_scores: list[DemoScore] = []

    for i, demo in enumerate(demos):
        ctx = _build_context(mode, demos, i)
        result, _trace = traced_execute(program, demo.input, ctx, demo.output)

        if result is None:
            demo_scores.append(DemoScore(
                demo_idx=i,
                passed=False,
                dims_correct=False,
                pixel_diff_count=None,
                execution_error=True,
                palette_overlap=0.0,
                total_pixels=int(demo.output.size),
            ))
            continue

        expected = demo.output
        dims_correct = result.shape == expected.shape
        passed = dims_correct and bool(np.array_equal(result, expected))

        if dims_correct:
            pixel_diff = int(np.sum(result != expected))
        else:
            pixel_diff = None

        demo_scores.append(DemoScore(
            demo_idx=i,
            passed=passed,
            dims_correct=dims_correct,
            pixel_diff_count=pixel_diff,
            execution_error=False,
            palette_overlap=_palette_overlap(result, expected),
            total_pixels=int(expected.size),
        ))

    demos_passed = sum(1 for ds in demo_scores if ds.passed)
    dims_correct = sum(1 for ds in demo_scores if ds.dims_correct)
    execution_errors = sum(1 for ds in demo_scores if ds.execution_error)

    pixel_diffs = [
        ds.pixel_diff_count
        for ds in demo_scores
        if ds.pixel_diff_count is not None
    ]
    pixel_diff_total = sum(pixel_diffs) if pixel_diffs else -1

    palette_overlaps = [
        ds.palette_overlap for ds in demo_scores if not ds.execution_error
    ]
    palette_overlap_avg = (
        sum(palette_overlaps) / len(palette_overlaps) if palette_overlaps else 0.0
    )

    return CandidateScore(
        passed=all(ds.passed for ds in demo_scores) and len(demo_scores) > 0,
        demos_passed=demos_passed,
        total_demos=len(demos),
        dims_correct=dims_correct,
        pixel_diff_total=pixel_diff_total,
        execution_errors=execution_errors,
        palette_overlap_avg=palette_overlap_avg,
        demo_scores=tuple(demo_scores),
    )


def _build_context(
    mode: VerifyMode,
    demos: tuple[DemoPair, ...],
    demo_idx: int,
) -> TaskContext | None:
    if mode == VerifyMode.STATELESS:
        return None
    if mode == VerifyMode.LEAVE_ONE_OUT:
        ctx_demos = demos[:demo_idx] + demos[demo_idx + 1:]
        return TaskContext(demos=ctx_demos)
    if demo_idx == 0:
        return TaskContext(demos=())
    return TaskContext(demos=demos[:demo_idx])


def _palette_overlap(actual: Grid, expected: Grid) -> float:
    expected_palette = set(int(v) for v in np.unique(expected))
    if not expected_palette:
        return 1.0
    actual_palette = set(int(v) for v in np.unique(actual))
    return len(expected_palette & actual_palette) / len(expected_palette)
