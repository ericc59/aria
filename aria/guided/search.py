"""Multi-step search over the grammar.

Builds programs step-by-step: at each point, enumerate possible next
actions, apply them, and verify. Programs are extended incrementally
until STOP is reached and the program verifies on all train demos.

This is compositional search — multi-step programs emerge naturally.
"""

from __future__ import annotations

from typing import Any
from collections import deque

import numpy as np

from aria.guided.grammar import (
    Program, Action, Act, Support, Target, Rewrite,
    execute_program,
)
from aria.guided.workspace import Workspace, build_workspace, _detect_bg
from aria.types import Grid


class SearchResult:
    __slots__ = ("solved", "program", "candidates_tried", "train_diff")

    def __init__(self, solved: bool, program: Program | None,
                 candidates_tried: int, train_diff: int):
        self.solved = solved
        self.program = program
        self.candidates_tried = candidates_tried
        self.train_diff = train_diff


def search(
    train: list[tuple[Grid, Grid]],
    max_candidates: int = 2000,
    max_steps: int = 4,
) -> SearchResult:
    """Build programs step-by-step and verify on train demos.

    Uses BFS over partial programs, extending each with valid next actions.
    """
    if not train:
        return SearchResult(False, None, 0, 0)

    bgs = [_detect_bg(inp) for inp, _ in train]
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)
    candidates_tried = 0

    # BFS queue of partial programs
    queue: deque[Program] = deque()
    queue.append(Program())

    while queue and candidates_tried < max_candidates:
        partial = queue.popleft()
        n_steps = sum(1 for a in partial.actions if a.act in (Act.NEXT, Act.STOP))

        if n_steps >= max_steps:
            continue

        # Try extending with each valid next action sequence
        for extension in _enumerate_extensions(partial, train, bgs):
            candidates_tried += 1
            candidate = partial.copy()
            for action in extension:
                candidate.append(action)

            # If extension ends with STOP, verify
            if extension[-1].act == Act.STOP:
                ok, diff = _verify(candidate, train, bgs)
                if ok:
                    return SearchResult(True, candidate, candidates_tried, 0)
                if diff < best_diff:
                    best_diff = diff

            # If extension ends with NEXT, enqueue for further expansion
            if extension[-1].act == Act.NEXT:
                queue.append(candidate)

            if candidates_tried >= max_candidates:
                break

    return SearchResult(False, None, candidates_tried, best_diff)


def _verify(prog: Program, train: list[tuple[Grid, Grid]], bgs: list[int]) -> tuple[bool, int]:
    """Verify a complete program on all train demos."""
    total_diff = 0
    for (inp, out), bg in zip(train, bgs):
        try:
            pred = execute_program(prog, inp, bg)
        except Exception:
            return False, sum(out.size for _, out in train)
        if not np.array_equal(pred, out):
            if pred.shape == out.shape:
                total_diff += int(np.sum(pred != out))
            else:
                total_diff += out.size
    return total_diff == 0, total_diff


def _enumerate_extensions(
    partial: Program,
    train: list[tuple[Grid, Grid]],
    bgs: list[int],
) -> list[list[Action]]:
    """Enumerate valid next action sequences (one complete step).

    A step is: [SELECT_TARGET, REWRITE, BIND*, STOP_or_NEXT]
    """
    extensions = []

    # Determine what colors exist in all train grids (inputs AND outputs)
    all_colors = set()
    for inp, out in train:
        all_colors.update(int(v) for v in np.unique(inp))
        all_colors.update(int(v) for v in np.unique(out))

    # Each extension is a complete step ending with STOP or NEXT
    for target in Target:
        for rewrite in _compatible_rewrites(target):
            for bindings in _enumerate_bindings(rewrite, target, all_colors, bgs[0]):
                # Try with STOP
                step = [Action(Act.SELECT_TARGET, target), Action(Act.REWRITE, rewrite)]
                step.extend(bindings)
                step.append(Action(Act.STOP))
                extensions.append(step)

                # Try with NEXT (for multi-step programs)
                step_next = [Action(Act.SELECT_TARGET, target), Action(Act.REWRITE, rewrite)]
                step_next.extend(bindings)
                step_next.append(Action(Act.NEXT))
                extensions.append(step_next)

    return extensions


def _compatible_rewrites(target: Target) -> list[Rewrite]:
    """Which rewrites make sense for a given target type."""
    if target == Target.ENCLOSED_BG:
        return [Rewrite.FILL]
    if target == Target.BY_COLOR:
        return [Rewrite.RECOLOR, Rewrite.DELETE]
    if target == Target.ASYMMETRIC:
        return [Rewrite.SYMMETRIZE]
    if target == Target.ANOMALY:
        return [Rewrite.PERIODIC_REPAIR]
    if target == Target.SINGLETONS:
        return [Rewrite.DELETE, Rewrite.COPY_STAMP]
    if target == Target.ADJACENT_TO:
        return [Rewrite.RECOLOR]
    if target == Target.ALL_CELLS:
        return [Rewrite.RECOLOR, Rewrite.FILL]
    return []


def _enumerate_bindings(
    rewrite: Rewrite,
    target: Target,
    colors: set[int],
    bg: int,
) -> list[list[Action]]:
    """Enumerate binding combinations for a rewrite."""
    results = []

    if rewrite == Rewrite.FILL:
        # Literal color
        for c in colors:
            if c != bg:
                results.append([
                    Action(Act.BIND, param_name="color", param_value=c),
                    Action(Act.BIND, param_name="color_source", param_value="literal"),
                ])
        # From-context: frame color
        results.append([
            Action(Act.BIND, param_name="color_source", param_value="from_context:frame"),
        ])

    elif rewrite == Rewrite.RECOLOR:
        if target == Target.BY_COLOR:
            for from_c in colors:
                for to_c in colors:
                    if from_c != to_c and from_c != bg:
                        results.append([
                            Action(Act.BIND, param_name="from_color", param_value=from_c),
                            Action(Act.BIND, param_name="color", param_value=to_c),
                        ])
        elif target == Target.ADJACENT_TO:
            results.append([
                Action(Act.BIND, param_name="color_source", param_value="from_context:adjacent_singleton"),
            ])
        else:
            for to_c in colors:
                if to_c != bg:
                    results.append([
                        Action(Act.BIND, param_name="color", param_value=to_c),
                    ])

    elif rewrite == Rewrite.DELETE:
        if target == Target.BY_COLOR:
            for c in colors:
                if c != bg:
                    results.append([
                        Action(Act.BIND, param_name="from_color", param_value=c),
                    ])
        else:
            results.append([])

    elif rewrite == Rewrite.SYMMETRIZE:
        for axis in ["h", "v"]:
            results.append([
                Action(Act.BIND, param_name="axis", param_value=axis),
            ])

    elif rewrite == Rewrite.PERIODIC_REPAIR:
        for axis in ["row", "col"]:
            results.append([
                Action(Act.BIND, param_name="axis", param_value=axis),
            ])

    elif rewrite == Rewrite.COPY_STAMP:
        for sc in colors:
            for mc in colors:
                if sc != mc and sc != bg and mc != bg:
                    results.append([
                        Action(Act.BIND, param_name="stamp_color", param_value=sc),
                        Action(Act.BIND, param_name="marker_color", param_value=mc),
                    ])

    if not results:
        results.append([])

    return results


def predict_test(prog: Program, test_input: Grid) -> Grid | None:
    """Execute a verified program on a test input."""
    bg = _detect_bg(test_input)
    try:
        return execute_program(prog, test_input, bg)
    except Exception:
        return None
