"""Guided multi-step search: selector ranks candidate extensions.

Same BFS structure as unguided search, but at each node the selector
scores all candidate next-steps and we explore in score order.
This is beam search with learned ordering.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from aria.guided.grammar import (
    Program, Action, Act, Target, Rewrite, execute_program,
)
from aria.guided.search import (
    SearchResult, _verify, _enumerate_extensions, _compatible_rewrites,
    _enumerate_bindings, predict_test,
)
from aria.guided.selector_model import StepSelector
from aria.guided.training_data import TARGET_TO_IDX, REWRITE_TO_IDX, _featurize_workspace
from aria.guided.workspace import build_workspace, _detect_bg
from aria.types import Grid


def guided_search(
    train: list[tuple[Grid, Grid]],
    model: StepSelector,
    max_candidates: int = 2000,
    max_steps: int = 4,
    beam_width: int = 20,
) -> SearchResult:
    """Guided multi-step search with learned step ordering.

    Uses a priority queue (best-first) instead of plain BFS.
    The selector scores each candidate extension; we explore
    highest-scoring first.
    """
    if not train:
        return SearchResult(False, None, 0, 0)

    bgs = [_detect_bg(inp) for inp, _ in train]
    ws = build_workspace(train[0][0], train[0][1])
    features = _featurize_workspace(ws)
    best_diff = sum(int(np.sum(inp != out)) for inp, out in train)
    candidates_tried = 0

    # Priority queue: (-score, program)
    # Higher score = explored first (negated for min-heap)
    counter = 0
    heap: list[tuple[float, int, Program]] = []
    heapq.heappush(heap, (0.0, counter, Program()))
    counter += 1

    while heap and candidates_tried < max_candidates:
        neg_score, _, partial = heapq.heappop(heap)
        n_steps = sum(1 for a in partial.actions if a.act in (Act.NEXT, Act.STOP))
        if n_steps >= max_steps:
            continue

        # Score and sort extensions
        extensions = _enumerate_extensions(partial, train, bgs)
        scored_ext = []
        for ext in extensions:
            # Extract target and rewrite from extension
            target_idx = 0
            rewrite_idx = 0
            is_last = ext[-1].act == Act.STOP
            for a in ext:
                if a.act == Act.SELECT_TARGET and a.choice is not None:
                    target_idx = TARGET_TO_IDX.get(a.choice, 0)
                if a.act == Act.REWRITE and a.choice is not None:
                    rewrite_idx = REWRITE_TO_IDX.get(a.choice, 0)
            score = model.score_step(features, target_idx, rewrite_idx, is_last)
            scored_ext.append((score, ext))

        # Take top beam_width extensions
        scored_ext.sort(key=lambda x: -x[0])
        for score, ext in scored_ext[:beam_width]:
            candidates_tried += 1
            candidate = partial.copy()
            for action in ext:
                candidate.append(action)

            if ext[-1].act == Act.STOP:
                ok, diff = _verify(candidate, train, bgs)
                if ok:
                    return SearchResult(True, candidate, candidates_tried, 0)
                if diff < best_diff:
                    best_diff = diff

            if ext[-1].act == Act.NEXT:
                heapq.heappush(heap, (-score + neg_score, counter, candidate))
                counter += 1

            if candidates_tried >= max_candidates:
                break

    return SearchResult(False, None, candidates_tried, best_diff)
