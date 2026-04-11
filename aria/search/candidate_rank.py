"""Candidate ranking for `aria/search`.

This is a small inference-policy layer above exact verification:
- score partially-correct SearchPrograms with generic verifier-style features
- combine that with the persistent proposal prior and macro library
- reorder candidates before exact verification

No task-specific logic. No learned model yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.search.proposal_memory import SearchProposalPrior
from aria.search.proposal_model import SearchFamilyModel
from aria.search.macros import MacroLibrary
from aria.search.sketch import SearchProgram


@dataclass(frozen=True)
class SearchCandidateScore:
    demos_passed: int
    dims_correct: int
    pixel_diff_total: int
    execution_errors: int
    palette_overlap_avg: float
    prior_score: float
    model_score: float
    macro_score: float
    step_count: int

    @property
    def rank_key(self) -> tuple:
        return (
            -self.demos_passed,
            -self.dims_correct,
            self.execution_errors,
            self.pixel_diff_total,
            -self.palette_overlap_avg,
            -self.macro_score,
            -self.model_score,
            -self.prior_score,
            self.step_count,
        )


def score_search_program(
    prog: SearchProgram,
    demos: list[tuple[np.ndarray, np.ndarray]],
    *,
    task_signatures: frozenset[str],
    prior: SearchProposalPrior,
    model: SearchFamilyModel | None = None,
    macro_library: MacroLibrary | None = None,
    max_demos: int = 2,
) -> SearchCandidateScore:
    demos_passed = 0
    dims_correct = 0
    pixel_diff_total = 0
    execution_errors = 0
    palette_overlaps: list[float] = []

    for inp, out in demos[:max_demos]:
        try:
            pred = prog.execute(inp)
        except Exception:
            execution_errors += 1
            continue

        if pred.shape == out.shape:
            dims_correct += 1
            diff = int(np.sum(pred != out))
            pixel_diff_total += diff
            if diff == 0:
                demos_passed += 1
            palette_overlaps.append(_palette_overlap(pred, out))
        else:
            pixel_diff_total += out.size
            palette_overlaps.append(0.0)

    palette_overlap_avg = (
        sum(palette_overlaps) / len(palette_overlaps) if palette_overlaps else 0.0
    )
    prior_score = prior.score_family(prog.signature, task_signatures)
    model_score = model.score_family(prog.signature, task_signatures) if model is not None else 0.0
    macro_score = macro_library.score_candidate(prog.signature, prog.provenance) if macro_library is not None else 0.0
    return SearchCandidateScore(
        demos_passed=demos_passed,
        dims_correct=dims_correct,
        pixel_diff_total=pixel_diff_total,
        execution_errors=execution_errors,
        palette_overlap_avg=palette_overlap_avg,
        prior_score=prior_score,
        model_score=model_score,
        macro_score=macro_score,
        step_count=len(prog.steps),
    )


def rank_search_candidates(
    programs: list[SearchProgram],
    demos: list[tuple[np.ndarray, np.ndarray]],
    *,
    task_signatures: frozenset[str],
    prior: SearchProposalPrior,
    model: SearchFamilyModel | None = None,
    macro_library: MacroLibrary | None = None,
    max_demos: int = 2,
) -> list[SearchProgram]:
    scored = [
        (
            score_search_program(
                prog,
                demos,
                task_signatures=task_signatures,
                prior=prior,
                model=model,
                macro_library=macro_library,
                max_demos=max_demos,
            ),
            idx,
            prog,
        )
        for idx, prog in enumerate(programs)
    ]
    scored.sort(key=lambda item: (item[0].rank_key, item[1]))
    return [prog for _, _, prog in scored]


def _palette_overlap(actual: np.ndarray, expected: np.ndarray) -> float:
    expected_palette = set(int(v) for v in np.unique(expected))
    if not expected_palette:
        return 1.0
    actual_palette = set(int(v) for v in np.unique(actual))
    return len(expected_palette & actual_palette) / len(expected_palette)
