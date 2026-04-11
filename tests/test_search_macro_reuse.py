"""Tests for macro reuse in search: loading, scoring, and ranking."""

from __future__ import annotations

import numpy as np

from aria.search.macros import Macro, MacroLibrary, load_default_macro_library
from aria.search.candidate_rank import (
    SearchCandidateScore,
    score_search_program,
    rank_search_candidates,
)
from aria.search.proposal_memory import SearchProposalPrior
from aria.search.sketch import SearchProgram, SearchStep, StepSelect


def test_load_default_macro_library():
    """Default macro library should load without error (may be empty)."""
    lib = load_default_macro_library()
    assert isinstance(lib, MacroLibrary)
    # Actual count depends on whether library file exists
    assert isinstance(lib.macros, list)


def test_macro_library_score_candidate_match():
    """score_candidate should return > 0 for matching macros."""
    lib = MacroLibrary()
    lib.add(Macro(name='test', frequency=5, solve_rate=1.0,
                  action_signature='recolor_map',
                  source_provenances=['derive:color_map']))
    lib.add(Macro(name='other', frequency=3, solve_rate=0.8,
                  action_signature='scale',
                  source_provenances=['derive:pixel_scale']))

    # Exact provenance match → full weight
    assert lib.score_candidate('recolor_map', 'derive:color_map') == 5.0
    # Action-only match (no provenance) → half weight
    assert abs(lib.score_candidate('recolor_map', '') - 2.5) < 1e-9
    # Exact provenance for scale
    assert abs(lib.score_candidate('scale', 'derive:pixel_scale') - 2.4) < 1e-9
    # No match
    assert lib.score_candidate('nonexistent') == 0.0


def test_macro_library_score_empty():
    """Empty macro library should score everything as 0."""
    lib = MacroLibrary()
    assert lib.score_candidate('recolor') == 0.0
    assert lib.score_candidate('', '') == 0.0


def test_candidate_score_includes_macro():
    """SearchCandidateScore should include macro_score in rank_key."""
    score_with = SearchCandidateScore(
        demos_passed=1, dims_correct=1, pixel_diff_total=0,
        execution_errors=0, palette_overlap_avg=1.0,
        prior_score=0, model_score=0, macro_score=5.0, step_count=1)
    score_without = SearchCandidateScore(
        demos_passed=1, dims_correct=1, pixel_diff_total=0,
        execution_errors=0, palette_overlap_avg=1.0,
        prior_score=0, model_score=0, macro_score=0.0, step_count=1)

    # Higher macro_score should rank better (lower rank_key)
    assert score_with.rank_key < score_without.rank_key


def test_rank_with_macro_library():
    """Candidates matching a macro should rank higher."""
    lib = MacroLibrary()
    lib.add(Macro(name='m', frequency=10, solve_rate=1.0,
                  action_signature='recolor_map'))

    prior = SearchProposalPrior.empty()

    # Two programs: one matches the macro signature, one doesn't
    prog_match = SearchProgram(
        steps=[SearchStep('recolor_map', {})],
        provenance='test_match')
    prog_no = SearchProgram(
        steps=[SearchStep('move', {'dr': 1, 'dc': 0})],
        provenance='test_no')

    # Create a trivial demo (doesn't matter for ranking, just execution)
    demo = (np.zeros((3, 3), dtype=np.int8), np.zeros((3, 3), dtype=np.int8))

    ranked = rank_search_candidates(
        [prog_no, prog_match],
        [demo],
        task_signatures=frozenset(),
        prior=prior,
        macro_library=lib,
    )

    # prog_match should come first due to macro_score
    assert ranked[0].provenance == 'test_match'


def test_rank_without_macro_library():
    """Search should work normally when no macro library is provided."""
    prior = SearchProposalPrior.empty()

    prog = SearchProgram(
        steps=[SearchStep('recolor', {'color': 1}, StepSelect('largest'))],
        provenance='test')

    demo = (np.zeros((3, 3), dtype=np.int8), np.zeros((3, 3), dtype=np.int8))

    ranked = rank_search_candidates(
        [prog],
        [demo],
        task_signatures=frozenset(),
        prior=prior,
        macro_library=None,
    )
    assert len(ranked) == 1


def test_search_programs_with_macros_smoke():
    """search_programs should run without error with macro library loaded."""
    from aria.search.search import search_programs

    # Simple task that should solve
    inp = np.array([[0, 1], [1, 0]], dtype=np.int8)
    out = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.int8)

    result = search_programs([(inp, out)])
    assert result is not None  # template_broadcast should solve this
