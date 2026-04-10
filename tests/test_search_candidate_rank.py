from __future__ import annotations

from aria.search.candidate_rank import rank_search_candidates, score_search_program
from aria.search.proposal_memory import SearchProposalPrior
from aria.search.sketch import SearchProgram, SearchStep
from aria.types import grid_from_list


def test_score_search_program_prefers_exact_match() -> None:
    inp = grid_from_list([[1, 2], [3, 4]])
    out = grid_from_list([[1, 2], [3, 4]])
    prior = SearchProposalPrior.empty()

    exact = SearchProgram(steps=[], provenance="empty")
    wrong = SearchProgram(steps=[SearchStep("flip_h")], provenance="flip_h")

    exact_score = score_search_program(
        exact,
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
    )
    wrong_score = score_search_program(
        wrong,
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
    )

    assert exact_score.rank_key < wrong_score.rank_key


def test_rank_search_candidates_uses_prior_as_tiebreak() -> None:
    inp = grid_from_list([[7]])
    out = grid_from_list([[7]])
    tile_like = SearchProgram(steps=[SearchStep("flip_h")], provenance="tile_like")
    scale_like = SearchProgram(steps=[SearchStep("flip_v")], provenance="scale_like")

    # Same execution score; prior should decide.
    prior = SearchProposalPrior(
        global_counts={"flip_h": 5, "flip_v": 1},
        by_signature={"dims:same": {"flip_h": 2}},
    )
    ranked = rank_search_candidates(
        [scale_like, tile_like],
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
    )
    assert ranked[0].signature == "flip_h"
