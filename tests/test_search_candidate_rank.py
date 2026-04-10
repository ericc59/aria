from __future__ import annotations

from aria.search.candidate_rank import rank_search_candidates, score_search_program
from aria.search.proposal_memory import SearchProposalPrior
from aria.search.proposal_model import SearchFamilyModel
from aria.search.sketch import SearchProgram, SearchStep
from aria.types import grid_from_list


def test_score_search_program_prefers_exact_match() -> None:
    inp = grid_from_list([[1, 2], [3, 4]])
    out = grid_from_list([[1, 2], [3, 4]])
    prior = SearchProposalPrior.empty()
    model = SearchFamilyModel.empty()

    exact = SearchProgram(steps=[], provenance="empty")
    wrong = SearchProgram(steps=[SearchStep("flip_h")], provenance="flip_h")

    exact_score = score_search_program(
        exact,
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
        model=model,
    )
    wrong_score = score_search_program(
        wrong,
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
        model=model,
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
    model = SearchFamilyModel(
        family_counts={"flip_h": 3, "flip_v": 1},
        signature_counts={"flip_h": {"dims:same": 3}, "flip_v": {"dims:different": 1}},
        vocabulary=("dims:different", "dims:same"),
        total_examples=4,
    )
    ranked = rank_search_candidates(
        [scale_like, tile_like],
        [(inp, out)],
        task_signatures=frozenset({"dims:same"}),
        prior=prior,
        model=model,
    )
    assert ranked[0].signature == "flip_h"
