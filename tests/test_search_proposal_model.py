from __future__ import annotations

from pathlib import Path

from aria.search.proposal_corpus import SearchProposalExample
from aria.search.proposal_model import SearchFamilyModel


def test_search_family_model_prefers_signature_compatible_family() -> None:
    examples = [
        SearchProposalExample(
            schema_version=1,
            task_id="a",
            task_signatures=("dims:different", "size:multiplicative"),
            family="tile",
            description="search: derive:exact_tile [tile]",
            source_report="r1.json",
        ),
        SearchProposalExample(
            schema_version=1,
            task_id="b",
            task_signatures=("dims:different", "size:multiplicative"),
            family="tile",
            description="search: derive:exact_tile [tile]",
            source_report="r1.json",
        ),
        SearchProposalExample(
            schema_version=1,
            task_id="c",
            task_signatures=("dims:same", "color:palette_same"),
            family="recolor_map",
            description="search: derive:color_map [recolor_map]",
            source_report="r1.json",
        ),
    ]
    model = SearchFamilyModel.train(examples)
    assert model.score_family("tile", frozenset({"size:multiplicative"})) > model.score_family(
        "recolor_map",
        frozenset({"size:multiplicative"}),
    )


def test_search_family_model_round_trip(tmp_path: Path) -> None:
    examples = [
        SearchProposalExample(
            schema_version=1,
            task_id="a",
            task_signatures=("dims:same",),
            family="flip_h",
            description="search: flip_h [flip_h]",
            source_report="r.json",
        )
    ]
    model = SearchFamilyModel.train(examples)
    path = tmp_path / "model.json"
    model.save_json(path)
    loaded = SearchFamilyModel.load_json(path)
    assert loaded == model
