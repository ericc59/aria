"""Tests for residual-to-edit priors."""

from __future__ import annotations

from aria.core.editor_env import ActionType
from aria.core.residual_priors import (
    EditPrior, get_edit_prior, get_all_priors,
)


def test_get_edit_prior_specific():
    prior = get_edit_prior("partial_match", "replication")
    assert isinstance(prior, EditPrior)
    assert ActionType.BIND in prior.action_types
    assert "__replicate__" in prior.binding_namespaces


def test_get_edit_prior_fallback():
    prior = get_edit_prior("partial_match", "unknown_lane")
    assert isinstance(prior, EditPrior)
    assert len(prior.action_types) > 0


def test_get_edit_prior_default():
    prior = get_edit_prior("unknown_category", "unknown_lane")
    assert isinstance(prior, EditPrior)
    assert prior.rationale == "default: generic parameter edits"


def test_large_mismatch_prefers_replacement():
    prior = get_edit_prior("large_mismatch")
    assert ActionType.REPLACE_SUBGRAPH in prior.action_types


def test_no_compile_prefers_structural():
    prior = get_edit_prior("no_compile")
    assert ActionType.ADD_NODE in prior.action_types


def test_near_perfect_prefers_bind():
    prior = get_edit_prior("near_perfect")
    assert ActionType.BIND in prior.action_types


def test_all_priors_inspectable():
    table = get_all_priors()
    assert isinstance(table, dict)
    assert len(table) >= 5
    for key, prior in table.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(prior.rationale, str)
        assert len(prior.rationale) > 0


def test_no_task_id():
    import inspect
    src = inspect.getsource(get_edit_prior) + str(get_all_priors())
    assert "1b59e163" not in src
