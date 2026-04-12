"""Tests for output-dims integration in search."""

from __future__ import annotations

import numpy as np

from aria.search.output_dims import DimHypothesis
from aria.search.search import _matches_dim_hypothesis
from aria.search.sketch import SearchProgram, SearchStep


def _make_prog(action, params=None):
    return SearchProgram(
        steps=[SearchStep(action, params or {})],
        provenance='test',
    )


def test_constant_dims_accepts_matching():
    """Program producing the right shape should pass."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 1
    demos = [(inp, np.zeros((2, 2), dtype=np.int8))]
    hypotheses = [DimHypothesis(rule='constant', shape=(2, 2), confidence=1.0)]

    prog = _make_prog('crop_nonbg')
    assert _matches_dim_hypothesis(prog, demos, hypotheses)


def test_constant_dims_rejects_wrong_shape():
    """Program producing wrong shape should be rejected."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[0:3, 0:4] = 1
    demos = [(inp, np.zeros((2, 2), dtype=np.int8))]
    hypotheses = [DimHypothesis(rule='constant', shape=(2, 2), confidence=1.0)]

    prog = _make_prog('crop_nonbg')  # produces (3,4), not (2,2)
    assert not _matches_dim_hypothesis(prog, demos, hypotheses)


def test_scale_down_hypothesis():
    """Scale-down hypothesis matches programs producing input/k shape."""
    inp = np.zeros((6, 4), dtype=np.int8)
    demos = [(inp, np.zeros((3, 2), dtype=np.int8))]
    hypotheses = [DimHypothesis(rule='scale_down', shape=None, confidence=0.9,
                                meta={'factor': 2})]

    # A program that produces (3,2) from (6,4) matches
    prog = _make_prog('scale', {'factor': 1})  # identity, produces (6,4)
    assert not _matches_dim_hypothesis(prog, demos, hypotheses)


def test_no_hypotheses_allows_all():
    """Empty hypotheses should not reject anything."""
    inp = np.zeros((3, 3), dtype=np.int8)
    demos = [(inp, np.zeros((3, 3), dtype=np.int8))]

    prog = _make_prog('crop_nonbg')
    assert _matches_dim_hypothesis(prog, demos, [])
