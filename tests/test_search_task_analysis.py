"""Tests for task analysis module."""

from __future__ import annotations

import numpy as np

from aria.search.task_analysis import analyze_task


def test_recolor_only():
    inp = np.array([[1, 0], [0, 2]], dtype=np.int8)
    out = np.array([[3, 0], [0, 4]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.diff_type == 'recolor_only'
    assert a.same_dims
    assert not a.dims_change


def test_additive():
    inp = np.array([[0, 0], [0, 0]], dtype=np.int8)
    out = np.array([[1, 0], [0, 0]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.diff_type == 'additive'


def test_subtractive():
    inp = np.array([[1, 2], [3, 0]], dtype=np.int8)
    out = np.array([[1, 0], [0, 0]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.diff_type == 'subtractive'


def test_rearrange():
    inp = np.array([[1, 2], [3, 0]], dtype=np.int8)
    out = np.array([[3, 1], [0, 2]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.diff_type == 'rearrange'


def test_dims_change():
    inp = np.array([[1, 2], [3, 4]], dtype=np.int8)
    out = np.array([[1, 2, 3, 4]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.dims_change
    assert not a.same_dims


def test_extraction():
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 7
    out = np.array([[7, 7], [7, 7]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.is_extraction


def test_construction():
    inp = np.array([[1, 1], [1, 1]], dtype=np.int8)
    out = np.array([[5, 5], [5, 5]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert a.is_construction


def test_new_removed_colors():
    inp = np.array([[1, 2]], dtype=np.int8)
    out = np.array([[1, 3]], dtype=np.int8)
    a = analyze_task([(inp, out)])
    assert 3 in a.new_colors
    assert 2 in a.removed_colors
