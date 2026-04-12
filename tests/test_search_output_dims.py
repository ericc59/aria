"""Tests for output-dims pre-solver."""

from __future__ import annotations

import numpy as np

from aria.search.task_analysis import analyze_task
from aria.search.output_dims import solve_output_dims


def test_constant_output():
    inp1 = np.zeros((3, 3), dtype=np.int8)
    out1 = np.zeros((2, 2), dtype=np.int8)
    inp2 = np.zeros((4, 4), dtype=np.int8)
    out2 = np.zeros((2, 2), dtype=np.int8)
    demos = [(inp1, out1), (inp2, out2)]
    a = analyze_task(demos)
    hs = solve_output_dims(demos, a)
    assert any(h.rule == 'constant' and h.shape == (2, 2) for h in hs)


def test_scale_up():
    inp = np.zeros((2, 3), dtype=np.int8)
    out = np.zeros((4, 6), dtype=np.int8)
    demos = [(inp, out)]
    a = analyze_task(demos)
    hs = solve_output_dims(demos, a)
    assert any(h.rule == 'scale_up' and h.meta.get('factor') == 2 for h in hs)


def test_scale_down():
    inp = np.zeros((6, 4), dtype=np.int8)
    out = np.zeros((3, 2), dtype=np.int8)
    demos = [(inp, out)]
    a = analyze_task(demos)
    hs = solve_output_dims(demos, a)
    assert any(h.rule == 'scale_down' and h.meta.get('factor') == 2 for h in hs)


def test_object_bbox():
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:4] = 3  # 2x3 object
    out = np.zeros((2, 3), dtype=np.int8)
    out[:] = 3
    demos = [(inp, out)]
    a = analyze_task(demos)
    hs = solve_output_dims(demos, a)
    assert any(h.rule == 'object_bbox' for h in hs)
