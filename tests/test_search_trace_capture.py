"""Tests for trace schema and trace capture."""

from __future__ import annotations

import json

import numpy as np

from aria.search.trace_schema import SolveTrace
from aria.search.trace_capture import capture_solve_trace
from aria.search.sketch import SearchProgram, SearchStep, StepSelect


def test_solve_trace_round_trip():
    """SolveTrace should survive to_dict / from_dict."""
    trace = SolveTrace(
        task_id='abc12345',
        task_signatures=('dims:same', 'role:has_marker'),
        provenance='derive:uniform_recolor_const',
        step_actions=('recolor',),
        step_selectors=('largest',),
        program_dict={'steps': [{'action': 'recolor', 'params': {'color': 3}}]},
        n_demos=3,
        n_steps=1,
        test_correct=True,
    )

    d = trace.to_dict()
    restored = SolveTrace.from_dict(d)

    assert restored.task_id == trace.task_id
    assert restored.task_signatures == trace.task_signatures
    assert restored.provenance == trace.provenance
    assert restored.step_actions == trace.step_actions
    assert restored.step_selectors == trace.step_selectors
    assert restored.n_steps == 1
    assert restored.test_correct is True


def test_solve_trace_json_round_trip():
    """SolveTrace should survive JSON serialization."""
    trace = SolveTrace(
        task_id='test_task',
        task_signatures=('dims:same',),
        provenance='derive:rank_recolor',
        step_actions=('recolor', 'recolor', 'recolor'),
        step_selectors=('largest', 'rule', 'smallest'),
        program_dict={},
        n_demos=2,
        n_steps=3,
    )

    j = trace.to_json()
    restored = SolveTrace.from_json(j)

    assert restored.step_actions == ('recolor', 'recolor', 'recolor')
    assert restored.provenance == 'derive:rank_recolor'


def test_solve_trace_signatures():
    """signature() and selector_signature() should produce stable strings."""
    trace = SolveTrace(
        task_id='t',
        task_signatures=(),
        provenance='p',
        step_actions=('recolor', 'remove'),
        step_selectors=('largest', 'singleton'),
        program_dict={},
    )

    assert trace.signature() == 'recolor -> remove'
    assert trace.selector_signature() == 'recolor(largest) -> remove(singleton)'


def test_capture_solve_trace_from_program():
    """capture_solve_trace should extract structure from a SearchProgram."""
    prog = SearchProgram(
        steps=[
            SearchStep('recolor', {'color': 1}, StepSelect('largest')),
            SearchStep('remove', {}, StepSelect('singleton')),
        ],
        provenance='derive:conditional_dispatch',
    )

    trace = capture_solve_trace(
        task_id='fake_task',
        task_signatures=('dims:same', 'change:additive'),
        program=prog,
        n_demos=3,
        test_correct=True,
    )

    assert trace.task_id == 'fake_task'
    assert trace.provenance == 'derive:conditional_dispatch'
    assert trace.step_actions == ('recolor', 'remove')
    assert trace.step_selectors == ('largest', 'singleton')
    assert trace.n_steps == 2
    assert trace.n_demos == 3
    assert trace.test_correct is True


def test_capture_with_rule_selector():
    """Capture should summarize rule-based selectors."""
    prog = SearchProgram(
        steps=[
            SearchStep('recolor', {'color': 2}, StepSelect('by_rule', {
                'rule': {'kind': 'dnf', 'clauses': [
                    {'atoms': [
                        {'field': 'is_rectangular', 'value': True},
                        {'field': 'interior', 'value': True},
                    ]}
                ]}
            })),
        ],
        provenance='derive:rank_recolor',
    )

    trace = capture_solve_trace(
        task_id='rule_task',
        task_signatures=(),
        program=prog,
    )

    assert trace.step_selectors[0] == 'is_rectangular=T & interior=T'


def test_capture_real_task_08ed6ac7():
    """Smoke test: capture trace from a real solved task."""
    from aria.datasets import get_dataset, load_arc_task
    from aria.search.derive import derive_programs

    ds = get_dataset('v1-train')
    task = load_arc_task(ds, '08ed6ac7')
    demos = [(p.input, p.output) for p in task.train]

    progs = derive_programs(demos)
    assert progs, "08ed6ac7 should solve"

    prog = progs[0]
    trace = capture_solve_trace(
        task_id='08ed6ac7',
        task_signatures=('dims:same',),
        program=prog,
        n_demos=len(demos),
        test_correct=True,
    )

    assert trace.provenance == 'derive:rank_recolor'
    assert len(trace.step_actions) == 4  # 4 recolor steps
    assert all(a == 'recolor' for a in trace.step_actions)
    assert trace.program_dict  # non-empty
    assert trace.signature() == 'recolor -> recolor -> recolor -> recolor'
