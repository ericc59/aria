"""Tests for macro mining from solved traces."""

from __future__ import annotations

import json
import tempfile

from aria.search.trace_schema import SolveTrace
from aria.search.macro_miner import mine_macros, load_traces_jsonl, _action_key
from aria.search.macros import MacroLibrary


def _make_trace(task_id, provenance, actions, selectors=None, test_correct=True):
    """Helper: build a SolveTrace with given structure."""
    if selectors is None:
        selectors = tuple('' for _ in actions)
    return SolveTrace(
        task_id=task_id,
        task_signatures=('dims:same',),
        provenance=provenance,
        step_actions=tuple(actions),
        step_selectors=tuple(selectors),
        program_dict={'steps': [{'action': a} for a in actions]},
        n_demos=3,
        n_steps=len(actions),
        test_correct=test_correct,
    )


def test_action_key():
    """_action_key should combine provenance + action signature."""
    t = _make_trace('t1', 'derive:rank_recolor', ['recolor', 'recolor'])
    key = _action_key(t)
    assert 'derive:rank_recolor' in key
    assert 'recolor -> recolor' in key


def test_mine_groups_by_provenance_and_actions():
    """Traces with same provenance+actions should group together."""
    traces = [
        _make_trace('t1', 'derive:uniform_recolor_const', ['recolor']),
        _make_trace('t2', 'derive:uniform_recolor_const', ['recolor']),
        _make_trace('t3', 'derive:uniform_recolor_const', ['recolor']),
        _make_trace('t4', 'derive:color_map', ['recolor_map']),
    ]

    lib = mine_macros(traces, min_frequency=2, min_steps=1)

    # The 3-trace group should produce a macro; the singleton should not
    assert len(lib.macros) == 1
    assert lib.macros[0].frequency == 3
    assert lib.macros[0].source_task_count == 3
    assert 'recolor' in lib.macros[0].action_signature


def test_mine_filters_by_frequency():
    """Groups below min_frequency should be excluded."""
    traces = [
        _make_trace('t1', 'derive:foo', ['recolor']),
        _make_trace('t2', 'derive:bar', ['move']),
    ]

    lib = mine_macros(traces, min_frequency=2)
    assert len(lib.macros) == 0


def test_mine_filters_by_steps():
    """Groups with fewer steps than min_steps should be excluded."""
    traces = [
        _make_trace('t1', 'derive:x', ['recolor']),
        _make_trace('t2', 'derive:x', ['recolor']),
    ]

    lib = mine_macros(traces, min_frequency=2, min_steps=2)
    assert len(lib.macros) == 0

    lib2 = mine_macros(traces, min_frequency=2, min_steps=1)
    assert len(lib2.macros) == 1


def test_mine_test_correct_filter():
    """require_test_correct should exclude test-failed traces."""
    traces = [
        _make_trace('t1', 'derive:x', ['recolor'], test_correct=True),
        _make_trace('t2', 'derive:x', ['recolor'], test_correct=True),
        _make_trace('t3', 'derive:x', ['recolor'], test_correct=False),
    ]

    lib_all = mine_macros(traces, min_frequency=2, require_test_correct=False)
    assert lib_all.macros[0].frequency == 3

    lib_strict = mine_macros(traces, min_frequency=2, require_test_correct=True)
    assert lib_strict.macros[0].frequency == 2


def test_mine_multi_step_macro():
    """Multi-step compositions should produce macros with correct fields."""
    traces = [
        _make_trace('t1', 'derive:rank_recolor',
                    ['recolor', 'recolor', 'recolor'],
                    ['largest', 'rule', 'smallest']),
        _make_trace('t2', 'derive:rank_recolor',
                    ['recolor', 'recolor', 'recolor'],
                    ['largest', 'rule', 'smallest']),
    ]

    lib = mine_macros(traces, min_frequency=2)
    assert len(lib.macros) == 1
    m = lib.macros[0]
    assert m.action_signature == 'recolor -> recolor -> recolor'
    assert m.source_task_count == 2
    assert 'rank_recolor' in m.name
    assert m.program_template  # not empty


def test_mine_distinct_action_sequences():
    """Different action sequences should produce separate macros."""
    traces = [
        _make_trace('t1', 'derive:dispatch', ['recolor', 'remove']),
        _make_trace('t2', 'derive:dispatch', ['recolor', 'remove']),
        _make_trace('t3', 'derive:dispatch', ['move', 'remove']),
        _make_trace('t4', 'derive:dispatch', ['move', 'remove']),
    ]

    lib = mine_macros(traces, min_frequency=2)
    assert len(lib.macros) == 2
    sigs = {m.action_signature for m in lib.macros}
    assert 'recolor -> remove' in sigs
    assert 'move -> remove' in sigs


def test_mine_sorted_by_frequency():
    """Macros should be sorted by frequency descending."""
    traces = [
        _make_trace(f't{i}', 'derive:a', ['recolor'])
        for i in range(5)
    ] + [
        _make_trace(f's{i}', 'derive:b', ['move'])
        for i in range(3)
    ]

    lib = mine_macros(traces, min_frequency=2)
    assert len(lib.macros) == 2
    assert lib.macros[0].frequency >= lib.macros[1].frequency


def test_load_traces_jsonl_round_trip():
    """Traces saved as JSONL should load back correctly."""
    traces = [
        _make_trace('t1', 'derive:x', ['recolor', 'move']),
        _make_trace('t2', 'derive:y', ['remove']),
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for t in traces:
            f.write(t.to_json() + '\n')
        path = f.name

    loaded = load_traces_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].task_id == 't1'
    assert loaded[0].step_actions == ('recolor', 'move')
    assert loaded[1].provenance == 'derive:y'


def test_macro_library_end_to_end():
    """Mine → save → load should preserve macros."""
    traces = [
        _make_trace(f't{i}', 'derive:recolor_const', ['recolor'])
        for i in range(4)
    ]

    lib = mine_macros(traces, min_frequency=2)
    assert len(lib.macros) == 1

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    lib.save_json(path)
    loaded = MacroLibrary.load_json(path)
    assert len(loaded.macros) == 1
    assert loaded.macros[0].frequency == 4
    assert loaded.macros[0].name == lib.macros[0].name
