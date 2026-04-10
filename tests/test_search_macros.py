"""Tests for macro schema."""

from __future__ import annotations

import json
import tempfile

from aria.search.macros import Macro, MacroLibrary


def test_macro_round_trip():
    """Macro should survive to_dict / from_dict."""
    macro = Macro(
        name='rank_recolor_4',
        description='Recolor 4 objects by size rank',
        program_template={'steps': [
            {'action': 'recolor', 'params': {'color': 1}},
            {'action': 'recolor', 'params': {'color': 2}},
            {'action': 'recolor', 'params': {'color': 3}},
            {'action': 'recolor', 'params': {'color': 4}},
        ]},
        source_provenances=['derive:rank_recolor'],
        source_task_count=3,
        frequency=3,
        solve_rate=1.0,
        action_signature='recolor -> recolor -> recolor -> recolor',
        selector_pattern='largest -> rule -> rule -> smallest',
    )

    d = macro.to_dict()
    restored = Macro.from_dict(d)

    assert restored.name == macro.name
    assert restored.frequency == 3
    assert restored.action_signature == macro.action_signature
    assert restored.source_task_count == 3


def test_macro_json_round_trip():
    """Macro should survive JSON round-trip."""
    macro = Macro(name='test_macro', frequency=5)
    j = macro.to_json()
    restored = Macro.from_json(j)
    assert restored.name == 'test_macro'
    assert restored.frequency == 5


def test_macro_library_find():
    """MacroLibrary.find_by_signature should filter correctly."""
    lib = MacroLibrary()
    lib.add(Macro(name='a', action_signature='recolor'))
    lib.add(Macro(name='b', action_signature='recolor -> remove'))
    lib.add(Macro(name='c', action_signature='recolor'))

    found = lib.find_by_signature('recolor')
    assert len(found) == 2
    assert {m.name for m in found} == {'a', 'c'}


def test_macro_library_save_load():
    """MacroLibrary should persist to/from JSON."""
    lib = MacroLibrary()
    lib.add(Macro(name='m1', frequency=10, action_signature='move'))
    lib.add(Macro(name='m2', frequency=5, action_signature='recolor'))

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    lib.save_json(path)
    loaded = MacroLibrary.load_json(path)

    assert len(loaded.macros) == 2
    assert loaded.macros[0].name == 'm1'
    assert loaded.macros[1].frequency == 5
