"""Tests for fragment mining."""

from __future__ import annotations

from aria.core.fragment_mine import (
    FragmentCatalog, MinedFragment, mine_fragments, format_catalog,
)
from aria.types import Bind, Call, Literal, Program, Ref, Type


def _make_prog(*ops):
    steps = []
    for i, op in enumerate(ops):
        ref = Ref("input") if i == 0 else Ref(f"v{i-1}")
        steps.append(Bind(f"v{i}", Type.GRID, Call(op, (ref,))))
    return Program(steps=tuple(steps), output=f"v{len(ops)-1}")


def test_mine_extracts_singles():
    progs = [("t1", _make_prog("rotate_grid")), ("t2", _make_prog("rotate_grid"))]
    catalog = mine_fragments(progs)
    assert any(f.ops == ("rotate_grid",) and f.count == 2 for f in catalog.singles)


def test_mine_extracts_pairs():
    progs = [("t1", _make_prog("find_objects", "paint_objects"))]
    catalog = mine_fragments(progs)
    assert any(f.ops == ("find_objects", "paint_objects") for f in catalog.pairs)


def test_mine_deduplicates():
    progs = [("t1", _make_prog("rotate_grid")), ("t2", _make_prog("rotate_grid"))]
    catalog = mine_fragments(progs)
    rotate = [f for f in catalog.singles if f.ops == ("rotate_grid",)]
    assert len(rotate) == 1
    assert rotate[0].count == 2


def test_format_catalog():
    progs = [("t1", _make_prog("rotate_grid")), ("t2", _make_prog("reflect_grid"))]
    catalog = mine_fragments(progs)
    text = format_catalog(catalog)
    assert isinstance(text, str)
    assert "Mined Fragment Catalog" in text


def test_no_task_id():
    import inspect
    src = inspect.getsource(mine_fragments)
    assert "1b59e163" not in src
