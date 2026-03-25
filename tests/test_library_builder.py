"""Library builder and snapshot tests."""

from __future__ import annotations

import json

from aria.library.builder import build_library
from aria.program_store import ProgramStore
from aria.proposer.parser import parse_program
from aria.snapshot import write_snapshot
from aria.types import Type


def test_build_library_admits_reused_recipe(tmp_path):
    programs = [
        parse_program("""\
bind flipped: GRID = reflect_grid(HORIZONTAL, input)
bind tiled: GRID = stack_v(input, flipped)
bind result: GRID = transpose_grid(tiled)
yield result
"""),
        parse_program("""\
bind flipped: GRID = reflect_grid(HORIZONTAL, input)
bind tiled: GRID = stack_v(input, flipped)
bind result: GRID = tile_grid(tiled, 1, 2)
yield result
"""),
    ]

    library, report = build_library(programs, min_length=2, max_length=2, min_uses=2)

    assert report.corpus_programs == 2
    assert report.aggregated_candidates >= 1
    assert len(library.all_entries()) >= 1

    entry = library.all_entries()[0]
    assert entry.name.startswith("lib_reflect_grid_")
    assert entry.use_count == 2
    assert entry.params == (("arg0", Type.GRID),)


def test_write_snapshot_writes_manifest_and_files(tmp_path):
    store = ProgramStore()
    store.add_text("""\
bind result = reflect_grid(HORIZONTAL, input)
yield result
""", task_id="task-a", source="test")

    library, _report = build_library([], min_uses=2)
    manifest_path = write_snapshot(
        tmp_path / "snapshot",
        program_store=store,
        library=library,
        metadata={"benchmark": "arc"},
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 1
    assert manifest["program_count"] == 1
    assert manifest["library_entry_count"] == 0
    assert manifest["metadata"]["benchmark"] == "arc"
    assert (manifest_path.parent / "program_store.json").exists()
    assert (manifest_path.parent / "library.json").exists()
