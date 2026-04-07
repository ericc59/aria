"""Tests for seed collection."""

from __future__ import annotations

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier
from aria.core.graph import ComputationGraph
from aria.core.library import GraphLibrary
from aria.core.seeds import Seed, collect_seeds
from aria.types import DemoPair, grid_from_list


def _tile_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([
                [1, 2, 1, 2], [3, 4, 3, 4],
                [1, 2, 1, 2], [3, 4, 3, 4],
            ]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([
                [5, 6, 5, 6], [7, 8, 7, 8],
                [5, 6, 5, 6], [7, 8, 7, 8],
            ]),
        ),
    )


def _rotate_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2], [3, 4]]),
            output=grid_from_list([[4, 3], [2, 1]]),
        ),
        DemoPair(
            input=grid_from_list([[5, 6], [7, 8]]),
            output=grid_from_list([[8, 7], [6, 5]]),
        ),
    )


def _impossible_task():
    return (
        DemoPair(
            input=grid_from_list([[1, 2, 3]]),
            output=grid_from_list([[9, 8, 7]]),
        ),
    )


def test_collect_seeds_from_fitters():
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=False,
    )
    assert len(seeds) >= 1
    assert all(isinstance(s, Seed) for s in seeds)
    assert all(isinstance(s.graph, ComputationGraph) for s in seeds)
    # Rotate should be verified by fitter
    fitter_seeds = [s for s in seeds if s.provenance == "fitter"]
    assert any(s.already_verified for s in fitter_seeds)


def test_collect_seeds_includes_templates():
    demos = _impossible_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
        include_templates=True,
    )
    template_seeds = [s for s in seeds if s.provenance == "template"]
    assert len(template_seeds) >= 1
    # Templates should not be verified
    assert all(not s.already_verified for s in template_seeds)


def test_collect_seeds_from_library():
    # First solve a task to populate library
    from aria.core.learn import learn_and_propose
    lib = GraphLibrary()
    learn_and_propose(
        [("rotate", _rotate_task())],
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        library=lib,
    )
    assert lib.size >= 1

    # Now collect seeds for a different task using the library
    demos = _tile_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="tile",
        library=lib,
    )
    library_seeds = [s for s in seeds if s.provenance == "library"]
    assert len(library_seeds) >= 1


def test_collect_seeds_deduplicates():
    demos = _rotate_task()
    seeds = collect_seeds(
        examples=demos,
        fitter=ARCFitter(),
        specializer=ARCSpecializer(),
        compiler=ARCCompiler(),
        verifier=ARCVerifier(),
        task_id="test",
    )
    # No duplicate structure keys
    keys = set()
    for s in seeds:
        from aria.core.seeds import _structure_key
        k = _structure_key(s.graph)
        assert k not in keys, f"duplicate seed: {k}"
        keys.add(k)
