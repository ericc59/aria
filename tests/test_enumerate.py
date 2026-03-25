"""Tests for the program enumerator."""

from __future__ import annotations

import numpy as np
import pytest

from aria.types import (
    Bind,
    Call,
    DemoPair,
    Literal,
    Program,
    Ref,
    Task,
    Type,
    grid_eq,
    grid_from_list,
    make_grid,
)
from aria.enumerate import (
    TypeEnv,
    _can_fill_params,
    _enumerable_ops,
    _verify_candidate,
    enumerate_programs,
    generate_synthetic_task,
    build_corpus,
)
from aria.runtime.ops import OpSignature, all_ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_task() -> Task:
    """Task where output == input."""
    g1 = grid_from_list([[1, 2], [3, 4]])
    g2 = grid_from_list([[5, 0], [0, 5]])
    return Task(
        train=(DemoPair(input=g1, output=g1), DemoPair(input=g2, output=g2)),
        test=(DemoPair(input=g1, output=g1),),
    )


def _recolor_task(src: int, dst: int) -> Task:
    """Task where all cells of color `src` become `dst`, everything else stays."""
    def transform(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == src] = dst
        return result

    g1 = grid_from_list([
        [0, src, 0],
        [src, 0, src],
        [0, src, 0],
    ])
    g2 = grid_from_list([
        [src, 0, 0],
        [0, 0, 0],
        [0, 0, src],
    ])
    g3 = grid_from_list([
        [src, src, src],
        [0, 0, 0],
        [0, 0, 0],
    ])
    return Task(
        train=(
            DemoPair(input=g1, output=transform(g1)),
            DemoPair(input=g2, output=transform(g2)),
            DemoPair(input=g3, output=transform(g3)),
        ),
        test=(DemoPair(input=g1, output=transform(g1)),),
    )


def _color_swap_task(c1: int, c2: int) -> Task:
    """Task where colors c1 and c2 are swapped."""
    def transform(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == c1] = c2
        result[grid == c2] = c1
        return result

    g1 = grid_from_list([
        [c1, c2, 0],
        [0, c1, c2],
    ])
    g2 = grid_from_list([
        [c2, 0, c1],
        [c1, c2, 0],
    ])
    return Task(
        train=(
            DemoPair(input=g1, output=transform(g1)),
            DemoPair(input=g2, output=transform(g2)),
        ),
        test=(DemoPair(input=g1, output=transform(g1)),),
    )


# ---------------------------------------------------------------------------
# TypeEnv tests
# ---------------------------------------------------------------------------

class TestTypeEnv:
    def test_initial_has_input_grid(self):
        env = TypeEnv()
        refs = env.refs_for_type(Type.GRID)
        assert len(refs) == 1
        assert refs[0].name == "input"

    def test_bind_and_lookup(self):
        env = TypeEnv()
        env.bind("objs", Type.OBJECT_SET)
        refs = env.refs_for_type(Type.OBJECT_SET)
        names = {r.name for r in refs}
        assert "objs" in names

    def test_has_type(self):
        env = TypeEnv()
        assert env.has_type(Type.GRID)
        assert not env.has_type(Type.OBJECT_SET)
        env.bind("x", Type.OBJECT_SET)
        assert env.has_type(Type.OBJECT_SET)

    def test_args_includes_literals(self):
        env = TypeEnv()
        args = env.args_for_type(Type.COLOR)
        # Should have literals 0-9 (no refs of type COLOR initially)
        assert len(args) >= 10
        literal_vals = {a.value for a in args if isinstance(a, Literal)}
        assert 0 in literal_vals
        assert 9 in literal_vals

    def test_copy_independence(self):
        env = TypeEnv()
        env2 = env.copy()
        env2.bind("extra", Type.INT)
        assert not env.has_type(Type.INT)
        assert env2.has_type(Type.INT)


# ---------------------------------------------------------------------------
# Type-guided filtering tests
# ---------------------------------------------------------------------------

class TestTypeGuidedSearch:
    def test_enumerable_ops_excludes_context(self):
        ops = _enumerable_ops()
        assert "demo_count" not in ops
        assert "infer_step" not in ops

    def test_enumerable_ops_excludes_higher_order(self):
        ops = _enumerable_ops()
        assert "compose" not in ops
        assert "map_obj" not in ops

    def test_enumerable_ops_includes_core(self):
        ops = _enumerable_ops()
        assert "find_objects" in ops
        assert "apply_color_map" in ops
        assert "recolor" in ops

    def test_can_fill_params_with_grid_only(self):
        env = TypeEnv()  # only has GRID
        ops = _enumerable_ops()
        # find_objects takes a GRID, should be fillable
        assert _can_fill_params(ops["find_objects"], env)
        # where takes (PREDICATE, OBJECT_SET), not fillable from just GRID
        assert not _can_fill_params(ops["where"], env)

    def test_can_fill_params_with_literals(self):
        env = TypeEnv()
        ops = _enumerable_ops()
        # new_grid takes (DIMS, COLOR), both have literals
        assert _can_fill_params(ops["new_grid"], env)


# ---------------------------------------------------------------------------
# Identity task test
# ---------------------------------------------------------------------------

class TestEnumerateIdentity:
    def test_finds_identity_program(self):
        task = _identity_task()
        programs = enumerate_programs(task, max_steps=2, max_candidates=100,
                                      timeout_sec=5)
        assert len(programs) >= 1
        # The identity program should output "input"
        identity_found = any(p.output == "input" and len(p.steps) == 0
                             for p in programs)
        assert identity_found, (
            f"Expected identity program, got: "
            f"{[(p.output, len(p.steps)) for p in programs]}"
        )


# ---------------------------------------------------------------------------
# Recolor task tests
# ---------------------------------------------------------------------------

class TestEnumerateRecolor:
    def test_finds_recolor_program(self):
        """Test that enumerator finds a single-color remapping program."""
        task = _recolor_task(src=3, dst=7)
        programs = enumerate_programs(task, max_steps=4, max_candidates=5000,
                                      timeout_sec=10)
        assert len(programs) >= 1, "Should find at least one recolor program"

        # Verify all found programs actually work
        for prog in programs:
            for demo in task.train:
                from aria.runtime.executor import execute
                result = execute(prog, demo.input)
                assert grid_eq(result, demo.output)

    def test_finds_color_swap(self):
        """Test two-color swap: c1 <-> c2."""
        task = _color_swap_task(c1=2, c2=5)
        programs = enumerate_programs(task, max_steps=4, max_candidates=5000,
                                      timeout_sec=10)
        assert len(programs) >= 1, "Should find at least one color swap program"


# ---------------------------------------------------------------------------
# Verify candidate
# ---------------------------------------------------------------------------

class TestVerifyCandidate:
    def test_correct_program_verifies(self):
        task = _identity_task()
        prog = Program(steps=(), output="input")
        assert _verify_candidate(prog, task.train)

    def test_wrong_program_fails(self):
        task = _recolor_task(src=1, dst=2)
        # Identity program should fail on a recolor task
        prog = Program(steps=(), output="input")
        assert not _verify_candidate(prog, task.train)

    def test_crashing_program_fails(self):
        task = _identity_task()
        # Program that references a nonexistent variable
        prog = Program(
            steps=(
                Bind(name="x", typ=Type.GRID,
                     expr=Ref(name="nonexistent")),
            ),
            output="x",
        )
        assert not _verify_candidate(prog, task.train)


# ---------------------------------------------------------------------------
# Synthetic task generation
# ---------------------------------------------------------------------------

class TestSyntheticTaskGeneration:
    def test_generates_valid_task(self):
        import random
        rng = random.Random(42)
        result = generate_synthetic_task(min_steps=2, max_steps=5, rng=rng)
        # May return None if unlucky, but with a fixed seed it should work
        # Try multiple seeds
        for seed in range(42, 60):
            rng = random.Random(seed)
            result = generate_synthetic_task(min_steps=2, max_steps=5, rng=rng)
            if result is not None:
                break

        if result is None:
            pytest.skip("Synthetic generation did not produce a task "
                        "(stochastic — re-run)")
            return

        task, program = result
        assert len(task.train) == 3
        assert len(task.test) == 1

        # The program should verify against its own demos
        assert _verify_candidate(program, task.train)

    def test_synthetic_task_nontrivial(self):
        """Output should differ from input."""
        import random
        for seed in range(100, 120):
            rng = random.Random(seed)
            result = generate_synthetic_task(min_steps=2, max_steps=5, rng=rng)
            if result is not None:
                task, _ = result
                for demo in task.train:
                    assert not grid_eq(demo.input, demo.output), (
                        "Synthetic task output should differ from input"
                    )
                return
        pytest.skip("Could not generate non-trivial synthetic task")


# ---------------------------------------------------------------------------
# Build corpus
# ---------------------------------------------------------------------------

class TestBuildCorpus:
    def test_build_corpus_basic(self):
        tasks = [_identity_task(), _recolor_task(1, 2)]
        result = build_corpus(
            tasks,
            max_steps=4,
            timeout_per_task=5,
            max_candidates_per_task=3000,
        )
        assert "programs" in result
        assert "tasks_solved" in result
        assert "total_programs" in result
        assert "time_sec" in result
        assert "per_task" in result
        assert len(result["per_task"]) == 2
        # Identity task should definitely be solved
        assert result["tasks_solved"] >= 1

    def test_build_corpus_stats(self):
        tasks = [_identity_task()]
        result = build_corpus(tasks, max_steps=2, timeout_per_task=3,
                              max_candidates_per_task=100)
        assert result["tasks_solved"] == 1
        assert result["total_programs"] >= 1
        assert result["time_sec"] > 0
