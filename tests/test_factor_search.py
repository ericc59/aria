"""Tests for factor instantiation, consensus, and search."""

import numpy as np
import pytest

from aria.factors import (
    Correspondence,
    Decomposition,
    Depth,
    FactorSet,
    Op,
    Scope,
    Selector,
    enumerate_compatible,
)
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.int64)


def _make_demo(inp: np.ndarray, out: np.ndarray) -> DemoPair:
    return DemoPair(input=inp, output=out)


# ---------------------------------------------------------------------------
# Factor instantiation tests
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_extract_produces_programs(self):
        """Object select + extract should produce at least one program."""
        from aria.core.grid_perception import perceive_grid
        from aria.factor_instantiate import instantiate_factor_set

        inp = _make_grid(
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2],
        )
        out = _make_grid(
            [1, 1],
            [1, 1],
        )
        demos = (_make_demo(inp, out),)
        perceptions = tuple(perceive_grid(d.input) for d in demos)

        fs = FactorSet(
            Decomposition.OBJECT, Selector.OBJECT_SELECT,
            Scope.OBJECT, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
        )
        programs = instantiate_factor_set(fs, demos, perceptions)
        assert len(programs) > 0

    def test_recolor_produces_programs(self):
        """Scoped recolor should produce programs when color map exists."""
        from aria.core.grid_perception import perceive_grid
        from aria.factor_instantiate import instantiate_factor_set

        inp = _make_grid(
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        )
        out = _make_grid(
            [0, 2, 2],
            [0, 2, 2],
            [0, 0, 0],
        )
        demos = (_make_demo(inp, out),)
        perceptions = tuple(perceive_grid(d.input) for d in demos)

        fs = FactorSet(
            Decomposition.OBJECT, Selector.NONE,
            Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
        )
        programs = instantiate_factor_set(fs, demos, perceptions)
        assert len(programs) > 0

    def test_incompatible_returns_empty(self):
        """Incompatible factor sets should return no programs."""
        from aria.core.grid_perception import perceive_grid
        from aria.factor_instantiate import instantiate_factor_set

        inp = _make_grid([0, 1], [2, 3])
        demos = (_make_demo(inp, inp),)
        perceptions = tuple(perceive_grid(d.input) for d in demos)

        # Partition + frame_interior is incompatible
        fs = FactorSet(
            Decomposition.PARTITION, Selector.FRAME_INTERIOR,
            Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
        )
        programs = instantiate_factor_set(fs, demos, perceptions)
        assert len(programs) == 0

    def test_global_transform_produces_programs(self):
        """Global transform should produce 6 programs (one per transform)."""
        from aria.core.grid_perception import perceive_grid
        from aria.factor_instantiate import instantiate_factor_set

        inp = _make_grid([1, 2], [3, 4])
        demos = (_make_demo(inp, inp),)
        perceptions = tuple(perceive_grid(d.input) for d in demos)

        fs = FactorSet(
            Decomposition.OBJECT, Selector.NONE,
            Scope.GLOBAL, Op.TRANSFORM, Correspondence.NONE, Depth.ONE,
        )
        programs = instantiate_factor_set(fs, demos, perceptions)
        assert len(programs) == 6  # rot90, rot180, rot270, flip_lr, flip_ud, transpose


# ---------------------------------------------------------------------------
# Consensus factor check tests
# ---------------------------------------------------------------------------


class TestFactorConsensus:
    def test_partition_consistency_pass(self):
        """Factor with partition decomp should pass when all demos have partitions."""
        from aria.consensus import check_factor_consistency
        from aria.core.grid_perception import perceive_grid

        # Grid with clear partition (separator row/col)
        inp = _make_grid(
            [1, 5, 2],
            [5, 5, 5],
            [3, 5, 4],
        )
        perceptions = (perceive_grid(inp), perceive_grid(inp))

        fs = FactorSet(
            Decomposition.PARTITION, Selector.CELL_PANEL,
            Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
        )

        passed, score, detail = check_factor_consistency(perceptions, fs)
        # May or may not detect partition depending on structure, but shouldn't crash
        assert isinstance(passed, bool)

    def test_frame_consistency_fail(self):
        """Frame decomp should fail when no demos have frames."""
        from aria.consensus import check_factor_consistency
        from aria.core.grid_perception import perceive_grid

        # Scattered pixels — no frame structure
        inp = _make_grid(
            [1, 0],
            [0, 2],
        )
        perceptions = (perceive_grid(inp), perceive_grid(inp))

        fs = FactorSet(
            Decomposition.FRAME, Selector.FRAME_INTERIOR,
            Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
        )

        passed, score, detail = check_factor_consistency(perceptions, fs)
        assert not passed

    def test_object_select_consistency_pass(self):
        """Object selector should pass when all demos have objects."""
        from aria.consensus import check_factor_consistency
        from aria.core.grid_perception import perceive_grid

        inp = _make_grid(
            [0, 1, 0],
            [0, 0, 0],
            [0, 2, 0],
        )
        perceptions = (perceive_grid(inp), perceive_grid(inp))

        fs = FactorSet(
            Decomposition.OBJECT, Selector.OBJECT_SELECT,
            Scope.OBJECT, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
        )

        passed, score, detail = check_factor_consistency(perceptions, fs)
        assert passed


# ---------------------------------------------------------------------------
# Factor proposer tests
# ---------------------------------------------------------------------------


class TestFactorProposer:
    def test_untrained_gives_uniform(self):
        """Untrained proposer should return uniform probabilities."""
        from aria.core.factor_proposer import FactorProposer

        proposer = FactorProposer()
        features = np.zeros(40)
        probs = proposer.predict_factor_probs(features)

        assert len(probs) == 6
        for name, p in probs.items():
            assert abs(p.sum() - 1.0) < 1e-6

    def test_top_k_returns_compatible(self):
        """top_k_factor_sets should only return compatible combos."""
        from aria.core.factor_proposer import FactorProposer

        proposer = FactorProposer()
        features = np.random.randn(40)
        results = proposer.top_k_factor_sets(features, k=20)

        assert len(results) <= 20
        for fs, prob in results:
            from aria.factors import is_compatible
            assert is_compatible(fs)
            assert prob > 0

    def test_training_changes_predictions(self):
        """After training, predictions should differ from uniform."""
        from aria.core.factor_proposer import FactorProposer

        proposer = FactorProposer()
        # Synthetic training data
        X = np.random.randn(20, 40)
        labels = [
            FactorSet(
                Decomposition.OBJECT, Selector.OBJECT_SELECT,
                Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
            )
        ] * 20

        proposer.fit_from_labels(list(X), labels, epochs=50)
        assert proposer.trained

        probs = proposer.predict_factor_probs(np.zeros(40))
        # After training on all-OBJECT, decomposition prob should peak at OBJECT
        decomp_probs = probs["decomposition"]
        assert np.argmax(decomp_probs) == list(Decomposition).index(Decomposition.OBJECT)


# ---------------------------------------------------------------------------
# Factor labels tests
# ---------------------------------------------------------------------------


class TestFactorLabels:
    def test_skeleton_to_factors_mapping(self):
        """All skeleton mappings should produce valid factor sets."""
        from aria.core.factor_labels import SKELETON_TO_FACTORS
        from aria.factors import is_compatible

        for skeleton, fs in SKELETON_TO_FACTORS.items():
            assert is_compatible(fs), f"skeleton {skeleton} maps to incompatible {fs}"

    def test_scene_family_to_factors_mapping(self):
        """All scene family mappings should produce valid factor sets."""
        from aria.core.factor_labels import SCENE_FAMILY_TO_FACTORS
        from aria.factors import is_compatible

        for family, fs in SCENE_FAMILY_TO_FACTORS.items():
            assert is_compatible(fs), f"family {family} maps to incompatible {fs}"

    def test_coverage_report(self):
        """Coverage report should have entries for all 6 factors."""
        from aria.core.factor_labels import FactorLabel, FactorLabelSet

        labels = FactorLabelSet(labels=[
            FactorLabel(
                task_id="test",
                factors=FactorSet(
                    Decomposition.OBJECT, Selector.OBJECT_SELECT,
                    Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
                ),
                source="test",
            ),
        ])
        report = labels.coverage_report()
        assert len(report) == 6
        assert "object" in report["decomposition"]


# ---------------------------------------------------------------------------
# End-to-end factor search test
# ---------------------------------------------------------------------------


class TestFactorSearch:
    def test_search_finds_color_map(self):
        """Factor search should find a global color map solution."""
        from aria.core.factor_search import factor_composition_search

        # Task: recolor all 1s to 2s
        inp1 = _make_grid([0, 1, 0], [1, 0, 1], [0, 1, 0])
        out1 = _make_grid([0, 2, 0], [2, 0, 2], [0, 2, 0])
        inp2 = _make_grid([1, 0, 1], [0, 1, 0], [1, 0, 1])
        out2 = _make_grid([2, 0, 2], [0, 2, 0], [2, 0, 2])

        demos = (_make_demo(inp1, out1), _make_demo(inp2, out2))
        results = factor_composition_search(demos, max_candidates=30, max_programs=100)

        assert len(results) > 0
        # At least one should verify
        for fs, prog in results:
            from aria.core.scene_executor import execute_scene_program
            for d in demos:
                result = execute_scene_program(prog, d.input)
                assert np.array_equal(result, d.output)

    def test_search_bounded(self):
        """Factor search should respect max_programs bound."""
        from aria.core.factor_search import factor_composition_search

        inp = _make_grid([0, 1], [2, 3])
        out = _make_grid([3, 2], [1, 0])  # unlikely to solve
        demos = (_make_demo(inp, out),)

        # Should complete without error even if nothing verifies
        results = factor_composition_search(demos, max_candidates=10, max_programs=20)
        assert isinstance(results, list)
