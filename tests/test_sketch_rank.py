"""Tests for learned proposal ranking hooks."""

from __future__ import annotations

from aria.sketch import Sketch, SketchStep, PrimitiveFamily
from aria.sketch_rank import (
    DecompRankInput,
    RankingReport,
    SketchRankInput,
    export_decomp_rank_example,
    export_sketch_rank_example,
    heuristic_decomp_ranker,
    heuristic_sketch_ranker,
    rank_decompositions,
    rank_sketches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sketch(family: str, confidence: float = 1.0) -> Sketch:
    return Sketch(
        task_id="test",
        steps=(SketchStep(name="s1", primitive=PrimitiveFamily.FIND_OBJECTS),),
        output_ref="s1",
        confidence=confidence,
        metadata={"family": family},
    )


# ---------------------------------------------------------------------------
# Sketch ranking
# ---------------------------------------------------------------------------


def test_rank_sketches_no_ranker():
    """Without ranker, order is preserved."""
    sketches = [_make_sketch("a"), _make_sketch("b")]
    reordered, report = rank_sketches(sketches, (), {})
    assert reordered == sketches
    assert report.order_changed is False
    assert report.policy_name == "none"
    assert report.candidates_ranked == 2


def test_rank_sketches_with_ranker():
    """With a ranker that reverses, order should change."""
    sketches = [_make_sketch("a"), _make_sketch("b")]

    def reverse_ranker(inp, sks):
        return tuple(reversed(range(len(sks))))

    reordered, report = rank_sketches(sketches, (), {}, ranker=reverse_ranker)
    assert reordered[0].metadata["family"] == "b"
    assert reordered[1].metadata["family"] == "a"
    assert report.order_changed is True
    assert report.ranked_order == (1, 0)


def test_rank_sketches_empty():
    """Empty list returns empty."""
    reordered, report = rank_sketches([], (), {})
    assert reordered == []
    assert report.candidates_ranked == 0


def test_heuristic_sketch_ranker_prefers_composite_when_bg_rotates():
    """Composite alignment should rank first when colors rotate."""
    sketches = [_make_sketch("framed_periodic_repair"), _make_sketch("composite_role_alignment")]
    inp = SketchRankInput(
        task_signatures=("dims:same", "role:has_frame"),
        sketch_families=("framed_periodic_repair", "composite_role_alignment"),
        sketch_confidences=(1.0, 1.0),
        bg_rotates=True,
        same_dims=True,
        n_demos=2,
    )
    order = heuristic_sketch_ranker(inp, sketches)
    # Composite should come first when bg_rotates
    assert order[0] == 1  # composite_role_alignment


def test_heuristic_sketch_ranker_prefers_periodic_when_has_frame():
    """Periodic repair should rank first when has_frame and no rotation."""
    sketches = [_make_sketch("composite_role_alignment"), _make_sketch("framed_periodic_repair")]
    inp = SketchRankInput(
        task_signatures=("dims:same", "role:has_frame"),
        sketch_families=("composite_role_alignment", "framed_periodic_repair"),
        sketch_confidences=(1.0, 1.0),
        bg_rotates=False,
        same_dims=True,
        n_demos=2,
    )
    order = heuristic_sketch_ranker(inp, sketches)
    # Periodic should come first when has_frame and no rotation
    assert order[0] == 1  # framed_periodic_repair


def test_heuristic_sketch_ranker_confidence_tiebreak():
    """Higher confidence should win when family bonuses are equal."""
    sketches = [_make_sketch("unknown_a", 0.5), _make_sketch("unknown_b", 0.9)]
    inp = SketchRankInput(
        task_signatures=(),
        sketch_families=("unknown_a", "unknown_b"),
        sketch_confidences=(0.5, 0.9),
        bg_rotates=False,
        same_dims=True,
        n_demos=2,
    )
    order = heuristic_sketch_ranker(inp, sketches)
    assert order[0] == 1  # higher confidence


# ---------------------------------------------------------------------------
# Decomposition ranking
# ---------------------------------------------------------------------------


def test_rank_decomp_no_ranker():
    """Without ranker, order preserved."""
    views = ["objects", "framed_regions", "composites", "markers"]
    reordered, report = rank_decompositions(views, (), {})
    assert reordered == views
    assert report.order_changed is False


def test_rank_decomp_with_ranker():
    """With heuristic ranker, composites first when bg_rotates."""
    views = ["objects", "framed_regions", "composites", "markers"]
    reordered, report = rank_decompositions(
        views, ("dims:same",), {"bg_rotates": True},
        ranker=heuristic_decomp_ranker,
    )
    assert reordered[0] == "composites"
    assert report.order_changed is True


def test_rank_decomp_empty():
    reordered, report = rank_decompositions([], (), {})
    assert reordered == []
    assert report.candidates_ranked == 0


def test_heuristic_decomp_ranker_marker_priority():
    """Markers should rank high when has_marker signature present."""
    views = ["objects", "markers"]
    inp = DecompRankInput(
        task_signatures=("role:has_marker",),
        view_names=("objects", "markers"),
        same_dims=True,
        bg_rotates=False,
        n_demos=2,
    )
    order = heuristic_decomp_ranker(inp, views)
    assert order[0] == 1  # markers first


# ---------------------------------------------------------------------------
# Training data export
# ---------------------------------------------------------------------------


def test_export_sketch_rank_example():
    ex = export_sketch_rank_example(
        task_id="test_task",
        task_signatures=("dims:same", "role:has_marker"),
        sketch_families=("framed_periodic_repair", "composite_role_alignment"),
        winning_family="composite_role_alignment",
        verified_indices=(1,),
    )
    assert ex["task_type"] == "SKETCH_RANK"
    assert ex["task_id"] == "test_task"
    assert ex["label"]["winning_index"] == 1
    assert ex["label"]["winning_family"] == "composite_role_alignment"


def test_export_sketch_rank_no_winner():
    ex = export_sketch_rank_example(
        task_id="test",
        task_signatures=(),
        sketch_families=("a", "b"),
        winning_family=None,
        verified_indices=(),
    )
    assert ex["label"]["winning_index"] == -1


def test_export_decomp_rank_example():
    ex = export_decomp_rank_example(
        task_id="test_task",
        task_signatures=("dims:same",),
        view_names=("objects", "framed_regions", "composites"),
        useful_views=("composites",),
    )
    assert ex["task_type"] == "DECOMP_RANK"
    assert ex["label"]["useful_indices"] == [2]
    assert ex["label"]["useful_views"] == ["composites"]


# ---------------------------------------------------------------------------
# Ranking report structure
# ---------------------------------------------------------------------------


def test_ranking_report_fields():
    report = RankingReport(
        task="SKETCH_RANK",
        candidates_ranked=3,
        order_changed=True,
        original_order=(0, 1, 2),
        ranked_order=(2, 0, 1),
        policy_name="heuristic",
    )
    assert report.task == "SKETCH_RANK"
    assert report.candidates_ranked == 3
    assert report.order_changed is True


# ---------------------------------------------------------------------------
# Integration: ranking in refinement
# ---------------------------------------------------------------------------


def test_refinement_sketch_ranking_report():
    """Sketch refinement should carry ranking metadata."""
    from aria.refinement import _run_sketch_refinement
    from aria.types import DemoPair, grid_from_list

    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 8, 4, 8, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 8, 4, 8, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 1, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 3, 1, 3, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
        ),
    )

    # Without ranker
    result = _run_sketch_refinement(demos)
    assert result.ranking_policy == "none"

    # With heuristic ranker
    from aria.sketch_rank import heuristic_sketch_ranker
    result = _run_sketch_refinement(
        demos,
        sketch_ranker=heuristic_sketch_ranker,
        task_signatures=("dims:same",),
    )
    assert result.ranking_policy == "heuristic_sketch_ranker"
    assert result.sketches_proposed >= 1


def test_deterministic_fallback_without_ranker():
    """Without ranker, sketch order should be deterministic."""
    from aria.refinement import _run_sketch_refinement
    from aria.types import DemoPair, grid_from_list

    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )
    r1 = _run_sketch_refinement(demos)
    r2 = _run_sketch_refinement(demos)
    assert r1.sketch_families == r2.sketch_families


# ---------------------------------------------------------------------------
# Decomposition ranking in live sketch fitting
# ---------------------------------------------------------------------------


def _composite_demos():
    """Task with composites — decomp ranking should prefer composites view."""
    from aria.types import DemoPair, grid_from_list
    return (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 8, 4, 8, 0, 0],
                [0, 8, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 8, 4, 8, 0],
                [0, 0, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 1, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 3, 1, 3, 0],
                [0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
        ),
    )


def test_fit_sketches_with_report_no_ranker():
    """Without decomp ranker, default view order preserved."""
    from aria.sketch_fit import fit_sketches_with_report
    demos = _composite_demos()
    result = fit_sketches_with_report(demos, "test")
    # Default order: framed_regions first, composites second
    assert result.decomp_views_tried == ("framed_regions", "composites")
    assert result.decomp_ranking_applied is False
    assert result.decomp_ranking_policy == "none"


def test_fit_sketches_with_report_heuristic_ranker():
    """With heuristic decomp ranker, ranking is applied and reported."""
    from aria.sketch_fit import fit_sketches_with_report
    demos = _composite_demos()
    result = fit_sketches_with_report(
        demos, "test",
        decomp_ranker=heuristic_decomp_ranker,
        task_signatures=("dims:same",),
    )
    assert result.decomp_ranking_applied is True
    assert result.decomp_ranking_policy == "heuristic_decomp_ranker"
    # Views should be ordered (may or may not change depending on signatures)
    assert len(result.decomp_views_tried) == 2


def test_fit_sketches_heuristic_reorders_when_has_frame():
    """With has_frame signature, heuristic should put framed_regions first."""
    from aria.sketch_fit import fit_sketches_with_report
    demos = _composite_demos()
    result = fit_sketches_with_report(
        demos, "test",
        decomp_ranker=heuristic_decomp_ranker,
        task_signatures=("dims:same", "role:has_frame"),
    )
    # framed_regions should be first (has_frame bonus > no composites bonus when !bg_rotates)
    assert result.decomp_views_tried[0] == "framed_regions"


def test_fit_sketches_custom_ranker_reorders():
    """A custom ranker that reverses order should produce composites first."""
    from aria.sketch_fit import fit_sketches_with_report
    demos = _composite_demos()

    def reverse_ranker(inp, views):
        return tuple(reversed(range(len(views))))

    result = fit_sketches_with_report(
        demos, "test",
        decomp_ranker=reverse_ranker,
        task_signatures=("dims:same",),
    )
    # Custom reverse ranker: default [framed_regions, composites] → [composites, framed_regions]
    assert result.decomp_views_tried == ("composites", "framed_regions")
    assert result.decomp_ranking_changed_order is True


def test_decomp_ranking_changes_sketch_order():
    """Decomp ranking should change the order sketches appear in fit result."""
    from aria.sketch_fit import fit_sketches_with_report
    demos = _composite_demos()

    # Without ranker: default order
    r1 = fit_sketches_with_report(demos, "test")

    # With ranker: composites view first → composite sketch appears first
    r2 = fit_sketches_with_report(
        demos, "test",
        decomp_ranker=heuristic_decomp_ranker,
        task_signatures=("dims:same",),
    )

    # If both produced sketches, the ranked result should have composite first
    if len(r2.sketches) >= 1:
        families = [s.metadata.get("family") for s in r2.sketches]
        if "composite_role_alignment" in families:
            assert families[0] == "composite_role_alignment"


def test_refinement_decomp_ranking_metadata():
    """Refinement result should carry decomp ranking metadata."""
    from aria.refinement import _run_sketch_refinement
    demos = _composite_demos()

    # Without ranker
    result = _run_sketch_refinement(demos)
    assert result.decomp_ranking_applied is False
    assert result.decomp_ranking_policy == "none"

    # With ranker
    result = _run_sketch_refinement(
        demos,
        decomp_ranker=heuristic_decomp_ranker,
        task_signatures=("dims:same",),
    )
    assert result.decomp_ranking_applied is True
    assert result.decomp_ranking_policy == "heuristic_decomp_ranker"


def test_decomp_ranking_no_effect_on_empty_task():
    """Tasks with no sketch proposals should still report decomp ranking."""
    from aria.refinement import _run_sketch_refinement
    from aria.types import DemoPair, grid_from_list

    demos = (
        DemoPair(
            input=grid_from_list([[0, 0], [0, 0]]),
            output=grid_from_list([[0, 0], [0, 0]]),
        ),
    )
    result = _run_sketch_refinement(
        demos,
        decomp_ranker=heuristic_decomp_ranker,
        task_signatures=("dims:same",),
    )
    assert result.sketches_proposed == 0
    # Decomp ranking still applied even though no sketches resulted
    assert result.decomp_ranking_policy == "heuristic_decomp_ranker"
