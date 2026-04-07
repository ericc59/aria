"""Tests for the factorized target space."""

from aria.factors import (
    Correspondence,
    Decomposition,
    Depth,
    FactorSet,
    Op,
    Scope,
    Selector,
    FACTOR_ENUMS,
    FACTOR_NAMES,
    FACTOR_SIZES,
    enumerate_compatible,
    enumerate_compatible_for,
    is_compatible,
)


def test_factor_enums_complete():
    """All 6 factor dimensions are defined."""
    assert len(FACTOR_NAMES) == 6
    for name in FACTOR_NAMES:
        assert name in FACTOR_ENUMS
        assert FACTOR_SIZES[name] == len(FACTOR_ENUMS[name])


def test_factor_sizes():
    assert FACTOR_SIZES["decomposition"] == 7
    assert FACTOR_SIZES["selector"] == 7
    assert FACTOR_SIZES["scope"] == 8
    assert FACTOR_SIZES["op"] == 8
    assert FACTOR_SIZES["correspondence"] == 6
    assert FACTOR_SIZES["depth"] == 3


def test_factorset_creation():
    fs = FactorSet(
        Decomposition.OBJECT,
        Selector.OBJECT_SELECT,
        Scope.OBJECT,
        Op.RECOLOR,
        Correspondence.NONE,
        Depth.ONE,
    )
    assert fs.decomposition == Decomposition.OBJECT
    assert fs.depth == Depth.ONE


def test_factorset_hashable():
    fs1 = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    )
    fs2 = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    )
    assert fs1 == fs2
    assert hash(fs1) == hash(fs2)
    assert len({fs1, fs2}) == 1


def test_compatibility_basic_valid():
    """A simple object-level recolor should be valid."""
    fs = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    )
    assert is_compatible(fs)


def test_compatibility_partition_frame_invalid():
    """Partition decomp + frame_interior selector is invalid."""
    fs = FactorSet(
        Decomposition.PARTITION, Selector.FRAME_INTERIOR,
        Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    )
    assert not is_compatible(fs)


def test_compatibility_copy_needs_correspondence():
    """COPY_STAMP without correspondence is invalid."""
    fs = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.COPY_STAMP, Correspondence.NONE, Depth.TWO,
    )
    assert not is_compatible(fs)


def test_compatibility_copy_with_correspondence():
    """COPY_STAMP with positional correspondence at depth=2 is valid."""
    fs = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.COPY_STAMP, Correspondence.POSITIONAL, Depth.TWO,
    )
    assert is_compatible(fs)


def test_compatibility_depth1_no_correspondence():
    """Depth=1 with correspondence is invalid (need 2+ steps)."""
    fs = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.RECOLOR, Correspondence.POSITIONAL, Depth.ONE,
    )
    assert not is_compatible(fs)


def test_compatibility_depth3_restricted():
    """Depth=3 is restricted to certain ops."""
    fs_ok = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.TRANSFORM, Correspondence.NONE, Depth.THREE,
    )
    assert is_compatible(fs_ok)

    fs_bad = FactorSet(
        Decomposition.OBJECT, Selector.OBJECT_SELECT,
        Scope.OBJECT, Op.COMBINE, Correspondence.NONE, Depth.THREE,
    )
    assert not is_compatible(fs_bad)


def test_enumerate_compatible_bounded():
    """Compatible combos should be much less than the full cartesian product."""
    all_combos = enumerate_compatible()
    total_cartesian = 7 * 7 * 8 * 8 * 6 * 3  # 56448
    assert len(all_combos) > 50, "too few compatible combos"
    assert len(all_combos) < 3000, "too many compatible combos"
    assert len(all_combos) < total_cartesian * 0.05


def test_enumerate_compatible_all_valid():
    """Every returned combo must pass is_compatible."""
    for fs in enumerate_compatible():
        assert is_compatible(fs), f"invalid combo in enumerate: {fs}"


def test_enumerate_compatible_for_pinned():
    """Pinning a factor should restrict results."""
    all_obj = enumerate_compatible_for(decomposition=Decomposition.OBJECT)
    all_part = enumerate_compatible_for(decomposition=Decomposition.PARTITION)
    all_total = enumerate_compatible()
    assert len(all_obj) > 0
    assert len(all_part) > 0
    assert len(all_obj) + len(all_part) <= len(all_total)


def test_known_families_are_compatible():
    """Factor combos corresponding to existing solved families must be valid."""
    known = [
        # derivation_clone
        FactorSet(Decomposition.REGION, Selector.OBJECT_SELECT,
                  Scope.OBJECT, Op.EXTRACT, Correspondence.NONE, Depth.ONE),
        # geometric_transform
        FactorSet(Decomposition.OBJECT, Selector.NONE,
                  Scope.GLOBAL, Op.TRANSFORM, Correspondence.NONE, Depth.ONE),
        # global_color_map
        FactorSet(Decomposition.OBJECT, Selector.NONE,
                  Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE),
        # fill_enclosed
        FactorSet(Decomposition.REGION, Selector.ENCLOSED,
                  Scope.ENCLOSED_SUBSET, Op.FILL, Correspondence.NONE, Depth.ONE),
        # mask_repair
        FactorSet(Decomposition.MASK, Selector.MARKER,
                  Scope.LOCAL_SUPPORT, Op.REPAIR, Correspondence.NONE, Depth.ONE),
        # partition_cell_select
        FactorSet(Decomposition.PARTITION, Selector.CELL_PANEL,
                  Scope.PARTITION_CELL, Op.EXTRACT, Correspondence.NONE, Depth.ONE),
        # scoped_color_map
        FactorSet(Decomposition.OBJECT, Selector.OBJECT_SELECT,
                  Scope.OBJECT_BBOX, Op.RECOLOR, Correspondence.NONE, Depth.ONE),
        # frame_interior_edit
        FactorSet(Decomposition.FRAME, Selector.FRAME_INTERIOR,
                  Scope.FRAME_INTERIOR, Op.RECOLOR, Correspondence.NONE, Depth.ONE),
        # object_transform (depth=2)
        FactorSet(Decomposition.OBJECT, Selector.OBJECT_SELECT,
                  Scope.OBJECT, Op.TRANSFORM, Correspondence.NONE, Depth.TWO),
    ]
    for fs in known:
        assert is_compatible(fs), f"known family is incompatible: {fs}"


def test_factorset_repr():
    fs = FactorSet(
        Decomposition.OBJECT, Selector.NONE,
        Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
    )
    r = repr(fs)
    assert "object" in r
    assert "recolor" in r
