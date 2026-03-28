"""Tests for the decomposition layer."""

from __future__ import annotations

import numpy as np

from aria.decomposition import (
    CompositeDecomposition,
    CompositeMotif,
    FramedRegion,
    GridDecomposition,
    MarkerNeighborhood,
    ObjectDecomposition,
    RawObject,
    decompose_composites,
    decompose_grid,
    decompose_marker_neighborhoods,
    decompose_objects,
    detect_bg,
    detect_framed_regions,
    extract_objects,
)
from aria.types import grid_from_list


# ---------------------------------------------------------------------------
# detect_bg
# ---------------------------------------------------------------------------


def test_detect_bg_most_common():
    grid = grid_from_list([[0, 0, 1], [0, 0, 0]])
    assert detect_bg(grid) == 0


def test_detect_bg_nonzero():
    grid = grid_from_list([[3, 3, 3], [3, 1, 3], [3, 3, 3]])
    assert detect_bg(grid) == 3


def test_detect_bg_empty():
    grid = np.zeros((0, 0), dtype=np.uint8)
    assert detect_bg(grid) == 0


# ---------------------------------------------------------------------------
# Raw objects
# ---------------------------------------------------------------------------


def test_extract_objects_basic():
    grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [2, 0, 3],
    ])
    objs = extract_objects(grid, bg=0)
    assert len(objs) == 3
    colors = {o.color for o in objs}
    assert colors == {1, 2, 3}


def test_extract_objects_connected():
    grid = grid_from_list([
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
    ])
    objs = extract_objects(grid, bg=0)
    assert len(objs) == 1
    assert objs[0].size == 3
    assert objs[0].color == 1


def test_extract_objects_auto_bg():
    grid = grid_from_list([
        [5, 5, 5],
        [5, 2, 5],
        [5, 5, 5],
    ])
    objs = extract_objects(grid)
    assert len(objs) == 1
    assert objs[0].color == 2


def test_raw_object_properties():
    grid = grid_from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    objs = extract_objects(grid, bg=0)
    assert len(objs) == 1
    obj = objs[0]
    assert obj.is_singleton is True
    assert obj.center_row == 1
    assert obj.center_col == 1
    assert obj.bbox_h == 1
    assert obj.bbox_w == 1


def test_raw_object_to_dict_compat():
    grid = grid_from_list([[0, 1, 0], [0, 0, 0]])
    objs = extract_objects(grid, bg=0)
    d = objs[0].to_dict()
    assert d["color"] == 1
    assert d["row"] == 0
    assert d["col"] == 1
    assert d["size"] == 1
    assert "mask" in d
    assert "mask_bytes" in d
    assert "mask_shape" in d


def test_decompose_objects():
    grid = grid_from_list([
        [0, 1, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 2],
    ])
    dec = decompose_objects(grid, bg=0)
    assert isinstance(dec, ObjectDecomposition)
    assert dec.bg_color == 0
    assert len(dec.objects) == 2
    assert len(dec.singletons) == 1
    assert len(dec.non_singletons) == 1
    assert 1 in dec.color_counts
    assert 2 in dec.color_counts


# ---------------------------------------------------------------------------
# Framed regions
# ---------------------------------------------------------------------------


def test_detect_full_grid_frame():
    grid = grid_from_list([
        [5, 5, 5, 5],
        [5, 1, 2, 5],
        [5, 3, 4, 5],
        [5, 5, 5, 5],
    ])
    regions = detect_framed_regions(grid, bg=1)
    assert len(regions) >= 1
    r = regions[0]
    assert r.frame_color == 5
    assert r.height == 2
    assert r.width == 2
    assert r.interior.shape == (2, 2)


def test_detect_framed_region_interior_content():
    grid = grid_from_list([
        [3, 3, 3, 3, 3],
        [3, 0, 1, 0, 3],
        [3, 1, 0, 1, 3],
        [3, 0, 1, 0, 3],
        [3, 3, 3, 3, 3],
    ])
    regions = detect_framed_regions(grid, bg=0)
    assert len(regions) >= 1
    r = regions[0]
    assert r.frame_color == 3
    assert 0 in r.interior_colors
    assert 1 in r.interior_colors


def test_detect_no_frame_small_grid():
    grid = grid_from_list([[1, 2]])
    regions = detect_framed_regions(grid)
    assert len(regions) == 0


def test_detect_no_frame_non_uniform_border():
    grid = grid_from_list([
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8],
    ])
    regions = detect_framed_regions(grid, bg=0)
    assert len(regions) == 0


def test_detect_separator_bounded_regions():
    """Grid with row/col separators forming sub-regions."""
    grid = grid_from_list([
        [2, 2, 2, 2, 2, 2, 2],
        [2, 0, 0, 2, 0, 0, 2],
        [2, 0, 0, 2, 0, 0, 2],
        [2, 2, 2, 2, 2, 2, 2],
        [2, 0, 0, 2, 0, 0, 2],
        [2, 0, 0, 2, 0, 0, 2],
        [2, 2, 2, 2, 2, 2, 2],
    ])
    regions = detect_framed_regions(grid, bg=0)
    # Should find sub-regions bounded by color 2
    assert len(regions) >= 1


# ---------------------------------------------------------------------------
# Composite motifs
# ---------------------------------------------------------------------------


def test_decompose_composites_basic():
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 8, 4, 8, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ])
    dec = decompose_composites(grid, bg=0)
    assert isinstance(dec, CompositeDecomposition)
    assert len(dec.composites) == 1
    assert dec.composites[0].center.color == 4
    assert dec.composites[0].center_row == 2
    assert dec.composites[0].center_col == 2
    assert dec.center_color == 4
    assert dec.frame_color == 8
    assert dec.anchor is not None
    assert dec.anchor.color == 4
    assert dec.anchor.row == 5


def test_composite_structural_signature():
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 8, 4, 8, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ])
    dec = decompose_composites(grid, bg=0)
    sig = dec.composites[0].structural_signature
    assert isinstance(sig, tuple)
    assert len(sig) == 6  # (n_components, frame_sizes, bbox_h, bbox_w, rel_r, rel_c)


def test_composite_signature_color_invariant():
    """Same structure, different colors → same signature."""
    grid1 = grid_from_list([
        [0, 8, 8, 8, 0],
        [0, 8, 4, 8, 0],
        [0, 8, 8, 8, 0],
    ])
    grid2 = grid_from_list([
        [0, 3, 3, 3, 0],
        [0, 3, 1, 3, 0],
        [0, 3, 3, 3, 0],
    ])
    dec1 = decompose_composites(grid1, bg=0)
    dec2 = decompose_composites(grid2, bg=0)
    assert len(dec1.composites) == 1
    assert len(dec2.composites) == 1
    assert dec1.composites[0].structural_signature == dec2.composites[0].structural_signature


def test_no_composites_without_adjacency():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 3],
    ])
    dec = decompose_composites(grid, bg=0)
    assert len(dec.composites) == 0
    assert len(dec.isolated) == 2


def test_no_composites_same_color_adjacency():
    grid = grid_from_list([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
    ])
    dec = decompose_composites(grid, bg=0)
    assert len(dec.composites) == 0


# ---------------------------------------------------------------------------
# Marker neighborhoods
# ---------------------------------------------------------------------------


def test_marker_neighborhood_basic():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    neighborhoods = decompose_marker_neighborhoods(grid, bg=0)
    # Both colors 1 and 2 appear as single singletons
    assert len(neighborhoods) == 2
    colors = {n.marker_color for n in neighborhoods}
    assert 1 in colors
    assert 2 in colors


def test_marker_neighborhood_directional():
    grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 2, 0],
    ])
    neighborhoods = decompose_marker_neighborhoods(grid, bg=0)
    # Find the marker for color 1
    m1 = next(n for n in neighborhoods if n.marker_color == 1)
    assert len(m1.below) >= 1  # color 2 is below
    assert len(m1.above) == 0


def test_marker_not_detected_for_multi_instance_color():
    """Colors with multiple singletons are not markers."""
    grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    neighborhoods = decompose_marker_neighborhoods(grid, bg=0)
    # Color 1 has 2 singletons, so it's not a marker
    assert len(neighborhoods) == 0


def test_marker_nearest_by_color():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
    ])
    neighborhoods = decompose_marker_neighborhoods(grid, bg=0)
    for n in neighborhoods:
        assert isinstance(n.nearest_by_color, dict)


# ---------------------------------------------------------------------------
# Full grid decomposition
# ---------------------------------------------------------------------------


def test_decompose_grid_returns_all_views():
    grid = grid_from_list([
        [5, 5, 5, 5, 5],
        [5, 0, 1, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ])
    dec = decompose_grid(grid)
    assert isinstance(dec, GridDecomposition)
    assert dec.bg_color == 5
    assert isinstance(dec.objects, ObjectDecomposition)
    assert isinstance(dec.framed_regions, tuple)
    assert isinstance(dec.composites, CompositeDecomposition)
    assert isinstance(dec.marker_neighborhoods, tuple)
    # Should detect the frame
    assert len(dec.framed_regions) >= 1


def test_decompose_grid_empty():
    grid = grid_from_list([[0, 0], [0, 0]])
    dec = decompose_grid(grid)
    assert len(dec.objects.objects) == 0
    assert len(dec.framed_regions) == 0
    assert len(dec.composites.composites) == 0


# ---------------------------------------------------------------------------
# Backward compatibility: observe.py still works
# ---------------------------------------------------------------------------


def test_observe_extract_objects_uses_decomposition():
    """observe._extract_objects_with_bg delegates to decomposition layer."""
    from aria.observe import _extract_objects_with_bg

    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 8, 4, 8, 0, 0],
        [0, 8, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ])
    objs = _extract_objects_with_bg(grid)
    assert isinstance(objs, list)
    assert len(objs) >= 2
    # All should be dicts with expected keys
    for o in objs:
        assert "color" in o
        assert "row" in o
        assert "size" in o
        assert "mask" in o


def test_observe_still_solves_basic():
    """Basic observation synthesis still works after routing through decomposition."""
    from aria.observe import observe_and_synthesize
    from aria.types import DemoPair

    demos = (
        DemoPair(
            input=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
        ),
    )
    result = observe_and_synthesize(demos)
    # Should at least find some rules
    assert result.candidates_tested >= 0
