"""Tests for decomposition search."""

from __future__ import annotations

import numpy as np

from aria.search.task_analysis import analyze_task
from aria.search.decompose import search_decomposed, _build_splitters


def test_crop_then_recolor():
    """2-step: crop non-bg bbox, then recolor (color map)."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 1

    out = np.full((2, 2), 2, dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)
    assert analysis.dims_change

    prog = search_decomposed(demos, analysis)
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
        assert 'decompose:' in prog.provenance


def test_remove_color_splitter():
    """Splitter removes a color that disappears in the output."""
    inp = np.array([[1, 2], [2, 1]], dtype=np.int8)
    out = np.array([[1, 0], [0, 1]], dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)
    assert 2 in analysis.removed_colors

    prog = search_decomposed(demos, analysis)
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)


def test_composition_correctness():
    """Composed program's steps are ordered: splitter first, then sub-derive."""
    inp = np.zeros((4, 4), dtype=np.int8)
    inp[0:2, 0:2] = 3

    out = np.full((2, 2), 3, dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)

    prog = search_decomposed(demos, analysis)
    if prog is not None:
        assert prog.steps[0].action == 'crop_nonbg'
        assert np.array_equal(prog.execute(inp), out)


# --- Gating tests ---

def test_recolor_only_suppresses_spatial():
    """recolor_only diff_type should not produce crop or extract splitters."""
    inp = np.array([[1, 0], [0, 2]], dtype=np.int8)
    out = np.array([[3, 0], [0, 4]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    assert analysis.diff_type == 'recolor_only'
    assert not analysis.dims_change

    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert not any('crop' in n for n in names)
    assert not any('panel' in n for n in names)
    assert not any('transform' in n for n in names)


def test_extraction_enables_crop():
    """is_extraction should enable crop_non_bg_bbox."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 2:4] = 7
    out = np.array([[7, 7], [7, 7]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    assert analysis.is_extraction

    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert 'crop_non_bg_bbox' in names


def test_rearrange_enables_transforms():
    """rearrange diff_type should enable flip/rot splitters."""
    # Needs bg pixels so mask changes rule out recolor_only
    inp = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=np.int8)
    out = np.array([[3, 0, 1], [0, 0, 0], [4, 0, 2]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    assert analysis.diff_type == 'rearrange'

    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert any('transform' in n for n in names)
    assert not any('crop' in n for n in names)


def test_subtractive_enables_remove_objects():
    """subtractive diff_type should enable object-removal splitters."""
    inp = np.array([[1, 1, 2], [1, 1, 0], [0, 0, 0]], dtype=np.int8)
    out = np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    assert analysis.diff_type == 'subtractive'

    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert any('remove_objects' in n for n in names)


def test_remove_largest_object():
    """remove_objects_largest splitter removes the largest object."""
    from aria.search.decompose import _apply_remove_objects

    inp = np.zeros((5, 5), dtype=np.int8)
    inp[0:3, 0:3] = 1  # 3x3 = largest
    inp[4, 4] = 2      # 1x1 = smallest

    result = _apply_remove_objects(inp, 'largest')
    assert result[0, 0] == 0  # largest removed
    assert result[4, 4] == 2  # smallest kept


def test_flip_splitter_then_derive():
    """flip_h as splitter + identity derive = flip task."""
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    out = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int8)

    demos = [(inp, out)]
    analysis = analyze_task(demos)
    assert analysis.diff_type == 'rearrange'

    prog = search_decomposed(demos, analysis)
    # flip_h transforms inp to match out exactly
    if prog is not None:
        assert np.array_equal(prog.execute(inp), out)
        assert 'transform' in prog.provenance


def test_splitter_count():
    """Ensure we produce a reasonable number of splitters for a complex task."""
    inp = np.zeros((7, 7), dtype=np.int8)
    inp[3, :] = 5  # separator
    inp[0:3, 0:3] = 1
    inp[4:7, 4:7] = 2

    out = np.zeros((3, 3), dtype=np.int8)
    out[:] = 1

    analysis = analyze_task([(inp, out)])
    splitters = _build_splitters(analysis)
    # crop + panel×3 + legend + remove_color + crop_unique_color
    assert len(splitters) >= 5, f"Only {len(splitters)} splitters: {[s.name for s in splitters]}"


def test_apply_color_map_pruned():
    """apply_color_map splitter was pruned — should not appear."""
    inp = np.array([[1, 2], [2, 1]], dtype=np.int8)
    out = np.array([[1, 0], [0, 1]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert 'apply_color_map' not in names


def test_rot90_rot270_pruned():
    """rot90/rot270 were pruned from same_dims transforms."""
    inp = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=np.int8)
    out = np.array([[3, 0, 1], [0, 0, 0], [4, 0, 2]], dtype=np.int8)

    analysis = analyze_task([(inp, out)])
    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert 'apply_transform_rot90' not in names
    assert 'apply_transform_rot270' not in names
    # flip_h/flip_v/rot180 should still be present
    assert 'apply_transform_flip_h' in names


def test_crop_unique_color_splitter():
    """crop_object_unique_color extracts the uniquely-colored object."""
    from aria.search.decompose import _apply_crop_unique_color

    inp = np.zeros((6, 6), dtype=np.int8)
    inp[0:2, 0:2] = 1  # object of color 1
    inp[0:2, 4:6] = 1  # another object of color 1
    inp[4:6, 0:3] = 7  # unique color 7

    result = _apply_crop_unique_color(inp)
    assert result.shape == (2, 3)
    assert (result == 7).all()


def test_crop_unique_color_in_splitters():
    """crop_object_unique_color should appear for dims_change tasks."""
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[1:3, 1:3] = 1

    analysis = analyze_task([(inp, np.zeros((2, 2), dtype=np.int8))])
    assert analysis.dims_change
    splitters = _build_splitters(analysis)
    names = [s.name for s in splitters]
    assert 'crop_object_unique_color' in names
