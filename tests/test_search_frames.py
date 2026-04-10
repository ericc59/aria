import numpy as np

from aria.datasets import get_dataset, load_arc_task
from aria.search.frames import extract_rect_frames, extract_rect_items, group_frames_by_color, render_frame_family
from aria.search.search import search_programs


def test_extract_rect_frames_covers_b5ca7ac4_family_tiles():
    task = load_arc_task(get_dataset("v2-eval"), "b5ca7ac4")
    counts = []
    colors = []
    for ex in task.train:
        frames = extract_rect_frames(ex.input, bg=0, min_span=4)
        counts.append(len(frames))
        colors.append(sorted({frame.color for frame in frames}))
    assert counts == [6, 8, 6]
    assert colors == [[2, 8], [2, 8], [2, 8]]


def test_extract_rect_frames_covers_2ba387bc_box_items():
    task = load_arc_task(get_dataset("v2-eval"), "2ba387bc")
    counts = []
    for ex in task.train:
        frames = extract_rect_frames(ex.input, bg=0, min_span=4)
        counts.append(len(frames))
    assert counts == [3, 2, 2, 3]


def test_extract_rect_items_covers_solid_and_framed_pack_items():
    task = load_arc_task(get_dataset("v2-eval"), "2ba387bc")
    kinds = []
    counts = []
    for ex in task.train:
        items = extract_rect_items(ex.input, bg=0, min_span=4)
        counts.append(len(items))
        kinds.append(sorted({item.kind for item in items}))
    assert counts == [7, 5, 4, 5]
    assert kinds == [["frame", "solid"]] * 4


def test_render_frame_family_compacts_columns_but_preserves_rows():
    task = load_arc_task(get_dataset("v2-eval"), "b5ca7ac4")
    ex = task.train[0]
    frames = extract_rect_frames(ex.input, bg=0, min_span=4)
    grouped = group_frames_by_color(frames)
    bg = int(np.bincount(ex.output.ravel()).argmax())

    left_family = render_frame_family(grouped[8], shape=ex.input.shape, bg=bg, compact_rows=False, compact_cols=True)
    right_family = render_frame_family(grouped[2], shape=ex.input.shape, bg=bg, compact_rows=False, compact_cols=True)

    assert left_family.shape[0] == ex.input.shape[0]
    assert right_family.shape[0] == ex.input.shape[0]
    assert left_family.shape[1] < ex.input.shape[1]
    assert right_family.shape[1] < ex.input.shape[1]


def test_search_still_solves_2ba387bc_via_frame_bbox_pack():
    task = load_arc_task(get_dataset("v2-eval"), "2ba387bc")
    demos = [(ex.input, ex.output) for ex in task.train]
    prog = search_programs(demos, time_budget=5.0)
    assert prog is not None
    assert "frame_bbox_pack" in prog.description
    assert all(np.array_equal(prog.execute(inp), out) for inp, out in demos)
