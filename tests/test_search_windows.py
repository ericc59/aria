from __future__ import annotations

from collections import Counter

from aria.datasets import get_dataset, load_arc_task
from aria.search.binding import derive_scene_binding
from aria.search.windows import classify_bar_window_roles, extract_bar_windows


def test_bar_window_extraction_covers_271d71e2_training_geometry():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "271d71e2")

    expected_counts = [2, 6, 4]
    expected_side_counts = [
        Counter({"top": 2}),
        Counter({"left": 4, "right": 1, "top": 1}),
        Counter({"left": 3, "right": 1}),
    ]

    for idx, pair in enumerate(task.train):
        windows = extract_bar_windows(pair.input, bg=6)
        side_counts = Counter(window.side for window in windows)

        assert len(windows) == expected_counts[idx]
        assert side_counts == expected_side_counts[idx]


def test_bar_window_role_hints_identify_sources_targets_and_fillers():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "271d71e2")

    roles = classify_bar_window_roles(extract_bar_windows(task.train[1].input, bg=6))
    counts = Counter(role for _, role, _, _ in roles)

    assert counts["SOURCE"] == 1
    assert counts["TARGET"] == 3
    assert counts["WORKSPACE"] == 2


def test_scene_binding_exposes_window_entities_for_g06_layouts():
    ds = get_dataset("v2-eval")
    task = load_arc_task(ds, "271d71e2")
    binding = derive_scene_binding([(pair.input, pair.output) for pair in task.train])

    window_roles = [(ra.entity.kind, ra.role.name) for ra in binding.roles if ra.entity.kind == "window"]

    assert window_roles
    assert any(role == "SOURCE" for _, role in window_roles)
