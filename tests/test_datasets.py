"""Tests for aria.datasets — dataset registry and task loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aria.datasets import (
    DatasetInfo,
    dataset_names,
    get_dataset,
    iter_tasks,
    list_task_ids,
    load_arc_task,
)


def _make_arc_task_json() -> dict:
    """Minimal valid ARC task JSON."""
    return {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
        ],
        "test": [
            {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
        ],
    }


def _make_dataset(tmp_path: Path, name: str = "test-ds") -> DatasetInfo:
    """Create a tiny on-disk dataset for testing."""
    root = tmp_path / name
    root.mkdir()
    for tid in ("aaa", "bbb", "ccc"):
        (root / f"{tid}.json").write_text(json.dumps(_make_arc_task_json()))
    return DatasetInfo(name=name, version=99, split="testing", root=root)


# --- registry ---


def test_dataset_names_includes_known_splits():
    names = dataset_names()
    assert "v1-train" in names
    assert "v2-eval" in names


def test_get_dataset_returns_info():
    info = get_dataset("v2-eval")
    assert info.version == 2
    assert info.split == "evaluation"
    assert info.short_version == "ARC-2"


def test_get_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("v99-fake")


# --- list / load / iter ---


def test_list_task_ids(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    ids = list_task_ids(ds)
    assert ids == ["aaa", "bbb", "ccc"]


def test_list_task_ids_missing_dir(tmp_path: Path):
    ds = DatasetInfo("x", 1, "training", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        list_task_ids(ds)


def test_load_arc_task(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    task = load_arc_task(ds, "aaa")
    assert len(task.train) == 1
    assert len(task.test) == 1
    assert task.train[0].input.shape == (2, 2)


def test_load_arc_task_missing(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_arc_task(ds, "nonexistent")


def test_iter_tasks_all(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    pairs = list(iter_tasks(ds))
    assert len(pairs) == 3
    assert [tid for tid, _ in pairs] == ["aaa", "bbb", "ccc"]


def test_iter_tasks_limit(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    pairs = list(iter_tasks(ds, limit=2))
    assert len(pairs) == 2


def test_iter_tasks_explicit_ids(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    pairs = list(iter_tasks(ds, task_ids=["ccc", "aaa"]))
    assert [tid for tid, _ in pairs] == ["ccc", "aaa"]


def test_iter_tasks_limit_with_explicit_ids(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    pairs = list(iter_tasks(ds, task_ids=["ccc", "aaa", "bbb"], limit=1))
    assert len(pairs) == 1
    assert pairs[0][0] == "ccc"


# --- DatasetInfo ---


def test_dataset_info_short_version():
    info = DatasetInfo("v1-train", 1, "training", Path("/tmp"))
    assert info.short_version == "ARC-1"
