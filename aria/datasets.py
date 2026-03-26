"""Dataset loading for ARC-AGI benchmarks.

Provides a uniform interface over ARC-1 and ARC-2 splits so that
scripts can select a dataset by name without path-wrangling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from aria.solver import load_task
from aria.types import Task


_ARC_BENCHMARKING_ROOT = Path(
    os.environ.get(
        "ARC_DATA_ROOT",
        os.path.expanduser("~/dev/arcagi/arc-agi-benchmarking/data"),
    )
)


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a named ARC dataset split."""

    name: str          # e.g. "v2-eval"
    version: int       # 1 or 2
    split: str         # "training" or "evaluation"
    root: Path         # directory containing per-task JSON files

    @property
    def short_version(self) -> str:
        return f"ARC-{self.version}"


def _default_datasets() -> dict[str, DatasetInfo]:
    """Build the registry from the standard benchmarking checkout."""
    base = _ARC_BENCHMARKING_ROOT
    return {
        "v1-train": DatasetInfo("v1-train", 1, "training", base / "public-v1" / "training"),
        "v1-eval": DatasetInfo("v1-eval", 1, "evaluation", base / "public-v1" / "evaluation"),
        "v2-train": DatasetInfo("v2-train", 2, "training", base / "public-v2" / "training"),
        "v2-eval": DatasetInfo("v2-eval", 2, "evaluation", base / "public-v2" / "evaluation"),
    }


DATASETS: dict[str, DatasetInfo] = _default_datasets()


def dataset_names() -> list[str]:
    return sorted(DATASETS.keys())


def get_dataset(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Known: {', '.join(dataset_names())}"
        )
    return DATASETS[name]


def list_task_ids(ds: DatasetInfo) -> list[str]:
    """Return sorted task IDs available in a dataset."""
    if not ds.root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds.root}")
    return sorted(
        f.stem for f in ds.root.iterdir() if f.suffix == ".json"
    )


def load_arc_task(ds: DatasetInfo, task_id: str) -> Task:
    """Load a single task by ID from a dataset."""
    path = ds.root / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Task not found: {path}")
    with open(path) as f:
        return load_task(json.load(f))


def iter_tasks(
    ds: DatasetInfo,
    *,
    limit: int = 0,
    task_ids: list[str] | None = None,
) -> Iterator[tuple[str, Task]]:
    """Yield (task_id, Task) pairs from a dataset.

    If *task_ids* is given, iterate only those (in the given order).
    Otherwise iterate all tasks sorted by ID.
    *limit* caps the total count (0 = unlimited).
    """
    ids = task_ids if task_ids is not None else list_task_ids(ds)
    for i, tid in enumerate(ids):
        if limit > 0 and i >= limit:
            break
        yield tid, load_arc_task(ds, tid)
