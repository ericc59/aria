"""Validate state graph extraction on all ARC tasks and collect statistics.

Usage:
    python -m scripts.validate_extraction --dataset v1-train --limit 20
    python -m scripts.validate_extraction --dataset v1-eval
    python -m scripts.validate_extraction --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aria.graph.extract import extract, extract_with_delta
from aria.graph.zones import find_zones
from aria.types import Delta, Shape, Symmetry, GlobalSymmetry, StateGraph, grid_from_list


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path.home() / "dev" / "arcagi" / "arc-agi-benchmarking" / "data" / "public-v1"

DATASETS: dict[str, Path] = {
    "v1-train": _DATA_ROOT / "training",
    "v1-eval": _DATA_ROOT / "evaluation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_task_file(path: Path) -> dict:
    """Load a single ARC task JSON file."""
    with open(path) as f:
        return json.load(f)


def safe_serialize(obj: Any) -> Any:
    """Convert non-JSON-serializable types for the report."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, frozenset):
        return sorted(str(v) for v in obj)
    if isinstance(obj, set):
        return sorted(str(v) for v in obj)
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


# ---------------------------------------------------------------------------
# Per-grid statistics
# ---------------------------------------------------------------------------

def collect_grid_stats(sg: StateGraph) -> dict[str, Any]:
    """Collect statistics from a single StateGraph."""
    shapes = [obj.shape.name for obj in sg.objects]
    symmetries: list[str] = []
    for obj in sg.objects:
        symmetries.extend(s.name for s in obj.symmetry)

    return {
        "dims": sg.context.dims,
        "bg_color": sg.context.bg_color,
        "obj_count": sg.context.obj_count,
        "is_tiled": sg.context.is_tiled,
        "global_symmetry": [s.name for s in sg.context.symmetry],
        "palette_size": len(sg.context.palette),
        "shapes": shapes,
        "symmetries": symmetries,
        "relation_count": len(sg.relations),
    }


def collect_delta_stats(delta: Delta) -> dict[str, Any]:
    """Collect statistics from a Delta."""
    mod_fields = [field for (_, field, _, _) in delta.modified]
    return {
        "added_count": len(delta.added),
        "removed_count": len(delta.removed),
        "modified_count": len(delta.modified),
        "modified_fields": mod_fields,
        "dims_changed": delta.dims_changed is not None,
    }


def collect_zone_stats(grid: np.ndarray) -> dict[str, Any]:
    """Detect zones and return stats."""
    zones = find_zones(grid)
    return {
        "zone_count": len(zones),
        "has_zones": len(zones) > 1,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class StatsAggregator:
    """Accumulates statistics across all tasks."""

    def __init__(self) -> None:
        self.total_tasks = 0
        self.success_count = 0
        self.failure_count = 0
        self.errors: list[dict[str, str]] = []

        # Per-grid stats
        self.obj_counts: list[int] = []
        self.shape_counter: Counter[str] = Counter()
        self.symmetry_counter: Counter[str] = Counter()
        self.global_sym_counter: Counter[str] = Counter()
        self.bg_color_counter: Counter[int] = Counter()
        self.palette_sizes: list[int] = []
        self.grid_dims: list[tuple[int, int]] = []

        # Tiling
        self.tiled_grids = 0
        self.total_grids = 0

        # Zone detection
        self.grids_with_zones = 0

        # Delta stats
        self.total_pairs = 0
        self.dims_changed_count = 0
        self.delta_added: list[int] = []
        self.delta_removed: list[int] = []
        self.delta_modified: list[int] = []
        self.delta_mod_fields: Counter[str] = Counter()

        # Timing
        self.extraction_times: list[float] = []

    def record_grid(self, gs: dict[str, Any]) -> None:
        self.total_grids += 1
        self.obj_counts.append(gs["obj_count"])
        self.shape_counter.update(gs["shapes"])
        self.symmetry_counter.update(gs["symmetries"])
        self.global_sym_counter.update(gs["global_symmetry"])
        self.bg_color_counter[gs["bg_color"]] += 1
        self.palette_sizes.append(gs["palette_size"])
        self.grid_dims.append(gs["dims"])
        if gs["is_tiled"] is not None:
            self.tiled_grids += 1

    def record_zones(self, zs: dict[str, Any]) -> None:
        if zs["has_zones"]:
            self.grids_with_zones += 1

    def record_delta(self, ds: dict[str, Any]) -> None:
        self.total_pairs += 1
        self.delta_added.append(ds["added_count"])
        self.delta_removed.append(ds["removed_count"])
        self.delta_modified.append(ds["modified_count"])
        self.delta_mod_fields.update(ds["modified_fields"])
        if ds["dims_changed"]:
            self.dims_changed_count += 1

    def record_time(self, elapsed: float) -> None:
        self.extraction_times.append(elapsed)

    def summary(self) -> dict[str, Any]:
        """Build the final summary dict."""
        total_times = self.extraction_times
        avg_time = sum(total_times) / len(total_times) if total_times else 0.0

        obj_counter = Counter(self.obj_counts)

        return {
            "tasks": {
                "total": self.total_tasks,
                "success": self.success_count,
                "failure": self.failure_count,
                "error_tasks": [e["task"] for e in self.errors],
            },
            "grids": {
                "total": self.total_grids,
                "avg_extraction_time_ms": round(avg_time * 1000, 2),
                "median_extraction_time_ms": round(
                    float(np.median(total_times)) * 1000, 2
                ) if total_times else 0.0,
                "p95_extraction_time_ms": round(
                    float(np.percentile(total_times, 95)) * 1000, 2
                ) if total_times else 0.0,
                "max_extraction_time_ms": round(
                    max(total_times) * 1000, 2
                ) if total_times else 0.0,
            },
            "objects": {
                "total_objects": sum(self.obj_counts),
                "obj_count_distribution": dict(
                    sorted(obj_counter.items(), key=lambda x: x[0])
                ),
                "avg_per_grid": round(
                    sum(self.obj_counts) / len(self.obj_counts), 2
                ) if self.obj_counts else 0.0,
                "max_per_grid": max(self.obj_counts) if self.obj_counts else 0,
            },
            "shapes": {
                "distribution": dict(self.shape_counter.most_common()),
            },
            "symmetry": {
                "object_symmetry_distribution": dict(
                    self.symmetry_counter.most_common()
                ),
                "global_symmetry_distribution": dict(
                    self.global_sym_counter.most_common()
                ),
            },
            "background_colors": dict(
                sorted(self.bg_color_counter.items(), key=lambda x: -x[1])
            ),
            "tiling": {
                "grids_with_tiling": self.tiled_grids,
                "total_grids": self.total_grids,
                "tiling_rate": round(
                    self.tiled_grids / self.total_grids, 4
                ) if self.total_grids else 0.0,
            },
            "zones": {
                "grids_with_multi_zones": self.grids_with_zones,
                "zone_detection_rate": round(
                    self.grids_with_zones / self.total_grids, 4
                ) if self.total_grids else 0.0,
            },
            "deltas": {
                "total_pairs": self.total_pairs,
                "dims_changed_count": self.dims_changed_count,
                "dims_change_rate": round(
                    self.dims_changed_count / self.total_pairs, 4
                ) if self.total_pairs else 0.0,
                "avg_added": round(
                    sum(self.delta_added) / len(self.delta_added), 2
                ) if self.delta_added else 0.0,
                "avg_removed": round(
                    sum(self.delta_removed) / len(self.delta_removed), 2
                ) if self.delta_removed else 0.0,
                "avg_modified": round(
                    sum(self.delta_modified) / len(self.delta_modified), 2
                ) if self.delta_modified else 0.0,
                "modified_field_distribution": dict(
                    self.delta_mod_fields.most_common()
                ),
            },
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_task(
    task_id: str,
    task_data: dict,
    agg: StatsAggregator,
) -> dict[str, Any]:
    """Process a single task, return per-task detail record."""
    task_record: dict[str, Any] = {
        "task_id": task_id,
        "pairs": [],
        "success": True,
        "error": None,
    }

    try:
        for pair_idx, pair in enumerate(task_data["train"]):
            in_grid = grid_from_list(pair["input"])
            out_grid = grid_from_list(pair["output"])

            # Time the full extraction + delta
            t0 = time.perf_counter()
            sg_in, sg_out, delta = extract_with_delta(in_grid, out_grid)
            elapsed = time.perf_counter() - t0

            # Collect stats for input grid
            gs_in = collect_grid_stats(sg_in)
            agg.record_grid(gs_in)
            agg.record_time(elapsed / 2)  # attribute half to each grid

            # Collect stats for output grid
            gs_out = collect_grid_stats(sg_out)
            agg.record_grid(gs_out)
            agg.record_time(elapsed / 2)

            # Zone detection on input
            zs_in = collect_zone_stats(in_grid)
            agg.record_zones(zs_in)

            # Zone detection on output
            zs_out = collect_zone_stats(out_grid)
            agg.record_zones(zs_out)

            # Delta stats
            ds = collect_delta_stats(delta)
            agg.record_delta(ds)

            pair_record = {
                "pair_idx": pair_idx,
                "input": gs_in,
                "output": gs_out,
                "delta": ds,
                "zones_in": zs_in,
                "zones_out": zs_out,
                "extraction_time_ms": round(elapsed * 1000, 2),
            }
            task_record["pairs"].append(pair_record)

        agg.success_count += 1

    except Exception as exc:
        task_record["success"] = False
        task_record["error"] = f"{type(exc).__name__}: {exc}"
        agg.failure_count += 1
        agg.errors.append({
            "task": task_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })

    agg.total_tasks += 1
    return task_record


def resolve_dataset_paths(dataset: str) -> list[tuple[str, Path]]:
    """Return list of (dataset_name, directory_path) for the requested dataset."""
    if dataset == "all":
        paths = []
        for name, p in sorted(DATASETS.items()):
            paths.append((name, p))
        return paths
    if dataset in DATASETS:
        return [(dataset, DATASETS[dataset])]
    print(f"Unknown dataset: {dataset}. Available: {', '.join(DATASETS)} or 'all'")
    sys.exit(1)


def gather_task_files(dataset_paths: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Collect all task JSON files, tagged with dataset prefix."""
    files: list[tuple[str, Path]] = []
    for ds_name, ds_dir in dataset_paths:
        if not ds_dir.exists():
            print(f"WARNING: dataset directory not found: {ds_dir}")
            continue
        for f in sorted(ds_dir.glob("*.json")):
            task_id = f"{ds_name}/{f.stem}"
            files.append((task_id, f))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate state graph extraction on ARC tasks."
    )
    parser.add_argument(
        "--dataset",
        default="v1-train",
        help="Which dataset to process: v1-train, v1-eval, or all. Default: v1-train",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks to process (0 = no limit). Default: 0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "extraction_report.json"),
        help="Path to save the detailed JSON report.",
    )
    args = parser.parse_args()

    dataset_paths = resolve_dataset_paths(args.dataset)
    task_files = gather_task_files(dataset_paths)

    if args.limit > 0:
        task_files = task_files[: args.limit]

    print(f"Processing {len(task_files)} tasks from dataset '{args.dataset}'")
    print(f"Report will be saved to: {args.output}")
    print("-" * 60)

    agg = StatsAggregator()
    task_records: list[dict[str, Any]] = []

    t_start = time.perf_counter()

    for idx, (task_id, task_path) in enumerate(task_files):
        task_data = load_task_file(task_path)
        record = process_task(task_id, task_data, agg)
        task_records.append(record)

        # Progress indicator every 50 tasks or on failure
        if (idx + 1) % 50 == 0 or not record["success"]:
            status = "OK" if record["success"] else f"FAIL: {record['error']}"
            print(f"  [{idx + 1}/{len(task_files)}] {task_id}: {status}")

    total_wall = time.perf_counter() - t_start

    # Print summary
    summary = agg.summary()
    print()
    print("=" * 60)
    print("EXTRACTION VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTasks: {summary['tasks']['total']} total, "
          f"{summary['tasks']['success']} success, "
          f"{summary['tasks']['failure']} failures")

    if summary["tasks"]["failure"] > 0:
        print(f"  Failed tasks: {summary['tasks']['error_tasks']}")

    print(f"\nGrids analyzed: {summary['grids']['total']}")
    print(f"  Avg extraction time: {summary['grids']['avg_extraction_time_ms']:.2f} ms")
    print(f"  Median extraction time: {summary['grids']['median_extraction_time_ms']:.2f} ms")
    print(f"  P95 extraction time: {summary['grids']['p95_extraction_time_ms']:.2f} ms")
    print(f"  Max extraction time: {summary['grids']['max_extraction_time_ms']:.2f} ms")

    print(f"\nObjects: {summary['objects']['total_objects']} total, "
          f"avg {summary['objects']['avg_per_grid']}/grid, "
          f"max {summary['objects']['max_per_grid']}/grid")
    print(f"  Object count distribution (top 10):")
    obj_dist = summary["objects"]["obj_count_distribution"]
    for count, freq in list(sorted(obj_dist.items(), key=lambda x: -x[1]))[:10]:
        print(f"    {count} objects: {freq} grids")

    print(f"\nShapes:")
    for shape, freq in list(summary["shapes"]["distribution"].items())[:10]:
        print(f"    {shape}: {freq}")

    print(f"\nObject symmetries:")
    for sym, freq in list(summary["symmetry"]["object_symmetry_distribution"].items())[:10]:
        print(f"    {sym}: {freq}")

    print(f"\nGlobal symmetries:")
    if summary["symmetry"]["global_symmetry_distribution"]:
        for sym, freq in summary["symmetry"]["global_symmetry_distribution"].items():
            print(f"    {sym}: {freq}")
    else:
        print("    (none detected)")

    print(f"\nBackground colors:")
    for color, freq in summary["background_colors"].items():
        print(f"    color {color}: {freq} grids")

    print(f"\nTiling: {summary['tiling']['grids_with_tiling']}/{summary['tiling']['total_grids']} "
          f"grids ({summary['tiling']['tiling_rate'] * 100:.1f}%)")

    print(f"\nZone detection: {summary['zones']['grids_with_multi_zones']}/{summary['grids']['total']} "
          f"grids have multiple zones ({summary['zones']['zone_detection_rate'] * 100:.1f}%)")

    print(f"\nDeltas ({summary['deltas']['total_pairs']} train pairs):")
    print(f"  Dimension changes: {summary['deltas']['dims_changed_count']} "
          f"({summary['deltas']['dims_change_rate'] * 100:.1f}%)")
    print(f"  Avg objects added: {summary['deltas']['avg_added']:.2f}")
    print(f"  Avg objects removed: {summary['deltas']['avg_removed']:.2f}")
    print(f"  Avg modifications: {summary['deltas']['avg_modified']:.2f}")
    print(f"  Modified fields:")
    for field, freq in summary["deltas"]["modified_field_distribution"].items():
        print(f"    {field}: {freq}")

    print(f"\nTotal wall time: {total_wall:.1f}s")
    print("=" * 60)

    # Save report
    report = {
        "meta": {
            "dataset": args.dataset,
            "limit": args.limit,
            "task_count": len(task_files),
            "wall_time_s": round(total_wall, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
        "tasks": task_records,
    }

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=safe_serialize)

    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
