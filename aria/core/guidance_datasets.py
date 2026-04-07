"""Training datasets for bounded guidance subproblems.

Extracts compact (features, label) examples from guidance exports/traces
for each "ready" subproblem. Datasets are explicit, inspectable, and
small enough for simple models.

No task-id logic. No heavy ML dependencies.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_perception_features(demos: tuple[DemoPair, ...]) -> dict[str, float]:
    """Extract a compact numeric feature vector from task demos.

    Returns a dict of named features suitable for simple classifiers.
    All values are numeric (int/float). No task-id leakage.
    """
    from aria.core.grid_perception import perceive_grid
    from aria.decomposition import detect_bg

    if not demos:
        return {}

    states = [perceive_grid(d.input) for d in demos]
    first = states[0]

    same_dims = all(d.input.shape == d.output.shape for d in demos)
    in_rows, in_cols = first.dims
    out_rows, out_cols = demos[0].output.shape

    n_colors = len(first.palette)
    n_non_bg = len(first.non_bg_colors)
    n_obj4 = len(first.objects.objects)
    n_obj8 = len(first.objects8.objects)
    n_framed = len(first.framed_regions)
    n_boxed = len(first.boxed_regions)
    n_zones = len(first.zones)
    has_partition = first.partition is not None
    has_legend = first.legend is not None

    # Partition dims if present
    part_rows = first.partition.n_rows if has_partition else 0
    part_cols = first.partition.n_cols if has_partition else 0

    # Size ratios
    area_in = in_rows * in_cols
    area_out = out_rows * out_cols
    area_ratio = area_out / max(area_in, 1)
    row_ratio = out_rows / max(in_rows, 1)
    col_ratio = out_cols / max(in_cols, 1)

    return {
        "same_dims": float(same_dims),
        "in_rows": float(in_rows),
        "in_cols": float(in_cols),
        "out_rows": float(out_rows),
        "out_cols": float(out_cols),
        "area_ratio": area_ratio,
        "row_ratio": row_ratio,
        "col_ratio": col_ratio,
        "n_colors": float(n_colors),
        "n_non_bg": float(n_non_bg),
        "n_obj4": float(n_obj4),
        "n_obj8": float(n_obj8),
        "n_framed": float(n_framed),
        "n_boxed": float(n_boxed),
        "n_zones": float(n_zones),
        "has_partition": float(has_partition),
        "has_legend": float(has_legend),
        "part_rows": float(part_rows),
        "part_cols": float(part_cols),
        "n_demos": float(len(demos)),
        "bg_color": float(first.bg_color),
    }


FEATURE_NAMES = [
    "same_dims", "in_rows", "in_cols", "out_rows", "out_cols",
    "area_ratio", "row_ratio", "col_ratio",
    "n_colors", "n_non_bg", "n_obj4", "n_obj8",
    "n_framed", "n_boxed", "n_zones",
    "has_partition", "has_legend", "part_rows", "part_cols",
    "n_demos", "bg_color",
]


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert feature dict to a fixed-order numpy array."""
    return np.array([features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset example types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidanceExample:
    """One training example for a guidance subproblem."""
    features: np.ndarray        # shape (n_features,)
    label: int                  # class index
    label_name: str             # human-readable label
    candidates: tuple[str, ...]  # all candidate names in order
    task_id: str = ""           # for debugging only, never used for dispatch


@dataclass
class GuidanceDataset:
    """A collection of guidance examples for one subproblem."""
    name: str
    label_names: tuple[str, ...]  # index -> name mapping
    examples: list[GuidanceExample] = field(default_factory=list)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays for training."""
        if not self.examples:
            return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=int)
        X = np.stack([e.features for e in self.examples])
        y = np.array([e.label for e in self.examples], dtype=int)
        return X, y

    def label_distribution(self) -> dict[str, int]:
        counts = Counter(e.label for e in self.examples)
        return {self.label_names[k]: v for k, v in sorted(counts.items())}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_examples": self.n_examples,
            "n_classes": self.n_classes,
            "label_names": list(self.label_names),
            "label_distribution": self.label_distribution(),
        }


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


# Group rare size modes into families for learnability
SIZE_MODE_FAMILIES = {
    "same_as_input": "same_dims",
    "transpose_input": "transpose",
    "scale_input": "scale",
    "scale_input_by_palette_size": "scale",
    "additive_input": "additive",
    "square_canvas_scaled": "scale",
    "separator_cell_size": "partition_derived",
    "separator_panel_size": "partition_derived",
    "selected_partition_cell_size": "partition_derived",
    "partition_grid_shape": "partition_derived",
    "frame_interior_size": "frame_derived",
    "selected_boxed_region_interior": "frame_derived",
    "tight_bbox_of_non_bg": "bbox_derived",
    "bbox_of_selected_object": "bbox_derived",
    "bbox_of_selected_color": "bbox_derived",
    "scaled_bbox_of_selected_object": "bbox_derived",
    "object_position_grid_shape": "structure_derived",
    "solid_rectangle_layout_shape": "structure_derived",
    "fixed_output_dims": "fixed",
}

SIZE_FAMILY_NAMES = tuple(sorted(set(SIZE_MODE_FAMILIES.values()) | {"other"}))


def build_output_size_dataset(
    task_demos_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> GuidanceDataset:
    """Build dataset for output-size mode family prediction."""
    from aria.core.output_stage1 import infer_output_stage1_spec

    label_to_idx = {name: i for i, name in enumerate(SIZE_FAMILY_NAMES)}
    ds = GuidanceDataset(name="output_size_mode", label_names=SIZE_FAMILY_NAMES)

    for task_id, demos in task_demos_pairs:
        stage1 = infer_output_stage1_spec(demos)
        if stage1 is None:
            continue

        mode = stage1.size_spec.mode
        family = SIZE_MODE_FAMILIES.get(mode, "other")
        label_idx = label_to_idx[family]

        features = extract_perception_features(demos)
        ds.examples.append(GuidanceExample(
            features=features_to_array(features),
            label=label_idx,
            label_name=family,
            candidates=SIZE_FAMILY_NAMES,
            task_id=task_id,
        ))

    return ds


AXIS_NAMES = ("row", "col")
PERIOD_NAMES = ("2", "3", "4", "5")
TRANSFORM_NAMES = ("rotate_90", "rotate_180", "rotate_270", "reflect_row", "reflect_col", "transpose")
LANE_NAMES = ("periodic_repair", "replication", "relocation", "grid_transform")


def build_periodic_axis_dataset(
    task_demos_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> GuidanceDataset:
    """Build dataset for periodic repair axis prediction."""
    from aria.core.guidance_deep_traces import trace_param_alternatives

    ds = GuidanceDataset(name="periodic_axis", label_names=AXIS_NAMES)
    label_to_idx = {n: i for i, n in enumerate(AXIS_NAMES)}

    for task_id, demos in task_demos_pairs:
        ep = trace_param_alternatives(task_id, demos)
        # Find verified periodic repair alternatives
        winner_axis = None
        for a in ep.alternatives:
            if a.family == "periodic_repair" and a.verified:
                winner_axis = a.param_set.get("axis")
                break
        if winner_axis is None or winner_axis not in label_to_idx:
            continue

        features = extract_perception_features(demos)
        ds.examples.append(GuidanceExample(
            features=features_to_array(features),
            label=label_to_idx[winner_axis],
            label_name=winner_axis,
            candidates=AXIS_NAMES,
            task_id=task_id,
        ))

    return ds


def build_periodic_period_dataset(
    task_demos_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> GuidanceDataset:
    """Build dataset for periodic repair period prediction."""
    from aria.core.guidance_deep_traces import trace_param_alternatives

    ds = GuidanceDataset(name="periodic_period", label_names=PERIOD_NAMES)
    label_to_idx = {n: i for i, n in enumerate(PERIOD_NAMES)}

    for task_id, demos in task_demos_pairs:
        ep = trace_param_alternatives(task_id, demos)
        winner_period = None
        for a in ep.alternatives:
            if a.family == "periodic_repair" and a.verified:
                winner_period = str(a.param_set.get("period"))
                break
        if winner_period is None or winner_period not in label_to_idx:
            continue

        features = extract_perception_features(demos)
        ds.examples.append(GuidanceExample(
            features=features_to_array(features),
            label=label_to_idx[winner_period],
            label_name=winner_period,
            candidates=PERIOD_NAMES,
            task_id=task_id,
        ))

    return ds


def build_transform_dataset(
    task_demos_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> GuidanceDataset:
    """Build dataset for grid transform choice prediction."""
    from aria.core.guidance_deep_traces import trace_param_alternatives

    ds = GuidanceDataset(name="transform_choice", label_names=TRANSFORM_NAMES)
    label_to_idx = {n: i for i, n in enumerate(TRANSFORM_NAMES)}

    # Map param_set to label name
    def _transform_label(ps: dict) -> str | None:
        if "degrees" in ps:
            return f"rotate_{ps['degrees']}"
        if "axis" in ps:
            return f"reflect_{ps['axis']}"
        if ps.get("transform") == "transpose":
            return "transpose"
        return None

    for task_id, demos in task_demos_pairs:
        ep = trace_param_alternatives(task_id, demos)
        winner_label = None
        for a in ep.alternatives:
            if a.family == "grid_transform" and a.verified:
                winner_label = _transform_label(a.param_set)
                break
        if winner_label is None or winner_label not in label_to_idx:
            continue

        features = extract_perception_features(demos)
        ds.examples.append(GuidanceExample(
            features=features_to_array(features),
            label=label_to_idx[winner_label],
            label_name=winner_label,
            candidates=TRANSFORM_NAMES,
            task_id=task_id,
        ))

    return ds


def build_lane_ranking_dataset(
    task_demos_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> GuidanceDataset:
    """Build dataset for lane/family ranking prediction."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve

    ds = GuidanceDataset(name="lane_ranking", label_names=LANE_NAMES)
    label_to_idx = {n: i for i, n in enumerate(LANE_NAMES)}

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    for task_id, demos in task_demos_pairs:
        result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)
        if not result.solved:
            continue

        # Determine winning family from the winning graph
        winning_family = None
        for attempt in result.attempts:
            if attempt.verified:
                winning_family = attempt.graph.metadata.get("family", "")
                break

        if not winning_family or winning_family not in label_to_idx:
            continue

        features = extract_perception_features(demos)
        ds.examples.append(GuidanceExample(
            features=features_to_array(features),
            label=label_to_idx[winning_family],
            label_name=winning_family,
            candidates=LANE_NAMES,
            task_id=task_id,
        ))

    return ds


# ---------------------------------------------------------------------------
# Dataset IO
# ---------------------------------------------------------------------------


def save_dataset(ds: GuidanceDataset, path: str | Path) -> None:
    """Save dataset as npz + metadata json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    X, y = ds.to_arrays()
    np.savez(p.with_suffix(".npz"), X=X, y=y)
    meta = ds.to_dict()
    meta["feature_names"] = FEATURE_NAMES
    with open(p.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
