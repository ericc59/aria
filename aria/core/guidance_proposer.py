"""Learned cross-demo program proposer for ARC-2.

Looks at all demos jointly and ranks likely decomposition/skeleton families
to guide bounded symbolic search. Does NOT emit final grids.

Target space (small, inspectable):
- decomposition_type: object, frame, partition, zone, mask_repair, derivation, render
- skeleton: select_apply, select_fill, select_transform, correspond_apply,
            combine_stamp, mask_repair, derivation_clone, tiled_render,
            color_map, zone_summary, geometric_transform

The model uses cross-demo perception features (consistency signals,
change patterns, structure counts) to rank families.
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
# Target space
# ---------------------------------------------------------------------------

DECOMP_TYPES = (
    "derivation",       # output = direct extraction from input
    "render_transform",  # output = geometric/tiled transform of input
    "color_map",         # output = color substitution
    "fill_enclosed",     # output = input + filled enclosed regions
    "zone_summary",      # output = per-zone property summary
    "partition_edit",    # output = partition-cell operation
    "object_edit",       # output = object-level transform
    "frame_edit",        # output = frame-interior operation
    "mask_repair",       # output = repaired masked region
    "scene_program",     # output = short scene program
)

SKELETON_TYPES = (
    "derivation_clone",
    "derivation_interior",
    "tiled_input",
    "geometric_transform",
    "global_color_map",
    "scoped_color_map",
    "fill_enclosed",
    "zone_summary_grid",
    "partition_cell_select",
    "partition_per_cell",
    "partition_combine",
    "mask_repair",
    "object_transform",
    "frame_interior_edit",
    "bbox_extract",
)


# ---------------------------------------------------------------------------
# Cross-demo feature extraction
# ---------------------------------------------------------------------------


def extract_cross_demo_features(demos: tuple[DemoPair, ...]) -> np.ndarray:
    """Extract a feature vector from ALL demos jointly.

    Unlike single-demo features, this captures cross-demo consistency
    signals that are critical for ARC rule inference.
    """
    from aria.core.grid_perception import perceive_grid
    from aria.decomposition import detect_bg

    n = len(demos)
    if n == 0:
        return np.zeros(40, dtype=np.float64)

    states = [perceive_grid(d.input) for d in demos]

    # Basic dims
    same_dims = float(all(d.input.shape == d.output.shape for d in demos))
    in_rows = [s.dims[0] for s in states]
    in_cols = [s.dims[1] for s in states]
    out_rows = [d.output.shape[0] for d in demos]
    out_cols = [d.output.shape[1] for d in demos]
    dims_consistent = float(len(set(zip(in_rows, in_cols))) == 1)
    out_dims_consistent = float(len(set(zip(out_rows, out_cols))) == 1)

    # Change patterns
    change_fracs = []
    bg_change_fracs = []
    for d in demos:
        if d.input.shape == d.output.shape:
            diff = d.input != d.output
            total = d.input.size
            n_changed = int(np.sum(diff))
            change_fracs.append(n_changed / max(total, 1))
            bg = detect_bg(d.input)
            if np.any(diff):
                bg_changes = sum(1 for r, c in zip(*np.where(diff)) if int(d.input[r, c]) == bg)
                bg_change_fracs.append(bg_changes / max(n_changed, 1))
            else:
                bg_change_fracs.append(0.0)
        else:
            change_fracs.append(1.0)  # different dims = fully changed
            bg_change_fracs.append(0.0)

    # Structure counts (consistent across demos?)
    n_partitions = [1 if s.partition is not None else 0 for s in states]
    n_frames = [len(s.framed_regions) for s in states]
    n_obj4 = [len(s.objects.objects) for s in states]
    n_zones = [len(s.zones) for s in states]
    n_colors = [len(s.palette) for s in states]

    partition_consistent = float(len(set(n_partitions)) == 1 and n_partitions[0] > 0)
    frame_consistent = float(len(set([min(x, 1) for x in n_frames])) == 1 and n_frames[0] > 0)
    obj_count_consistent = float(max(n_obj4) - min(n_obj4) <= 2)

    # Output analysis
    area_ratios = [d.output.size / max(d.input.size, 1) for d in demos]
    output_smaller = float(all(r < 0.5 for r in area_ratios))
    output_same = float(all(abs(r - 1.0) < 0.01 for r in area_ratios))

    # Color map consistency (only for same-dims tasks)
    color_map_consistent = 0.0
    global_map: dict[int, int] = {}
    n_color_swaps = 0
    if same_dims > 0.5:
        color_map_consistent = 1.0
        for d in demos:
            if d.input.shape != d.output.shape:
                continue
            for r in range(d.input.shape[0]):
                for c in range(d.input.shape[1]):
                    ic, oc = int(d.input[r, c]), int(d.output[r, c])
                    if ic != oc:
                        if ic in global_map and global_map[ic] != oc:
                            color_map_consistent = 0.0
                        global_map[ic] = oc
        n_color_swaps = len(global_map)

    # Symmetry signals
    has_legend = float(any(s.legend is not None for s in states))

    # Mask detection
    has_solid_marker = 0.0
    for d in demos:
        bg = detect_bg(d.input)
        for color in range(10):
            if color == bg:
                continue
            mask = d.input == color
            if not np.any(mask):
                continue
            pos = np.argwhere(mask)
            r0, c0 = pos.min(axis=0)
            r1, c1 = pos.max(axis=0)
            rh, rw = r1 - r0 + 1, c1 - c0 + 1
            if rh * rw == int(np.sum(mask)) and rh * rw >= 4:
                has_solid_marker = 1.0
                break
        break  # check first demo only

    features = np.array([
        same_dims,                          # 0
        dims_consistent,                    # 1
        out_dims_consistent,                # 2
        np.mean(change_fracs),              # 3
        np.std(change_fracs),               # 4
        np.mean(bg_change_fracs),           # 5
        partition_consistent,               # 6
        frame_consistent,                   # 7
        obj_count_consistent,               # 8
        np.mean(n_partitions),              # 9
        np.mean(n_frames),                  # 10
        np.mean(n_obj4),                    # 11
        np.mean(n_zones),                   # 12
        np.mean(n_colors),                  # 13
        np.mean(area_ratios),               # 14
        output_smaller,                     # 15
        output_same,                        # 16
        color_map_consistent,               # 17
        float(n_color_swaps),               # 18
        has_legend,                         # 19
        has_solid_marker,                   # 20
        float(n),                           # 21
        np.mean(in_rows),                   # 22
        np.mean(in_cols),                   # 23
        np.mean(out_rows),                  # 24
        np.mean(out_cols),                  # 25
        np.std(in_rows),                    # 26
        np.std(in_cols),                    # 27
        float(max(n_obj4) - min(n_obj4)),   # 28
        float(np.mean(n_obj4) > 10),        # 29
        float(np.mean(change_fracs) < 0.05),# 30
        float(np.mean(change_fracs) > 0.3), # 31
        float(np.mean(bg_change_fracs) > 0.8), # 32
        float(all(nf > 0 for nf in n_frames)), # 33
        float(any(s.partition is not None and s.partition.n_rows * s.partition.n_cols >= 4 for s in states)), # 34
        float(np.mean(area_ratios) < 0.3),  # 35
        float(np.mean(area_ratios) > 1.5),  # 36
        float(n_color_swaps <= 2),           # 37
        float(n_color_swaps == 0),           # 38
        float(has_solid_marker and not same_dims), # 39
    ], dtype=np.float64)

    return features


FEATURE_NAMES = [
    "same_dims", "dims_consistent", "out_dims_consistent",
    "mean_change_frac", "std_change_frac", "mean_bg_change_frac",
    "partition_consistent", "frame_consistent", "obj_count_consistent",
    "mean_partitions", "mean_frames", "mean_obj4", "mean_zones", "mean_colors",
    "mean_area_ratio", "output_smaller", "output_same",
    "color_map_consistent", "n_color_swaps", "has_legend", "has_solid_marker",
    "n_demos", "mean_in_rows", "mean_in_cols", "mean_out_rows", "mean_out_cols",
    "std_in_rows", "std_in_cols", "obj_count_range", "many_objects",
    "sparse_change", "dense_change", "mostly_bg_change", "all_have_frames",
    "has_large_partition", "output_much_smaller", "output_much_larger",
    "few_color_swaps", "no_change_at_all", "marker_and_diff_dims",
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Dataset builder for proposer training
# ---------------------------------------------------------------------------


@dataclass
class ProposerExample:
    features: np.ndarray
    skeleton_label: int
    skeleton_name: str
    task_id: str = ""


@dataclass
class ProposerDataset:
    name: str
    skeleton_names: tuple[str, ...]
    examples: list[ProposerExample] = field(default_factory=list)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.examples:
            return np.zeros((0, N_FEATURES)), np.zeros(0, dtype=int)
        X = np.stack([e.features for e in self.examples])
        y = np.array([e.skeleton_label for e in self.examples], dtype=int)
        return X, y


def _family_to_skeleton(family: str) -> str:
    """Map solved-task family to skeleton label."""
    mapping = {
        "tiled_input_pattern": "tiled_input",
        "geometric_transform": "geometric_transform",
        "global_color_map": "global_color_map",
        "zone_summary_grid": "zone_summary_grid",
        "fill_enclosed": "fill_enclosed",
        "partition_cell_select": "partition_cell_select",
        "mask_repair": "mask_repair",
    }
    if family in mapping:
        return mapping[family]
    if family.startswith("scene_program"):
        if "color_map" in family:
            return "scoped_color_map"
        if "per_cell" in family:
            return "partition_per_cell"
        return "scene_program"
    if family.startswith("deriv:") and "clone" in family:
        return "derivation_clone"
    if family.startswith("deriv:") and "interior" in family:
        return "derivation_interior"
    if family.startswith("deriv:") and "border" in family:
        return "derivation_interior"
    if family.startswith("size:"):
        return "bbox_extract"
    return "scene_program"


def build_proposer_dataset(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
) -> ProposerDataset:
    """Build training dataset from solved tasks."""
    from aria.core.output_stage1 import infer_output_stage1_spec, compile_stage1_program
    from aria.core.arc import ARCVerifier

    verifier = ARCVerifier()
    skeleton_to_idx = {s: i for i, s in enumerate(SKELETON_TYPES)}
    ds = ProposerDataset(name="proposer", skeleton_names=SKELETON_TYPES)

    for task_id, demos in task_demo_pairs:
        stage1 = infer_output_stage1_spec(demos)
        if stage1 is None:
            continue
        prog = compile_stage1_program(stage1)
        if prog is None:
            continue
        vr = verifier.verify(prog, demos)
        if not vr.passed:
            continue

        # Determine family
        if stage1.render_spec:
            family = stage1.render_spec.get("kind", "?")
        elif stage1.derivation_spec:
            family = f"deriv:{stage1.derivation_spec.candidate_kind}/{stage1.derivation_spec.relation}"
        else:
            family = f"size:{stage1.size_spec.mode}"

        skeleton = _family_to_skeleton(family)
        if skeleton not in skeleton_to_idx:
            continue

        features = extract_cross_demo_features(demos)
        ds.examples.append(ProposerExample(
            features=features,
            skeleton_label=skeleton_to_idx[skeleton],
            skeleton_name=skeleton,
            task_id=task_id,
        ))

    return ds


# ---------------------------------------------------------------------------
# Model: linear softmax ranker over skeletons
# ---------------------------------------------------------------------------


@dataclass
class ProposerModel:
    """Learned proposer: ranks skeleton families from cross-demo features."""
    weights: np.ndarray     # (N_FEATURES, N_SKELETONS)
    bias: np.ndarray        # (N_SKELETONS,)
    mean: np.ndarray        # (N_FEATURES,) normalization
    std: np.ndarray         # (N_FEATURES,) normalization
    skeleton_names: tuple[str, ...]

    def rank_skeletons(self, features: np.ndarray) -> list[tuple[str, float]]:
        """Return skeleton names ranked by predicted probability."""
        Xn = (features - self.mean) / self.std
        logits = Xn @ self.weights + self.bias
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        order = np.argsort(-probs)
        return [(self.skeleton_names[i], float(probs[i])) for i in order]

    def top_k(self, features: np.ndarray, k: int = 5) -> list[str]:
        """Return top-k skeleton names."""
        return [name for name, _ in self.rank_skeletons(features)[:k]]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".npz"), weights=self.weights, bias=self.bias,
                 mean=self.mean, std=self.std)
        with open(p.with_suffix(".json"), "w") as f:
            json.dump({"skeleton_names": list(self.skeleton_names)}, f)

    @staticmethod
    def load(path: str | Path) -> ProposerModel:
        p = Path(path)
        arrays = np.load(p.with_suffix(".npz"))
        with open(p.with_suffix(".json")) as f:
            meta = json.load(f)
        return ProposerModel(
            weights=arrays["weights"], bias=arrays["bias"],
            mean=arrays["mean"], std=arrays["std"],
            skeleton_names=tuple(meta["skeleton_names"]),
        )


def train_proposer(dataset: ProposerDataset, lr: float = 0.05, epochs: int = 300) -> ProposerModel:
    """Train the proposer model from solved-task examples."""
    X, y = dataset.to_arrays()
    n_classes = len(dataset.skeleton_names)

    if len(X) == 0:
        return ProposerModel(
            weights=np.zeros((N_FEATURES, n_classes)),
            bias=np.zeros(n_classes),
            mean=np.zeros(N_FEATURES),
            std=np.ones(N_FEATURES),
            skeleton_names=dataset.skeleton_names,
        )

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    W = np.zeros((N_FEATURES, n_classes))
    b = np.zeros(n_classes)

    for _ in range(epochs):
        logits = Xn @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        targets = np.zeros_like(probs)
        targets[np.arange(len(y)), y] = 1.0
        grad = probs - targets
        W -= lr * (Xn.T @ grad) / len(y)
        b -= lr * grad.mean(axis=0)

    return ProposerModel(
        weights=W, bias=b, mean=mean, std=std,
        skeleton_names=dataset.skeleton_names,
    )


# ---------------------------------------------------------------------------
# Integration: proposer-guided symbolic search
# ---------------------------------------------------------------------------


def proposer_guided_solve(
    demos: tuple[DemoPair, ...],
    model: ProposerModel,
    task_id: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """Use the learned proposer to guide symbolic search.

    Returns a trace dict with:
    - ranked_skeletons: model's ranking
    - attempted: which skeletons were instantiated
    - solved: whether any verified
    - winning_skeleton: if solved
    """
    import numpy as np
    from aria.core.output_stage1 import infer_output_stage1_spec, compile_stage1_program
    from aria.core.arc import ARCVerifier

    features = extract_cross_demo_features(demos)
    ranking = model.rank_skeletons(features)
    top_names = [name for name, _ in ranking[:top_k]]

    verifier = ARCVerifier()
    trace = {
        "task_id": task_id,
        "ranked_skeletons": [(n, round(p, 4)) for n, p in ranking[:top_k]],
        "attempted": [],
        "solved": False,
        "winning_skeleton": None,
    }

    # Stage 1: render/derivation/mask_repair
    stage1 = infer_output_stage1_spec(demos)
    if stage1 is not None:
        prog = compile_stage1_program(stage1)
        if prog is not None:
            vr = verifier.verify(prog, demos)
            if vr.passed:
                trace["solved"] = True
                if stage1.render_spec:
                    trace["winning_skeleton"] = stage1.render_spec.get("kind", "stage1")
                elif stage1.derivation_spec:
                    trace["winning_skeleton"] = "derivation"
                else:
                    trace["winning_skeleton"] = "stage1_compile"
                return trace

    # Scene-program proposals (guided by ranking)
    from aria.core.scene_propose import propose_scene_programs, verify_scene_proposals

    proposals = propose_scene_programs(demos)
    if proposals:
        def _skeleton_priority(name: str) -> int:
            for i, skel in enumerate(top_names):
                if skel in name or name in skel:
                    return i
            return len(top_names)

        proposals_sorted = sorted(proposals, key=lambda p: _skeleton_priority(p[0]))

        for name, prog, verified in verify_scene_proposals(demos, proposals_sorted):
            trace["attempted"].append(name)
            if verified:
                trace["solved"] = True
                trace["winning_skeleton"] = name
                return trace

    # Local rule synthesis
    from aria.core.local_rule_synth import synthesize_local_rule, apply_rule
    local_rule = synthesize_local_rule(demos, max_conjunction_size=2)
    if local_rule is not None:
        # Verify by applying to each demo
        from aria.decomposition import detect_bg as _detect_bg
        all_ok = True
        for d in demos:
            bg = _detect_bg(d.input)
            result = apply_rule(d.input, bg, local_rule)
            if not np.array_equal(result, d.output):
                all_ok = False
                break
        if all_ok:
            trace["solved"] = True
            trace["winning_skeleton"] = f"local_rule:{local_rule.description}"
            return trace

    # Seed-guided repair: fitter seeds + near-miss repair
    from aria.core.fitter_seeds import solve_from_seeds
    repaired, seeds = solve_from_seeds(demos, task_id=task_id)
    trace["seeds_collected"] = len(seeds)
    if repaired is not None:
        trace["solved"] = True
        trace["winning_skeleton"] = "seed_repair"
        return trace

    # Full pipeline: fitter → specialize → compile → verify
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler
    from aria.core.protocol import solve as core_solve

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()

    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)
    if result.solved:
        trace["solved"] = True
        trace["winning_skeleton"] = "static_pipeline"
        return trace

    return trace
