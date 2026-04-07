"""Learned sketch proposer — predicts bounded symbolic hypotheses from demos.

Proposes factored sketch tuples: (decomposition, scope, selector, action, depth).
The symbolic system instantiates and verifies them.

No LLM. No end-to-end grid prediction. Model outputs are inspectable symbolic factors.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Sketch factor space
# ---------------------------------------------------------------------------

DECOMP_TYPES = ("full_grid", "object", "region", "frame", "partition", "mask")
SCOPE_TYPES = ("global", "object", "bbox", "frame_interior", "partition_cell", "enclosed", "mask_bbox")
SELECTOR_TYPES = ("none", "local_rule", "cell_property", "object_predicate", "marker", "size_selector")
ACTION_TYPES = ("recolor", "fill", "erase", "transform", "tile_render", "extract", "repair", "relocate", "compose")
DEPTH_TYPES = ("1", "2", "3")

ALL_FACTOR_NAMES = ("decomp", "scope", "selector", "action", "depth")
ALL_FACTOR_SPACES = (DECOMP_TYPES, SCOPE_TYPES, SELECTOR_TYPES, ACTION_TYPES, DEPTH_TYPES)


# ---------------------------------------------------------------------------
# Cross-demo feature extraction (reused from guidance_proposer, extended)
# ---------------------------------------------------------------------------


def extract_sketch_features(demos: tuple[DemoPair, ...]) -> np.ndarray:
    """Extract cross-demo features for sketch prediction."""
    from aria.core.grid_perception import perceive_grid
    from aria.decomposition import detect_bg

    n = len(demos)
    if n == 0:
        return np.zeros(50, dtype=np.float64)

    states = [perceive_grid(d.input) for d in demos]

    same_dims = float(all(d.input.shape == d.output.shape for d in demos))
    in_rows = np.mean([s.dims[0] for s in states])
    in_cols = np.mean([s.dims[1] for s in states])
    out_rows = np.mean([d.output.shape[0] for d in demos])
    out_cols = np.mean([d.output.shape[1] for d in demos])
    area_ratio = np.mean([d.output.size / max(d.input.size, 1) for d in demos])

    # Change analysis
    change_fracs = []
    bg_fracs = []
    nbg_to_bg_fracs = []
    nbg_to_nbg_fracs = []
    for d in demos:
        if d.input.shape != d.output.shape:
            change_fracs.append(1.0)
            bg_fracs.append(0.0)
            nbg_to_bg_fracs.append(0.0)
            nbg_to_nbg_fracs.append(0.0)
            continue
        bg = detect_bg(d.input)
        diff = d.input != d.output
        nc = int(np.sum(diff))
        total = d.input.size
        change_fracs.append(nc / max(total, 1))
        if nc > 0:
            rows, cols = np.where(diff)
            bg_ch = sum(1 for r, c in zip(rows, cols) if int(d.input[r, c]) == bg)
            nbg_bg = sum(1 for r, c in zip(rows, cols)
                         if int(d.input[r, c]) != bg and int(d.output[r, c]) == bg)
            nbg_nbg = sum(1 for r, c in zip(rows, cols)
                          if int(d.input[r, c]) != bg and int(d.output[r, c]) != bg)
            bg_fracs.append(bg_ch / nc)
            nbg_to_bg_fracs.append(nbg_bg / nc)
            nbg_to_nbg_fracs.append(nbg_nbg / nc)
        else:
            bg_fracs.append(0.0)
            nbg_to_bg_fracs.append(0.0)
            nbg_to_nbg_fracs.append(0.0)

    # Structure
    n_partitions = [1 if s.partition is not None else 0 for s in states]
    n_frames = [len(s.framed_regions) for s in states]
    n_obj4 = [len(s.objects.objects) for s in states]
    n_obj8 = [len(s.objects8.objects) for s in states]
    n_zones = [len(s.zones) for s in states]
    n_colors = [len(s.palette) for s in states]

    # Conservation check (relocation signature)
    conservation = 0.0
    if same_dims > 0.5:
        d = demos[0]
        bg = detect_bg(d.input)
        diff = d.input != d.output
        if np.any(diff):
            erased = Counter()
            added = Counter()
            for r, c in zip(*np.where(diff)):
                ic, oc = int(d.input[r, c]), int(d.output[r, c])
                if ic != bg:
                    erased[ic] += 1
                if oc != bg:
                    added[oc] += 1
            conservation = float(erased == added and len(erased) > 0)

    # Solid marker detection
    has_marker = 0.0
    marker_area_frac = 0.0
    d0 = demos[0]
    bg0 = detect_bg(d0.input)
    for color in range(10):
        if color == bg0:
            continue
        mask = d0.input == color
        if not np.any(mask):
            continue
        pos = np.argwhere(mask)
        r0, c0 = pos.min(axis=0)
        r1, c1 = pos.max(axis=0)
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if rh * rw == int(np.sum(mask)) and rh * rw >= 4:
            has_marker = 1.0
            marker_area_frac = rh * rw / d0.input.size
            break

    features = np.array([
        same_dims,
        float(in_rows), float(in_cols),
        float(out_rows), float(out_cols),
        float(area_ratio),
        float(np.mean(change_fracs)),
        float(np.std(change_fracs)),
        float(np.mean(bg_fracs)),
        float(np.mean(nbg_to_bg_fracs)),
        float(np.mean(nbg_to_nbg_fracs)),
        float(np.mean(n_partitions)),
        float(np.mean(n_frames)),
        float(np.mean(n_obj4)),
        float(np.mean(n_obj8)),
        float(np.mean(n_zones)),
        float(np.mean(n_colors)),
        float(all(p > 0 for p in n_partitions)),
        float(all(f > 0 for f in n_frames)),
        float(np.mean(n_obj4) > 10),
        float(np.std(n_obj4)),
        conservation,
        has_marker,
        marker_area_frac,
        float(area_ratio < 0.3),
        float(area_ratio > 1.5),
        float(np.mean(change_fracs) < 0.05),
        float(np.mean(change_fracs) > 0.3),
        float(n),
        # Color map consistency
        float(_color_map_consistent(demos, same_dims > 0.5)),
        # Size ratios
        float(np.std([d.output.shape[0] for d in demos])),
        float(np.std([d.output.shape[1] for d in demos])),
        float(in_rows * in_cols),
        float(out_rows * out_cols),
        float(max(n_obj4) - min(n_obj4)),
        float(np.mean(n_obj4) <= 3),
        float(np.mean(n_frames) >= 2),
        float(np.mean(n_zones) >= 4),
        float(np.mean(nbg_to_bg_fracs) > 0.5),
        float(conservation and np.mean(change_fracs) < 0.15),
    ], dtype=np.float64)

    # Pad to fixed size
    if len(features) < 50:
        features = np.pad(features, (0, 50 - len(features)))
    return features[:50]


def _color_map_consistent(demos, same_dims):
    if not same_dims:
        return 0.0
    from aria.decomposition import detect_bg
    cm = {}
    for d in demos:
        if d.input.shape != d.output.shape:
            return 0.0
        for r in range(d.input.shape[0]):
            for c in range(d.input.shape[1]):
                ic, oc = int(d.input[r, c]), int(d.output[r, c])
                if ic != oc:
                    if ic in cm and cm[ic] != oc:
                        return 0.0
                    cm[ic] = oc
    return 1.0 if cm else 0.0


N_FEATURES = 50


# ---------------------------------------------------------------------------
# Sketch label extraction from solved tasks
# ---------------------------------------------------------------------------


def _mechanism_to_sketch(mechanism: str) -> tuple[str, str, str, str, str]:
    """Map solved mechanism to sketch factors."""
    m = mechanism
    if m.startswith("render:tiled"):
        return ("full_grid", "global", "none", "tile_render", "1")
    if m.startswith("render:geometric"):
        return ("full_grid", "global", "none", "transform", "1")
    if m.startswith("render:global_color"):
        return ("full_grid", "global", "none", "recolor", "1")
    if m.startswith("render:zone_summary"):
        return ("partition", "partition_cell", "cell_property", "extract", "1")
    if m.startswith("render:fill_enclosed"):
        return ("region", "enclosed", "none", "fill", "1")
    if m.startswith("render:mask_repair"):
        return ("mask", "mask_bbox", "marker", "repair", "1")
    if m.startswith("render:partition_cell"):
        return ("partition", "partition_cell", "cell_property", "extract", "1")
    if m.startswith("render:scene_program"):
        return ("full_grid", "global", "none", "compose", "2")
    if m.startswith("deriv:"):
        return ("object", "bbox", "size_selector", "extract", "1")
    if m.startswith("size:"):
        return ("object", "bbox", "size_selector", "extract", "1")
    if m.startswith("local_rule:set_bg") or m.startswith("local_rule:set_color"):
        return ("full_grid", "global", "local_rule", "erase", "1")
    if m.startswith("local_rule:set_dominant"):
        return ("full_grid", "global", "local_rule", "recolor", "1")
    if m.startswith("pipeline:canvas"):
        return ("full_grid", "global", "none", "tile_render", "1")
    if m.startswith("pipeline:object_movement"):
        return ("object", "object", "object_predicate", "relocate", "1")
    if m.startswith("pipeline:framed_periodic"):
        return ("frame", "frame_interior", "none", "repair", "1")
    return ("full_grid", "global", "none", "compose", "1")


# ---------------------------------------------------------------------------
# Multi-head factor model
# ---------------------------------------------------------------------------


@dataclass
class SketchProposerModel:
    """Multi-head model: one softmax head per factor."""
    heads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    # factor_name -> (W, b, mean, std) for softmax classification

    def predict_factors(self, features: np.ndarray) -> dict[str, list[tuple[str, float]]]:
        """Predict top-k factor values for each factor dimension."""
        results = {}
        for fi, (fname, fspace) in enumerate(zip(ALL_FACTOR_NAMES, ALL_FACTOR_SPACES)):
            W, b, mean, std = self.heads[fname]
            Xn = (features - mean) / std
            logits = Xn @ W + b
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            order = np.argsort(-probs)
            results[fname] = [(fspace[i], float(probs[i])) for i in order]
        return results

    def top_sketches(self, features: np.ndarray, k: int = 10) -> list[tuple[tuple[str, ...], float]]:
        """Generate top-k sketch tuples by combining top factor predictions."""
        factor_preds = self.predict_factors(features)

        # Take top-2 per factor, combine greedily
        sketches = []
        top_per_factor = {fn: preds[:3] for fn, preds in factor_preds.items()}

        # Greedy: top-1 across all factors
        top1 = tuple(preds[0][0] for preds in factor_preds.values())
        score1 = np.prod([preds[0][1] for preds in factor_preds.values()])
        sketches.append((top1, score1))

        # Vary each factor independently
        for fi, fname in enumerate(ALL_FACTOR_NAMES):
            for val, prob in top_per_factor[fname][1:]:
                sketch = list(top1)
                sketch[fi] = val
                score = score1 * (prob / max(factor_preds[fname][0][1], 1e-6))
                sketches.append((tuple(sketch), score))

        sketches.sort(key=lambda x: -x[1])
        return sketches[:k]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for fname, (W, b, mean, std) in self.heads.items():
            arrays[f"{fname}_W"] = W
            arrays[f"{fname}_b"] = b
            arrays[f"{fname}_mean"] = mean
            arrays[f"{fname}_std"] = std
        np.savez(p.with_suffix(".npz"), **arrays)

    @staticmethod
    def load(path: str | Path) -> SketchProposerModel:
        p = Path(path)
        arrays = np.load(p.with_suffix(".npz"))
        heads = {}
        for fname in ALL_FACTOR_NAMES:
            heads[fname] = (
                arrays[f"{fname}_W"],
                arrays[f"{fname}_b"],
                arrays[f"{fname}_mean"],
                arrays[f"{fname}_std"],
            )
        return SketchProposerModel(heads=heads)


def train_sketch_proposer(
    task_demo_pairs: list[tuple[str, tuple[DemoPair, ...]]],
    lr: float = 0.05,
    epochs: int = 300,
) -> SketchProposerModel:
    """Train from solved tasks."""
    from aria.core.output_stage1 import infer_output_stage1_spec, compile_stage1_program
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.core.local_rule_synth import synthesize_local_rule

    verifier = ARCVerifier()

    # Collect training examples
    examples: list[tuple[np.ndarray, dict[str, int]]] = []

    for task_id, demos in task_demo_pairs:
        # Determine mechanism
        mechanism = None

        stage1 = infer_output_stage1_spec(demos)
        if stage1 is not None:
            prog = compile_stage1_program(stage1)
            if prog is not None and verifier.verify(prog, demos).passed:
                if stage1.render_spec:
                    mechanism = f"render:{stage1.render_spec.get('kind', '?')}"
                elif stage1.derivation_spec:
                    mechanism = f"deriv:{stage1.derivation_spec.candidate_kind}/{stage1.derivation_spec.relation}"
                else:
                    mechanism = f"size:{stage1.size_spec.mode}"

        if mechanism is None and all(d.input.shape == d.output.shape for d in demos):
            rule = synthesize_local_rule(demos)
            if rule:
                mechanism = f"local_rule:{rule.action}"

        if mechanism is None:
            fitter = ARCFitter()
            specializer = ARCSpecializer()
            compiler = ARCCompiler()
            result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)
            if result.solved:
                for att in result.attempts:
                    if att.verified:
                        mechanism = f"pipeline:{att.graph.metadata.get('family', '?')}"
                        break

        if mechanism is None:
            continue

        sketch = _mechanism_to_sketch(mechanism)
        features = extract_sketch_features(demos)

        label_dict = {}
        for fi, (fname, fspace) in enumerate(zip(ALL_FACTOR_NAMES, ALL_FACTOR_SPACES)):
            val = sketch[fi]
            if val in fspace:
                label_dict[fname] = fspace.index(val)
            else:
                label_dict[fname] = 0

        examples.append((features, label_dict))

    if not examples:
        # Empty model
        heads = {}
        for fname, fspace in zip(ALL_FACTOR_NAMES, ALL_FACTOR_SPACES):
            d = N_FEATURES
            k = len(fspace)
            heads[fname] = (np.zeros((d, k)), np.zeros(k), np.zeros(d), np.ones(d))
        return SketchProposerModel(heads=heads)

    # Train one head per factor
    X = np.stack([f for f, _ in examples])
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    heads = {}
    for fname, fspace in zip(ALL_FACTOR_NAMES, ALL_FACTOR_SPACES):
        k = len(fspace)
        y = np.array([labels[fname] for _, labels in examples])
        W = np.zeros((N_FEATURES, k))
        b = np.zeros(k)

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

        heads[fname] = (W, b, mean, std)

    return SketchProposerModel(heads=heads)


# ---------------------------------------------------------------------------
# Sketch-guided symbolic search
# ---------------------------------------------------------------------------


def sketch_guided_solve(
    demos: tuple[DemoPair, ...],
    model: SketchProposerModel,
    task_id: str = "",
) -> dict[str, Any]:
    """Use learned sketch proposer to guide symbolic search."""
    import numpy as np
    from aria.core.output_stage1 import infer_output_stage1_spec, compile_stage1_program
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.core.local_rule_synth import synthesize_local_rule, apply_rule
    from aria.decomposition import detect_bg

    verifier = ARCVerifier()
    features = extract_sketch_features(demos)
    sketches = model.top_sketches(features, k=10)

    trace = {
        "task_id": task_id,
        "top_sketches": [(s, round(sc, 4)) for s, sc in sketches[:5]],
        "solved": False,
        "winning_sketch": None,
        "winning_mechanism": None,
    }

    # Phase 1: Stage-1 (always try, independent of sketch)
    stage1 = infer_output_stage1_spec(demos)
    if stage1 is not None:
        prog = compile_stage1_program(stage1)
        if prog is not None and verifier.verify(prog, demos).passed:
            trace["solved"] = True
            trace["winning_mechanism"] = "stage1"
            return trace

    # Phase 2: Local-rule synthesis (if sketch suggests it)
    top_actions = [s[3] for s, _ in sketches[:3]]
    if any(a in ("erase", "recolor") for a in top_actions):
        if all(d.input.shape == d.output.shape for d in demos):
            rule = synthesize_local_rule(demos)
            if rule:
                bg = detect_bg(demos[0].input)
                all_ok = all(np.array_equal(apply_rule(d.input, detect_bg(d.input), rule), d.output) for d in demos)
                if all_ok:
                    trace["solved"] = True
                    trace["winning_mechanism"] = f"local_rule:{rule.description}"
                    return trace

    # Phase 3: Scene-program proposals
    from aria.core.scene_propose import propose_scene_programs, verify_scene_proposals
    proposals = propose_scene_programs(demos)
    if proposals:
        for name, prog, verified in verify_scene_proposals(demos, proposals):
            if verified:
                trace["solved"] = True
                trace["winning_mechanism"] = f"scene:{name}"
                return trace

    # Phase 4: Full pipeline
    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=task_id)
    if result.solved:
        trace["solved"] = True
        trace["winning_mechanism"] = "pipeline"
        return trace

    return trace
