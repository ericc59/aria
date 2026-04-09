#!/usr/bin/env python3
"""Real ARC-2 factor evaluation — incremental, flushed output.

Run: python scripts/eval_factor_real.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

OUT_DIR = Path("/tmp/factor_eval")
OUT_DIR.mkdir(exist_ok=True)


def p(msg):
    print(msg, flush=True)


def main():
    from aria.datasets import get_dataset, iter_tasks, list_task_ids
    from aria.core.arc import solve_arc_task
    from aria.core.guidance_proposer import extract_cross_demo_features
    from aria.core.factor_labels import SKELETON_TO_FACTORS, SCENE_FAMILY_TO_FACTORS, FactorLabel
    from aria.factors import (
        FactorSet, Decomposition, Selector, Scope, Op, Correspondence, Depth,
        FACTOR_NAMES, FACTOR_ENUMS, is_compatible,
    )

    # ---------------------------------------------------------------
    # PHASE 1: Collect solved-task labels from v2-train
    # ---------------------------------------------------------------
    p("=" * 60)
    p("PHASE 1: Collecting solved-task factor labels")
    p("=" * 60)

    ds = get_dataset("v2-train")
    task_ids = list_task_ids(ds)
    p(f"Total v2-train tasks: {len(task_ids)}")

    features_list = []
    labels_list = []
    labeled_ids = []
    solve_count = 0
    family_dist = Counter()

    t0 = time.time()
    for i, (task_id, task) in enumerate(iter_tasks(ds)):
        demos = task.train
        try:
            result = solve_arc_task(demos, task_id=task_id, use_editor_search=False)
        except Exception:
            continue

        if not result.solved:
            continue
        solve_count += 1

        # Classify the solved program into factor labels
        label = _classify_solved(task_id, demos, result)
        if label is None:
            continue

        feats = extract_cross_demo_features(demos)
        features_list.append(feats)
        labels_list.append(label.factors)
        labeled_ids.append(task_id)
        family_dist[label.source] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            p(f"  {i+1}/{len(task_ids)}: {solve_count} solved, {len(labeled_ids)} labeled ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    p(f"\nPhase 1 done: {solve_count} solved, {len(labeled_ids)} labeled in {elapsed:.0f}s")
    p(f"Family distribution: {dict(family_dist.most_common())}")

    n = len(labeled_ids)
    if n < 3:
        p("ERROR: Too few labeled tasks. Cannot evaluate proposer.")
        return

    # Save labels
    np.save(OUT_DIR / "features.npy", np.stack(features_list))
    with open(OUT_DIR / "labels.json", "w") as f:
        json.dump({"task_ids": labeled_ids, "n": n, "families": dict(family_dist)}, f)

    # ---------------------------------------------------------------
    # PHASE 2: Per-factor class balance and LOO accuracy
    # ---------------------------------------------------------------
    p("\n" + "=" * 60)
    p("PHASE 2: Per-factor evaluation (leave-one-out)")
    p("=" * 60)

    from aria.core.factor_proposer import FactorHead, FactorProposer

    enum_lists = {name: list(FACTOR_ENUMS[name]) for name in FACTOR_NAMES}
    enum_to_idx = {
        name: {v: i for i, v in enumerate(vals)}
        for name, vals in enum_lists.items()
    }

    X = np.stack(features_list)

    for factor_name in FACTOR_NAMES:
        y = np.array([
            enum_to_idx[factor_name][getattr(fs, factor_name)]
            for fs in labels_list
        ], dtype=int)

        # Class balance
        counts = Counter(y.tolist())
        balance = {enum_lists[factor_name][k].value if hasattr(enum_lists[factor_name][k], 'value') else str(k): v
                   for k, v in sorted(counts.items(), key=lambda x: -x[1])}
        n_classes = len(enum_lists[factor_name])

        # LOO
        correct1 = 0
        correct3 = 0
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            head = FactorHead(factor_name=factor_name, n_classes=n_classes,
                              class_names=tuple(str(m) for m in enum_lists[factor_name]))
            head.fit(X_train, y_train, epochs=200)
            probs = head.predict_proba(X[i])
            if np.argmax(probs) == y[i]:
                correct1 += 1
            if y[i] in np.argsort(-probs)[:min(3, n_classes)]:
                correct3 += 1

        p(f"\n  {factor_name}: top1={correct1/n:.3f} top3={correct3/n:.3f} classes={n_classes}")
        p(f"    balance: {balance}")

    # ---------------------------------------------------------------
    # PHASE 3: Joint factor combo evaluation
    # ---------------------------------------------------------------
    p("\n" + "=" * 60)
    p("PHASE 3: Joint factor combo quality")
    p("=" * 60)

    proposer = FactorProposer()
    proposer.fit_from_labels(features_list, labels_list, epochs=300)

    joint_top1 = 0
    recall_at = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    not_found = 0
    ranks = []

    for i in range(n):
        true_fs = labels_list[i]
        ranked = proposer.top_k_factor_sets(features_list[i], k=50)
        predicted = [fs for fs, _ in ranked]

        rank = -1
        for j, fs in enumerate(predicted):
            if fs == true_fs:
                rank = j
                break

        if rank < 0:
            not_found += 1
        else:
            ranks.append(rank)
            for k in recall_at:
                if rank < k:
                    recall_at[k] += 1
        if rank == 0:
            joint_top1 += 1

    p(f"  Joint top-1: {joint_top1/n:.3f}")
    for k, v in sorted(recall_at.items()):
        p(f"  Recall@{k}: {v/n:.3f}")
    p(f"  Not found in top-50: {not_found}/{n}")
    if ranks:
        p(f"  Mean rank (when found): {np.mean(ranks):.1f}")

    # ---------------------------------------------------------------
    # PHASE 4: Factor interaction diagnostics
    # ---------------------------------------------------------------
    p("\n" + "=" * 60)
    p("PHASE 4: Factor interaction diagnostics")
    p("=" * 60)

    from aria.factors import _DECOMP_SELECTORS

    incompatible_top1 = 0
    decomp_sel_mismatch = 0

    for i in range(n):
        probs = proposer.predict_factor_probs(features_list[i])
        top1 = {}
        for name in FACTOR_NAMES:
            elist = list(FACTOR_ENUMS[name])
            top1[name] = elist[np.argmax(probs[name])]

        fs1 = FactorSet(**top1)
        if not is_compatible(fs1):
            incompatible_top1 += 1

        # Check decomp/selector compatibility
        if top1["selector"] not in _DECOMP_SELECTORS.get(top1["decomposition"], set()):
            decomp_sel_mismatch += 1

    p(f"  Incompatible top-1 combos: {incompatible_top1}/{n} ({incompatible_top1/n:.2%})")
    p(f"  Decomp-selector mismatches: {decomp_sel_mismatch}/{n} ({decomp_sel_mismatch/n:.2%})")

    # ---------------------------------------------------------------
    # PHASE 5: ARC-2 v2-eval comparison
    # ---------------------------------------------------------------
    p("\n" + "=" * 60)
    p("PHASE 5: v2-eval factorized vs flat-family comparison")
    p("=" * 60)

    from aria.core.scene_solve import infer_scene_programs, verify_scene_program
    from aria.core.factor_search import factor_composition_search
    from aria.core.grid_perception import perceive_grid

    ds_eval = get_dataset("v2-eval")
    eval_ids = list_task_ids(ds_eval)
    p(f"v2-eval tasks: {len(eval_ids)}")

    flat_solved = []
    factor_solved = []
    factor_only = []
    flat_only = []
    flat_candidates_total = 0
    factor_candidates_total = 0

    t0 = time.time()
    for i, (task_id, task) in enumerate(iter_tasks(ds_eval)):
        demos = task.train

        # Flat-family search
        try:
            flat_progs = infer_scene_programs(demos)
        except Exception:
            flat_progs = ()
        flat_candidates_total += len(flat_progs)

        flat_ok = False
        for prog in flat_progs:
            if verify_scene_program(prog, demos):
                flat_ok = True
                break

        # Factorized search
        try:
            factor_results = factor_composition_search(
                demos, max_candidates=50, max_programs=200,
            )
        except Exception:
            factor_results = []
        factor_candidates_total += len(factor_results)

        factor_ok = len(factor_results) > 0

        if flat_ok:
            flat_solved.append(task_id)
        if factor_ok:
            factor_solved.append(task_id)
        if factor_ok and not flat_ok:
            factor_only.append(task_id)
        if flat_ok and not factor_ok:
            flat_only.append(task_id)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            p(f"  {i+1}/{len(eval_ids)}: flat={len(flat_solved)} factor={len(factor_solved)} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    p(f"\nv2-eval done in {elapsed:.0f}s")
    p(f"  Flat solves: {len(flat_solved)}")
    p(f"  Factor solves: {len(factor_solved)}")
    p(f"  Factor-only: {len(factor_only)} {factor_only}")
    p(f"  Flat-only: {len(flat_only)} {flat_only}")
    p(f"  Avg flat candidates/task: {flat_candidates_total/max(len(eval_ids),1):.1f}")
    p(f"  Avg factor candidates/task: {factor_candidates_total/max(len(eval_ids),1):.3f}")

    # ---------------------------------------------------------------
    # PHASE 6: Summary
    # ---------------------------------------------------------------
    p("\n" + "=" * 60)
    p("FINAL SUMMARY")
    p("=" * 60)
    p(f"Labeled tasks: {n}")
    p(f"Incompatible top-1 rate: {incompatible_top1/n:.2%}")
    p(f"Joint top-1: {joint_top1/n:.3f}")
    p(f"Recall@10: {recall_at.get(10,0)/n:.3f}")
    p(f"v2-eval flat solves: {len(flat_solved)}")
    p(f"v2-eval factor solves: {len(factor_solved)}")
    p(f"v2-eval factor-only NEW solves: {len(factor_only)}")

    # Save full report
    report = {
        "n_labeled": n,
        "class_balance": dict(family_dist),
        "joint_top1": joint_top1 / n,
        "recall_at": {str(k): v / n for k, v in recall_at.items()},
        "incompatible_top1_rate": incompatible_top1 / n,
        "v2_eval_flat_solved": flat_solved,
        "v2_eval_factor_solved": factor_solved,
        "v2_eval_factor_only": factor_only,
        "v2_eval_flat_only": flat_only,
    }
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    p(f"\nFull report saved to {OUT_DIR / 'report.json'}")


def _classify_solved(task_id, demos, result):
    """Classify a solved task's result into factor labels."""
    from aria.core.factor_labels import SKELETON_TO_FACTORS, SCENE_FAMILY_TO_FACTORS, FactorLabel
    from aria.factors import FactorSet, Decomposition, Selector, Scope, Op, Correspondence, Depth

    prog = result.winning_program
    if prog is None:
        return None

    # Try to match program steps to scene families
    if hasattr(prog, 'steps') and hasattr(prog.steps[0] if prog.steps else None, 'op'):
        step_ops = [s.op.value for s in prog.steps]
        ops_str = "->".join(step_ops)

        if "select_entity" in ops_str:
            if "canonicalize" in ops_str:
                return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS.get(
                    "select_extract_transform",
                    FactorSet(Decomposition.OBJECT, Selector.OBJECT_SELECT,
                              Scope.OBJECT, Op.TRANSFORM, Correspondence.NONE, Depth.TWO)
                ), "select_extract_transform")
            elif "recolor" in ops_str:
                return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS.get(
                    "select_extract_colormap",
                    FactorSet(Decomposition.OBJECT, Selector.OBJECT_SELECT,
                              Scope.OBJECT, Op.RECOLOR, Correspondence.NONE, Depth.TWO)
                ), "select_extract_colormap")
            else:
                return FactorLabel(task_id, FactorSet(
                    Decomposition.OBJECT, Selector.OBJECT_SELECT,
                    Scope.OBJECT, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
                ), "select_extract")

        if "boolean_combine" in ops_str:
            return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS["boolean_combine"], "boolean_combine")

        if "fill_enclosed" in ops_str:
            return FactorLabel(task_id, SKELETON_TO_FACTORS["fill_enclosed"], "fill_enclosed")

        if "recolor_object" in ops_str:
            return FactorLabel(task_id, FactorSet(
                Decomposition.OBJECT, Selector.NONE,
                Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
            ), "recolor")

        if "for_each_entity" in ops_str:
            return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS.get(
                "per_object_operation",
                FactorSet(Decomposition.OBJECT, Selector.OBJECT_SELECT,
                          Scope.OBJECT_BBOX, Op.FILL, Correspondence.NONE, Depth.TWO)
            ), "per_object")

        if "map_over_entities" in ops_str:
            return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS["map_over_panels_summary"], "map_over")

        if "extend_periodic" in ops_str:
            return FactorLabel(task_id, SKELETON_TO_FACTORS["mask_repair"], "repair")

        if "combine_cells" in ops_str:
            return FactorLabel(task_id, SCENE_FAMILY_TO_FACTORS["combine_to_output"], "combine_cells")

        if "assign_cells" in ops_str:
            return FactorLabel(task_id, FactorSet(
                Decomposition.PARTITION, Selector.CELL_PANEL,
                Scope.PARTITION_CELL, Op.COMBINE, Correspondence.POSITIONAL, Depth.TWO,
            ), "cell_assignment")

    # Fallback: classify by program string and demo structure
    prog_str = str(prog).lower()

    for skel, fs in SKELETON_TO_FACTORS.items():
        if skel.replace("_", " ") in prog_str or skel in prog_str:
            return FactorLabel(task_id, fs, f"text:{skel}")

    # Last resort: structural heuristic
    d0 = demos[0]
    same_dims = all(d.input.shape == d.output.shape for d in demos)
    if not same_dims:
        return FactorLabel(task_id, FactorSet(
            Decomposition.REGION, Selector.OBJECT_SELECT,
            Scope.OBJECT, Op.EXTRACT, Correspondence.NONE, Depth.ONE,
        ), "heuristic:extract")

    diff = d0.input != d0.output
    change_frac = np.sum(diff) / max(d0.input.size, 1)
    if change_frac < 0.15:
        return FactorLabel(task_id, FactorSet(
            Decomposition.OBJECT, Selector.NONE,
            Scope.GLOBAL, Op.RECOLOR, Correspondence.NONE, Depth.ONE,
        ), "heuristic:recolor")

    return FactorLabel(task_id, FactorSet(
        Decomposition.OBJECT, Selector.NONE,
        Scope.GLOBAL, Op.TRANSFORM, Correspondence.NONE, Depth.ONE,
    ), "heuristic:transform")


if __name__ == "__main__":
    main()
