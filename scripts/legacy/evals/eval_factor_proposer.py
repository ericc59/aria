#!/usr/bin/env python3
"""Evaluate the factorized proposer on real ARC-2 data.

Parts:
A. Train proposer on solved tasks, evaluate per-factor accuracy
B. Inspect top-k factor combo quality
C. Compare factorized vs flat-family search on ARC-2
D. Diagnose factor interaction failures
E. Stress-test instantiation paths
F. ARC-2 before/after report
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aria.core.arc import solve_arc_task
from aria.core.factor_labels import (
    SKELETON_TO_FACTORS,
    SCENE_FAMILY_TO_FACTORS,
    FactorLabel,
    FactorLabelSet,
    label_from_skeleton,
)
from aria.core.factor_proposer import FactorProposer, uniform_factor_ranking
from aria.core.factor_search import factor_composition_search
from aria.core.grid_perception import perceive_grid
from aria.core.guidance_proposer import extract_cross_demo_features
from aria.core.scene_solve import infer_scene_programs, verify_scene_program
from aria.datasets import get_dataset, iter_tasks, list_task_ids
from aria.factor_instantiate import instantiate_factor_set
from aria.factors import (
    FACTOR_NAMES,
    FACTOR_ENUMS,
    FactorSet,
    enumerate_compatible,
    is_compatible,
)
from aria.types import DemoPair


# ---------------------------------------------------------------------------
# Part A: Train and evaluate proposer
# ---------------------------------------------------------------------------


def collect_solved_labels(dataset_name: str = "v2-train") -> tuple[
    list[np.ndarray], list[FactorSet], list[str]
]:
    """Collect (features, factor_labels, task_ids) from solvable tasks.

    Runs solve_arc_task on each task, extracts the winning family,
    maps it to factor labels.
    """
    ds = get_dataset(dataset_name)
    features_list: list[np.ndarray] = []
    labels_list: list[FactorSet] = []
    task_ids: list[str] = []
    solved_families: list[str] = []

    for task_id, task in iter_tasks(ds):
        demos = task.train
        try:
            result = solve_arc_task(demos, task_id=task_id, use_editor_search=False)
        except Exception:
            continue

        if not result.solved:
            continue

        # Extract family label
        label = _infer_factor_label(task_id, result)
        if label is None:
            continue

        feats = extract_cross_demo_features(demos)
        features_list.append(feats)
        labels_list.append(label.factors)
        task_ids.append(task_id)
        solved_families.append(label.source)

    return features_list, labels_list, task_ids


def _infer_factor_label(task_id: str, result: Any) -> FactorLabel | None:
    """Infer factor label from a solve result."""
    prog = result.winning_program
    if prog is None:
        return None

    # Try to determine family from program structure
    prog_type = type(prog).__name__

    # Check if it's a scene program wrapper
    if hasattr(prog, 'steps'):
        step_names = [s.op.value if hasattr(s, 'op') else str(s) for s in prog.steps]
        family = _classify_scene_program(step_names)
        fl = None
        if family:
            # Try scene family mapping
            for key, fs in SCENE_FAMILY_TO_FACTORS.items():
                if key in family:
                    fl = FactorLabel(task_id=task_id, factors=fs, source=f"scene:{key}")
                    break
        if fl is None:
            # Try skeleton mapping
            for skel, fs in SKELETON_TO_FACTORS.items():
                if skel in family if family else False:
                    fl = FactorLabel(task_id=task_id, factors=fs, source=f"skeleton:{skel}")
                    break
        return fl

    # Fallback: try to classify by program text
    prog_str = str(prog)
    for skel, fs in SKELETON_TO_FACTORS.items():
        if skel in prog_str:
            return FactorLabel(task_id=task_id, factors=fs, source=f"text:{skel}")

    return None


def _classify_scene_program(step_names: list[str]) -> str | None:
    """Classify a scene program by its step sequence."""
    steps_str = "->".join(step_names)

    if "select_entity" in steps_str and "canonicalize" in steps_str:
        return "select_extract_transform"
    if "select_entity" in steps_str and "recolor" in steps_str:
        return "select_extract_colormap"
    if "boolean_combine" in steps_str:
        return "boolean_combine"
    if "fill_enclosed" in steps_str:
        return "consensus_compose:enclosed"
    if "recolor_object" in steps_str:
        return "consensus_compose:frame"
    if "for_each_entity" in steps_str:
        return "per_object_operation"
    if "map_over_entities" in steps_str:
        return "map_over_panels_summary"
    if "select_entity" in steps_str:
        return "select_panel_extract"
    return None


def evaluate_proposer_per_factor(
    features_list: list[np.ndarray],
    labels_list: list[FactorSet],
    task_ids: list[str],
) -> dict[str, Any]:
    """Leave-one-out evaluation per factor head."""
    n = len(features_list)
    if n < 3:
        return {"error": f"too few labeled tasks: {n}"}

    enum_lists = {name: list(FACTOR_ENUMS[name]) for name in FACTOR_NAMES}
    enum_to_idx = {
        name: {v: i for i, v in enumerate(vals)}
        for name, vals in enum_lists.items()
    }

    results: dict[str, Any] = {}

    # Class balance per factor
    balance: dict[str, dict[str, int]] = {}
    for name in FACTOR_NAMES:
        counts: Counter = Counter()
        for fs in labels_list:
            val = getattr(fs, name)
            counts[val.value if hasattr(val, 'value') else str(val)] += 1
        balance[name] = dict(counts.most_common())
    results["class_balance"] = balance

    # Leave-one-out per factor
    X = np.stack(features_list)
    per_factor_results: dict[str, dict[str, float]] = {}

    for factor_name in FACTOR_NAMES:
        y = np.array([
            enum_to_idx[factor_name][getattr(fs, factor_name)]
            for fs in labels_list
        ], dtype=int)

        n_classes = len(enum_lists[factor_name])
        correct_top1 = 0
        correct_top3 = 0

        for i in range(n):
            # Leave one out
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            y_test = y[i]

            # Train a small head
            from aria.core.factor_proposer import FactorHead
            head = FactorHead(
                factor_name=factor_name,
                n_classes=n_classes,
                class_names=tuple(
                    m.value if hasattr(m, 'value') else str(m)
                    for m in enum_lists[factor_name]
                ),
            )
            head.fit(X_train, y_train, epochs=200)
            probs = head.predict_proba(X_test[0])

            # Top-1
            if np.argmax(probs) == y_test:
                correct_top1 += 1

            # Top-3
            top3 = np.argsort(-probs)[:min(3, n_classes)]
            if y_test in top3:
                correct_top3 += 1

        per_factor_results[factor_name] = {
            "top1_accuracy": correct_top1 / n,
            "top3_recall": correct_top3 / n,
            "n_classes": n_classes,
            "n_examples": n,
        }

    results["per_factor"] = per_factor_results

    # Joint factor combo evaluation
    proposer = FactorProposer()
    proposer.fit_from_labels(features_list, labels_list, epochs=300)

    joint_top1 = 0
    joint_topk = 0
    for i in range(n):
        feats = features_list[i]
        true_fs = labels_list[i]
        ranked = proposer.top_k_factor_sets(feats, k=50)
        predicted_sets = [fs for fs, _ in ranked]

        if predicted_sets and predicted_sets[0] == true_fs:
            joint_top1 += 1
        if true_fs in predicted_sets:
            joint_topk += 1

    results["joint"] = {
        "top1_accuracy": joint_top1 / n,
        "top50_recall": joint_topk / n,
        "n_examples": n,
    }

    return results


# ---------------------------------------------------------------------------
# Part B: Top-k factor combo quality
# ---------------------------------------------------------------------------


def inspect_topk_quality(
    features_list: list[np.ndarray],
    labels_list: list[FactorSet],
    task_ids: list[str],
) -> dict[str, Any]:
    """For solved tasks, report top-k quality of factor combo ranking."""
    n = len(features_list)
    if n < 3:
        return {"error": f"too few: {n}"}

    proposer = FactorProposer()
    proposer.fit_from_labels(features_list, labels_list, epochs=300)

    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    novel_combos_found = 0
    per_task_rank: dict[str, int] = {}

    for i in range(n):
        feats = features_list[i]
        true_fs = labels_list[i]
        ranked = proposer.top_k_factor_sets(feats, k=50)
        predicted_sets = [fs for fs, _ in ranked]

        # Find rank of true FactorSet
        rank = -1
        for j, fs in enumerate(predicted_sets):
            if fs == true_fs:
                rank = j
                break

        per_task_rank[task_ids[i]] = rank

        for k in recall_at_k:
            if 0 <= rank < k:
                recall_at_k[k] += 1

        # Check if true combo is novel (not in flat skeleton types)
        is_novel = true_fs not in SKELETON_TO_FACTORS.values()
        if is_novel and rank >= 0:
            novel_combos_found += 1

    return {
        "recall_at_k": {k: v / n for k, v in recall_at_k.items()},
        "novel_combos_found": novel_combos_found,
        "per_task_rank": per_task_rank,
        "mean_rank": np.mean([r for r in per_task_rank.values() if r >= 0]),
        "not_found_count": sum(1 for r in per_task_rank.values() if r < 0),
    }


# ---------------------------------------------------------------------------
# Part C: Factorized vs flat-family ARC-2 comparison
# ---------------------------------------------------------------------------


def compare_search_methods(
    dataset_name: str = "v2-eval",
    limit: int = 0,
) -> dict[str, Any]:
    """Compare factorized search vs flat-family search on ARC-2."""
    ds = get_dataset(dataset_name)
    task_ids = list_task_ids(ds)
    if limit > 0:
        task_ids = task_ids[:limit]

    flat_solved: list[str] = []
    factor_solved: list[str] = []
    both_solved: list[str] = []
    factor_only: list[str] = []
    flat_only: list[str] = []
    factor_candidate_counts: list[int] = []
    flat_candidate_counts: list[int] = []
    factor_program_counts: list[int] = []
    near_misses: list[tuple[str, int]] = []

    for task_id, task in iter_tasks(ds, task_ids=task_ids):
        demos = task.train
        test_demos = task.test

        # --- Flat-family search (existing pipeline) ---
        try:
            flat_progs = infer_scene_programs(demos)
            flat_count = len(flat_progs)
        except Exception:
            flat_progs = ()
            flat_count = 0
        flat_candidate_counts.append(flat_count)

        flat_verifies = False
        for prog in flat_progs:
            if verify_scene_program(prog, demos):
                flat_verifies = True
                break

        if flat_verifies:
            flat_solved.append(task_id)

        # --- Factorized search ---
        try:
            factor_results = factor_composition_search(
                demos, max_candidates=50, max_programs=200,
            )
        except Exception:
            factor_results = []
        factor_candidate_counts.append(len(factor_results))

        # Count instantiated programs
        try:
            perceptions = tuple(perceive_grid(d.input) for d in demos)
            prog_count = 0
            for fs in uniform_factor_ranking(max_combos=50):
                passed, _, _ = check_factor_consistency_safe(perceptions, fs)
                if passed:
                    progs = instantiate_factor_set(fs, demos, perceptions)
                    prog_count += len(progs)
            factor_program_counts.append(prog_count)
        except Exception:
            factor_program_counts.append(0)

        factor_verifies = len(factor_results) > 0
        if factor_verifies:
            factor_solved.append(task_id)

        # Categorize
        if flat_verifies and factor_verifies:
            both_solved.append(task_id)
        elif factor_verifies and not flat_verifies:
            factor_only.append(task_id)
        elif flat_verifies and not factor_verifies:
            flat_only.append(task_id)

        # Near-miss check for factor search
        if not factor_verifies and factor_results:
            near_misses.append((task_id, len(factor_results)))

    return {
        "dataset": dataset_name,
        "n_tasks": len(task_ids),
        "flat_solved": len(flat_solved),
        "flat_solved_ids": flat_solved,
        "factor_solved": len(factor_solved),
        "factor_solved_ids": factor_solved,
        "both_solved": len(both_solved),
        "factor_only": len(factor_only),
        "factor_only_ids": factor_only,
        "flat_only": len(flat_only),
        "flat_only_ids": flat_only,
        "avg_flat_candidates": np.mean(flat_candidate_counts) if flat_candidate_counts else 0,
        "avg_factor_candidates": np.mean(factor_candidate_counts) if factor_candidate_counts else 0,
        "avg_factor_programs": np.mean(factor_program_counts) if factor_program_counts else 0,
        "near_misses": near_misses[:10],
    }


def check_factor_consistency_safe(perceptions, fs):
    try:
        from aria.consensus import check_factor_consistency
        return check_factor_consistency(perceptions, fs)
    except Exception:
        return True, 1.0, ""


# ---------------------------------------------------------------------------
# Part D: Factor interaction failures
# ---------------------------------------------------------------------------


def diagnose_interaction_failures(
    features_list: list[np.ndarray],
    labels_list: list[FactorSet],
    task_ids: list[str],
) -> dict[str, Any]:
    """Diagnose where independent factor heads produce bad joint combos."""
    n = len(features_list)
    if n < 3:
        return {"error": f"too few: {n}"}

    proposer = FactorProposer()
    proposer.fit_from_labels(features_list, labels_list, epochs=300)

    incompatible_top1 = 0
    marginal_correct_joint_wrong = 0
    decomp_selector_mismatches = 0
    scope_op_mismatches = 0
    examples: list[dict] = []

    for i in range(n):
        feats = features_list[i]
        true_fs = labels_list[i]
        probs = proposer.predict_factor_probs(feats)

        # Check if top-1 per-factor produces a compatible combo
        top1_factors = {}
        for name in FACTOR_NAMES:
            enum_list = list(FACTOR_ENUMS[name])
            top1_factors[name] = enum_list[np.argmax(probs[name])]

        top1_fs = FactorSet(**top1_factors)
        if not is_compatible(top1_fs):
            incompatible_top1 += 1

        # Check if each marginal is correct but joint is wrong
        marginals_correct = True
        for name in FACTOR_NAMES:
            enum_list = list(FACTOR_ENUMS[name])
            true_val = getattr(true_fs, name)
            pred_val = enum_list[np.argmax(probs[name])]
            if pred_val != true_val:
                marginals_correct = False
                break

        ranked = proposer.top_k_factor_sets(feats, k=10)
        joint_correct = any(fs == true_fs for fs, _ in ranked)

        if marginals_correct and not joint_correct:
            marginal_correct_joint_wrong += 1

        # Specific mismatch types
        decomp_top = list(FACTOR_ENUMS["decomposition"])[np.argmax(probs["decomposition"])]
        sel_top = list(FACTOR_ENUMS["selector"])[np.argmax(probs["selector"])]
        from aria.factors import _DECOMP_SELECTORS
        if sel_top not in _DECOMP_SELECTORS.get(decomp_top, set()):
            decomp_selector_mismatches += 1

        if not joint_correct:
            examples.append({
                "task_id": task_ids[i],
                "true": repr(true_fs),
                "top1_compatible": is_compatible(top1_fs),
                "top1": repr(top1_fs),
            })

    return {
        "incompatible_top1_count": incompatible_top1,
        "incompatible_top1_rate": incompatible_top1 / n,
        "marginal_correct_joint_wrong": marginal_correct_joint_wrong,
        "decomp_selector_mismatches": decomp_selector_mismatches,
        "decomp_selector_mismatch_rate": decomp_selector_mismatches / n,
        "failure_examples": examples[:5],
    }


# ---------------------------------------------------------------------------
# Part E: Stress-test instantiation
# ---------------------------------------------------------------------------


def stress_test_instantiation(
    dataset_name: str = "v2-train",
    limit: int = 20,
) -> dict[str, Any]:
    """Check which factor combos produce programs vs empty instantiation."""
    ds = get_dataset(dataset_name)
    task_ids = list_task_ids(ds)
    if limit > 0:
        task_ids = task_ids[:limit]

    # Pick first task for testing
    task_id, task = next(iter_tasks(ds, task_ids=task_ids[:1]))
    demos = task.train
    perceptions = tuple(perceive_grid(d.input) for d in demos)

    all_combos = enumerate_compatible()
    depth_counts = Counter()
    instantiation_counts = Counter()
    empty_factors: list[str] = []
    total_programs = 0

    for fs in all_combos[:200]:  # Sample
        programs = instantiate_factor_set(fs, demos, perceptions)
        n_progs = len(programs)
        total_programs += n_progs
        depth_counts[fs.depth.value] += 1

        if n_progs > 0:
            instantiation_counts["has_programs"] += 1
        else:
            instantiation_counts["empty"] += 1
            empty_factors.append(repr(fs))

    # Check specific weak areas
    corr_combos = [fs for fs in all_combos if fs.correspondence.value != "none"][:50]
    corr_empty = 0
    for fs in corr_combos:
        progs = instantiate_factor_set(fs, demos, perceptions)
        if not progs:
            corr_empty += 1

    depth3_combos = [fs for fs in all_combos if fs.depth.value == 3][:50]
    depth3_empty = 0
    for fs in depth3_combos:
        progs = instantiate_factor_set(fs, demos, perceptions)
        if not progs:
            depth3_empty += 1

    growth_combos = [fs for fs in all_combos if fs.op.value == "grow_propagate"][:20]
    growth_empty = sum(
        1 for fs in growth_combos
        if not instantiate_factor_set(fs, demos, perceptions)
    )

    return {
        "sampled_combos": 200,
        "has_programs": instantiation_counts.get("has_programs", 0),
        "empty": instantiation_counts.get("empty", 0),
        "total_programs_generated": total_programs,
        "correspondence_empty_rate": corr_empty / max(len(corr_combos), 1),
        "depth3_empty_rate": depth3_empty / max(len(depth3_combos), 1),
        "growth_empty_rate": growth_empty / max(len(growth_combos), 1),
        "empty_factor_examples": empty_factors[:5],
        "weakness_summary": {
            "proposer": "N/A (not trained yet)" if True else "",
            "compatibility": f"{instantiation_counts.get('empty', 0)} / 200 combos produce no programs",
            "instantiation": f"correspondence={corr_empty}/{len(corr_combos)} empty, "
                             f"depth3={depth3_empty}/{len(depth3_combos)} empty, "
                             f"growth={growth_empty}/{len(growth_combos)} empty",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("FACTORIZED PROPOSER EVALUATION")
    print("=" * 70)

    # Part A: Collect labels and train
    print("\n--- Part A: Collecting solved-task labels ---")
    t0 = time.time()
    features, labels, task_ids = collect_solved_labels("v2-train")
    t1 = time.time()
    print(f"Collected {len(labels)} labeled tasks in {t1-t0:.1f}s")

    if len(labels) < 3:
        print("ERROR: Too few labeled tasks to evaluate. Falling back to stress test only.")
        print("\n--- Part E: Stress-test instantiation ---")
        stress = stress_test_instantiation(limit=5)
        _print_dict("Instantiation stress test", stress)
        return

    # Part A continued: evaluate
    print("\n--- Part A: Per-factor evaluation ---")
    t0 = time.time()
    eval_results = evaluate_proposer_per_factor(features, labels, task_ids)
    t1 = time.time()
    _print_dict("Class balance", eval_results.get("class_balance", {}))
    _print_dict("Per-factor accuracy", eval_results.get("per_factor", {}))
    _print_dict("Joint accuracy", eval_results.get("joint", {}))
    print(f"  (evaluation took {t1-t0:.1f}s)")

    # Part B: Top-k quality
    print("\n--- Part B: Top-k factor combo quality ---")
    topk = inspect_topk_quality(features, labels, task_ids)
    _print_dict("Top-k recall", topk)

    # Part D: Interaction failures
    print("\n--- Part D: Factor interaction failures ---")
    diag = diagnose_interaction_failures(features, labels, task_ids)
    _print_dict("Interaction diagnostics", diag)

    # Part E: Stress test
    print("\n--- Part E: Stress-test instantiation ---")
    stress = stress_test_instantiation(limit=5)
    _print_dict("Instantiation stress test", stress)

    # Part C: ARC-2 comparison (v2-eval)
    print("\n--- Part C: Factorized vs flat-family on v2-eval ---")
    t0 = time.time()
    comparison = compare_search_methods("v2-eval", limit=0)
    t1 = time.time()
    _print_dict("v2-eval comparison", comparison)
    print(f"  (comparison took {t1-t0:.1f}s)")

    # Part F: Summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(f"Labeled tasks: {len(labels)}")
    for name in FACTOR_NAMES:
        pf = eval_results.get("per_factor", {}).get(name, {})
        print(f"  {name}: top1={pf.get('top1_accuracy', 0):.2f} top3={pf.get('top3_recall', 0):.2f} classes={pf.get('n_classes', 0)}")
    joint = eval_results.get("joint", {})
    print(f"Joint: top1={joint.get('top1_accuracy', 0):.2f} top50={joint.get('top50_recall', 0):.2f}")
    print(f"Top-k recall@10: {topk.get('recall_at_k', {}).get(10, 0):.2f}")
    print(f"Incompatible top-1 rate: {diag.get('incompatible_top1_rate', 0):.2f}")
    print(f"v2-eval flat solves: {comparison.get('flat_solved', 0)}")
    print(f"v2-eval factor solves: {comparison.get('factor_solved', 0)}")
    print(f"v2-eval factor-only: {comparison.get('factor_only', 0)}")
    print(f"v2-eval flat-only: {comparison.get('flat_only', 0)}")


def _print_dict(label: str, d: dict, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{label}:")
    for k, v in d.items():
        if isinstance(v, dict):
            _print_dict(str(k), v, indent + 2)
        elif isinstance(v, float):
            print(f"{prefix}  {k}: {v:.4f}")
        elif isinstance(v, list) and len(v) > 5:
            print(f"{prefix}  {k}: [{len(v)} items]")
        else:
            print(f"{prefix}  {k}: {v}")


if __name__ == "__main__":
    main()
