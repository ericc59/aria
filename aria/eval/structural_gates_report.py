"""Compact report generation for structural gates evaluation.

Produces human-readable and machine-readable reports that are easy
to compare between branches.
"""

from __future__ import annotations

import json
from typing import Any

from aria.eval.structural_gates_runner import RunReport
from aria.eval.structural_gates_scorer import GATE_ORDER


def format_text_report(report: RunReport) -> str:
    """Generate a compact text report for terminal display."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("STRUCTURAL GATES EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Dataset: {report.dataset}")
    lines.append(f"Tasks:   {report.n_tasks}")
    lines.append(f"Top-K:   {report.top_k}")
    lines.append(f"Time:    {report.elapsed_total_sec:.1f}s")
    lines.append("")

    # --- Aggregate metrics ---
    lines.append("-" * 72)
    lines.append("AGGREGATE GATE METRICS")
    lines.append("-" * 72)
    lines.append(f"{'Gate':<20} {'Pass Rate':>10} {'Mean Recall':>12}")
    lines.append("-" * 72)
    for gate in GATE_ORDER:
        pr = report.gate_pass_rate(gate)
        recall = report.gate_recall(gate)
        lines.append(f"{gate:<20} {pr:>9.0%} {recall:>11.2f}")
    lines.append("-" * 72)
    lines.append(
        f"{'exact_solve':<20} "
        f"{report.exact_solve_count}/{report.n_tasks}  "
        f"({report.exact_solve_count / max(report.n_tasks, 1):.0%})"
    )

    mcf = report.most_common_failing_gate()
    if mcf:
        lines.append(f"\nMost common first-failing gate: {mcf}")

    # --- Per-task details ---
    lines.append("")
    lines.append("-" * 72)
    lines.append("PER-TASK GATE OUTCOMES")
    lines.append("-" * 72)

    # Header
    gate_cols = [g[:6] for g in GATE_ORDER]
    header = f"{'task_id':<12} " + " ".join(f"{c:>6}" for c in gate_cols)
    header += f" {'solve':>6} {'1st_fail':<14} {'solve_path':<16} {'exec_path':<14}"
    lines.append(header)
    lines.append("-" * 90)

    for result in report.results:
        gr = result.gate_results
        cols = []
        for gate in GATE_ORDER:
            g = gr.gate_by_name(gate)
            if g is None:
                cols.append("  n/a ")
            elif g.passed:
                cols.append("    \u2713 ")
            else:
                cols.append(f" {g.score:.2f}")

        first_fail = gr.first_failing_gate or ""
        solve_mark = "\u2713" if result.exact_solve else "\u2717"
        solve_path = result.solve_path or ""
        exec_path = result.artifacts.executor_path or ""

        tid = result.task_id[:12]
        line = f"{tid:<12} " + " ".join(f"{c:>6}" for c in cols)
        line += f" {'  ' + solve_mark:>6} {first_fail:<14} {solve_path:<16} {exec_path:<14}"
        lines.append(line)

    # --- Path attribution summary ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("EXECUTOR PATH ATTRIBUTION")
    lines.append("-" * 90)
    for result in report.results:
        a = result.artifacts
        tid = result.task_id[:12]
        produced = ",".join(a.executor_paths_produced) or "none"
        ran = ",".join(a.executor_paths_ran) or "none"
        verified = ",".join(a.executor_paths_verified) or "none"
        lines.append(
            f"{tid:<12}  produced=[{produced}]  ran=[{ran}]  verified=[{verified}]"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def format_json_report(report: RunReport) -> dict[str, Any]:
    """Generate a machine-readable JSON report for branch comparison."""
    aggregate = {}
    for gate in GATE_ORDER:
        aggregate[f"{gate}_pass_rate"] = round(report.gate_pass_rate(gate), 3)
        aggregate[f"{gate}_mean_recall"] = round(report.gate_recall(gate), 3)
    aggregate["exact_solve_rate"] = round(
        report.exact_solve_count / max(report.n_tasks, 1), 3
    )
    aggregate["exact_solve_count"] = report.exact_solve_count
    aggregate["most_common_failing_gate"] = report.most_common_failing_gate()

    per_task = []
    for result in report.results:
        gr = result.gate_results
        task_entry: dict[str, Any] = {
            "task_id": result.task_id,
            "exact_solve": result.exact_solve,
            "solve_path": result.solve_path,
            "first_failing_gate": gr.first_failing_gate,
            "elapsed_sec": result.elapsed_sec,
            "executor_path": result.artifacts.executor_path,
            "executor_paths_produced": result.artifacts.executor_paths_produced,
            "executor_paths_ran": result.artifacts.executor_paths_ran,
            "executor_paths_verified": result.artifacts.executor_paths_verified,
        }
        for gate in GATE_ORDER:
            g = gr.gate_by_name(gate)
            if g:
                task_entry[f"{gate}_passed"] = g.passed
                task_entry[f"{gate}_score"] = round(g.score, 3)
                task_entry[f"{gate}_details"] = _serialize_details(g.details)
        per_task.append(task_entry)

    return {
        "meta": {
            "dataset": report.dataset,
            "n_tasks": report.n_tasks,
            "top_k": report.top_k,
            "elapsed_total_sec": report.elapsed_total_sec,
        },
        "aggregate": aggregate,
        "per_task": per_task,
    }


def _serialize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Make details JSON-serializable."""
    result = {}
    for k, v in details.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, (list, tuple)):
            result[k] = [str(x) for x in v]
        else:
            result[k] = str(v)
    return result
