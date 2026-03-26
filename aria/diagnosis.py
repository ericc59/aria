"""Gap analysis from eval results.

Answers: "where is the system falling short, and why?"
Takes eval task outcomes and produces a structured diagnosis that
identifies the biggest opportunities for improvement.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def diagnose(task_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a structured gap analysis from eval task outcomes.

    The diagnosis is organized around actionable categories:
    - What's working (solved tasks by source)
    - What's close (near-misses that almost worked)
    - What's missing (tasks with no useful hypotheses)
    - Where the library helps vs doesn't
    """
    solved = [t for t in task_outcomes if t.get("solved")]
    unsolved = [t for t in task_outcomes if not t.get("solved")]

    return {
        "summary": _summary(task_outcomes, solved, unsolved),
        "solve_breakdown": _solve_breakdown(solved),
        "near_misses": _near_miss_analysis(unsolved),
        "hypothesis_coverage": _hypothesis_coverage(task_outcomes),
        "gap_categories": _gap_categories(unsolved),
        "top_failing_signatures": _top_failing_signatures(unsolved),
    }


def format_diagnosis(diag: dict[str, Any]) -> str:
    """Format a diagnosis dict as a human-readable report."""
    lines: list[str] = []
    s = diag["summary"]

    lines.append(f"Solved: {s['solved']}/{s['total']} ({s['solve_rate']})")
    lines.append("")

    # Solve breakdown
    sb = diag["solve_breakdown"]
    lines.append("Solve sources:")
    for source, count in sorted(sb.items(), key=lambda x: -x[1]):
        lines.append(f"  {source}: {count}")
    lines.append("")

    # Hypothesis coverage
    hc = diag["hypothesis_coverage"]
    lines.append("Hypothesis coverage:")
    lines.append(f"  tasks with hints available: {hc['hints_available']}")
    lines.append(f"  tasks with skeletons tested: {hc['skeletons_tested']}")
    lines.append(f"  solved by skeleton: {hc['solved_by_skeleton']}")
    lines.append(f"  solved with retrieved abstraction: {hc['solved_with_abstraction']}")
    lines.append("")

    # Near-miss analysis
    nm = diag["near_misses"]
    lines.append(f"Near-misses (unsolved with skeleton near-miss): {nm['count']}")
    if nm["by_error_type"]:
        lines.append("  by error type:")
        for et, count in sorted(nm["by_error_type"].items(), key=lambda x: -x[1]):
            lines.append(f"    {et}: {count}")
    if nm["examples"]:
        lines.append("  closest misses:")
        for ex in nm["examples"][:5]:
            lines.append(
                f"    {ex['task_id']}: {ex['source']} "
                f"({ex['error_type']})"
            )
            for line in ex["program_text"].splitlines():
                lines.append(f"      {line}")
    lines.append("")

    # Gap categories
    gc = diag["gap_categories"]
    lines.append("Gap categories (unsolved tasks):")
    lines.append(f"  no hints available: {gc['no_hints']}")
    lines.append(f"  hints available, no near-miss: {gc['hints_no_near_miss']}")
    lines.append(f"  near-miss but not solved: {gc['near_miss_not_solved']}")
    lines.append(f"  no refinement result: {gc['no_refinement']}")
    lines.append("")

    # Top failing signatures
    ts = diag["top_failing_signatures"]
    if ts:
        lines.append("Top signatures in unsolved tasks:")
        for sig, count in ts[:10]:
            lines.append(f"  {sig}: {count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis components
# ---------------------------------------------------------------------------


def _summary(
    all_tasks: list[dict], solved: list[dict], unsolved: list[dict],
) -> dict[str, Any]:
    total = len(all_tasks)
    return {
        "total": total,
        "solved": len(solved),
        "unsolved": len(unsolved),
        "solve_rate": f"{100 * len(solved) / max(total, 1):.1f}%",
    }


def _solve_breakdown(solved: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t in solved:
        source = t.get("solve_source", "unknown")
        if t.get("solved_by_skeleton"):
            source = "skeleton"
        counts[source] += 1
    return dict(counts)


def _near_miss_analysis(unsolved: list[dict]) -> dict[str, Any]:
    near_misses = [
        t for t in unsolved if "skeleton_near_miss" in t
    ]
    by_error_type: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for t in near_misses:
        nm = t["skeleton_near_miss"]
        et = nm.get("error_type", "unknown")
        by_error_type[et] += 1
        examples.append({
            "task_id": t["task_id"],
            "source": nm.get("source", ""),
            "error_type": et,
            "program_text": nm.get("program_text", ""),
            "task_signatures": t.get("task_signatures", []),
        })

    # Sort examples: wrong_output first (closest to solving), then by task_id
    examples.sort(key=lambda x: (0 if x["error_type"] == "wrong_output" else 1, x["task_id"]))

    return {
        "count": len(near_misses),
        "by_error_type": dict(by_error_type),
        "examples": examples,
    }


def _hypothesis_coverage(all_tasks: list[dict]) -> dict[str, int]:
    return {
        "hints_available": sum(
            1 for t in all_tasks if t.get("abstraction_hints_available")
        ),
        "skeletons_tested": sum(
            1 for t in all_tasks if t.get("skeleton_hypotheses_tested", 0) > 0
        ),
        "solved_by_skeleton": sum(
            1 for t in all_tasks if t.get("solved_by_skeleton")
        ),
        "solved_with_abstraction": sum(
            1 for t in all_tasks if t.get("solved_with_retrieved_abstraction")
        ),
    }


def _gap_categories(unsolved: list[dict]) -> dict[str, int]:
    no_hints = 0
    hints_no_near_miss = 0
    near_miss_not_solved = 0
    no_refinement = 0

    for t in unsolved:
        has_hints = t.get("abstraction_hints_available", False)
        has_near_miss = "skeleton_near_miss" in t
        has_skeletons = t.get("skeleton_hypotheses_tested", 0) > 0

        if not has_hints and not has_skeletons:
            if t.get("refinement_rounds", 0) == 0 and not t.get("searched"):
                no_refinement += 1
            else:
                no_hints += 1
        elif has_near_miss:
            near_miss_not_solved += 1
        else:
            hints_no_near_miss += 1

    return {
        "no_hints": no_hints,
        "hints_no_near_miss": hints_no_near_miss,
        "near_miss_not_solved": near_miss_not_solved,
        "no_refinement": no_refinement,
    }


def _top_failing_signatures(unsolved: list[dict]) -> list[tuple[str, int]]:
    sig_counts: Counter[str] = Counter()
    for t in unsolved:
        for sig in t.get("task_signatures", []):
            sig_counts[sig] += 1
    return sig_counts.most_common(15)
