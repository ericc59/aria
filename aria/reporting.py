"""Solve report helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def extract_op_names(text: str) -> set[str]:
    """Pull op names from a program text."""
    return set(re.findall(r"(\w+)\(", text))


def extract_library_ops_used(program_text: str, library_names: list[str]) -> list[str]:
    """Return sorted library op names referenced in a program."""
    library_name_set = set(library_names)
    return sorted(op for op in extract_op_names(program_text) if op in library_name_set)


def task_solve_source(task: dict[str, Any]) -> str:
    """Classify a task outcome for reporting."""
    explicit = task.get("solve_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    if task.get("solved"):
        return "retrieval" if task.get("retrieved") else "proposal"
    return "unsolved"


def build_solve_report(tasks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Build the persisted solve report with benchmark-focused summary fields."""
    total = len(tasks)
    solved = sum(1 for t in tasks if t.get("solved"))
    test_correct = sum(
        1 for t in tasks if t.get("test_results")
        for tr in t.get("test_results", []) if tr.get("correct")
    )
    test_total = sum(
        len(t.get("test_results", [])) for t in tasks if t.get("test_results")
    )

    all_missing: Counter[str] = Counter()
    solve_sources: Counter[str] = Counter()
    retrieval_sources: Counter[str] = Counter()
    library_ops: Counter[str] = Counter()

    for task in tasks:
        solve_sources[task_solve_source(task)] += 1
        for op, cnt in task.get("missing_ops", {}).items():
            all_missing[op] += cnt
        for source in task.get("retrieval_sources", []):
            retrieval_sources[source] += 1
        for op in task.get("library_ops_used", []):
            library_ops[op] += 1

    retrieval_hits = solve_sources["retrieval"]
    proposal_solves = solve_sources["proposal"]
    search_solves = solve_sources["search"]

    report: dict[str, Any] = {
        "total": total,
        "solved": solved,
        "solve_rate": f"{100*solved/max(total,1):.1f}%",
        "test_correct": test_correct,
        "test_total": test_total,
        "config": config,
        "tasks": tasks,
        "source_counts": dict(solve_sources),
        "retrieval_hit_count": retrieval_hits,
        "retrieval_hit_rate": f"{100*retrieval_hits/max(total,1):.1f}%",
        "retrieval_share_of_solves": f"{100*retrieval_hits/max(solved,1):.1f}%",
        "proposal_solve_count": proposal_solves,
        "proposal_solve_rate": f"{100*proposal_solves/max(total,1):.1f}%",
        "search_solve_count": search_solves,
        "search_solve_rate": f"{100*search_solves/max(total,1):.1f}%",
    }
    if retrieval_sources:
        report["retrieval_sources_global"] = dict(retrieval_sources.most_common(20))
    if library_ops:
        report["library_ops_global"] = dict(library_ops.most_common(20))
    if all_missing:
        report["missing_ops_global"] = dict(all_missing.most_common(20))

    return report
