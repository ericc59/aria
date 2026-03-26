"""Audit the verified program corpus and refinement traces for training readiness.

Reads persisted stores under results/ and produces a structured report
at training/CORPUS_AUDIT.md with counts, coverage gaps, and recommendations.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    total_programs: int = 0
    unique_task_ids: int = 0
    unique_programs: int = 0
    step_count_distribution: dict[int, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    op_usage: Counter = field(default_factory=Counter)
    ops_used_once: list[str] = field(default_factory=list)
    task_ids_with_multiple_programs: int = 0
    programs_with_signatures: int = 0
    signature_set: set[str] = field(default_factory=set)


@dataclass
class SolveStats:
    total_tasks: int = 0
    solved_tasks: int = 0
    solve_rate: float = 0.0
    rounds_distribution: dict[int, int] = field(default_factory=dict)
    solve_sources: dict[str, int] = field(default_factory=dict)
    task_signatures_available: int = 0
    unique_signatures: set[str] = field(default_factory=set)


@dataclass
class TraceStats:
    total_records: int = 0
    solved_records: int = 0
    failed_records: int = 0
    total_rounds: int = 0
    total_candidates: int = 0
    focus_distribution: dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    avg_candidates_per_record: float = 0.0
    records_with_near_miss: int = 0
    near_miss_threshold: float = 350.0


@dataclass
class LibraryStats:
    total_entries: int = 0
    total_use_count: int = 0
    return_type_distribution: dict[str, int] = field(default_factory=dict)
    max_level: int = 0


@dataclass
class AuditReport:
    corpus: CorpusStats = field(default_factory=CorpusStats)
    train_solve: SolveStats = field(default_factory=SolveStats)
    eval_solve: SolveStats = field(default_factory=SolveStats)
    traces: TraceStats = field(default_factory=TraceStats)
    library: LibraryStats = field(default_factory=LibraryStats)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_ops(program_text: str) -> list[str]:
    return re.findall(r"(\w+)\(", program_text)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def audit_program_store(path: Path) -> CorpusStats:
    data = _load_json(path)
    if data is None or not isinstance(data, dict):
        return CorpusStats()

    programs = data.get("programs", [])
    stats = CorpusStats(
        total_programs=len(programs),
        unique_programs=len(programs),
    )

    all_task_ids: set[str] = set()
    task_id_to_programs: dict[str, int] = Counter()

    for p in programs:
        text = p.get("program", "")
        task_ids = p.get("task_ids", [])
        sources = p.get("sources", [])
        sigs = p.get("signatures", [])

        steps = sum(
            1 for line in text.splitlines()
            if line.startswith(("let ", "bind ", "assert "))
        )
        stats.step_count_distribution[steps] = (
            stats.step_count_distribution.get(steps, 0) + 1
        )

        for src in sources:
            stats.source_distribution[src] = (
                stats.source_distribution.get(src, 0) + 1
            )

        for op in _extract_ops(text):
            stats.op_usage[op] += 1

        all_task_ids.update(task_ids)
        for tid in task_ids:
            task_id_to_programs[tid] += 1

        if sigs:
            stats.programs_with_signatures += 1
            stats.signature_set.update(sigs)

    stats.unique_task_ids = len(all_task_ids)
    stats.task_ids_with_multiple_programs = sum(
        1 for count in task_id_to_programs.values() if count > 1
    )
    stats.ops_used_once = sorted(
        op for op, count in stats.op_usage.items() if count == 1
    )

    return stats


def audit_solve_report(path: Path) -> SolveStats:
    data = _load_json(path)
    if data is None or not isinstance(data, dict):
        return SolveStats()

    tasks = data.get("tasks", [])
    stats = SolveStats(
        total_tasks=len(tasks),
        solved_tasks=sum(1 for t in tasks if t.get("solved")),
    )
    stats.solve_rate = (
        stats.solved_tasks / max(stats.total_tasks, 1) * 100.0
    )

    for t in tasks:
        rounds = t.get("rounds", 0)
        stats.rounds_distribution[rounds] = (
            stats.rounds_distribution.get(rounds, 0) + 1
        )
        if t.get("solved"):
            src = t.get("solve_source", "search")
            stats.solve_sources[src] = stats.solve_sources.get(src, 0) + 1

        task_sigs = t.get("task_signatures", [])
        if task_sigs:
            stats.task_signatures_available += 1
            stats.unique_signatures.update(task_sigs)

    return stats


def audit_trace_store(path: Path) -> TraceStats:
    data = _load_json(path)
    if data is None or not isinstance(data, dict):
        return TraceStats()

    records = data.get("records", [])
    stats = TraceStats(total_records=len(records))

    total_candidates = 0
    for rec in records:
        if rec.get("solved"):
            stats.solved_records += 1
        else:
            stats.failed_records += 1

        candidates = rec.get("candidates_tried", 0)
        total_candidates += candidates

        rounds = rec.get("rounds", [])
        stats.total_rounds += len(rounds)

        has_near_miss = False
        for rnd in rounds:
            feedback = rnd.get("feedback", {})
            focus = feedback.get("suggested_focus", "unknown")
            stats.focus_distribution[focus] = (
                stats.focus_distribution.get(focus, 0) + 1
            )

            trace = rnd.get("trace", [])
            for entry in trace:
                depth = entry.get("depth", 0)
                if depth > stats.max_depth:
                    stats.max_depth = depth

                score = entry.get("score")
                if (
                    score is not None
                    and score >= stats.near_miss_threshold
                    and not entry.get("passed")
                ):
                    has_near_miss = True

        if has_near_miss:
            stats.records_with_near_miss += 1

    stats.total_candidates = total_candidates
    stats.avg_candidates_per_record = (
        total_candidates / max(len(records), 1)
    )
    return stats


def audit_library(path: Path) -> LibraryStats:
    data = _load_json(path)
    if data is None or not isinstance(data, dict):
        return LibraryStats()

    entries = data.get("entries", [])
    stats = LibraryStats(total_entries=len(entries))

    for e in entries:
        use_count = e.get("use_count", 0)
        stats.total_use_count += use_count

        rt = e.get("return_type", "unknown")
        stats.return_type_distribution[rt] = (
            stats.return_type_distribution.get(rt, 0) + 1
        )

        level = e.get("level", 0)
        if level > stats.max_level:
            stats.max_level = level

    return stats


# ---------------------------------------------------------------------------
# Weakness detection
# ---------------------------------------------------------------------------

def detect_weaknesses(report: AuditReport) -> list[str]:
    weaknesses: list[str] = []
    c = report.corpus
    t = report.traces

    if c.total_programs < 200:
        weaknesses.append(
            f"CRITICAL: Only {c.total_programs} verified programs "
            f"(target: 800-1000 for Phase 2). Far too few for any model training."
        )

    if c.unique_task_ids < 30:
        weaknesses.append(
            f"Only {c.unique_task_ids} unique task IDs covered. "
            "Need broad coverage across ARC-1 (400 tasks)."
        )

    trivial = sum(
        count for steps, count in c.step_count_distribution.items()
        if steps <= 2
    )
    total = max(c.total_programs, 1)
    if trivial / total > 0.4:
        weaknesses.append(
            f"Size skew: {trivial}/{c.total_programs} programs "
            f"({100*trivial/total:.0f}%) are trivial (<=2 steps). "
            "Model will overfit to short programs."
        )

    if c.programs_with_signatures == 0:
        weaknesses.append(
            "No programs have task signatures attached. "
            "Cannot train NEXT_FOCUS without signature labels."
        )

    if t.total_records == 0:
        weaknesses.append(
            "No refinement traces found. "
            "Cannot train NEXT_EDIT without (feedback, edit) pairs."
        )
    elif t.records_with_near_miss < 10:
        weaknesses.append(
            f"Only {t.records_with_near_miss} traces with near-miss candidates "
            f"(score >= {t.near_miss_threshold}). "
            "Need hard near-miss data for contrastive NEXT_EDIT training."
        )

    if t.total_records > 0 and t.failed_records > 0:
        focus_counts = t.focus_distribution
        if focus_counts.get("marker_geometry", 0) == 0:
            weaknesses.append(
                "No marker_geometry focus in traces. "
                "Missing an important refinement category."
            )
        if focus_counts.get("color_map", 0) == 0:
            weaknesses.append(
                "No color_map focus in traces. "
                "Missing an important refinement category."
            )

    if report.eval_solve.total_tasks > 0 and report.eval_solve.solved_tasks == 0:
        weaknesses.append(
            f"Eval solve rate is 0% ({report.eval_solve.total_tasks} tasks). "
            "Retrieval-transfer is not working yet."
        )

    if report.library.total_entries < 20:
        weaknesses.append(
            f"Library has only {report.library.total_entries} entries "
            f"(target: 80-120). Abstractions are thin."
        )

    long_programs = sum(
        count for steps, count in c.step_count_distribution.items()
        if steps >= 8
    )
    if long_programs < 10:
        weaknesses.append(
            f"Only {long_programs} programs with >=8 steps. "
            "Need complex multi-step examples for SKETCH training."
        )

    return weaknesses


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(report: AuditReport) -> list[str]:
    recs: list[str] = []
    c = report.corpus

    if c.total_programs < 200:
        recs.append(
            "PRIORITY 1: Run batch solve on full ARC-1 training set (400 tasks) "
            "with higher search budgets (max_candidates=50000, max_steps=5, "
            "max_refinement_rounds=4). Target: 200+ verified programs."
        )

    if c.programs_with_signatures == 0:
        recs.append(
            "PRIORITY 2: Backfill task signatures into program_store.json. "
            "Re-run solve with signature persistence enabled, or write a "
            "migration script to recompute signatures for existing programs."
        )

    if report.traces.total_records == 0:
        recs.append(
            "PRIORITY 3: Enable trace persistence in solve.py "
            "(RefinementTraceStore.save_json). Every solve attempt — "
            "successful or not — should emit a trace."
        )

    enough_for_sketch = c.total_programs >= 200 and sum(
        count for steps, count in c.step_count_distribution.items()
        if steps >= 3
    ) >= 80
    enough_for_next_edit = (
        report.traces.total_records >= 100
        and report.traces.records_with_near_miss >= 20
    )
    enough_for_next_focus = (
        c.programs_with_signatures >= 100
        and len(report.traces.focus_distribution) >= 3
    )

    if not enough_for_sketch:
        recs.append(
            "NOT READY for SKETCH training. Need at least 200 verified programs "
            "with 80+ having >=3 steps. Current: "
            f"{c.total_programs} programs total."
        )
    else:
        recs.append(
            "SKETCH training is feasible. Proceed with "
            "(state_graph, deltas) -> program pairs."
        )

    if not enough_for_next_edit:
        recs.append(
            "NOT READY for NEXT_EDIT training. Need 100+ refinement traces "
            f"with 20+ near-misses. Current: {report.traces.total_records} traces, "
            f"{report.traces.records_with_near_miss} near-misses."
        )
    else:
        recs.append(
            "NEXT_EDIT training is feasible. Proceed with "
            "(feedback, current_program) -> refined_program pairs."
        )

    if not enough_for_next_focus:
        recs.append(
            "Train NEXT_FOCUS first — it requires less data and "
            "improves refinement quality for collecting NEXT_EDIT pairs. "
            f"Need: 100 programs with signatures and 3+ focus labels. "
            f"Current: {c.programs_with_signatures} with signatures, "
            f"{len(report.traces.focus_distribution)} focus labels."
        )

    recs.append(
        "Generate synthetic tasks (Phase 2 in DESIGN.md) to multiply "
        "training volume. Target: 200K random step sequences length 3-10 "
        "for clean, high-volume training data."
    )

    if report.library.total_entries < 40:
        recs.append(
            "Grow the library before training. More library entries = "
            "more compositional programs in training data. "
            "Run library mining on a larger program corpus first."
        )

    return recs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def run_audit(results_dir: Path) -> AuditReport:
    report = AuditReport()

    report.corpus = audit_program_store(results_dir / "program_store.json")

    report.train_solve = audit_solve_report(results_dir / "v1-train.json")
    report.eval_solve = audit_solve_report(results_dir / "v1-eval.json")

    trace_path = results_dir / "refinement_traces.json"
    if not trace_path.exists():
        trace_path = results_dir / "trace_store.json"
    report.traces = audit_trace_store(trace_path)

    report.library = audit_library(results_dir / "library.json")

    report.weaknesses = detect_weaknesses(report)
    report.recommendations = generate_recommendations(report)

    return report


def format_report(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("# Corpus Audit Report")
    lines.append("")

    # --- Corpus ---
    c = report.corpus
    lines.append("## 1. Verified Program Corpus")
    lines.append("")
    lines.append(f"- **Total programs**: {c.total_programs}")
    lines.append(f"- **Unique task IDs**: {c.unique_task_ids}")
    lines.append(f"- **Programs with signatures**: {c.programs_with_signatures}")
    lines.append(f"- **Unique signatures**: {len(c.signature_set)}")
    lines.append(f"- **Tasks with multiple programs**: {c.task_ids_with_multiple_programs}")
    lines.append("")
    lines.append("### Step-count distribution")
    lines.append("")
    lines.append("| Steps | Count |")
    lines.append("|-------|-------|")
    for steps in sorted(c.step_count_distribution):
        lines.append(f"| {steps} | {c.step_count_distribution[steps]} |")
    lines.append("")
    lines.append("### Source distribution")
    lines.append("")
    for src, count in sorted(c.source_distribution.items()):
        lines.append(f"- `{src}`: {count}")
    lines.append("")

    # --- Op usage ---
    lines.append("### Library-op usage (top 20)")
    lines.append("")
    lines.append("| Op | Count |")
    lines.append("|-----|-------|")
    for op, count in c.op_usage.most_common(20):
        lines.append(f"| `{op}` | {count} |")
    lines.append("")
    lines.append(f"Total unique ops: {len(c.op_usage)}")
    if c.ops_used_once:
        lines.append(f"  Ops used only once: {', '.join(c.ops_used_once[:15])}")
    lines.append("")

    # --- Solve ---
    lines.append("## 2. Solve Results")
    lines.append("")
    for label, s in [("Train (v1-train)", report.train_solve), ("Eval (v1-eval)", report.eval_solve)]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Tasks: {s.total_tasks}")
        lines.append(f"- Solved: {s.solved_tasks} ({s.solve_rate:.1f}%)")
        lines.append(f"- Tasks with signatures: {s.task_signatures_available}")
        if s.rounds_distribution:
            lines.append(f"- Rounds distribution: {dict(sorted(s.rounds_distribution.items()))}")
        if s.solve_sources:
            lines.append(f"- Solve sources: {dict(s.solve_sources)}")
        lines.append("")

    # --- Traces ---
    t = report.traces
    lines.append("## 3. Refinement Traces")
    lines.append("")
    if t.total_records == 0:
        lines.append("**No refinement trace store found.**")
    else:
        lines.append(f"- Total records: {t.total_records}")
        lines.append(f"- Solved: {t.solved_records}")
        lines.append(f"- Failed: {t.failed_records}")
        lines.append(f"- Total rounds: {t.total_rounds}")
        lines.append(f"- Total candidates tried: {t.total_candidates}")
        lines.append(f"- Avg candidates/record: {t.avg_candidates_per_record:.1f}")
        lines.append(f"- Max trace depth: {t.max_depth}")
        lines.append(f"- Records with near-miss (score>={t.near_miss_threshold}): {t.records_with_near_miss}")
        lines.append("")
        lines.append("### Focus label distribution")
        lines.append("")
        for focus, count in sorted(t.focus_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"- `{focus}`: {count}")
    lines.append("")

    # --- Library ---
    lib = report.library
    lines.append("## 4. Library")
    lines.append("")
    lines.append(f"- Entries: {lib.total_entries}")
    lines.append(f"- Total use count: {lib.total_use_count}")
    lines.append(f"- Max level: {lib.max_level}")
    if lib.return_type_distribution:
        lines.append(f"- Return types: {dict(sorted(lib.return_type_distribution.items()))}")
    lines.append("")

    # --- Weaknesses ---
    lines.append("## 5. Weaknesses")
    lines.append("")
    if report.weaknesses:
        for i, w in enumerate(report.weaknesses, 1):
            lines.append(f"{i}. {w}")
    else:
        lines.append("No critical weaknesses detected.")
    lines.append("")

    # --- Recommendations ---
    lines.append("## 6. Recommendations")
    lines.append("")
    for i, r in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {r}")
    lines.append("")

    # --- Verdict ---
    lines.append("## 7. Verdict")
    lines.append("")
    ready = c.total_programs >= 200 and t.total_records >= 50
    if ready:
        lines.append("**Proceed with training.** Corpus meets minimum thresholds.")
    else:
        lines.append("**Do not train yet.** Collect more data first.")
        lines.append("")
        lines.append("Priority actions before training:")
        critical = [w for w in report.weaknesses if "CRITICAL" in w]
        for w in critical:
            lines.append(f"- {w}")
        if not critical:
            lines.append("- See weaknesses and recommendations above.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir: str | None = None) -> AuditReport:
    base = Path(results_dir) if results_dir else Path("results")
    report = run_audit(base)
    md = format_report(report)

    output_dir = Path("training")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "CORPUS_AUDIT.md"
    output_path.write_text(md)

    print(md)
    print(f"\nWritten to {output_path}")
    return report


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(results_path)
