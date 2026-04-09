#!/usr/bin/env python3
"""Export Opus-ready task dossiers from ARIA's current system state.

Fast default:
    python scripts/export_task_dossiers.py --out-dir /tmp/aria-dossiers

Richer but slower:
    python scripts/export_task_dossiers.py --out-dir /tmp/aria-dossiers \
        --exact-solve --solver-summary --include-trace-json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.eval.task_dossiers import export_task_dossier_bundle


DEFAULT_GOLD_PATH = (
    Path(__file__).resolve().parent.parent
    / "aria"
    / "eval"
    / "structural_gates_tasks.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export structured task dossiers for large-context analysis"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the dossier bundle",
    )
    parser.add_argument(
        "--dataset",
        default="v2-eval",
        help="Dataset name (default: v2-eval)",
    )
    parser.add_argument(
        "--gold-path",
        default=str(DEFAULT_GOLD_PATH),
        help="Path to structural-gates gold task annotations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for structural-gates recall metrics",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Limit export to specific task IDs (repeatable)",
    )
    parser.add_argument(
        "--exact-solve",
        action="store_true",
        help="Run exact solve checks inside structural-gates run (slower)",
    )
    parser.add_argument(
        "--solver-summary",
        action="store_true",
        help="Run solve_task per dossier and include best-candidate summaries (slower)",
    )
    parser.add_argument(
        "--include-trace-json",
        action="store_true",
        help="Embed local trace_<task>.json content when available",
    )
    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
        help="solve_task max_search_steps for dossier solver summaries",
    )
    parser.add_argument(
        "--max-search-candidates",
        type=int,
        default=10000,
        help="solve_task max_search_candidates for dossier solver summaries",
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=2,
        help="solve_task max_refinement_rounds for dossier solver summaries",
    )
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent.parent / "logs"),
        help="Directory to scan for local trace_<task>.json files",
    )
    args = parser.parse_args()

    bundle_dir = export_task_dossier_bundle(
        out_dir=args.out_dir,
        dataset_name=args.dataset,
        gold_path=args.gold_path,
        task_ids=args.tasks,
        top_k=args.top_k,
        run_exact_solve=args.exact_solve,
        include_trace_json=args.include_trace_json,
        include_solver_summary=args.solver_summary,
        max_search_steps=args.max_search_steps,
        max_search_candidates=args.max_search_candidates,
        max_refinement_rounds=args.max_refinement_rounds,
        log_dir=args.log_dir,
    )

    print(f"Wrote dossier bundle to {bundle_dir}")
    print(f"  System summary:      {bundle_dir / 'SYSTEM_CONTEXT.md'}")
    print(f"  Gate report (text):  {bundle_dir / 'STRUCTURAL_GATES_REPORT.md'}")
    print(f"  Gate report (json):  {bundle_dir / 'STRUCTURAL_GATES_REPORT.json'}")
    print(f"  Opus prompt:         {bundle_dir / 'OPUS_ANALYSIS_PROMPT.md'}")
    print(f"  Monolithic bundle:   {bundle_dir / 'OPUS_BUNDLE.md'}")


if __name__ == "__main__":
    main()
