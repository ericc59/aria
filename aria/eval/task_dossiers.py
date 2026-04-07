"""Task dossier export for large-context failure analysis.

Builds a structured bundle combining:
- system summary
- structural-gates report
- per-task raw ARC data
- induced stage artifacts
- solver summaries

The output is designed for offline analysis in a large-context model.
It is strictly observational: no solver logic depends on these dossiers.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aria.datasets import get_dataset, load_arc_task
from aria.eval.structural_gates_report import format_json_report, format_text_report
from aria.eval.structural_gates_runner import RunReport, TaskRunResult, run_structural_gates
from aria.eval.structural_gates_schema import GoldTask, load_gold_tasks_map
from aria.eval.structural_gates_trace import extract_stage_artifacts
from aria.library.store import Library
from aria.runtime import program_to_text
from aria.solver import SolveResult, solve_task


_DEFAULT_GOLD_PATH = (
    Path(__file__).resolve().parent / "structural_gates_tasks.yaml"
)
_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass(frozen=True)
class BestCandidateSummary:
    program_text: str
    depth: int
    failed_demo: int | None
    error_type: str | None
    pixel_diff_count: int | None
    wrong_row_count: int | None
    wrong_col_count: int | None
    diff: dict[str, Any] | None


@dataclass(frozen=True)
class SolverSummary:
    solved: bool
    retrieved: bool
    searched: bool
    refined: bool
    retrieval_candidates_tried: int
    search_candidates_tried: int
    refinement_rounds: int
    task_signatures: tuple[str, ...]
    winning_program_text: str | None
    best_candidates: tuple[BestCandidateSummary, ...]


def _grid_to_list(grid: Any) -> list[list[int]]:
    if hasattr(grid, "tolist"):
        return [[int(v) for v in row] for row in grid.tolist()]
    return [[int(v) for v in row] for row in grid]


def _grid_to_text(grid: Any) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in _grid_to_list(grid))


def _task_to_jsonable(task: Any) -> dict[str, Any]:
    return {
        "train": [
            {
                "input": _grid_to_list(pair.input),
                "output": _grid_to_list(pair.output),
            }
            for pair in task.train
        ],
        "test": [
            {
                "input": _grid_to_list(pair.input),
                "output": _grid_to_list(pair.output),
            }
            for pair in task.test
        ],
    }


def _gold_to_jsonable(gold: GoldTask | None) -> dict[str, Any] | None:
    if gold is None:
        return None
    return {
        "task_id": gold.task_id,
        "decomposition": gold.decomposition.value,
        "entities": [
            {
                "name": entity.name,
                "kind": entity.kind.value,
                "selector_note": entity.selector_note,
            }
            for entity in gold.entities
        ],
        "relations": [
            {
                "kind": relation.kind.value,
                "source": relation.source,
                "target": relation.target,
            }
            for relation in gold.relations
        ],
        "template": gold.template.value,
        "critical_slots": dict(gold.critical_slots),
    }


def _best_trace_candidates(result: SolveResult, *, limit: int = 5) -> tuple[BestCandidateSummary, ...]:
    ref = result.refinement_result
    if ref is None:
        return ()
    entries: list[Any] = []
    for round_result in ref.rounds:
        entries.extend(round_result.trace)

    scored: list[tuple[tuple[int, int, int, int], Any]] = []
    for entry in entries:
        diff = entry.diff or {}
        pixel_diff = diff.get("pixel_diff_count")
        wrong_rows = diff.get("wrong_rows")
        wrong_cols = diff.get("wrong_cols")
        score_key = (
            0 if pixel_diff is not None else 1,
            int(pixel_diff) if pixel_diff is not None else 10**9,
            len(wrong_rows) if isinstance(wrong_rows, list) else 10**6,
            len(wrong_cols) if isinstance(wrong_cols, list) else 10**6,
        )
        scored.append((score_key, entry))
    scored.sort(key=lambda item: item[0])

    best: list[BestCandidateSummary] = []
    seen_programs: set[str] = set()
    for _, entry in scored:
        if entry.program_text in seen_programs:
            continue
        seen_programs.add(entry.program_text)
        diff = entry.diff or {}
        best.append(
            BestCandidateSummary(
                program_text=entry.program_text,
                depth=int(entry.depth),
                failed_demo=entry.failed_demo,
                error_type=entry.error_type,
                pixel_diff_count=diff.get("pixel_diff_count"),
                wrong_row_count=len(diff.get("wrong_rows", [])) if isinstance(diff.get("wrong_rows"), list) else None,
                wrong_col_count=len(diff.get("wrong_cols", [])) if isinstance(diff.get("wrong_cols"), list) else None,
                diff=diff if diff else None,
            )
        )
        if len(best) >= limit:
            break
    return tuple(best)


def _summarize_solver(
    task_id: str,
    task: Any,
    *,
    max_search_steps: int,
    max_search_candidates: int,
    max_refinement_rounds: int,
) -> SolverSummary:
    result = solve_task(
        task,
        Library(),
        task_id=task_id,
        max_search_steps=max_search_steps,
        max_search_candidates=max_search_candidates,
        max_refinement_rounds=max_refinement_rounds,
    )
    winning_program_text = None
    if result.winning_program is not None:
        winning_program_text = program_to_text(result.winning_program)
    return SolverSummary(
        solved=result.solved,
        retrieved=result.retrieved,
        searched=result.searched,
        refined=result.refined,
        retrieval_candidates_tried=result.retrieval_candidates_tried,
        search_candidates_tried=result.search_candidates_tried,
        refinement_rounds=result.refinement_rounds,
        task_signatures=result.task_signatures,
        winning_program_text=winning_program_text,
        best_candidates=_best_trace_candidates(result),
    )


def _local_trace_path(task_id: str, log_dir: Path) -> Path | None:
    path = log_dir / f"trace_{task_id}.json"
    return path if path.exists() else None


def _artifact_to_jsonable(task_result: TaskRunResult, *, include_trace_json: bool, log_dir: Path) -> dict[str, Any]:
    artifacts = task_result.artifacts
    trace_path = _local_trace_path(task_result.task_id, log_dir)
    local_trace_json = None
    if include_trace_json and trace_path is not None:
        try:
            local_trace_json = json.loads(trace_path.read_text())
        except Exception:
            local_trace_json = None

    return {
        "decomposition_hypotheses": list(artifacts.decomposition_hypotheses),
        "entities": [asdict(entity) for entity in artifacts.entities],
        "relations": [asdict(relation) for relation in artifacts.relations],
        "template_hypotheses": list(artifacts.template_hypotheses),
        "slot_candidates": dict(artifacts.slot_candidates),
        "executor_attempted": artifacts.executor_attempted,
        "executor_ran": artifacts.executor_ran,
        "executor_error": artifacts.executor_error,
        "local_trace_path": str(trace_path) if trace_path is not None else None,
        "local_trace_json": local_trace_json,
    }


def _empty_gate_jsonable(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "first_failing_gate": None,
        "all_passed": False,
        "exact_solve": False,
        "gates": {},
        "note": "no gold structural-gates annotation for this task",
    }


def _gate_jsonable(task_result: TaskRunResult) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for gate in task_result.gate_results.gates:
        gates[gate.gate_name] = {
            "passed": gate.passed,
            "score": gate.score,
            "details": gate.details,
        }
    return {
        "first_failing_gate": task_result.gate_results.first_failing_gate,
        "all_passed": task_result.gate_results.all_passed,
        "exact_solve": task_result.exact_solve,
        "gates": gates,
    }


def render_task_markdown(dossier: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Task {dossier['task_id']}")
    lines.append("")
    lines.append("## Structural Gates")
    lines.append("")
    lines.append(f"- First failing gate: `{dossier['gates']['first_failing_gate']}`")
    lines.append(f"- Exact solve: `{dossier['gates']['exact_solve']}`")
    if dossier["gates"].get("note"):
        lines.append(f"- Note: {dossier['gates']['note']}")
    lines.append("")
    for gate_name, gate in dossier["gates"]["gates"].items():
        lines.append(f"- `{gate_name}`: passed={gate['passed']} score={gate['score']}")
    lines.append("")

    if dossier.get("gold") is not None:
        lines.append("## Gold Annotation")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(dossier["gold"], indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Train/Test Grids")
    lines.append("")
    for i, pair in enumerate(dossier["task"]["train"]):
        lines.append(f"### Train {i} Input")
        lines.append("```text")
        lines.append(_grid_to_text(pair["input"]))
        lines.append("```")
        lines.append(f"### Train {i} Output")
        lines.append("```text")
        lines.append(_grid_to_text(pair["output"]))
        lines.append("```")
    for i, pair in enumerate(dossier["task"]["test"]):
        lines.append(f"### Test {i} Input")
        lines.append("```text")
        lines.append(_grid_to_text(pair["input"]))
        lines.append("```")
    lines.append("")

    lines.append("## Stage Artifacts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dossier["artifacts"], indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Solver Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dossier["solver_summary"], indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _system_context_markdown(report: RunReport, *, dataset_name: str, top_k: int) -> str:
    return f"""# ARIA System Context

## Purpose

This bundle is for offline diagnosis of ARIA's current failure modes on ARC.
The goal is not to invent task-specific hacks. The goal is to identify
generalized missing capabilities in the system.

## Hard Constraints

- ARC-2 first
- no task-id logic
- no benchmark-specific hacks
- exact verification remains final arbiter
- no end-to-end grid prediction
- no LLM-generated solver logic
- prefer reusable operators and compositional schemas over family-zoo additions

## Current ARIA Architecture (high level)

- factorized/sketch guidance and proposal
- structural-gates harness for decomposition/entity/relation/template/slot/executor
- scene IR + scene executor
- local-rule synthesis
- correspondence-conditioned actions
- exact verification on train demos and test execution through symbolic programs

## Current Structural-Gates Run

- dataset: `{dataset_name}`
- top_k: `{top_k}`
- tasks in bundle: `{report.n_tasks}`
- exact solves on slice: `{report.exact_solve_count}/{report.n_tasks}`
- most common first-failing gate: `{report.most_common_failing_gate()}`

## Aggregate Structural-Gates Report

```text
{format_text_report(report)}
```
"""


def _analysis_prompt_markdown(bundle_dir: Path) -> str:
    return f"""# Opus Analysis Prompt

You are analyzing ARIA's current solver limitations from a structured bundle.

Read these files in this bundle:
- `SYSTEM_CONTEXT.md`
- `STRUCTURAL_GATES_REPORT.md`
- every file under `tasks/`

Your job:
1. Cluster failures into a small number of generalized missing capabilities.
2. Distinguish:
   - decomposition/world-model gaps
   - relation/slot induction gaps
   - executor/program-schema expressiveness gaps
   - DSL/operator coverage gaps
3. Propose the smallest generalized architectural direction that explains the most failures.
4. Explicitly reject:
   - task-specific hacks
   - benchmark-specific logic
   - another handwritten family zoo
5. Stay aligned with NDEA:
   - latent structured programs
   - reusable operators
   - symbolic execution
   - exact verification

Output format:
1. Top 3 generalized missing capabilities
2. For each capability:
   - what tasks it explains
   - why ARIA currently fails
   - what reusable abstraction is missing
3. The single highest-ROI next architecture direction
4. The smallest Stage 1 implementation slice for that direction
5. What should be stopped immediately

Important:
- Do not propose hardcoded task families.
- Do not propose shallow search tuning.
- Do not propose brute-force operator growth.
- Favor factorized relational program representations and reusable slot/value derivations.

Bundle root:
- `{bundle_dir}`
"""


def export_task_dossier_bundle(
    *,
    out_dir: str | Path,
    dataset_name: str = "v2-eval",
    gold_path: str | Path = _DEFAULT_GOLD_PATH,
    task_ids: list[str] | None = None,
    top_k: int = 5,
    run_exact_solve: bool = True,
    include_trace_json: bool = False,
    include_solver_summary: bool = True,
    max_search_steps: int = 3,
    max_search_candidates: int = 10000,
    max_refinement_rounds: int = 2,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tasks_dir = out_path / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    gold_map = load_gold_tasks_map(gold_path)
    report = run_structural_gates(
        gold_path=gold_path,
        dataset_name=dataset_name,
        top_k=top_k,
        run_exact_solve=run_exact_solve,
        task_ids=task_ids,
    )
    ds = get_dataset(dataset_name)
    log_dir = Path(log_dir)

    # Bundle-level files
    (out_path / "STRUCTURAL_GATES_REPORT.md").write_text(format_text_report(report))
    (out_path / "STRUCTURAL_GATES_REPORT.json").write_text(
        json.dumps(format_json_report(report), indent=2)
    )
    (out_path / "SYSTEM_CONTEXT.md").write_text(
        _system_context_markdown(report, dataset_name=dataset_name, top_k=top_k)
    )
    (out_path / "OPUS_ANALYSIS_PROMPT.md").write_text(
        _analysis_prompt_markdown(out_path)
    )

    bundle_index: list[dict[str, Any]] = []
    reported_ids = {result.task_id for result in report.results}

    def _write_one(
        *,
        task_id: str,
        task_result: TaskRunResult | None,
        gold: GoldTask | None,
    ) -> None:
        task = load_arc_task(ds, task_id)
        ad_hoc_artifacts = None
        if task_result is None:
            ad_hoc_artifacts = extract_stage_artifacts(task_id, task.train)
        artifacts_json = (
            _artifact_to_jsonable(
                task_result,
                include_trace_json=include_trace_json,
                log_dir=log_dir,
            )
            if task_result is not None
            else {
                "decomposition_hypotheses": list(ad_hoc_artifacts.decomposition_hypotheses),
                "entities": [
                    asdict(entity)
                    for entity in ad_hoc_artifacts.entities
                ],
                "relations": [
                    asdict(relation)
                    for relation in ad_hoc_artifacts.relations
                ],
                "template_hypotheses": list(ad_hoc_artifacts.template_hypotheses),
                "slot_candidates": dict(ad_hoc_artifacts.slot_candidates),
                "executor_attempted": ad_hoc_artifacts.executor_attempted,
                "executor_ran": ad_hoc_artifacts.executor_ran,
                "executor_error": ad_hoc_artifacts.executor_error,
                "local_trace_path": str(_local_trace_path(task_id, log_dir)) if _local_trace_path(task_id, log_dir) is not None else None,
                "local_trace_json": (
                    json.loads(_local_trace_path(task_id, log_dir).read_text())
                    if include_trace_json and _local_trace_path(task_id, log_dir) is not None
                    else None
                ),
            }
        )
        solver_summary = None
        if include_solver_summary:
            solver_summary = asdict(
                _summarize_solver(
                    task_id,
                    task,
                    max_search_steps=max_search_steps,
                    max_search_candidates=max_search_candidates,
                    max_refinement_rounds=max_refinement_rounds,
                )
            )
        dossier = {
            "task_id": task_id,
            "dataset": dataset_name,
            "gold": _gold_to_jsonable(gold),
            "task": _task_to_jsonable(task),
            "gates": _gate_jsonable(task_result) if task_result is not None else _empty_gate_jsonable(task_id),
            "artifacts": artifacts_json,
            "solver_summary": solver_summary,
        }
        json_path = tasks_dir / f"{task_id}.json"
        md_path = tasks_dir / f"{task_id}.md"
        json_path.write_text(json.dumps(dossier, indent=2))
        md_path.write_text(render_task_markdown(dossier))
        bundle_index.append(
            {
                "task_id": task_id,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            }
        )

    for task_result in report.results:
        _write_one(
            task_id=task_result.task_id,
            task_result=task_result,
            gold=gold_map.get(task_result.task_id),
        )

    extra_task_ids = [
        task_id for task_id in (task_ids or [])
        if task_id not in reported_ids
    ]
    for task_id in extra_task_ids:
        _write_one(
            task_id=task_id,
            task_result=None,
            gold=gold_map.get(task_id),
        )

    (out_path / "INDEX.json").write_text(json.dumps(bundle_index, indent=2))

    mega_lines: list[str] = []
    mega_lines.append((out_path / "SYSTEM_CONTEXT.md").read_text())
    mega_lines.append("\n# Per-Task Dossiers\n")
    for item in bundle_index:
        mega_lines.append(Path(item["markdown_path"]).read_text())
    mega_lines.append("\n")
    mega_lines.append((out_path / "OPUS_ANALYSIS_PROMPT.md").read_text())
    (out_path / "OPUS_BUNDLE.md").write_text("\n\n".join(mega_lines))
    return out_path
