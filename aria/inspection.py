"""Task inspection helpers for perception, retrieval, and offline search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import aria.runtime  # noqa: F401
from aria.graph.extract import extract, extract_with_delta
from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.offline_search import SearchTraceEntry, search_program
from aria.program_store import ProgramStore
from aria.proposer.parser import ParseError, parse_program
from aria.runtime.ops import get_op
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import Delta, DemoPair, Grid, GridContext, StateGraph, TaskContext, Type
from aria.verify.verifier import verify


@dataclass(frozen=True)
class RetrievalInspection:
    rank: int
    task_ids: tuple[str, ...]
    sources: tuple[str, ...]
    use_count: int
    step_count: int
    signature_overlap: int
    matching_signatures: tuple[str, ...]
    status: str
    failed_demo: int | None = None
    error_type: str | None = None
    program_text: str = ""


def inspect_task(
    demos: tuple[DemoPair, ...],
    *,
    library: Library,
    program_store: ProgramStore,
    test_inputs: tuple[Grid, ...] = (),
    retrieval_limit: int = 10,
    max_search_steps: int = 3,
    max_search_candidates: int = 200,
    search_trace_limit: int = 20,
    include_core_ops: bool = True,
) -> dict[str, Any]:
    """Collect a human/debug-friendly snapshot of task processing."""
    task_signatures = compute_task_signatures(demos)
    demo_summaries = []
    for index, demo in enumerate(demos):
        sg_in, sg_out, delta = extract_with_delta(demo.input, demo.output)
        demo_summaries.append({
            "demo_idx": index,
            "input": summarize_state_graph(sg_in),
            "output": summarize_state_graph(sg_out),
            "delta": summarize_delta(delta),
        })

    retrieval = inspect_retrieval(
        demos,
        program_store,
        limit=retrieval_limit,
    )

    traces: list[SearchTraceEntry] = []

    def observer(entry: SearchTraceEntry) -> None:
        if len(traces) < search_trace_limit:
            traces.append(entry)

    search_result = search_program(
        demos,
        library,
        program_store=program_store,
        max_steps=max_search_steps,
        max_candidates=max_search_candidates,
        include_core_ops=include_core_ops,
        observer=observer,
    )

    return {
        "task_signatures": sorted(task_signatures),
        "output_size": inspect_output_size(demos, test_inputs=test_inputs),
        "demos": demo_summaries,
        "retrieval": [asdict(item) for item in retrieval],
        "search": {
            "solved": search_result.solved,
            "candidates_tried": search_result.candidates_tried,
            "winning_program": (
                program_to_text(search_result.winning_program)
                if search_result.winning_program is not None
                else None
            ),
            "trace": [asdict(entry) for entry in traces],
        },
    }


def inspect_output_size(
    demos: tuple[DemoPair, ...],
    *,
    test_inputs: tuple[Grid, ...] = (),
) -> dict[str, Any]:
    predictor = get_op("predict_dims")[1]
    full_ctx = TaskContext(demos=demos)
    classification, details = classify_output_size_pattern(demos)

    demo_predictions = []
    for index, demo in enumerate(demos):
        input_dims = grid_dims(demo.input)
        output_dims = grid_dims(demo.output)
        full_prediction = tuple(int(v) for v in predictor(full_ctx, demo.input))
        loo_ctx = TaskContext(demos=tuple(
            pair for pair_index, pair in enumerate(demos) if pair_index != index
        ))
        loo_prediction = tuple(int(v) for v in predictor(loo_ctx, demo.input))
        demo_predictions.append({
            "demo_idx": index,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "full_context_prediction": full_prediction,
            "full_context_matches_output": full_prediction == output_dims,
            "loo_prediction": loo_prediction,
            "loo_matches_output": loo_prediction == output_dims,
        })

    test_predictions = []
    for index, grid in enumerate(test_inputs):
        input_dims = grid_dims(grid)
        predicted_dims = tuple(int(v) for v in predictor(full_ctx, grid))
        test_predictions.append({
            "test_idx": index,
            "input_dims": input_dims,
            "predicted_output_dims": predicted_dims,
        })

    return {
        "classification": classification,
        "details": details,
        "demos": demo_predictions,
        "tests": test_predictions,
    }


def inspect_retrieval(
    demos: tuple[DemoPair, ...],
    program_store: ProgramStore,
    *,
    limit: int = 10,
) -> list[RetrievalInspection]:
    task_signatures = compute_task_signatures(demos)
    results: list[RetrievalInspection] = []

    for rank, record in enumerate(program_store.ranked_records(task_signatures), start=1):
        if limit and rank > limit:
            break

        overlap = tuple(sorted(task_signatures & set(record.signatures)))

        try:
            program = parse_program(record.program_text)
        except ParseError as exc:
            results.append(RetrievalInspection(
                rank=rank,
                task_ids=record.task_ids,
                sources=record.sources,
                use_count=record.use_count,
                step_count=record.step_count,
                signature_overlap=len(overlap),
                matching_signatures=overlap,
                status="parse_error",
                error_type=str(exc),
                program_text=record.program_text,
            ))
            continue

        type_errors = type_check(
            program,
            initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
        )
        if type_errors:
            results.append(RetrievalInspection(
                rank=rank,
                task_ids=record.task_ids,
                sources=record.sources,
                use_count=record.use_count,
                step_count=record.step_count,
                signature_overlap=len(overlap),
                matching_signatures=overlap,
                status="type_error",
                error_type="; ".join(type_errors[:4]),
                program_text=record.program_text,
            ))
            continue

        verify_result = verify(program, demos)
        results.append(RetrievalInspection(
            rank=rank,
            task_ids=record.task_ids,
            sources=record.sources,
            use_count=record.use_count,
            step_count=record.step_count,
            signature_overlap=len(overlap),
            matching_signatures=overlap,
            status="pass" if verify_result.passed else "verify_fail",
            failed_demo=verify_result.failed_demo,
            error_type=verify_result.error_type,
            program_text=record.program_text,
        ))

    return results


def summarize_state_graph(state_graph: StateGraph) -> dict[str, Any]:
    context = summarize_context(state_graph.context)
    partition = None
    if state_graph.partition is not None:
        partition = {
            "separator_color": state_graph.partition.separator_color,
            "n_rows": state_graph.partition.n_rows,
            "n_cols": state_graph.partition.n_cols,
            "is_uniform": state_graph.partition.is_uniform_partition,
            "cells": [
                {
                    "row_idx": cell.row_idx,
                    "col_idx": cell.col_idx,
                    "bbox": cell.bbox,
                    "dims": cell.dims,
                    "background": cell.background,
                    "palette": sorted(cell.palette),
                    "obj_count": cell.obj_count,
                }
                for cell in state_graph.partition.cells
            ],
        }

    legend = None
    if state_graph.legend is not None:
        legend = {
            "edge": state_graph.legend.edge,
            "bbox": state_graph.legend.region_bbox,
            "entries": [
                {"key_color": entry.key_color, "value_color": entry.value_color}
                for entry in state_graph.legend.entries
            ],
        }

    return {
        "context": context,
        "object_bboxes": [obj.bbox for obj in state_graph.objects],
        "relation_count": len(state_graph.relations),
        "roles": [
            {
                "role": binding.role.name,
                "object_id": binding.object_id,
                "bbox": binding.bbox,
                "color": binding.color,
                "tags": list(binding.tags),
            }
            for binding in state_graph.roles
        ],
        "partition": partition,
        "legend": legend,
    }


def summarize_context(context: GridContext) -> dict[str, Any]:
    return {
        "dims": context.dims,
        "bg_color": context.bg_color,
        "palette": sorted(context.palette),
        "obj_count": context.obj_count,
        "is_tiled": context.is_tiled,
        "symmetry": sorted(sym.name for sym in context.symmetry),
    }


def summarize_delta(delta: Delta) -> dict[str, Any]:
    return {
        "added": len(delta.added),
        "removed": len(delta.removed),
        "modified": len(delta.modified),
        "dims_changed": delta.dims_changed,
        "modified_fields": [field for _obj_id, field, _old, _new in delta.modified],
    }


def classify_output_size_pattern(demos: tuple[DemoPair, ...]) -> tuple[str, dict[str, Any]]:
    if not demos:
        return ("unknown", {})

    dims = [
        (*grid_dims(demo.input), *grid_dims(demo.output))
        for demo in demos
    ]

    out_dims_set = {(out_rows, out_cols) for _, _, out_rows, out_cols in dims}
    if len(out_dims_set) == 1:
        out_rows, out_cols = next(iter(out_dims_set))
        return ("fixed_output_dims", {"output_dims": (out_rows, out_cols)})

    if all(in_rows == out_rows and in_cols == out_cols for in_rows, in_cols, out_rows, out_cols in dims):
        return ("same_as_input", {})

    row_ratios = [out_rows / in_rows for in_rows, _, out_rows, _ in dims if in_rows > 0]
    col_ratios = [out_cols / in_cols for _, in_cols, _, out_cols in dims if in_cols > 0]
    if row_ratios and col_ratios and _all_close(row_ratios) and _all_close(col_ratios):
        return (
            "multiplicative",
            {"row_ratio": row_ratios[0], "col_ratio": col_ratios[0]},
        )

    row_deltas = [out_rows - in_rows for in_rows, _, out_rows, _ in dims]
    col_deltas = [out_cols - in_cols for _, in_cols, _, out_cols in dims]
    if _all_equal(row_deltas) and _all_equal(col_deltas):
        return (
            "additive",
            {"row_delta": row_deltas[0], "col_delta": col_deltas[0]},
        )

    return ("content_derived_or_unresolved", {})


def grid_dims(grid: Grid) -> tuple[int, int]:
    return (int(grid.shape[0]), int(grid.shape[1]))


def _all_close(values: list[float], tol: float = 1e-9) -> bool:
    if not values:
        return True
    ref = values[0]
    return all(abs(value - ref) < tol for value in values)


def _all_equal(values: list[int]) -> bool:
    if not values:
        return True
    return all(value == values[0] for value in values)
