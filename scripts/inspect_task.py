#!/usr/bin/env python3
"""Inspect how ARIA perceives and searches a single ARC task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aria.inspection import inspect_task
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.solver import load_task


DATA_ROOTS = {
    "v1-train": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v1/training"
    ),
    "v1-eval": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v1/evaluation"
    ),
    "v2-train": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v2/training"
    ),
    "v2-eval": os.path.expanduser(
        "~/dev/arcagi/arc-agi-benchmarking/data/public-v2/evaluation"
    ),
}

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_PROGRAM_STORE = RESULTS_DIR / "program_store.json"
DEFAULT_LIBRARY_STORE = RESULTS_DIR / "library.json"


def _load_program_store(store_path: Path, extra_sources: list[str]) -> ProgramStore:
    store = ProgramStore.load_json(store_path)
    sources = [Path(source) for source in extra_sources]
    if not store and not sources:
        sources = [
            RESULTS_DIR / "v1-train.json",
            Path(__file__).parent / "corpus_report.json",
        ]

    for source in sources:
        if source == store_path or not source.exists():
            continue
        store.import_path(source)

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single ARC task in the offline ARIA system")
    parser.add_argument("--task", required=True, help="Task ID (e.g. 60c09cac)")
    parser.add_argument("--dataset", default="v1-eval", choices=DATA_ROOTS.keys())
    parser.add_argument("--program-store", default=str(DEFAULT_PROGRAM_STORE))
    parser.add_argument("--program-source", action="append", default=[])
    parser.add_argument("--library-store", default=str(DEFAULT_LIBRARY_STORE))
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--max-search-steps", type=int, default=3)
    parser.add_argument("--max-search-candidates", type=int, default=200)
    parser.add_argument("--search-trace-limit", type=int, default=20)
    parser.add_argument("--library-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        program_store_path = snapshot_dir / "program_store.json"
        library_path = snapshot_dir / "library.json"
    else:
        program_store_path = Path(args.program_store)
        library_path = Path(args.library_store)

    task_path = Path(DATA_ROOTS[args.dataset]) / f"{args.task}.json"
    if not task_path.exists():
        raise SystemExit(f"Task not found: {task_path}")

    with open(task_path) as f:
        task = load_task(json.load(f))

    program_store = _load_program_store(program_store_path, args.program_source)
    library = Library.load_json(library_path)

    inspection = inspect_task(
        task.train,
        library=library,
        program_store=program_store,
        test_inputs=tuple(pair.input for pair in task.test),
        retrieval_limit=args.retrieval_limit,
        max_search_steps=args.max_search_steps,
        max_search_candidates=args.max_search_candidates,
        search_trace_limit=args.search_trace_limit,
        include_core_ops=not args.library_only,
    )

    if args.json:
        print(json.dumps(inspection, indent=2, default=str))
        return

    print(f"Task: {args.task} ({args.dataset})")
    print(f"Train demos: {len(task.train)}")
    print(f"Program store: {program_store_path} ({len(program_store)} programs)")
    print(f"Library: {library_path} ({len(library.all_entries())} entries)")
    print()

    print("Signatures")
    print(", ".join(inspection["task_signatures"]) or "(none)")
    print()

    print("Output Size")
    output_size = inspection["output_size"]
    print(f"  classification: {output_size['classification']}")
    if output_size["details"]:
        print(f"  details: {output_size['details']}")
    for item in output_size["demos"]:
        print(
            f"  demo {item['demo_idx']}: "
            f"in={item['input_dims']} out={item['output_dims']} "
            f"full={item['full_context_prediction']} "
            f"full_match={item['full_context_matches_output']} "
            f"loo={item['loo_prediction']} "
            f"loo_match={item['loo_matches_output']}"
        )
    for item in output_size["tests"]:
        print(
            f"  test {item['test_idx']}: "
            f"in={item['input_dims']} predicted={item['predicted_output_dims']}"
        )
    print()

    print("Perception")
    for demo in inspection["demos"]:
        print(f"Demo {demo['demo_idx']}")
        print(f"  Input context: {demo['input']['context']}")
        if demo["input"]["partition"] is not None:
            print(f"  Input partition: {demo['input']['partition']}")
        if demo["input"]["legend"] is not None:
            print(f"  Input legend: {demo['input']['legend']}")
        if demo["input"]["roles"]:
            print(f"  Input roles: {demo['input']['roles']}")
        print(f"  Output context: {demo['output']['context']}")
        print(f"  Delta: {demo['delta']}")
        print()

    print("Retrieval")
    if not inspection["retrieval"]:
        print("  No retrieval candidates inspected")
    for item in inspection["retrieval"]:
        print(
            f"  #{item['rank']} {item['status']} "
            f"overlap={item['signature_overlap']} use={item['use_count']} "
            f"tasks={list(item['task_ids'])}"
        )
        if item["matching_signatures"]:
            print(f"    matching: {', '.join(item['matching_signatures'])}")
        print(f"    sources: {', '.join(item['sources'])}")
        print("    program:")
        for line in item["program_text"].splitlines():
            print(f"      {line}")
    print()

    print("Search")
    search = inspection["search"]
    print(
        f"  solved={search['solved']} "
        f"candidates_tried={search['candidates_tried']}"
    )
    if search["winning_program"]:
        print("  winning_program:")
        for line in search["winning_program"].splitlines():
            print(f"    {line}")
    if search["trace"]:
        print("  trace:")
        for entry in search["trace"]:
            status = "PASS" if entry["passed"] else f"FAIL/{entry['error_type']}"
            print(f"    [{entry['candidate_num']}] depth={entry['depth']} {status}")
            if entry["failed_demo"] is not None:
                print(f"      failed_demo={entry['failed_demo']}")
            if entry["diff"] is not None:
                print(f"      diff={entry['diff']}")
            for line in entry["program_text"].splitlines():
                print(f"      {line}")


if __name__ == "__main__":
    main()
