"""Build a persistent Level-1 library and optional frozen snapshot.

Usage:
    python scripts/build_library.py
    python scripts/build_library.py --program-source results/v1-train.json
    python scripts/build_library.py --snapshot-dir results/snapshots/v1_bootstrap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria.library.builder import build_library_from_store
from aria.library.graph import build_abstraction_graph
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.snapshot import write_snapshot


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_PROGRAM_STORE = RESULTS_DIR / "program_store.json"
DEFAULT_LIBRARY_STORE = RESULTS_DIR / "library.json"
DEFAULT_STATS_OUTPUT = RESULTS_DIR / "library_build.json"
DEFAULT_GRAPH_OUTPUT = RESULTS_DIR / "abstraction_graph.json"


def _load_program_store(store_path: Path, extra_sources: list[str]) -> tuple[ProgramStore, list[str]]:
    store = ProgramStore.load_json(store_path)
    imported_notes: list[str] = []

    sources = [Path(source) for source in extra_sources]
    if not store and not sources:
        sources = [
            RESULTS_DIR / "v1-train.json",
            Path(__file__).parent / "corpus_report.json",
        ]

    for source in sources:
        if source == store_path or not source.exists():
            continue
        imported = store.import_path(source)
        if imported:
            imported_notes.append(f"{imported} from {source}")

    if imported_notes:
        store.save_json(store_path)

    return store, imported_notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent library from verified programs")
    parser.add_argument("--program-store", default=str(DEFAULT_PROGRAM_STORE))
    parser.add_argument("--program-source", action="append", default=[])
    parser.add_argument("--library-output", default=str(DEFAULT_LIBRARY_STORE))
    parser.add_argument("--stats-output", default=str(DEFAULT_STATS_OUTPUT))
    parser.add_argument("--graph-output", default=str(DEFAULT_GRAPH_OUTPUT))
    parser.add_argument("--snapshot-dir", help="Write a frozen snapshot directory")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument("--min-uses", type=int, default=2)
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--prune-weak", action="store_true",
                        help="Remove entries with single-task support and zero MDL gain")
    args = parser.parse_args()

    program_store_path = Path(args.program_store)
    program_store, imported = _load_program_store(program_store_path, args.program_source)
    if len(program_store) == 0:
        raise SystemExit("No verified programs available to build a library")

    library, report = build_library_from_store(
        program_store,
        min_length=args.min_length,
        max_length=args.max_length,
        min_uses=args.min_uses,
        max_entries=args.max_entries,
    )
    pruned_count = 0
    if args.prune_weak:
        pruned_count = library.prune_weak()
    library_output = Path(args.library_output)
    library.save_json(library_output)
    graph = build_abstraction_graph(program_store, library)
    graph_output = Path(args.graph_output)
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    graph_data = graph.to_dict()
    with open(graph_output, "w") as f:
        json.dump(graph_data, f, indent=2)

    def _strength(entry):
        if len(entry.support_task_ids) >= 2 and entry.mdl_gain > 0:
            return "strong"
        return "weak"

    stats = {
        "program_store": str(program_store_path),
        "library_output": str(library_output),
        "graph_output": str(graph_output),
        "corpus_programs": report.corpus_programs,
        "corpus_tasks": report.corpus_tasks,
        "candidates_mined": report.candidates_mined,
        "aggregated_candidates": report.aggregated_candidates,
        "admitted_count": len(report.admitted_entries),
        "admitted_entries": [
            {
                "name": entry.name,
                "params": [[name, typ.name] for name, typ in entry.params],
                "return_type": entry.return_type.name,
                "use_count": entry.use_count,
                "step_count": len(entry.steps),
                "support_task_ids": list(entry.support_task_ids),
                "support_program_count": entry.support_program_count,
                "mdl_gain": entry.mdl_gain,
                "strength": _strength(entry),
            }
            for entry in report.admitted_entries
        ],
        "rejected_reasons": report.rejected_reasons,
        "imported": imported,
        "graph_summary": graph_data.get("summary", {}),
        "pruned_weak": pruned_count,
        "config": {
            "min_length": args.min_length,
            "max_length": args.max_length,
            "min_uses": args.min_uses,
            "max_entries": args.max_entries,
            "prune_weak": args.prune_weak,
        },
    }
    stats_output = Path(args.stats_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_output, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Program store: {program_store_path} ({len(program_store)} programs)")
    for note in imported:
        print(f"Imported {note}")
    print(f"Library: {library_output} ({len(library.all_entries())} entries)")
    if pruned_count:
        print(f"Pruned: {pruned_count} weak entries")
    print(f"Graph: {graph_output}")
    print(f"Stats: {stats_output}")
    summary = stats.get("graph_summary", {})
    if summary:
        print(
            f"Transfer quality: "
            f"strong={summary.get('strong_abstractions', 0)}, "
            f"weak={summary.get('weak_abstractions', 0)}, "
            f"transfer-backed leaves={summary.get('transfer_backed_leaves', 0)}, "
            f"one-off leaves={summary.get('one_off_leaves', 0)}"
        )

    if args.snapshot_dir:
        manifest = write_snapshot(
            args.snapshot_dir,
            program_store=program_store,
            library=library,
            metadata={
                "builder": "scripts/build_library.py",
                "program_store": str(program_store_path),
                "library_output": str(library_output),
                "config": stats["config"],
            },
        )
        print(f"Snapshot: {manifest.parent}")
        print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
