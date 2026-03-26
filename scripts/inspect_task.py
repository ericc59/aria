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
    parser.add_argument("--max-refinement-rounds", type=int, default=2)
    parser.add_argument("--search-trace-limit", type=int, default=20)
    parser.add_argument("--library-only", action="store_true")
    parser.add_argument("--beam-width", type=int, default=0, help="Beam width (0=disabled)")
    parser.add_argument("--beam-rounds", type=int, default=3)
    parser.add_argument(
        "--policy",
        choices=["heuristic", "local_lm_dry"],
        default="heuristic",
        help="Refinement policy mode (default: heuristic)",
    )
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

    local_policy = None
    if args.policy == "local_lm_dry":
        from aria.local_policy import LocalCausalLMPolicy
        local_policy = LocalCausalLMPolicy(dry_run=True)

    inspection = inspect_task(
        task.train,
        library=library,
        program_store=program_store,
        test_inputs=tuple(pair.input for pair in task.test),
        retrieval_limit=args.retrieval_limit,
        max_search_steps=args.max_search_steps,
        max_search_candidates=args.max_search_candidates,
        max_refinement_rounds=args.max_refinement_rounds,
        search_trace_limit=args.search_trace_limit,
        include_core_ops=not args.library_only,
        beam_width=args.beam_width,
        beam_rounds=args.beam_rounds,
        local_policy=local_policy,
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
        dtc = item.get("distinct_task_count", 0)
        echo = item.get("retrieval_echo_count", 0)
        non_retr = item.get("has_non_retrieval_source", False)
        transfer_tag = " [transfer]" if dtc >= 2 else " [leaf]"
        print(
            f"  #{item['rank']} {item['status']}{transfer_tag} "
            f"overlap={item['signature_overlap']} use={item['use_count']} "
            f"distinct_tasks={dtc} echo={echo} "
            f"non_retrieval={non_retr} "
            f"tasks={list(item['task_ids'])}"
        )
        if item["matching_signatures"]:
            print(f"    matching: {', '.join(item['matching_signatures'])}")
        print(f"    sources: {', '.join(item['sources'])}")
        print("    program:")
        for line in item["program_text"].splitlines():
            print(f"      {line}")
    print()

    abs_ret = inspection.get("abstraction_retrieval", {})
    if abs_ret.get("hints"):
        print("Abstraction Retrieval")
        for h in abs_ret["hints"]:
            used_mark = " *USED*" if h["name"] in abs_ret.get("used_in_winning_program", []) else ""
            print(
                f"  {h['name']} [{h['strength']}] "
                f"score={h['score']} "
                f"tasks={h['support_task_count']} "
                f"programs={h['support_program_count']} "
                f"mdl_gain={h['mdl_gain']}{used_mark}"
            )
        if abs_ret.get("solved_with_retrieved_abstraction"):
            print(f"  -> Solved with retrieved abstraction: {abs_ret['used_in_winning_program']}")
        sk = abs_ret.get("skeleton_hypotheses")
        if sk:
            print(
                f"  Skeleton hypotheses: {sk['skeletons_tested']} tested / "
                f"{sk['skeletons_generated']} generated"
                f"{' -> SOLVED' if sk['solved'] else ''}"
            )
            for hyp in sk.get("hypotheses", []):
                status = "PASS" if hyp["passed"] else f"FAIL/{hyp['error_type']}"
                print(f"    [{status}] {hyp['source']}")
                for line in hyp["program_text"].splitlines():
                    print(f"      {line}")
        print()

    print("Refinement")
    refinement = inspection["refinement"]
    print(
        f"  solved={refinement['solved']} "
        f"candidates_tried={refinement['candidates_tried']}"
    )
    if refinement["winning_program"]:
        print("  winning_program:")
        for line in refinement["winning_program"].splitlines():
            print(f"    {line}")
    if refinement["rounds"]:
        print("  rounds:")
        for round_data in refinement["rounds"]:
            plan = round_data["plan"]
            feedback = round_data["feedback"]
            print(
                f"    round {round_data['round_idx']}: "
                f"{plan['name']} "
                f"(steps={plan['max_steps']}, cand={plan['max_candidates']}, tried={round_data['candidates_tried']}) "
                f"focus={feedback['suggested_focus']} policy={round_data.get('policy_source', 'heuristic')} solved={round_data['solved']}"
            )
            if plan["allowed_ops"]:
                print(f"      allowed_ops={plan['allowed_ops']}")
            if feedback["best_candidate_num"] is not None:
                print(
                    "      best_candidate="
                    f"{feedback['best_candidate_num']} "
                    f"score={feedback['best_candidate_score']:.1f} "
                    f"err={feedback['best_candidate_error_type']} "
                    f"dims_match={feedback['best_candidate_dims_match']} "
                    f"pixel_diff={feedback['best_candidate_pixel_diff_count']} "
                    f"wrong_rows={feedback['best_candidate_wrong_row_count']} "
                    f"wrong_cols={feedback['best_candidate_wrong_col_count']}"
                )
                if round_data["best_candidate_program"]:
                    print("      best_candidate_program:")
                    for line in round_data["best_candidate_program"].splitlines():
                        print(f"        {line}")
            if round_data["winning_program"]:
                print("      winning_program:")
                for line in round_data["winning_program"].splitlines():
                    print(f"        {line}")
            if round_data["trace"]:
                print("      trace:")
                for entry in round_data["trace"]:
                    status = "PASS" if entry["passed"] else f"FAIL/{entry['error_type']}"
                    score_suffix = (
                        f" score={entry['score']:.1f}"
                        if entry["score"] is not None
                        else ""
                    )
                    print(
                        f"        [{entry['candidate_num']}] depth={entry['depth']} {status}{score_suffix}"
                    )
                    if entry["failed_demo"] is not None:
                        print(f"          failed_demo={entry['failed_demo']}")
                    if entry["diff"] is not None:
                        print(f"          diff={entry['diff']}")
                    if entry["score_reasons"]:
                        print(f"          score_reasons={entry['score_reasons']}")
                    for line in entry["program_text"].splitlines():
                        print(f"          {line}")

    beam = refinement.get("beam")
    if beam is not None:
        print()
        print("Beam Refinement")
        print(
            f"  solved={beam['solved']} "
            f"candidates_scored={beam['candidates_scored']} "
            f"rounds={beam['rounds_completed']}"
        )
        if beam["best_program_text"]:
            print(f"  best_score={beam['best_score']}")
            print("  best_program:")
            for line in beam["best_program_text"].splitlines():
                print(f"    {line}")
        for rs in beam.get("round_summaries", []):
            print(
                f"  round {rs['round_idx']}: "
                f"mutations={rs['mutations_tried']} "
                f"improvements={rs['improvements']} "
                f"best_score={rs['best_score']}"
            )
        top_improvements = beam.get("top_improvements", [])
        if top_improvements:
            print(f"  top improvements ({len(top_improvements)}):")
            for t in top_improvements[:10]:
                print(
                    f"    round={t['round_idx']} {t['edit_kind']}: "
                    f"{t['detail']}"
                )
                print(
                    f"      score: {t['parent_score']} -> {t['child_score']}"
                )


if __name__ == "__main__":
    main()
