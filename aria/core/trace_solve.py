"""Instrumented canonical solve path — produces a TaskTrace artifact.

Runs the full canonical pipeline (static -> deterministic -> learned)
with tracing hooks that capture every seed, edit, compile, and verify
event for single-task debugging.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from aria.core.arc import ARCCompiler, ARCFitter, ARCSpecializer, ARCVerifier
from aria.core.graph import CompileSuccess, Specialization
from aria.core.output_stage1 import (
    compile_stage1_program,
    infer_output_stage1_spec,
)
from aria.core.protocol import solve as core_solve
from aria.core.trace import TaskTrace, _safe_val
from aria.runtime.program import program_to_text


def _stage1_spec_to_dict(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    return _safe_val(asdict(spec))


def _trace_relation_diagnostics(demos: tuple) -> dict[str, Any]:
    """Gather relation layer diagnostics for tracing."""
    try:
        from aria.core.grid_perception import perceive_grid
        from aria.core.relations import (
            build_legend_mapping,
            detect_slot_grid,
            verify_zone_summary_grid,
        )

        if not demos:
            return {}

        state = perceive_grid(demos[0].input)
        info: dict[str, Any] = {}

        sg = detect_slot_grid(state)
        if sg is not None:
            info["slot_grid"] = f"{sg.n_rows}x{sg.n_cols}"

        lm = build_legend_mapping(state)
        if lm is not None:
            info["legend"] = {
                "edge": lm.edge,
                "entries": dict(lm.key_to_value),
            }

        info["zones"] = len(state.zones)
        info["roles"] = [
            {"role": r.role.name, "color": r.color}
            for r in state.roles
        ]

        zsm = verify_zone_summary_grid(demos)
        if zsm is not None:
            info["zone_summary"] = {
                "kind": zsm.mapping_kind,
                "property": zsm.params.get("property"),
            }

        return info
    except Exception:
        return {}


def _try_scene_programs_in_trace(trace, demos) -> bool:
    """Try scene program inference and return True if solved."""
    try:
        from aria.core.scene_solve import infer_scene_programs, verify_scene_program
        scene_progs = infer_scene_programs(demos)
        for sp in scene_progs:
            if verify_scene_program(sp, demos):
                step_names = [s.op.value for s in sp.steps]
                trace.add_event(
                    "scene",
                    "scene_program_solved",
                    steps=step_names,
                    n_steps=len(sp.steps),
                )
                trace.static_result = {
                    "solved": True,
                    "graphs_proposed": 0,
                    "graphs_compiled": 0,
                    "graphs_verified": 0,
                }
                trace.solved = True
                trace.solver = "scene"
                return True
        if scene_progs:
            trace.add_event(
                "scene",
                "scene_program_attempted",
                n_candidates=len(scene_progs),
                verified=False,
            )
    except Exception:
        pass
    return False


def solve_with_trace(
    demos: tuple,
    task_id: str = "",
    *,
    use_deterministic: bool = True,
    use_learned: bool = True,
) -> TaskTrace:
    """Run the canonical pipeline on one task and return the full trace."""
    trace = TaskTrace(task_id=task_id)
    trace.set_demos(demos)

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    # ---- Stage 1: output-size / direct-derivation gate ----
    trace.add_event("stage1", "phase_start")
    stage1 = infer_output_stage1_spec(demos)

    # Trace relation layer diagnostics on first demo
    relation_info = _trace_relation_diagnostics(demos)
    if relation_info:
        trace.add_event("stage1", "relation_diagnostics", **relation_info)

    trace.add_event(
        "stage1",
        "output_spec",
        verified=stage1 is not None,
        spec=_stage1_spec_to_dict(stage1),
        note="stage1 = infer output size first; only then infer direct derivation",
    )
    trace.add_event("stage1", "phase_end", verified=stage1 is not None)

    if stage1 is None:
        # Try scene programs even without stage-1 size
        scene_solved = _try_scene_programs_in_trace(trace, demos)
        if scene_solved:
            return trace

        trace.static_result = {
            "solved": False,
            "graphs_proposed": 0,
            "graphs_compiled": 0,
            "graphs_verified": 0,
        }
        trace.add_event(
            "static",
            "skipped",
            reason="stage1_gate_failed",
            note="no verified output-size spec; canonical solve path not attempted",
        )
        return trace

    stage1_program = compile_stage1_program(stage1)
    if stage1_program is not None:
        vr = verifier.verify(stage1_program, demos)
        trace.add_event(
            "stage1", "direct_stage1_program",
            compiled=True, verified=vr.passed,
            program_text=program_to_text(stage1_program),
        )
        if vr.passed:
            trace.static_result = {
                "solved": True,
                "graphs_proposed": 0,
                "graphs_compiled": 0,
                "graphs_verified": 0,
            }
            trace.solved = True
            trace.solver = "stage1"
            return trace

    # ---- Phase 0.5: multi-step scene programs ----
    scene_solved = _try_scene_programs_in_trace(trace, demos)
    if scene_solved:
        return trace

    # ---- Phase 1: static pipeline ----
    trace.add_event("static", "phase_start")

    static_result = core_solve(
        examples=demos,
        fitter=fitter,
        specializer=specializer,
        compiler=compiler,
        verifier=verifier,
        task_id=task_id,
    )

    trace.static_result = {
        "solved": static_result.solved,
        "graphs_proposed": static_result.graphs_proposed,
        "graphs_compiled": static_result.graphs_compiled,
        "graphs_verified": static_result.graphs_verified,
    }

    # Log each attempt from static pipeline
    for i, attempt in enumerate(static_result.attempts):
        from aria.core.trace import _graph_to_dict, _spec_to_dict, _compile_result_to_dict
        trace.add_event("static", "attempt",
            index=i,
            graph=_graph_to_dict(attempt.graph),
            specialization=_spec_to_dict(attempt.specialization),
            compile_result=_compile_result_to_dict(attempt.compile_result),
            verified=attempt.verified,
        )

    # Compute mechanism evidence for trace (class_fit only — executable_fit comes from compile)
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    evidence, ranking = compute_evidence_and_rank(demos)
    trace.add_event("static", "mechanism_class_fit",
        evidence={k: v for k, v in asdict(evidence).items()},
        ranking=[{"name": c.name, "class_score": round(c.class_score, 3),
                  "exec_hint": round(c.exec_hint, 3),
                  "anti_evidence": c.anti_evidence,
                  "final_score": round(c.final_score, 3),
                  "rationale": c.rationale, "gate_pass": c.gate_pass}
                 for c in ranking.lanes],
        top_hypothesis=ranking.description,
        note="class_fit = structural hypothesis; exec_hint = sufficiency for current op; anti_evidence = contradiction signals",
    )

    trace.add_event("static", "phase_end", solved=static_result.solved)

    if static_result.solved:
        trace.solved = True
        trace.solver = "static"
        return trace

    # ---- Collect seeds (shared for phases 2 and 3) ----
    from aria.core.seeds import collect_seeds

    seeds = collect_seeds(
        examples=demos,
        fitter=fitter,
        specializer=specializer,
        compiler=compiler,
        verifier=verifier,
        task_id=task_id,
    )

    for seed in seeds:
        trace.add_seed(
            provenance=seed.provenance,
            graph=seed.graph,
            spec=seed.specialization,
            verified=seed.already_verified,
        )

    trace.add_event("seeds", "collected", count=len(seeds),
                     by_provenance={
                         p: sum(1 for s in seeds if s.provenance == p)
                         for p in set(s.provenance for s in seeds)
                     })

    # ---- Phase 2: deterministic editor search ----
    if use_deterministic:
        trace.add_event("deterministic", "phase_start")

        from aria.core.editor_search import search_from_seeds
        from aria.core.editor_env import GraphEditEnv, EditAction, ActionType

        det_result = search_from_seeds(
            seeds=seeds,
            examples=demos,
            specializer=specializer,
            compiler=compiler,
            verifier=verifier,
            task_id=task_id,
        )

        trace.deterministic_result = {
            "solved": det_result.solved,
            "frontier_expansions": det_result.frontier_expansions,
            "compiles_attempted": det_result.compiles_attempted,
            "unique_states_seen": det_result.unique_states_seen,
            "max_depth_reached": det_result.max_depth_reached,
            "description": det_result.description,
        }

        trace.add_event("deterministic", "phase_end",
                         **trace.deterministic_result)

        if det_result.solved:
            trace.solved = True
            trace.solver = "deterministic"
            return trace

    # ---- Phase 3: learned editor ----
    if use_learned:
        trace.add_event("learned", "phase_start")

        from aria.core.editor_train import train_and_solve

        learn_result = train_and_solve(
            seeds=seeds,
            examples=demos,
            specializer=specializer,
            compiler=compiler,
            verifier=verifier,
            task_id=task_id,
            n_rounds=3,
            n_trajectories=8,
        )

        trace.learned_result = {
            "solved": learn_result.solved,
            "rounds_run": learn_result.rounds_run,
            "trajectories_sampled": learn_result.trajectories_sampled,
            "elites_kept": learn_result.elites_kept,
            "compiles_attempted": learn_result.compiles_attempted,
            "max_depth_reached": learn_result.max_depth_reached,
            "description": learn_result.description,
        }

        trace.add_event("learned", "phase_end",
                         **trace.learned_result)

        if learn_result.solved:
            trace.solved = True
            trace.solver = "learned"

    return trace
