"""Search engine: enumerate and verify SearchPrograms.

Uses seed schemas to generate candidates, verifies against demos,
returns verified programs as AST.

This is the flat-search layer. Future: add beam search, MCTS,
learned proposal/value models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.search.sketch import SearchProgram
from aria.search.seeds import build_seed_registry, SeedSchema, _color_selects
from aria.search.ast import ASTNode, ASTProgram
from aria.search.proposal_memory import load_default_search_prior
from aria.search.candidate_rank import rank_search_candidates
from aria.search.proposal_model import load_default_search_family_model
from aria.search.macros import load_default_macro_library
from aria.graph.signatures import compute_task_signatures
from aria.types import DemoPair


def search_programs(
    demos: list[tuple[np.ndarray, np.ndarray]],
    time_budget: float = 10.0,
) -> ASTProgram | None:
    """Search for a program that solves all demos.

    Priority order:
    1. Correspondence-derived programs (structural analysis)
    2. Seed schema enumeration (blind parameter search)
    3. 2-step compositions
    """
    import time as _time
    import signal as _signal
    _deadline = _time.monotonic() + time_budget

    class _SearchTimeout(BaseException):
        pass

    def _alarm_handler(signum, frame):
        raise _SearchTimeout()

    _old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
    _old_timer = _signal.setitimer(_signal.ITIMER_REAL, max(time_budget, 0.05))

    def _expired():
        return _time.monotonic() > _deadline

    try:
        return _search_programs_inner(demos, time_budget, _deadline, _expired)
    except _SearchTimeout:
        return None
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, _old_handler)


def _search_programs_inner(demos, time_budget, _deadline, _expired):
    """Core search logic, separated so the signal timer can interrupt it."""
    task_signatures = compute_task_signatures(
        tuple(DemoPair(input=inp, output=out) for inp, out in demos)
    )
    proposal_prior = load_default_search_prior()
    proposal_model = load_default_search_family_model()
    macro_library = load_default_macro_library()

    # Task analysis (cheap, run once, reusable by other tracks)
    from aria.search.task_analysis import analyze_task
    from aria.search.output_dims import solve_output_dims
    analysis = analyze_task(demos)
    dim_hypotheses = solve_output_dims(demos, analysis) if analysis.dims_change else []

    # Phase 0a: Correspondence-derived programs (structural transitions)
    from aria.search.derive import derive_programs
    derived = proposal_prior.rank_programs(derive_programs(demos, deadline=_deadline), task_signatures)
    for prog in derived:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc, search_program=prog)

    if _expired():
        return None

    # Phase 0b: Cross-panel structural reasoning
    from aria.search.panels import derive_panel_programs
    panel_progs = proposal_prior.rank_programs(derive_panel_programs(demos), task_signatures)
    for prog in panel_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc, search_program=prog)

    if _expired():
        return None

    # Phase 0c: Panel algebra (odd-select, majority-select, etc.)
    from aria.search.panel_ops import derive_panel_algebra_programs
    panel_alg_progs = proposal_prior.rank_programs(
        derive_panel_algebra_programs(demos),
        task_signatures,
    )
    for prog in panel_alg_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc, search_program=prog)

    if _expired():
        return None

    # Phase 0d: Region decode/transfer programs (panel/legend tasks)
    from aria.search.decode import derive_region_programs
    region_progs = proposal_prior.rank_programs(derive_region_programs(demos), task_signatures)
    for prog in region_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc, search_program=prog)

    if _expired():
        return None

    # Phase 0e: Binding-guided decode (uses role/relation substrate)
    from aria.search.binding_derive import derive_from_binding
    binding_progs = proposal_prior.rank_programs(derive_from_binding(demos), task_signatures)
    for prog in binding_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc, search_program=prog)

    if _expired():
        return None

    # Phase 0f: Decomposition search (analysis-gated splitter + sub-derive)
    from aria.search.decompose import search_decomposed
    decomp = search_decomposed(demos, analysis)
    if decomp is not None:
        ast = decomp.to_ast()
        desc = f"search: {decomp.provenance} [{decomp.signature}]"
        return ASTProgram(ast, desc, search_program=decomp)

    if _expired():
        return None

    registry = proposal_prior.rank_schemas(build_seed_registry(), task_signatures)

    # Phase 1: single-step schemas
    for schema in registry:
        if _expired():
            break
        candidates = schema.enumerate(demos)
        # Also try color-based selectors for object actions
        if schema.select_options != [None]:
            color_sels = _color_selects(demos)
            for sel in color_sels:
                for params in schema.get_param_options(demos):
                    from aria.search.sketch import SearchStep
                    step = SearchStep(action=schema.action, params=params, select=sel)
                    candidates.append(SearchProgram(steps=[step], provenance=schema.name))

        candidates = rank_search_candidates(
            candidates,
            demos,
            task_signatures=task_signatures,
            prior=proposal_prior,
            model=proposal_model,
            macro_library=macro_library,
            max_demos=min(2, len(demos)),
        )
        for prog in candidates:
            if prog.verify(demos):
                ast = prog.to_ast()
                desc = f"search: {prog.provenance} [{prog.signature}]"
                return ASTProgram(ast, desc)

    if _expired():
        return None

    # Phase 2: 2-step compositions (future: beam search)
    # For now: try pairs of single-step programs
    verified_singles = []
    for schema in registry:
        if _expired():
            break
        partials = rank_search_candidates(
            schema.enumerate(demos),
            demos,
            task_signatures=task_signatures,
            prior=proposal_prior,
            model=proposal_model,
            macro_library=macro_library,
            max_demos=min(2, len(demos)),
        )
        for prog in partials[:8]:
            try:
                mid = prog.execute(demos[0][0])
                if not np.array_equal(mid, demos[0][0]):
                    verified_singles.append(prog)
            except Exception:
                pass
        if len(verified_singles) > 50:
            break

    verified_singles = rank_search_candidates(
        verified_singles,
        demos,
        task_signatures=task_signatures,
        prior=proposal_prior,
        model=proposal_model,
        max_demos=min(2, len(demos)),
    )

    for p1 in verified_singles[:20]:
        if _expired():
            break
        for p2 in verified_singles[:20]:
            if _expired():
                break
            if p1.signature == p2.signature:
                continue
            combined = SearchProgram(
                steps=p1.steps + p2.steps,
                provenance=f"{p1.provenance}+{p2.provenance}",
            )
            if combined.verify(demos):
                ast = combined.to_ast()
                desc = f"search: {combined.provenance} [{combined.signature}]"
                return ASTProgram(ast, desc)

    return None
