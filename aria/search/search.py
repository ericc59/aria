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
    # Phase 0a: Correspondence-derived programs (structural transitions)
    from aria.search.derive import derive_programs
    derived = derive_programs(demos)
    for prog in derived:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc)

    # Phase 0b: Cross-panel structural reasoning
    from aria.search.panels import derive_panel_programs
    panel_progs = derive_panel_programs(demos)
    for prog in panel_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc)

    # Phase 0c: Panel algebra (odd-select, majority-select, etc.)
    from aria.search.panel_ops import derive_panel_algebra_programs
    panel_alg_progs = derive_panel_algebra_programs(demos)
    for prog in panel_alg_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc)

    # Phase 0d: Region decode/transfer programs (panel/legend tasks)
    from aria.search.decode import derive_region_programs
    region_progs = derive_region_programs(demos)
    for prog in region_progs:
        ast = prog.to_ast()
        desc = f"search: {prog.provenance} [{prog.signature}]"
        return ASTProgram(ast, desc)

    registry = build_seed_registry()

    # Phase 1: single-step schemas
    for schema in registry:
        candidates = schema.enumerate(demos)
        # Also try color-based selectors for object actions
        if schema.select_options != [None]:
            color_sels = _color_selects(demos)
            for sel in color_sels:
                for params in schema.get_param_options(demos):
                    from aria.search.sketch import SearchStep
                    step = SearchStep(action=schema.action, params=params, select=sel)
                    candidates.append(SearchProgram(steps=[step], provenance=schema.name))

        for prog in candidates:
            if prog.verify(demos):
                ast = prog.to_ast()
                desc = f"search: {prog.provenance} [{prog.signature}]"
                return ASTProgram(ast, desc)

    # Phase 2: 2-step compositions (future: beam search)
    # For now: try pairs of single-step programs
    verified_singles = []
    for schema in registry:
        for prog in schema.enumerate(demos):
            # Check if this is a useful partial (reduces error)
            try:
                mid = prog.execute(demos[0][0])
                if not np.array_equal(mid, demos[0][0]):
                    verified_singles.append(prog)
            except Exception:
                pass
            if len(verified_singles) > 50:
                break

    for p1 in verified_singles[:20]:
        for p2 in verified_singles[:20]:
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
