"""Parameterized structural template search.

Uses grid-analysis-based parameter proposal to efficiently search
high-arity ops that brute-force enumeration can't reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.core.param_propose import (
    propose_color_map_params,
    propose_crop_params,
    propose_derive_output_params,
    propose_fill_params,
    propose_geometric_params,
    propose_relocate_params,
    propose_repair_params,
    propose_tiling_params,
)
from aria.runtime.program import program_to_text
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.verify.verifier import verify


@dataclass(frozen=True)
class TemplateSearchResult:
    solved: bool
    winning_program: Program | None = None
    winning_text: str = ""
    template_name: str = ""
    candidates_tested: int = 0


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

# Each template: (name, op_name, param_proposer, max_candidates)
_TEMPLATES = [
    # High-arity ops that the generic search can't reach
    ("tiling", "render_tiled_input_pattern", propose_tiling_params, 500),
    ("derive_output", "derive_output_from_input", propose_derive_output_params, 1000),
    # Ops where grid-derived params help
    ("relocate", "relocate_objects", propose_relocate_params, 49),
    ("relocate2", "match_and_place", propose_relocate_params, 49),
    ("repair_masked", "repair_masked_region", None, 8),
    # Multi-step templates
]

# Depth-2 template combinations
_DEPTH2_TEMPLATES = [
    # repair_framed_lines → repair_framed_2d_motif
    ("repair_chain", [
        ("repair_framed_lines", [(0, 2), (0, 3), (0, 4), (0, 5),
                                  (1, 2), (1, 3), (1, 4), (1, 5)]),
        ("repair_framed_2d_motif", [()]),
    ]),
    ("repair_then_fill", [
        ("repair_masked_region", [(i,) for i in range(8)]),
        ("fill_enclosed_regions_auto", [()]),
    ]),
]


def search_templates(
    demos: tuple[DemoPair, ...],
    *,
    max_candidates: int = 3000,
) -> TemplateSearchResult:
    """Search parameterized structural templates with proposed slot values."""
    total_tested = 0
    seen: set[str] = set()

    # Phase 1: single-op templates with proposed params
    for tpl_name, op_name, proposer, budget in _TEMPLATES:
        if proposer is not None:
            param_candidates = proposer(demos)
        else:
            # Simple enumeration for low-arity
            param_candidates = [(i,) for i in range(-1, 10)]

        for params in param_candidates[:budget]:
            if total_tested >= max_candidates:
                return TemplateSearchResult(solved=False, candidates_tested=total_tested)

            if isinstance(params, (int, float)):
                params = (params,)

            args = [Ref(name="input")] + [Literal(value=int(v), typ=Type.INT) for v in params]
            prog = Program(
                steps=(Bind(name="v0", typ=Type.GRID,
                            expr=Call(op=op_name, args=tuple(args))),),
                output="v0",
            )

            key = program_to_text(prog)
            if key in seen:
                continue
            seen.add(key)

            total_tested += 1
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    return TemplateSearchResult(
                        solved=True,
                        winning_program=prog,
                        winning_text=key,
                        template_name=tpl_name,
                        candidates_tested=total_tested,
                    )
            except Exception:
                pass

    # Phase 2: depth-2 template chains
    for chain_name, steps_def in _DEPTH2_TEMPLATES:
        if len(steps_def) != 2:
            continue

        op1_name, op1_params_list = steps_def[0]
        op2_name, op2_params_list = steps_def[1]

        for p1 in op1_params_list:
            for p2 in op2_params_list:
                if total_tested >= max_candidates:
                    return TemplateSearchResult(solved=False, candidates_tested=total_tested)

                args1 = [Ref(name="input")] + [Literal(value=int(v), typ=Type.INT) for v in p1]
                step1 = Bind(name="v0", typ=Type.GRID,
                             expr=Call(op=op1_name, args=tuple(args1)))

                args2 = [Ref(name="v0")] + [Literal(value=int(v), typ=Type.INT) for v in p2]
                step2 = Bind(name="v1", typ=Type.GRID,
                             expr=Call(op=op2_name, args=tuple(args2)))

                prog = Program(steps=(step1, step2), output="v1")
                key = program_to_text(prog)
                if key in seen:
                    continue
                seen.add(key)

                total_tested += 1
                try:
                    vr = verify(prog, demos)
                    if vr.passed:
                        return TemplateSearchResult(
                            solved=True,
                            winning_program=prog,
                            winning_text=key,
                            template_name=chain_name,
                            candidates_tested=total_tested,
                        )
                except Exception:
                    pass

    # Phase 3: grid-derived fill operations
    fill_params = propose_fill_params(demos)
    for fc_tuple in fill_params:
        fc = fc_tuple[0] if isinstance(fc_tuple, tuple) else fc_tuple
        for op_name in ("fill_enclosed", "fill_enclosed_regions",
                         "fill_enclosed_regions_auto", "fill_enclosed_per_composite",
                         "fill_enclosed_rarest_singleton"):
            if total_tested >= max_candidates:
                return TemplateSearchResult(solved=False, candidates_tested=total_tested)

            if op_name.endswith("_auto") or op_name.endswith("_singleton") or op_name.endswith("_composite"):
                args = [Ref(name="input")]
            else:
                args = [Ref(name="input"), Literal(value=int(fc), typ=Type.INT)]

            prog = Program(
                steps=(Bind(name="v0", typ=Type.GRID,
                            expr=Call(op=op_name, args=tuple(args))),),
                output="v0",
            )
            key = program_to_text(prog)
            if key in seen:
                continue
            seen.add(key)

            total_tested += 1
            try:
                vr = verify(prog, demos)
                if vr.passed:
                    return TemplateSearchResult(
                        solved=True,
                        winning_program=prog,
                        winning_text=key,
                        template_name="fill_guided",
                        candidates_tested=total_tested,
                    )
            except Exception:
                pass

    # Phase 4: derive_output_from_input (7-param, needs proposal)
    derive_params = propose_derive_output_params(demos)
    for params in derive_params[:1000]:
        if total_tested >= max_candidates:
            break

        args = [Ref(name="input")] + [Literal(value=int(v), typ=Type.INT) for v in params]
        prog = Program(
            steps=(Bind(name="v0", typ=Type.GRID,
                        expr=Call(op="derive_output_from_input", args=tuple(args))),),
            output="v0",
        )
        key = program_to_text(prog)
        if key in seen:
            continue
        seen.add(key)

        total_tested += 1
        try:
            vr = verify(prog, demos)
            if vr.passed:
                return TemplateSearchResult(
                    solved=True,
                    winning_program=prog,
                    winning_text=key,
                    template_name="derive_output",
                    candidates_tested=total_tested,
                )
        except Exception:
            pass

    return TemplateSearchResult(solved=False, candidates_tested=total_tested)
