"""Executor for the next-generation solver.

Given a UnifiedRule, executes it on a new input to produce an output.
Also provides train verification and the main solver entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.decomposition import detect_bg
from aria.ngs.ir import UnifiedRule
from aria.ngs.output_units import decompose_output_units, OutputUnit
from aria.ngs.backward_explain import explain_unit, explain_all_units, Explanation
from aria.ngs.unify import unify_across_demos, _execute_abstract_rule
from aria.types import Grid, DemoPair


# ---------------------------------------------------------------------------
# Solver result
# ---------------------------------------------------------------------------

@dataclass
class NGSResult:
    """Result of the NGS solver on one task."""
    task_id: str
    train_verified: bool
    test_outputs: list[Grid]
    unified_rule: UnifiedRule | None
    per_demo_explanations: list[list[Explanation]]
    train_diff: int
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def ngs_solve(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> NGSResult:
    """Run the next-generation solver on one task.

    1. Decompose each demo's output into units
    2. For each unit, search for primitive-graph explanations
    3. Unify across demos into one abstract rule
    4. Verify on all demos
    5. Return the result
    """
    if not demos:
        return _empty_result(task_id)

    bg = detect_bg(demos[0].input)

    # Phase 1: Decompose and explain each demo
    per_demo_units: list[list[OutputUnit]] = []
    per_demo_explanations: list[list[Explanation]] = []

    for di, demo in enumerate(demos):
        demo_bg = detect_bg(demo.input)
        units = decompose_output_units(demo.input, demo.output, demo_bg)
        per_demo_units.append(units)

        # For Stage 1: if multiple units, try whole-grid first
        # (we only unify whole-grid explanations for now)
        if len(units) == 1:
            exps = explain_unit(demo.input, demo.output, units[0], demo_bg)
        else:
            # Try whole-grid explanation too
            from aria.ngs.output_units import _whole_unit
            whole = _whole_unit(demo.input, demo.output, demo_bg)
            exps = explain_unit(demo.input, demo.output, whole, demo_bg)
            # Also try per-unit explanations and aggregate
            per_unit_exps = explain_all_units(demo.input, demo.output, units, demo_bg)
            for ue in per_unit_exps:
                exps.extend(ue)

        per_demo_explanations.append(exps)

    # Phase 2: Unify across demos
    unified = unify_across_demos(per_demo_explanations, demos)

    if unified is not None and unified.train_verified:
        return NGSResult(
            task_id=task_id,
            train_verified=True,
            test_outputs=[],
            unified_rule=unified,
            per_demo_explanations=per_demo_explanations,
            train_diff=0,
            details={
                "strategy": "unified_abstract_rule",
                "description": unified.rule.description,
                "graph_depth": unified.rule.graph.depth(),
                "graph_size": unified.rule.graph.size(),
            },
        )

    # Phase 3: If no unified rule, report best partial result
    from aria.ngs.ir import AbstractRule
    best_diff = 0
    for di, demo in enumerate(demos):
        if per_demo_explanations[di]:
            best_exp = per_demo_explanations[di][0]
            demo_bg = detect_bg(demo.input)
            arule = AbstractRule(graph=best_exp.graph, description=best_exp.description)
            predicted = _execute_abstract_rule(arule, demo.input, demo_bg, di, demos)
            if predicted is not None and predicted.shape == demo.output.shape:
                best_diff += int(np.sum(predicted != demo.output))
            else:
                best_diff += demo.output.size
        else:
            best_diff += int(np.sum(demo.input != demo.output)) if demo.input.shape == demo.output.shape else demo.output.size

    return NGSResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        unified_rule=unified,
        per_demo_explanations=per_demo_explanations,
        train_diff=best_diff,
        details={
            "strategy": "no_unified_rule",
            "n_demo_explanations": [len(e) for e in per_demo_explanations],
        },
    )


def ngs_predict(
    unified_rule: UnifiedRule,
    test_input: Grid,
) -> Grid | None:
    """Execute a unified rule on a test input."""
    bg = detect_bg(test_input)
    return _execute_abstract_rule(
        unified_rule.rule, test_input, bg, 0, (),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(task_id: str) -> NGSResult:
    return NGSResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        unified_rule=None,
        per_demo_explanations=[],
        train_diff=0,
        details={"strategy": "empty"},
    )
