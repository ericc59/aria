"""Budget allocation policy — spend compile effort where expected value is highest.

Uses mechanism evidence and exec_hints to allocate per-lane compile
budgets. Lanes with strong anti-evidence get zero budget. Lanes with
high exec_hint get more of the total budget.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class LaneBudget:
    """Compile budget allocated to one lane."""
    lane: str
    max_combos: int       # max parameter combinations to try
    rationale: str


@dataclass(frozen=True)
class BudgetPolicy:
    """Full budget allocation for one task."""
    lane_budgets: dict[str, LaneBudget]
    editor_compiles: int
    editor_expansions: int
    total_compiles: int   # sum of all lane budgets + editor


# Default lane max combos (before evidence adjustment)
_DEFAULT_MAX = {
    "periodic_repair": 24,
    "replication": 8,
    "relocation": 49,
}


def allocate_budget(
    demos: Sequence[Any],
    *,
    total_compile_cap: int = 60,
    editor_compile_cap: int = 20,
    editor_expansion_cap: int = 100,
) -> BudgetPolicy:
    """Allocate compile budget across lanes using evidence signals.

    Policy:
    1. Lanes with anti-evidence get 0 budget
    2. Lanes with high exec_hint get full default budget
    3. Lanes with low exec_hint get reduced budget
    4. Editor gets remaining budget after lane allocation
    """
    from aria.core.mechanism_evidence import compute_evidence_and_rank

    evidence, ranking = compute_evidence_and_rank(demos)

    budgets: dict[str, LaneBudget] = {}
    total_lane = 0

    for candidate in ranking.lanes:
        name = candidate.name
        if name not in _DEFAULT_MAX:
            continue

        default = _DEFAULT_MAX[name]

        if candidate.anti_evidence:
            # Anti-evidence: skip this lane entirely
            budgets[name] = LaneBudget(
                lane=name, max_combos=0,
                rationale=f"skipped: {candidate.anti_evidence}",
            )
        elif candidate.exec_hint >= 0.5:
            # High hint: full budget
            cap = min(default, total_compile_cap - total_lane)
            budgets[name] = LaneBudget(
                lane=name, max_combos=cap,
                rationale=f"full: exec_hint={candidate.exec_hint:.2f}",
            )
            total_lane += cap
        elif candidate.exec_hint > 0:
            # Low hint: reduced budget (half default)
            cap = min(default // 2, total_compile_cap - total_lane)
            budgets[name] = LaneBudget(
                lane=name, max_combos=max(cap, 0),
                rationale=f"reduced: exec_hint={candidate.exec_hint:.2f}",
            )
            total_lane += max(cap, 0)
        else:
            # No hint: minimal budget
            cap = min(4, total_compile_cap - total_lane)
            budgets[name] = LaneBudget(
                lane=name, max_combos=max(cap, 0),
                rationale=f"minimal: exec_hint=0",
            )
            total_lane += max(cap, 0)

    # Editor gets remaining budget
    editor_compiles = min(editor_compile_cap, total_compile_cap - total_lane)

    return BudgetPolicy(
        lane_budgets=budgets,
        editor_compiles=max(editor_compiles, 0),
        editor_expansions=editor_expansion_cap,
        total_compiles=total_lane + max(editor_compiles, 0),
    )


def format_budget(policy: BudgetPolicy) -> str:
    lines = [f"=== Budget Policy (total={policy.total_compiles}) ==="]
    for name, lb in sorted(policy.lane_budgets.items()):
        lines.append(f"  {name:20s} max={lb.max_combos:3d}  {lb.rationale}")
    lines.append(f"  {'editor':20s} compiles={policy.editor_compiles} expansions={policy.editor_expansions}")
    return "\n".join(lines)
