"""Consensus trace — diagnostics for stepwise all-demo consensus.

Shows why each branch lived or died during consensus-gated search.
Wired into scene_solve and sketch_compile for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aria.consensus import BranchState, ConsistencyCheck, SharedHypothesis


@dataclass(frozen=True)
class ConsensusTraceEntry:
    """One step in the consensus trace for a branch."""

    branch_id: str
    step_index: int
    step_description: str
    consistency_score: float
    checks: tuple[ConsistencyCheck, ...]
    pruned: bool
    prune_reason: str
    hypothesis: SharedHypothesis
    per_demo_signatures: tuple[str, ...]


@dataclass
class ConsensusTrace:
    """Accumulates trace entries during a consensus-gated search."""

    entries: list[ConsensusTraceEntry] = field(default_factory=list)
    branches_created: int = 0
    branches_pruned: int = 0
    branches_survived: int = 0

    def record(self, branch: BranchState, step_description: str) -> None:
        """Record a branch state as a trace entry."""
        self.entries.append(ConsensusTraceEntry(
            branch_id=branch.branch_id,
            step_index=branch.step_index,
            step_description=step_description,
            consistency_score=branch.consistency_score,
            checks=branch.consistency_checks,
            pruned=branch.pruned,
            prune_reason=branch.prune_reason,
            hypothesis=branch.hypothesis,
            per_demo_signatures=tuple(
                pd.structural_signature for pd in branch.per_demo
            ),
        ))
        self.branches_created += 1
        if branch.pruned:
            self.branches_pruned += 1
        else:
            self.branches_survived += 1

    def summary(self) -> dict[str, int]:
        """Return summary counts."""
        return {
            "branches_created": self.branches_created,
            "branches_pruned": self.branches_pruned,
            "branches_survived": self.branches_survived,
            "trace_entries": len(self.entries),
        }


def format_consensus_trace(trace: ConsensusTrace) -> str:
    """Human-readable trace showing why branches lived or died."""
    lines: list[str] = []
    lines.append(
        f"Consensus trace: {trace.branches_created} created, "
        f"{trace.branches_pruned} pruned, "
        f"{trace.branches_survived} survived"
    )
    lines.append("")

    for entry in trace.entries:
        status = "PRUNED" if entry.pruned else "alive"
        lines.append(
            f"  [{entry.branch_id}] step={entry.step_index} "
            f"score={entry.consistency_score:.2f} {status}"
        )
        lines.append(f"    desc: {entry.step_description}")
        if entry.hypothesis.rule_family:
            lines.append(f"    hypothesis: {_format_hypothesis(entry.hypothesis)}")
        for check in entry.checks:
            mark = "OK" if check.passed else "FAIL"
            lines.append(
                f"    {mark} {check.name} (score={check.score:.2f})"
                + (f" — {check.detail}" if check.detail else "")
            )
        if entry.pruned:
            lines.append(f"    prune reason: {entry.prune_reason}")
        lines.append("")

    return "\n".join(lines)


def _format_hypothesis(h: SharedHypothesis) -> str:
    parts = []
    if h.rule_family:
        parts.append(f"rule={h.rule_family}")
    if h.selector_family:
        parts.append(f"sel={h.selector_family}")
    if h.transform_family:
        parts.append(f"xform={h.transform_family}")
    if h.scope_family:
        parts.append(f"scope={h.scope_family}")
    return " ".join(parts)
