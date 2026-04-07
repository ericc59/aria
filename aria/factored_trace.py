"""Factored retrieval trace — diagnostics for inspecting retrieval decisions.

Shows which records matched, why, which factors were borrowed, and
whether retrieval improved ranking or helped repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aria.factored_memory import PerceptionKey
from aria.factored_retrieval import RetrievalGuidance, RetrievalMatch


# ---------------------------------------------------------------------------
# Trace entry — one retrieved record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactoredRetrievalTraceEntry:
    """One retrieved record with usage info."""

    record_id: str
    score: float
    match_reasons: tuple[str, ...]
    factors_used: tuple[str, ...]  # which factors from this record were adopted


# ---------------------------------------------------------------------------
# Retrieval trace — full retrieval session
# ---------------------------------------------------------------------------


@dataclass
class FactoredRetrievalTrace:
    """Accumulates retrieval diagnostics during a solve attempt."""

    query_perception_key: PerceptionKey | None = None
    query_task_signatures: tuple[str, ...] = ()
    matches_returned: int = 0
    top_match_score: float = 0.0
    guidance_used: RetrievalGuidance | None = None
    factors_borrowed: tuple[str, ...] = ()
    retrieval_improved_ranking: bool | None = None
    retrieval_helped_repair: bool | None = None
    # Per-mechanism tracking
    decomp_bias_active: bool = False
    decomp_bias_changed_order: bool = False
    op_bias_count: int = 0
    repair_bias_active: bool = False
    repair_mutations_suggested: int = 0
    entries: list[FactoredRetrievalTraceEntry] = field(default_factory=list)

    def record_matches(self, matches: list[RetrievalMatch]) -> None:
        """Record the retrieval matches."""
        self.matches_returned = len(matches)
        self.top_match_score = matches[0].score if matches else 0.0
        self.entries = [
            FactoredRetrievalTraceEntry(
                record_id=m.record.record_id,
                score=m.score,
                match_reasons=m.match_reasons,
                factors_used=(),  # filled in later when factors are adopted
            )
            for m in matches
        ]

    def record_guidance(self, guidance: RetrievalGuidance) -> None:
        """Record the aggregated guidance."""
        self.guidance_used = guidance
        borrowed: list[str] = []
        if guidance.preferred_decomposition_types:
            borrowed.append(f"decomp:{guidance.preferred_decomposition_types[0]}")
            self.decomp_bias_active = True
        if guidance.preferred_op_families:
            borrowed.append(f"op:{guidance.preferred_op_families[0]}")
            self.op_bias_count = len(guidance.preferred_op_families)
        if guidance.preferred_selector_families:
            borrowed.append(f"selector:{guidance.preferred_selector_families[0]}")
        if guidance.candidate_repair_paths:
            self.repair_bias_active = True
            self.repair_mutations_suggested = sum(
                len(rp.mutations_applied) for rp in guidance.candidate_repair_paths
            )
        self.factors_borrowed = tuple(borrowed)

    def summary(self) -> dict[str, Any]:
        """Return summary for reporting."""
        return {
            "matches_returned": self.matches_returned,
            "top_match_score": round(self.top_match_score, 2),
            "factors_borrowed": list(self.factors_borrowed),
            "retrieval_improved_ranking": self.retrieval_improved_ranking,
            "retrieval_helped_repair": self.retrieval_helped_repair,
            "decomp_bias_active": self.decomp_bias_active,
            "decomp_bias_changed_order": self.decomp_bias_changed_order,
            "op_bias_count": self.op_bias_count,
            "repair_bias_active": self.repair_bias_active,
            "repair_mutations_suggested": self.repair_mutations_suggested,
        }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_factored_retrieval_trace(trace: FactoredRetrievalTrace) -> str:
    """Human-readable trace of factored retrieval decisions."""
    lines: list[str] = []
    lines.append(
        f"Factored retrieval: {trace.matches_returned} matches, "
        f"top score {trace.top_match_score:.1f}"
    )

    if trace.factors_borrowed:
        lines.append(f"  Factors borrowed: {', '.join(trace.factors_borrowed)}")

    if trace.guidance_used:
        g = trace.guidance_used
        if g.preferred_decomposition_types:
            lines.append(f"  Preferred decomp: {', '.join(g.preferred_decomposition_types[:3])}")
        if g.preferred_op_families:
            lines.append(f"  Preferred ops: {', '.join(g.preferred_op_families[:5])}")
        if g.preferred_selector_families:
            lines.append(f"  Preferred selectors: {', '.join(g.preferred_selector_families[:3])}")
        if g.candidate_repair_paths:
            lines.append(f"  Repair paths available: {len(g.candidate_repair_paths)}")

    if trace.entries:
        lines.append("")
        lines.append("  Top matches:")
        for entry in trace.entries[:5]:
            reasons = ", ".join(entry.match_reasons[:3])
            lines.append(f"    {entry.record_id[:8]}  score={entry.score:.1f}  [{reasons}]")

    if trace.retrieval_improved_ranking is not None:
        lines.append(f"  Improved ranking: {trace.retrieval_improved_ranking}")
    if trace.retrieval_helped_repair is not None:
        lines.append(f"  Helped repair: {trace.retrieval_helped_repair}")

    return "\n".join(lines)
