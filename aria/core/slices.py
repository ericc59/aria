"""Benchmark slice curriculum — repeatable, evidence-defined task slices.

Each slice has an explicit structural entry rule, not hand-picked task IDs.
Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class SliceDefinition:
    """One benchmark slice with entry rule and rationale."""
    name: str
    rationale: str
    entry_rule: Callable  # (task_id, demos, evidence, ranking) -> bool


@dataclass
class SliceResult:
    """Tasks selected for a slice."""
    name: str
    task_ids: list[str] = field(default_factory=list)
    size: int = 0


def build_slices(
    task_ids: list[str],
    demos_fn,
) -> dict[str, SliceResult]:
    """Build all slices from a task pool."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    results = {s.name: SliceResult(name=s.name) for s in SLICES}

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        ev, ranking = compute_evidence_and_rank(demos)
        solve_result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=tid)
        solved = solve_result.solved

        for sdef in SLICES:
            try:
                if sdef.entry_rule(tid, demos, ev, ranking, solved):
                    results[sdef.name].task_ids.append(tid)
            except Exception:
                pass

    for sr in results.values():
        sr.size = len(sr.task_ids)

    return results


def format_slices(slices: dict[str, SliceResult]) -> str:
    lines = ["=== Benchmark Slices ==="]
    for name, sr in sorted(slices.items()):
        sdef = next((s for s in SLICES if s.name == name), None)
        rationale = sdef.rationale if sdef else ""
        lines.append(f"\n{name} ({sr.size} tasks): {rationale}")
        if sr.size <= 10:
            for tid in sr.task_ids:
                lines.append(f"  {tid}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slice definitions
# ---------------------------------------------------------------------------


def _solved_sanity(tid, demos, ev, ranking, solved):
    return solved


def _replication_highfit(tid, demos, ev, ranking, solved):
    repl = next((c for c in ranking.lanes if c.name == "replication"), None)
    return repl is not None and repl.gate_pass and not solved


def _periodic_highfit(tid, demos, ev, ranking, solved):
    per = next((c for c in ranking.lanes if c.name == "periodic_repair"), None)
    return per is not None and per.gate_pass and per.class_score >= 0.6 and not solved


def _relocation_highfit(tid, demos, ev, ranking, solved):
    reloc = next((c for c in ranking.lanes if c.name == "relocation"), None)
    return (reloc is not None and reloc.gate_pass
            and ev.shapes_shift_position and ev.same_dims and not solved)


def _composition_candidate(tid, demos, ev, ranking, solved):
    if solved:
        return False
    gated = [c for c in ranking.lanes if c.gate_pass]
    return len(gated) >= 2 or (ev.has_framed_region and ev.n_input_singles > 0)


def _dims_change(tid, demos, ev, ranking, solved):
    return not ev.same_dims and not solved


SLICES = [
    SliceDefinition("solved_sanity", "Tasks that currently solve (regression guard)", _solved_sanity),
    SliceDefinition("replication_candidates", "High replication fit, unsolved", _replication_highfit),
    SliceDefinition("periodic_candidates", "High periodic fit, unsolved", _periodic_highfit),
    SliceDefinition("relocation_candidates", "Relocation fit with shifting shapes, unsolved", _relocation_highfit),
    SliceDefinition("composition_candidates", "Multi-lane structure, unsolved", _composition_candidate),
    SliceDefinition("dims_change", "Output dims differ from input, unsolved", _dims_change),
]
