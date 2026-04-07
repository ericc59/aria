"""Fragment mining from verified programs.

Extracts small reusable graph fragments from verified programs,
canonicalizes them, and scores by frequency.

Part of the canonical architecture.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MinedFragment:
    """A graph fragment extracted from a verified program."""
    ops: tuple[str, ...]       # op sequence
    source_tasks: tuple[str, ...]  # task IDs where this fragment appeared
    count: int                 # how many times it appeared
    verified: bool = True      # came from a verified program


@dataclass
class FragmentCatalog:
    """Collection of mined fragments, scored by frequency."""
    singles: list[MinedFragment] = field(default_factory=list)
    pairs: list[MinedFragment] = field(default_factory=list)
    sequences: list[MinedFragment] = field(default_factory=list)


def mine_fragments(
    verified_programs: list[tuple[str, Any]],  # (task_id, Program)
) -> FragmentCatalog:
    """Extract and deduplicate fragments from verified programs."""
    single_counts: dict[str, list[str]] = {}  # op -> [task_ids]
    pair_counts: dict[tuple[str, str], list[str]] = {}
    seq_counts: dict[tuple[str, ...], list[str]] = {}

    for task_id, program in verified_programs:
        ops = _extract_ops(program)
        if not ops:
            continue

        # Single ops
        for op in ops:
            single_counts.setdefault(op, []).append(task_id)

        # Adjacent pairs
        for i in range(len(ops) - 1):
            pair = (ops[i], ops[i + 1])
            pair_counts.setdefault(pair, []).append(task_id)

        # Full sequence (canonical)
        seq = tuple(ops)
        seq_counts.setdefault(seq, []).append(task_id)

    catalog = FragmentCatalog()

    for op, tids in sorted(single_counts.items(), key=lambda x: -len(x[1])):
        catalog.singles.append(MinedFragment(
            ops=(op,), source_tasks=tuple(tids), count=len(tids),
        ))

    for pair, tids in sorted(pair_counts.items(), key=lambda x: -len(x[1])):
        catalog.pairs.append(MinedFragment(
            ops=pair, source_tasks=tuple(tids), count=len(tids),
        ))

    for seq, tids in sorted(seq_counts.items(), key=lambda x: -len(x[1])):
        catalog.sequences.append(MinedFragment(
            ops=seq, source_tasks=tuple(tids), count=len(tids),
        ))

    return catalog


def _extract_ops(program: Any) -> list[str]:
    """Extract operation names from a Program."""
    ops = []
    try:
        for step in program.steps:
            if hasattr(step.expr, 'op'):
                ops.append(step.expr.op)
    except Exception:
        pass
    return ops


def format_catalog(catalog: FragmentCatalog) -> str:
    """Format the fragment catalog for inspection."""
    lines = ["=== Mined Fragment Catalog ==="]

    recurring_singles = [f for f in catalog.singles if f.count >= 2]
    if recurring_singles:
        lines.append(f"\nRecurring single ops ({len(recurring_singles)}):")
        for f in recurring_singles:
            lines.append(f"  {f.ops[0]}: {f.count}x from {f.source_tasks}")

    recurring_pairs = [f for f in catalog.pairs if f.count >= 2]
    if recurring_pairs:
        lines.append(f"\nRecurring op pairs ({len(recurring_pairs)}):")
        for f in recurring_pairs:
            lines.append(f"  {f.ops}: {f.count}x from {f.source_tasks}")

    lines.append(f"\nAll sequences ({len(catalog.sequences)}):")
    for f in catalog.sequences:
        lines.append(f"  {f.ops}: {f.count}x from {f.source_tasks}")

    return "\n".join(lines)
