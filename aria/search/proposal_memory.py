"""Persistent proposal priors for the canonical `aria/search` stack.

This is the first small amortized-inference layer for search.

We mine past *search* solves from evaluation reports, keyed by existing
task signatures, and use that memory to rank:

- derived/search candidate programs
- seed schemas

The prior is intentionally weak and conservative:
- exact verification still decides correctness
- we only reorder existing candidates
- no task-specific logic or learned black boxes
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from aria.search.seeds import SeedSchema
from aria.search.sketch import SearchProgram


def _results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _iter_eval_reports(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("eval_*.json"))


def _family_from_description(description: str) -> str | None:
    if not description.startswith("search:"):
        return None
    if "[" in description and "]" in description:
        start = description.rfind("[")
        end = description.rfind("]")
        if 0 <= start < end:
            family = description[start + 1:end].strip()
            if family:
                return family
    body = description.split(":", 1)[1].strip()
    if body:
        return body
    return None


@dataclass(frozen=True)
class SearchProposalPrior:
    """A lightweight, persistent prior over search families."""

    global_counts: dict[str, int]
    by_signature: dict[str, dict[str, int]]

    @classmethod
    def empty(cls) -> "SearchProposalPrior":
        return cls(global_counts={}, by_signature={})

    @classmethod
    def from_eval_reports(cls, paths: list[Path]) -> "SearchProposalPrior":
        global_counts: Counter[str] = Counter()
        by_signature: dict[str, Counter[str]] = defaultdict(Counter)

        for path in paths:
            try:
                with open(path) as f:
                    payload = json.load(f)
            except Exception:
                continue

            for task in payload.get("tasks", []):
                if not task.get("solved"):
                    continue
                if task.get("solve_source") != "search":
                    continue
                family = _family_from_description(str(task.get("description", "")))
                if not family:
                    continue
                global_counts[family] += 1
                for sig in task.get("task_signatures", []):
                    if isinstance(sig, str) and sig:
                        by_signature[sig][family] += 1

        return cls(
            global_counts=dict(global_counts),
            by_signature={sig: dict(counts) for sig, counts in by_signature.items()},
        )

    def score_family(self, family: str, task_signatures: frozenset[str]) -> float:
        if not family:
            return 0.0
        score = float(self.global_counts.get(family, 0))
        for sig in task_signatures:
            score += 3.0 * self.by_signature.get(sig, {}).get(family, 0)
        return score

    def rank_programs(
        self,
        programs: list[SearchProgram],
        task_signatures: frozenset[str],
    ) -> list[SearchProgram]:
        def key(prog: SearchProgram) -> tuple[float, int, str]:
            family = prog.signature
            return (
                -self.score_family(family, task_signatures),
                len(prog.steps),
                prog.provenance,
            )

        return sorted(programs, key=key)

    def rank_schemas(
        self,
        schemas: list[SeedSchema],
        task_signatures: frozenset[str],
    ) -> list[SeedSchema]:
        def key(schema: SeedSchema) -> tuple[float, str, str]:
            family = schema.action or schema.name
            return (
                -self.score_family(family, task_signatures),
                schema.action,
                schema.name,
            )

        return sorted(schemas, key=key)


@lru_cache(maxsize=1)
def load_default_search_prior() -> SearchProposalPrior:
    """Load persistent search priors from local eval reports.

    This keeps the current search stack self-contained:
    if no reports exist, the prior is simply empty.
    """

    root = _results_dir()
    paths = _iter_eval_reports(root)
    if not paths:
        return SearchProposalPrior.empty()
    return SearchProposalPrior.from_eval_reports(paths)
