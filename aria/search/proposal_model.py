"""Small trainable family model for `aria/search`.

This is intentionally simple and local:
- train from the solved-search proposal corpus
- predict family log-scores from task signatures
- act as an optional learned-ish signal on top of proposal memory

Current model: Bernoulli Naive Bayes over task-signature presence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

from aria.search.proposal_corpus import SearchProposalExample


def _results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def default_model_path() -> Path:
    return _results_dir() / "search_family_model.json"


@dataclass(frozen=True)
class SearchFamilyModel:
    family_counts: dict[str, int]
    signature_counts: dict[str, dict[str, int]]
    vocabulary: tuple[str, ...]
    total_examples: int

    @classmethod
    def empty(cls) -> "SearchFamilyModel":
        return cls(
            family_counts={},
            signature_counts={},
            vocabulary=(),
            total_examples=0,
        )

    @classmethod
    def train(cls, examples: list[SearchProposalExample]) -> "SearchFamilyModel":
        family_counts: Counter[str] = Counter()
        signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocab: set[str] = set()

        for ex in examples:
            family_counts[ex.family] += 1
            for sig in set(ex.task_signatures):
                vocab.add(sig)
                signature_counts[ex.family][sig] += 1

        return cls(
            family_counts=dict(family_counts),
            signature_counts={fam: dict(counts) for fam, counts in signature_counts.items()},
            vocabulary=tuple(sorted(vocab)),
            total_examples=len(examples),
        )

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "family_counts": dict(self.family_counts),
            "signature_counts": {
                fam: dict(counts)
                for fam, counts in self.signature_counts.items()
            },
            "vocabulary": list(self.vocabulary),
            "total_examples": self.total_examples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchFamilyModel":
        if int(data.get("version", 0)) != 1:
            raise ValueError("Unsupported search family model format")
        return cls(
            family_counts={str(k): int(v) for k, v in data.get("family_counts", {}).items()},
            signature_counts={
                str(fam): {str(sig): int(v) for sig, v in counts.items()}
                for fam, counts in data.get("signature_counts", {}).items()
            },
            vocabulary=tuple(str(v) for v in data.get("vocabulary", [])),
            total_examples=int(data.get("total_examples", 0)),
        )

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "SearchFamilyModel":
        source = Path(path)
        if not source.exists():
            return cls.empty()
        with open(source) as f:
            return cls.from_dict(json.load(f))

    def score_family(self, family: str, task_signatures: frozenset[str]) -> float:
        if family not in self.family_counts or self.total_examples <= 0:
            return 0.0

        n_families = max(len(self.family_counts), 1)
        family_total = self.family_counts[family]
        logp = math.log((family_total + 1) / (self.total_examples + n_families))

        fam_sig_counts = self.signature_counts.get(family, {})
        denom = family_total + 2.0
        for sig in task_signatures:
            count = fam_sig_counts.get(sig, 0)
            logp += math.log((count + 1.0) / denom)

        # Penalize missing positive evidence when the family usually relies on
        # signatures not present in the task.
        for sig, count in fam_sig_counts.items():
            if count <= 0 or sig in task_signatures:
                continue
            logp += math.log((family_total - count + 1.0) / denom)
        return logp

    def score_all(self, task_signatures: frozenset[str]) -> dict[str, float]:
        return {
            family: self.score_family(family, task_signatures)
            for family in self.family_counts
        }


@lru_cache(maxsize=1)
def load_default_search_family_model() -> SearchFamilyModel:
    path = default_model_path()
    if path.exists():
        return SearchFamilyModel.load_json(path)
    return SearchFamilyModel.empty()
