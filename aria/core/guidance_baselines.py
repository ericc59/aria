"""Simple guidance baselines — validate data interfaces before serious models.

Lightweight baselines that consume exported records and labels:
1. Perception-signature retrieval (k-nearest by feature overlap)
2. Heuristic output-size ranker (frequency-based from solved traces)
3. Feature-rule classifier (decision stump over perception features)

No ML dependencies. No solver changes. No task-id logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Perception signature — hashable feature vector for retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionSignature:
    """Discrete perception features for retrieval similarity."""
    same_dims: bool
    has_partition: bool
    has_frame: bool
    has_legend: bool
    has_slot_grid: bool
    n_objects_bucket: str   # "0", "1-3", "4-9", "10+"
    n_colors: int

    @staticmethod
    def from_record(record: dict) -> PerceptionSignature:
        """Extract signature from an exported guidance record."""
        perceptions = record.get("perception_summaries", [])
        first = perceptions[0] if perceptions else {}

        n_obj = first.get("n_objects_4", 0)
        if n_obj == 0:
            bucket = "0"
        elif n_obj <= 3:
            bucket = "1-3"
        elif n_obj <= 9:
            bucket = "4-9"
        else:
            bucket = "10+"

        palette = first.get("palette", [])

        # Check dims consistency across demos
        demos = record.get("train_demos", [])
        same_dims = True
        for d in demos:
            inp = d.get("input", [[]])
            out = d.get("output", [[]])
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                same_dims = False
                break

        return PerceptionSignature(
            same_dims=same_dims,
            has_partition=first.get("partition") is not None,
            has_frame=first.get("n_framed_regions", 0) > 0,
            has_legend=first.get("has_legend", False),
            has_slot_grid=record.get("slot_grid") is not None,
            n_objects_bucket=bucket,
            n_colors=len(palette),
        )

    def overlap(self, other: PerceptionSignature) -> int:
        """Count matching features between two signatures."""
        matches = 0
        if self.same_dims == other.same_dims:
            matches += 1
        if self.has_partition == other.has_partition:
            matches += 1
        if self.has_frame == other.has_frame:
            matches += 1
        if self.has_legend == other.has_legend:
            matches += 1
        if self.has_slot_grid == other.has_slot_grid:
            matches += 1
        if self.n_objects_bucket == other.n_objects_bucket:
            matches += 1
        if self.n_colors == other.n_colors:
            matches += 1
        return matches


# ---------------------------------------------------------------------------
# Retrieval baseline
# ---------------------------------------------------------------------------


@dataclass
class RetrievalIndex:
    """Index of solved traces for k-nearest retrieval."""
    entries: list[tuple[PerceptionSignature, dict]] = field(default_factory=list)

    def add(self, record: dict) -> None:
        sig = PerceptionSignature.from_record(record)
        self.entries.append((sig, record))

    def query(self, record: dict, k: int = 5) -> list[dict]:
        """Return k-nearest records by perception signature overlap."""
        query_sig = PerceptionSignature.from_record(record)
        scored = [
            (sig.overlap(query_sig), rec)
            for sig, rec in self.entries
        ]
        scored.sort(key=lambda x: -x[0])
        return [rec for _, rec in scored[:k]]

    def query_families(self, record: dict, k: int = 5) -> list[str]:
        """Return program families from k-nearest solved traces."""
        neighbors = self.query(record, k)
        families = []
        for n in neighbors:
            fam = n.get("program_family", "")
            if fam:
                families.append(fam)
        return families

    def query_size_modes(self, record: dict, k: int = 5) -> list[str]:
        """Return output-size modes from k-nearest solved traces."""
        neighbors = self.query(record, k)
        modes = []
        for n in neighbors:
            ss = n.get("size_spec")
            if ss and "mode" in ss:
                modes.append(ss["mode"])
        return modes

    @staticmethod
    def from_records(records: list[dict], solved_only: bool = True) -> RetrievalIndex:
        """Build an index from exported records."""
        index = RetrievalIndex()
        for rec in records:
            if solved_only and rec.get("verify_result") != "solved":
                continue
            index.add(rec)
        return index


# ---------------------------------------------------------------------------
# Heuristic output-size ranker
# ---------------------------------------------------------------------------


@dataclass
class SizeModeRanker:
    """Rank output-size modes by frequency, optionally conditioned on features."""

    # Global mode counts from solved traces
    mode_counts: Counter = field(default_factory=Counter)
    # Conditional counts: feature_key -> mode -> count
    conditional_counts: dict[str, Counter] = field(default_factory=dict)

    def fit(self, records: list[dict]) -> None:
        """Fit from solved records."""
        for rec in records:
            if rec.get("verify_result") != "solved":
                continue
            ss = rec.get("size_spec")
            if not ss or "mode" not in ss:
                continue
            mode = ss["mode"]
            self.mode_counts[mode] += 1

            # Conditional on perception features
            sig = PerceptionSignature.from_record(rec)
            for feat_key in [
                f"same_dims={sig.same_dims}",
                f"has_partition={sig.has_partition}",
                f"has_frame={sig.has_frame}",
                f"n_obj={sig.n_objects_bucket}",
            ]:
                if feat_key not in self.conditional_counts:
                    self.conditional_counts[feat_key] = Counter()
                self.conditional_counts[feat_key][mode] += 1

    def rank(self, record: dict, top_k: int = 5) -> list[str]:
        """Rank modes for a new task, boosting by feature match."""
        sig = PerceptionSignature.from_record(record)
        scores: Counter = Counter()

        # Base frequency
        for mode, count in self.mode_counts.items():
            scores[mode] += count

        # Feature boost
        for feat_key in [
            f"same_dims={sig.same_dims}",
            f"has_partition={sig.has_partition}",
            f"has_frame={sig.has_frame}",
            f"n_obj={sig.n_objects_bucket}",
        ]:
            cond = self.conditional_counts.get(feat_key)
            if cond:
                for mode, count in cond.items():
                    scores[mode] += count * 2  # 2x weight for conditional match

        return [mode for mode, _ in scores.most_common(top_k)]


# ---------------------------------------------------------------------------
# Feature-rule classifier (decision stump)
# ---------------------------------------------------------------------------


@dataclass
class FeatureRule:
    """A single feature rule: if feature == value then predict label."""
    feature: str
    value: Any
    prediction: str
    accuracy: float
    support: int


def fit_decision_stumps(
    records: list[dict],
    target_key: str = "size_spec.mode",
) -> list[FeatureRule]:
    """Fit single-feature rules from exported records.

    Returns rules sorted by accuracy (best first).
    """
    solved = [r for r in records if r.get("verify_result") == "solved"]
    if not solved:
        return []

    # Extract targets
    targets = []
    for rec in solved:
        if target_key == "size_spec.mode":
            ss = rec.get("size_spec")
            targets.append(ss["mode"] if ss and "mode" in ss else None)
        elif target_key == "program_family":
            targets.append(rec.get("program_family", ""))
        else:
            targets.append(None)

    # Extract feature vectors
    features: list[dict] = []
    for rec in solved:
        sig = PerceptionSignature.from_record(rec)
        features.append({
            "same_dims": sig.same_dims,
            "has_partition": sig.has_partition,
            "has_frame": sig.has_frame,
            "has_legend": sig.has_legend,
            "has_slot_grid": sig.has_slot_grid,
            "n_objects_bucket": sig.n_objects_bucket,
            "n_colors": sig.n_colors,
        })

    # Find best prediction per (feature, value) pair
    rules: list[FeatureRule] = []
    for feat_name in features[0].keys():
        # Group targets by feature value
        groups: dict[Any, list] = {}
        for i, fv in enumerate(features):
            val = fv[feat_name]
            groups.setdefault(val, []).append(targets[i])

        for val, group_targets in groups.items():
            valid = [t for t in group_targets if t is not None]
            if not valid:
                continue
            counter = Counter(valid)
            best_pred, best_count = counter.most_common(1)[0]
            accuracy = best_count / len(valid)
            rules.append(FeatureRule(
                feature=feat_name,
                value=val,
                prediction=best_pred,
                accuracy=accuracy,
                support=len(valid),
            ))

    rules.sort(key=lambda r: (-r.accuracy, -r.support))
    return rules
