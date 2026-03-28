"""Canonical training-data export for future local Qwen-class models.

Reads verified programs from a ProgramStore and refinement trajectories
from a RefinementTraceStore, then emits deterministic JSONL examples
grouped by task type.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

TASK_TYPES = frozenset({
    "NEXT_FOCUS", "NEXT_EDIT", "SKETCH", "CANDIDATE_RANK",
    "DECOMP_RANK", "SKETCH_RANK",
})

# Diff-derived fields we propagate from feedback into training examples.
_DIFF_PROGRESS_KEYS = (
    "best_candidate_score",
    "best_candidate_dims_match",
    "best_candidate_pixel_diff_count",
    "best_candidate_wrong_row_count",
    "best_candidate_wrong_col_count",
    "best_candidate_palette_expected_coverage",
    "best_candidate_palette_precision",
    "best_candidate_preserved_input_ratio",
    "best_candidate_changed_cells_ratio",
)


@dataclass(frozen=True)
class TrainingExample:
    schema_version: int
    task_type: str
    task_id: str | None
    task_signatures: tuple[str, ...]
    current_program: str | None
    round_index: int | None
    feedback: dict[str, Any] | None
    target: dict[str, Any]
    winning_program: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "task_id": self.task_id,
            "task_signatures": list(self.task_signatures),
            "current_program": self.current_program,
            "round_index": self.round_index,
            "feedback": self.feedback,
            "target": self.target,
            "winning_program": self.winning_program,
        }


def examples_from_program_store(records: list[dict]) -> list[TrainingExample]:
    """Emit SKETCH examples from verified programs in the program store.

    Each stored program that solved at least one task becomes a SKETCH example:
    given the task signatures, produce this program.
    """
    examples: list[TrainingExample] = []
    for record in records:
        program_text = record.get("program")
        if not isinstance(program_text, str) or not program_text.strip():
            continue
        task_ids = record.get("task_ids", [])
        signatures = tuple(sorted(record.get("signatures", [])))
        if not signatures:
            continue
        primary_task_id = task_ids[0] if task_ids else None
        examples.append(TrainingExample(
            schema_version=SCHEMA_VERSION,
            task_type="SKETCH",
            task_id=primary_task_id,
            task_signatures=signatures,
            current_program=None,
            round_index=None,
            feedback=None,
            target={"program": program_text},
            winning_program=program_text,
        ))
    return examples


def examples_from_refinement_traces(records: list[dict]) -> list[TrainingExample]:
    """Emit NEXT_FOCUS, CANDIDATE_RANK, and NEXT_EDIT examples."""
    examples: list[TrainingExample] = []
    for record in records:
        task_id = record.get("task_id")
        rounds = record.get("rounds", [])
        overall_winning = record.get("winning_program")

        for round_index, rnd in enumerate(rounds):
            feedback = rnd.get("feedback", {})
            task_sigs = tuple(sorted(feedback.get("task_signatures", [])))
            plan = rnd.get("plan", {})
            round_winning = rnd.get("winning_program")
            trace = rnd.get("trace", [])

            # --- NEXT_FOCUS ---
            if round_index > 0:
                prior_feedback = rounds[round_index - 1].get("feedback", {})
                examples.append(TrainingExample(
                    schema_version=SCHEMA_VERSION,
                    task_type="NEXT_FOCUS",
                    task_id=task_id,
                    task_signatures=task_sigs,
                    current_program=None,
                    round_index=round_index,
                    feedback=prior_feedback,
                    target={
                        "focus": plan.get("name", "generic"),
                        "allowed_ops": plan.get("allowed_ops"),
                    },
                    winning_program=overall_winning,
                ))

            # --- CANDIDATE_RANK ---
            _emit_candidate_rank(
                examples,
                task_id=task_id,
                task_sigs=task_sigs,
                round_index=round_index,
                feedback=feedback,
                trace=trace,
                round_winning=round_winning,
                overall_winning=overall_winning,
            )

            # --- NEXT_EDIT ---
            _emit_next_edit(
                examples,
                task_id=task_id,
                task_sigs=task_sigs,
                round_index=round_index,
                rounds=rounds,
                rnd=rnd,
                round_winning=round_winning,
                overall_winning=overall_winning,
            )

    return examples


# ---------------------------------------------------------------------------
# NEXT_EDIT: use best-candidate scoring for before/after pairs
# ---------------------------------------------------------------------------

def _emit_next_edit(
    examples: list[TrainingExample],
    *,
    task_id: str | None,
    task_sigs: tuple[str, ...],
    round_index: int,
    rounds: list[dict],
    rnd: dict,
    round_winning: str | None,
    overall_winning: str | None,
) -> None:
    """Emit NEXT_EDIT examples with quality tiering based on available data."""
    if round_index == 0:
        return

    prior_rnd = rounds[round_index - 1]
    prior_feedback = prior_rnd.get("feedback", {})
    current_feedback = rnd.get("feedback", {})

    # Determine the "before" program: prior round's best failing candidate.
    before_program = _best_candidate_program_from_round(prior_rnd)
    before_score = prior_feedback.get("best_candidate_score")

    # Determine the "after" program and its quality tier.
    after_program: str | None = None
    after_score: float | None = None
    edit_quality: str | None = None

    if round_winning is not None and before_program is not None:
        # Strong: best failing candidate in prior round -> exact winner this round.
        after_program = round_winning
        after_score = 1_000_000.0  # exact pass sentinel
        edit_quality = "strong"
    elif round_winning is not None and before_program is None:
        # We have a winner but no best-candidate anchor from prior round.
        # Fall back to the old coarse signal.
        after_program = round_winning
        after_score = 1_000_000.0
        edit_quality = "weak"
    elif before_program is not None:
        # No winner this round, but check if this round's best candidate
        # improved over the prior round's best candidate.
        current_best = _best_candidate_program_from_round(rnd)
        current_best_score = current_feedback.get("best_candidate_score")
        if (
            current_best is not None
            and current_best_score is not None
            and before_score is not None
            and current_best_score > before_score
        ):
            after_program = current_best
            after_score = current_best_score
            edit_quality = "medium"

    if after_program is None:
        return

    score_delta = (
        after_score - before_score
        if after_score is not None and before_score is not None
        else None
    )

    target: dict[str, Any] = {
        "before_program": before_program,
        "after_program": after_program,
        "edit_quality": edit_quality,
    }
    if before_score is not None:
        target["before_score"] = before_score
    if after_score is not None:
        target["after_score"] = after_score
    if score_delta is not None:
        target["score_delta"] = score_delta

    # Attach diff-progress summaries for before/after rounds.
    before_progress = _extract_diff_progress(prior_feedback)
    after_progress = _extract_diff_progress(current_feedback)
    if before_progress:
        target["before_feedback"] = before_progress
    if after_progress:
        target["after_feedback"] = after_progress

    examples.append(TrainingExample(
        schema_version=SCHEMA_VERSION,
        task_type="NEXT_EDIT",
        task_id=task_id,
        task_signatures=task_sigs,
        current_program=before_program,
        round_index=round_index,
        feedback=current_feedback,
        target=target,
        winning_program=overall_winning,
    ))


def _best_candidate_program_from_round(rnd: dict) -> str | None:
    """Find the best failing candidate's program text in a round."""
    feedback = rnd.get("feedback", {})
    best_num = feedback.get("best_candidate_num")
    if best_num is None:
        return None
    for entry in rnd.get("trace", []):
        if entry.get("candidate_num") == best_num and not entry.get("passed"):
            return entry.get("program_text")
    return None


def _extract_diff_progress(feedback: dict) -> dict[str, Any]:
    """Pull diff-derived progress fields from feedback, dropping Nones."""
    result: dict[str, Any] = {}
    for key in _DIFF_PROGRESS_KEYS:
        value = feedback.get(key)
        if value is not None:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# CANDIDATE_RANK: score-aware hard-negative selection
# ---------------------------------------------------------------------------

def _emit_candidate_rank(
    examples: list[TrainingExample],
    *,
    task_id: str | None,
    task_sigs: tuple[str, ...],
    round_index: int,
    feedback: dict[str, Any],
    trace: list[dict],
    round_winning: str | None,
    overall_winning: str | None,
) -> None:
    """Emit CANDIDATE_RANK examples using score-aware negative selection.

    Prefers hard negatives (high score but still failing) and diverse
    error types over trivially bad candidates.
    """
    if not trace or round_winning is None:
        return

    failing = [
        entry for entry in trace
        if not entry.get("passed") and entry.get("program_text")
    ]
    if not failing:
        return

    sampled = _select_hard_negatives(failing, k=3)

    for fail_entry in sampled:
        fail_score = fail_entry.get("score")
        preferred_score = 1_000_000.0  # exact pass sentinel

        rank_target: dict[str, Any] = {
            "preferred": round_winning,
            "rejected": fail_entry["program_text"],
            "rejected_error_type": fail_entry.get("error_type"),
        }

        if fail_score is not None:
            rank_target["rejected_score"] = fail_score
            rank_target["preferred_score"] = preferred_score
            rank_target["score_delta"] = preferred_score - fail_score

        fail_reasons = fail_entry.get("score_reasons")
        if fail_reasons:
            rank_target["rejected_score_reasons"] = (
                list(fail_reasons) if isinstance(fail_reasons, (list, tuple))
                else fail_reasons
            )

        examples.append(TrainingExample(
            schema_version=SCHEMA_VERSION,
            task_type="CANDIDATE_RANK",
            task_id=task_id,
            task_signatures=task_sigs,
            current_program=fail_entry["program_text"],
            round_index=round_index,
            feedback=feedback,
            target=rank_target,
            winning_program=overall_winning,
        ))


def _select_hard_negatives(failing: list[dict], k: int) -> list[dict]:
    """Select up to k failing candidates, preferring hard negatives.

    Strategy:
    1. Sort by score descending (highest-scored failures first = hardest negatives).
    2. Pick the top candidate.
    3. For remaining slots, prefer candidates with different error types.
    4. Break ties deterministically by (score desc, candidate_num asc).
    """
    if len(failing) <= k:
        return sorted(
            failing,
            key=lambda e: (-_safe_score(e), e.get("candidate_num", 0)),
        )

    # Sort all by score descending, then candidate_num for determinism.
    ranked = sorted(
        failing,
        key=lambda e: (-_safe_score(e), e.get("candidate_num", 0)),
    )

    selected: list[dict] = [ranked[0]]
    seen_error_types: set[str | None] = {ranked[0].get("error_type")}
    remaining = ranked[1:]

    while len(selected) < k and remaining:
        # Prefer a candidate with an unseen error type.
        diverse_pick = None
        diverse_idx = None
        for idx, entry in enumerate(remaining):
            if entry.get("error_type") not in seen_error_types:
                diverse_pick = entry
                diverse_idx = idx
                break

        if diverse_pick is not None and diverse_idx is not None:
            selected.append(diverse_pick)
            seen_error_types.add(diverse_pick.get("error_type"))
            remaining.pop(diverse_idx)
        else:
            # No new error type available; take next highest scored.
            selected.append(remaining.pop(0))

    return selected


def _safe_score(entry: dict) -> float:
    """Return score or a very low sentinel for deterministic sorting."""
    score = entry.get("score")
    return float(score) if score is not None else -1e9


# ---------------------------------------------------------------------------
# Export / IO
# ---------------------------------------------------------------------------

def export_examples(
    examples: list[TrainingExample],
    output_dir: str | Path,
) -> dict[str, int]:
    """Write examples to per-task-type JSONL files. Returns counts per type."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_type.setdefault(ex.task_type, []).append(ex)

    counts: dict[str, int] = {}
    for task_type in sorted(by_type):
        items = by_type[task_type]
        # Sort deterministically
        items.sort(key=lambda e: json.dumps(e.to_dict(), sort_keys=True))
        path = out / f"{task_type.lower()}.jsonl"
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
        counts[task_type] = len(items)

    return counts


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read back a JSONL file as a list of dicts."""
    result: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"schema_version", "task_type", "task_id", "task_signatures",
                    "current_program", "round_index", "feedback", "target",
                    "winning_program"}

_TARGET_REQUIRED: dict[str, set[str]] = {
    "SKETCH": {"program"},
    "NEXT_FOCUS": {"focus"},
    "NEXT_EDIT": {"after_program", "edit_quality"},
    "CANDIDATE_RANK": {"preferred", "rejected"},
    "DECOMP_RANK": {"useful_views"},
    "SKETCH_RANK": {"winning_family"},
}


@dataclass
class ValidationResult:
    valid: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)


def validate_example(ex: TrainingExample) -> list[str]:
    """Return a list of validation errors for a single example (empty = valid)."""
    errors: list[str] = []
    d = ex.to_dict()

    missing = _REQUIRED_FIELDS - set(d.keys())
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")

    if ex.task_type not in TASK_TYPES:
        errors.append(f"unknown task_type: {ex.task_type}")
        return errors

    if not ex.task_signatures:
        errors.append("empty task_signatures")

    target = ex.target
    required_target = _TARGET_REQUIRED.get(ex.task_type, set())
    missing_target = required_target - set(target.keys())
    if missing_target:
        errors.append(f"target missing: {sorted(missing_target)}")

    if ex.task_type == "NEXT_EDIT":
        quality = target.get("edit_quality")
        if quality not in ("strong", "medium", "weak"):
            errors.append(f"invalid edit_quality: {quality}")
        if quality in ("strong", "medium") and not target.get("before_program"):
            errors.append("strong/medium edit missing before_program")

    if ex.task_type == "SKETCH":
        prog = target.get("program", "")
        if not isinstance(prog, str) or not prog.strip():
            errors.append("SKETCH target has empty program")

    return errors


def validate_examples(examples: list[TrainingExample]) -> ValidationResult:
    """Validate a batch of examples, returning aggregate results."""
    result = ValidationResult()
    for ex in examples:
        errs = validate_example(ex)
        if errs:
            result.invalid += 1
            result.errors.extend(
                f"[{ex.task_type}:{ex.task_id}] {e}" for e in errs
            )
        else:
            result.valid += 1
    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _example_content_hash(ex: TrainingExample) -> str:
    """Deterministic hash of the semantically meaningful fields."""
    key = json.dumps({
        "task_type": ex.task_type,
        "task_signatures": list(ex.task_signatures),
        "target": ex.target,
        "current_program": ex.current_program,
        "round_index": ex.round_index,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def deduplicate(examples: list[TrainingExample]) -> tuple[list[TrainingExample], int]:
    """Remove duplicate examples by content hash. Returns (deduped, num_removed)."""
    seen: set[str] = set()
    unique: list[TrainingExample] = []
    for ex in examples:
        h = _example_content_hash(ex)
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique, len(examples) - len(unique)


# ---------------------------------------------------------------------------
# Dataset-level statistics
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    total: int = 0
    by_task_type: dict[str, int] = field(default_factory=dict)
    by_edit_quality: dict[str, int] = field(default_factory=dict)
    unique_task_ids: int = 0
    duplicates_removed: int = 0
    validation_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_task_type": self.by_task_type,
            "by_edit_quality": self.by_edit_quality if self.by_edit_quality else None,
            "unique_task_ids": self.unique_task_ids,
            "duplicates_removed": self.duplicates_removed,
            "validation_errors": self.validation_errors,
        }


def compute_dataset_stats(examples: list[TrainingExample]) -> DatasetStats:
    """Compute summary statistics for a set of training examples."""
    by_type: Counter[str] = Counter()
    by_quality: Counter[str] = Counter()
    task_ids: set[str] = set()

    for ex in examples:
        by_type[ex.task_type] += 1
        if ex.task_id:
            task_ids.add(ex.task_id)
        if ex.task_type == "NEXT_EDIT":
            q = ex.target.get("edit_quality", "unset")
            by_quality[q] += 1

    return DatasetStats(
        total=len(examples),
        by_task_type=dict(sorted(by_type.items())),
        by_edit_quality=dict(sorted(by_quality.items())),
        unique_task_ids=len(task_ids),
    )
