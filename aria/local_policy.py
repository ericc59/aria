"""Local learned-policy interface for refinement guidance.

Defines the contract that a future trained causal-LM (e.g. Qwen) must satisfy
to replace heuristic refinement planning.  Ships two concrete implementations:

  1. HeuristicBaselinePolicy – deterministic mock for testing and baselines.
  2. LocalCausalLMPolicy   – stub that loads a HuggingFace-style local model.

No remote providers.  No runtime-solver coupling.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Data containers passed to / returned from the policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyInput:
    """Read-only snapshot the policy uses to make decisions.

    Fields intentionally mirror the data already available in
    ``RefinementFeedback`` and ``SearchTraceEntry`` so the caller can
    build a ``PolicyInput`` from existing types without new plumbing.
    """
    task_signatures: tuple[str, ...]
    round_index: int
    prior_focuses: tuple[str, ...]          # suggested_focus history
    prior_error_types: tuple[str | None, ...]
    candidate_ops: tuple[str, ...] = ()     # ops available in current library
    # Best-candidate diff signals from the most recent round's feedback.
    best_candidate_score: float | None = None
    best_candidate_error_type: str | None = None
    best_candidate_dims_match: bool | None = None
    best_candidate_pixel_diff_count: int | None = None
    best_candidate_wrong_row_count: int | None = None
    best_candidate_wrong_col_count: int | None = None
    best_candidate_palette_expected_coverage: float | None = None
    best_candidate_palette_precision: float | None = None
    best_candidate_preserved_input_ratio: float | None = None
    best_candidate_changed_cells_ratio: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FocusPrediction:
    """Policy output for the NEXT_FOCUS task."""
    focus: str                     # e.g. "marker_geometry", "size", "color_map", "generic"
    confidence: float = 1.0        # 0-1, for ranking / thresholding


@dataclass(frozen=True)
class ActionRanking:
    """Policy output for action/edit ranking."""
    ranked_actions: tuple[str, ...]   # best-first
    scores: tuple[float, ...] = ()    # optional logit / probability per action


@dataclass(frozen=True)
class Sketch:
    """Optional programme sketch emitted by the policy."""
    steps: tuple[str, ...]     # pseudo-DSL step strings
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LocalPolicy(abc.ABC):
    """Contract that any local learned policy must satisfy."""

    @abc.abstractmethod
    def predict_next_focus(self, inp: PolicyInput) -> FocusPrediction:
        """Predict which refinement focus to try next."""

    @abc.abstractmethod
    def rank_actions(self, inp: PolicyInput, candidates: Sequence[str]) -> ActionRanking:
        """Rank a set of candidate actions / op names."""

    def emit_sketch(self, inp: PolicyInput) -> Sketch | None:
        """Optionally emit a programme sketch.  Default: no sketch."""
        return None


# ---------------------------------------------------------------------------
# 1. Heuristic / mock baseline
# ---------------------------------------------------------------------------

_FOCUS_PRIORITY: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset({"change:additive", "role:has_marker", "dims:same"}),
        "marker_geometry",
    ),
    (
        frozenset({"color:new_in_output", "dims:same"}),
        "color_map",
    ),
    (
        frozenset({"color:palette_subset", "dims:same"}),
        "color_map",
    ),
)


class HeuristicBaselinePolicy(LocalPolicy):
    """Deterministic policy that mirrors the hand-built rules in refinement.py."""

    def predict_next_focus(self, inp: PolicyInput) -> FocusPrediction:
        sigs = frozenset(inp.task_signatures)

        # If we already tried a non-generic focus, escalate to generic.
        if inp.prior_focuses and inp.prior_focuses[-1] != "generic":
            return FocusPrediction(focus="generic", confidence=0.5)

        for required, focus in _FOCUS_PRIORITY:
            if required <= sigs:
                return FocusPrediction(focus=focus, confidence=0.8)

        same_size = "dims:same" in sigs
        if not same_size:
            return FocusPrediction(focus="size", confidence=0.7)

        return FocusPrediction(focus="generic", confidence=0.4)

    def rank_actions(self, inp: PolicyInput, candidates: Sequence[str]) -> ActionRanking:
        # Stable sort: keep original order (no learned signal).
        return ActionRanking(ranked_actions=tuple(candidates))


# ---------------------------------------------------------------------------
# 2. Local causal-LM wrapper stub
# ---------------------------------------------------------------------------

class LocalCausalLMPolicy(LocalPolicy):
    """Stub for a HuggingFace-style local causal-LM policy.

    The constructor accepts the same kwargs that ``AutoModelForCausalLM.from_pretrained``
    would, but does NOT import transformers at module level.  This lets the rest
    of the codebase import the interface without pulling in heavy ML deps.

    When ``dry_run=True`` (the default), all methods return neutral / empty
    results and never touch the GPU.  This is the mode used in unit tests and
    in CI where no model weights are present.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        *,
        device: str = "cpu",
        dry_run: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dry_run = dry_run
        self.model_kwargs = model_kwargs or {}
        self._model: Any = None
        self._tokenizer: Any = None

        if not dry_run:
            self._load_model()

    # -- lazy model loading ---------------------------------------------------

    def _load_model(self) -> None:
        """Import transformers and load weights.  Raises ImportError when
        the ``transformers`` package is not installed."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, **self.model_kwargs
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **self.model_kwargs
        ).to(self.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- interface methods ----------------------------------------------------

    def predict_next_focus(self, inp: PolicyInput) -> FocusPrediction:
        if self.dry_run:
            return FocusPrediction(focus="generic", confidence=0.0)
        return self._generate_focus(inp)

    def rank_actions(self, inp: PolicyInput, candidates: Sequence[str]) -> ActionRanking:
        if self.dry_run:
            return ActionRanking(ranked_actions=tuple(candidates))
        return self._score_and_rank(inp, candidates)

    def emit_sketch(self, inp: PolicyInput) -> Sketch | None:
        if self.dry_run:
            return None
        return self._generate_sketch(inp)

    # -- private generation stubs (to be filled when weights exist) -----------

    def _generate_focus(self, inp: PolicyInput) -> FocusPrediction:
        """Score each focus label via constrained generation."""
        # Future: tokenize prompt, run model.generate, parse label.
        raise NotImplementedError("Model inference not yet implemented")

    def _score_and_rank(self, inp: PolicyInput, candidates: Sequence[str]) -> ActionRanking:
        """Score candidates by log-likelihood under the model."""
        raise NotImplementedError("Model inference not yet implemented")

    def _generate_sketch(self, inp: PolicyInput) -> Sketch | None:
        """Autoregressively generate a programme sketch."""
        raise NotImplementedError("Model inference not yet implemented")
