"""Shared types for the Layered Workspace Solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.types import Grid, DemoPair


# ---------------------------------------------------------------------------
# Stage outputs — each stage produces one of these per demo
# ---------------------------------------------------------------------------

@dataclass
class Scaffold:
    """Stage 0: output size and coarse structure."""
    output_shape: tuple[int, int]
    same_shape: bool
    bg: int


@dataclass
class Workspace:
    """Stage 1: initialized output canvas."""
    canvas: Grid               # starts as bg-filled, accumulates layers
    bg: int
    palette: set[int]


@dataclass
class PreservedSupport:
    """Stage 2: what's preserved from input."""
    preserved_mask: np.ndarray   # bool: True = explained as preserved
    residual_mask: np.ndarray    # bool: True = still needs explanation
    n_preserved: int
    n_residual: int


@dataclass
class OutputUnit:
    """Stage 3: one piece of the residual to explain."""
    unit_id: str
    mask: np.ndarray            # bool mask in output grid coords
    content: Grid               # output values at this unit
    input_support: Grid | None  # corresponding input subgrid
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alignment:
    """Stage 4: input-support alignment for one output unit."""
    unit: OutputUnit
    input_region: Grid          # the input subgrid that supports this unit
    input_bbox: tuple[int, int, int, int]
    alignment_type: str         # "same_pos", "same_panel", "enclosing_obj", etc.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    """Stage 5: which cells within the alignment are targets."""
    target_mask: np.ndarray     # bool mask in output grid coords
    description: str
    confidence: float


@dataclass
class RewriteRule:
    """Stage 6: the local transformation for the residual."""
    rule_type: str              # "periodic_repair", "recolor_to_singleton", etc.
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def canon_key(self) -> tuple:
        """Structural key for cross-demo comparison (strips data-specific params)."""
        return (self.rule_type, tuple(sorted(
            (k, v) for k, v in self.params.items()
            if not isinstance(v, np.ndarray)
        )))


@dataclass
class UnifiedRule:
    """Stage 7: shared rule across all demos."""
    rule: RewriteRule
    train_verified: bool
    train_diff: int


@dataclass
class SolveResult:
    """Stage 9: final result."""
    task_id: str
    train_verified: bool
    test_outputs: list[Grid]
    unified_rule: UnifiedRule | None
    train_diff: int
    details: dict[str, Any] = field(default_factory=dict)
    stage_trace: dict[str, Any] = field(default_factory=dict)
