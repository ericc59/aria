"""Residual-to-edit priors — structured mapping from failure labels to edit preferences.

An explicit inspectable table that maps (residual_category, lane, failure_label)
to preferred edit types. Used by editor search to prioritize edits based on
the current state's failure diagnosis.

Not a learned model. A compact rule table.
Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aria.core.editor_env import ActionType


@dataclass(frozen=True)
class EditPrior:
    """One entry in the residual-to-edit mapping."""
    action_types: tuple[ActionType, ...]   # preferred action types, in priority order
    binding_namespaces: tuple[str, ...]    # preferred specialization namespaces
    rationale: str


# ---------------------------------------------------------------------------
# The mapping table
# ---------------------------------------------------------------------------

# Key: (residual_category, lane_name) -> EditPrior
# lane_name can be "" for lane-independent entries

_PRIOR_TABLE: dict[tuple[str, str], EditPrior] = {
    # --- Near-perfect residuals: tiny tweaks ---
    ("near_perfect", ""): EditPrior(
        action_types=(ActionType.BIND, ActionType.SET_SLOT),
        binding_namespaces=("__task__", "__placement__", "__replicate__"),
        rationale="near-perfect: try small binding/slot adjustments",
    ),

    # --- Partial match: parameter-level repair ---
    ("partial_match", "periodic_repair"): EditPrior(
        action_types=(ActionType.BIND,),
        binding_namespaces=("__task__", "__periodic__"),
        rationale="periodic partial: try axis/period/mode alternatives",
    ),
    ("partial_match", "replication"): EditPrior(
        action_types=(ActionType.BIND,),
        binding_namespaces=("__replicate__",),
        rationale="replication partial: try key_rule/source_policy alternatives",
    ),
    ("partial_match", "relocation"): EditPrior(
        action_types=(ActionType.BIND,),
        binding_namespaces=("__placement__",),
        rationale="relocation partial: try match_rule/alignment alternatives",
    ),
    ("partial_match", ""): EditPrior(
        action_types=(ActionType.BIND, ActionType.SET_SLOT),
        binding_namespaces=("__task__",),
        rationale="partial: try parameter adjustments",
    ),

    # --- Wrong placement: alignment edits ---
    ("wrong_placement", ""): EditPrior(
        action_types=(ActionType.BIND,),
        binding_namespaces=("__placement__",),
        rationale="wrong placement: try alignment mode changes",
    ),

    # --- Wrong content: slot/op edits ---
    ("wrong_content", ""): EditPrior(
        action_types=(ActionType.SET_SLOT, ActionType.SET_NODE_OP),
        binding_namespaces=("__task__",),
        rationale="wrong content: try slot evidence or op changes",
    ),

    # --- Large mismatch: structural replacement ---
    ("large_mismatch", ""): EditPrior(
        action_types=(ActionType.REPLACE_SUBGRAPH, ActionType.SET_NODE_OP),
        binding_namespaces=(),
        rationale="large mismatch: try structural replacement or op swap",
    ),

    # --- No compile: structural edits ---
    ("no_compile", ""): EditPrior(
        action_types=(ActionType.ADD_NODE, ActionType.SET_NODE_OP, ActionType.ADD_EDGE),
        binding_namespaces=(),
        rationale="no compile: try structural graph edits",
    ),
}


def get_edit_prior(
    residual_category: str,
    lane_name: str = "",
) -> EditPrior:
    """Look up the preferred edit prior for a (residual, lane) pair.

    Tries (category, lane) first, then (category, "") as fallback.
    """
    key = (residual_category, lane_name)
    if key in _PRIOR_TABLE:
        return _PRIOR_TABLE[key]
    fallback = (residual_category, "")
    if fallback in _PRIOR_TABLE:
        return _PRIOR_TABLE[fallback]
    return EditPrior(
        action_types=(ActionType.BIND, ActionType.SET_SLOT),
        binding_namespaces=("__task__",),
        rationale="default: generic parameter edits",
    )


def get_all_priors() -> dict[tuple[str, str], EditPrior]:
    """Return the full mapping table for inspection."""
    return dict(_PRIOR_TABLE)
