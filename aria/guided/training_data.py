"""Training data: extract per-step supervision from latent programs.

For each synthetic task, the latent program tells us the correct
action sequence. We extract supervised examples for:
1. First-step prediction (what target+rewrite to try first)
2. Binding prediction (what parameters to bind)
3. Continue/stop prediction (NEXT vs STOP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.guided.grammar import Act, Target, Rewrite, Action, Program
from aria.guided.workspace import Workspace, build_workspace
from aria.guided.synthetic import SyntheticTask


# Vocabularies
TARGETS = list(Target)
REWRITES = list(Rewrite)
TARGET_TO_IDX = {t: i for i, t in enumerate(TARGETS)}
REWRITE_TO_IDX = {r: i for i, r in enumerate(REWRITES)}


@dataclass
class StepExample:
    """One supervised step example."""
    task_id: str
    step_idx: int           # which step in the program (0, 1, ...)
    features: np.ndarray    # workspace features
    # Labels
    target_idx: int         # which Target
    rewrite_idx: int        # which Rewrite
    bindings: dict[str, Any]
    is_last: bool           # STOP (True) or NEXT (False)


def extract_step_examples(tasks: list[SyntheticTask]) -> list[StepExample]:
    """Extract per-step supervised examples from latent programs."""
    examples = []
    for task in tasks:
        prog = task.latent_program
        # Use ALL demo pairs for training, not just demo 0
        for inp, out in task.train:
            ws = build_workspace(inp, out)
            features = _featurize_workspace(ws)

            step_idx = 0
            current_target = None
            current_rewrite = None
            current_bindings: dict[str, Any] = {}

            for action in prog.actions:
                if action.act == Act.SELECT_TARGET:
                    current_target = action.choice
                elif action.act == Act.REWRITE:
                    current_rewrite = action.choice
                elif action.act == Act.BIND:
                    current_bindings[action.param_name] = action.param_value
                elif action.act in (Act.NEXT, Act.STOP):
                    if current_target is not None and current_rewrite is not None:
                        examples.append(StepExample(
                            task_id=task.task_id,
                            step_idx=step_idx,
                            features=features,
                            target_idx=TARGET_TO_IDX.get(current_target, 0),
                            rewrite_idx=REWRITE_TO_IDX.get(current_rewrite, 0),
                            bindings=dict(current_bindings),
                            is_last=(action.act == Act.STOP),
                        ))
                    step_idx += 1
                    current_target = None
                    current_rewrite = None
                    current_bindings = {}

    return examples


def _featurize_workspace(ws: Workspace) -> np.ndarray:
    """Convert workspace to a fixed-size feature vector."""
    f = ws.serialize()
    objs = f.get("objects", [])
    units = f.get("residual_units", [])
    rels = f.get("relations", [])

    obj_color_hist = [0] * 10
    for o in objs:
        obj_color_hist[o["color"]] += 1

    n_adj = sum(1 for r in rels if r["type"] == "adjacent")
    n_contains = sum(1 for r in rels if r["type"] == "contains")

    sizes = [o["size"] for o in objs]

    x = np.array([
        f["rows"], f["cols"], f["bg"],
        f["n_preserved"], f["n_residual"], f["preservation_ratio"],
        f["n_objects"], len(rels), len(units),
        sum(1 for u in units if u["change_type"] == "add"),
        sum(1 for u in units if u["change_type"] == "delete"),
        sum(1 for u in units if u["change_type"] == "recolor"),
        sum(1 for u in units if u["change_type"] == "mixed"),
        max(sizes, default=0), min(sizes, default=0),
        sum(1 for o in objs if o["singleton"]),
        len(set(o["color"] for o in objs)),
        n_adj, n_contains,
        max((u["n_pixels"] for u in units), default=0),
        *obj_color_hist,
    ], dtype=np.float32)
    return x
