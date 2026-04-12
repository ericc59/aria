"""Output-dimensions pre-solver.

Predicts output shape hypotheses before search. Cheap, deterministic,
run once per task. Used by decomposition planner and dims_change gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.search.task_analysis import TaskAnalysis


@dataclass(frozen=True)
class DimHypothesis:
    rule: str
    shape: tuple[int, int] | None
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)


def solve_output_dims(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
) -> list[DimHypothesis]:
    """Generate output-shape hypotheses consistent across all demos.

    Returns hypotheses sorted by confidence (descending).
    """
    hypotheses = []

    # 1) Constant output shape
    h = _try_constant_shape(demos)
    if h:
        hypotheses.append(h)

    # 2) Input scaled by factor k
    h = _try_scale_up(demos)
    if h:
        hypotheses.append(h)

    # 3) Input divided by k
    h = _try_scale_down(demos)
    if h:
        hypotheses.append(h)

    # 4) Object bbox match
    hs = _try_object_bbox(demos)
    hypotheses.extend(hs)

    # 5) Panel size match
    if analysis.has_separators:
        hs = _try_panel_size(demos)
        hypotheses.extend(hs)

    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses


def _try_constant_shape(demos):
    """All demos have the same output shape."""
    shapes = {out.shape for _, out in demos}
    if len(shapes) == 1:
        s = next(iter(shapes))
        return DimHypothesis(
            rule='constant', shape=s, confidence=1.0,
            meta={'note': 'all demos have identical output shape'},
        )
    return None


def _try_scale_up(demos):
    """Output = input * k for some k in 2..5."""
    for k in range(2, 6):
        if all(out.shape[0] == inp.shape[0] * k and
               out.shape[1] == inp.shape[1] * k
               for inp, out in demos):
            return DimHypothesis(
                rule='scale_up', shape=None, confidence=0.9,
                meta={'factor': k},
            )
    return None


def _try_scale_down(demos):
    """Output = input / k for some k in 2..5."""
    for k in range(2, 6):
        if all(inp.shape[0] % k == 0 and inp.shape[1] % k == 0 and
               out.shape[0] == inp.shape[0] // k and
               out.shape[1] == inp.shape[1] // k
               for inp, out in demos):
            return DimHypothesis(
                rule='scale_down', shape=None, confidence=0.9,
                meta={'factor': k},
            )
    return None


def _try_object_bbox(demos):
    """Output dims match the bbox of some object in each input."""
    from aria.guided.perceive import perceive

    hypotheses = []
    # For each demo, find objects whose bbox matches output dims
    # Then check if the same structural property holds across demos
    for di, (inp, out) in enumerate(demos):
        facts = perceive(inp)
        oh, ow = out.shape
        for obj in facts.objects:
            if obj.height == oh and obj.width == ow:
                hypotheses.append(DimHypothesis(
                    rule='object_bbox', shape=(oh, ow), confidence=0.7,
                    meta={'demo': di, 'color': obj.color},
                ))
                break  # One per demo is enough
        else:
            return []  # No match in this demo → can't generalize

    if not hypotheses:
        return []

    # Check consistency: all demos found a match
    if len(hypotheses) == len(demos):
        return [DimHypothesis(
            rule='object_bbox', shape=hypotheses[0].shape, confidence=0.7,
            meta={'note': 'output dims match object bbox in all demos'},
        )]
    return []


def _try_panel_size(demos):
    """Output dims match a panel/region size from separator detection."""
    from aria.guided.perceive import perceive

    hypotheses = []
    for di, (inp, out) in enumerate(demos):
        facts = perceive(inp)
        oh, ow = out.shape
        for region in facts.regions:
            rh = region.r1 - region.r0
            rw = region.c1 - region.c0
            if rh == oh and rw == ow:
                hypotheses.append(DimHypothesis(
                    rule='panel_size', shape=(oh, ow), confidence=0.8,
                    meta={'demo': di},
                ))
                break
        else:
            return []

    if len(hypotheses) == len(demos):
        return [DimHypothesis(
            rule='panel_size', shape=hypotheses[0].shape, confidence=0.8,
            meta={'note': 'output dims match panel size in all demos'},
        )]
    return []
