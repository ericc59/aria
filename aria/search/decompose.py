"""Analysis-gated decomposition search.

Replaces brute-force N-step composition with a structured planner
that applies compatible splitters, then runs derive on the sub-problem.
Max depth 3 (splitter + sub-splitter + leaf derive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep
from aria.search.task_analysis import TaskAnalysis


@dataclass(frozen=True)
class Splitter:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    program: SearchProgram
    compatible: Callable[[TaskAnalysis], bool]


def _build_splitters(analysis: TaskAnalysis) -> list[Splitter]:
    """Build the set of compatible splitters for this task."""
    splitters = []

    # crop_non_bg_bbox: extract bounding box of non-bg content
    if analysis.is_extraction or analysis.dims_change:
        splitters.append(Splitter(
            name='crop_non_bg_bbox',
            apply=_apply_crop_non_bg,
            program=SearchProgram(
                steps=[SearchStep('crop_nonbg', {})],
                provenance='splitter:crop_non_bg_bbox',
            ),
            compatible=lambda a: a.is_extraction or a.dims_change,
        ))

    # remove_color_c: zero out a specific color
    for c in analysis.removed_colors:
        if c == 0:
            continue
        splitters.append(Splitter(
            name=f'remove_color_{c}',
            apply=lambda inp, color=c: _apply_remove_color(inp, color),
            program=SearchProgram(
                steps=[SearchStep('recolor_map', {'color_map': {c: 0}})],
                provenance=f'splitter:remove_color_{c}',
            ),
            compatible=lambda a: True,
        ))

    # extract_panel_0: extract first panel region
    if analysis.has_panels:
        splitters.append(Splitter(
            name='extract_panel_0',
            apply=_apply_extract_panel_0,
            program=SearchProgram(
                steps=[SearchStep('crop_fixed', {'r0': 0, 'c0': 0, 'h': 1, 'w': 1})],
                provenance='splitter:extract_panel_0',
            ),
            compatible=lambda a: a.has_panels,
        ))

    return splitters


def search_decomposed(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
    *,
    max_depth: int = 3,
) -> SearchProgram | None:
    """Try to decompose the task via splitter + sub-derive.

    For each compatible splitter:
      1) Apply to all demo inputs → intermediate grids.
      2) Run derive on (intermediate, output) pairs.
      3) If sub-program verifies, compose splitter + sub-program.
      4) If max_depth > 1, recurse on sub-problem.
    """
    if max_depth <= 0:
        return None

    splitters = _build_splitters(analysis)

    for splitter in splitters:
        # Apply splitter to all demo inputs
        sub_demos = []
        ok = True
        for inp, out in demos:
            try:
                mid = splitter.apply(inp)
            except Exception:
                ok = False
                break
            if mid.shape == inp.shape and np.array_equal(mid, inp):
                ok = False
                break
            sub_demos.append((mid, out))
        if not ok or not sub_demos:
            continue

        # Try derive on the sub-problem
        from aria.search.derive import derive_programs
        sub_progs = derive_programs(sub_demos)
        for sub_prog in sub_progs:
            composed = SearchProgram(
                steps=splitter.program.steps + sub_prog.steps,
                provenance=f'decompose:{splitter.name}+{sub_prog.provenance}',
            )
            if composed.verify(demos):
                return composed

        # Recurse if depth allows
        if max_depth > 1:
            from aria.search.task_analysis import analyze_task
            sub_analysis = analyze_task(sub_demos)
            sub_result = search_decomposed(
                sub_demos, sub_analysis, max_depth=max_depth - 1)
            if sub_result is not None:
                composed = SearchProgram(
                    steps=splitter.program.steps + sub_result.steps,
                    provenance=f'decompose:{splitter.name}+{sub_result.provenance}',
                )
                if composed.verify(demos):
                    return composed

    return None


# ---------------------------------------------------------------------------
# Splitter implementations
# ---------------------------------------------------------------------------

def _apply_crop_non_bg(inp):
    """Crop to bounding box of non-bg content."""
    bg = 0
    nz = np.argwhere(inp != bg)
    if len(nz) == 0:
        return inp
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return inp[r0:r1, c0:c1].copy()


def _apply_remove_color(inp, color):
    """Zero out all pixels of the given color."""
    result = inp.copy()
    result[result == color] = 0
    return result


def _apply_extract_panel_0(inp):
    """Extract the first (top-left) panel region."""
    from aria.guided.perceive import perceive
    facts = perceive(inp)
    if not facts.regions:
        return inp
    r = facts.regions[0]
    return inp[r.r0:r.r1, r.c0:r.c1].copy()
