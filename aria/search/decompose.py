"""Analysis-gated decomposition search.

Replaces brute-force N-step composition with a structured planner
that applies compatible splitters, then runs derive on the sub-problem.
Max depth 3 (splitter + sub-splitter + leaf derive).

Gating rules (keep decomposition tractable):
  - Spatial splitters (crop, extract) only for dims_change or is_extraction.
  - Color splitters only when removed_colors or diff_type == 'recolor_only'.
  - Transform splitters only for same_dims + rearrange or mixed.
  - Object-removal splitters only for subtractive or mixed.
  - Panel splitters only when has_panels or has_separators.
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
    """Build the set of compatible splitters for this task.

    Each splitter is gated by analysis fields to avoid wasted search.
    """
    splitters = []

    # --- Spatial / extraction splitters (dims_change or extraction) ---

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

    # --- Panel splitters (has_panels or has_separators) ---

    if analysis.has_panels or analysis.has_separators:
        for idx in range(3):
            splitters.append(Splitter(
                name=f'extract_panel_{idx}',
                apply=lambda inp, i=idx: _apply_extract_panel(inp, i),
                program=SearchProgram(
                    steps=[SearchStep('extract_panel', {'index': idx})],
                    provenance=f'splitter:extract_panel_{idx}',
                ),
                compatible=lambda a: a.has_panels or a.has_separators,
            ))

        splitters.append(Splitter(
            name='extract_legend_region',
            apply=_apply_extract_legend_region,
            program=SearchProgram(
                steps=[SearchStep('extract_panel', {'mode': 'smallest'})],
                provenance='splitter:extract_legend_region',
            ),
            compatible=lambda a: a.has_panels or a.has_separators,
        ))

    # --- Color splitters ---

    # Remove specific colors (gated: only when colors actually disappear)
    for c in analysis.removed_colors:
        if c == 0:
            continue
        splitters.append(Splitter(
            name=f'remove_color_{c}',
            apply=lambda inp, color=c: _apply_remove_color(inp, color),
            program=SearchProgram(
                steps=[SearchStep('remove_color', {'color': c})],
                provenance=f'splitter:remove_color_{c}',
            ),
            compatible=lambda a: True,
        ))

    # NOTE: apply_color_map pruned — redundant with remove_color_c splitters
    # and had a bg=0 invariant mismatch.

    # --- Transform splitters (same_dims + rearrange or mixed) ---
    # Only flip_h, flip_v, rot180 preserve dims. rot90/rot270 change dims
    # so they're always identity-rejected under same_dims gating.

    if analysis.same_dims and analysis.diff_type in ('rearrange', 'mixed'):
        for xform_name, xfn in [
            ('flip_h', lambda g: g[:, ::-1]),
            ('flip_v', lambda g: g[::-1, :]),
            ('rot180', lambda g: np.rot90(g, 2)),
        ]:
            splitters.append(Splitter(
                name=f'apply_transform_{xform_name}',
                apply=lambda inp, fn=xfn: fn(inp).copy(),
                program=SearchProgram(
                    steps=[SearchStep('grid_transform', {'xform': xform_name})],
                    provenance=f'splitter:apply_transform_{xform_name}',
                ),
                compatible=lambda a: a.same_dims and a.diff_type in ('rearrange', 'mixed'),
            ))

    # --- Object-removal splitters (subtractive or mixed) ---

    if analysis.diff_type in ('subtractive', 'mixed'):
        for sel_name, sel_desc in [
            ('largest', 'remove largest object'),
            ('smallest', 'remove smallest object'),
            ('touches_border', 'remove border-touching objects'),
        ]:
            splitters.append(Splitter(
                name=f'remove_objects_{sel_name}',
                apply=lambda inp, s=sel_name: _apply_remove_objects(inp, s),
                program=SearchProgram(
                    steps=[SearchStep('remove', {}, _sel_from_name(sel_name))],
                    provenance=f'splitter:remove_objects_{sel_name}',
                ),
                compatible=lambda a: a.diff_type in ('subtractive', 'mixed'),
            ))

    # --- Extraction by unique-color object (dims_change or extraction) ---

    if analysis.dims_change or analysis.is_extraction:
        splitters.append(Splitter(
            name='crop_object_unique_color',
            apply=_apply_crop_unique_color,
            program=SearchProgram(
                steps=[SearchStep('crop_object', {'predicate': 'unique_color'})],
                provenance='splitter:crop_object_unique_color',
            ),
            compatible=lambda a: a.dims_change or a.is_extraction,
        ))

    return splitters


def _sel_from_name(name):
    """Build a StepSelect from a selector name."""
    from aria.search.sketch import StepSelect
    return StepSelect(role=name)


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
            # Reject invalid grids (empty, negative values, wrong dtype)
            if mid.size == 0 or (hasattr(mid, 'min') and int(mid.min()) < 0):
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
    from aria.guided.perceive import perceive
    bg = perceive(inp).bg
    nz = np.argwhere(inp != bg)
    if len(nz) == 0:
        return inp
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return inp[r0:r1, c0:c1].copy()


def _apply_remove_color(inp, color):
    """Replace all pixels of the given color with the background color."""
    from aria.guided.perceive import perceive
    bg = perceive(inp).bg
    result = inp.copy()
    result[result == color] = bg
    return result


def _apply_color_map_dict(inp, cmap):
    """Apply an explicit color mapping dict to the grid."""
    result = inp.copy()
    for src, tgt in cmap.items():
        result[inp == int(src)] = int(tgt)
    return result


def _apply_extract_panel(inp, index):
    """Extract a panel region by index."""
    from aria.guided.perceive import perceive
    facts = perceive(inp)
    if not facts.regions or index >= len(facts.regions):
        return inp  # unchanged → will be rejected by the identity check
    r = facts.regions[index]
    return inp[r.r0:r.r1, r.c0:r.c1].copy()


def _apply_extract_legend_region(inp):
    """Extract the smallest panel region (likely the legend)."""
    from aria.guided.perceive import perceive
    facts = perceive(inp)
    if not facts.regions or len(facts.regions) < 2:
        return inp
    smallest = min(facts.regions,
                   key=lambda r: (r.r1 - r.r0) * (r.c1 - r.c0))
    return inp[smallest.r0:smallest.r1, smallest.c0:smallest.c1].copy()


def _apply_derived_color_map(inp, analysis):
    """Apply a simple color map: removed_colors → bg."""
    from aria.guided.perceive import perceive
    bg = perceive(inp).bg
    result = inp.copy()
    for c in analysis.removed_colors:
        if c != bg:
            result[result == c] = bg
    return result


def _apply_remove_objects(inp, selector_name):
    """Remove objects matching a structural selector."""
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    facts = perceive(inp)
    bg = facts.bg

    name_to_pred = {
        'largest': Pred.IS_LARGEST,
        'smallest': Pred.IS_SMALLEST,
        'touches_border': Pred.TOUCHES_BORDER,
    }
    pred = name_to_pred.get(selector_name)
    if pred is None:
        return inp

    targets = prim_select(facts, [Predicate(pred)])
    if not targets:
        return inp

    result = inp.copy()
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg
    return result


def _apply_crop_unique_color(inp):
    """Crop to the bounding box of the object with a unique color."""
    from aria.guided.perceive import perceive
    from collections import Counter

    facts = perceive(inp)
    color_counts = Counter(o.color for o in facts.objects)
    unique = [o for o in facts.objects if color_counts[o.color] == 1]
    if len(unique) != 1:
        return inp  # identity → rejected by splitter loop
    obj = unique[0]
    return inp[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width].copy()
