"""Goal-directed multi-step planner.

Replaces shallow decomposition for 3-5 step tasks by reasoning about
intermediate goals. Each reducer must improve the GoalState (lower
changed_pixel_fraction or closer to target_shape) or it's rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.search.sketch import SearchProgram, SearchStep
from aria.search.task_analysis import TaskAnalysis, analyze_task


@dataclass(frozen=True)
class GoalState:
    target_shape: tuple[int, int] | None
    diff_type: str
    required_colors: frozenset[int]
    removed_colors: frozenset[int]
    changed_pixel_fraction: float


def _goal_from_analysis(demos, analysis):
    """Derive initial GoalState from TaskAnalysis and demo outputs."""
    shapes = {out.shape for _, out in demos}
    target_shape = next(iter(shapes)) if len(shapes) == 1 else None
    return GoalState(
        target_shape=target_shape,
        diff_type=analysis.diff_type,
        required_colors=analysis.new_colors,
        removed_colors=analysis.removed_colors,
        changed_pixel_fraction=analysis.changed_pixel_fraction,
    )


def _improves(old_goal, new_goal):
    """Check if new_goal is strictly closer to solved than old_goal."""
    # Lower changed_pixel_fraction is always an improvement
    if new_goal.changed_pixel_fraction < old_goal.changed_pixel_fraction - 0.01:
        return True
    # Fewer removed colors to handle
    if len(new_goal.removed_colors) < len(old_goal.removed_colors):
        return True
    # Diff type simplified (mixed → something more specific)
    if old_goal.diff_type == 'mixed' and new_goal.diff_type != 'mixed':
        return True
    return False


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def _build_reducers(analysis):
    """Build gated reducers for the planner."""
    reducers = []

    if analysis.is_extraction or analysis.dims_change:
        reducers.append(_Reducer(
            name='crop_nonbg',
            apply=_r_crop_nonbg,
            program=SearchProgram(
                steps=[SearchStep('crop_nonbg', {})],
                provenance='planner:crop_nonbg',
            ),
            compatible=lambda a: a.is_extraction or a.dims_change,
        ))

    if analysis.has_panels or analysis.has_separators:
        reducers.append(_Reducer(
            name='extract_panel_0',
            apply=_r_extract_panel_0,
            program=SearchProgram(
                steps=[SearchStep('extract_panel', {'index': 0})],
                provenance='planner:extract_panel_0',
            ),
            compatible=lambda a: a.has_panels or a.has_separators,
        ))

    for c in analysis.removed_colors:
        if c == 0:
            continue
        reducers.append(_Reducer(
            name=f'remove_color_{c}',
            apply=lambda inp, color=c: _r_remove_color(inp, color),
            program=SearchProgram(
                steps=[SearchStep('remove_color', {'color': c})],
                provenance=f'planner:remove_color_{c}',
            ),
            compatible=lambda a: True,
        ))

    if analysis.same_dims and analysis.diff_type in ('rearrange', 'mixed'):
        for xname, xfn in [('flip_h', lambda g: g[:, ::-1]),
                            ('flip_v', lambda g: g[::-1, :]),
                            ('rot180', lambda g: np.rot90(g, 2))]:
            reducers.append(_Reducer(
                name=f'transform_{xname}',
                apply=lambda inp, fn=xfn: fn(inp).copy(),
                program=SearchProgram(
                    steps=[SearchStep('grid_transform', {'xform': xname})],
                    provenance=f'planner:transform_{xname}',
                ),
                compatible=lambda a: a.same_dims and a.diff_type in ('rearrange', 'mixed'),
            ))

    return reducers


@dataclass(frozen=True)
class _Reducer:
    name: str
    apply: object  # Callable[[np.ndarray], np.ndarray]
    program: SearchProgram
    compatible: object  # Callable[[TaskAnalysis], bool]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan_search(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
    *,
    max_depth: int = 4,
) -> SearchProgram | None:
    """Goal-directed multi-step search.

    At each depth:
      1) Try derive on the current sub-problem.
      2) For each compatible reducer that improves the goal, recurse.
    """
    goal = _goal_from_analysis(demos, analysis)
    return _plan_step(demos, analysis, goal, [], max_depth)


def _plan_step(demos, analysis, goal, prefix_steps, depth):
    """Recursive planner step."""
    if depth <= 0:
        return None

    # Try to solve the current sub-problem directly
    from aria.search.derive import derive_programs
    derived = derive_programs(demos)
    if derived:
        prog = derived[0]
        composed = SearchProgram(
            steps=prefix_steps + prog.steps,
            provenance=f'plan:{"+".join(s.action for s in prefix_steps)}+{prog.provenance}' if prefix_steps else prog.provenance,
        )
        # Verify composed program on the ORIGINAL demos (not sub-demos)
        # But we don't have the originals here — just verify on current sub-demos
        if prog.verify(demos):
            return composed

    # Try each reducer
    reducers = _build_reducers(analysis)
    for reducer in reducers:
        sub_demos = []
        ok = True
        for inp, out in demos:
            try:
                mid = reducer.apply(inp)
            except Exception:
                ok = False
                break
            if mid.shape == inp.shape and np.array_equal(mid, inp):
                ok = False
                break
            if hasattr(mid, 'min') and int(mid.min()) < 0:
                ok = False
                break
            sub_demos.append((mid, out))
        if not ok or not sub_demos:
            continue

        # Re-analyze sub-problem
        sub_analysis = analyze_task(sub_demos)
        sub_goal = _goal_from_analysis(sub_demos, sub_analysis)

        # Only proceed if the reducer improves the goal
        if not _improves(goal, sub_goal):
            continue

        result = _plan_step(
            sub_demos, sub_analysis, sub_goal,
            prefix_steps + reducer.program.steps,
            depth - 1,
        )
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Reducer implementations
# ---------------------------------------------------------------------------

def _r_crop_nonbg(inp):
    from aria.guided.perceive import perceive
    bg = perceive(inp).bg
    nz = np.argwhere(inp != bg)
    if len(nz) == 0:
        return inp
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return inp[r0:r1, c0:c1].copy()


def _r_extract_panel_0(inp):
    from aria.guided.perceive import perceive
    facts = perceive(inp)
    if not facts.regions:
        return inp
    r = facts.regions[0]
    return inp[r.r0:r.r1, r.c0:r.c1].copy()


def _r_remove_color(inp, color):
    from aria.guided.perceive import perceive
    bg = perceive(inp).bg
    result = inp.copy()
    result[result == color] = bg
    return result
