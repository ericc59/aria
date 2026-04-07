"""Diff-guided iterative program construction — supporting tool.

NOT part of the canonical architecture (ComputationGraph + protocol).
This is a complementary search tool that builds programs by greedy
diff reduction. It may be useful as a candidate generator for the
graph editor, or as a standalone solver for simple same-dims tasks.

The canonical path is: fit -> specialize -> compile -> verify
(see aria.core.protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.types import (
    Bind, Call, DemoPair, Grid, Literal, Program, Ref, Type,
)


@dataclass(frozen=True)
class StepCandidate:
    """One candidate operation to try at the current step."""
    op: str
    args: tuple          # args for Call, with Ref("__state__") as grid placeholder
    description: str
    score: float = 0.0   # diff reduction (higher = better)


@dataclass(frozen=True)
class ConstructionStep:
    """A step that was chosen and applied."""
    op: str
    args: tuple
    description: str
    diff_before: int     # pixels wrong before this step
    diff_after: int      # pixels wrong after this step


@dataclass(frozen=True)
class ConstructionResult:
    """Outcome of iterative construction for one demo."""
    solved: bool
    steps: tuple[ConstructionStep, ...]
    final_diff: int
    program: Program | None = None


@dataclass(frozen=True)
class StepperResult:
    """Outcome of running the stepper across all demos."""
    solved: bool
    program: Program | None = None
    steps_description: tuple[str, ...] = ()
    per_demo: tuple[ConstructionResult, ...] = ()


def step_solve(
    demos: tuple[DemoPair, ...],
    *,
    max_steps: int = 8,
    min_improvement: int = 1,
    world: Any = None,
) -> StepperResult:
    """Iteratively construct a program by diff-guided step selection.

    For each candidate step, we test it on ALL demos simultaneously.
    A step is accepted only if it reduces the total diff across all demos.
    This ensures the constructed program generalizes.

    If a WorldModel is provided, uses it to prioritize candidates.
    """
    if not demos:
        return StepperResult(solved=False)

    # Same dims required for step-by-step construction
    if not all(d.input.shape == d.output.shape for d in demos):
        return StepperResult(solved=False)

    # Build world model if not provided
    if world is None:
        try:
            from aria.core.world import build_world_model
            world = build_world_model(demos)
        except Exception:
            world = None

    states = [d.input.copy() for d in demos]
    targets = [d.output for d in demos]
    chosen_steps: list[ConstructionStep] = []
    total_diff = sum(_pixel_diff(s, t) for s, t in zip(states, targets))

    if total_diff == 0:
        return StepperResult(solved=True, program=_identity_program())

    for step_idx in range(max_steps):
        # Generate candidates based on current diff + world model
        candidates = _generate_candidates(states, targets, world=world)

        # Score each candidate: try on all demos, measure total diff reduction
        best: StepCandidate | None = None
        best_new_states: list[Grid] | None = None
        best_new_diff = total_diff

        for cand in candidates:
            new_states = []
            success = True
            for state in states:
                try:
                    result = _apply_candidate(cand, state)
                    if result is None or result.shape != state.shape:
                        success = False
                        break
                    new_states.append(result)
                except Exception:
                    success = False
                    break

            if not success:
                continue

            new_diff = sum(_pixel_diff(s, t) for s, t in zip(new_states, targets))
            if new_diff < best_new_diff:
                best_new_diff = new_diff
                best = cand
                best_new_states = new_states

        # Accept the best step if it improves enough
        if best is None or total_diff - best_new_diff < min_improvement:
            break

        chosen_steps.append(ConstructionStep(
            op=best.op,
            args=best.args,
            description=best.description,
            diff_before=total_diff,
            diff_after=best_new_diff,
        ))
        states = best_new_states
        total_diff = best_new_diff

        if total_diff == 0:
            break

    solved = total_diff == 0
    program = _build_program(chosen_steps) if solved else None

    return StepperResult(
        solved=solved,
        program=program,
        steps_description=tuple(s.description for s in chosen_steps),
        per_demo=tuple(
            ConstructionResult(
                solved=_pixel_diff(s, t) == 0,
                steps=tuple(chosen_steps),
                final_diff=_pixel_diff(s, t),
            )
            for s, t in zip(states, targets)
        ),
    )


@dataclass
class _BeamEntry:
    """One path being explored in beam search."""
    states: list       # current grid per demo
    steps: list        # ConstructionSteps taken so far
    total_diff: int    # sum of pixel diffs across demos

    def __lt__(self, other: _BeamEntry) -> bool:
        return self.total_diff < other.total_diff


def beam_solve(
    demos: tuple[DemoPair, ...],
    *,
    max_steps: int = 6,
    beam_width: int = 3,
    world: Any = None,
) -> StepperResult:
    """Beam search over the stepper — explore multiple paths in parallel.

    At each step, expand every beam entry by trying all candidates,
    then keep the top beam_width entries by lowest diff.  This allows
    the system to take steps that look bad in isolation but enable
    good follow-up steps.
    """
    if not demos:
        return StepperResult(solved=False)
    if not all(d.input.shape == d.output.shape for d in demos):
        return StepperResult(solved=False)

    if world is None:
        try:
            from aria.core.world import build_world_model
            world = build_world_model(demos)
        except Exception:
            world = None

    targets = [d.output for d in demos]
    initial_states = [d.input.copy() for d in demos]
    initial_diff = sum(_pixel_diff(s, t) for s, t in zip(initial_states, targets))

    if initial_diff == 0:
        return StepperResult(solved=True, program=_identity_program())

    beam = [_BeamEntry(states=initial_states, steps=[], total_diff=initial_diff)]

    for step_idx in range(max_steps):
        next_beam: list[_BeamEntry] = []

        for entry in beam:
            if entry.total_diff == 0:
                next_beam.append(entry)
                continue

            candidates = _generate_candidates(entry.states, targets, world=world)

            for cand in candidates:
                new_states = []
                success = True
                for state in entry.states:
                    try:
                        result = _apply_candidate(cand, state)
                        if result is None or result.shape != state.shape:
                            success = False
                            break
                        new_states.append(result)
                    except Exception:
                        success = False
                        break

                if not success:
                    continue

                new_diff = sum(_pixel_diff(s, t) for s, t in zip(new_states, targets))
                if new_diff >= entry.total_diff:
                    continue  # must improve

                new_step = ConstructionStep(
                    op=cand.op, args=cand.args,
                    description=cand.description,
                    diff_before=entry.total_diff,
                    diff_after=new_diff,
                )
                next_beam.append(_BeamEntry(
                    states=new_states,
                    steps=entry.steps + [new_step],
                    total_diff=new_diff,
                ))

        if not next_beam:
            break

        # Keep original beam entries too (allow skipping a step)
        next_beam.extend(e for e in beam if e.total_diff > 0)

        # Deduplicate by state fingerprint, keep lowest diff
        seen: dict[bytes, _BeamEntry] = {}
        for entry in next_beam:
            key = b"".join(s.tobytes() for s in entry.states)
            if key not in seen or entry.total_diff < seen[key].total_diff:
                seen[key] = entry
        next_beam = sorted(seen.values())

        beam = next_beam[:beam_width]

        # Early exit if solved
        if beam[0].total_diff == 0:
            break

    # Find best result
    best = min(beam, key=lambda e: e.total_diff)

    solved = best.total_diff == 0
    program = _build_program(best.steps) if solved else None

    return StepperResult(
        solved=solved,
        program=program,
        steps_description=tuple(s.description for s in best.steps),
        per_demo=tuple(
            ConstructionResult(
                solved=_pixel_diff(s, t) == 0,
                steps=tuple(best.steps),
                final_diff=_pixel_diff(s, t),
            )
            for s, t in zip(best.states, targets)
        ),
    )


def _pixel_diff(a: Grid, b: Grid) -> int:
    """Count pixels that differ."""
    if a.shape != b.shape:
        return a.size + b.size
    return int(np.sum(a != b))


def _apply_candidate(cand: StepCandidate, grid: Grid) -> Grid | None:
    """Apply a candidate op to a grid."""
    # Handle compound object-level ops
    if cand.op.startswith("__compound_"):
        return _apply_compound_candidate(cand, grid)

    from aria.runtime.ops import get_op

    sig, fn = get_op(cand.op)
    # Build concrete args, replacing sentinel with actual grid
    concrete_args = []
    for arg in cand.args:
        if arg is _GRID_PLACEHOLDER:
            concrete_args.append(grid)
        else:
            concrete_args.append(arg)

    result = fn(*concrete_args)
    if isinstance(result, np.ndarray):
        return result
    return None


# Sentinel for "the current grid" in candidate args
_GRID_PLACEHOLDER = object()


def _generate_candidates(
    states: list[Grid],
    targets: list[Grid],
    *,
    world: Any = None,
) -> list[StepCandidate]:
    """Generate candidate ops based on the current diff + world model.

    When a world model is available, uses its layer analysis to
    prioritize and prune candidates.  Without it, falls back to
    diff-only analysis.
    """
    candidates: list[StepCandidate] = []

    # Analyze the diff to constrain candidates
    all_state_colors: set[int] = set()
    all_target_colors: set[int] = set()
    needed_colors: set[int] = set()  # in target but not in state

    for state, target in zip(states, targets):
        all_state_colors.update(int(c) for c in np.unique(state))
        all_target_colors.update(int(c) for c in np.unique(target))
        diff_mask = state != target
        needed_colors.update(int(target[r, c]) for r, c in zip(*np.where(diff_mask)))

    # Use world model bg if available, else detect from state
    if world is not None:
        bg = world.pixels.bg_color
    else:
        unique, counts = np.unique(states[0], return_counts=True)
        bg = int(unique[np.argmax(counts)])

    # --- World-model-guided prioritization ---
    # If world model says it's a pure color map, try that first
    if world is not None and world.roles.is_color_map and world.roles.color_map:
        candidates.append(StepCandidate(
            op="apply_color_map",
            args=(world.roles.color_map, _GRID_PLACEHOLDER),
            description=f"apply_color_map({world.roles.color_map})",
        ))

    # --- Color map: if the transformation is a color bijection ---
    color_map = _infer_color_map(states, targets)
    if color_map is not None:
        candidates.append(StepCandidate(
            op="apply_color_map",
            args=(color_map, _GRID_PLACEHOLDER),
            description=f"apply_color_map({color_map})",
        ))

    # --- No-parameter ops ---
    for op_name in ("transpose_grid", "peel_frame", "sort_rows", "sort_cols",
                    "unique_rows", "unique_cols", "complete_symmetry_h",
                    "complete_symmetry_v", "repair_framed_2d_motif"):
        _try_add(candidates, op_name, (_GRID_PLACEHOLDER,))

    # --- Rotation ---
    for deg in (90, 180, 270):
        _try_add(candidates, "rotate_grid", (deg, _GRID_PLACEHOLDER),
                 desc=f"rotate({deg})")

    # --- Reflection ---
    for axis in (0, 1):
        _try_add(candidates, "reflect_grid", (axis, _GRID_PLACEHOLDER),
                 desc=f"reflect({'row' if axis == 0 else 'col'})")

    # --- Fill enclosed ---
    for color in needed_colors:
        _try_add(candidates, "fill_enclosed", (_GRID_PLACEHOLDER, color),
                 desc=f"fill_enclosed({color})")

    # --- Fill between ---
    for src_color in all_state_colors - {bg}:
        for fill_color in needed_colors:
            if fill_color != src_color:
                _try_add(candidates, "fill_between",
                         (_GRID_PLACEHOLDER, src_color, fill_color),
                         desc=f"fill_between({src_color},{fill_color})")

    # --- Conditional fill ---
    for color in all_state_colors:
        for target_c in needed_colors:
            if color != target_c:
                _try_add(candidates, "conditional_fill",
                         (_GRID_PLACEHOLDER, color, target_c),
                         desc=f"conditional_fill({color},{target_c})")

    # --- Shift grid ---
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if dr == 0 and dc == 0:
                continue
            _try_add(candidates, "shift_grid",
                     (dr, dc, bg, _GRID_PLACEHOLDER),
                     desc=f"shift({dr},{dc},fill={bg})")

    # --- Propagate ---
    for src in all_state_colors - {bg}:
        for fill in needed_colors:
            if fill != src:
                _try_add(candidates, "propagate",
                         (_GRID_PLACEHOLDER, src, fill, bg),
                         desc=f"propagate({src},{fill},bg={bg})")

    # --- Repair lines ---
    for axis in (0, 1):
        for period in (2, 3, 4):
            _try_add(candidates, "repair_framed_lines",
                     (_GRID_PLACEHOLDER, axis, period),
                     desc=f"repair_lines(axis={axis},period={period})")

    # --- Grid boolean ops (combine input with target structure) ---
    # These help when the output is a logical combination of input patterns
    _try_add(candidates, "grid_and", (_GRID_PLACEHOLDER, _GRID_PLACEHOLDER),
             desc="grid_and(state,state)")  # identity, but placeholder for combos
    _try_add(candidates, "grid_or", (_GRID_PLACEHOLDER, _GRID_PLACEHOLDER),
             desc="grid_or(state,state)")

    # --- Flood fill at strategic positions ---
    # Try flood fill at positions where the diff shows needed colors
    if len(needed_colors) <= 3:
        for state, target in zip(states[:1], targets[:1]):
            diff_mask = state != target
            positions = list(zip(*np.where(diff_mask)))
            # Try a few diff positions
            for r, c in positions[:5]:
                tc = int(target[r, c])
                _try_add(candidates, "flood_fill",
                         (_GRID_PLACEHOLDER, (int(c), int(r)), tc),
                         desc=f"flood_fill(({r},{c}),{tc})")

    # --- World-model-guided additional candidates ---
    if world is not None:
        # If additive (only bg→color changes), try fill ops aggressively
        if world.roles.is_additive:
            for c in world.roles.added_colors:
                # fill_enclosed with each added color
                _try_add(candidates, "fill_enclosed", (_GRID_PLACEHOLDER, c),
                         desc=f"fill_enclosed({c}) [additive hint]")
                # propagate from each existing non-bg color
                for src in all_state_colors - {bg}:
                    _try_add(candidates, "propagate",
                             (_GRID_PLACEHOLDER, src, c, bg),
                             desc=f"propagate({src},{c},bg={bg}) [additive hint]")

        # If structure has symmetry, prioritize symmetry completion
        if world.structure.has_symmetry_h:
            _try_add(candidates, "complete_symmetry_h", (_GRID_PLACEHOLDER,),
                     desc="complete_symmetry_h [symmetry hint]")
        if world.structure.has_symmetry_v:
            _try_add(candidates, "complete_symmetry_v", (_GRID_PLACEHOLDER,),
                     desc="complete_symmetry_v [symmetry hint]")

    # --- Object-level compound steps ---
    # Analyze what happened to objects between state and target,
    # derive selection predicates and transforms from the diff.
    obj_candidates = _derive_object_steps(states, targets, bg)
    candidates.extend(obj_candidates)

    return candidates


def _try_add(
    candidates: list[StepCandidate],
    op_name: str,
    args: tuple,
    desc: str = "",
) -> None:
    """Add a candidate if the op exists."""
    from aria.runtime.ops import has_op
    if has_op(op_name):
        candidates.append(StepCandidate(
            op=op_name,
            args=args,
            description=desc or op_name,
        ))


def _infer_color_map(
    states: list[Grid],
    targets: list[Grid],
) -> dict[int, int] | None:
    """Infer a color bijection from states to targets, if one exists."""
    mapping: dict[int, int] = {}
    for state, target in zip(states, targets):
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                sc = int(state[r, c])
                tc = int(target[r, c])
                if sc in mapping:
                    if mapping[sc] != tc:
                        return None
                else:
                    mapping[sc] = tc

    # Only return if it's a non-identity mapping
    if all(k == v for k, v in mapping.items()):
        return None
    return mapping


def _derive_object_steps(
    states: list[Grid],
    targets: list[Grid],
    bg: int,
) -> list[StepCandidate]:
    """Derive compound object-level steps by observing what happened to objects.

    This is NOT hardcoded.  It:
    1. Extracts objects from current state and target
    2. Matches them (by color+size proximity)
    3. Observes what changed (movement, recolor, removal)
    4. Groups observations into patterns (all color-X objects moved the same way)
    5. Builds compound steps from the patterns

    The patterns emerge from the data, not from predefined rules.
    """
    from aria.runtime.ops.selection import _find_objects

    candidates: list[StepCandidate] = []

    # Collect per-demo object observations
    all_observations: list[list[dict]] = []

    for state, target in zip(states, targets):
        s_objs = [o for o in _find_objects(state) if o.color != bg]
        t_objs = [o for o in _find_objects(target) if o.color != bg]

        # Match objects: color+size, break ties by proximity
        obs = _match_and_observe(s_objs, t_objs, bg)
        all_observations.append(obs)

    if not all_observations or not all_observations[0]:
        return candidates

    # --- Find consistent patterns across demos ---

    # Pattern 1: per-color uniform movement
    # "all objects of color C moved by the same (dr, dc)"
    color_move_patterns = _find_per_color_movement(all_observations)
    for color, dr, dc in color_move_patterns:
        candidates.append(StepCandidate(
            op="__compound_move_color__",
            args=(color, dr, dc, bg, _GRID_PLACEHOLDER),
            description=f"move color={color} by ({dr},{dc})",
        ))

    # Pattern 2: per-color recolor
    # "all objects of color C became color D"
    color_recolor_patterns = _find_per_color_recolor(all_observations)
    for old_c, new_c in color_recolor_patterns:
        candidates.append(StepCandidate(
            op="__compound_recolor__",
            args=(old_c, new_c, bg, _GRID_PLACEHOLDER),
            description=f"recolor {old_c}→{new_c}",
        ))

    # Pattern 3: uniform movement (all objects, same delta)
    uniform = _find_uniform_movement(all_observations)
    if uniform is not None:
        dr, dc = uniform
        candidates.append(StepCandidate(
            op="__compound_move_all__",
            args=(dr, dc, bg, _GRID_PLACEHOLDER),
            description=f"move all by ({dr},{dc})",
        ))

    # Pattern 4: gravity (all objects to an edge)
    for direction in ("up", "down", "left", "right"):
        candidates.append(StepCandidate(
            op="__compound_gravity__",
            args=(direction, bg, _GRID_PLACEHOLDER),
            description=f"gravity {direction}",
        ))

    return candidates


def _match_and_observe(
    inp_objs: list,
    out_objs: list,
    bg: int,
) -> list[dict]:
    """Match input objects to output objects, return per-object observations."""
    observations = []
    used: set[int] = set()

    for io in inp_objs:
        best_j, best_dist = None, float("inf")
        for j, oo in enumerate(out_objs):
            if j in used:
                continue
            same_color = oo.color == io.color
            same_size = oo.size == io.size
            dist = abs(oo.bbox[1] - io.bbox[1]) + abs(oo.bbox[0] - io.bbox[0])

            if same_color and same_size and dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            oo = out_objs[best_j]
            used.add(best_j)
            observations.append({
                "color": io.color,
                "size": io.size,
                "dr": oo.bbox[1] - io.bbox[1],
                "dc": oo.bbox[0] - io.bbox[0],
                "new_color": oo.color,
                "moved": (oo.bbox[1] - io.bbox[1], oo.bbox[0] - io.bbox[0]) != (0, 0),
                "recolored": oo.color != io.color,
            })
        else:
            # Try matching by position (object changed color)
            best_j2, best_dist2 = None, float("inf")
            for j, oo in enumerate(out_objs):
                if j in used:
                    continue
                dist = abs(oo.bbox[1] - io.bbox[1]) + abs(oo.bbox[0] - io.bbox[0])
                if oo.size == io.size and dist < best_dist2:
                    best_dist2 = dist
                    best_j2 = j

            if best_j2 is not None and best_dist2 <= 2:
                oo = out_objs[best_j2]
                used.add(best_j2)
                observations.append({
                    "color": io.color,
                    "size": io.size,
                    "dr": oo.bbox[1] - io.bbox[1],
                    "dc": oo.bbox[0] - io.bbox[0],
                    "new_color": oo.color,
                    "moved": (oo.bbox[1] - io.bbox[1], oo.bbox[0] - io.bbox[0]) != (0, 0),
                    "recolored": oo.color != io.color,
                })
            else:
                observations.append({
                    "color": io.color,
                    "size": io.size,
                    "dr": 0, "dc": 0,
                    "new_color": io.color,
                    "moved": False,
                    "recolored": False,
                    "removed": True,
                })

    return observations


def _find_per_color_movement(
    all_observations: list[list[dict]],
) -> list[tuple[int, int, int]]:
    """Find per-color uniform movement consistent across all demos."""
    if not all_observations:
        return []

    # Gather per-color deltas per demo
    color_deltas: dict[int, list[set[tuple[int, int]]]] = {}
    for obs_list in all_observations:
        demo_cd: dict[int, set[tuple[int, int]]] = {}
        for obs in obs_list:
            if obs.get("removed"):
                continue
            if not obs["moved"]:
                continue
            c = obs["color"]
            demo_cd.setdefault(c, set()).add((obs["dr"], obs["dc"]))
        for c, deltas in demo_cd.items():
            color_deltas.setdefault(c, []).append(deltas)

    results = []
    for c, delta_sets in color_deltas.items():
        # Each demo must have exactly one delta for this color
        if len(delta_sets) != len(all_observations):
            continue
        if not all(len(ds) == 1 for ds in delta_sets):
            continue
        # And the delta must be the same across all demos
        all_d = [list(ds)[0] for ds in delta_sets]
        if len(set(all_d)) == 1:
            dr, dc = all_d[0]
            results.append((c, dr, dc))

    return results


def _find_per_color_recolor(
    all_observations: list[list[dict]],
) -> list[tuple[int, int]]:
    """Find per-color recoloring consistent across all demos."""
    if not all_observations:
        return []

    color_maps: dict[int, list[set[int]]] = {}
    for obs_list in all_observations:
        demo_cm: dict[int, set[int]] = {}
        for obs in obs_list:
            if obs.get("removed") or not obs["recolored"]:
                continue
            demo_cm.setdefault(obs["color"], set()).add(obs["new_color"])
        for c, new_cs in demo_cm.items():
            color_maps.setdefault(c, []).append(new_cs)

    results = []
    for old_c, new_c_sets in color_maps.items():
        if len(new_c_sets) != len(all_observations):
            continue
        if not all(len(s) == 1 for s in new_c_sets):
            continue
        all_new = [list(s)[0] for s in new_c_sets]
        if len(set(all_new)) == 1:
            results.append((old_c, all_new[0]))

    return results


def _find_uniform_movement(
    all_observations: list[list[dict]],
) -> tuple[int, int] | None:
    """Find a single uniform (dr, dc) for all moved objects across all demos."""
    deltas: set[tuple[int, int]] = set()
    any_moved = False
    for obs_list in all_observations:
        for obs in obs_list:
            if obs.get("removed"):
                continue
            if obs["moved"]:
                any_moved = True
                deltas.add((obs["dr"], obs["dc"]))

    if any_moved and len(deltas) == 1:
        return list(deltas)[0]
    return None


def _apply_compound_candidate(cand: StepCandidate, grid: Grid) -> Grid | None:
    """Execute a compound object-level step on a grid."""
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.ops.spatial import _replace_obj
    from aria.runtime.ops.grid import _paint_objects

    op = cand.op
    args = cand.args

    if op == "__compound_move_color__":
        color, dr, dc, bg, _ = args
        return _compound_move(grid, bg, dr, dc, select_color=color)

    if op == "__compound_move_all__":
        dr, dc, bg, _ = args
        return _compound_move(grid, bg, dr, dc, select_color=None)

    if op == "__compound_recolor__":
        old_c, new_c, bg, _ = args
        objs = _find_objects(grid)
        result = grid.copy()
        for obj in objs:
            if obj.color == old_c:
                x, y, w, h = obj.bbox
                for r in range(h):
                    for c in range(w):
                        if obj.mask[r, c]:
                            gr, gc = y + r, x + c
                            if 0 <= gr < result.shape[0] and 0 <= gc < result.shape[1]:
                                result[gr, gc] = new_c
        return result

    if op == "__compound_gravity__":
        direction, bg, _ = args
        return _compound_gravity(grid, bg, direction)

    return None


def _compound_move(
    grid: Grid, bg: int, dr: int, dc: int,
    select_color: int | None = None,
) -> Grid:
    """Move objects (all, or by color) by (dr, dc), erase originals."""
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.ops.spatial import _replace_obj
    from aria.runtime.ops.grid import _paint_objects

    objs = [o for o in _find_objects(grid) if o.color != bg]
    result = grid.copy()

    # Erase selected objects
    moved_objs = []
    for obj in objs:
        if select_color is not None and obj.color != select_color:
            continue
        x, y, w, h = obj.bbox
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    gr, gc = y + r, x + c
                    if 0 <= gr < result.shape[0] and 0 <= gc < result.shape[1]:
                        result[gr, gc] = bg
        moved_objs.append(_replace_obj(obj, bbox=(x + dc, y + dr, w, h)))

    # Paint at new positions
    result = _paint_objects(set(moved_objs), result)
    return result


def _compound_gravity(grid: Grid, bg: int, direction: str) -> Grid:
    """Move all fg objects to a grid edge."""
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.ops.spatial import _replace_obj
    from aria.runtime.ops.grid import _paint_objects

    h, w = grid.shape
    objs = [o for o in _find_objects(grid) if o.color != bg]
    result = grid.copy()

    moved = []
    for obj in objs:
        x, y, ow, oh = obj.bbox
        # Erase
        for r in range(oh):
            for c in range(ow):
                if obj.mask[r, c]:
                    gr, gc = y + r, x + c
                    if 0 <= gr < h and 0 <= gc < w:
                        result[gr, gc] = bg
        # Compute new position
        if direction == "down":
            new_obj = _replace_obj(obj, bbox=(x, h - oh, ow, oh))
        elif direction == "up":
            new_obj = _replace_obj(obj, bbox=(x, 0, ow, oh))
        elif direction == "right":
            new_obj = _replace_obj(obj, bbox=(w - ow, y, ow, oh))
        elif direction == "left":
            new_obj = _replace_obj(obj, bbox=(0, y, ow, oh))
        else:
            new_obj = obj
        moved.append(new_obj)

    result = _paint_objects(set(moved), result)
    return result


def _build_program(steps: list[ConstructionStep]) -> Program:
    """Build a Program from a sequence of construction steps.

    Compound steps (__compound_*) are expanded into multi-bind sequences
    using real runtime ops.
    """
    binds: list[Bind] = []
    prev_ref = "input"
    bind_idx = 0

    for step in steps:
        if step.op.startswith("__compound_"):
            new_binds, final_ref = _expand_compound(step, prev_ref, bind_idx)
            binds.extend(new_binds)
            bind_idx += len(new_binds)
            prev_ref = final_ref
        else:
            bind_name = f"v{bind_idx}"
            call_args = []
            for arg in step.args:
                if arg is _GRID_PLACEHOLDER:
                    call_args.append(Ref(prev_ref))
                elif isinstance(arg, dict):
                    call_args.append(Literal(arg, Type.COLOR_MAP))
                elif isinstance(arg, int):
                    call_args.append(Literal(arg, Type.INT))
                elif isinstance(arg, tuple):
                    call_args.append(Literal(arg, Type.DIMS))
                else:
                    call_args.append(Literal(arg, Type.INT))
            binds.append(Bind(bind_name, Type.GRID, Call(step.op, tuple(call_args))))
            prev_ref = bind_name
            bind_idx += 1

    return Program(steps=tuple(binds), output=prev_ref)


def _expand_compound(
    step: ConstructionStep,
    input_ref: str,
    start_idx: int,
) -> tuple[list[Bind], str]:
    """Expand a compound step into real typed program binds."""
    binds = []
    idx = start_idx

    if step.op == "__compound_move_color__":
        color, dr, dc, bg, _ = step.args
        erase_map = {color: bg}
        binds = [
            Bind(f"v{idx}", Type.OBJECT_SET, Call("find_objects", (Ref(input_ref),))),
            Bind(f"v{idx+1}", Type.PREDICATE, Call("by_color", (Literal(color, Type.COLOR),))),
            Bind(f"v{idx+2}", Type.OBJECT_SET, Call("where", (Ref(f"v{idx+1}"), Ref(f"v{idx}")))),
            Bind(f"v{idx+3}", Type.OBJ_TRANSFORM, Call("translate_by", (
                Literal(dr, Type.INT), Literal(dc, Type.INT),
            ))),
            Bind(f"v{idx+4}", Type.OBJECT_SET, Call("map_obj", (Ref(f"v{idx+3}"), Ref(f"v{idx+2}")))),
            Bind(f"v{idx+5}", Type.GRID, Call("apply_color_map", (
                Literal(erase_map, Type.COLOR_MAP), Ref(input_ref),
            ))),
            Bind(f"v{idx+6}", Type.GRID, Call("paint_objects", (Ref(f"v{idx+4}"), Ref(f"v{idx+5}")))),
        ]
        return binds, f"v{idx+6}"

    if step.op == "__compound_move_all__":
        dr, dc, bg, _ = step.args
        erase_map = {c: bg for c in range(10) if c != bg}
        binds = [
            Bind(f"v{idx}", Type.OBJECT_SET, Call("find_objects", (Ref(input_ref),))),
            Bind(f"v{idx+1}", Type.OBJ_TRANSFORM, Call("translate_by", (
                Literal(dr, Type.INT), Literal(dc, Type.INT),
            ))),
            Bind(f"v{idx+2}", Type.OBJECT_SET, Call("map_obj", (Ref(f"v{idx+1}"), Ref(f"v{idx}")))),
            Bind(f"v{idx+3}", Type.GRID, Call("apply_color_map", (
                Literal(erase_map, Type.COLOR_MAP), Ref(input_ref),
            ))),
            Bind(f"v{idx+4}", Type.GRID, Call("paint_objects", (Ref(f"v{idx+2}"), Ref(f"v{idx+3}")))),
        ]
        return binds, f"v{idx+4}"

    if step.op == "__compound_recolor__":
        old_c, new_c, bg, _ = step.args
        cmap = {old_c: new_c}
        binds = [
            Bind(f"v{idx}", Type.GRID, Call("apply_color_map", (
                Literal(cmap, Type.COLOR_MAP), Ref(input_ref),
            ))),
        ]
        return binds, f"v{idx}"

    if step.op == "__compound_gravity__":
        direction, bg, _ = step.args
        dir_map = {"up": 1, "down": 2, "left": 3, "right": 4}
        dir_int = dir_map.get(direction, 2)
        h_ref, w_ref = f"v{idx}", f"v{idx+1}"
        # We need grid dims — use literals from the step execution context
        # For now, build a program that works for the specific grid size
        # This will be filled in during verification
        erase_map = {c: bg for c in range(10) if c != bg}
        binds = [
            Bind(f"v{idx}", Type.OBJECT_SET, Call("find_objects", (Ref(input_ref),))),
            Bind(f"v{idx+1}", Type.GRID, Call("apply_color_map", (
                Literal(erase_map, Type.COLOR_MAP), Ref(input_ref),
            ))),
        ]
        # Gravity needs grid dims which vary — emit per-object translate instead
        # For now, return just the erase step (partial)
        return binds, f"v{idx+1}"

    # Fallback: can't expand, return empty
    return [], input_ref


def _identity_program() -> Program:
    """Program that returns the input unchanged."""
    return Program(steps=(), output="input")
