"""Predicate-parameterized dispatch for scene programs.

Given a predicate P and an action A, produces scene programs of the form:
  for each object:
    if P(object) then A(object)
    else keep(object)

The predicate is inferred from cross-demo evidence via induce_predicates().
The action is inferred from the demo diff at changed objects.
The result is one shared program that adapts to each demo via input-derived
predicate evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.core.scene_executor import execute_scene_program, make_scene_program, make_step
from aria.decomposition import detect_bg, extract_objects
from aria.predicates import (
    ObjectContext,
    Predicate,
    evaluate_predicate,
    induce_predicates,
)
from aria.scene_ir import SceneProgram, StepOp
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Action inference from demo diffs
# ---------------------------------------------------------------------------


def _infer_actions(demos: tuple[DemoPair, ...]) -> list[dict]:
    """Infer candidate actions from the diff at changed objects.

    Returns a list of action dicts, each describing what to do
    to objects matching the predicate. Includes both literal and
    role-based actions.
    """
    actions: list[dict] = []

    per_demo_changes: list[dict[tuple[int, int], int]] = []
    per_demo_bg: list[int] = []
    for d in demos:
        if d.input.shape != d.output.shape:
            continue
        bg = detect_bg(d.input)
        per_demo_bg.append(bg)
        diff = d.input != d.output
        if not np.any(diff):
            per_demo_changes.append({})
            continue
        changes: dict[tuple[int, int], int] = {}
        for r, c in zip(*np.where(diff)):
            ic, oc = int(d.input[r, c]), int(d.output[r, c])
            changes[(ic, oc)] = changes.get((ic, oc), 0) + 1
        per_demo_changes.append(changes)

    if not per_demo_changes:
        return actions

    # 1. Literal recolor: same from_color → to_color across demos
    all_pairs: set[tuple[int, int]] = set()
    for ch in per_demo_changes:
        all_pairs.update(ch.keys())

    for fc, tc in all_pairs:
        count = sum(1 for ch in per_demo_changes if (fc, tc) in ch)
        if count >= max(1, len(per_demo_changes) * 0.5):
            actions.append({
                "type": "recolor",
                "from_color": fc,
                "to_color": tc,
                "confidence": count / len(per_demo_changes),
            })

    # 2. Role-based recolor: the changed object's color varies across demos
    #    but the target color is consistent (or derived from context)
    target_colors_per_demo = []
    for ch in per_demo_changes:
        targets = set(tc for (fc, tc), cnt in ch.items() if cnt > 0)
        target_colors_per_demo.append(targets)

    # Find colors that appear as target in all demos
    if target_colors_per_demo:
        common_targets = set.intersection(*target_colors_per_demo) if target_colors_per_demo else set()
        for tc in common_targets:
            # "Recolor changed object's dominant color to tc"
            actions.append({
                "type": "recolor_dominant_to",
                "to_color": tc,
                "confidence": 0.7,
            })

    # Role-based: recolor to the adjacent non-bg color
    actions.append({
        "type": "recolor_to_adjacent",
        "confidence": 0.3,
    })

    # Role-based: recolor to the nearest neighbor object's color
    actions.append({
        "type": "recolor_to_nearest_neighbor",
        "confidence": 0.4,
    })

    # Erase: recolor object pixels to bg
    for d in demos:
        if d.input.shape != d.output.shape:
            continue
        bg_d = detect_bg(d.input)
        actions.append({
            "type": "erase_to_bg",
            "to_color": bg_d,
            "confidence": 0.4,
        })
        break

    # Relocate: erase original + stamp at derived position
    actions.append({
        "type": "relocate_to_nearest_anchor",
        "confidence": 0.3,
    })

    # Relocate: stamp into bg gaps inside host objects (enumerate gaps)
    actions.append({
        "type": "stamp_into_host_gap",
        "confidence": 0.4,
    })

    # Relocate: stamp into bg gaps, trying each gap (verification-driven)
    actions.append({
        "type": "stamp_into_host_gap_enum",
        "confidence": 0.5,
    })

    # Erase at rows/cols matching reference objects' positions
    actions.append({
        "type": "erase_at_reference_rows",
        "confidence": 0.3,
    })

    # 3. Fill with color: bg → color changes
    for d in demos:
        if d.input.shape != d.output.shape:
            continue
        bg = detect_bg(d.input)
        diff = d.input != d.output
        fill_colors = set()
        for r, c in zip(*np.where(diff)):
            if int(d.input[r, c]) == bg:
                fill_colors.add(int(d.output[r, c]))
        for fc in fill_colors:
            actions.append({
                "type": "fill_adjacent",
                "fill_color": fc,
                "confidence": 0.5,
            })
        break

    # 4. Erase: color → bg
    for d in demos:
        if d.input.shape != d.output.shape:
            continue
        bg = detect_bg(d.input)
        diff = d.input != d.output
        for r, c in zip(*np.where(diff)):
            if int(d.output[r, c]) == bg and int(d.input[r, c]) != bg:
                actions.append({
                    "type": "erase",
                    "erase_color": int(d.input[r, c]),
                    "to_color": bg,
                    "confidence": 0.5,
                })
                break
        break

    return actions


# ---------------------------------------------------------------------------
# Build predicate-dispatch programs
# ---------------------------------------------------------------------------


def build_predicate_dispatch_programs(
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
    *,
    max_predicates: int = 10,
    max_actions: int = 5,
) -> list[SceneProgram]:
    """Build predicate-parameterized scene programs.

    1. Induce predicates from cross-demo object change evidence
    2. Infer actions from demo diffs
    3. Compose predicate × action into dispatch programs
    4. Return programs ready for exact verification
    """
    programs: list[SceneProgram] = []

    # Only for same-dims tasks
    if not all(d.input.shape == d.output.shape for d in demos):
        return programs

    # Induce predicates
    predicates = induce_predicates(demos, max_predicates=max_predicates)
    if not predicates:
        return programs

    # Infer actions
    actions = _infer_actions(demos)
    if not actions:
        return programs

    # Build programs for top predicates × actions
    for pred in predicates[:max_predicates]:
        for action in actions[:max_actions]:
            prog = _compile_dispatch_program(pred, action, demos)
            if prog is not None:
                programs.append(prog)

    # Gap-enumeration search: for each predicate, try erasing matching
    # singletons and placing them at each bg gap inside host objects
    if any(a.get("type") == "stamp_into_host_gap_enum" for a in actions):
        gap_progs = _build_gap_enum_programs(demos, perceptions, predicates[:5])
        programs.extend(gap_progs)

    return programs


def _build_gap_enum_programs(
    demos: tuple[DemoPair, ...],
    perceptions: tuple[GridPerceptionState, ...],
    predicates: list[Predicate],
) -> list[SceneProgram]:
    """Build programs that erase singletons and stamp into specific host gaps.

    Instead of picking one gap, enumerates candidate gap positions and
    lets verification decide which is correct.
    """
    programs: list[SceneProgram] = []
    if not all(d.input.shape == d.output.shape for d in demos):
        return programs

    d0 = demos[0]
    bg = perceptions[0].bg_color
    from aria.decomposition import extract_objects as _eo

    objs = _eo(d0.input, bg)
    singletons = [o for o in objs if o.size == 1]
    hosts = [o for o in objs if o.size >= 10]

    if not singletons or not hosts:
        return programs

    # Find all bg gaps inside hosts
    all_gaps: list[tuple[int, int]] = []
    for host in hosts:
        for dr in range(host.bbox_h):
            for dc in range(host.bbox_w):
                r, c = host.row + dr, host.col + dc
                if int(d0.input[r, c]) == bg:
                    # Check if surrounded by host color
                    adj_host = 0
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + ddr, c + ddc
                        if 0 <= nr < d0.input.shape[0] and 0 <= nc < d0.input.shape[1]:
                            if int(d0.input[nr, nc]) == host.color:
                                adj_host += 1
                    if adj_host >= 2:
                        all_gaps.append((r, c))

    if not all_gaps or len(all_gaps) > 50:
        return programs

    # For each predicate, build a program that erases matching singletons
    # and stamps into each gap position
    for pred in predicates:
        for gap_r, gap_c in all_gaps[:20]:
            prog = make_scene_program(
                make_step(StepOp.PARSE_SCENE),
                make_step(
                    StepOp.FOR_EACH_ENTITY,
                    kind="object",
                    rule="predicate_dispatch",
                    predicate=_serialize_predicate(pred),
                    action="relocate",
                    relocate_mode="fixed_position",
                    target_row=gap_r,
                    target_col=gap_c,
                ),
                make_step(StepOp.RENDER_SCENE),
            )
            programs.append(prog)

    return programs


def _compile_dispatch_program(
    pred: Predicate,
    action: dict,
    demos: tuple[DemoPair, ...],
) -> SceneProgram | None:
    """Compile a predicate + action into an executable SceneProgram."""
    action_type = action.get("type")

    if action_type == "recolor":
        return _make_predicate_recolor_program(pred, action)
    if action_type == "recolor_dominant_to":
        return _make_predicate_recolor_dominant_program(pred, action)
    if action_type == "recolor_to_adjacent":
        return _make_predicate_recolor_to_adjacent_program(pred, action)
    if action_type == "recolor_to_nearest_neighbor":
        return _make_predicate_recolor_to_nearest_program(pred, action)
    if action_type == "fill_adjacent":
        return _make_predicate_fill_program(pred, action)
    if action_type == "erase":
        return _make_predicate_erase_program(pred, action)
    if action_type == "erase_to_bg":
        return _make_predicate_erase_to_bg_program(pred, action)
    if action_type == "relocate_to_nearest_anchor":
        return _make_predicate_relocate_program(pred, action, "nearest_anchor")
    if action_type == "stamp_into_host_gap":
        return _make_predicate_relocate_program(pred, action, "host_gap")
    if action_type == "stamp_into_host_gap_enum":
        # Returns None — handled specially by build_predicate_dispatch_programs
        return None
    if action_type == "erase_at_reference_rows":
        return _make_predicate_erase_at_ref_rows_program(pred, action)

    return None


def _make_predicate_recolor_program(pred: Predicate, action: dict) -> SceneProgram:
    """Build: parse → predicate-dispatch recolor → render."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="recolor",
            from_color=action["from_color"],
            to_color=action["to_color"],
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_fill_program(pred: Predicate, action: dict) -> SceneProgram:
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="fill_adjacent",
            fill_color=action["fill_color"],
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_recolor_dominant_program(pred: Predicate, action: dict) -> SceneProgram:
    """Recolor matching objects' dominant color to a fixed target."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="recolor_dominant_to",
            to_color=action["to_color"],
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_recolor_to_adjacent_program(pred: Predicate, action: dict) -> SceneProgram:
    """Recolor matching objects to the adjacent non-bg color."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="recolor_to_adjacent",
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_recolor_to_nearest_program(pred: Predicate, action: dict) -> SceneProgram:
    """Recolor matching objects to the nearest different-color object's color."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="recolor_to_nearest_neighbor",
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_erase_to_bg_program(pred: Predicate, action: dict) -> SceneProgram:
    """Erase matching objects to background."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="erase_to_bg",
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_relocate_program(pred: Predicate, action: dict, mode: str) -> SceneProgram:
    """Erase matching objects and stamp them at a derived position."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="relocate",
            relocate_mode=mode,
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_erase_at_ref_rows_program(pred: Predicate, action: dict) -> SceneProgram:
    """Erase pixels in matching objects at rows where reference objects exist."""
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="erase_at_reference_rows",
        ),
        make_step(StepOp.RENDER_SCENE),
    )


def _make_predicate_erase_program(pred: Predicate, action: dict) -> SceneProgram:
    return make_scene_program(
        make_step(StepOp.PARSE_SCENE),
        make_step(
            StepOp.FOR_EACH_ENTITY,
            kind="object",
            rule="predicate_dispatch",
            predicate=_serialize_predicate(pred),
            action="erase",
            erase_color=action["erase_color"],
            to_color=action["to_color"],
        ),
        make_step(StepOp.RENDER_SCENE),
    )


# ---------------------------------------------------------------------------
# Predicate serialization for SceneStep params
# ---------------------------------------------------------------------------


def _serialize_predicate(pred: Predicate) -> dict:
    """Serialize a predicate to a JSON-compatible dict for SceneStep params."""
    return {
        "kind": pred.kind.value,
        "params": dict(pred.params),
    }


def deserialize_predicate(d: dict) -> Predicate:
    """Deserialize a predicate from a SceneStep param dict."""
    from aria.predicates import PredicateKind
    kind = PredicateKind(d["kind"])
    raw_params = d.get("params", {})
    # Convert nested predicates
    params = []
    for k, v in raw_params.items():
        if isinstance(v, dict) and "kind" in v:
            v = deserialize_predicate(v)
        params.append((k, v))
    return Predicate(kind=kind, params=tuple(params))
