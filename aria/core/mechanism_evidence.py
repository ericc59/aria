"""Mechanism evidence layer — structural features for lane selection.

Computes inspectable evidence from task demos to rank candidate
compilation lanes before trying them. Evidence is purely structural
(object counts, singleton patterns, periodicity cues, etc.) and does
not depend on task IDs or family labels.

Part of the canonical architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MechanismEvidence:
    """Inspectable structural evidence for lane selection.

    Each field is a named, typed feature computed from the demos.
    """
    # Object counts
    n_input_shapes: int = 0         # multi-pixel non-bg objects in input
    n_output_shapes: int = 0        # multi-pixel non-bg objects in output
    shape_count_change: int = 0     # output - input (positive = growth)
    n_input_singles: int = 0        # single-pixel non-bg objects in input
    n_output_singles: int = 0       # single-pixel non-bg in output
    single_count_change: int = 0

    # Replication cues
    has_anchored_exemplars: bool = False  # shapes with adjacent diff-color singletons
    n_anchored_exemplars: int = 0
    replication_target_count: int = 0     # singletons sharing anchor key colors
    output_grows_shapes: bool = False     # more shapes in output than input

    # Relocation cues
    shapes_shift_position: bool = False   # shapes move between input and output
    one_to_one_shape_count: bool = False  # same number of shapes in/out

    # Periodic repair cues
    has_framed_region: bool = False
    has_periodic_pattern: bool = False
    periodic_axis: str = ""
    periodic_period: int = 0

    # Grid transform cues
    same_dims: bool = False
    same_shape_set: bool = False    # same shapes, possibly at different positions

    # Confidence scores per lane (0-1)
    replication_score: float = 0.0
    relocation_score: float = 0.0
    periodic_repair_score: float = 0.0
    grid_transform_score: float = 0.0


@dataclass(frozen=True)
class LaneRanking:
    """Ranked list of candidate lanes with evidence rationale."""
    lanes: tuple[LaneCandidate, ...]
    evidence: MechanismEvidence
    description: str = ""


@dataclass(frozen=True)
class LaneCandidate:
    """One candidate lane in the ranking."""
    name: str
    class_score: float        # broad structural fit (0-1)
    exec_hint: float          # sufficiency hint for current executable (0-1)
    anti_evidence: str        # contradiction signals (empty = none)
    final_score: float        # combined score used for ranking
    rationale: str
    gate_pass: bool

    @property
    def score(self) -> float:
        """Backward compat alias."""
        return self.final_score


def compute_evidence_and_rank(demos: Sequence[Any]) -> tuple[MechanismEvidence, LaneRanking]:
    """Compute evidence and produce a ranked lane selection in one call."""
    evidence, extras = _compute_evidence_raw(demos)
    ranking = rank_lanes(evidence, **extras)
    return evidence, ranking


def compute_evidence(demos: Sequence[Any]) -> MechanismEvidence:
    """Compute mechanism evidence only (backward compat)."""
    evidence, _ = _compute_evidence_raw(demos)
    return evidence


def _compute_evidence_raw(demos: Sequence[Any]) -> tuple[MechanismEvidence, dict]:
    """Compute evidence + sufficiency data for rank_lanes.

    All evidence is structural and inspectable. No task-ID checks.
    """
    from aria.decomposition import detect_bg, detect_framed_regions
    from aria.runtime.ops.selection import _find_objects
    from aria.runtime.ops.replicate import _find_anchor, KEY_ADJACENT_DIFF_COLOR

    n_demos = len(demos)
    if n_demos == 0:
        return MechanismEvidence(), {"repl_match_ratio": 0.0, "frame_consistent": False, "n_demos": 0}

    total_in_shapes = 0
    total_out_shapes = 0
    total_in_singles = 0
    total_out_singles = 0
    total_anchored = 0
    total_repl_targets = 0
    total_repl_output_matches = 0  # clones found at predicted positions
    has_framed = False
    demos_with_frame = 0           # how many demos have a framed region
    has_periodic = False
    p_axis = ""
    p_period = 0
    same_dims = True
    shapes_shift = False
    same_shape_set_all = True

    for demo in demos:
        bg = detect_bg(demo.input)
        in_objs = list(_find_objects(demo.input))
        out_objs = list(_find_objects(demo.output))
        in_shapes = [o for o in in_objs if o.color != bg and o.size > 1]
        out_shapes = [o for o in out_objs if o.color != bg and o.size > 1]
        in_singles = [o for o in in_objs if o.color != bg and o.size == 1]
        out_singles = [o for o in out_objs if o.color != bg and o.size == 1]

        total_in_shapes += len(in_shapes)
        total_out_shapes += len(out_shapes)
        total_in_singles += len(in_singles)
        total_out_singles += len(out_singles)

        if demo.input.shape != demo.output.shape:
            same_dims = False

        # Replication: anchored exemplars + output match check
        for shape in in_shapes:
            anchor, offset = _find_anchor(shape, in_singles, KEY_ADJACENT_DIFF_COLOR)
            if anchor is not None:
                key_color = anchor.color
                targets = [m for m in in_singles if m.color == key_color and m is not anchor]
                if targets:
                    total_anchored += 1
                    total_repl_targets += len(targets)
                    # Sufficiency check: do output shapes appear at predicted clone positions?
                    for t in targets:
                        tx, ty = t.bbox[0], t.bbox[1]
                        clone_r = ty - offset[0]
                        clone_c = tx - offset[1]
                        if any(so.color == shape.color and so.size == shape.size
                               and so.bbox[0] == clone_c and so.bbox[1] == clone_r
                               for so in out_shapes):
                            total_repl_output_matches += 1

        # Check if shapes shift position
        in_positions = {(o.bbox[0], o.bbox[1]) for o in in_shapes}
        out_positions = {(o.bbox[0], o.bbox[1]) for o in out_shapes}
        if in_positions != out_positions and in_shapes:
            shapes_shift = True

        # Check same shape set (same sizes/colors, possibly different positions)
        in_sig = sorted((o.color, o.size) for o in in_shapes)
        out_sig = sorted((o.color, o.size) for o in out_shapes)
        if in_sig != out_sig:
            same_shape_set_all = False

        # Framed regions — track per-demo consistency
        try:
            regions = detect_framed_regions(demo.input)
            if regions:
                has_framed = True
                demos_with_frame += 1
        except Exception:
            pass

    # Average across demos
    avg_in_shapes = total_in_shapes / max(n_demos, 1)
    avg_out_shapes = total_out_shapes / max(n_demos, 1)
    shape_change = total_out_shapes - total_in_shapes

    output_grows = shape_change > 0
    one_to_one = abs(shape_change) == 0 and total_in_shapes > 0

    # Frame consistency across demos
    frame_consistent = demos_with_frame == n_demos if n_demos > 0 else False

    # Replication output match ratio: how many predicted clones actually appear
    repl_match_ratio = (total_repl_output_matches / max(total_repl_targets, 1)
                        if total_repl_targets > 0 else 0.0)

    # --- Class scores (broad structural fit) ---
    repl_score = 0.0
    if total_anchored > 0 and total_repl_targets > 0:
        repl_score = 0.7
        if output_grows:
            repl_score += 0.2
        if total_repl_targets > total_anchored:
            repl_score += 0.1

    reloc_score = 0.0
    if shapes_shift and one_to_one:
        reloc_score = 0.7
    elif shapes_shift:
        reloc_score = 0.4
    elif one_to_one and total_in_singles > 0:
        reloc_score = 0.3
    if same_shape_set_all and reloc_score > 0:
        reloc_score += 0.1
    if total_anchored > 0 and total_repl_targets > 0:
        reloc_score *= 0.5

    periodic_score = 0.0
    if has_framed:
        periodic_score = 0.6
        if same_dims and not output_grows:
            periodic_score = 0.8

    transform_score = 0.0
    if same_dims and not has_framed and total_in_singles == 0:
        transform_score = 0.5
    elif same_dims:
        transform_score = 0.3

    ev = MechanismEvidence(
        n_input_shapes=total_in_shapes,
        n_output_shapes=total_out_shapes,
        shape_count_change=shape_change,
        n_input_singles=total_in_singles,
        n_output_singles=total_out_singles,
        single_count_change=total_out_singles - total_in_singles,
        has_anchored_exemplars=total_anchored > 0,
        n_anchored_exemplars=total_anchored,
        replication_target_count=total_repl_targets,
        output_grows_shapes=output_grows,
        shapes_shift_position=shapes_shift,
        one_to_one_shape_count=one_to_one,
        has_framed_region=has_framed,
        has_periodic_pattern=has_periodic,
        periodic_axis=p_axis,
        periodic_period=p_period,
        same_dims=same_dims,
        same_shape_set=same_shape_set_all,
        replication_score=min(repl_score, 1.0),
        relocation_score=min(reloc_score, 1.0),
        periodic_repair_score=min(periodic_score, 1.0),
        grid_transform_score=min(transform_score, 1.0),
    )
    extras = {
        "repl_match_ratio": repl_match_ratio,
        "frame_consistent": frame_consistent,
        "n_demos": n_demos,
    }
    return ev, extras


def rank_lanes(
    evidence: MechanismEvidence,
    repl_match_ratio: float = 0.0,
    frame_consistent: bool = False,
    n_demos: int = 0,
) -> LaneRanking:
    """Rank candidate lanes with class_fit, exec_hint, and anti_evidence.

    Three levels per lane:
      class_score:  broad structural resemblance
      exec_hint:    sufficiency hint for current executable
      anti_evidence: contradiction signals that demote the lane
    """
    # --- Replication ---
    repl_class = evidence.replication_score
    repl_exec = 0.0
    repl_anti = ""
    if repl_class > 0:
        # Hard anti-evidence: output must grow in shape count for replication
        if not evidence.output_grows_shapes:
            repl_anti = "output does not grow in shape count (not replication)"
            repl_exec = 0.0
        elif repl_match_ratio > 0.5:
            repl_exec = 0.8
        elif repl_match_ratio > 0:
            repl_exec = 0.4
        else:
            repl_anti = "no output clones at predicted anchor-offset positions"
            repl_exec = 0.1

    repl_final = repl_class * (0.4 + 0.6 * repl_exec) if repl_class > 0 else 0.0
    if repl_anti:
        repl_final *= 0.3

    # --- Relocation ---
    reloc_class = evidence.relocation_score
    reloc_exec = 0.5 if reloc_class > 0 else 0.0
    reloc_anti = ""
    if evidence.output_grows_shapes:
        reloc_anti = "output has more shapes than input (suggests replication, not relocation)"
        reloc_exec *= 0.3
    if not evidence.same_dims:
        reloc_anti += ("; " if reloc_anti else "") + "different grid dims"
        reloc_exec *= 0.5

    reloc_final = reloc_class * (0.4 + 0.6 * reloc_exec) if reloc_class > 0 else 0.0
    if reloc_anti:
        reloc_final *= 0.5

    # --- Periodic repair ---
    per_class = evidence.periodic_repair_score
    per_exec = 0.0
    per_anti = ""
    if per_class > 0:
        if frame_consistent and n_demos > 1:
            per_exec = 0.7  # frame present in all demos
        elif evidence.has_framed_region:
            per_exec = 0.4
            if n_demos > 1:
                per_anti = "frame not detected in every demo"

    per_final = per_class * (0.4 + 0.6 * per_exec) if per_class > 0 else 0.0
    if per_anti:
        per_final *= 0.7  # mild demotion for soft warnings

    # --- Grid transform ---
    trans_class = evidence.grid_transform_score
    trans_exec = 0.5 if trans_class > 0 else 0.0
    trans_anti = ""
    if evidence.has_framed_region:
        trans_anti = "framed region present (suggests periodic, not plain transform)"
        trans_exec *= 0.3
    trans_final = trans_class * (0.4 + 0.6 * trans_exec) if trans_class > 0 else 0.0

    candidates = [
        LaneCandidate(
            name="replication",
            class_score=repl_class,
            exec_hint=round(repl_exec, 3),
            anti_evidence=repl_anti,
            final_score=round(repl_final, 3),
            rationale=_repl_rationale(evidence),
            gate_pass=(evidence.has_anchored_exemplars
                      and evidence.replication_target_count > 0
                      and evidence.output_grows_shapes),
        ),
        LaneCandidate(
            name="relocation",
            class_score=reloc_class,
            exec_hint=round(reloc_exec, 3),
            anti_evidence=reloc_anti,
            final_score=round(reloc_final, 3),
            rationale=_reloc_rationale(evidence),
            gate_pass=evidence.n_input_shapes > 0 and evidence.n_input_singles > 0,
        ),
        LaneCandidate(
            name="periodic_repair",
            class_score=per_class,
            exec_hint=round(per_exec, 3),
            anti_evidence=per_anti,
            final_score=round(per_final, 3),
            rationale=_periodic_rationale(evidence),
            gate_pass=evidence.has_framed_region,
        ),
        LaneCandidate(
            name="grid_transform",
            class_score=trans_class,
            exec_hint=round(trans_exec, 3),
            anti_evidence=trans_anti,
            final_score=round(trans_final, 3),
            rationale=_transform_rationale(evidence),
            gate_pass=evidence.same_dims,
        ),
    ]

    candidates.sort(key=lambda c: (-int(c.gate_pass), -c.final_score))

    top = candidates[0] if candidates else None
    desc = (f"top hypothesis={top.name}({top.final_score:.2f})"
            if top else "no candidates")

    return LaneRanking(lanes=tuple(candidates), evidence=evidence, description=desc)


def _repl_rationale(ev: MechanismEvidence) -> str:
    parts = []
    if ev.has_anchored_exemplars:
        parts.append(f"{ev.n_anchored_exemplars} anchored exemplars")
    if ev.replication_target_count > 0:
        parts.append(f"{ev.replication_target_count} targets")
    if ev.output_grows_shapes:
        parts.append(f"shapes grow ({ev.shape_count_change:+d})")
    return "; ".join(parts) if parts else "no replication cues"


def _reloc_rationale(ev: MechanismEvidence) -> str:
    parts = []
    if ev.shapes_shift_position:
        parts.append("shapes shift")
    if ev.one_to_one_shape_count:
        parts.append("1:1 shape count")
    if ev.same_shape_set:
        parts.append("same shape set")
    return "; ".join(parts) if parts else "no relocation cues"


def _periodic_rationale(ev: MechanismEvidence) -> str:
    parts = []
    if ev.has_framed_region:
        parts.append("framed region")
    if ev.same_dims:
        parts.append("same dims")
    return "; ".join(parts) if parts else "no periodic cues"


def _transform_rationale(ev: MechanismEvidence) -> str:
    parts = []
    if ev.same_dims:
        parts.append("same dims")
    return "; ".join(parts) if parts else "no transform cues"
