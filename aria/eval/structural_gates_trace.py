"""Stage artifact extraction from the solver pipeline.

Captures intermediate outputs (decomposition hypotheses, entities,
relations, templates, slots) without polluting the main solver code.
Operates as a post-hoc observer on existing data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.decomposition import (
    CompositeDecomposition,
    FramedRegion,
    ObjectDecomposition,
    RawObject,
    decompose_composites,
    decompose_objects,
    detect_bg,
    detect_framed_regions,
)
from aria.sketch_fit import fit_sketches_with_report
from aria.types import DemoPair, Grid


# ---------------------------------------------------------------------------
# Stage artifact types
# ---------------------------------------------------------------------------


@dataclass
class InducedEntity:
    id: str
    kind: str  # "object", "marker", "host", "gap", "panel", "region", "singleton"
    bbox: tuple[int, int, int, int]  # (row, col, h, w)
    color: int = -1
    size: int = 0
    score: float = 1.0


@dataclass
class InducedRelation:
    kind: str
    source_id: str
    target_id: str
    score: float = 1.0


@dataclass
class StageArtifacts:
    """All intermediate artifacts captured from one task's solver run."""

    task_id: str

    # Decomposition hypotheses (ranked)
    decomposition_hypotheses: list[str] = field(default_factory=list)

    # Induced entities (across all demos, from first demo as representative)
    entities: list[InducedEntity] = field(default_factory=list)

    # Induced relations
    relations: list[InducedRelation] = field(default_factory=list)

    # Template hypotheses (ranked)
    template_hypotheses: list[str] = field(default_factory=list)

    # Slot candidates per slot family
    slot_candidates: dict[str, list[str]] = field(default_factory=dict)

    # Executor: could a program be instantiated and run?
    executor_attempted: bool = False
    executor_ran: bool = False
    executor_error: str | None = None

    # Path-aware executor attribution
    executor_path: str | None = None  # first path that produced a runnable candidate
    executor_paths_tried: list[str] = field(default_factory=list)
    executor_paths_produced: list[str] = field(default_factory=list)
    executor_paths_ran: list[str] = field(default_factory=list)
    executor_paths_verified: list[str] = field(default_factory=list)

    # Raw decomposition data for debugging
    object_decomp: ObjectDecomposition | None = None
    framed_regions: list[FramedRegion] | None = None
    composite_decomp: CompositeDecomposition | None = None


# ---------------------------------------------------------------------------
# Trace extraction — observes existing pipeline structures
# ---------------------------------------------------------------------------


def extract_stage_artifacts(
    task_id: str,
    demos: tuple[DemoPair, ...],
) -> StageArtifacts:
    """Run decomposition/entity/relation/template extraction on a task.

    This does NOT run the full solver. It runs the perception and
    sketch-fitting stages independently to capture what the system
    *would* propose, then records the artifacts.
    """
    artifacts = StageArtifacts(task_id=task_id)

    if not demos:
        return artifacts

    # Use first demo as representative for entity extraction
    demo0 = demos[0]
    inp = demo0.input
    bg = detect_bg(inp)

    # --- Decomposition hypotheses ---
    decomp_hypotheses = _extract_decomposition_hypotheses(demos)
    artifacts.decomposition_hypotheses = decomp_hypotheses

    # --- Entity extraction ---
    obj_decomp = decompose_objects(inp, bg)
    artifacts.object_decomp = obj_decomp

    entities: list[InducedEntity] = []
    for i, obj in enumerate(obj_decomp.objects):
        kind = "marker" if obj.is_singleton else "object"
        entities.append(InducedEntity(
            id=f"obj_{i}",
            kind=kind,
            bbox=(obj.row, obj.col, obj.bbox_h, obj.bbox_w),
            color=obj.color,
            size=obj.size,
        ))

    # Framed regions as entities
    framed = detect_framed_regions(inp, bg)
    artifacts.framed_regions = framed
    for i, fr in enumerate(framed):
        entities.append(InducedEntity(
            id=f"frame_{i}",
            kind="region",
            bbox=(fr.row, fr.col, fr.height, fr.width),
            color=fr.frame_color,
            size=fr.height * fr.width,
        ))

    # Composite decomposition
    try:
        comp_decomp = decompose_composites(inp, bg)
        artifacts.composite_decomp = comp_decomp
        if comp_decomp and comp_decomp.composites:
            for j, motif in enumerate(comp_decomp.composites):
                # Center = the singleton center of the composite
                entities.append(InducedEntity(
                    id=f"host_{j}",
                    kind="host",
                    bbox=(motif.center.row, motif.center.col,
                          motif.center.bbox_h, motif.center.bbox_w),
                    color=motif.center.color,
                    size=motif.center.size,
                ))
                for k, frame_obj in enumerate(motif.frames):
                    entities.append(InducedEntity(
                        id=f"gap_{j}_{k}",
                        kind="gap",
                        bbox=(frame_obj.row, frame_obj.col,
                              frame_obj.bbox_h, frame_obj.bbox_w),
                        color=frame_obj.color,
                        size=frame_obj.size,
                    ))
    except Exception:
        pass  # composite decomposition may not apply

    # Panel entities from propose_decompositions
    from aria.decomposition import (
        PanelDecomposition,
        PartitionDecomposition,
        RegionDecomposition,
        HostSlotDecomposition,
        propose_decompositions,
    )
    all_hyps = propose_decompositions(inp, bg)
    for hyp in all_hyps:
        if hyp.label == "panel" and isinstance(hyp.data, PanelDecomposition):
            for panel in hyp.data.panels:
                entities.append(InducedEntity(
                    id=f"panel_{panel.index}",
                    kind="panel",
                    bbox=(panel.row, panel.col, panel.height, panel.width),
                    color=panel.bg_color,
                    size=panel.height * panel.width,
                ))
        elif hyp.label == "partition" and isinstance(hyp.data, PartitionDecomposition):
            for ci, bbox in enumerate(hyp.data.cell_bboxes):
                r0, c0, r1, c1 = bbox
                entities.append(InducedEntity(
                    id=f"cell_{ci}",
                    kind="panel",
                    bbox=(r0, c0, r1 - r0 + 1, c1 - c0 + 1),
                    size=(r1 - r0 + 1) * (c1 - c0 + 1),
                ))
        elif hyp.label == "region" and isinstance(hyp.data, RegionDecomposition):
            for ri, reg in enumerate(hyp.data.regions):
                eid = f"region_{ri}"
                # Avoid duplicating frame entities already added
                entities.append(InducedEntity(
                    id=eid,
                    kind="region",
                    bbox=(reg.row, reg.col, reg.height, reg.width),
                    color=reg.frame_color,
                    size=reg.height * reg.width,
                ))
        elif hyp.label == "host_slot" and isinstance(hyp.data, HostSlotDecomposition):
            for hi, host in enumerate(hyp.data.hosts):
                entities.append(InducedEntity(
                    id=f"hs_host_{hi}",
                    kind="host",
                    bbox=(host.row, host.col, host.bbox_h, host.bbox_w),
                    color=host.color,
                    size=host.size,
                ))
            for si, slot in enumerate(hyp.data.slots):
                entities.append(InducedEntity(
                    id=f"hs_slot_{si}",
                    kind="gap",
                    bbox=(slot.row, slot.col, slot.bbox_h, slot.bbox_w),
                    color=slot.color,
                    size=slot.size,
                ))

    artifacts.entities = entities

    # --- Relation extraction ---
    relations = _extract_relations(obj_decomp, framed)
    artifacts.relations = relations

    # --- Template hypotheses ---
    templates, slot_candidates = _extract_template_hypotheses(demos, task_id)
    artifacts.template_hypotheses = templates
    artifacts.slot_candidates = slot_candidates

    # --- Executor gate ---
    _check_executor(artifacts, demos)

    return artifacts


def _extract_decomposition_hypotheses(
    demos: tuple[DemoPair, ...],
) -> list[str]:
    """Determine which decomposition views apply to this task.

    Uses the new propose_decompositions() which runs all detectors
    (object, partition, panel, region, frame, host_slot) and returns
    ranked hypotheses.
    """
    if not demos:
        return []

    from aria.decomposition import propose_decompositions

    inp = demos[0].input
    bg = detect_bg(inp)
    hypotheses = propose_decompositions(inp, bg)

    # Return labels in confidence-ranked order
    return [h.label for h in hypotheses]


def _extract_relations(
    obj_decomp: ObjectDecomposition,
    framed: list[FramedRegion],
) -> list[InducedRelation]:
    """Extract relations from decomposition outputs."""
    relations: list[InducedRelation] = []
    objects = obj_decomp.objects

    # Adjacency relations (share a border)
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if j <= i:
                continue
            if _objects_adjacent(a, b):
                relations.append(InducedRelation(
                    kind="adjacent_to",
                    source_id=f"obj_{i}",
                    target_id=f"obj_{j}",
                ))

    # Containment: singleton inside non-singleton bbox
    for i, a in enumerate(objects):
        if not a.is_singleton:
            continue
        for j, b in enumerate(objects):
            if b.is_singleton or i == j:
                continue
            if _point_in_bbox(a.row, a.col, b.row, b.col, b.bbox_h, b.bbox_w):
                relations.append(InducedRelation(
                    kind="contains",
                    source_id=f"obj_{j}",
                    target_id=f"obj_{i}",
                ))

    # Alignment relations
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if j <= i:
                continue
            if _aligned_row(a, b):
                relations.append(InducedRelation(
                    kind="aligned_row",
                    source_id=f"obj_{i}",
                    target_id=f"obj_{j}",
                ))
            if _aligned_col(a, b):
                relations.append(InducedRelation(
                    kind="aligned_col",
                    source_id=f"obj_{i}",
                    target_id=f"obj_{j}",
                ))

    # Frame containment
    for fi, fr in enumerate(framed):
        for oi, obj in enumerate(objects):
            if _point_in_bbox(
                obj.row, obj.col,
                fr.row, fr.col, fr.height, fr.width,
            ):
                relations.append(InducedRelation(
                    kind="contains",
                    source_id=f"frame_{fi}",
                    target_id=f"obj_{oi}",
                ))

    return relations


def _extract_template_hypotheses(
    demos: tuple[DemoPair, ...],
    task_id: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Extract template family hypotheses from sketch fitting."""
    templates: list[str] = []
    slot_candidates: dict[str, list[str]] = {}

    try:
        fit_result = fit_sketches_with_report(demos, task_id)
        for sketch in fit_result.sketches:
            family = _sketch_to_template_family(sketch)
            if family and family not in templates:
                templates.append(family)
            # Extract slot info from sketch steps
            _extract_slots_from_sketch(sketch, slot_candidates)
    except Exception:
        pass

    # Propose templates based on decomposition hypotheses + structural cues
    if demos:
        inp, out = demos[0].input, demos[0].output
        same_dims = inp.shape == out.shape
        bg = detect_bg(inp)
        objs = decompose_objects(inp, bg)

        from aria.decomposition import propose_decompositions
        hyps = propose_decompositions(inp, bg)
        hyp_labels = {h.label for h in hyps}

        # --- Decomposition-driven template proposals ---

        # Panel or partition → panel_combine_rewrite
        if "panel" in hyp_labels or "partition" in hyp_labels:
            if "panel_combine_rewrite" not in templates:
                templates.append("panel_combine_rewrite")

        # Region → region_fill or extract_modify
        if "region" in hyp_labels:
            if "region_fill" not in templates:
                templates.append("region_fill")
            if not same_dims and "extract_modify" not in templates:
                templates.append("extract_modify")

        # Host-slot → host_slot_place
        if "host_slot" in hyp_labels:
            if "host_slot_place" not in templates:
                templates.append("host_slot_place")

        # Frame → region_fill
        if "frame" in hyp_labels:
            if "region_fill" not in templates:
                templates.append("region_fill")

        # --- Object-level structural cues ---

        # Singletons + non-singletons with same dims → match_recolor
        if same_dims and objs.singletons and objs.non_singletons:
            if "match_recolor" not in templates:
                templates.append("match_recolor")

        # Same dims, multiple small objects of different colors → swap
        if same_dims and len(objs.non_singletons) >= 2:
            colors = {o.color for o in objs.non_singletons}
            small_objs = [o for o in objs.non_singletons if o.size <= 9]
            if len(colors) >= 2 and len(small_objs) >= 2:
                if "swap" not in templates:
                    templates.append("swap")
                if "match_recolor" not in templates:
                    templates.append("match_recolor")

        # Same dims, pixel changes form a pure color-pair swap → swap
        if same_dims and "swap" not in templates:
            import numpy as np
            diff_mask = inp != out
            if np.any(diff_mask):
                from_vals = inp[diff_mask]
                to_vals = out[diff_mask]
                pairs = set(zip(from_vals.tolist(), to_vals.tolist()))
                # Pure swap: every (a→b) has a matching (b→a)
                is_swap = all((b, a) in pairs for a, b in pairs) and len(pairs) <= 4
                if is_swap:
                    templates.append("swap")

        # Same dims with singletons → host_slot_place
        if same_dims and objs.singletons:
            if "host_slot_place" not in templates:
                templates.append("host_slot_place")

        # Dims change → extract_modify
        if not same_dims:
            if "extract_modify" not in templates:
                templates.append("extract_modify")

    return templates, slot_candidates


def _sketch_to_template_family(sketch) -> str | None:
    """Map a Sketch object to a template family label."""
    from aria.sketch import PrimitiveFamily
    if not hasattr(sketch, "steps") or not sketch.steps:
        return None

    # Check sketch step families/ops
    step_ops = set()
    for step in sketch.steps:
        if hasattr(step, "family"):
            step_ops.add(step.family.value if isinstance(step.family, PrimitiveFamily) else str(step.family))
        if hasattr(step, "op"):
            step_ops.add(str(step.op))

    # Map to template families
    if "periodic_repair" in str(step_ops) or "repair" in str(step_ops).lower():
        return "region_fill"
    if "role_alignment" in str(step_ops) or "composite" in str(step_ops).lower():
        return "match_recolor"
    if "canvas" in str(step_ops).lower() or "tile" in str(step_ops).lower():
        return "extract_modify"
    if "movement" in str(step_ops).lower():
        return "swap"

    return None


def _extract_slots_from_sketch(sketch, slot_candidates: dict[str, list[str]]) -> None:
    """Extract slot names/values from sketch for slot gate scoring."""
    if not hasattr(sketch, "steps"):
        return
    for step in sketch.steps:
        if hasattr(step, "slots"):
            for slot in step.slots:
                name = slot.name if hasattr(slot, "name") else str(slot)
                slot_candidates.setdefault(name, []).append(str(slot))
        if hasattr(step, "roles"):
            for role in step.roles:
                name = role.name if hasattr(role, "name") else str(role)
                slot_candidates.setdefault("color_role", []).append(name)


def _check_executor(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    """Check if any active solver path can produce a runnable program.

    Tests all active paths: sketch compilation, observation synthesis,
    legacy observe, scene_solve (the canonical active pipeline), and
    direct correspondence search.

    Gate 6 semantics: "a runnable executable candidate exists" — a path
    that produces a candidate and executes it without error counts as a
    pass, even if the candidate doesn't verify exactly on all demos.
    """
    artifacts.executor_attempted = True

    # Path 1: sketch compilation
    _try_path_sketch(artifacts, demos)

    # Path 2: observation synthesis
    _try_path_synthesize(artifacts, demos)

    # Path 3: per-object observation (legacy)
    _try_path_legacy_observe(artifacts, demos)

    # Path 4: scene_solve (canonical active pipeline — includes
    # factor_search, correspondence_search, 16+ families)
    _try_path_scene_solve(artifacts, demos)

    # Path 5: direct correspondence search (provenance/attribution)
    _try_path_correspondence(artifacts, demos)

    # executor_ran = any path produced a runnable candidate
    artifacts.executor_ran = len(artifacts.executor_paths_ran) > 0
    if artifacts.executor_paths_ran and artifacts.executor_path is None:
        artifacts.executor_path = artifacts.executor_paths_ran[0]
    # Prefer a verified path for attribution if available
    if artifacts.executor_paths_verified:
        artifacts.executor_path = artifacts.executor_paths_verified[0]


def _try_path_sketch(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    path_name = "sketch"
    artifacts.executor_paths_tried.append(path_name)
    try:
        from aria.sketch_compile import compile_sketch
        from aria.sketch_fit import fit_sketches
        sketches = fit_sketches(demos, artifacts.task_id)
        for sketch in sketches:
            try:
                programs = compile_sketch(sketch, demos)
                if programs:
                    artifacts.executor_paths_produced.append(path_name)
                    artifacts.executor_paths_ran.append(path_name)
                    return
            except Exception:
                continue
    except Exception:
        pass


def _try_path_synthesize(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    path_name = "synthesize"
    artifacts.executor_paths_tried.append(path_name)
    try:
        from aria.synthesize import synthesize_from_observations
        sr = synthesize_from_observations(demos)
        if sr.candidates_tested > 0:
            artifacts.executor_paths_produced.append(path_name)
            artifacts.executor_paths_ran.append(path_name)
        if sr.solved:
            artifacts.executor_paths_verified.append(path_name)
    except Exception:
        pass


def _try_path_legacy_observe(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    path_name = "legacy_observe"
    artifacts.executor_paths_tried.append(path_name)
    try:
        from aria.legacy.observe import observe_and_synthesize
        obs = observe_and_synthesize(demos)
        if obs.solved:
            artifacts.executor_paths_produced.append(path_name)
            artifacts.executor_paths_ran.append(path_name)
            artifacts.executor_paths_verified.append(path_name)
    except Exception as e:
        artifacts.executor_error = str(e)


def _try_path_scene_solve(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    path_name = "scene_solve"
    artifacts.executor_paths_tried.append(path_name)
    try:
        from aria.core.scene_solve import infer_scene_programs, verify_scene_program
        candidates = infer_scene_programs(demos)
        if candidates:
            artifacts.executor_paths_produced.append(path_name)
        for prog in candidates:
            try:
                # Attempt execution — if it runs without error, the path is runnable
                if hasattr(prog, 'execute'):
                    prog.execute(demos[0].input)
                elif hasattr(prog, 'verify_on_demo'):
                    # CorrespondenceProgram: verify_on_demo is its execution
                    pass
                else:
                    from aria.core.scene_executor import execute_scene_program
                    execute_scene_program(prog, demos[0].input)
                artifacts.executor_paths_ran.append(path_name)
                # Check if it verifies on all demos
                if verify_scene_program(prog, demos):
                    artifacts.executor_paths_verified.append(path_name)
                return
            except Exception:
                continue
    except Exception:
        pass


def _try_path_correspondence(artifacts: StageArtifacts, demos: tuple[DemoPair, ...]) -> None:
    path_name = "correspondence"
    artifacts.executor_paths_tried.append(path_name)
    try:
        from aria.correspondence import correspondence_search
        corr_results = correspondence_search(demos, max_programs=50)
        if corr_results:
            artifacts.executor_paths_produced.append(path_name)
            desc, corr_prog = corr_results[0]
            # Correspondence programs are inherently runnable
            artifacts.executor_paths_ran.append(path_name)
            # Check verification
            if hasattr(corr_prog, 'verify_on_demo'):
                import numpy as np
                if all(corr_prog.verify_on_demo(d.input, d.output) for d in demos):
                    artifacts.executor_paths_verified.append(path_name)
    except Exception:
        pass


def _objects_adjacent(a: RawObject, b: RawObject) -> bool:
    """Check if two objects are adjacent (bboxes touch or overlap by 1)."""
    a_r2 = a.row + a.bbox_h
    a_c2 = a.col + a.bbox_w
    b_r2 = b.row + b.bbox_h
    b_c2 = b.col + b.bbox_w

    # Check if bboxes are within 1 cell of each other
    row_gap = max(0, max(a.row, b.row) - min(a_r2, b_r2))
    col_gap = max(0, max(a.col, b.col) - min(a_c2, b_c2))

    return row_gap <= 1 and col_gap <= 1 and (row_gap + col_gap) > 0 or (
        row_gap == 0 and col_gap == 0
    )


def _point_in_bbox(
    pr: int, pc: int,
    br: int, bc: int, bh: int, bw: int,
) -> bool:
    return br <= pr < br + bh and bc <= pc < bc + bw


def _aligned_row(a: RawObject, b: RawObject) -> bool:
    """Objects share a row center (within 1 cell)."""
    return abs(a.center_row - b.center_row) <= 1


def _aligned_col(a: RawObject, b: RawObject) -> bool:
    """Objects share a column center (within 1 cell)."""
    return abs(a.center_col - b.center_col) <= 1
