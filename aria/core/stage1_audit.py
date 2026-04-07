"""Stage-1 gap audit — identify missing first-stage transforms for composition.

Analyzes composition-candidate tasks to classify what stage-1 operation
would be needed before an existing stage-2 lane could apply.

Labels:
  isolate_region     — need to extract/isolate a subregion (frame interior, object bbox)
  crop_subgrid       — need to crop output to a smaller subgrid from input
  keep_subset        — need to keep some objects/cells and erase others
  erase_subset       — need to erase some objects/cells and keep the rest
  normalize_origin   — need to normalize position/alignment before stage-2
  local_cleanup      — need to fix/clean a localized region before stage-2
  split_objects      — need to decompose composite objects into parts
  no_clear_gap       — no clear single stage-1 transform would help
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


LABEL_ISOLATE_REGION = "isolate_region"
LABEL_CROP_SUBGRID = "crop_subgrid"
LABEL_KEEP_SUBSET = "keep_subset"
LABEL_ERASE_SUBSET = "erase_subset"
LABEL_NORMALIZE_ORIGIN = "normalize_origin"
LABEL_LOCAL_CLEANUP = "local_cleanup"
LABEL_SPLIT_OBJECTS = "split_objects"
LABEL_NO_CLEAR_GAP = "no_clear_gap"


@dataclass
class Stage1Record:
    task_id: str
    label: str
    rationale: str
    dims_change: bool = False
    has_frame: bool = False
    n_in_shapes: int = 0
    n_out_shapes: int = 0


@dataclass
class Stage1Report:
    n_tasks: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    records: list[Stage1Record] = field(default_factory=list)


def audit_stage1_gaps(
    task_ids: list[str],
    demos_fn,
) -> Stage1Report:
    """Audit composition-candidate tasks for missing stage-1 transforms."""
    from aria.core.mechanism_evidence import compute_evidence_and_rank
    from aria.core.arc import ARCFitter, ARCSpecializer, ARCCompiler, ARCVerifier
    from aria.core.protocol import solve as core_solve
    from aria.decomposition import detect_bg, detect_framed_regions
    from aria.runtime.ops.selection import _find_objects

    fitter = ARCFitter()
    specializer = ARCSpecializer()
    compiler = ARCCompiler()
    verifier = ARCVerifier()

    report = Stage1Report()

    for tid in task_ids:
        try:
            demos = demos_fn(tid)
        except Exception:
            continue

        result = core_solve(demos, fitter, specializer, compiler, verifier, task_id=tid)
        if result.solved:
            continue

        ev, ranking = compute_evidence_and_rank(demos)
        demo = demos[0]
        bg = detect_bg(demo.input)
        dims_change = demo.input.shape != demo.output.shape

        has_frame = False
        try:
            regions = detect_framed_regions(demo.input)
            has_frame = len(regions) > 0
        except Exception:
            pass

        in_objs = list(_find_objects(demo.input))
        out_objs = list(_find_objects(demo.output))
        in_shapes = [o for o in in_objs if o.color != bg and o.size > 1]
        out_shapes = [o for o in out_objs if o.color != detect_bg(demo.output) and o.size > 1]

        # Classify the stage-1 gap
        label, rationale = _classify_gap(
            demo, bg, dims_change, has_frame, in_shapes, out_shapes, ev,
        )

        report.n_tasks += 1
        report.label_counts[label] = report.label_counts.get(label, 0) + 1
        report.records.append(Stage1Record(
            task_id=tid, label=label, rationale=rationale,
            dims_change=dims_change, has_frame=has_frame,
            n_in_shapes=len(in_shapes), n_out_shapes=len(out_shapes),
        ))

    return report


def _classify_gap(demo, bg, dims_change, has_frame, in_shapes, out_shapes, ev):
    """Classify the missing stage-1 transform."""
    ir, ic = demo.input.shape
    or_, oc = demo.output.shape

    # Dims change: output is smaller -> crop
    if dims_change and or_ < ir and oc < ic:
        return LABEL_CROP_SUBGRID, f"output smaller ({ir}x{ic}->{or_}x{oc}): needs crop"

    # Dims change: output is larger -> this is canvas construction (already a lane)
    if dims_change and or_ > ir and oc > ic:
        return LABEL_NO_CLEAR_GAP, f"output larger: canvas lane should handle"

    # Dims change: mixed (one dim grows, one shrinks, or asymmetric)
    if dims_change:
        if or_ < ir or oc < ic:
            return LABEL_CROP_SUBGRID, f"partial crop ({ir}x{ic}->{or_}x{oc})"
        return LABEL_ISOLATE_REGION, f"dims change with complex reshape"

    # Same dims, framed region: might need to isolate interior
    if has_frame and len(in_shapes) > len(out_shapes):
        return LABEL_ERASE_SUBSET, f"framed + fewer output shapes: erase subset"

    if has_frame and len(out_shapes) > len(in_shapes):
        return LABEL_KEEP_SUBSET, f"framed + more output shapes: keep/generate subset"

    # Same dims, many objects: might need to isolate a region
    if len(in_shapes) > 5 and len(out_shapes) < len(in_shapes):
        return LABEL_ERASE_SUBSET, f"many objects, fewer in output: selective erasure"

    if len(in_shapes) > 1 and has_frame:
        return LABEL_ISOLATE_REGION, f"framed + objects: isolate region first"

    # Objects present but need normalization
    if ev.shapes_shift_position and len(in_shapes) == len(out_shapes):
        return LABEL_NORMALIZE_ORIGIN, f"shapes shift: normalize position"

    # Fallback
    if has_frame:
        return LABEL_LOCAL_CLEANUP, f"framed region: local cleanup needed"

    return LABEL_NO_CLEAR_GAP, "no clear single-stage gap identified"


def format_stage1_audit(report: Stage1Report) -> str:
    lines = [f"=== Stage-1 Gap Audit ({report.n_tasks} unsolved tasks) ===", ""]
    lines.append("Label distribution:")
    for label, count in sorted(report.label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)
