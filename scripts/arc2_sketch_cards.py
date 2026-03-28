"""Per-task ARC-2 sketch-card generator.

Produces an individual card for every ARC-2 eval task describing:
- decomposition: how the task breaks into sub-problems
- invariants: what stays the same between input and output
- construction: how the output is built from the input
- system_failure: what the current system assumed wrongly
- sketch: what kind of program would solve this task

Cards are task-specific. Shared tags appear only as secondary metadata.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

import aria.runtime  # noqa: F401  — register ops
from aria.datasets import get_dataset, iter_tasks, load_arc_task
from aria.graph.extract import extract
from aria.graph.signatures import compute_task_signatures
from aria.observe import observe_and_synthesize, dims_change_reconstruct
from aria.types import DemoPair, Grid

ROOT = Path("/Users/ericc59/Dev/aria")


# ---------------------------------------------------------------------------
# Per-task analysis primitives
# ---------------------------------------------------------------------------


def _dominant_color(grid: Grid) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _color_roles_per_demo(demos: tuple[DemoPair, ...]) -> list[dict]:
    """Per-demo color role assignment: bg, fg colors, added/removed colors."""
    roles = []
    for d in demos:
        bg_in = _dominant_color(d.input)
        bg_out = _dominant_color(d.output)
        in_colors = set(int(v) for v in np.unique(d.input))
        out_colors = set(int(v) for v in np.unique(d.output))
        roles.append({
            "bg_in": bg_in,
            "bg_out": bg_out,
            "bg_same": bg_in == bg_out,
            "fg_in": sorted(in_colors - {bg_in}),
            "fg_out": sorted(out_colors - {bg_out}),
            "colors_added": sorted(out_colors - in_colors),
            "colors_removed": sorted(in_colors - out_colors),
        })
    return roles


def _pixel_diff_analysis(demos: tuple[DemoPair, ...]) -> dict | None:
    """Detailed pixel-level diff for same-dims tasks."""
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    per_demo = []
    for d in demos:
        diff_mask = d.input != d.output
        n_changed = int(diff_mask.sum())
        if n_changed == 0:
            per_demo.append({
                "changed_pixels": 0,
                "total_pixels": int(d.input.size),
                "changed_rows": [],
                "changed_cols": [],
                "value_transitions": {},
            })
            continue

        changed_positions = np.argwhere(diff_mask)
        changed_rows = sorted(set(int(r) for r in changed_positions[:, 0]))
        changed_cols = sorted(set(int(c) for c in changed_positions[:, 1]))

        transitions: Counter = Counter()
        for r, c in changed_positions:
            transitions[(int(d.input[r, c]), int(d.output[r, c]))] += 1

        per_demo.append({
            "changed_pixels": n_changed,
            "total_pixels": int(d.input.size),
            "changed_rows": changed_rows,
            "changed_cols": changed_cols,
            "value_transitions": {
                f"{src}->{dst}": count
                for (src, dst), count in transitions.most_common()
            },
        })

    return {
        "per_demo": per_demo,
        "max_changed": max(p["changed_pixels"] for p in per_demo),
        "all_same_transition_set": (
            len(set(frozenset(p["value_transitions"].keys()) for p in per_demo)) == 1
        ),
    }


def _object_census(grid: Grid, bg: int) -> list[dict]:
    """Extract connected-component census: color, size, bbox, position."""
    objects = []
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labeled == label_id)
            min_r, max_r = int(ys.min()), int(ys.max())
            min_c, max_c = int(xs.min()), int(xs.max())
            size = int((labeled == label_id).sum())
            objects.append({
                "color": color,
                "size": size,
                "bbox": (min_r, min_c, max_r - min_r + 1, max_c - min_c + 1),
                "center": (min_r + (max_r - min_r) // 2, min_c + (max_c - min_c) // 2),
            })
    return objects


def _object_structure(demos: tuple[DemoPair, ...]) -> dict:
    """Summarize object-level structure across demos."""
    per_demo = []
    for d in demos:
        bg = _dominant_color(d.input)
        inp_objs = _object_census(d.input, bg)
        out_objs = _object_census(d.output, _dominant_color(d.output))

        inp_by_color = Counter(o["color"] for o in inp_objs)
        out_by_color = Counter(o["color"] for o in out_objs)
        inp_sizes = sorted(set(o["size"] for o in inp_objs))
        out_sizes = sorted(set(o["size"] for o in out_objs))

        per_demo.append({
            "n_input_objects": len(inp_objs),
            "n_output_objects": len(out_objs),
            "input_color_counts": dict(inp_by_color),
            "output_color_counts": dict(out_by_color),
            "input_size_set": inp_sizes,
            "output_size_set": out_sizes,
            "objects_preserved": len(inp_objs) == len(out_objs),
            "has_singleton_anchor": any(o["size"] == 1 for o in inp_objs),
        })
    return {
        "per_demo": per_demo,
        "object_count_stable": len(set(p["n_input_objects"] for p in per_demo)) == 1,
        "any_singletons": any(p["has_singleton_anchor"] for p in per_demo),
    }


def _spatial_regularity(grid: Grid, bg: int) -> dict:
    """Detect spatial regularity: periodicity, symmetry, frame structure."""
    rows, cols = grid.shape
    fg_mask = grid != bg

    # Frame detection: is the border all one color?
    border_pixels = set()
    for r in range(rows):
        border_pixels.add(int(grid[r, 0]))
        border_pixels.add(int(grid[r, cols - 1]))
    for c in range(cols):
        border_pixels.add(int(grid[0, c]))
        border_pixels.add(int(grid[rows - 1, c]))
    has_frame = len(border_pixels) == 1

    # Row/col periodicity in foreground
    row_patterns = []
    for r in range(rows):
        row_patterns.append(tuple(int(v) for v in grid[r]))
    unique_rows = len(set(row_patterns))

    col_patterns = []
    for c in range(cols):
        col_patterns.append(tuple(int(grid[r, c]) for r in range(rows)))
    unique_cols = len(set(col_patterns))

    # Symmetry checks
    h_sym = np.array_equal(grid, np.fliplr(grid))
    v_sym = np.array_equal(grid, np.flipud(grid))
    rot180 = np.array_equal(grid, np.rot90(grid, 2))

    return {
        "has_frame": has_frame,
        "frame_color": border_pixels.pop() if has_frame else None,
        "unique_rows": unique_rows,
        "unique_cols": unique_cols,
        "row_repetition": unique_rows < rows,
        "col_repetition": unique_cols < cols,
        "h_symmetric": h_sym,
        "v_symmetric": v_sym,
        "rot180_symmetric": rot180,
    }


def _dims_relationship(demos: tuple[DemoPair, ...]) -> dict | None:
    """For dims-change tasks, describe the relationship."""
    if all(d.input.shape == d.output.shape for d in demos):
        return None

    entries = []
    for d in demos:
        ir, ic = d.input.shape
        orr, oc = d.output.shape
        entries.append({
            "input_shape": (ir, ic),
            "output_shape": (orr, oc),
            "row_ratio": round(orr / max(ir, 1), 3),
            "col_ratio": round(oc / max(ic, 1), 3),
            "area_ratio": round((orr * oc) / max(ir * ic, 1), 3),
            "shrink": orr * oc < ir * ic,
        })

    # Check if output dims are consistent across demos
    output_shapes = set(e["output_shape"] for e in entries)
    input_shapes = set(e["input_shape"] for e in entries)

    return {
        "per_demo": entries,
        "output_shape_fixed": len(output_shapes) == 1,
        "input_shape_fixed": len(input_shapes) == 1,
        "all_shrink": all(e["shrink"] for e in entries),
        "all_grow": all(not e["shrink"] for e in entries),
    }


# ---------------------------------------------------------------------------
# Decomposition / invariant / construction inference
# ---------------------------------------------------------------------------


def _infer_decomposition(
    demos: tuple[DemoPair, ...],
    color_roles: list[dict],
    obj_struct: dict,
    pixel_diff: dict | None,
    spatial: dict,
    dims_rel: dict | None,
) -> list[str]:
    """Infer candidate decomposition steps for this specific task."""
    steps = []
    same_dims = all(d.input.shape == d.output.shape for d in demos)

    # Color role rotation across demos?
    bg_colors = [cr["bg_in"] for cr in color_roles]
    if len(set(bg_colors)) > 1:
        steps.append("normalize color roles across demos (bg/fg/marker colors rotate)")

    if not same_dims and dims_rel:
        if dims_rel["all_shrink"]:
            steps.append("identify region of interest in input")
            steps.append("extract or crop to output dimensions")
        elif dims_rel["all_grow"]:
            steps.append("determine output canvas size from input structure")
            steps.append("tile, extend, or construct output from input content")
        if dims_rel["output_shape_fixed"]:
            steps.append("output shape is constant — derive from structural element, not input dims")
        elif not dims_rel["input_shape_fixed"]:
            steps.append("output shape varies with input — infer size rule per demo")

    if same_dims and pixel_diff:
        if pixel_diff["max_changed"] <= 5:
            steps.append("identify small set of pixels to change")
            steps.append("infer selection rule for which pixels change")
        elif pixel_diff["max_changed"] <= 30:
            steps.append("identify changed region(s)")
            steps.append("infer per-region transformation rule")
        else:
            steps.append("significant grid-wide transformation")
            steps.append("analyze structural correspondence between input and output objects")

    if spatial["has_frame"]:
        steps.append(f"detect frame (color {spatial['frame_color']}), analyze interior")

    if obj_struct["any_singletons"]:
        steps.append("identify singleton anchor objects and their spatial relationships")

    if not steps:
        steps.append("full structural analysis needed — no clear decomposition shortcut")

    return steps


def _infer_invariants(
    demos: tuple[DemoPair, ...],
    color_roles: list[dict],
    obj_struct: dict,
    pixel_diff: dict | None,
    sigs: frozenset[str],
) -> list[str]:
    """Infer what stays the same between input and output for this task."""
    invariants = []
    same_dims = all(d.input.shape == d.output.shape for d in demos)

    if same_dims:
        invariants.append("grid dimensions preserved")

    if "color:palette_same" in sigs:
        invariants.append("color palette identical in input and output")
    elif "color:palette_subset" in sigs:
        invariants.append("output uses a subset of input colors")

    if "change:additive" in sigs:
        invariants.append("all non-background input pixels preserved in output")
    if "change:bg_preserved" in sigs:
        invariants.append("background pixels unchanged")

    # Check if object count is preserved
    if obj_struct["object_count_stable"]:
        for p in obj_struct["per_demo"]:
            if p["objects_preserved"]:
                invariants.append("number of objects preserved between input and output")
                break

    # Check if specific colors are always stationary
    if same_dims and pixel_diff:
        for p in pixel_diff["per_demo"]:
            if p["changed_pixels"] > 0:
                changed_in_colors = set()
                for k in p["value_transitions"]:
                    src = int(k.split("->")[0])
                    changed_in_colors.add(src)
                all_in_colors = set(int(v) for v in np.unique(demos[0].input))
                stationary = all_in_colors - changed_in_colors
                if stationary:
                    invariants.append(f"colors {sorted(stationary)} never change")
                break

    # Color role consistency across demos
    bg_set = set(cr["bg_in"] for cr in color_roles)
    if len(bg_set) > 1:
        invariants.append("structural role pattern is constant even though colors rotate")

    if not invariants:
        invariants.append("no obvious invariant detected — requires deeper analysis")

    return invariants


def _infer_construction(
    demos: tuple[DemoPair, ...],
    color_roles: list[dict],
    obj_struct: dict,
    pixel_diff: dict | None,
    dims_rel: dict | None,
    spatial: dict,
) -> str:
    """Describe how the output is constructed for this specific task."""
    same_dims = all(d.input.shape == d.output.shape for d in demos)

    if same_dims and pixel_diff:
        max_ch = pixel_diff["max_changed"]
        if max_ch == 0:
            return "output is identical to input (identity)"
        if max_ch <= 3:
            # Very few pixels change — describe what changes
            transitions = pixel_diff["per_demo"][0]["value_transitions"]
            return (
                f"modify {max_ch} pixel(s): transitions {transitions}. "
                f"Rule selects specific positions based on local or global context."
            )
        if max_ch <= 20:
            return (
                f"modify {max_ch} pixels across specific positions. "
                f"Changes likely governed by object relationships, periodicity, or spatial rule."
            )

        # Many pixels change — describe at object level
        for p in obj_struct["per_demo"]:
            if p["objects_preserved"]:
                return (
                    f"rearrange or transform {p['n_input_objects']} objects in place. "
                    f"Object count preserved; positions, colors, or shapes change."
                )
        return (
            f"reconstruct grid content: {max_ch}/{pixel_diff['per_demo'][0]['total_pixels']} "
            f"pixels change. Object-level or region-level transformation."
        )

    if dims_rel:
        if dims_rel["all_shrink"]:
            if dims_rel["output_shape_fixed"]:
                return (
                    "extract a fixed-size region from the input. "
                    "Selection rule determines which region."
                )
            return (
                "crop or extract variable-size region from input. "
                "Output size determined by input structure (e.g., object bounding box, partition cell)."
            )
        if dims_rel["all_grow"]:
            return (
                "construct larger output from input content. "
                "May involve tiling, recursive expansion, or assembly."
            )
        return "reshape: output dimensions differ from input by a task-specific rule."

    return "construction rule unclear from surface analysis"


def _infer_system_failure(
    obs_result,
    dims_result,
    sigs: frozenset[str],
    color_roles: list[dict],
    pixel_diff: dict | None,
    dims_rel: dict | None,
) -> str:
    """Describe what the current system assumed wrongly for this task."""
    parts = []

    # Did observation fire useful rules?
    move_rules = [r for r in obs_result.rules if r.kind == "move"]
    nonzero_moves = [r for r in move_rules
                     if r.details.get("dr", 0) != 0 or r.details.get("dc", 0) != 0]

    if obs_result.solved:
        return "system solved this task (observation)"
    if dims_result and dims_result.solved:
        return "system solved this task (dims reconstruction)"

    # Color role rotation
    bg_set = set(cr["bg_in"] for cr in color_roles)
    if len(bg_set) > 1:
        parts.append(
            "system treats color literally (by_color(3)) but this task assigns "
            "structural roles to colors that change between demos"
        )

    if not obs_result.rules:
        parts.append("observation module found no rules — task doesn't match any known rule family")
    elif not nonzero_moves and move_rules:
        parts.append(
            "observation found move rules but all have zero delta — "
            "objects stay put under the system's object model but something else changes"
        )
    elif nonzero_moves and not obs_result.solved:
        parts.append(
            "observation found movement rules but couldn't build a verified program — "
            "the movement may be conditional, role-dependent, or compositional"
        )

    if dims_rel and not (dims_result and dims_result.solved):
        parts.append(
            "system couldn't determine output dimensions or populate the output canvas — "
            "no dims reconstruction strategy matched"
        )

    if pixel_diff and pixel_diff["max_changed"] <= 10 and not obs_result.solved:
        parts.append(
            "only a few pixels change but the system can't identify the selection rule — "
            "the rule likely depends on spatial context the observation module doesn't analyze"
        )

    if not parts:
        parts.append(
            "search exhausted budget without finding a valid program — "
            "the required composition depth or op vocabulary exceeds current capabilities"
        )

    return "; ".join(parts)


def _infer_sketch(
    demos: tuple[DemoPair, ...],
    decomposition: list[str],
    invariants: list[str],
    construction: str,
    color_roles: list[dict],
    obj_struct: dict,
    pixel_diff: dict | None,
    dims_rel: dict | None,
    spatial: dict,
) -> str:
    """Propose a concrete sketch shape for a program that would solve this task."""
    same_dims = all(d.input.shape == d.output.shape for d in demos)
    bg_rotates = len(set(cr["bg_in"] for cr in color_roles)) > 1

    parts = []

    if bg_rotates:
        parts.append("1. identify_roles(input) → {bg, fg_colors, marker, frame}")

    if spatial["has_frame"]:
        parts.append(f"{'2' if bg_rotates else '1'}. extract_interior(input, frame_color={spatial['frame_color']}) → interior_grid")

    if same_dims and pixel_diff:
        max_ch = pixel_diff["max_changed"]
        if max_ch <= 10:
            # Small change — need context-dependent pixel selection
            parts.append("find_target_positions(grid, context_rule) → positions")
            parts.append("for each position: compute_new_value(position, neighbors, global_context)")
            parts.append("paint(grid, positions, new_values) → output")
        else:
            if obj_struct["any_singletons"]:
                parts.append("find_objects(input) → objects")
                parts.append("identify_anchor(objects) → anchor")
                parts.append("for each non-anchor object: compute_transform(object, anchor) → new_position/shape")
                parts.append("paint_objects(transformed, canvas) → output")
            else:
                parts.append("find_objects(input) → objects")
                parts.append("for each object: apply_rule(object, context) → transformed_object")
                parts.append("compose(transformed_objects, background) → output")

    elif dims_rel:
        if dims_rel["all_shrink"]:
            parts.append("identify_target_region(input) → bbox or mask")
            parts.append("extract(input, region) → output")
        elif dims_rel["all_grow"]:
            parts.append("compute_output_dims(input) → (rows, cols)")
            parts.append("construct_canvas(dims) → blank")
            parts.append("populate(blank, input_content, expansion_rule) → output")
        else:
            parts.append("compute_output_dims(input) → (rows, cols)")
            parts.append("transform_and_place(input, output_canvas) → output")

    if not parts:
        parts.append("// no clear sketch — requires manual task inspection")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


@dataclass
class SketchCard:
    task_id: str
    same_dims: bool
    input_shape_range: str
    output_shape_range: str
    n_demos: int
    decomposition: list[str]
    invariants: list[str]
    construction: str
    system_failure: str
    sketch: str
    color_roles_summary: str
    pixel_diff_summary: str
    object_summary: str
    spatial_summary: str
    task_signatures: list[str]
    evidence: dict[str, Any]


def build_card(task_id: str, task) -> SketchCard:
    """Build a sketch card for one task."""
    demos = task.train
    sigs = compute_task_signatures(demos)
    obs_result = observe_and_synthesize(demos)
    dims_result = dims_change_reconstruct(demos)
    same_dims = all(d.input.shape == d.output.shape for d in demos)

    color_roles = _color_roles_per_demo(demos)
    pixel_diff = _pixel_diff_analysis(demos)
    obj_struct = _object_structure(demos)
    dims_rel = _dims_relationship(demos)

    bg0 = _dominant_color(demos[0].input)
    spatial = _spatial_regularity(demos[0].input, bg0)

    decomposition = _infer_decomposition(demos, color_roles, obj_struct, pixel_diff, spatial, dims_rel)
    invariants = _infer_invariants(demos, color_roles, obj_struct, pixel_diff, sigs)
    construction = _infer_construction(demos, color_roles, obj_struct, pixel_diff, dims_rel, spatial)
    system_failure = _infer_system_failure(obs_result, dims_result, sigs, color_roles, pixel_diff, dims_rel)
    sketch = _infer_sketch(demos, decomposition, invariants, construction,
                           color_roles, obj_struct, pixel_diff, dims_rel, spatial)

    # Summaries
    input_shapes = [d.input.shape for d in demos]
    output_shapes = [d.output.shape for d in demos]
    input_range = f"{min(s[0] for s in input_shapes)}x{min(s[1] for s in input_shapes)}" \
                  f"..{max(s[0] for s in input_shapes)}x{max(s[1] for s in input_shapes)}"
    output_range = f"{min(s[0] for s in output_shapes)}x{min(s[1] for s in output_shapes)}" \
                   f"..{max(s[0] for s in output_shapes)}x{max(s[1] for s in output_shapes)}"

    bg_colors = [cr["bg_in"] for cr in color_roles]
    color_summary = f"bg={bg_colors}, roles_rotate={len(set(bg_colors)) > 1}"

    if pixel_diff:
        px_summary = f"max_changed={pixel_diff['max_changed']}, " \
                     f"same_transitions={pixel_diff['all_same_transition_set']}"
    else:
        px_summary = "n/a (dims change)"

    obj_summary = (
        f"objects={obj_struct['per_demo'][0]['n_input_objects']}->"
        f"{obj_struct['per_demo'][0]['n_output_objects']}, "
        f"singletons={obj_struct['any_singletons']}"
    )

    spatial_summary = (
        f"frame={spatial['has_frame']}, "
        f"h_sym={spatial['h_symmetric']}, v_sym={spatial['v_symmetric']}, "
        f"row_rep={spatial['row_repetition']}, col_rep={spatial['col_repetition']}"
    )

    evidence = {
        "obs_solved": obs_result.solved,
        "obs_candidates_tested": obs_result.candidates_tested,
        "obs_rule_count": len(obs_result.rules),
        "obs_rule_kinds": list(Counter(r.kind for r in obs_result.rules).items()),
        "dims_solved": dims_result.solved if dims_result else False,
        "dims_mode": dims_result.mode if dims_result else "n/a",
    }

    return SketchCard(
        task_id=task_id,
        same_dims=same_dims,
        input_shape_range=input_range,
        output_shape_range=output_range,
        n_demos=len(demos),
        decomposition=decomposition,
        invariants=invariants,
        construction=construction,
        system_failure=system_failure,
        sketch=sketch,
        color_roles_summary=color_summary,
        pixel_diff_summary=px_summary,
        object_summary=obj_summary,
        spatial_summary=spatial_summary,
        task_signatures=sorted(sigs),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def card_to_dict(card: SketchCard) -> dict:
    return asdict(card)


def card_to_markdown(card: SketchCard) -> str:
    lines = []
    lines.append(f"### `{card.task_id}`")
    lines.append(f"- **dims**: {'same' if card.same_dims else 'change'} "
                 f"| input {card.input_shape_range} → output {card.output_shape_range} "
                 f"| {card.n_demos} demos")
    lines.append(f"- **colors**: {card.color_roles_summary}")
    lines.append(f"- **objects**: {card.object_summary}")
    lines.append(f"- **spatial**: {card.spatial_summary}")
    lines.append(f"- **pixels**: {card.pixel_diff_summary}")

    lines.append(f"\n**Decomposition**:")
    for step in card.decomposition:
        lines.append(f"1. {step}")

    lines.append(f"\n**Invariants**:")
    for inv in card.invariants:
        lines.append(f"- {inv}")

    lines.append(f"\n**Construction**: {card.construction}")
    lines.append(f"\n**System failure**: {card.system_failure}")

    lines.append(f"\n**Sketch**:")
    lines.append(f"```")
    lines.append(card.sketch)
    lines.append(f"```")

    lines.append(f"\n<details><summary>evidence</summary>\n")
    lines.append(f"- signatures: `{', '.join(card.task_signatures)}`")
    for k, v in card.evidence.items():
        lines.append(f"- {k}: `{v}`")
    lines.append(f"\n</details>\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ds = get_dataset("v2-eval")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = ROOT / "logs" / f"arc2_sketch_cards_{timestamp}.json"
    out_md = ROOT / "logs" / f"arc2_sketch_cards_{timestamp}.md"

    cards: list[dict] = []
    md_sections: list[str] = []

    md_sections.append("# ARC-2 Sketch Cards\n")
    md_sections.append(
        "Per-task analysis: decomposition, invariants, construction, "
        "system failure mode, and proposed sketch.\n"
    )
    md_sections.append(
        "Each card describes one specific task. There are no bucket labels — "
        "shared tags appear only as secondary evidence.\n"
    )

    for idx, (task_id, task) in enumerate(iter_tasks(ds), start=1):
        try:
            card = build_card(task_id, task)
        except Exception as exc:
            print(f"[{idx:03d}] {task_id} ERROR: {exc}", flush=True)
            continue

        cards.append(card_to_dict(card))
        md_sections.append(card_to_markdown(card))

        print(
            f"[{idx:03d}] {task_id} "
            f"dims={'same' if card.same_dims else 'chng'} "
            f"decomp={len(card.decomposition)} "
            f"inv={len(card.invariants)} "
            f"obs_rules={card.evidence['obs_rule_count']}",
            flush=True,
        )

    report = {
        "dataset": ds.name,
        "generated": timestamp,
        "task_count": len(cards),
        "cards": cards,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text("\n".join(md_sections))

    print(f"\nWROTE {out_json}")
    print(f"WROTE {out_md}")
    print(f"{len(cards)} cards generated")


if __name__ == "__main__":
    main()
