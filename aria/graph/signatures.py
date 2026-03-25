"""Cheap structural task signatures for retrieval and learning."""

from __future__ import annotations

from collections import Counter

import numpy as np

from aria.graph.extract import extract
from aria.types import DemoPair, GlobalSymmetry, MatchRel, SceneRole, TopoRel


def compute_task_signatures(demos: tuple[DemoPair, ...]) -> frozenset[str]:
    """Return a small set of structural tags describing a task."""
    signatures: set[str] = set()
    if not demos:
        return frozenset()

    same_dims = all(demo.input.shape == demo.output.shape for demo in demos)
    signatures.add("dims:same" if same_dims else "dims:different")

    if not same_dims:
        row_ratios = [
            demo.output.shape[0] / max(demo.input.shape[0], 1)
            for demo in demos
        ]
        col_ratios = [
            demo.output.shape[1] / max(demo.input.shape[1], 1)
            for demo in demos
        ]
        if _all_close(row_ratios) and _all_close(col_ratios):
            signatures.add("size:multiplicative")
            row_ratio = row_ratios[0]
            col_ratio = col_ratios[0]
            if _is_int_like(row_ratio):
                signatures.add(f"size:rows_x{int(round(row_ratio))}")
            if _is_int_like(col_ratio):
                signatures.add(f"size:cols_x{int(round(col_ratio))}")
            if _is_int_like(row_ratio) and _is_int_like(col_ratio) and round(row_ratio) == round(col_ratio):
                signatures.add(f"size:scale_{int(round(row_ratio))}x")

        row_deltas = [
            int(demo.output.shape[0] - demo.input.shape[0])
            for demo in demos
        ]
        col_deltas = [
            int(demo.output.shape[1] - demo.input.shape[1])
            for demo in demos
        ]
        if _all_equal(row_deltas) and _all_equal(col_deltas):
            signatures.add("size:additive")
            signatures.add(f"size:rows_delta_{row_deltas[0]}")
            signatures.add(f"size:cols_delta_{col_deltas[0]}")

        ratios = [
            (demo.output.shape[0] * demo.output.shape[1])
            / max(demo.input.shape[0] * demo.input.shape[1], 1)
            for demo in demos
        ]
        avg_ratio = float(np.mean(ratios))
        if avg_ratio < 0.5:
            signatures.add("size:shrink")
        elif avg_ratio > 2.0:
            signatures.add("size:grow")
        else:
            signatures.add("size:reshape")

    output_shapes = {demo.output.shape for demo in demos}
    if len(output_shapes) == 1:
        signatures.add("size:fixed_output_shape")

    all_input_colors: set[int] = set()
    all_output_colors: set[int] = set()
    for demo in demos:
        all_input_colors |= {int(value) for value in np.unique(demo.input)}
        all_output_colors |= {int(value) for value in np.unique(demo.output)}

    if all_output_colors <= all_input_colors:
        signatures.add("color:palette_subset")
    if all_output_colors == all_input_colors:
        signatures.add("color:palette_same")
    if all_output_colors - all_input_colors:
        signatures.add("color:new_in_output")
    color_count = len(all_input_colors | all_output_colors)
    if color_count <= 3:
        signatures.add("color:few_colors")
    elif color_count >= 7:
        signatures.add("color:many_colors")

    if same_dims:
        additive = True
        bg_preserved = True
        for demo in demos:
            input_graph = extract(demo.input)
            bg = input_graph.context.bg_color
            non_bg = demo.input != bg
            bg_mask = demo.input == bg
            if additive and not np.all(demo.output[non_bg] == demo.input[non_bg]):
                additive = False
            if bg_preserved and not np.all(demo.output[bg_mask] == bg):
                bg_preserved = False
        if additive:
            signatures.add("change:additive")
        if bg_preserved:
            signatures.add("change:bg_preserved")

    input_graphs = [extract(demo.input) for demo in demos]
    output_graphs = [extract(demo.output) for demo in demos]
    required_count = max(1, (len(demos) // 2) + 1)

    _add_majority_symmetry_signatures(signatures, "input", input_graphs, required_count)
    _add_majority_symmetry_signatures(signatures, "output", output_graphs, required_count)

    obj_bucket_counts = Counter(_object_bucket(len(graph.objects)) for graph in input_graphs)
    if obj_bucket_counts:
        signatures.add(max(obj_bucket_counts.items(), key=lambda item: (item[1], item[0]))[0])

    role_counts = Counter()
    rel_counts = Counter()
    partition_tag_counts = Counter()
    legend_tag_counts = Counter()

    for graph in input_graphs:
        seen_role_tags: set[str] = set()
        for binding in graph.roles:
            if binding.role == SceneRole.SEPARATOR:
                seen_role_tags.add("role:has_separator")
            elif binding.role == SceneRole.LEGEND:
                seen_role_tags.add("role:has_legend")
            elif binding.role == SceneRole.FRAME:
                seen_role_tags.add("role:has_frame")
            elif binding.role == SceneRole.MARKER:
                seen_role_tags.add("role:has_marker")
        role_counts.update(seen_role_tags)

        seen_rel_tags: set[str] = set()
        for edge in graph.relations:
            if MatchRel.SAME_SHAPE in edge.match:
                seen_rel_tags.add("rel:same_shape_pair")
            if MatchRel.SAME_COLOR in edge.match:
                seen_rel_tags.add("rel:same_color_pair")
            if TopoRel.CONTAINS in edge.topo:
                seen_rel_tags.add("rel:contains_pair")
            if TopoRel.ADJACENT in edge.topo:
                seen_rel_tags.add("rel:adjacent_pair")
        rel_counts.update(seen_rel_tags)

        if graph.partition is not None:
            partition = graph.partition
            tags = {
                "partition:has_separator_grid",
                f"partition:cell_rows_{partition.n_rows}",
                f"partition:cell_cols_{partition.n_cols}",
            }
            if partition.is_uniform_partition:
                tags.add("partition:uniform_cells")

            palettes = {cell.palette for cell in partition.cells}
            obj_counts = {cell.obj_count for cell in partition.cells}
            if len(palettes) > 1:
                tags.add("partition:cell_palette_varies")
            if len(obj_counts) > 1:
                tags.add("partition:cell_object_count_varies")
            if not same_dims:
                tags.add("partition:cell_summary_task")
            partition_tag_counts.update(tags)

        if graph.legend is not None:
            legend_tag_counts.update({
                "legend:present",
                f"legend:edge:{graph.legend.edge}",
                f"legend:entries:{len(graph.legend.entries)}",
            })

    for tag, count in role_counts.items():
        if count >= required_count:
            signatures.add(tag)

    for tag, count in rel_counts.items():
        if count >= required_count:
            signatures.add(tag)

    for tag, count in partition_tag_counts.items():
        if count >= required_count:
            signatures.add(tag)

    for tag, count in legend_tag_counts.items():
        if count >= required_count:
            signatures.add(tag)

    return frozenset(signatures)


def _extend_symmetry_signatures(
    signatures: set[str],
    prefix: str,
    symmetries: frozenset[GlobalSymmetry],
) -> None:
    if GlobalSymmetry.GLOBAL_REFL in symmetries:
        signatures.add(f"sym:{prefix}_reflective")
    if GlobalSymmetry.GLOBAL_ROT in symmetries:
        signatures.add(f"sym:{prefix}_rotational")
    if GlobalSymmetry.PERIODIC in symmetries:
        signatures.add(f"sym:{prefix}_periodic")


def _add_majority_symmetry_signatures(
    signatures: set[str],
    prefix: str,
    graphs,
    required_count: int,
) -> None:
    counts = Counter()
    for graph in graphs:
        tags: set[str] = set()
        _extend_symmetry_signatures(tags, prefix, graph.context.symmetry)
        counts.update(tags)
    for tag, count in counts.items():
        if count >= required_count:
            signatures.add(tag)


def _object_bucket(count: int) -> str:
    if count == 0:
        return "obj:none"
    if count == 1:
        return "obj:single"
    if count <= 5:
        return "obj:few"
    return "obj:many"


def _all_close(values: list[float], tol: float = 1e-9) -> bool:
    if not values:
        return True
    ref = values[0]
    return all(abs(value - ref) < tol for value in values)


def _all_equal(values: list[int]) -> bool:
    if not values:
        return True
    return all(value == values[0] for value in values)


def _is_int_like(value: float, tol: float = 1e-9) -> bool:
    return abs(value - round(value)) < tol
