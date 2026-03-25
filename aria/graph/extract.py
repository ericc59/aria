"""Top-level state graph extraction orchestrator.

Runs all analysis passes (labeling, background detection, shape
classification, symmetry, relations, zones) to produce a StateGraph.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from aria.types import (
    Delta,
    GlobalSymmetry,
    Grid,
    GridContext,
    ObjectNode,
    StateGraph,
)
from aria.graph.background import detect_bg
from aria.graph.cc_label import label_4conn
from aria.graph.delta import compute_delta
from aria.graph.legend import detect_legend
from aria.graph.relations import compute_relations
from aria.graph.partition import detect_partition
from aria.graph.roles import infer_roles
from aria.graph.shapes import classify_shape
from aria.graph.symmetry import detect_global_symmetry, detect_obj_symmetry
from aria.graph.zones import detect_tiling


def extract(grid: Grid) -> StateGraph:
    """Extract a full StateGraph from a grid.

    Pipeline:
    1. Detect background color.
    2. Label connected components (4-conn, ignoring background).
    3. Classify shapes and detect symmetry for each object.
    4. Compute pairwise relations.
    5. Detect global properties (tiling, symmetry, palette).
    6. Assemble the StateGraph.

    Parameters
    ----------
    grid : Grid
        2D array of color values (0-9).

    Returns
    -------
    StateGraph
        Complete state graph for the input grid.
    """
    bg = detect_bg(grid)

    # Label foreground objects
    raw_objects = label_4conn(grid, ignore_color=bg)

    # Enrich each object with shape and symmetry
    objects: list[ObjectNode] = []
    for obj in raw_objects:
        shape = classify_shape(obj.mask)
        sym = detect_obj_symmetry(obj.mask)
        # Replace placeholder fields using __class__ constructor
        # (ObjectNode is frozen, so we rebuild)
        enriched = ObjectNode(
            id=obj.id,
            color=obj.color,
            mask=obj.mask,
            bbox=obj.bbox,
            shape=shape,
            symmetry=sym,
            size=obj.size,
        )
        objects.append(enriched)

    partition = detect_partition(grid, background=bg)
    legend = detect_legend(grid, background=bg, partition=partition)

    # Compute relations
    relations = compute_relations(objects)
    roles = infer_roles(tuple(objects), partition, legend)

    # Global context
    tiling = detect_tiling(grid)
    global_sym = detect_global_symmetry(grid)
    palette = frozenset(int(c) for c in np.unique(grid))

    context = GridContext(
        dims=(int(grid.shape[0]), int(grid.shape[1])),
        bg_color=bg,
        is_tiled=tiling,
        symmetry=global_sym,
        palette=palette,
        obj_count=len(objects),
    )

    return StateGraph(
        objects=tuple(objects),
        relations=tuple(relations),
        context=context,
        grid=grid,
        partition=partition,
        legend=legend,
        roles=roles,
    )


def extract_with_delta(
    in_grid: Grid,
    out_grid: Grid,
) -> tuple[StateGraph, StateGraph, Delta]:
    """Extract StateGraphs for both grids and compute the delta.

    Parameters
    ----------
    in_grid : Grid
        Input grid.
    out_grid : Grid
        Output grid.

    Returns
    -------
    tuple[StateGraph, StateGraph, Delta]
        (input_sg, output_sg, delta)
    """
    sg_in = extract(in_grid)
    sg_out = extract(out_grid)
    delta = compute_delta(sg_in, sg_out)
    return sg_in, sg_out, delta
