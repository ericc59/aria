"""Multi-layer world model for grid transformation tasks — supporting tool.

NOT part of the canonical architecture (ComputationGraph + protocol).
Provides structured task understanding that can feed into graph proposal
or editor state encoding. Currently used by the stepper as a candidate
prioritizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.types import DemoPair, Grid


@dataclass
class PixelLayer:
    """Layer 0: raw pixel-level analysis."""
    dims: tuple[int, int]
    same_dims: bool
    total_pixels: int
    changed_pixels: int
    change_fraction: float
    bg_color: int
    input_palette: frozenset[int]
    output_palette: frozenset[int]
    added_colors: frozenset[int]      # in output but not input
    removed_colors: frozenset[int]    # in input but not output
    diff_positions: list[tuple[int, int]]  # (row, col) of changed pixels


@dataclass
class ObjectInfo:
    """Summary of one connected component."""
    color: int
    row: int
    col: int
    width: int
    height: int
    size: int  # pixel count


@dataclass
class ObjectLayer:
    """Layer 1: object-level analysis."""
    input_objects: list[ObjectInfo]
    output_objects: list[ObjectInfo]
    objects_added: int
    objects_removed: int
    same_count: bool
    input_colors: frozenset[int]
    output_colors: frozenset[int]


@dataclass
class RegionInfo:
    """One detected framed region."""
    frame_color: int
    row: int
    col: int
    height: int
    width: int


@dataclass
class StructureLayer:
    """Layer 2: structural analysis."""
    framed_regions: list[RegionInfo]
    has_frames: bool
    has_symmetry_h: bool
    has_symmetry_v: bool
    is_periodic_rows: bool
    is_periodic_cols: bool


@dataclass
class RoleLayer:
    """Layer 4: change role analysis."""
    pixels_added: int         # bg → non-bg
    pixels_removed: int       # non-bg → bg
    pixels_modified: int      # non-bg → different non-bg
    added_colors: frozenset[int]
    removed_colors: frozenset[int]
    changes_inside_frames: bool
    is_color_map: bool        # transformation is a pure color bijection
    color_map: dict[int, int] | None
    is_additive: bool         # only bg pixels changed
    is_subtractive: bool      # only non-bg pixels became bg


@dataclass
class WorldModel:
    """Multi-layer understanding of a task, built from all demo pairs."""
    task_id: str
    n_demos: int
    pixels: PixelLayer
    objects: ObjectLayer
    structure: StructureLayer
    roles: RoleLayer


def build_world_model(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> WorldModel:
    """Construct a world model from demo pairs.

    Analyzes all demos and merges observations into a single model.
    """
    from aria.decomposition import detect_bg, detect_framed_regions
    from aria.runtime.ops.selection import _find_objects

    if not demos:
        raise ValueError("need at least one demo")

    d0 = demos[0]
    bg = detect_bg(d0.input)
    same_dims = all(d.input.shape == d.output.shape for d in demos)

    # --- Pixel layer (aggregate across demos) ---
    total_changed = 0
    total_pixels = 0
    all_input_colors: set[int] = set()
    all_output_colors: set[int] = set()
    all_diff_positions: list[tuple[int, int]] = []

    for d in demos:
        total_pixels += d.input.size
        all_input_colors.update(int(c) for c in np.unique(d.input))
        all_output_colors.update(int(c) for c in np.unique(d.output))
        if same_dims:
            mask = d.input != d.output
            total_changed += int(mask.sum())
            all_diff_positions.extend((int(r), int(c)) for r, c in zip(*np.where(mask)))

    pixels = PixelLayer(
        dims=d0.input.shape,
        same_dims=same_dims,
        total_pixels=total_pixels,
        changed_pixels=total_changed,
        change_fraction=total_changed / total_pixels if total_pixels > 0 else 0,
        bg_color=bg,
        input_palette=frozenset(all_input_colors),
        output_palette=frozenset(all_output_colors),
        added_colors=frozenset(all_output_colors - all_input_colors),
        removed_colors=frozenset(all_input_colors - all_output_colors),
        diff_positions=all_diff_positions,
    )

    # --- Object layer ---
    all_inp_objs: list[ObjectInfo] = []
    all_out_objs: list[ObjectInfo] = []
    for d in demos:
        d_bg = detect_bg(d.input)
        for o in _find_objects(d.input):
            if o.color != d_bg:
                all_inp_objs.append(ObjectInfo(
                    color=o.color, row=o.bbox[1], col=o.bbox[0],
                    width=o.bbox[2], height=o.bbox[3], size=o.size,
                ))
        for o in _find_objects(d.output):
            if o.color != d_bg:
                all_out_objs.append(ObjectInfo(
                    color=o.color, row=o.bbox[1], col=o.bbox[0],
                    width=o.bbox[2], height=o.bbox[3], size=o.size,
                ))

    objects = ObjectLayer(
        input_objects=all_inp_objs,
        output_objects=all_out_objs,
        objects_added=max(0, len(all_out_objs) - len(all_inp_objs)),
        objects_removed=max(0, len(all_inp_objs) - len(all_out_objs)),
        same_count=len(all_inp_objs) == len(all_out_objs),
        input_colors=frozenset(o.color for o in all_inp_objs),
        output_colors=frozenset(o.color for o in all_out_objs),
    )

    # --- Structure layer ---
    regions_per_demo = []
    for d in demos:
        d_bg = detect_bg(d.input)
        try:
            regions = detect_framed_regions(d.input, d_bg)
            regions_per_demo.append(regions)
        except Exception:
            regions_per_demo.append([])

    all_regions = [
        RegionInfo(
            frame_color=r.frame_color, row=r.row, col=r.col,
            height=r.height, width=r.width,
        )
        for regions in regions_per_demo for r in regions
    ]

    # Symmetry check
    def _check_sym_h(grid: Grid) -> bool:
        return bool(np.array_equal(grid, grid[::-1]))

    def _check_sym_v(grid: Grid) -> bool:
        return bool(np.array_equal(grid, grid[:, ::-1]))

    has_sym_h = all(_check_sym_h(d.output) for d in demos) if same_dims else False
    has_sym_v = all(_check_sym_v(d.output) for d in demos) if same_dims else False

    structure = StructureLayer(
        framed_regions=all_regions,
        has_frames=len(all_regions) > 0,
        has_symmetry_h=has_sym_h,
        has_symmetry_v=has_sym_v,
        is_periodic_rows=False,  # computed below if needed
        is_periodic_cols=False,
    )

    # --- Role layer ---
    total_added = 0
    total_removed = 0
    total_modified = 0
    added_colors: set[int] = set()
    removed_colors: set[int] = set()
    changes_inside = False

    # Color map detection
    color_map: dict[int, int] = {}
    is_color_map = same_dims

    for d in demos:
        d_bg = detect_bg(d.input)
        if same_dims:
            diff = d.input != d.output
            for r, c in zip(*np.where(diff)):
                ic = int(d.input[r, c])
                oc = int(d.output[r, c])
                if ic == d_bg:
                    total_added += 1
                    added_colors.add(oc)
                elif oc == d_bg:
                    total_removed += 1
                    removed_colors.add(ic)
                else:
                    total_modified += 1

                if is_color_map:
                    if ic in color_map:
                        if color_map[ic] != oc:
                            is_color_map = False
                    else:
                        color_map[ic] = oc

    # Also add unchanged pixels to color map
    if is_color_map and same_dims:
        for d in demos:
            same_mask = d.input == d.output
            for r, c in zip(*np.where(same_mask)):
                ic = int(d.input[r, c])
                if ic in color_map:
                    if color_map[ic] != ic:
                        is_color_map = False
                        break
                else:
                    color_map[ic] = ic
            if not is_color_map:
                break

    if is_color_map and all(k == v for k, v in color_map.items()):
        is_color_map = False  # identity is not interesting

    # Check if changes are inside framed regions
    if same_dims and all_regions:
        for d in demos:
            diff = d.input != d.output
            for ri in all_regions:
                region_mask = np.zeros_like(diff)
                r0, c0 = ri.row, ri.col
                r1, c1 = r0 + ri.height, c0 + ri.width
                if r1 <= diff.shape[0] and c1 <= diff.shape[1]:
                    region_mask[r0:r1, c0:c1] = True
                    if np.any(diff & region_mask):
                        changes_inside = True
                        break
            if changes_inside:
                break

    is_additive = same_dims and total_added > 0 and total_removed == 0 and total_modified == 0
    is_subtractive = same_dims and total_removed > 0 and total_added == 0 and total_modified == 0

    roles = RoleLayer(
        pixels_added=total_added,
        pixels_removed=total_removed,
        pixels_modified=total_modified,
        added_colors=frozenset(added_colors),
        removed_colors=frozenset(removed_colors),
        changes_inside_frames=changes_inside,
        is_color_map=is_color_map,
        color_map=color_map if is_color_map else None,
        is_additive=is_additive,
        is_subtractive=is_subtractive,
    )

    return WorldModel(
        task_id=task_id,
        n_demos=len(demos),
        pixels=pixels,
        objects=objects,
        structure=structure,
        roles=roles,
    )
