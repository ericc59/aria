"""Root primitive set for the next-generation solver.

Each primitive is a pure function: typed inputs -> typed output.
No primitive encodes a task family. Each is a single atomic operation
that appears frequently across many ARC tasks.

Primitives are grouped by role:
- SELECT: pick structural units from a grid
- EXTRACT: pull out subgrids, masks, sequences, properties
- DETECT: find regularity, symmetry, anomalies, relations
- COMPUTE: derive offsets, correspondences, color maps
- RENDER: recolor, fill, stamp, move, copy, compose

Each primitive is registered with its name, input types, output type,
and implementation function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from aria.decomposition import RawObject, detect_bg, extract_objects
from aria.ngs.ir import VType
from aria.types import Grid


# ---------------------------------------------------------------------------
# Primitive registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimDef:
    name: str
    input_types: tuple[VType, ...]
    output_type: VType
    impl: Callable[..., Any]
    doc: str = ""


_PRIMS: dict[str, PrimDef] = {}


def register_prim(pdef: PrimDef) -> PrimDef:
    _PRIMS[pdef.name] = pdef
    return pdef


def get_prim(name: str) -> PrimDef:
    return _PRIMS[name]


def all_prims() -> dict[str, PrimDef]:
    return dict(_PRIMS)


def execute_prim(name: str, *args: Any, **params: Any) -> Any:
    pdef = _PRIMS[name]
    return pdef.impl(*args, **params)


# ---------------------------------------------------------------------------
# SELECT primitives — pick structural units
# ---------------------------------------------------------------------------

def _select_objects(grid: Grid, bg: int = 0) -> list[RawObject]:
    """Extract all non-bg connected components."""
    return extract_objects(grid, bg, connectivity=4)

register_prim(PrimDef(
    "select_objects", (VType.GRID, VType.COLOR), VType.OBJECTS,
    _select_objects, "extract connected components excluding bg",
))


def _select_objects_8conn(grid: Grid, bg: int = 0) -> list[RawObject]:
    """Extract all non-bg connected components with 8-connectivity."""
    return extract_objects(grid, bg, connectivity=8)

register_prim(PrimDef(
    "select_objects_8conn", (VType.GRID, VType.COLOR), VType.OBJECTS,
    _select_objects_8conn, "extract connected components (8-conn) excluding bg",
))


def _select_by_color(objects: list[RawObject], color: int) -> list[RawObject]:
    """Filter objects by color."""
    return [o for o in objects if o.color == color]

register_prim(PrimDef(
    "select_by_color", (VType.OBJECTS, VType.COLOR), VType.OBJECTS,
    _select_by_color, "filter objects to those with given color",
))


def _select_by_size(objects: list[RawObject], min_size: int = 1, max_size: int = 999) -> list[RawObject]:
    """Filter objects by pixel count."""
    return [o for o in objects if min_size <= o.size <= max_size]

register_prim(PrimDef(
    "select_by_size", (VType.OBJECTS, VType.INT, VType.INT), VType.OBJECTS,
    _select_by_size, "filter objects by size range",
))


def _select_singletons(objects: list[RawObject]) -> list[RawObject]:
    """Select singleton (1-pixel) objects."""
    return [o for o in objects if o.is_singleton]

register_prim(PrimDef(
    "select_singletons", (VType.OBJECTS,), VType.OBJECTS,
    _select_singletons, "filter to singleton objects",
))


# ---------------------------------------------------------------------------
# EXTRACT primitives — pull out subgrids, masks, properties
# ---------------------------------------------------------------------------

def _extract_subgrid(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    """Extract subgrid at bbox (r0, c0, r1, c1) inclusive."""
    r0, c0, r1, c1 = bbox
    return grid[r0:r1 + 1, c0:c1 + 1].copy()

register_prim(PrimDef(
    "extract_subgrid", (VType.GRID, VType.BBOX), VType.GRID,
    _extract_subgrid, "crop grid to bbox",
))


def _extract_object_mask(obj: RawObject, grid_shape: tuple[int, int]) -> np.ndarray:
    """Get full-grid boolean mask for one object."""
    mask = np.zeros(grid_shape, dtype=bool)
    r0, c0 = obj.row, obj.col
    mask[r0:r0 + obj.bbox_h, c0:c0 + obj.bbox_w] |= obj.mask
    return mask

register_prim(PrimDef(
    "extract_object_mask", (VType.OBJECT, VType.BBOX), VType.MASK,
    _extract_object_mask, "get full-grid boolean mask for an object",
))


def _extract_diff_mask(grid_a: Grid, grid_b: Grid) -> np.ndarray:
    """Positions where two same-shape grids differ."""
    assert grid_a.shape == grid_b.shape
    return grid_a != grid_b

register_prim(PrimDef(
    "extract_diff_mask", (VType.GRID, VType.GRID), VType.MASK,
    _extract_diff_mask, "boolean mask of differing positions",
))


def _extract_color_at(grid: Grid, r: int, c: int) -> int:
    return int(grid[r, c])

register_prim(PrimDef(
    "extract_color_at", (VType.GRID, VType.INT, VType.INT), VType.COLOR,
    _extract_color_at, "get color at position",
))


def _extract_row(grid: Grid, r: int) -> np.ndarray:
    return grid[r, :].copy()

register_prim(PrimDef(
    "extract_row", (VType.GRID, VType.INT), VType.SEQ,
    _extract_row, "get one row as 1D array",
))


def _extract_col(grid: Grid, c: int) -> np.ndarray:
    return grid[:, c].copy()

register_prim(PrimDef(
    "extract_col", (VType.GRID, VType.INT), VType.SEQ,
    _extract_col, "get one column as 1D array",
))


def _extract_palette(grid: Grid) -> set[int]:
    """Unique colors in grid."""
    return set(int(v) for v in np.unique(grid))

register_prim(PrimDef(
    "extract_palette", (VType.GRID,), VType.INT,  # returns set but typed as INT for simplicity
    _extract_palette, "set of unique colors",
))


def _extract_bbox(obj: RawObject) -> tuple[int, int, int, int]:
    """Bounding box of an object (r0, c0, r1, c1)."""
    return (obj.row, obj.col, obj.row + obj.bbox_h - 1, obj.col + obj.bbox_w - 1)

register_prim(PrimDef(
    "extract_bbox", (VType.OBJECT,), VType.BBOX,
    _extract_bbox, "bounding box of an object",
))


# ---------------------------------------------------------------------------
# DETECT primitives — find regularity, symmetry, anomalies
# ---------------------------------------------------------------------------

def _detect_symmetry_h(grid: Grid) -> bool:
    """Is grid symmetric under horizontal reflection?"""
    return bool(np.array_equal(grid, grid[:, ::-1]))

register_prim(PrimDef(
    "detect_symmetry_h", (VType.GRID,), VType.BOOL,
    _detect_symmetry_h, "horizontal reflection symmetry",
))


def _detect_symmetry_v(grid: Grid) -> bool:
    """Is grid symmetric under vertical reflection?"""
    return bool(np.array_equal(grid, grid[::-1, :]))

register_prim(PrimDef(
    "detect_symmetry_v", (VType.GRID,), VType.BOOL,
    _detect_symmetry_v, "vertical reflection symmetry",
))


def _detect_period_row(seq: np.ndarray) -> int:
    """Detect smallest period in a 1D sequence. Returns len if aperiodic."""
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if n % p != 0:
            continue
        tile = seq[:p]
        if all(np.array_equal(seq[i:i + p], tile) for i in range(p, n, p)):
            return p
    return n

register_prim(PrimDef(
    "detect_period", (VType.SEQ,), VType.INT,
    _detect_period_row, "smallest period of a 1D sequence",
))


def _detect_anomaly_positions(seq: np.ndarray, period: int) -> list[int]:
    """Positions in seq that violate the periodic pattern."""
    if period >= len(seq):
        return []
    tile = seq[:period].copy()
    anomalies = []
    for i in range(len(seq)):
        if seq[i] != tile[i % period]:
            anomalies.append(i)
    return anomalies

register_prim(PrimDef(
    "detect_anomaly_positions", (VType.SEQ, VType.INT), VType.INT,  # list[int]
    _detect_anomaly_positions, "positions that violate periodic pattern",
))


def _detect_enclosed_mask(grid: Grid, bg: int = 0) -> np.ndarray:
    """Find bg cells that are fully enclosed by non-bg cells."""
    non_bg = grid != bg
    # Flood fill from edges
    edge_connected = np.zeros_like(non_bg)
    rows, cols = grid.shape
    seed = np.zeros((rows + 2, cols + 2), dtype=bool)
    seed[0, :] = True
    seed[-1, :] = True
    seed[:, 0] = True
    seed[:, -1] = True
    padded = np.pad(~non_bg, 1, constant_values=False)
    # BFS from edges through bg cells
    from collections import deque
    visited = np.zeros_like(padded, dtype=bool)
    q = deque()
    for r in range(padded.shape[0]):
        for c in range(padded.shape[1]):
            if seed[r, c] and padded[r, c]:
                q.append((r, c))
                visited[r, c] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1]:
                if padded[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    # Enclosed = bg cells not reached from edges
    inner = visited[1:-1, 1:-1]
    enclosed = (~non_bg) & (~inner)
    return enclosed

register_prim(PrimDef(
    "detect_enclosed", (VType.GRID, VType.COLOR), VType.MASK,
    _detect_enclosed_mask, "mask of bg cells enclosed by non-bg",
))


def _detect_adjacency(obj_a: RawObject, obj_b: RawObject,
                       grid_shape: tuple[int, int]) -> bool:
    """Are two objects 4-adjacent (touching but not overlapping)?"""
    mask_a = np.zeros(grid_shape, dtype=bool)
    mask_a[obj_a.row:obj_a.row + obj_a.bbox_h,
           obj_a.col:obj_a.col + obj_a.bbox_w] |= obj_a.mask
    # Dilate a by 1
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = ndimage.binary_dilation(mask_a, structure=struct)
    mask_b = np.zeros(grid_shape, dtype=bool)
    mask_b[obj_b.row:obj_b.row + obj_b.bbox_h,
           obj_b.col:obj_b.col + obj_b.bbox_w] |= obj_b.mask
    return bool(np.any(dilated & mask_b))

register_prim(PrimDef(
    "detect_adjacency", (VType.OBJECT, VType.OBJECT, VType.BBOX), VType.BOOL,
    _detect_adjacency, "are two objects 4-adjacent?",
))


# ---------------------------------------------------------------------------
# COMPUTE primitives — derive mappings, offsets, correspondences
# ---------------------------------------------------------------------------

def _compute_color_map(grid_a: Grid, grid_b: Grid) -> dict[int, int] | None:
    """Derive a consistent per-pixel color map from a->b.

    Returns None if any position maps one input color to multiple output colors.
    """
    assert grid_a.shape == grid_b.shape
    cmap: dict[int, int] = {}
    for r in range(grid_a.shape[0]):
        for c in range(grid_a.shape[1]):
            a, b = int(grid_a[r, c]), int(grid_b[r, c])
            if a in cmap:
                if cmap[a] != b:
                    return None
            else:
                cmap[a] = b
    return cmap

register_prim(PrimDef(
    "compute_color_map", (VType.GRID, VType.GRID), VType.COLOR_MAP,
    _compute_color_map, "consistent per-pixel color mapping a->b",
))


def _compute_offset(obj_a: RawObject, obj_b: RawObject) -> tuple[int, int]:
    """Offset from a's top-left to b's top-left."""
    return (obj_b.row - obj_a.row, obj_b.col - obj_a.col)

register_prim(PrimDef(
    "compute_offset", (VType.OBJECT, VType.OBJECT), VType.OFFSET,
    _compute_offset, "positional offset between two objects",
))


def _compute_object_correspondence(
    objects_in: list[RawObject],
    objects_out: list[RawObject],
) -> list[tuple[RawObject, RawObject]] | None:
    """Match input objects to output objects by shape+position.

    Simple greedy: for each output obj, find the input obj with
    same shape mask and minimum positional distance.
    Returns None if no valid matching exists.
    """
    if not objects_in or not objects_out:
        return None
    used: set[int] = set()
    pairs: list[tuple[RawObject, RawObject]] = []
    for out_obj in objects_out:
        best_idx = -1
        best_dist = float("inf")
        for i, in_obj in enumerate(objects_in):
            if i in used:
                continue
            if in_obj.mask.shape != out_obj.mask.shape:
                continue
            if not np.array_equal(in_obj.mask, out_obj.mask):
                continue
            dist = abs(in_obj.row - out_obj.row) + abs(in_obj.col - out_obj.col)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            pairs.append((objects_in[best_idx], out_obj))
    if len(pairs) != len(objects_out):
        return None
    return pairs

register_prim(PrimDef(
    "compute_correspondence", (VType.OBJECTS, VType.OBJECTS), VType.CORRESPONDENCE,
    _compute_object_correspondence, "match objects by shape, minimize position distance",
))


# ---------------------------------------------------------------------------
# RENDER primitives — apply transformations to produce output
# ---------------------------------------------------------------------------

def _render_color_map(grid: Grid, cmap: dict[int, int]) -> Grid:
    """Apply a color map to every pixel."""
    out = grid.copy()
    for from_c, to_c in cmap.items():
        out[grid == from_c] = to_c
    return out

register_prim(PrimDef(
    "render_color_map", (VType.GRID, VType.COLOR_MAP), VType.GRID,
    _render_color_map, "recolor grid by color map",
))


def _render_fill_mask(grid: Grid, mask: np.ndarray, color: int) -> Grid:
    """Fill masked positions with a color."""
    out = grid.copy()
    out[mask] = color
    return out

register_prim(PrimDef(
    "render_fill_mask", (VType.GRID, VType.MASK, VType.COLOR), VType.GRID,
    _render_fill_mask, "fill masked positions with color",
))


def _render_fill_enclosed(grid: Grid, fill_color: int, bg: int = 0) -> Grid:
    """Fill enclosed bg regions with fill_color."""
    mask = _detect_enclosed_mask(grid, bg)
    out = grid.copy()
    out[mask] = fill_color
    return out

register_prim(PrimDef(
    "render_fill_enclosed", (VType.GRID, VType.COLOR, VType.COLOR), VType.GRID,
    _render_fill_enclosed, "fill enclosed bg regions with color",
))


def _render_stamp(canvas: Grid, stamp: Grid, r: int, c: int, bg: int = 0) -> Grid:
    """Stamp a subgrid onto canvas, treating bg in stamp as transparent."""
    out = canvas.copy()
    sh, sw = stamp.shape
    for dr in range(sh):
        for dc in range(sw):
            if stamp[dr, dc] != bg:
                tr, tc = r + dr, c + dc
                if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                    out[tr, tc] = stamp[dr, dc]
    return out

register_prim(PrimDef(
    "render_stamp", (VType.GRID, VType.GRID, VType.INT, VType.INT, VType.COLOR), VType.GRID,
    _render_stamp, "stamp subgrid onto canvas (bg transparent)",
))


def _render_move_object(grid: Grid, obj: RawObject, dr: int, dc: int, bg: int = 0) -> Grid:
    """Move an object by (dr, dc), erasing its old position."""
    out = grid.copy()
    # Erase old position
    for r in range(obj.bbox_h):
        for c in range(obj.bbox_w):
            if obj.mask[r, c]:
                out[obj.row + r, obj.col + c] = bg
    # Paint new position
    for r in range(obj.bbox_h):
        for c in range(obj.bbox_w):
            if obj.mask[r, c]:
                nr, nc = obj.row + r + dr, obj.col + c + dc
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    out[nr, nc] = obj.color
    return out

register_prim(PrimDef(
    "render_move_object", (VType.GRID, VType.OBJECT, VType.INT, VType.INT, VType.COLOR), VType.GRID,
    _render_move_object, "move object by offset, erase old position",
))


def _render_reflect_h(grid: Grid) -> Grid:
    """Reflect grid horizontally (left-right)."""
    return grid[:, ::-1].copy()

register_prim(PrimDef(
    "render_reflect_h", (VType.GRID,), VType.GRID,
    _render_reflect_h, "reflect grid left-right",
))


def _render_reflect_v(grid: Grid) -> Grid:
    """Reflect grid vertically (top-bottom)."""
    return grid[::-1, :].copy()

register_prim(PrimDef(
    "render_reflect_v", (VType.GRID,), VType.GRID,
    _render_reflect_v, "reflect grid top-bottom",
))


def _render_rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, k=-1).copy()

register_prim(PrimDef(
    "render_rotate_90", (VType.GRID,), VType.GRID,
    _render_rotate_90, "rotate grid 90° clockwise",
))


def _render_tile(grid: Grid, rows: int, cols: int) -> Grid:
    """Tile grid into rows x cols copies."""
    return np.tile(grid, (rows, cols)).copy()

register_prim(PrimDef(
    "render_tile", (VType.GRID, VType.INT, VType.INT), VType.GRID,
    _render_tile, "tile grid into rows x cols",
))


def _render_crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Crop grid to bbox."""
    return grid[r0:r1 + 1, c0:c1 + 1].copy()

register_prim(PrimDef(
    "render_crop", (VType.GRID, VType.INT, VType.INT, VType.INT, VType.INT), VType.GRID,
    _render_crop, "crop grid to bounding box",
))


def _render_upscale(grid: Grid, factor: int) -> Grid:
    """Upscale grid by integer factor."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1).copy()

register_prim(PrimDef(
    "render_upscale", (VType.GRID, VType.INT), VType.GRID,
    _render_upscale, "upscale grid by integer factor",
))


def _render_overlay(top: Grid, bottom: Grid, bg: int = 0) -> Grid:
    """Overlay top onto bottom, treating bg in top as transparent."""
    assert top.shape == bottom.shape
    out = bottom.copy()
    mask = top != bg
    out[mask] = top[mask]
    return out

register_prim(PrimDef(
    "render_overlay", (VType.GRID, VType.GRID, VType.COLOR), VType.GRID,
    _render_overlay, "overlay top onto bottom (bg=transparent)",
))


def _render_periodic_complete(seq: np.ndarray, period: int) -> np.ndarray:
    """Complete a periodic sequence by filling anomalies with the pattern."""
    if period >= len(seq) or period < 1:
        return seq.copy()
    tile = seq[:period].copy()
    out = seq.copy()
    for i in range(len(seq)):
        out[i] = tile[i % period]
    return out

register_prim(PrimDef(
    "render_periodic_complete", (VType.SEQ, VType.INT), VType.SEQ,
    _render_periodic_complete, "complete sequence to match periodic pattern",
))


def _render_new_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    """Create a new grid filled with a color."""
    return np.full((rows, cols), fill, dtype=np.uint8)

register_prim(PrimDef(
    "render_new_grid", (VType.INT, VType.INT, VType.COLOR), VType.GRID,
    _render_new_grid, "create new grid filled with color",
))
