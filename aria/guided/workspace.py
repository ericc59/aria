"""Structured workspace representation for one task.

The workspace exposes all structural information a search agent needs
to construct an explanation, serializable for model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from aria.types import Grid


# ---------------------------------------------------------------------------
# Core workspace
# ---------------------------------------------------------------------------

@dataclass
class ObjectInfo:
    """One connected component."""
    oid: int
    color: int
    row: int
    col: int
    height: int
    width: int
    size: int
    mask: np.ndarray          # bool within bbox
    is_singleton: bool

    def serialize(self) -> dict:
        return {
            "oid": self.oid,
            "color": self.color,
            "row": self.row,
            "col": self.col,
            "h": self.height,
            "w": self.width,
            "size": self.size,
            "singleton": self.is_singleton,
        }


@dataclass
class Relation:
    """A relation between two objects."""
    src: int  # oid
    dst: int  # oid
    rel_type: str  # "adjacent", "contains", "aligned_h", "aligned_v", "same_color", "same_shape"

    def serialize(self) -> dict:
        return {"src": self.src, "dst": self.dst, "type": self.rel_type}


@dataclass
class ResidualUnit:
    """One piece of the output that needs explanation."""
    uid: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray           # bool in grid coords
    n_pixels: int
    change_type: str           # "add", "delete", "recolor", "mixed"
    output_colors: tuple[int, ...]
    input_colors: tuple[int, ...]
    adjacent_colors: tuple[int, ...]

    def serialize(self) -> dict:
        return {
            "uid": self.uid,
            "bbox": list(self.bbox),
            "n_pixels": self.n_pixels,
            "change_type": self.change_type,
            "output_colors": list(self.output_colors),
            "input_colors": list(self.input_colors),
            "adjacent_colors": list(self.adjacent_colors),
        }


@dataclass
class Workspace:
    """Complete structured workspace for one demo pair."""
    # Grid data
    input_grid: Grid
    output_grid: Grid
    rows: int
    cols: int
    bg: int
    same_shape: bool

    # Preservation
    preserved_mask: np.ndarray   # bool
    residual_mask: np.ndarray    # bool
    n_preserved: int
    n_residual: int

    # Structural decomposition
    objects: list[ObjectInfo] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    residual_units: list[ResidualUnit] = field(default_factory=list)

    # Palette
    palette: tuple[int, ...] = ()

    def serialize(self) -> dict:
        """Serialize to dict for model input."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "bg": self.bg,
            "same_shape": self.same_shape,
            "n_preserved": self.n_preserved,
            "n_residual": self.n_residual,
            "preservation_ratio": self.n_preserved / max(1, self.n_preserved + self.n_residual),
            "palette": list(self.palette),
            "n_objects": len(self.objects),
            "objects": [o.serialize() for o in self.objects],
            "relations": [r.serialize() for r in self.relations],
            "residual_units": [u.serialize() for u in self.residual_units],
        }


# ---------------------------------------------------------------------------
# Build workspace from a demo pair
# ---------------------------------------------------------------------------

def build_workspace(inp: Grid, out: Grid) -> Workspace:
    """Build a complete workspace from one demo pair."""
    rows, cols = inp.shape
    bg = _detect_bg(inp)
    same_shape = inp.shape == out.shape

    # Preservation
    if same_shape:
        preserved = inp == out
        residual = ~preserved
    else:
        preserved = np.zeros((rows, cols), dtype=bool)
        residual = np.ones(out.shape, dtype=bool)

    # Objects
    objects = _extract_objects(inp, bg)

    # Relations
    relations = _compute_relations(objects, inp.shape)

    # Residual units
    residual_units = _decompose_residual(inp, out, residual, bg) if same_shape else []

    palette = tuple(sorted(set(int(v) for v in np.unique(inp))))

    return Workspace(
        input_grid=inp,
        output_grid=out,
        rows=rows,
        cols=cols,
        bg=bg,
        same_shape=same_shape,
        preserved_mask=preserved,
        residual_mask=residual,
        n_preserved=int(np.sum(preserved)),
        n_residual=int(np.sum(residual)),
        objects=objects,
        relations=relations,
        residual_units=residual_units,
        palette=palette,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_bg(grid: Grid) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _extract_objects(grid: Grid, bg: int) -> list[ObjectInfo]:
    rows, cols = grid.shape
    objects = []
    oid = 0
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary, structure=struct4)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labeled == label_id)
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            mask = labeled[r0:r1 + 1, c0:c1 + 1] == label_id
            size = int(mask.sum())
            objects.append(ObjectInfo(
                oid=oid,
                color=color,
                row=r0, col=c0,
                height=r1 - r0 + 1,
                width=c1 - c0 + 1,
                size=size,
                mask=mask,
                is_singleton=(size == 1),
            ))
            oid += 1
    return objects


def _compute_relations(objects: list[ObjectInfo], grid_shape: tuple[int, int]) -> list[Relation]:
    rels = []
    rows, cols = grid_shape
    # Build full masks for adjacency
    full_masks = {}
    for obj in objects:
        m = np.zeros((rows, cols), dtype=bool)
        m[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] |= obj.mask
        full_masks[obj.oid] = m

    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for i, a in enumerate(objects):
        dil_a = ndimage.binary_dilation(full_masks[a.oid], structure=struct4)
        for b in objects[i + 1:]:
            # Adjacent?
            if np.any(dil_a & full_masks[b.oid]):
                rels.append(Relation(a.oid, b.oid, "adjacent"))
            # Contains?
            if (a.row <= b.row and a.col <= b.col and
                    a.row + a.height >= b.row + b.height and
                    a.col + a.width >= b.col + b.width and a.size > b.size):
                rels.append(Relation(a.oid, b.oid, "contains"))
            elif (b.row <= a.row and b.col <= a.col and
                  b.row + b.height >= a.row + a.height and
                  b.col + b.width >= a.col + a.width and b.size > a.size):
                rels.append(Relation(b.oid, a.oid, "contains"))
            # Same color
            if a.color == b.color:
                rels.append(Relation(a.oid, b.oid, "same_color"))
            # Aligned
            if a.row == b.row or a.row + a.height == b.row + b.height:
                rels.append(Relation(a.oid, b.oid, "aligned_h"))
            if a.col == b.col or a.col + a.width == b.col + b.width:
                rels.append(Relation(a.oid, b.oid, "aligned_v"))

    return rels


def _decompose_residual(
    inp: Grid, out: Grid, residual: np.ndarray, bg: int,
) -> list[ResidualUnit]:
    if not np.any(residual):
        return []

    labeled, n = ndimage.label(residual, structure=np.ones((3, 3)))
    units = []
    struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    for uid in range(1, n + 1):
        comp = labeled == uid
        ys, xs = np.where(comp)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())

        in_vals = inp[comp]
        out_vals = out[comp]
        is_add = np.all(in_vals == bg)
        is_del = np.all(out_vals == bg)
        is_recol = np.all(in_vals != bg) and np.all(out_vals != bg)
        if is_add:
            ct = "add"
        elif is_del:
            ct = "delete"
        elif is_recol:
            ct = "recolor"
        else:
            ct = "mixed"

        # Adjacent colors
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        adj = sorted(set(int(v) for v in np.unique(inp[border])) - {bg}) if np.any(border) else []

        units.append(ResidualUnit(
            uid=uid,
            bbox=(r0, c0, r1, c1),
            mask=comp,
            n_pixels=int(np.sum(comp)),
            change_type=ct,
            output_colors=tuple(sorted(set(int(v) for v in np.unique(out_vals)))),
            input_colors=tuple(sorted(set(int(v) for v in np.unique(in_vals)))),
            adjacent_colors=tuple(adj),
        ))
    return units
