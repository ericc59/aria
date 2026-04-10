"""Registration helpers for anchor-conditioned component transfer tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from aria.types import Grid


_STRUCTURE4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


@dataclass(frozen=True)
class AnchoredShape:
    color: int
    row: int
    col: int
    height: int
    width: int
    patch: np.ndarray
    anchor_color: int
    anchors_global: tuple[tuple[int, int], ...]
    anchors_local: tuple[tuple[int, int], ...]

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.patch == self.color))


@dataclass(frozen=True)
class MovableModule:
    component_indices: tuple[int, ...]
    anchored: bool


def extract_anchored_shapes(
    grid: Grid,
    *,
    shape_color: int,
    anchor_color: int,
    margin: int = 1,
) -> list[AnchoredShape]:
    """Extract shape-color components plus nearby anchor cells."""
    labels, count = ndimage.label(grid == shape_color, structure=_STRUCTURE4)
    shapes: list[AnchoredShape] = []
    rows, cols = grid.shape
    for idx in range(1, count + 1):
        coords = np.argwhere(labels == idx)
        if len(coords) == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[r0:r1 + 1, c0:c1 + 1].copy()
        mr0 = max(0, int(r0) - margin)
        mc0 = max(0, int(c0) - margin)
        mr1 = min(rows - 1, int(r1) + margin)
        mc1 = min(cols - 1, int(c1) + margin)
        anchor_coords = np.argwhere(grid[mr0:mr1 + 1, mc0:mc1 + 1] == anchor_color)
        anchors_global = tuple(sorted((int(mr0 + r), int(mc0 + c)) for r, c in anchor_coords))
        anchors_local = tuple(sorted((int(r - r0), int(c - c0)) for r, c in anchors_global))
        shapes.append(
            AnchoredShape(
                color=int(shape_color),
                row=int(r0),
                col=int(c0),
                height=int(r1 - r0 + 1),
                width=int(c1 - c0 + 1),
                patch=patch,
                anchor_color=int(anchor_color),
                anchors_global=anchors_global,
                anchors_local=anchors_local,
            )
        )
    shapes.sort(key=lambda shape: (shape.row, shape.col, shape.height * shape.width))
    return shapes


def cluster_movable_modules(
    shapes: list[AnchoredShape],
    *,
    gap: int = 1,
    base_index: int | None = None,
) -> tuple[int | None, list[MovableModule]]:
    """Cluster non-base shapes into anchored movable modules.

    The largest shape defaults to the fixed base. Remaining anchor-bearing shapes seed
    modules; small unanchored neighbors can attach to the nearest anchored module when
    they sit within the specified bbox gap.
    """
    if not shapes:
        return None, []
    if base_index is None:
        base_index = max(range(len(shapes)), key=lambda idx: shapes[idx].area)

    def rect_gap(a: AnchoredShape, b: AnchoredShape) -> int:
        ar0, ac0 = a.row, a.col
        ar1, ac1 = a.row + a.height - 1, a.col + a.width - 1
        br0, bc0 = b.row, b.col
        br1, bc1 = b.row + b.height - 1, b.col + b.width - 1
        dr = max(0, max(br0 - ar1 - 1, ar0 - br1 - 1))
        dc = max(0, max(bc0 - ac1 - 1, ac0 - bc1 - 1))
        return max(dr, dc)

    seeded: list[list[int]] = []
    unanchored: list[int] = []
    for idx, shape in enumerate(shapes):
        if idx == base_index:
            continue
        if shape.anchors_global:
            seeded.append([idx])
        else:
            unanchored.append(idx)

    if not seeded:
        seeded = [[idx] for idx in range(len(shapes)) if idx != base_index]
        return base_index, [MovableModule(tuple(group), anchored=bool(shapes[group[0]].anchors_global)) for group in seeded]

    for idx in unanchored:
        best: tuple[int, int] | None = None
        for group_idx, group in enumerate(seeded):
            gap_val = min(rect_gap(shapes[idx], shapes[j]) for j in group)
            if gap_val > gap:
                continue
            best = (gap_val, group_idx) if best is None or (gap_val, group_idx) < best else best
        if best is not None:
            seeded[best[1]].append(idx)
        else:
            seeded.append([idx])

    modules = [
        MovableModule(tuple(sorted(group)), anchored=any(bool(shapes[idx].anchors_global) for idx in group))
        for group in seeded
    ]
    modules.sort(key=lambda module: min(shapes[idx].row for idx in module.component_indices))
    return base_index, modules
