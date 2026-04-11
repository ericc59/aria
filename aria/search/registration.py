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


@dataclass(frozen=True)
class RegistrationCandidate:
    shift_row: int
    shift_col: int
    target_site: tuple[int, int]
    source_anchor: tuple[int, int]
    canvas: np.ndarray


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


def module_anchor_patch(
    grid: Grid,
    shapes: list[AnchoredShape],
    module: MovableModule,
    *,
    shape_color: int,
    anchor_color: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Return an anchor-inclusive module patch plus paint mask and source anchors."""
    idxs = module.component_indices
    rows = [shapes[i].row for i in idxs]
    cols = [shapes[i].col for i in idxs]
    bottoms = [shapes[i].row + shapes[i].height - 1 for i in idxs]
    rights = [shapes[i].col + shapes[i].width - 1 for i in idxs]
    anchor_rows = [ar for i in idxs for ar, _ in shapes[i].anchors_global]
    anchor_cols = [ac for i in idxs for _, ac in shapes[i].anchors_global]
    r0 = min(rows + anchor_rows)
    c0 = min(cols + anchor_cols)
    r1 = max(bottoms + anchor_rows)
    c1 = max(rights + anchor_cols)
    patch = grid[r0:r1 + 1, c0:c1 + 1].copy()
    mask = (patch == shape_color) | (patch == anchor_color)
    source_anchors = tuple(
        sorted({(ar - r0, ac - c0) for i in idxs for ar, ac in shapes[i].anchors_global})
    )
    return patch, mask, source_anchors


def module_anchor_origin(
    shapes: list[AnchoredShape],
    module: MovableModule,
) -> tuple[int, int]:
    """Return the (r0, c0) origin used by module_anchor_patch."""
    idxs = module.component_indices
    rows = [shapes[i].row for i in idxs]
    cols = [shapes[i].col for i in idxs]
    bottoms = [shapes[i].row + shapes[i].height - 1 for i in idxs]
    rights = [shapes[i].col + shapes[i].width - 1 for i in idxs]
    anchor_rows = [ar for i in idxs for ar, _ in shapes[i].anchors_global]
    anchor_cols = [ac for i in idxs for _, ac in shapes[i].anchors_global]
    r0 = min(rows + anchor_rows) if anchor_rows else min(rows)
    c0 = min(cols + anchor_cols) if anchor_cols else min(cols)
    r1 = max(bottoms + anchor_rows) if anchor_rows else max(bottoms)
    c1 = max(rights + anchor_cols) if anchor_cols else max(rights)
    if r0 > r1 or c0 > c1:
        r0 = min(rows)
        c0 = min(cols)
    return int(r0), int(c0)


def base_registration_patch(
    base: AnchoredShape,
    *,
    shape_color: int,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Normalize the base patch and enumerate candidate attachment sites."""
    patch = base.patch.copy()
    sites: list[tuple[int, int]] = []
    for ar, ac in base.anchors_global:
        lr, lc = ar - base.row, ac - base.col
        if 0 <= lr < base.height and 0 <= lc < base.width:
            sites.append((lr, lc))
            patch[lr, lc] = shape_color
        else:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = lr + dr, lc + dc
                if 0 <= nr < base.height and 0 <= nc < base.width and patch[nr, nc] == shape_color:
                    sites.append((nr, nc))
    expanded: list[tuple[int, int]] = []
    for lr, lc in sites:
        if (lr, lc) not in expanded:
            expanded.append((lr, lc))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = lr + dr, lc + dc
            if 0 <= nr < base.height and 0 <= nc < base.width and patch[nr, nc] == shape_color and (nr, nc) not in expanded:
                expanded.append((nr, nc))
    return patch, tuple(expanded)


def overlay_registration_candidates(
    base_patch: np.ndarray,
    target_sites: tuple[tuple[int, int], ...],
    module_patch: np.ndarray,
    module_mask: np.ndarray,
    source_anchors: tuple[tuple[int, int], ...],
    *,
    bg_color: int,
    shape_color: int,
    anchor_color: int,
) -> list[RegistrationCandidate]:
    """Enumerate anchored overlay candidates with tight non-background crops."""
    normalized_module = module_patch.copy()
    normalized_module[normalized_module == anchor_color] = shape_color
    candidates: list[RegistrationCandidate] = []
    for target_site in target_sites:
        for source_anchor in source_anchors:
            shift_row = int(target_site[0] - source_anchor[0])
            shift_col = int(target_site[1] - source_anchor[1])
            min_r = min(0, shift_row)
            min_c = min(0, shift_col)
            max_r = max(base_patch.shape[0], shift_row + normalized_module.shape[0])
            max_c = max(base_patch.shape[1], shift_col + normalized_module.shape[1])
            canvas = np.full((max_r - min_r, max_c - min_c), bg_color, dtype=base_patch.dtype)
            ro = -min_r
            co = -min_c
            base_mask = base_patch != bg_color
            canvas[ro:ro + base_patch.shape[0], co:co + base_patch.shape[1]][base_mask] = base_patch[base_mask]
            rr = shift_row + ro
            cc = shift_col + co
            canvas[rr:rr + normalized_module.shape[0], cc:cc + normalized_module.shape[1]][module_mask] = normalized_module[module_mask]
            nz = np.argwhere(canvas != bg_color)
            cropped = canvas[nz[:, 0].min():nz[:, 0].max() + 1, nz[:, 1].min():nz[:, 1].max() + 1]
            candidates.append(
                RegistrationCandidate(
                    shift_row=shift_row,
                    shift_col=shift_col,
                    target_site=target_site,
                    source_anchor=source_anchor,
                    canvas=cropped,
                )
            )
    return candidates
