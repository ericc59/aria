"""Output decomposition for the backward solver.

Decomposes an output grid into bounded substructures (regions) that
can be independently explained by local rules applied to corresponding
input substructures.

Each output region is a rectangular subgrid with:
- bbox (r0, c0, r1, c1)
- content (the pixel values)
- role: "changed" (differs from input) or "unchanged"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.types import Grid


@dataclass(frozen=True)
class OutputRegion:
    """A bounded substructure of the output grid."""
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    content: Grid                     # the output subgrid
    input_content: Grid | None        # corresponding input subgrid (if same shape)
    changed: bool                     # whether this region differs from input
    diff_count: int                   # number of pixels that changed
    region_id: str


def decompose_output(
    input_grid: Grid,
    output_grid: Grid,
    bg: int = 0,
) -> list[OutputRegion]:
    """Decompose an output grid into bounded substructures.

    Strategy: find connected components of changed pixels, then expand
    each to a bounding box. Also partition unchanged regions.

    For same-shape tasks, the changed regions are the interesting ones.
    For diff-shape tasks, decompose the output directly.
    """
    if input_grid.shape == output_grid.shape:
        return _decompose_same_shape(input_grid, output_grid, bg)
    else:
        return _decompose_diff_shape(input_grid, output_grid, bg)


def _decompose_same_shape(
    input_grid: Grid,
    output_grid: Grid,
    bg: int,
) -> list[OutputRegion]:
    """For same-shape tasks: find regions where output differs from input."""
    from scipy import ndimage

    diff_mask = input_grid != output_grid
    if not np.any(diff_mask):
        # No changes — single region covering whole grid
        rows, cols = output_grid.shape
        return [OutputRegion(
            bbox=(0, 0, rows - 1, cols - 1),
            content=output_grid.copy(),
            input_content=input_grid.copy(),
            changed=False,
            diff_count=0,
            region_id="full",
        )]

    # Label connected components of changed pixels (8-connectivity)
    labeled, n_components = ndimage.label(diff_mask, structure=np.ones((3, 3)))

    regions: list[OutputRegion] = []
    for label_id in range(1, n_components + 1):
        component_mask = labeled == label_id
        rows_with, cols_with = np.where(component_mask)
        r0, r1 = int(rows_with.min()), int(rows_with.max())
        c0, c1 = int(cols_with.min()), int(cols_with.max())

        # Expand bbox by 1 pixel for context (clamped to grid bounds)
        rows, cols = output_grid.shape
        r0_exp = max(0, r0 - 1)
        c0_exp = max(0, c0 - 1)
        r1_exp = min(rows - 1, r1 + 1)
        c1_exp = min(cols - 1, c1 + 1)

        out_sub = output_grid[r0_exp:r1_exp + 1, c0_exp:c1_exp + 1].copy()
        in_sub = input_grid[r0_exp:r1_exp + 1, c0_exp:c1_exp + 1].copy()
        diff_count = int(np.sum(in_sub != out_sub))

        regions.append(OutputRegion(
            bbox=(r0_exp, c0_exp, r1_exp, c1_exp),
            content=out_sub,
            input_content=in_sub,
            changed=True,
            diff_count=diff_count,
            region_id=f"change_{label_id}",
        ))

    return regions


def _decompose_diff_shape(
    input_grid: Grid,
    output_grid: Grid,
    bg: int,
) -> list[OutputRegion]:
    """For diff-shape tasks: decompose output into structural subregions.

    Uses separator detection on the output to find bounded cells,
    or treats the whole output as one region.
    """
    from aria.decomposition import detect_panels, detect_bg

    rows, cols = output_grid.shape
    out_bg = detect_bg(output_grid)

    # Try partition/panel decomposition of output
    panels = detect_panels(output_grid, out_bg)
    if panels is not None and panels.n_panels >= 2:
        regions = []
        for p in panels.panels:
            regions.append(OutputRegion(
                bbox=(p.row, p.col, p.row + p.height - 1, p.col + p.width - 1),
                content=p.grid.copy(),
                input_content=None,
                changed=True,
                diff_count=p.grid.size,
                region_id=f"panel_{p.index}",
            ))
        return regions

    # Fallback: whole output is one region
    return [OutputRegion(
        bbox=(0, 0, rows - 1, cols - 1),
        content=output_grid.copy(),
        input_content=None,
        changed=True,
        diff_count=output_grid.size,
        region_id="full",
    )]
