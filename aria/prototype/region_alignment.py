"""Input↔output substructure alignment for the backward solver.

For each output region, finds plausible corresponding input support:
- Same bbox (for same-shape tasks)
- Structurally similar subgrid in input (by color signature, shape)
- Partition cell / panel at same index
- Nearest object with matching properties
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.prototype.output_regions import OutputRegion
from aria.types import Grid


@dataclass(frozen=True)
class RegionAlignment:
    """An aligned pair of input and output substructures."""
    output_region: OutputRegion
    input_bbox: tuple[int, int, int, int]  # input region bbox
    input_content: Grid                     # input subgrid at that bbox
    output_content: Grid                    # output subgrid
    alignment_type: str                     # "same_bbox", "partition_cell", etc.
    confidence: float


def align_regions(
    input_grid: Grid,
    output_grid: Grid,
    output_regions: list[OutputRegion],
    bg: int = 0,
) -> list[RegionAlignment]:
    """Find input support for each output region."""
    alignments = []

    for region in output_regions:
        alignment = _find_best_alignment(input_grid, output_grid, region, bg)
        if alignment is not None:
            alignments.append(alignment)

    return alignments


def _find_best_alignment(
    input_grid: Grid,
    output_grid: Grid,
    region: OutputRegion,
    bg: int,
) -> RegionAlignment | None:
    """Find the best input support for one output region."""
    candidates = []

    # Strategy 1: same bbox (for same-shape tasks)
    if input_grid.shape == output_grid.shape:
        r0, c0, r1, c1 = region.bbox
        if r1 < input_grid.shape[0] and c1 < input_grid.shape[1]:
            in_sub = input_grid[r0:r1 + 1, c0:c1 + 1].copy()
            candidates.append(RegionAlignment(
                output_region=region,
                input_bbox=region.bbox,
                input_content=in_sub,
                output_content=region.content,
                alignment_type="same_bbox",
                confidence=1.0,
            ))

    # Strategy 2: search for similar subgrid in input
    if not candidates:
        oH, oW = region.content.shape
        iH, iW = input_grid.shape
        if oH <= iH and oW <= iW:
            best_match = None
            best_diff = float('inf')
            # Check color signature match
            out_colors = set(int(v) for v in np.unique(region.content)) - {bg}
            for r in range(iH - oH + 1):
                for c in range(iW - oW + 1):
                    in_sub = input_grid[r:r + oH, c:c + oW]
                    in_colors = set(int(v) for v in np.unique(in_sub)) - {bg}
                    # Color overlap score
                    overlap = len(out_colors & in_colors)
                    if overlap == 0 and out_colors:
                        continue
                    diff = int(np.sum(in_sub != region.content))
                    if diff < best_diff:
                        best_diff = diff
                        best_match = (r, c, r + oH - 1, c + oW - 1, in_sub, diff)

            if best_match is not None:
                r0, c0, r1, c1, in_sub, diff = best_match
                total = region.content.size
                conf = 1.0 - (diff / total) if total > 0 else 0.0
                candidates.append(RegionAlignment(
                    output_region=region,
                    input_bbox=(r0, c0, r1, c1),
                    input_content=in_sub.copy(),
                    output_content=region.content,
                    alignment_type="best_subgrid_match",
                    confidence=conf,
                ))

    if not candidates:
        return None

    return max(candidates, key=lambda a: a.confidence)
