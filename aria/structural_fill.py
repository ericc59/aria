"""Structural fill: fill enclosed bg regions with role-derived colors.

Fills enclosed (interior) bg components with a color determined by
a structural role, not a literal. Verified per-demo with independent
region subset selection.

Supported color roles:
- rarest_color: the color with fewest total pixels in input
- unique_singleton: the singleton color that appears exactly once
- boundary_color: the color surrounding the enclosed region
- minority_non_bg: the non-bg color with fewest pixels
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects
from aria.types import DemoPair, Grid


@dataclass(frozen=True)
class StructuralFillProgram:
    """Executable: fill enclosed bg regions with a role-derived color."""
    color_role: str  # "rarest_color", "unique_singleton", "boundary_color"

    def verify_on_demo(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Verify by trying all subsets of enclosed regions."""
        bg = detect_bg(input_grid)
        fill_color = _resolve_color_role(self.color_role, input_grid, bg)
        if fill_color is None:
            return np.array_equal(input_grid, output_grid)

        regions = _find_enclosed_regions(input_grid, bg)
        if not regions:
            return np.array_equal(input_grid, output_grid)

        # Try all subsets of regions to fill
        for size in range(1, len(regions) + 1):
            for subset in combinations(regions, size):
                result = input_grid.copy()
                for cells in subset:
                    for r, c in cells:
                        result[r, c] = fill_color
                if np.array_equal(result, output_grid):
                    return True

        return False

    def execute(self, input_grid: Grid) -> Grid:
        """Fill all enclosed regions (test-time fallback)."""
        bg = detect_bg(input_grid)
        fill_color = _resolve_color_role(self.color_role, input_grid, bg)
        if fill_color is None:
            return input_grid.copy()

        regions = _find_enclosed_regions(input_grid, bg)
        result = input_grid.copy()
        for cells in regions:
            for r, c in cells:
                result[r, c] = fill_color
        return result


def solve_structural_fill(
    demos: tuple[DemoPair, ...],
) -> StructuralFillProgram | None:
    """Try structural fill with different color roles."""
    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    # Only consider additive tasks (all changes are bg -> non-bg)
    for d in demos:
        bg = detect_bg(d.input)
        diff = d.input != d.output
        if not np.any(diff):
            continue
        if not all(int(d.input[r, c]) == bg for r, c in zip(*np.where(diff))):
            return None  # Not purely additive

    for role in ["rarest_color", "unique_singleton", "boundary_color", "minority_non_bg"]:
        prog = StructuralFillProgram(color_role=role)
        all_ok = True
        for d in demos:
            if not prog.verify_on_demo(d.input, d.output):
                all_ok = False
                break
        if all_ok:
            return prog

    return None


def _resolve_color_role(role: str, grid: Grid, bg: int) -> int | None:
    """Resolve a color role to a literal color value."""
    objs = extract_objects(grid, bg)
    if not objs:
        return None

    color_pixels: dict[int, int] = {}
    color_obj_count: dict[int, int] = {}
    for o in objs:
        if o.color == bg:
            continue
        color_pixels[o.color] = color_pixels.get(o.color, 0) + o.size
        color_obj_count[o.color] = color_obj_count.get(o.color, 0) + 1

    if not color_pixels:
        return None

    if role == "rarest_color":
        return min(color_pixels, key=color_pixels.get)

    if role == "unique_singleton":
        # Color that appears exactly once as a singleton
        for color, count in color_obj_count.items():
            if count == 1 and color_pixels[color] == 1:
                return color
        return None

    if role == "minority_non_bg":
        return min(color_pixels, key=color_pixels.get)

    if role == "boundary_color":
        # Not applicable as a global role — would need per-region resolution
        return None

    return None


def _find_enclosed_regions(grid: Grid, bg: int) -> list[list[tuple[int, int]]]:
    """Find all enclosed (interior) bg components."""
    bg_mask = grid == bg
    labeled, n = ndimage.label(bg_mask)
    rows, cols = grid.shape

    border_labels: set[int] = set()
    for r in range(rows):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, cols - 1] > 0:
            border_labels.add(labeled[r, cols - 1])
    for c in range(cols):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[rows - 1, c] > 0:
            border_labels.add(labeled[rows - 1, c])

    regions: list[list[tuple[int, int]]] = []
    for lbl in range(1, n + 1):
        if lbl in border_labels:
            continue
        cells = list(zip(*np.where(labeled == lbl)))
        if cells:
            regions.append(cells)

    return regions
