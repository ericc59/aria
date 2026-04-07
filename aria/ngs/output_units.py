"""Output-unit decomposition for the backward explanation engine.

Decomposes output grids into atomic units that can be independently
explained by primitive graphs. Units are chosen to maximize the chance
that a small primitive graph can explain each one.

Unit types:
- WHOLE: the entire output grid (simplest case)
- DIFF_REGION: connected component of changed pixels (same-shape tasks)
- PANEL: sub-grid from separator-based partitioning
- OBJECT: a single connected component in the output
- ROW/COL: a single row or column (for line-based tasks)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.types import Grid, DemoPair


class UnitType(Enum):
    WHOLE = auto()
    DIFF_REGION = auto()
    PANEL = auto()
    OBJECT = auto()
    ROW = auto()
    COL = auto()


@dataclass(frozen=True)
class OutputUnit:
    """One atomic piece of the output to be explained."""
    unit_type: UnitType
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    content: Grid                     # the output subgrid
    input_support: Grid | None        # corresponding input region (if known)
    input_bbox: tuple[int, int, int, int] | None  # where in input
    mask: np.ndarray | None           # which pixels in bbox are "this unit" (for non-rectangular)
    unit_id: str


def decompose_output_units(
    inp: Grid,
    out: Grid,
    bg: int | None = None,
) -> list[OutputUnit]:
    """Decompose one demo's output into explanation targets.

    Strategy priority:
    1. If same shape and few changes → DIFF_REGION units
    2. If output has separators → PANEL units
    3. If output has distinct objects → OBJECT units
    4. Fallback → WHOLE unit

    Returns a list of units, each to be independently explained.
    """
    if bg is None:
        bg = detect_bg(inp)

    same_shape = inp.shape == out.shape

    if same_shape:
        diff_units = _diff_region_units(inp, out, bg)
        if diff_units:
            return diff_units
        # No diff → identity, return whole
        return [_whole_unit(inp, out, bg)]

    # Different shape: try panels, then objects, then whole
    panel_units = _panel_units(out, inp, bg)
    if panel_units and len(panel_units) >= 2:
        return panel_units

    obj_units = _object_units(out, inp, bg)
    if obj_units and len(obj_units) >= 2:
        return obj_units

    return [_whole_unit(inp, out, bg)]


def _whole_unit(inp: Grid, out: Grid, bg: int) -> OutputUnit:
    rows, cols = out.shape
    inp_support = inp if inp.shape == out.shape else None
    inp_bbox = (0, 0, inp.shape[0] - 1, inp.shape[1] - 1) if inp_support is not None else None
    return OutputUnit(
        unit_type=UnitType.WHOLE,
        bbox=(0, 0, rows - 1, cols - 1),
        content=out.copy(),
        input_support=inp_support,
        input_bbox=inp_bbox,
        mask=None,
        unit_id="whole",
    )


def _diff_region_units(inp: Grid, out: Grid, bg: int) -> list[OutputUnit]:
    """For same-shape tasks: find connected regions of changed pixels."""
    diff_mask = inp != out
    if not np.any(diff_mask):
        return []

    labeled, n = ndimage.label(diff_mask, structure=np.ones((3, 3)))
    if n == 0:
        return []

    # If changes are >50% of grid, return WHOLE instead
    if np.sum(diff_mask) > 0.5 * diff_mask.size:
        return [_whole_unit(inp, out, bg)]

    units = []
    for label_id in range(1, n + 1):
        component = labeled == label_id
        rows_w, cols_w = np.where(component)
        r0, r1 = int(rows_w.min()), int(rows_w.max())
        c0, c1 = int(cols_w.min()), int(cols_w.max())

        # Expand by 1 for context
        rows, cols = out.shape
        r0e = max(0, r0 - 1)
        c0e = max(0, c0 - 1)
        r1e = min(rows - 1, r1 + 1)
        c1e = min(cols - 1, c1 + 1)

        out_sub = out[r0e:r1e + 1, c0e:c1e + 1].copy()
        in_sub = inp[r0e:r1e + 1, c0e:c1e + 1].copy()

        units.append(OutputUnit(
            unit_type=UnitType.DIFF_REGION,
            bbox=(r0e, c0e, r1e, c1e),
            content=out_sub,
            input_support=in_sub,
            input_bbox=(r0e, c0e, r1e, c1e),
            mask=component[r0e:r1e + 1, c0e:c1e + 1],
            unit_id=f"diff_{label_id}",
        ))

    return units


def _panel_units(out: Grid, inp: Grid, bg: int) -> list[OutputUnit]:
    """Detect separator-based panels in the output."""
    from aria.decomposition import detect_panels
    out_bg = detect_bg(out)
    panels = detect_panels(out, out_bg)
    if panels is None or panels.n_panels < 2:
        return []

    units = []
    for p in panels.panels:
        r0 = p.row
        c0 = p.col
        r1 = p.row + p.height - 1
        c1 = p.col + p.width - 1
        units.append(OutputUnit(
            unit_type=UnitType.PANEL,
            bbox=(r0, c0, r1, c1),
            content=p.grid.copy(),
            input_support=None,
            input_bbox=None,
            mask=None,
            unit_id=f"panel_{p.index}",
        ))
    return units


def _object_units(out: Grid, inp: Grid, bg: int) -> list[OutputUnit]:
    """Decompose output into individual objects."""
    out_bg = detect_bg(out)
    objects = extract_objects(out, out_bg, connectivity=4)
    if len(objects) < 2:
        return []

    units = []
    for i, obj in enumerate(objects):
        r0, c0 = obj.row, obj.col
        r1, c1 = r0 + obj.bbox_h - 1, c0 + obj.bbox_w - 1
        sub = out[r0:r1 + 1, c0:c1 + 1].copy()
        units.append(OutputUnit(
            unit_type=UnitType.OBJECT,
            bbox=(r0, c0, r1, c1),
            content=sub,
            input_support=None,
            input_bbox=None,
            mask=obj.mask,
            unit_id=f"obj_{i}_{obj.color}",
        ))
    return units
