"""Residual object analysis: for each output object not on the canvas,
find what caused it to be there.

Causal categories:
  SAME_POS_RECOLOR  — same shape+position in input, different color
  MOVED             — same shape+color in input, different position
  MOVED_RECOLOR     — same shape in input, different position AND color
  FILLED            — bg region in input that became non-bg
  NEW               — no matching shape in input at all
  TILED             — a template object repeated to multiple positions

For each category, the cause can be described as a relation:
  "recolored because adjacent to singleton of color X"
  "moved to enclosed region"
  "filled with enclosing frame's color"
  "tiled from template at position (r,c)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aria.guided.construct import ConstructedCanvas, construct_canvas
from aria.guided.workspace import _detect_bg, _extract_objects, ObjectInfo
from aria.types import Grid


@dataclass
class ResidualObject:
    """One output object that needs to be placed on the canvas."""
    # What it is
    color: int
    row: int
    col: int
    height: int
    width: int
    size: int
    mask: np.ndarray

    # What caused it
    cause: str            # SAME_POS_RECOLOR, MOVED, MOVED_RECOLOR, FILLED, NEW, TILED
    source_obj: ObjectInfo | None  # the input object it came from (if any)
    source_color: int     # original color (-1 if no source)
    new_color: int        # the output color


@dataclass
class DemoResidual:
    """All residual objects for one demo."""
    objects: list[ResidualObject]
    canvas: ConstructedCanvas


def analyze_residual_objects(
    demos: list[tuple[Grid, Grid]],
) -> list[DemoResidual]:
    """For each demo, find residual objects and their causes."""
    canvases = construct_canvas(demos)
    results = []

    for (inp, out), cv in zip(demos, canvases):
        if cv.canvas.shape != out.shape:
            results.append(DemoResidual([], cv))
            continue

        bg = cv.bg
        out_objs = _extract_objects(out, bg)
        in_objs = _extract_objects(inp, bg)

        residuals = []
        for obj in out_objs:
            if _is_on_canvas(obj, cv):
                continue
            cause, source = _find_cause(obj, in_objs, inp, bg)
            residuals.append(ResidualObject(
                color=obj.color,
                row=obj.row, col=obj.col,
                height=obj.height, width=obj.width,
                size=obj.size, mask=obj.mask,
                cause=cause,
                source_obj=source,
                source_color=source.color if source else -1,
                new_color=obj.color,
            ))

        results.append(DemoResidual(residuals, cv))

    return results


def _is_on_canvas(obj: ObjectInfo, cv: ConstructedCanvas) -> bool:
    """Check if all of this object's pixels match the canvas."""
    for r in range(obj.height):
        for c in range(obj.width):
            if obj.mask[r, c]:
                gr, gc = obj.row + r, obj.col + c
                if gr >= cv.canvas.shape[0] or gc >= cv.canvas.shape[1]:
                    return False
                if cv.canvas[gr, gc] != obj.color:
                    return False
    return True


def _find_cause(
    out_obj: ObjectInfo,
    in_objs: list[ObjectInfo],
    inp: Grid,
    bg: int,
) -> tuple[str, ObjectInfo | None]:
    """Determine what caused this output object.

    Returns (cause_type, source_object_or_None).
    """
    # Check: same position, same shape, different color?
    for in_obj in in_objs:
        if (in_obj.row == out_obj.row and in_obj.col == out_obj.col and
                in_obj.height == out_obj.height and in_obj.width == out_obj.width and
                np.array_equal(in_obj.mask, out_obj.mask)):
            if in_obj.color != out_obj.color:
                return "SAME_POS_RECOLOR", in_obj
            # Same everything — should be on canvas
            return "PRESERVED", in_obj

    # Check: same shape+color, different position? (moved)
    for in_obj in in_objs:
        if (in_obj.color == out_obj.color and
                in_obj.height == out_obj.height and in_obj.width == out_obj.width and
                np.array_equal(in_obj.mask, out_obj.mask)):
            return "MOVED", in_obj

    # Check: same shape, different position AND color? (moved + recolored)
    for in_obj in in_objs:
        if (in_obj.height == out_obj.height and in_obj.width == out_obj.width and
                np.array_equal(in_obj.mask, out_obj.mask)):
            return "MOVED_RECOLOR", in_obj

    # Check: was this position bg in the input? (filled)
    all_bg = True
    for r in range(out_obj.height):
        for c in range(out_obj.width):
            if out_obj.mask[r, c]:
                gr, gc = out_obj.row + r, out_obj.col + c
                if 0 <= gr < inp.shape[0] and 0 <= gc < inp.shape[1]:
                    if inp[gr, gc] != bg:
                        all_bg = False
                        break
        if not all_bg:
            break
    if all_bg:
        return "FILLED", None

    # No match found
    return "NEW", None
