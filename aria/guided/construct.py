"""Output construction — build the answer layer by layer.

Step 1: Infer output size
Step 2: Determine background color
Step 3: Paint background canvas
Step 4: Copy preserved objects (legends, walls, scaffolds, static structure)
Step 5: Return canvas + residual mask (what still needs to be filled)

This module does NOT synthesize the residual — it just prepares the canvas
and identifies what's left to explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import Counter

import numpy as np
from scipy import ndimage

from aria.guided.workspace import build_workspace, _detect_bg, _extract_objects, ObjectInfo
from aria.guided.output_size import infer_output_size
from aria.guided.perceive import perceive
from aria.types import Grid


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ConstructedCanvas:
    """The partially-built output."""
    canvas: Grid                  # the output so far
    residual_mask: np.ndarray     # bool: True = still needs filling
    output_shape: tuple[int, int]
    bg: int
    size_mode: str                # "same", "static", "dynamic"
    n_preserved: int              # cells copied from input
    n_residual: int               # cells still empty
    preserved_objects: list[int]  # oids of copied objects


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def construct_canvas(
    demos: list[tuple[Grid, Grid]],
) -> list[ConstructedCanvas]:
    """Construct a canvas for each demo pair.

    Returns one ConstructedCanvas per demo, using cross-demo inference
    for output size and background.
    """
    if not demos:
        return []

    # Step 1: Infer output size (via the unified SizeRule engine)
    size_rule = infer_output_size(demos)
    in_shapes = [inp.shape for inp, _ in demos]
    out_shapes = [out.shape for _, out in demos]

    if size_rule is not None:
        size_mode = size_rule.mode
        output_shapes = []
        for inp, _ in demos:
            facts = perceive(inp)
            output_shapes.append(size_rule.predict(facts))
    else:
        # Fallback: use actual output shapes from demos
        size_mode = "same" if all(i == o for i, o in zip(in_shapes, out_shapes)) else "dynamic"
        output_shapes = list(out_shapes)

    # Step 2: Determine background color
    bg = _infer_background(demos, size_mode)

    # Steps 3-4: For each demo, build canvas and copy preserved objects
    results = []
    for i, (inp, out) in enumerate(demos):
        shape = output_shapes[i]
        canvas = np.full(shape, bg, dtype=np.uint8)

        # Copy preserved structure from input
        preserved_oids, residual = _copy_preserved(inp, out, canvas, bg, size_mode)

        results.append(ConstructedCanvas(
            canvas=canvas,
            residual_mask=residual,
            output_shape=shape,
            bg=bg,
            size_mode=size_mode,
            n_preserved=int(np.sum(~residual)),
            n_residual=int(np.sum(residual)),
            preserved_objects=preserved_oids,
        ))

    return results



# ---------------------------------------------------------------------------
# Step 2: Background inference
# ---------------------------------------------------------------------------

def _infer_background(
    demos: list[tuple[Grid, Grid]],
    size_mode: str,
) -> int:
    """Infer the background color for the output.

    Strategy:
    - If same-shape: bg = most common color in input (usually matches output bg)
    - If diff-shape: check output bg directly
    - Cross-demo: bg should be consistent
    """
    # Try input bg
    input_bgs = [_detect_bg(inp) for inp, _ in demos]
    output_bgs = [_detect_bg(out) for _, out in demos]

    # If all outputs agree on bg, use that
    if len(set(output_bgs)) == 1:
        return output_bgs[0]

    # If all inputs agree, use that
    if len(set(input_bgs)) == 1:
        return input_bgs[0]

    # Most common across all outputs
    return Counter(output_bgs).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Steps 3-4: Build canvas and copy preserved structure
# ---------------------------------------------------------------------------

def _copy_preserved(
    inp: Grid,
    out: Grid,
    canvas: Grid,
    bg: int,
    size_mode: str,
) -> tuple[list[int], np.ndarray]:
    """Copy static structure from input to canvas, return preserved oids and residual mask.

    A cell is "preserved" if it has the same value in input and output
    at the same position. These get copied to the canvas.
    Everything else is marked as residual.
    """
    rows, cols = canvas.shape
    residual = np.ones((rows, cols), dtype=bool)
    preserved_oids = []

    if size_mode == "same":
        # Same shape: directly compare input and output cell by cell
        same = inp == out
        # Copy preserved cells to canvas
        canvas[same] = inp[same]
        residual[same] = False

        # Track which objects are fully preserved
        objs = _extract_objects(inp, bg)
        for obj in objs:
            obj_mask = np.zeros((rows, cols), dtype=bool)
            obj_mask[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] |= obj.mask
            if np.all(same[obj_mask]):
                preserved_oids.append(obj.oid)

    elif size_mode == "static" or size_mode == "dynamic":
        # Diff shape: look for objects/subgrids from input that appear in output
        # at the same relative position or at any position
        objs_in = _extract_objects(inp, bg)

        for obj in objs_in:
            # Does this object appear in the output at the same position?
            if (obj.row + obj.height <= rows and obj.col + obj.width <= cols):
                out_sub = out[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width]
                in_sub = inp[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width]

                # Check if the object's pixels match
                match = True
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            if in_sub[r, c] != out_sub[r, c]:
                                match = False
                                break
                    if not match:
                        break

                if match:
                    # Copy to canvas
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                canvas[obj.row + r, obj.col + c] = inp[obj.row + r, obj.col + c]
                                residual[obj.row + r, obj.col + c] = False
                    preserved_oids.append(obj.oid)

    return preserved_oids, residual
