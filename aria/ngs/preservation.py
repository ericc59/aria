"""Preservation factoring for the NGS solver.

Separates each demo's output into:
- preserved_mask: output cells directly explained by the input (identity copy)
- residual_mask: output cells that differ from input (need explanation)
- support_context: structural features of the preserved region that
  can condition the residual explanation

The key insight: most ARC-2 outputs are >90% identical to input.
By factoring out the preserved part, the backward explanation engine
only needs to explain the small residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.types import Grid


# ---------------------------------------------------------------------------
# Preservation result
# ---------------------------------------------------------------------------

@dataclass
class PreservationFactor:
    """Result of preservation factoring on one demo."""
    preserved_mask: np.ndarray       # bool, True = directly from input
    residual_mask: np.ndarray        # bool, True = needs explanation
    n_preserved: int
    n_residual: int
    preservation_ratio: float

    # Residual characterization
    residual_regions: list[ResidualRegion]
    n_residual_regions: int

    # Support context for conditioning residual explanation
    support: SupportContext


@dataclass
class ResidualRegion:
    """One connected component of the residual."""
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    mask: np.ndarray                 # bool mask within bbox
    input_patch: Grid                # input content at this bbox
    output_patch: Grid               # output content at this bbox
    n_pixels: int
    region_id: int

    # Classification
    change_type: str  # "add" | "delete" | "recolor" | "mixed"
    output_colors: set[int]          # unique colors in residual output
    input_colors: set[int]           # unique colors in residual input (at changed positions)

    # Structural context
    enclosing_object: RawObject | None  # the input object that encloses this residual
    enclosing_color: int | None         # color of the enclosing object
    adjacent_colors: set[int]           # colors adjacent to this residual in the input
    is_enclosed: bool                   # is this residual fully enclosed by non-bg in input?


@dataclass
class SupportContext:
    """Structural features of the preserved region."""
    bg_color: int
    palette: set[int]                # colors present in input
    objects: list[RawObject]         # all input objects
    n_objects: int
    object_colors: set[int]          # unique object colors
    has_frame: bool                  # is there a border/frame structure?
    has_separators: bool             # are there separator lines?


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def factor_preservation(
    inp: Grid,
    out: Grid,
    bg: int | None = None,
) -> PreservationFactor:
    """Factor output into preserved + residual.

    Only works for same-shape tasks. For diff-shape, returns
    a trivial factoring (everything is residual).
    """
    if bg is None:
        bg = detect_bg(inp)

    if inp.shape != out.shape:
        return _trivial_factor(inp, out, bg)

    # Preserved = positions where input == output
    preserved = inp == out
    residual = ~preserved

    n_preserved = int(np.sum(preserved))
    n_residual = int(np.sum(residual))
    total = inp.size
    ratio = n_preserved / total if total > 0 else 0.0

    # Decompose residual into connected regions
    residual_regions = _decompose_residual(inp, out, residual, bg)

    # Build support context
    support = _build_support_context(inp, bg)

    return PreservationFactor(
        preserved_mask=preserved,
        residual_mask=residual,
        n_preserved=n_preserved,
        n_residual=n_residual,
        preservation_ratio=ratio,
        residual_regions=residual_regions,
        n_residual_regions=len(residual_regions),
        support=support,
    )


# ---------------------------------------------------------------------------
# Residual decomposition
# ---------------------------------------------------------------------------

def _decompose_residual(
    inp: Grid,
    out: Grid,
    residual: np.ndarray,
    bg: int,
) -> list[ResidualRegion]:
    """Decompose the residual mask into classified regions."""
    if not np.any(residual):
        return []

    # Label connected components (8-connectivity for grouping nearby changes)
    labeled, n_components = ndimage.label(residual, structure=np.ones((3, 3)))
    objects = extract_objects(inp, bg, connectivity=4)

    regions: list[ResidualRegion] = []
    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        rows_w, cols_w = np.where(component)
        r0, r1 = int(rows_w.min()), int(rows_w.max())
        c0, c1 = int(cols_w.min()), int(cols_w.max())

        # Expand bbox by 1 for context (clamped)
        rows, cols = inp.shape
        r0e = max(0, r0 - 1)
        c0e = max(0, c0 - 1)
        r1e = min(rows - 1, r1 + 1)
        c1e = min(cols - 1, c1 + 1)

        mask_local = component[r0e:r1e + 1, c0e:c1e + 1]
        in_patch = inp[r0e:r1e + 1, c0e:c1e + 1].copy()
        out_patch = out[r0e:r1e + 1, c0e:c1e + 1].copy()

        # Classify change type
        changed_in = inp[component]
        changed_out = out[component]
        n_pixels = int(np.sum(component))

        is_add = np.all(changed_in == bg)
        is_delete = np.all(changed_out == bg)
        is_recolor = np.all(changed_in != bg) & np.all(changed_out != bg)

        if is_add:
            change_type = "add"
        elif is_delete:
            change_type = "delete"
        elif is_recolor:
            change_type = "recolor"
        else:
            change_type = "mixed"

        output_colors = set(int(v) for v in np.unique(changed_out))
        input_colors = set(int(v) for v in np.unique(changed_in))

        # Find enclosing object
        enclosing, enc_color = _find_enclosing_object(
            r0, c0, r1, c1, component, objects, inp, bg)

        # Find adjacent colors
        adj_colors = _adjacent_colors(component, inp, bg)

        # Is this residual fully enclosed?
        is_enclosed = _check_enclosed(component, inp, bg)

        regions.append(ResidualRegion(
            bbox=(r0e, c0e, r1e, c1e),
            mask=mask_local,
            input_patch=in_patch,
            output_patch=out_patch,
            n_pixels=n_pixels,
            region_id=label_id,
            change_type=change_type,
            output_colors=output_colors,
            input_colors=input_colors,
            enclosing_object=enclosing,
            enclosing_color=enc_color,
            adjacent_colors=adj_colors,
            is_enclosed=is_enclosed,
        ))

    return regions


def _find_enclosing_object(
    r0: int, c0: int, r1: int, c1: int,
    component: np.ndarray,
    objects: list[RawObject],
    inp: Grid, bg: int,
) -> tuple[RawObject | None, int | None]:
    """Find the input object whose bbox encloses this residual region."""
    best = None
    best_area = float("inf")
    for obj in objects:
        or0, oc0 = obj.row, obj.col
        or1, oc1 = or0 + obj.bbox_h - 1, oc0 + obj.bbox_w - 1
        if or0 <= r0 and oc0 <= c0 and or1 >= r1 and oc1 >= c1:
            area = obj.bbox_h * obj.bbox_w
            if area < best_area:
                best = obj
                best_area = area
    return best, (best.color if best else None)


def _adjacent_colors(
    component: np.ndarray,
    inp: Grid,
    bg: int,
) -> set[int]:
    """Colors in input that are 4-adjacent to the residual region."""
    struct = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    dilated = ndimage.binary_dilation(component, structure=struct)
    border = dilated & ~component
    if not np.any(border):
        return set()
    return set(int(v) for v in np.unique(inp[border])) - {bg}


def _check_enclosed(
    component: np.ndarray,
    inp: Grid,
    bg: int,
) -> bool:
    """Is this residual region fully enclosed by non-bg in the input?

    Check: can any changed cell's immediate bg neighbors reach the border?
    A non-bg pixel is "enclosed" if all surrounding bg cells are trapped
    (cannot reach the grid border through bg).
    """
    rows, cols = inp.shape
    from collections import deque

    # Build reachability: flood from border through bg
    reachable = np.zeros_like(inp, dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1):
                if inp[r, c] == bg:
                    reachable[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and inp[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))

    # A changed region is enclosed if NO adjacent bg cell can reach the border
    for r, c in zip(*np.where(component)):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if inp[nr, nc] == bg and reachable[nr, nc]:
                    return False
            else:
                # Changed cell is at grid border → not enclosed
                return False
    return True


# ---------------------------------------------------------------------------
# Support context
# ---------------------------------------------------------------------------

def _build_support_context(inp: Grid, bg: int) -> SupportContext:
    """Extract structural features from the preserved input."""
    objects = extract_objects(inp, bg, connectivity=4)
    palette = set(int(v) for v in np.unique(inp))
    obj_colors = set(o.color for o in objects)

    # Simple frame detection: is the border all one non-bg color?
    rows, cols = inp.shape
    border_vals = set()
    if rows >= 2 and cols >= 2:
        border_vals = set(int(v) for v in np.unique(np.concatenate([
            inp[0, :], inp[-1, :], inp[:, 0], inp[:, -1]
        ])))
    has_frame = len(border_vals) == 1 and bg not in border_vals

    # Simple separator detection: full rows or columns of single non-bg color
    has_sep = False
    for r in range(rows):
        row = inp[r, :]
        if len(np.unique(row)) == 1 and int(row[0]) != bg:
            has_sep = True
            break
    if not has_sep:
        for c in range(cols):
            col = inp[:, c]
            if len(np.unique(col)) == 1 and int(col[0]) != bg:
                has_sep = True
                break

    return SupportContext(
        bg_color=bg,
        palette=palette,
        objects=objects,
        n_objects=len(objects),
        object_colors=obj_colors,
        has_frame=has_frame,
        has_separators=has_sep,
    )


# ---------------------------------------------------------------------------
# Trivial factor (diff-shape or no preservation)
# ---------------------------------------------------------------------------

def _trivial_factor(inp: Grid, out: Grid, bg: int) -> PreservationFactor:
    """Everything is residual."""
    rows, cols = out.shape
    return PreservationFactor(
        preserved_mask=np.zeros((rows, cols), dtype=bool),
        residual_mask=np.ones((rows, cols), dtype=bool),
        n_preserved=0,
        n_residual=out.size,
        preservation_ratio=0.0,
        residual_regions=[],
        n_residual_regions=0,
        support=_build_support_context(inp, bg),
    )
