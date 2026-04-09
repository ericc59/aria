"""Motif extraction and panel-layout comparison.

Extracts translation-invariant, color-agnostic motif signatures from
grid regions. Compares panels by both motif identity and spatial layout.

Used in derive/binding to determine structural similarity between panels.
NOT an execution layer. NOT part of the AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Motif extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Motif:
    """A single motif: translation-invariant shape signature + metadata."""
    shape_sig: frozenset       # frozenset of (dr, dc) offsets from top-left
    n_cells: int
    bbox_h: int
    bbox_w: int

    def __eq__(self, other):
        return isinstance(other, Motif) and self.shape_sig == other.shape_sig

    def __hash__(self):
        return hash(self.shape_sig)


@dataclass(frozen=True)
class PlacedMotif:
    """A motif with its position inside a panel."""
    motif: Motif
    row: int                   # top-left row in the panel
    col: int                   # top-left col in the panel
    center_row: float          # centroid row
    center_col: float          # centroid col
    color: int                 # the color of this motif instance


def extract_motifs(
    panel: np.ndarray,
    bg: int = 0,
    ground: int = 8,
    min_cells: int = 2,
) -> list[PlacedMotif]:
    """Extract motifs from a panel.

    A motif = 4-connected component of cells that are neither bg nor ground.
    Returns PlacedMotifs with shape signature + position.
    """
    mask = np.ones(panel.shape, dtype=bool)
    for skip in (bg, ground):
        mask &= (panel != skip)

    labeled, n = label(mask)
    motifs = []

    for mi in range(1, n + 1):
        cells = list(zip(*np.where(labeled == mi)))
        if len(cells) < min_cells:
            continue

        r0 = min(r for r, c in cells)
        c0 = min(c for r, c in cells)
        r1 = max(r for r, c in cells)
        c1 = max(c for r, c in cells)

        shape_sig = frozenset((r - r0, c - c0) for r, c in cells)
        color = int(panel[cells[0][0], cells[0][1]])
        center_r = sum(r for r, c in cells) / len(cells)
        center_c = sum(c for r, c in cells) / len(cells)

        motifs.append(PlacedMotif(
            motif=Motif(shape_sig=shape_sig, n_cells=len(cells),
                        bbox_h=r1 - r0 + 1, bbox_w=c1 - c0 + 1),
            row=r0, col=c0,
            center_row=center_r, center_col=center_c,
            color=color,
        ))

    return motifs


# ---------------------------------------------------------------------------
# Shape-only signatures (color-agnostic, position-agnostic)
# ---------------------------------------------------------------------------

def shape_set(panel: np.ndarray, bg: int = 0, ground: int = 8) -> frozenset[Motif]:
    """Get the set of unique motif shapes in a panel."""
    return frozenset(pm.motif for pm in extract_motifs(panel, bg, ground))


# ---------------------------------------------------------------------------
# Layout-aware signatures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutSlot:
    """A motif placed in a normalized position within a panel."""
    motif: Motif
    bucket_row: int   # coarse row position (0 = top, 1 = mid, 2 = bottom)
    bucket_col: int   # coarse col position (bucketed by panel width)


def panel_layout(
    panel: np.ndarray,
    bg: int = 0,
    ground: int = 8,
    n_col_buckets: int = 4,
) -> frozenset[LayoutSlot]:
    """Get a layout-aware signature for a panel.

    Each motif's position is bucketed into coarse column slots
    (row bucketing uses the 3-row panel height → row = exact row).
    """
    placed = extract_motifs(panel, bg, ground)
    h, w = panel.shape
    col_bucket_size = max(w // n_col_buckets, 1)

    slots = []
    for pm in placed:
        bucket_r = min(pm.row, h - 1)  # for 3-row panels, row IS the bucket
        bucket_c = int(pm.center_col // col_bucket_size)
        slots.append(LayoutSlot(
            motif=pm.motif,
            bucket_row=bucket_r,
            bucket_col=bucket_c,
        ))

    return frozenset(slots)


def layout_overlap(layout_a: frozenset[LayoutSlot],
                   layout_b: frozenset[LayoutSlot]) -> int:
    """Count how many layout slots are shared between two panels."""
    return len(layout_a & layout_b)


# ---------------------------------------------------------------------------
# Positional motif matching (exact position, shape-only)
# ---------------------------------------------------------------------------

def positional_motif_set(
    panel: np.ndarray,
    bg: int = 0,
    ground: int = 8,
) -> frozenset[tuple[Motif, int, int]]:
    """Motif set with exact (row, col) anchor positions.

    This is the strictest comparison: same shape at same position.
    """
    placed = extract_motifs(panel, bg, ground)
    return frozenset((pm.motif, pm.row, pm.col) for pm in placed)
