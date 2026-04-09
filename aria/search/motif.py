"""Motif extraction and panel-layout comparison.

Extracts translation-invariant, color-agnostic motif signatures from
grid regions. Compares panels by both motif identity and spatial layout.

Used in derive/binding to determine structural similarity between panels.
NOT an execution layer. NOT part of the AST.
"""

from __future__ import annotations

from dataclasses import dataclass

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
# Panel fact extraction (for rule-based derivation)
# ---------------------------------------------------------------------------

@dataclass
class PanelFacts:
    """Derived relational facts about a panel relative to a legend (P0)."""
    index: int
    n_motifs: int
    shapes: frozenset          # set of Motif shapes in this panel
    colors: frozenset          # set of motif colors
    n_matched: int             # motifs whose shape is in p0_shapes
    n_unmatched: int
    full_match: bool           # all motifs match P0
    any_match: bool            # at least one motif matches P0
    color_overlap: frozenset   # colors shared with P0
    color_disjoint: bool       # no shared colors with P0
    fewer_motifs: bool         # fewer motifs than P0


def extract_panel_facts(
    panel: np.ndarray,
    index: int,
    p0_shapes: frozenset,
    p0_colors: frozenset,
    p0_n_motifs: int,
    bg: int = 0,
    ground: int = 8,
) -> PanelFacts:
    """Extract structured facts about a panel relative to P0."""
    motifs = extract_motifs(panel, bg, ground, min_cells=2)
    shapes = frozenset(m.motif for m in motifs)
    colors = frozenset(m.color for m in motifs)
    matched = [m for m in motifs if m.motif in p0_shapes]

    return PanelFacts(
        index=index,
        n_motifs=len(motifs),
        shapes=shapes,
        colors=colors,
        n_matched=len(matched),
        n_unmatched=len(motifs) - len(matched),
        full_match=len(matched) > 0 and len(matched) == len(motifs),
        any_match=len(matched) > 0,
        color_overlap=colors & p0_colors,
        color_disjoint=len(colors & p0_colors) == 0,
        fewer_motifs=len(motifs) < p0_n_motifs,
    )
