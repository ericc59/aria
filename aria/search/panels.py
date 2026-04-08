"""Cross-panel structural reasoning.

Typed substrate for multi-panel tasks: extract panels, compare
objects/shapes across panels, identify common and unique motifs,
render motifs into target canvas.

Not a collection of heuristic decode strategies. A reasoning layer
that panel-based search can build on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from aria.search.sketch import SearchStep, SearchProgram, StepSelect


# ---------------------------------------------------------------------------
# Typed panel representation
# ---------------------------------------------------------------------------

@dataclass
class Motif:
    """A small shape extracted from a panel. Normalized (position-free)."""
    mask: np.ndarray       # boolean, cropped to bbox
    color: int             # the non-bg color
    height: int
    width: int
    size: int              # pixel count

    @property
    def key(self) -> bytes:
        """Canonical key for exact shape matching."""
        return self.mask.tobytes() + bytes([self.height, self.width])

    def matches(self, other: 'Motif', allow_transform: bool = False) -> str | None:
        """Check if this motif matches another. Returns transform name or None."""
        if self.height == other.height and self.width == other.width:
            if np.array_equal(self.mask, other.mask):
                return 'identical'
        if not allow_transform:
            return None
        for xform_name, xfn in _TRANSFORMS:
            transformed = xfn(self.mask)
            if (transformed.shape == other.mask.shape and
                    np.array_equal(transformed, other.mask)):
                return xform_name
        return None


_TRANSFORMS = [
    ('flip_h', lambda m: m[:, ::-1]),
    ('flip_v', lambda m: m[::-1, :]),
    ('flip_hv', lambda m: m[::-1, ::-1]),
    ('rot90', lambda m: np.rot90(m)),
    ('rot180', lambda m: np.rot90(m, 2)),
    ('rot270', lambda m: np.rot90(m, 3)),
]


@dataclass
class PanelObject:
    """An object within a panel, with position relative to panel origin."""
    motif: Motif
    row: int               # position within panel
    col: int
    color: int


@dataclass
class Panel:
    """A single panel extracted from a grid."""
    grid: np.ndarray       # the panel's pixel data
    objects: list[PanelObject]
    bg: int
    origin_row: int        # position in the source grid
    origin_col: int

    @property
    def shape(self):
        return self.grid.shape


@dataclass
class PanelSet:
    """A collection of panels extracted from one grid.

    This is the typed substrate for cross-panel reasoning.
    """
    panels: list[Panel]
    bg: int
    separator_color: int = -1

    def common_motifs(self, min_size: int = 2, allow_transform: bool = True) -> list[dict]:
        """Find motifs that appear in ALL panels.

        Returns list of dicts with:
          'motif': the canonical Motif
          'occurrences': list of (panel_idx, PanelObject, transform) per panel
        """
        if len(self.panels) < 2:
            return []

        # Index motifs from panel 0
        p0_objs = [o for o in self.panels[0].objects if o.motif.size >= min_size]
        if not p0_objs:
            # Fall back to all objects including singletons
            p0_objs = self.panels[0].objects

        results = []
        for ref_obj in p0_objs:
            occurrences = [(0, ref_obj, 'identical')]
            all_panels_match = True

            for pi in range(1, len(self.panels)):
                panel = self.panels[pi]
                found = False
                for obj in panel.objects:
                    xform = ref_obj.motif.matches(obj.motif, allow_transform)
                    if xform:
                        occurrences.append((pi, obj, xform))
                        found = True
                        break
                if not found:
                    all_panels_match = False
                    break

            if all_panels_match:
                results.append({
                    'motif': ref_obj.motif,
                    'occurrences': occurrences,
                })

        # Deduplicate: if multiple ref objects match the same shape, keep largest
        seen_keys = set()
        deduped = []
        for r in sorted(results, key=lambda x: -x['motif'].size):
            k = r['motif'].key
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append(r)

        return deduped

    def unique_to_panel(self, panel_idx: int, min_size: int = 2) -> list[PanelObject]:
        """Find motifs that appear ONLY in the specified panel."""
        panel = self.panels[panel_idx]
        other_keys = set()
        for pi in range(len(self.panels)):
            if pi == panel_idx:
                continue
            for obj in self.panels[pi].objects:
                other_keys.add(obj.motif.key)
                for _, xfn in _TRANSFORMS:
                    other_keys.add(Motif(
                        mask=xfn(obj.motif.mask),
                        color=obj.color,
                        height=xfn(obj.motif.mask).shape[0],
                        width=xfn(obj.motif.mask).shape[1],
                        size=obj.motif.size,
                    ).key)

        unique = []
        for obj in panel.objects:
            if obj.motif.size >= min_size and obj.motif.key not in other_keys:
                unique.append(obj)
        return unique

    def panel_diff(self, a_idx: int, b_idx: int) -> dict:
        """Compare two panels: what's in A but not B, and vice versa."""
        a_keys = {o.motif.key: o for o in self.panels[a_idx].objects}
        b_keys = {o.motif.key: o for o in self.panels[b_idx].objects}
        only_a = [o for k, o in a_keys.items() if k not in b_keys]
        only_b = [o for k, o in b_keys.items() if k not in a_keys]
        common = [o for k, o in a_keys.items() if k in b_keys]
        return {'only_a': only_a, 'only_b': only_b, 'common': common}


# ---------------------------------------------------------------------------
# Panel extraction
# ---------------------------------------------------------------------------

def extract_panels(grid: np.ndarray, bg: int = None) -> PanelSet | None:
    """Extract panels from a grid using separator detection.

    Uses aria's perception for separator finding, then extracts
    panel regions and their objects.
    """
    from aria.guided.perceive import perceive

    facts = perceive(grid)
    if bg is None:
        bg = facts.bg

    if not facts.separators or not facts.regions:
        return None

    panels = []
    sep_color = facts.separators[0].color if facts.separators else -1

    for region in facts.regions:
        sub = grid[region.r0:region.r1 + 1, region.c0:region.c1 + 1].copy()
        objects = _extract_panel_objects(sub, bg)
        panels.append(Panel(
            grid=sub,
            objects=objects,
            bg=bg,
            origin_row=region.r0,
            origin_col=region.c0,
        ))

    if len(panels) < 2:
        return None

    return PanelSet(panels=panels, bg=bg, separator_color=sep_color)


def _extract_panel_objects(panel: np.ndarray, bg: int) -> list[PanelObject]:
    """Extract connected-component objects from a panel subgrid."""
    binary = (panel != bg).astype(np.int32)
    labeled, n = ndimage.label(binary)
    objects = []

    for label_id in range(1, n + 1):
        mask_full = labeled == label_id
        rows_where = np.where(mask_full.any(axis=1))[0]
        cols_where = np.where(mask_full.any(axis=0))[0]
        if len(rows_where) == 0:
            continue

        r0, r1 = rows_where[0], rows_where[-1]
        c0, c1 = cols_where[0], cols_where[-1]
        cropped = mask_full[r0:r1 + 1, c0:c1 + 1]

        # Get the color (first non-bg pixel)
        color = bg
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if mask_full[r, c]:
                    color = int(panel[r, c])
                    break
            if color != bg:
                break

        motif = Motif(
            mask=cropped,
            color=color,
            height=cropped.shape[0],
            width=cropped.shape[1],
            size=int(cropped.sum()),
        )
        objects.append(PanelObject(motif=motif, row=r0, col=c0, color=color))

    return objects


# ---------------------------------------------------------------------------
# Cross-panel program derivation
# ---------------------------------------------------------------------------

def derive_panel_programs(demos: list[tuple[np.ndarray, np.ndarray]]) -> list[SearchProgram]:
    """Derive programs using cross-panel structural reasoning.

    Extracts panels, finds common/unique motifs, attempts to explain
    the output as a function of cross-panel relationships.
    """
    results = []

    # Extract panel sets for all demos
    panel_sets = []
    for inp, out in demos:
        ps = extract_panels(inp)
        if ps is None:
            return []
        panel_sets.append(ps)

    # Generic panel boolean algebra (and, or, xor, nor, nand, diff, rdiff)
    progs = _try_panel_boolean_algebra(panel_sets, demos)
    results.extend(progs)

    # Common motif render
    prog = _try_common_motif_render(panel_sets, demos)
    if prog:
        results.append(prog)

    return results


def _try_panel_boolean_algebra(panel_sets, demos):
    """Try all boolean ops on panel occupancy masks."""
    results = []
    ps0 = panel_sets[0]
    out0 = demos[0][1]
    bg = ps0.bg

    _OPS = {
        'and': lambda a, b: a & b,
        'or': lambda a, b: a | b,
        'xor': lambda a, b: a ^ b,
        'nor': lambda a, b: ~a & ~b,
        'nand': lambda a, b: ~(a & b),
        'a-b': lambda a, b: a & ~b,
        'b-a': lambda a, b: b & ~a,
    }

    for i in range(len(ps0.panels)):
        for j in range(len(ps0.panels)):
            if i == j:
                continue
            a = ps0.panels[i].grid
            b = ps0.panels[j].grid
            if a.shape != b.shape or a.shape != out0.shape:
                continue

            ma = (a != bg)
            mb = (b != bg)
            out_mask = (out0 != bg)

            for op_name, op_fn in _OPS.items():
                result_mask = op_fn(ma, mb)
                if not np.array_equal(result_mask, out_mask):
                    continue

                # Output color must be constant
                out_colors = set(int(out0[r, c]) for r in range(out0.shape[0])
                                 for c in range(out0.shape[1]) if out0[r, c] != bg)
                if len(out_colors) != 1:
                    continue
                render_color = next(iter(out_colors))

                # Verify across all demos
                all_ok = True
                for di in range(1, len(demos)):
                    psi = panel_sets[di]
                    out_i = demos[di][1]
                    if i >= len(psi.panels) or j >= len(psi.panels):
                        all_ok = False
                        break
                    ai = psi.panels[i].grid
                    bi = psi.panels[j].grid
                    if ai.shape != out_i.shape or bi.shape != out_i.shape:
                        all_ok = False
                        break
                    bgi = psi.bg
                    ri = op_fn((ai != bgi), (bi != bgi))
                    canvas = np.full(out_i.shape, bgi, dtype=np.uint8)
                    canvas[ri] = render_color
                    if not np.array_equal(canvas, out_i):
                        all_ok = False
                        break

                if all_ok:
                    prog = SearchProgram(
                        steps=[SearchStep('panel_boolean',
                                           {'op': op_name, 'panel_a': i, 'panel_b': j,
                                            'color': render_color})],
                        provenance=f'panel:bool_{op_name}({i},{j})_c{render_color}',
                    )
                    results.append(prog)
                    return results

    return results


def _try_common_motif_render(panel_sets, demos):
    """Output = the motif common to all panels, rendered on a canvas."""
    ps0 = panel_sets[0]
    common = ps0.common_motifs(min_size=2, allow_transform=True)
    if not common:
        common = ps0.common_motifs(min_size=1, allow_transform=True)

    for motif_info in common:
        motif = motif_info['motif']

        # Try rendering the motif at its position from each panel
        # Check if the output matches for any panel's position
        for occ_idx, (panel_idx, obj, xform) in enumerate(motif_info['occurrences']):
            # Render motif on output-sized canvas at the object's position
            out0 = demos[0][1]
            canvas = np.full(out0.shape, ps0.bg, dtype=np.uint8)

            # Use the mask, possibly transformed
            mask = motif.mask
            if xform != 'identical':
                for xname, xfn in _TRANSFORMS:
                    if xname == xform:
                        mask = xfn(mask)
                        break

            # Place at the object's panel position
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if mask[r, c]:
                        pr, pc = obj.row + r, obj.col + c
                        if 0 <= pr < canvas.shape[0] and 0 <= pc < canvas.shape[1]:
                            canvas[pr, pc] = obj.color

            if np.array_equal(canvas, out0):
                # Verify across demos
                all_ok = _verify_common_motif(panel_sets, demos, min_size=motif.size,
                                               panel_idx=panel_idx)
                if all_ok:
                    prog = SearchProgram(
                        steps=[SearchStep('common_motif_render',
                                           {'panel_idx': panel_idx})],
                        provenance=f'panel:common_motif_from_panel{panel_idx}',
                    )
                    return prog

    return None


def _verify_common_motif(panel_sets, demos, min_size, panel_idx):
    """Verify common-motif-render pattern across all demos."""
    for i, (inp, out) in enumerate(demos):
        ps = panel_sets[i]
        common = ps.common_motifs(min_size=min_size, allow_transform=True)
        if not common:
            common = ps.common_motifs(min_size=1, allow_transform=True)
        if not common:
            return False

        # Find the motif and render from the specified panel
        found = False
        for motif_info in common:
            if panel_idx >= len(motif_info['occurrences']):
                continue
            _, obj, xform = motif_info['occurrences'][panel_idx]
            mask = motif_info['motif'].mask
            if xform != 'identical':
                for xname, xfn in _TRANSFORMS:
                    if xname == xform:
                        mask = xfn(mask)
                        break

            canvas = np.full(out.shape, ps.bg, dtype=np.uint8)
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if mask[r, c]:
                        pr, pc = obj.row + r, obj.col + c
                        if 0 <= pr < canvas.shape[0] and 0 <= pc < canvas.shape[1]:
                            canvas[pr, pc] = obj.color
            if np.array_equal(canvas, out):
                found = True
                break
        if not found:
            return False

    return True


def _try_unique_motif_render(panel_sets, demos):
    """Output = motif unique to one panel, rendered."""
    # TODO: implement when common motif isn't the pattern
    return None
