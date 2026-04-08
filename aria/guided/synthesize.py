"""Typed program synthesis over the DSL.

Programs are trees of typed operations. The synthesizer enumerates
programs bottom-up: start with input values, apply operations where
types match, verify intermediate results against demos.

This replaces the hardcoded _compose_* functions in dsl.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto

import numpy as np

from aria.guided.perceive import perceive, GridFacts, ObjFact
from aria.guided.dsl import (
    prim_select, prim_crop_bbox, prim_crop_interior,
    prim_find_frame, prim_split_at_separator,
    prim_combine, prim_render_mask, prim_crop,
    Program, _make_program,
)
from aria.types import Grid


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

class Ty(Enum):
    GRID = auto()       # 2D numpy array
    OBJECT = auto()     # ObjFact
    OBJECTS = auto()    # list[ObjFact]
    REGION = auto()     # 2D numpy array (subgrid)
    MASK = auto()       # 2D boolean array
    COLOR = auto()      # int
    INT = auto()        # int
    FRAME = auto()      # (r0, c0, r1, c1)
    PAIR = auto()       # (region, region)
    FACTS = auto()      # GridFacts (perception output: objects, separators, bg, dims)
    SELECTOR = auto()   # list[Predicate] — selects objects from facts
    BINDING = auto()    # (ObjFact, ObjFact) — a related pair (host/content, source/target)
    TRANSITION = auto() # ObjectTransition — observed change between input/output object
    TRANSITIONS = auto() # list[ObjectTransition] — all transitions for a demo


# ---------------------------------------------------------------------------
# Object transition — observed change between input and output
# ---------------------------------------------------------------------------

@dataclass
class ObjectTransition:
    """An observed correspondence between an input and output object.

    Bridges clause-style correspondence reasoning into the typed IR.
    """
    in_obj: Any         # ObjFact (input side, None if created)
    out_obj: Any        # ObjFact (output side, None if removed)
    match_type: str     # "identical", "recolored", "moved", "moved_recolored", "transformed", "modified", "removed", "new"
    dr: int = 0         # row offset (out.row - in.row)
    dc: int = 0         # col offset (out.col - in.col)
    color_from: int = -1
    color_to: int = -1
    transform: str = None  # for "transformed": flip_h, flip_v, rot90, etc.


def compute_transitions(in_facts, out_facts):
    """Build ObjectTransition list from correspondence.

    This is the typed bridge: correspondence output → first-class typed values.
    """
    from aria.guided.correspond import map_output_to_input, find_removed_objects

    mappings = map_output_to_input(out_facts, in_facts)
    transitions = []

    for m in mappings:
        dr = (m.out_obj.row - m.in_obj.row) if m.in_obj else 0
        dc = (m.out_obj.col - m.in_obj.col) if m.in_obj else 0
        transitions.append(ObjectTransition(
            in_obj=m.in_obj,
            out_obj=m.out_obj,
            match_type=m.match_type,
            dr=dr, dc=dc,
            color_from=m.color_from,
            color_to=m.color_to,
            transform=m.transform,
        ))

    # Add removed objects
    for in_obj in find_removed_objects(in_facts, mappings):
        transitions.append(ObjectTransition(
            in_obj=in_obj, out_obj=None,
            match_type="removed",
            color_from=in_obj.color, color_to=-1,
        ))

    return transitions


# ---------------------------------------------------------------------------
# Typed operations (the building blocks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypedOp:
    """A typed operation with input/output type signatures."""
    name: str
    in_types: tuple       # input type signature
    out_type: Ty          # output type
    fn: Any               # callable
    description: str = ""


# All available operations
def _build_op_library(bg=0):
    """Build the library of typed operations.

    Each op has a signature (input types → output type) and a function.
    The synthesizer composes these where types match.
    bg: background color from demos, used for render and combine operations.
    """
    ops = []

    # --- Extractors: Grid → derived values ---
    ops.append(TypedOp(
        'perceive', (Ty.GRID,), Ty.OBJECTS,
        lambda grid: perceive(grid).objects,
        'get all objects from grid',
    ))

    ops.append(TypedOp(
        'get_facts', (Ty.GRID,), Ty.FACTS,
        lambda grid: perceive(grid),
        'perceive grid facts',
    ))

    # --- Selection: Facts × Selector → Object ---
    ops.append(TypedOp(
        'select', (Ty.FACTS, Ty.SELECTOR), Ty.OBJECT,
        lambda facts, sel: _safe_select(facts, sel),
        'select object by predicate',
    ))

    # --- Crop: Grid × Object → Region ---
    ops.append(TypedOp(
        'crop_bbox', (Ty.GRID, Ty.OBJECT), Ty.REGION,
        lambda grid, obj: prim_crop_bbox(grid, obj),
        'crop object bounding box',
    ))

    # --- Frame + Interior: Grid × Object → Region ---
    ops.append(TypedOp(
        'crop_interior', (Ty.GRID, Ty.OBJECT), Ty.REGION,
        lambda grid, obj: _safe_crop_interior(grid, obj),
        'crop interior of object frame',
    ))

    # --- Split: Grid → Pair ---
    ops.append(TypedOp(
        'split', (Ty.GRID,), Ty.PAIR,
        lambda grid: _safe_split(grid),
        'split grid at separator',
    ))

    # --- Mirror across separator: Grid → Grid ---
    from aria.guided.dsl import prim_mirror_across
    for mirror_mode in ['col', 'col_rtl', 'row', 'row_btt']:
        ops.append(TypedOp(
            f'mirror_{mirror_mode}', (Ty.GRID,), Ty.GRID,
            lambda grid, _mm=mirror_mode: _safe_mirror(grid, _mm),
            f'mirror grid {mirror_mode} at separator',
        ))

    # --- Combine: Pair → Mask ---
    for op_name in ['and', 'or', 'xor', 'diff', 'rdiff']:
        ops.append(TypedOp(
            f'combine_{op_name}', (Ty.PAIR,), Ty.MASK,
            lambda pair, _op=op_name: _safe_combine(pair, _op),
            f'combine regions with {op_name}',
        ))

    # --- Render: Mask × Color → Grid ---
    ops.append(TypedOp(
        'render', (Ty.MASK, Ty.COLOR), Ty.GRID,
        lambda mask, color, _bg=bg: _safe_render(mask, color, _bg),
        'render mask with color',
    ))

    # --- Repair all frame interiors in the grid ---
    ops.append(TypedOp(
        'repair_all_frames', (Ty.GRID,), Ty.GRID,
        lambda grid: _repair_all_frames(grid),
        'find all frames, repair their periodic interiors',
    ))

    # --- Pair decomposition: extract halves ---
    ops.append(TypedOp(
        'first', (Ty.PAIR,), Ty.REGION,
        lambda pair: pair[0] if pair else None,
        'first element of pair',
    ))
    ops.append(TypedOp(
        'second', (Ty.PAIR,), Ty.REGION,
        lambda pair: pair[1] if pair else None,
        'second element of pair',
    ))

    # --- Grid transforms (not just Region) ---
    # flip_h = horizontal flip = reverse columns; flip_v = vertical flip = reverse rows
    for xform_name, xfn in [('flip_h', lambda r: r[:, ::-1]),
                              ('flip_v', lambda r: r[::-1, :]),
                              ('flip_hv', lambda r: r[::-1, ::-1]),
                              ('rot90', lambda r: np.rot90(r)),
                              ('rot180', lambda r: np.rot90(r, 2))]:
        ops.append(TypedOp(
            f'grid_{xform_name}', (Ty.GRID,), Ty.GRID,
            xfn,
            f'transform grid: {xform_name}',
        ))

    # --- Object attribute extraction ---
    ops.append(TypedOp(
        'get_color', (Ty.OBJECT,), Ty.COLOR,
        lambda obj: obj.color,
        'get object color',
    ))

    ops.append(TypedOp(
        'get_mask', (Ty.OBJECT,), Ty.MASK,
        lambda obj: obj.mask,
        'get object mask (bool)',
    ))

    # --- Transforms: Region → Region ---
    for xform_name, xfn in [('flip_h', lambda r: r[:, ::-1]),
                              ('flip_v', lambda r: r[::-1, :]),
                              ('flip_hv', lambda r: r[::-1, ::-1]),
                              ('rot90', lambda r: np.rot90(r)),
                              ('rot180', lambda r: np.rot90(r, 2)),
                              ('transpose', lambda r: r.T)]:
        ops.append(TypedOp(
            f'transform_{xform_name}', (Ty.REGION,), Ty.REGION,
            xfn,
            f'transform region: {xform_name}',
        ))

    # --- Binding constructors: Facts → Binding ---
    # bind_contains: find (host, content) pairs via containment
    ops.append(TypedOp(
        'bind_contains', (Ty.FACTS,), Ty.BINDING,
        lambda facts: _bind_contains(facts),
        'find containment pair (host, content)',
    ))
    ops.append(TypedOp(
        'bind_adjacent', (Ty.FACTS,), Ty.BINDING,
        lambda facts: _bind_adjacent(facts),
        'find adjacent pair',
    ))
    ops.append(TypedOp(
        'bind_same_shape', (Ty.FACTS,), Ty.BINDING,
        lambda facts: _bind_same_shape(facts),
        'find same-shape pair',
    ))

    # --- Binding decomposition: Binding → Object ---
    ops.append(TypedOp(
        'binding_host', (Ty.BINDING,), Ty.OBJECT,
        lambda b: b[0] if b else None,
        'get host (first) from binding',
    ))
    ops.append(TypedOp(
        'binding_content', (Ty.BINDING,), Ty.OBJECT,
        lambda b: b[1] if b else None,
        'get content (second) from binding',
    ))

    # --- Binding-derived regions ---
    ops.append(TypedOp(
        'crop_host_bbox', (Ty.GRID, Ty.BINDING), Ty.REGION,
        lambda grid, b: prim_crop_bbox(grid, b[0]) if b else None,
        'crop host bounding box',
    ))
    ops.append(TypedOp(
        'crop_content_bbox', (Ty.GRID, Ty.BINDING), Ty.REGION,
        lambda grid, b: prim_crop_bbox(grid, b[1]) if b else None,
        'crop content bounding box',
    ))

    # --- Transition accessors: Transition → derived values ---
    ops.append(TypedOp(
        'transition_source', (Ty.TRANSITION,), Ty.OBJECT,
        lambda t: t.in_obj if t else None,
        'get input object from transition',
    ))
    ops.append(TypedOp(
        'transition_target', (Ty.TRANSITION,), Ty.OBJECT,
        lambda t: t.out_obj if t else None,
        'get output object from transition',
    ))
    ops.append(TypedOp(
        'transition_color_to', (Ty.TRANSITION,), Ty.COLOR,
        lambda t: t.color_to if t and t.color_to >= 0 else None,
        'get output color from transition',
    ))
    ops.append(TypedOp(
        'transition_color_from', (Ty.TRANSITION,), Ty.COLOR,
        lambda t: t.color_from if t and t.color_from >= 0 else None,
        'get input color from transition',
    ))

    return ops


def _bind_contains(facts):
    """Find the most prominent containment pair (host, content)."""
    best = None
    best_score = -1
    for p in facts.pairs:
        if p.a_contains_b:
            a = next((o for o in facts.objects if o.oid == p.oid_a), None)
            b = next((o for o in facts.objects if o.oid == p.oid_b), None)
            if a and b:
                score = a.size - b.size  # prefer largest host with smallest content
                if score > best_score:
                    best = (a, b)
                    best_score = score
        elif p.b_contains_a:
            a = next((o for o in facts.objects if o.oid == p.oid_a), None)
            b = next((o for o in facts.objects if o.oid == p.oid_b), None)
            if a and b:
                score = b.size - a.size
                if score > best_score:
                    best = (b, a)
                    best_score = score
    return best


def _bind_adjacent(facts):
    """Find the most prominent adjacent pair."""
    best = None
    best_score = -1
    for p in facts.pairs:
        if p.adjacent and not p.same_color:
            a = next((o for o in facts.objects if o.oid == p.oid_a), None)
            b = next((o for o in facts.objects if o.oid == p.oid_b), None)
            if a and b:
                score = a.size + b.size
                if score > best_score:
                    best = (a, b)
                    best_score = score
    return best


def _bind_same_shape(facts):
    """Find two objects with identical shape but different color."""
    for i, a in enumerate(facts.objects):
        for b in facts.objects[i+1:]:
            if (a.color != b.color and a.height == b.height
                    and a.width == b.width and np.array_equal(a.mask, b.mask)):
                return (a, b)
    return None


def _guess_bg(region):
    """Guess background color of a region (most common value)."""
    vals, counts = np.unique(region, return_counts=True)
    return int(vals[np.argmax(counts)])


def _repair_all_frames(grid):
    """Find all framed objects, repair their periodic interiors.

    Skips frames that contain sub-frames (only repairs innermost).
    """
    from aria.guided.dsl import prim_find_frame, prim_repair_region
    facts = perceive(grid)
    framed = [(o, prim_find_frame(o, grid)) for o in facts.objects]
    framed = [(o, f) for o, f in framed if f is not None and o.size >= 8]
    result = grid.copy()
    changed = False
    for obj, (r0, c0, r1, c1) in framed:
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        if ih < 1 or iw < 1:
            continue
        # Skip if this frame's interior contains another frame with a real interior
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        int_colors = set(int(v) for v in np.unique(interior))
        has_sub = any(
            o2.color in int_colors and o2.color != obj.color
            for o2, (fr0, fc0, fr1, fc1) in framed
            if o2.oid != obj.oid
            and fr0 > r0 and fc0 > c0 and fr1 < r1 and fc1 < c1
            and fr1 - fr0 > 2 and fc1 - fc0 > 2  # sub-frame must have real interior
        )
        if has_sub:
            continue
        repaired = prim_repair_region(interior)
        if repaired is not None and not np.array_equal(repaired, interior):
            result[r0 + 1:r1, c0 + 1:c1] = repaired
            changed = True
    return result if changed else None


def _safe_repair_interior(grid, obj, axis=None):
    """Repair the interior of an object's frame.

    Only repairs INNERMOST frames — skips if the interior contains
    other framed objects (they should be repaired independently).
    """
    from aria.guided.dsl import prim_repair_region, prim_replace_interior, prim_find_frame
    facts = perceive(grid)

    # Check: does this object's frame contain other frames?
    my_frame = prim_find_frame(obj, grid)
    if my_frame is None:
        return None
    r0, c0, r1, c1 = my_frame

    for other in facts.objects:
        if other.oid == obj.oid:
            continue
        other_frame = prim_find_frame(other, grid)
        if other_frame is None:
            continue
        or0, oc0, or1, oc1 = other_frame
        if or0 >= r0 and oc0 >= c0 and or1 <= r1 and oc1 <= c1:
            # Contains a sub-frame — repair each sub-frame instead
            result = grid.copy()
            for sub_obj in facts.objects:
                sf = prim_find_frame(sub_obj, grid)
                if sf is None:
                    continue
                sr0, sc0, sr1, sc1 = sf
                if sr0 < r0 or sc0 < c0 or sr1 > r1 or sc1 > c1:
                    continue  # not inside our frame
                if sub_obj.oid == obj.oid:
                    continue
                # Check this sub-frame doesn't contain yet another frame
                contains_sub = any(
                    prim_find_frame(o3, grid) is not None and
                    prim_find_frame(o3, grid)[0] >= sr0 and
                    prim_find_frame(o3, grid)[2] <= sr1
                    for o3 in facts.objects
                    if o3.oid != sub_obj.oid and o3.oid != obj.oid
                )
                if contains_sub:
                    continue
                ih = sr1 - sr0 - 1
                iw = sc1 - sc0 - 1
                if ih < 1 or iw < 1:
                    continue
                interior = grid[sr0+1:sr1, sc0+1:sc1]
                repaired = prim_repair_region(interior)
                if repaired is not None:
                    result[sr0+1:sr1, sc0+1:sc1] = repaired
            if not np.array_equal(result, grid):
                return result
            return None

    # Simple case: no sub-frames
    interior = _safe_crop_interior(grid, obj)
    if interior is None:
        return None
    repaired = prim_repair_region(interior)
    if repaired is None or np.array_equal(repaired, interior):
        return None
    return prim_replace_interior(grid, obj, repaired)


def _safe_select(facts, selector):
    """Select a single object from facts using a predicate list."""
    from aria.guided.dsl import prim_select
    targets = prim_select(facts, selector)
    return targets[0] if len(targets) == 1 else None


def _safe_mirror(grid, mode):
    from aria.guided.dsl import prim_mirror_across
    facts = perceive(grid)
    if not facts.separators:
        return None
    sep = facts.separators[0]
    if mode.startswith('col') and sep.axis != 'col':
        return None
    if mode.startswith('row') and sep.axis != 'row':
        return None
    return prim_mirror_across(grid, mode, sep.index)


def _safe_crop_interior(grid, obj):
    frame = prim_find_frame(obj, grid)
    if frame is None:
        return None
    return prim_crop_interior(grid, frame)


def _safe_split(grid):
    facts = perceive(grid)
    if not facts.separators:
        return None
    return prim_split_at_separator(grid, facts, 0)


def _safe_combine(pair, op):
    if pair is None:
        return None
    a, b = pair
    bg = perceive(a).bg if a.size > 0 else 0
    return prim_combine(a, b, op, bg)


def _safe_render(mask, color, bg=0):
    if mask is None:
        return None
    return prim_render_mask(mask, color, bg)


# ---------------------------------------------------------------------------
# Program nodes (IR)
# ---------------------------------------------------------------------------

@dataclass
class PNode:
    """A node in a program tree."""
    op_name: str
    args: list            # list of PNode or literal values
    result_type: Ty
    _cached: Any = field(default=None, repr=False)

    def __repr__(self):
        if not self.args:
            return self.op_name
        args_str = ', '.join(str(a) for a in self.args)
        return f"{self.op_name}({args_str})"


@dataclass
class PLiteral(PNode):
    """A literal value (input grid, constant color, etc.)."""
    value: Any = None

    def __repr__(self):
        return f"${self.op_name}"


# ---------------------------------------------------------------------------
# Grid constructors — whole-grid patterns (not per-object clauses)
# ---------------------------------------------------------------------------

def _try_tile(demos):
    """Detect if output = input tiled NxM with per-tile transforms."""
    # Check: output dims are integer multiples of input in ALL demos
    tile_sizes = []
    for inp, out in demos:
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return None
        tile_sizes.append((oh // ih, ow // iw))
    if len(set(tile_sizes)) != 1:
        return None
    tile_r, tile_c = tile_sizes[0]
    if tile_r <= 1 and tile_c <= 1:
        return None

    # Detect per-tile transforms from demo 0
    inp0, out0 = demos[0]
    ih, iw = inp0.shape
    transforms = {}
    for tr in range(tile_r):
        for tc in range(tile_c):
            tile = out0[tr * ih:(tr + 1) * ih, tc * iw:(tc + 1) * iw]
            found = False
            for xform_name, xfn in [('none', lambda g: g),
                                      ('flip_h', lambda g: g[::-1, :]),
                                      ('flip_v', lambda g: g[:, ::-1]),
                                      ('flip_hv', lambda g: g[::-1, ::-1])]:
                if np.array_equal(tile, xfn(inp0)):
                    if xform_name != 'none':
                        transforms[(tr, tc)] = xform_name
                    found = True
                    break
            if not found:
                return None

    # Verify on ALL demos
    for inp, out in demos:
        ih, iw = inp.shape
        for tr in range(tile_r):
            for tc in range(tile_c):
                xform = transforms.get((tr, tc), 'none')
                tile_src = inp.copy()
                if xform == 'flip_h':
                    tile_src = tile_src[::-1, :]
                elif xform == 'flip_v':
                    tile_src = tile_src[:, ::-1]
                elif xform == 'flip_hv':
                    tile_src = tile_src[::-1, ::-1]
                actual = out[tr * ih:(tr + 1) * ih, tc * iw:(tc + 1) * iw]
                if not np.array_equal(actual, tile_src):
                    return None

    xform_desc = ', '.join(f'{k}={v}' for k, v in sorted(transforms.items())) or 'no transforms'
    _tr, _tc, _xf = tile_r, tile_c, dict(transforms)

    def _exec(inp):
        ih, iw = inp.shape
        bg = perceive(inp).bg
        tiled = np.full((ih * _tr, iw * _tc), bg, dtype=np.uint8)
        for tr in range(_tr):
            for tc in range(_tc):
                xform = _xf.get((tr, tc), 'none')
                tile = inp.copy()
                if xform == 'flip_h':
                    tile = tile[::-1, :]
                elif xform == 'flip_v':
                    tile = tile[:, ::-1]
                elif xform == 'flip_hv':
                    tile = tile[::-1, ::-1]
                elif xform == 'rot90':
                    tile = np.rot90(tile)
                tiled[tr * ih:(tr + 1) * ih, tc * iw:(tc + 1) * iw] = tile
        return tiled

    return _make_program(_exec, f"tile {tile_r}x{tile_c} ({xform_desc})")


def _try_self_tile(demos):
    """Detect if output = input used as its own tile placement mask."""
    for inp, out in demos:
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return None
        tr, tc = oh // ih, ow // iw
        if tr != ih or tc != iw:
            return None

    bg = perceive(demos[0][0]).bg
    for inp, out in demos:
        ih, iw = inp.shape
        for r in range(ih):
            for c in range(iw):
                tile = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
                if inp[r, c] != bg:
                    if not np.array_equal(tile, inp):
                        return None
                else:
                    if not np.all(tile == bg):
                        return None

    def _exec(inp):
        facts = perceive(inp)
        ih, iw = inp.shape
        tiled = np.full((ih * ih, iw * iw), facts.bg, dtype=np.uint8)
        for r in range(ih):
            for c in range(iw):
                if inp[r, c] != facts.bg:
                    tiled[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw] = inp
        return tiled

    return _make_program(_exec, "self-tile (input as placement mask)")


def _try_periodic_extend(demos):
    """Detect if output = input extended by continuing its period."""
    # All demos must have same-width output and input, output taller
    for inp, out in demos:
        if inp.shape[1] != out.shape[1]:
            return None
        if out.shape[0] <= inp.shape[0]:
            return None

    # Detect recolor: consistent color mapping across demos
    color_map = {}
    for inp, out in demos:
        ih = inp.shape[0]
        for r in range(ih):
            for c in range(inp.shape[1]):
                iv, ov = int(inp[r, c]), int(out[r, c])
                if iv != 0 and ov != 0 and iv != ov:
                    if iv in color_map and color_map[iv] != ov:
                        return None
                    color_map[iv] = ov

    # Verify: recolored input matches output prefix, and output is periodic
    for inp, out in demos:
        ih, iw = inp.shape
        oh = out.shape[0]
        src = inp.copy()
        for old_c, new_c in color_map.items():
            src[inp == old_c] = new_c
        if not np.array_equal(out[:ih, :], src):
            return None
        found_period = False
        for period in range(1, ih + 1):
            tile = out[:period, :]
            match = True
            for start in range(period, oh, period):
                end = min(start + period, oh)
                if not np.array_equal(out[start:end, :], tile[:end - start, :]):
                    match = False
                    break
            if match:
                found_period = True
                break
        if not found_period:
            return None

    # Need output shape — infer from demos (must be consistent rule)
    from aria.guided.output_size import infer_output_size
    size_rule = infer_output_size(demos)
    _cmap = dict(color_map)

    desc = "periodic extend"
    if color_map:
        desc += f" (recolor {color_map})"

    def _exec(inp):
        facts = perceive(inp)
        ih, iw = inp.shape
        # Determine output size
        if size_rule is not None:
            oh, ow = size_rule.predict(facts)
        else:
            oh, ow = ih, iw  # fallback
        # Apply recolor
        src = inp.copy()
        for old_c, new_c in _cmap.items():
            src[inp == old_c] = new_c
        # Find shortest period
        for period in range(1, ih + 1):
            tile = src[:period, :]
            match = True
            for start in range(period, ih, period):
                end = min(start + period, ih)
                if not np.array_equal(src[start:end, :], tile[:end - start, :]):
                    match = False
                    break
            if match:
                break
        # Build output by repeating
        extended = np.full((oh, ow), facts.bg, dtype=np.uint8)
        for start in range(0, oh, period):
            end = min(start + period, oh)
            extended[start:end, :ow] = tile[:end - start, :ow]
        return extended

    return _make_program(_exec, desc)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def synthesize(
    demos: list[tuple[Grid, Grid]],
    max_depth: int = 3,
) -> Program | None:
    """Synthesize a program from demo pairs.

    Pipeline:
    1. Grid constructors (tile, periodic_extend) — whole-grid patterns
    2. Clause engine — relational per-object rules
    3. Bottom-up typed synthesis — composable primitive search
    """
    # Phase 1: grid constructors (tile, periodic extend)
    prog = _try_tile(demos)
    if prog:
        return prog
    prog = _try_self_tile(demos)
    if prog:
        return prog
    prog = _try_periodic_extend(demos)
    if prog:
        return prog

    # Phase 2: clause engine (per-object relational rules)
    from aria.guided.induce import induce_program
    clause_prog = induce_program(demos)
    if clause_prog is not None:
        all_ok = all(
            np.array_equal(clause_prog.execute(inp), out)
            for inp, out in demos
        )
        if all_ok:
            desc = '; '.join(c.description for c in clause_prog.clauses)
            return _make_program(clause_prog.execute, f"clause: {desc}")

    # Phase 3: bottom-up typed synthesis
    demo_bg = perceive(demos[0][0]).bg
    op_lib = _build_op_library(bg=demo_bg)
    prog = _bottom_up_search(demos, op_lib, max_depth)
    if prog:
        return prog

    return None


def _bottom_up_search(demos, op_lib, max_depth):
    """Bottom-up enumeration of typed programs.

    Level 0: input grid, constants (colors from output)
    Level 1: apply one op to level-0 values
    Level 2: apply one op to level-0 or level-1 values
    ...

    At each level, check if any value matches the output.
    Cross-demo: a value must match the output in ALL demos.
    """
    # Infer output size rule (needed for different-shape task support)
    from aria.guided.output_size import infer_output_size
    size_rule = infer_output_size(demos)

    # Per-demo execution contexts
    contexts = []
    for inp, out in demos:
        facts = perceive(inp)
        ctx = {
            'input': inp,
            'output': out,
            'facts': facts,
            'size_rule': size_rule,
        }
        contexts.append(ctx)

    # Level 0: seed values
    # For each demo: the input grid, all objects, all colors from output
    seeds_per_demo = []
    for ctx in contexts:
        seeds = {}
        seeds[('input', Ty.GRID)] = ctx['input']
        # Facts (first-class perception output)
        seeds[('facts', Ty.FACTS)] = ctx['facts']
        # Individual objects
        for i, obj in enumerate(ctx['facts'].objects):
            seeds[(f'obj_{i}', Ty.OBJECT)] = obj
        # Predicate-selected objects (dynamic)
        _add_selected_objects(seeds, ctx['facts'])
        # Selector constants (predicate lists)
        _add_selectors(seeds, ctx['facts'])
        # Colors from the output (constants available for construction)
        out_colors = set(int(v) for v in np.unique(ctx['output'])) - {ctx['facts'].bg}
        for c in out_colors:
            seeds[(f'color_{c}', Ty.COLOR)] = c
        seeds_per_demo.append(seeds)

    # Track all values at each level: key → list of per-demo values
    # A key is valid if it exists in ALL demos
    all_values = {}  # key → [val_demo0, val_demo1, ...]
    for key_ty in seeds_per_demo[0]:
        key, ty = key_ty
        vals = []
        valid = True
        for demo_seeds in seeds_per_demo:
            if key_ty in demo_seeds:
                vals.append(demo_seeds[key_ty])
            else:
                valid = False
                break
        if valid:
            all_values[(key, ty)] = vals

    # Check level 0: does any seed value match the output?
    match = _check_output_match(all_values, contexts)
    if match:
        return match

    # Enumerate higher levels
    for level in range(1, max_depth + 1):
        new_values = _apply_ops(all_values, op_lib, contexts)
        if not new_values:
            break
        all_values.update(new_values)

        match = _check_output_match(new_values, contexts)
        if match:
            return match

    # Check write-back: output = input with a sub-region replaced by a computed value
    writeback = _check_writeback(all_values, contexts)
    if writeback:
        return writeback

    # Try transition-based programs: use correspondence to discover structure
    transition_match = _search_transition_programs(contexts)
    if transition_match:
        return transition_match

    # Try conditional dispatch: selector A → op X, selector B → op Y
    cond_match = _search_conditional_dispatch(all_values, contexts)
    if cond_match:
        return cond_match

    # Try MAP: per-row or per-object operations with derived parameters
    map_match = _search_map(contexts)
    if map_match:
        return map_match

    # Try FOR_EACH loops: apply a per-object operation across all matching objects
    loop_match = _search_loops(all_values, contexts)
    if loop_match:
        return loop_match

    return None


def _add_selected_objects(seeds, facts):
    """Add predicate-selected objects as seed values."""
    from aria.guided.clause import Predicate, Pred

    # Common single predicates
    predicates = {
        'largest': [Predicate(Pred.IS_LARGEST)],
        'smallest': [Predicate(Pred.IS_SMALLEST)],
        'unique_color': [Predicate(Pred.UNIQUE_COLOR)],
    }

    for name, preds in predicates.items():
        selected = prim_select(facts, preds)
        if len(selected) == 1:
            seeds[(f'sel_{name}', Ty.OBJECT)] = selected[0]


def _add_selectors(seeds, facts):
    """Add predicate selector constants for typed composition."""
    from aria.guided.clause import Predicate, Pred

    selectors = {
        'sel_largest': [Predicate(Pred.IS_LARGEST)],
        'sel_smallest': [Predicate(Pred.IS_SMALLEST)],
        'sel_unique_color': [Predicate(Pred.UNIQUE_COLOR)],
        'sel_topmost': [Predicate(Pred.IS_TOPMOST)],
        'sel_bottommost': [Predicate(Pred.IS_BOTTOMMOST)],
        'sel_leftmost': [Predicate(Pred.IS_LEFTMOST)],
        'sel_rightmost': [Predicate(Pred.IS_RIGHTMOST)],
    }

    # Color-based selectors for each color present
    for obj in facts.objects:
        name = f'sel_color_{obj.color}'
        if (name, Ty.SELECTOR) not in seeds:
            selectors[name] = [Predicate(Pred.COLOR_EQ, obj.color)]

    # Relational selectors (contained_by, contains, adjacent_to largest/smallest)
    for rel, rel_name in [(Pred.CONTAINED_BY, 'inside'),
                           (Pred.CONTAINS, 'hosts'),
                           (Pred.ADJACENT_TO, 'near')]:
        for inner_name, inner_pred in [('largest', Predicate(Pred.IS_LARGEST)),
                                        ('smallest', Predicate(Pred.IS_SMALLEST))]:
            sel = [Predicate(rel, inner_pred)]
            # Only add if it selects at least one object
            from aria.guided.dsl import prim_select
            if prim_select(facts, sel):
                selectors[f'sel_{rel_name}_{inner_name}'] = sel

    for name, preds in selectors.items():
        seeds[(name, Ty.SELECTOR)] = preds


def _search_transition_programs(contexts):
    """Search for programs by analyzing object transitions across demos.

    Uses correspondence to build ObjectTransition values, then discovers
    consistent patterns that can be expressed as typed programs.

    This bridges the gap between clause-style correspondence reasoning
    and typed program synthesis.
    """
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    n_demos = len(contexts)
    same_shape = all(ctx['input'].shape == ctx['output'].shape for ctx in contexts)
    size_rule = contexts[0].get('size_rule')
    if not same_shape and size_rule is None:
        return None

    # Compute transitions for all demos
    all_transitions = []
    for ctx in contexts:
        in_facts = ctx['facts']
        out_facts = perceive(ctx['output'])
        transitions = compute_transitions(in_facts, out_facts)
        all_transitions.append(transitions)

    if not all_transitions[0]:
        return None

    # --- Strategy 1: Uniform transition ---
    # All non-identical objects undergo the same action (e.g., all moved by same offset)
    prog = _try_uniform_transition(all_transitions, contexts)
    if prog:
        return prog

    # --- Strategy 2: Stamp/creation transitions ---
    # New objects in output whose shape matches an input object = stamp at offset
    prog = _try_stamp_transitions(all_transitions, contexts)
    if prog:
        return prog

    # --- Strategy 3: Multi-group transition dispatch ---
    # Different object groups undergo different transitions, partitioned by selectors
    prog = _try_transition_dispatch(all_transitions, contexts)
    if prog:
        return prog

    return None


def _try_uniform_transition(all_transitions, contexts):
    """Check if all non-identical transitions share the same action + parameters."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    for demo_transitions in all_transitions:
        changed = [t for t in demo_transitions if t.match_type != 'identical']
        if not changed:
            continue

        # All changed objects must have the same match_type
        types = set(t.match_type for t in changed)
        if len(types) != 1:
            return None

    # Check cross-demo consistency of the transition parameters
    match_type = None
    for demo_transitions in all_transitions:
        changed = [t for t in demo_transitions if t.match_type != 'identical']
        if not changed:
            return None
        types = set(t.match_type for t in changed)
        if len(types) != 1:
            return None
        mt = next(iter(types))
        if match_type is None:
            match_type = mt
        elif mt != match_type:
            return None

    if match_type == 'moved':
        prog = _try_uniform_move(all_transitions, contexts)
        if prog:
            return prog
        return _try_gravity_or_slide(all_transitions, contexts, 'moved')
    if match_type == 'recolored':
        return _try_uniform_recolor(all_transitions, contexts)
    if match_type == 'transformed':
        return _try_uniform_transform(all_transitions, contexts)
    if match_type == 'moved_recolored':
        prog = _try_uniform_move_recolor(all_transitions, contexts)
        if prog:
            return prog
        return _try_gravity_or_slide(all_transitions, contexts, 'moved_recolored')

    return None


def _try_uniform_move(all_transitions, contexts):
    """All changed objects moved by the same constant offset."""
    offsets = set()
    for demo_transitions in all_transitions:
        moved = [t for t in demo_transitions if t.match_type == 'moved']
        if not moved:
            return None
        demo_offsets = set((t.dr, t.dc) for t in moved)
        if len(demo_offsets) != 1:
            return None  # different objects moved different amounts
        offsets.add(next(iter(demo_offsets)))

    if len(offsets) != 1:
        return None  # offset differs across demos
    dr, dc = next(iter(offsets))

    # Find selector for moved objects
    ctx0 = contexts[0]
    moved_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == 'moved')
    sel = _find_transition_selector(moved_oids, ctx0['facts'])
    if sel is None:
        return None

    def _make_fn(_sel, _dr, _dc):
        def _exec(inp):
            facts = perceive(inp)
            bg = facts.bg
            result = inp.copy()
            targets = prim_select(facts, _sel)
            # Clear original positions
            for obj in targets:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            result[obj.row + r, obj.col + c] = bg
            # Place at new positions
            for obj in targets:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            nr, nc = obj.row + r + _dr, obj.col + c + _dc
                            if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                result[nr, nc] = obj.color
            return result
        return _exec

    fn = _make_fn(sel, dr, dc)
    if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
        desc = _describe_selector(sel)
        return _make_program(fn, f"synth: transition move [{desc}] by ({dr},{dc})")
    return None


def _try_gravity_or_slide(all_transitions, contexts, match_type):
    """Handle movement where offset varies — gravity (to border) or slide (until collision)."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    # Detect gravity: all moved objects end up at the same border
    # AND actually moved in that direction (not trivially at border due to full span)
    for direction in ['down', 'up', 'right', 'left']:
        consistent = True
        any_actually_moved = False
        for demo_idx, demo_transitions in enumerate(all_transitions):
            moved = [t for t in demo_transitions if t.match_type == match_type]
            if not moved:
                consistent = False
                break
            rows, cols = contexts[demo_idx]['input'].shape
            for t in moved:
                o = t.out_obj
                # Must be at the target border
                at_border = False
                if direction == 'down' and o.row + o.height == rows:
                    at_border = True
                elif direction == 'up' and o.row == 0:
                    at_border = True
                elif direction == 'right' and o.col + o.width == cols:
                    at_border = True
                elif direction == 'left' and o.col == 0:
                    at_border = True
                if not at_border:
                    consistent = False
                    break
                # Must have moved in the matching direction
                if direction == 'down' and t.dr > 0:
                    any_actually_moved = True
                elif direction == 'up' and t.dr < 0:
                    any_actually_moved = True
                elif direction == 'right' and t.dc > 0:
                    any_actually_moved = True
                elif direction == 'left' and t.dc < 0:
                    any_actually_moved = True
            if not consistent:
                break
        if not any_actually_moved:
            consistent = False

        if consistent:
            ctx0 = contexts[0]
            moved_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == match_type)
            sel = _find_transition_selector(moved_oids, ctx0['facts'])
            if sel is None:
                continue

            # Determine if recolored too
            recolor = None
            if match_type == 'moved_recolored':
                colors = set(t.color_to for t in all_transitions[0] if t.match_type == match_type)
                if len(colors) == 1:
                    recolor = next(iter(colors))
                else:
                    continue

            def _make_gravity_fn(_sel, _dir, _recolor):
                def _exec(inp):
                    facts = perceive(inp)
                    bg = facts.bg
                    result = inp.copy()
                    targets = prim_select(facts, _sel)
                    rows, cols = inp.shape
                    for obj in targets:
                        # Clear original
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = bg
                    for obj in targets:
                        color = _recolor if _recolor is not None else obj.color
                        if _dir == 'down':
                            nr = rows - obj.height
                        elif _dir == 'up':
                            nr = 0
                        elif _dir == 'right':
                            nc = cols - obj.width
                            nr = obj.row
                        elif _dir == 'left':
                            nc = 0
                            nr = obj.row
                        if _dir in ('down', 'up'):
                            nc = obj.col
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.mask[r, c]:
                                    pr, pc = nr + r, nc + c
                                    if 0 <= pr < rows and 0 <= pc < cols:
                                        result[pr, pc] = color
                    return result
                return _exec

            fn = _make_gravity_fn(sel, direction, recolor)
            if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
                desc = _describe_selector(sel)
                rc_desc = f' recolor({recolor})' if recolor else ''
                return _make_program(fn, f"synth: transition gravity({direction}) [{desc}]{rc_desc}")

    # Detect slide: objects move in a consistent direction until hitting non-bg
    for direction in ['down', 'up', 'right', 'left']:
        consistent = True
        for demo_transitions in all_transitions:
            moved = [t for t in demo_transitions if t.match_type == match_type]
            if not moved:
                consistent = False
                break
            for t in moved:
                # Check direction consistency
                if direction == 'down' and t.dr <= 0:
                    consistent = False
                elif direction == 'up' and t.dr >= 0:
                    consistent = False
                elif direction == 'right' and t.dc <= 0:
                    consistent = False
                elif direction == 'left' and t.dc >= 0:
                    consistent = False
                if not consistent:
                    break
            if not consistent:
                break

        if not consistent:
            continue

        ctx0 = contexts[0]
        moved_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == match_type)
        sel = _find_transition_selector(moved_oids, ctx0['facts'])
        if sel is None:
            continue

        recolor = None
        if match_type == 'moved_recolored':
            colors = set(t.color_to for t in all_transitions[0] if t.match_type == match_type)
            if len(colors) == 1:
                recolor = next(iter(colors))
            else:
                continue

        def _make_slide_fn(_sel, _dir, _recolor):
            def _exec(inp):
                facts = perceive(inp)
                bg = facts.bg
                result = inp.copy()
                targets = prim_select(facts, _sel)
                rows, cols = inp.shape
                dr = {'down': 1, 'up': -1, 'right': 0, 'left': 0}[_dir]
                dc = {'down': 0, 'up': 0, 'right': 1, 'left': -1}[_dir]
                for obj in targets:
                    # Clear original
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                result[obj.row + r, obj.col + c] = bg
                for obj in targets:
                    color = _recolor if _recolor is not None else obj.color
                    # Slide until hitting non-bg or border
                    shift = _compute_slide_shift(obj, dr, dc, inp, bg, rows, cols)
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                pr = obj.row + r + shift * dr
                                pc = obj.col + c + shift * dc
                                if 0 <= pr < rows and 0 <= pc < cols:
                                    result[pr, pc] = color
                return result
            return _exec

        fn = _make_slide_fn(sel, direction, recolor)
        if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
            desc = _describe_selector(sel)
            rc_desc = f' recolor({recolor})' if recolor else ''
            return _make_program(fn, f"synth: transition slide({direction}) [{desc}]{rc_desc}")

    return None


def _compute_slide_shift(obj, dr, dc, grid, bg, rows, cols):
    """Compute how far an object can slide in direction (dr,dc) before hitting non-bg."""
    max_shift = max(rows, cols)
    for shift in range(1, max_shift):
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    pr = obj.row + r + shift * dr
                    pc = obj.col + c + shift * dc
                    if pr < 0 or pr >= rows or pc < 0 or pc >= cols:
                        return shift - 1
                    # Check if target cell is non-bg AND not part of this object
                    if grid[pr, pc] != bg:
                        # Is this cell part of the same object?
                        lr, lc = pr - obj.row, pc - obj.col
                        if 0 <= lr < obj.height and 0 <= lc < obj.width and obj.mask[lr, lc]:
                            continue  # same object cell — keep going
                        return shift - 1
    return max_shift - 1


def _try_uniform_transform(all_transitions, contexts):
    """All changed objects undergo the same geometric transform in place."""
    from aria.guided.dsl import prim_select

    # Check: all transformed objects use the same transform across all demos
    xforms = set()
    for demo_transitions in all_transitions:
        transformed = [t for t in demo_transitions if t.match_type == 'transformed']
        if not transformed:
            return None
        demo_xforms = set(t.transform for t in transformed)
        if len(demo_xforms) != 1:
            return None
        xforms.add(next(iter(demo_xforms)))

    if len(xforms) != 1:
        return None
    xform = next(iter(xforms))

    # Find selector for transformed objects
    ctx0 = contexts[0]
    xform_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == 'transformed')
    sel = _find_transition_selector(xform_oids, ctx0['facts'])
    if sel is None:
        return None

    _xform_fns = {
        'flip_h': lambda m: m[:, ::-1],
        'flip_v': lambda m: m[::-1, :],
        'flip_hv': lambda m: m[::-1, ::-1],
        'rot90': lambda m: np.rot90(m),
        'rot180': lambda m: np.rot90(m, 2),
        'rot270': lambda m: np.rot90(m, 3),
    }

    def _make_fn(_sel, _xform):
        def _exec(inp):
            facts = perceive(inp)
            bg = facts.bg
            result = inp.copy()
            targets = prim_select(facts, _sel)
            xfn = _xform_fns.get(_xform)
            if xfn is None:
                return result
            for obj in targets:
                # Clear original
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            result[obj.row + r, obj.col + c] = bg
                # Place transformed
                new_mask = xfn(obj.mask)
                mh, mw = new_mask.shape
                for r in range(mh):
                    for c in range(mw):
                        if new_mask[r, c]:
                            nr, nc = obj.row + r, obj.col + c
                            if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                result[nr, nc] = obj.color
            return result
        return _exec

    fn = _make_fn(sel, xform)
    if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
        desc = _describe_selector(sel)
        return _make_program(fn, f"synth: transition transform({xform}) [{desc}]")
    return None


def _try_uniform_recolor(all_transitions, contexts):
    """All changed objects recolored — check if new color is derivable."""
    # Check: constant color across all demos?
    colors = set()
    for demo_transitions in all_transitions:
        recolored = [t for t in demo_transitions if t.match_type == 'recolored']
        for t in recolored:
            colors.add(t.color_to)

    if len(colors) == 1:
        # Constant recolor — clause engine likely handles this
        return None  # let clause engine handle it

    # Variable color — check if it's derivable from a support object
    # For each demo, find what the new color is and where it comes from
    for demo_idx, demo_transitions in enumerate(all_transitions):
        recolored = [t for t in demo_transitions if t.match_type == 'recolored']
        if not recolored:
            return None
        demo_colors = set(t.color_to for t in recolored)
        if len(demo_colors) != 1:
            return None  # different objects get different colors in same demo

    # All recolored objects get the same color per demo, but it varies across demos
    # Check: is this color = color of a structurally identifiable support object?
    ctx0 = contexts[0]
    demo0_color = next(iter(set(t.color_to for t in all_transitions[0] if t.match_type == 'recolored')))

    # Find which object has this color in the input
    from aria.guided.clause import Predicate, Pred
    source_preds = None
    for pred_list in [[Predicate(Pred.IS_LARGEST)], [Predicate(Pred.IS_SMALLEST)],
                       [Predicate(Pred.UNIQUE_COLOR)]]:
        support = prim_select(ctx0['facts'], pred_list)
        if support and support[0].color == demo0_color:
            # Verify across demos
            consistent = True
            for i, ctx in enumerate(contexts[1:], 1):
                demo_color = next(iter(set(t.color_to for t in all_transitions[i] if t.match_type == 'recolored')))
                s = prim_select(ctx['facts'], pred_list)
                if not s or s[0].color != demo_color:
                    consistent = False
                    break
            if consistent:
                source_preds = pred_list
                break

    if source_preds is None:
        return None

    # Find selector for recolored objects
    recolor_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == 'recolored')
    sel = _find_transition_selector(recolor_oids, ctx0['facts'])
    if sel is None:
        return None

    def _make_fn(_sel, _src_preds):
        def _exec(inp):
            facts = perceive(inp)
            result = inp.copy()
            support = prim_select(facts, _src_preds)
            if not support:
                return result
            new_color = support[0].color
            targets = prim_select(facts, _sel)
            for obj in targets:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            result[obj.row + r, obj.col + c] = new_color
            return result
        return _exec

    fn = _make_fn(sel, source_preds)
    if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
        sel_desc = _describe_selector(sel)
        src_desc = _describe_selector(source_preds)
        return _make_program(fn, f"synth: transition recolor [{sel_desc}] to color_of [{src_desc}]")
    return None


def _try_uniform_move_recolor(all_transitions, contexts):
    """All changed objects moved + recolored — combine offset + color derivation."""
    # Check constant offset
    offsets = set()
    for demo_transitions in all_transitions:
        changed = [t for t in demo_transitions if t.match_type == 'moved_recolored']
        if not changed:
            return None
        demo_offsets = set((t.dr, t.dc) for t in changed)
        if len(demo_offsets) != 1:
            return None
        offsets.add(next(iter(demo_offsets)))

    if len(offsets) != 1:
        return None
    dr, dc = next(iter(offsets))

    # Check derived color (same logic as uniform recolor)
    colors = set()
    for demo_transitions in all_transitions:
        changed = [t for t in demo_transitions if t.match_type == 'moved_recolored']
        for t in changed:
            colors.add(t.color_to)

    if len(colors) == 1:
        new_color = next(iter(colors))
        # Constant color + constant offset
        ctx0 = contexts[0]
        moved_oids = set(t.in_obj.oid for t in all_transitions[0] if t.match_type == 'moved_recolored')
        sel = _find_transition_selector(moved_oids, ctx0['facts'])
        if sel is None:
            return None

        def _make_fn(_sel, _dr, _dc, _color):
            def _exec(inp):
                facts = perceive(inp)
                bg = facts.bg
                result = inp.copy()
                targets = prim_select(facts, _sel)
                for obj in targets:
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                result[obj.row + r, obj.col + c] = bg
                for obj in targets:
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                nr, nc = obj.row + r + _dr, obj.col + c + _dc
                                if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                    result[nr, nc] = _color
                return result
            return _exec

        fn = _make_fn(sel, dr, dc, new_color)
        if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
            desc = _describe_selector(sel)
            return _make_program(fn, f"synth: transition move+recolor [{desc}] by ({dr},{dc}) color={new_color}")

    return None


def _try_stamp_transitions(all_transitions, contexts):
    """Handle new objects whose shape matches an input object — stamp at offset.

    Pattern: an input object is duplicated at a new position (possibly recolored).
    The stamp offset may be constant or derived from structure.
    """
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    for demo_transitions in all_transitions:
        new_objs = [t for t in demo_transitions if t.match_type == 'new']
        if not new_objs:
            return None

    # For each new output object, find the input object with the same shape
    # and compute the offset
    stamp_patterns = []  # per-demo list of (source_obj, new_obj, dr, dc, color)
    for demo_idx, demo_transitions in enumerate(all_transitions):
        facts = contexts[demo_idx]['facts']
        new_objs = [t for t in demo_transitions if t.match_type == 'new']
        demo_stamps = []
        for t in new_objs:
            out = t.out_obj
            # Find matching input object by shape
            source = None
            for in_obj in facts.objects:
                if (in_obj.height == out.height and in_obj.width == out.width
                        and np.array_equal(in_obj.mask, out.mask)):
                    source = in_obj
                    break
            if source is None:
                return None  # can't find source for this new object
            dr = out.row - source.row
            dc = out.col - source.col
            demo_stamps.append((source, out, dr, dc, out.color))
        stamp_patterns.append(demo_stamps)

    if not stamp_patterns[0]:
        return None

    # Check: constant offset across all stamps within and across demos?
    all_offsets = set()
    for demo_stamps in stamp_patterns:
        for _, _, dr, dc, _ in demo_stamps:
            all_offsets.add((dr, dc))

    if len(all_offsets) == 1:
        dr, dc = next(iter(all_offsets))
        # Constant stamp offset — find selector for source objects
        source_oids = set(s.oid for s, _, _, _, _ in stamp_patterns[0])
        sel = _find_transition_selector(source_oids, contexts[0]['facts'])
        if sel is None:
            return None

        # Check color: same as source or constant?
        stamp_colors = set(c for demo in stamp_patterns for _, _, _, _, c in demo)
        source_colors = set(s.color for demo in stamp_patterns for s, _, _, _, _ in demo)
        use_source_color = all(c == s.color for demo in stamp_patterns for s, _, _, _, c in demo)

        def _make_stamp_fn(_sel, _dr, _dc, _use_src_color, _const_color):
            def _exec(inp):
                facts = perceive(inp)
                result = inp.copy()
                targets = prim_select(facts, _sel)
                rows, cols = inp.shape
                for obj in targets:
                    color = obj.color if _use_src_color else _const_color
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                nr, nc = obj.row + r + _dr, obj.col + c + _dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    result[nr, nc] = color
                return result
            return _exec

        const_color = next(iter(stamp_colors)) if len(stamp_colors) == 1 else None
        fn = _make_stamp_fn(sel, dr, dc, use_source_color, const_color)
        if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
            desc = _describe_selector(sel)
            color_desc = 'same_color' if use_source_color else f'color={const_color}'
            return _make_program(fn, f"synth: transition stamp [{desc}] at ({dr},{dc}) {color_desc}")

    return None


def _try_transition_dispatch(all_transitions, contexts):
    """Different object groups undergo different transitions.

    Groups by match_type, finds selectors for each group, builds a
    multi-branch dispatch program.
    """
    from aria.guided.clause import Predicate, Pred

    ctx0 = contexts[0]
    facts0 = ctx0['facts']
    bg = facts0.bg

    # Group transitions by match_type in demo 0
    groups = {}
    for t in all_transitions[0]:
        if t.match_type == 'identical':
            continue
        if t.match_type not in groups:
            groups[t.match_type] = []
        groups[t.match_type].append(t)

    if len(groups) < 2:
        return None  # uniform or nothing — handled by strategy 1

    # Find selectors for each group
    group_selectors = {}
    for mtype, transitions in groups.items():
        oids = set(t.in_obj.oid for t in transitions if t.in_obj)
        if not oids:
            continue
        sel = _find_transition_selector(oids, facts0)
        if sel is None:
            return None  # can't select this group
        group_selectors[mtype] = sel

    if len(group_selectors) < 2:
        return None

    # Check cross-demo consistency of transition parameters per group
    branches = []  # (selector, match_type, params)
    for mtype, sel in group_selectors.items():
        transitions = groups[mtype]

        if mtype == 'removed':
            branches.append((sel, 'remove', {}))

        elif mtype == 'recolored':
            # Check: constant color or derived?
            color_set = set(t.color_to for t in transitions)
            if len(color_set) == 1:
                branches.append((sel, 'recolor', {'color': next(iter(color_set))}))
            else:
                return None  # per-object varying recolor — too complex for now

        elif mtype in ('moved', 'moved_recolored'):
            offsets = set((t.dr, t.dc) for t in transitions)
            colors = set(t.color_to for t in transitions) if mtype == 'moved_recolored' else set()
            if len(offsets) == 1:
                dr, dc = next(iter(offsets))
                if mtype == 'moved':
                    branches.append((sel, 'move', {'dr': dr, 'dc': dc}))
                elif len(colors) == 1:
                    branches.append((sel, 'move_recolor', {'dr': dr, 'dc': dc, 'color': next(iter(colors))}))
                else:
                    return None
            else:
                # Varying offset — check gravity (all to same border)
                rows, cols = contexts[0]['input'].shape
                grav_dir = _detect_gravity_direction(transitions, rows, cols)
                if grav_dir:
                    if mtype == 'moved_recolored' and len(colors) != 1:
                        return None
                    params = {'direction': grav_dir}
                    if mtype == 'moved_recolored' and len(colors) == 1:
                        params['color'] = next(iter(colors))
                    branches.append((sel, 'gravity', params))
                else:
                    # Check slide (all same direction, varying distance)
                    slide_dir = _detect_slide_direction(transitions)
                    if slide_dir:
                        if mtype == 'moved_recolored' and len(colors) != 1:
                            return None
                        params = {'direction': slide_dir}
                        if mtype == 'moved_recolored' and len(colors) == 1:
                            params['color'] = next(iter(colors))
                        branches.append((sel, 'slide', params))
                    else:
                        return None

        elif mtype == 'transformed':
            xforms = set(t.transform for t in transitions)
            if len(xforms) == 1:
                branches.append((sel, 'transform', {'xform': next(iter(xforms))}))
            else:
                return None

    # Build multi-branch program
    def _make_dispatch_fn(branch_list):
        def _exec(inp):
            facts = perceive(inp)
            _bg = facts.bg
            result = inp.copy()
            for sel, action, params in branch_list:
                targets = prim_select(facts, sel)
                for obj in targets:
                    h, w = obj.height, obj.width
                    if action == 'remove':
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = _bg
                    elif action == 'recolor':
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = params['color']
                    elif action in ('move', 'move_recolor'):
                        dr, dc = params['dr'], params['dc']
                        color = params.get('color', obj.color)
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = _bg
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    nr, nc = obj.row + r + dr, obj.col + c + dc
                                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                        result[nr, nc] = color
                    elif action == 'gravity':
                        d = params['direction']
                        color = params.get('color', obj.color)
                        _rows, _cols = result.shape
                        if d == 'down': new_r, new_c = _rows - h, obj.col
                        elif d == 'up': new_r, new_c = 0, obj.col
                        elif d == 'right': new_r, new_c = obj.row, _cols - w
                        elif d == 'left': new_r, new_c = obj.row, 0
                        else: continue
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = _bg
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    nr, nc = new_r + r, new_c + c
                                    if 0 <= nr < _rows and 0 <= nc < _cols:
                                        result[nr, nc] = color
                    elif action == 'slide':
                        d = params['direction']
                        color = params.get('color', obj.color)
                        _dr = {'down': 1, 'up': -1, 'right': 0, 'left': 0}[d]
                        _dc = {'down': 0, 'up': 0, 'right': 1, 'left': -1}[d]
                        shift = _compute_slide_shift(obj, _dr, _dc, inp, _bg,
                                                      result.shape[0], result.shape[1])
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    result[obj.row + r, obj.col + c] = _bg
                        for r in range(h):
                            for c in range(w):
                                if obj.mask[r, c]:
                                    nr = obj.row + r + shift * _dr
                                    nc = obj.col + c + shift * _dc
                                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                        result[nr, nc] = color
                    elif action == 'transform':
                        xfn_map = {
                            'flip_h': lambda m: m[:, ::-1],
                            'flip_v': lambda m: m[::-1, :],
                            'flip_hv': lambda m: m[::-1, ::-1],
                            'rot90': lambda m: np.rot90(m),
                            'rot180': lambda m: np.rot90(m, 2),
                            'rot270': lambda m: np.rot90(m, 3),
                        }
                        xfn = xfn_map.get(params['xform'])
                        if xfn:
                            for r in range(h):
                                for c in range(w):
                                    if obj.mask[r, c]:
                                        result[obj.row + r, obj.col + c] = _bg
                            new_mask = xfn(obj.mask)
                            mh, mw = new_mask.shape
                            for r in range(mh):
                                for c in range(mw):
                                    if new_mask[r, c]:
                                        nr, nc = obj.row + r, obj.col + c
                                        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                            result[nr, nc] = obj.color
            return result
        return _exec

    fn = _make_dispatch_fn(branches)
    if all(np.array_equal(fn(ctx['input']), ctx['output']) for ctx in contexts):
        parts = []
        for sel, action, params in branches:
            desc = _describe_selector(sel)
            if action == 'remove':
                parts.append(f"[{desc}] remove")
            elif action == 'recolor':
                parts.append(f"[{desc}] recolor({params['color']})")
            elif action == 'move':
                parts.append(f"[{desc}] move({params['dr']},{params['dc']})")
            elif action == 'move_recolor':
                parts.append(f"[{desc}] move({params['dr']},{params['dc']}) recolor({params['color']})")
            elif action == 'gravity':
                rc = f" recolor({params['color']})" if 'color' in params else ''
                parts.append(f"[{desc}] gravity({params['direction']}){rc}")
            elif action == 'slide':
                rc = f" recolor({params['color']})" if 'color' in params else ''
                parts.append(f"[{desc}] slide({params['direction']}){rc}")
            elif action == 'transform':
                parts.append(f"[{desc}] transform({params['xform']})")
        desc = '; '.join(parts)
        return _make_program(fn, f"synth: transition dispatch {desc}")

    return None


def _detect_gravity_direction(transitions, rows, cols):
    """Check if all transitions go to the same border. Returns direction or None."""
    for d in ['down', 'up', 'right', 'left']:
        all_match = True
        any_moved = False
        for t in transitions:
            o = t.out_obj
            at_border = ((d == 'down' and o.row + o.height == rows) or
                         (d == 'up' and o.row == 0) or
                         (d == 'right' and o.col + o.width == cols) or
                         (d == 'left' and o.col == 0))
            moved_toward = ((d == 'down' and t.dr > 0) or
                            (d == 'up' and t.dr < 0) or
                            (d == 'right' and t.dc > 0) or
                            (d == 'left' and t.dc < 0))
            if not at_border:
                all_match = False
                break
            if moved_toward:
                any_moved = True
        if all_match and any_moved:
            return d
    return None


def _detect_slide_direction(transitions):
    """Check if all transitions move in the same direction (possibly different distances)."""
    if not transitions:
        return None
    dirs = set()
    for t in transitions:
        if t.dr > 0 and t.dc == 0:
            dirs.add('down')
        elif t.dr < 0 and t.dc == 0:
            dirs.add('up')
        elif t.dc > 0 and t.dr == 0:
            dirs.add('right')
        elif t.dc < 0 and t.dr == 0:
            dirs.add('left')
        else:
            return None  # diagonal or no movement
    if len(dirs) == 1:
        return next(iter(dirs))
    return None


def _find_transition_selector(oids, facts):
    """Find a selector predicate list that selects exactly these objects."""
    from aria.guided.clause import Predicate, Pred
    from aria.guided.dsl import prim_select

    target_set = set(oids)

    # Color-based
    target_objs = [o for o in facts.objects if o.oid in target_set]
    if target_objs:
        colors = set(o.color for o in target_objs)
        for c in colors:
            sel = [Predicate(Pred.COLOR_EQ, c)]
            if set(o.oid for o in prim_select(facts, sel)) == target_set:
                return sel

    # Structural predicates
    for pred in [Pred.IS_LARGEST, Pred.IS_SMALLEST, Pred.UNIQUE_COLOR,
                 Pred.IS_SINGLETON, Pred.IS_RECTANGULAR, Pred.IS_LINE,
                 Pred.IS_SQUARE, Pred.TOUCHES_BORDER, Pred.NOT_TOUCHES_BORDER,
                 Pred.IS_TOPMOST, Pred.IS_BOTTOMMOST, Pred.IS_LEFTMOST, Pred.IS_RIGHTMOST]:
        sel = [Predicate(pred)]
        if set(o.oid for o in prim_select(facts, sel)) == target_set:
            return sel

    # NOT predicates
    for pred in [Pred.IS_SINGLETON, Pred.IS_RECTANGULAR, Pred.IS_LINE,
                 Pred.TOUCHES_BORDER]:
        sel = [Predicate(Pred.NOT, Predicate(pred))]
        if set(o.oid for o in prim_select(facts, sel)) == target_set:
            return sel

    # Relational
    inner_preds = [
        Predicate(Pred.IS_LARGEST), Predicate(Pred.IS_SMALLEST),
        Predicate(Pred.UNIQUE_COLOR),
    ]
    for o in facts.objects:
        inner_preds.append(Predicate(Pred.COLOR_EQ, o.color))
    for rel in [Pred.CONTAINED_BY, Pred.CONTAINS, Pred.ADJACENT_TO, Pred.SAME_SHAPE_AS]:
        for inner in inner_preds:
            sel = [Predicate(rel, inner)]
            if set(o.oid for o in prim_select(facts, sel)) == target_set:
                return sel

    return None


def _search_conditional_dispatch(all_values, contexts):
    """Search for conditional dispatch programs.

    Form: for each object in the grid, if it matches selector A apply action X,
    if selector B apply action Y, else keep as-is.

    This is the minimal conditional: two disjoint selectors with different
    per-object actions (recolor, remove, or writeback).
    """
    from aria.guided.clause import Predicate, Pred
    from aria.guided.dsl import prim_select

    n_demos = len(contexts)
    # Only for same-shape tasks
    if any(ctx['input'].shape != ctx['output'].shape for ctx in contexts):
        return None

    # Analyze per-object changes in demo 0
    ctx0 = contexts[0]
    facts0 = ctx0['facts']
    inp0, out0 = ctx0['input'], ctx0['output']
    bg = facts0.bg

    # Classify each object by what happened to it
    obj_actions = {}  # oid → (action_name, param)
    for obj in facts0.objects:
        action = _classify_object_action(obj, inp0, out0, bg)
        obj_actions[obj.oid] = action

    # Lift constant recolors to derived-color recolors when possible
    # If recolored objects get their new color from a related object,
    # replace ('recolor', constant) with ('recolor_derived', relation_type)
    obj_actions = _lift_derived_colors(obj_actions, facts0, contexts)

    # Group objects by action
    action_groups = {}
    for oid, (action, param) in obj_actions.items():
        key = (action, param)
        if key not in action_groups:
            action_groups[key] = []
        action_groups[key].append(oid)

    # Need at least 2 different actions to be a conditional
    if len(action_groups) < 2:
        return None

    # Try to find selectors that partition the objects into action groups
    # For each group, try common predicates
    def _find_selector(oids, all_objs, facts):
        """Find a predicate list that selects exactly these objects."""
        target_set = set(oids)
        candidates = []

        # Color-based
        target_objs = [o for o in all_objs if o.oid in target_set]
        if target_objs:
            colors = set(o.color for o in target_objs)
            for c in colors:
                sel = [Predicate(Pred.COLOR_EQ, c)]
                selected = set(o.oid for o in prim_select(facts, sel))
                if selected == target_set:
                    candidates.append(sel)

        # Structural predicates
        for pred in [Pred.IS_LARGEST, Pred.IS_SMALLEST, Pred.UNIQUE_COLOR,
                     Pred.IS_SINGLETON, Pred.IS_RECTANGULAR, Pred.IS_LINE,
                     Pred.IS_SQUARE, Pred.TOUCHES_BORDER, Pred.NOT_TOUCHES_BORDER,
                 Pred.IS_TOPMOST, Pred.IS_BOTTOMMOST, Pred.IS_LEFTMOST, Pred.IS_RIGHTMOST]:
            sel = [Predicate(pred)]
            selected = set(o.oid for o in prim_select(facts, sel))
            if selected == target_set:
                candidates.append(sel)

        # NOT(predicate) — complement selectors
        for pred in [Pred.IS_SINGLETON, Pred.IS_RECTANGULAR, Pred.IS_LINE,
                     Pred.IS_SQUARE, Pred.TOUCHES_BORDER]:
            sel = [Predicate(Pred.NOT, Predicate(pred))]
            selected = set(o.oid for o in prim_select(facts, sel))
            if selected == target_set:
                candidates.append(sel)

        # Two-predicate compounds for tighter selection
        for p1 in [Pred.IS_RECTANGULAR, Pred.IS_LINE, Pred.IS_SINGLETON,
                    Pred.TOUCHES_BORDER]:
            for p2 in [Pred.IS_RECTANGULAR, Pred.IS_LINE, Pred.IS_SINGLETON,
                        Pred.TOUCHES_BORDER]:
                if p1.value >= p2.value:
                    continue
                sel = [Predicate(p1), Predicate(p2)]
                selected = set(o.oid for o in prim_select(facts, sel))
                if selected == target_set:
                    candidates.append(sel)

        # --- Relational selectors ---
        # CONTAINED_BY(other_pred): objects contained by objects matching other_pred
        # CONTAINS(other_pred): objects that contain objects matching other_pred
        # ADJACENT_TO(other_pred): objects adjacent to objects matching other_pred
        # SAME_SHAPE_AS(other_pred): objects with same shape as objects matching other_pred
        #
        # For each relational predicate, try inner selectors that identify
        # the "other" object structurally (not by color literal when avoidable)
        inner_preds = [
            Predicate(Pred.IS_LARGEST),
            Predicate(Pred.IS_SMALLEST),
            Predicate(Pred.UNIQUE_COLOR),
            Predicate(Pred.IS_SINGLETON),
            Predicate(Pred.IS_RECTANGULAR),
            Predicate(Pred.NOT, Predicate(Pred.IS_SINGLETON)),
            Predicate(Pred.TOUCHES_BORDER),
            Predicate(Pred.NOT, Predicate(Pred.TOUCHES_BORDER)),
        ]
        # Also try color-based inner selectors
        seen_colors = set()
        for o in all_objs:
            if o.color not in seen_colors:
                inner_preds.append(Predicate(Pred.COLOR_EQ, o.color))
                seen_colors.add(o.color)

        for rel_pred in [Pred.CONTAINED_BY, Pred.CONTAINS,
                         Pred.ADJACENT_TO, Pred.SAME_SHAPE_AS]:
            for inner in inner_preds:
                sel = [Predicate(rel_pred, inner)]
                selected = set(o.oid for o in prim_select(facts, sel))
                if selected == target_set:
                    candidates.append(sel)

        return candidates

    # Separate keep (default) from active action groups
    keep_key = ('keep', None)
    active_groups = {k: v for k, v in action_groups.items() if k != keep_key}

    if not active_groups:
        return None

    # Find selectors for each active group
    group_selectors = {}  # key → list of selector candidates
    for key, oids in active_groups.items():
        sels = _find_selector(oids, facts0.objects, facts0)
        if not sels:
            break  # can't select this group → no dispatch possible
        group_selectors[key] = sels
    else:
        # All active groups have selectors — try to build dispatch
        # Support 1-3 active groups
        active_keys = list(active_groups.keys())
        if len(active_keys) > 3:
            return None  # too many groups

        # Try selector combinations
        from itertools import product
        sel_lists = [group_selectors[k][:3] for k in active_keys]
        for combo in product(*sel_lists):
            # Check disjointness
            all_selected = []
            disjoint = True
            for sel in combo:
                selected = set(o.oid for o in prim_select(facts0, sel))
                for prev in all_selected:
                    if selected & prev:
                        disjoint = False
                        break
                if not disjoint:
                    break
                all_selected.append(selected)
            if not disjoint:
                continue

            # Build dispatch: list of (selector, action, param)
            branches = list(zip(combo, active_keys))

            def _make_dispatch_fn(branch_list):
                def _exec(inp):
                    facts = perceive(inp)
                    _bg = facts.bg
                    result = inp.copy()
                    # Build match sets
                    match_sets = []
                    for sel, (act, prm) in branch_list:
                        matched = set(o.oid for o in prim_select(facts, sel))
                        match_sets.append((matched, act, prm))
                    for obj in facts.objects:
                        for matched, act, prm in match_sets:
                            if obj.oid in matched:
                                _apply_action_inplace(result, obj, act, prm, _bg, inp)
                                break
                    return result
                return _exec

            fn = _make_dispatch_fn(branches)

            # Verify on ALL demos
            all_ok = all(
                np.array_equal(fn(ctx['input']), ctx['output'])
                for ctx in contexts
            )
            if all_ok:
                parts = []
                for sel, (act, prm) in branches:
                    parts.append(f"if [{_describe_selector(sel)}] then {_describe_action(act, prm)}")
                desc = '; '.join(parts)
                return _make_program(fn, f"synth: dispatch {desc}")

    return None


def _lift_derived_colors(obj_actions, facts0, contexts):
    """Lift constant recolors to derived-color recolors when the new color
    comes from a related object and this relationship is consistent across demos.

    Returns updated obj_actions dict.
    """
    recolored_oids = [oid for oid, (act, _) in obj_actions.items() if act == 'recolor']
    if not recolored_oids:
        return obj_actions

    # For each recolored object in demo 0, check what relation provides the new color
    relation_types = ['adjacent', 'contained_by', 'contains']

    for oid in recolored_oids:
        _, new_color = obj_actions[oid]
        obj = next(o for o in facts0.objects if o.oid == oid)

        # Find which relation gives us the color
        for rel_type in relation_types:
            source = _find_color_source(obj, new_color, facts0, rel_type)
            if source is None:
                continue

            # Verify across ALL demos: does the same relation consistently
            # provide the recolor target?
            consistent = True
            for ctx in contexts[1:]:
                demo_facts = ctx['facts']
                demo_bg = demo_facts.bg
                # Find the corresponding object in this demo
                # (match by color since oid won't correspond across demos)
                demo_obj = _find_corresponding_obj(obj, demo_facts)
                if demo_obj is None:
                    consistent = False
                    break
                act, param = _classify_object_action(demo_obj, ctx['input'], ctx['output'], demo_bg)
                if act != 'recolor':
                    consistent = False
                    break
                # Check: does the same relation provide this color?
                demo_source = _find_color_source(demo_obj, param, demo_facts, rel_type)
                if demo_source is None:
                    consistent = False
                    break

            if consistent:
                obj_actions[oid] = ('recolor_derived', rel_type)
                break

    return obj_actions


def _find_color_source(obj, target_color, facts, rel_type):
    """Find an object related to obj whose color == target_color."""
    for pair in facts.pairs:
        if pair.oid_a == obj.oid:
            other_oid = pair.oid_b
        elif pair.oid_b == obj.oid:
            other_oid = pair.oid_a
        else:
            continue
        other = next((o for o in facts.objects if o.oid == other_oid), None)
        if other is None or other.color != target_color:
            continue
        if rel_type == 'adjacent' and pair.adjacent:
            return other
        if rel_type == 'contained_by':
            if (pair.oid_a == obj.oid and pair.b_contains_a) or \
               (pair.oid_b == obj.oid and pair.a_contains_b):
                return other
        if rel_type == 'contains':
            if (pair.oid_a == obj.oid and pair.a_contains_b) or \
               (pair.oid_b == obj.oid and pair.b_contains_a):
                return other
    return None


def _find_corresponding_obj(ref_obj, demo_facts):
    """Find an object in demo_facts that corresponds to ref_obj by color+shape."""
    for o in demo_facts.objects:
        if o.color == ref_obj.color:
            return o
    return None


def _classify_object_action(obj, inp, out, bg):
    """Classify what happened to an object between input and output.

    Returns (action_name, param) where action_name is one of:
    keep, remove, recolor, transform, gravity, fill_interior, stamp
    """
    h, w = obj.height, obj.width
    rows, cols = inp.shape

    # Extract what's at the object's mask position in the output
    out_colors_at_mask = set()
    for r in range(h):
        for c in range(w):
            if obj.mask[r, c]:
                out_colors_at_mask.add(int(out[obj.row + r, obj.col + c]))

    # Simple cases
    if out_colors_at_mask == {obj.color}:
        # Object pixels unchanged — but check if interior (bg holes) got filled
        if h >= 3 and w >= 3:
            interior_changed = False
            fill_color = None
            for r in range(h):
                for c in range(w):
                    if not obj.mask[r, c]:
                        gr, gc = obj.row + r, obj.col + c
                        if 0 <= gr < rows and 0 <= gc < cols:
                            iv, ov = int(inp[gr, gc]), int(out[gr, gc])
                            if iv == bg and ov != bg:
                                interior_changed = True
                                if fill_color is None:
                                    fill_color = ov
                                elif fill_color != ov:
                                    fill_color = None  # multi-color fill
            if interior_changed and fill_color is not None:
                return ('fill_interior', fill_color)
        return ('keep', None)

    if out_colors_at_mask == {bg}:
        # Object removed from original position — check if stamped elsewhere
        # (stamp = object appears at a new location while removed from original)
        return ('remove', None)

    if len(out_colors_at_mask) == 1:
        return ('recolor', next(iter(out_colors_at_mask)))

    # Check for in-place transform: object mask transformed at same position
    out_sub = out[obj.row:obj.row + h, obj.col:obj.col + w]
    inp_sub = inp[obj.row:obj.row + h, obj.col:obj.col + w]
    # Build the object's rendered patch (mask × color on bg)
    obj_patch = np.full((h, w), bg, dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if obj.mask[r, c]:
                obj_patch[r, c] = obj.color

    for xform_name, xfn in [('flip_h', lambda m: m[::-1, :]),
                              ('flip_v', lambda m: m[:, ::-1]),
                              ('flip_hv', lambda m: m[::-1, ::-1]),
                              ('rot90', lambda m: np.rot90(m)),
                              ('rot180', lambda m: np.rot90(m, 2))]:
        transformed = xfn(obj_patch)
        if transformed.shape == out_sub.shape and np.array_equal(transformed, out_sub):
            return ('transform', xform_name)

    # Check for gravity: object moved to grid border, same shape
    for direction in ['down', 'up', 'left', 'right']:
        if direction == 'down':
            new_row = rows - h
            new_col = obj.col
        elif direction == 'up':
            new_row = 0
            new_col = obj.col
        elif direction == 'right':
            new_row = obj.row
            new_col = cols - w
        elif direction == 'left':
            new_row = obj.row
            new_col = 0
        else:
            continue
        if new_row == obj.row and new_col == obj.col:
            continue
        if new_row + h > rows or new_col + w > cols:
            continue
        match = True
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    if out[new_row + r, new_col + c] != obj.color:
                        match = False
                        break
            if not match:
                break
        if match:
            return ('gravity', direction)

    return ('keep', None)  # unrecognized change


def _apply_action(canvas, obj, action, param, bg, inp=None):
    """Apply a per-object action to the canvas."""
    h, w = obj.height, obj.width
    rows, cols = canvas.shape

    if action == 'keep':
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    canvas[obj.row + r, obj.col + c] = obj.color

    elif action == 'recolor':
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    canvas[obj.row + r, obj.col + c] = param

    elif action == 'remove':
        pass  # leave as bg

    elif action == 'transform':
        xform = param  # 'flip_h', 'flip_v', etc.
        mask = obj.mask
        if xform == 'flip_h':
            mask = mask[::-1, :]
        elif xform == 'flip_v':
            mask = mask[:, ::-1]
        elif xform == 'flip_hv':
            mask = mask[::-1, ::-1]
        elif xform == 'rot90':
            mask = np.rot90(mask)
        elif xform == 'rot180':
            mask = np.rot90(mask, 2)
        mh, mw = mask.shape
        for r in range(mh):
            for c in range(mw):
                if mask[r, c]:
                    nr, nc = obj.row + r, obj.col + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        canvas[nr, nc] = obj.color

    elif action == 'gravity':
        direction = param
        if direction == 'down':
            new_row, new_col = rows - h, obj.col
        elif direction == 'up':
            new_row, new_col = 0, obj.col
        elif direction == 'right':
            new_row, new_col = obj.row, cols - w
        elif direction == 'left':
            new_row, new_col = obj.row, 0
        else:
            new_row, new_col = obj.row, obj.col
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    nr, nc = new_row + r, new_col + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        canvas[nr, nc] = obj.color

    elif action == 'recolor_derived':
        # Recolor to color of a related object (derived at execution time)
        rel_type = param  # 'adjacent', 'contained_by', 'contains'
        derived_color = _derive_color_from_relation(obj, canvas, bg, rel_type, inp)
        if derived_color is not None:
            for r in range(h):
                for c in range(w):
                    if obj.mask[r, c]:
                        nr, nc = obj.row + r, obj.col + c
                        if 0 <= nr < rows and 0 <= nc < cols:
                            canvas[nr, nc] = derived_color
        else:
            # Fallback: keep as-is
            for r in range(h):
                for c in range(w):
                    if obj.mask[r, c]:
                        canvas[nr, nc] = obj.color

    elif action == 'fill_interior':
        # Keep the object, fill bg holes with param color
        for r in range(h):
            for c in range(w):
                nr, nc = obj.row + r, obj.col + c
                if 0 <= nr < rows and 0 <= nc < cols:
                    if obj.mask[r, c]:
                        canvas[nr, nc] = obj.color
                    elif inp is not None and inp[nr, nc] == bg:
                        canvas[nr, nc] = param


def _describe_selector(sel):
    from aria.guided.clause import Pred
    parts = []
    for p in sel:
        if p.param is not None:
            parts.append(f"{p.pred.name}({p.param})")
        else:
            parts.append(p.pred.name)
    return ' & '.join(parts)


def _apply_action_inplace(result, obj, action, param, bg, inp):
    """Apply an action by mutating the grid in-place (not rebuilding from scratch)."""
    h, w = obj.height, obj.width
    rows, cols = result.shape

    if action == 'recolor':
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = param

    elif action == 'recolor_derived':
        derived_color = _derive_color_from_relation(obj, result, bg, param, inp)
        if derived_color is not None:
            for r in range(h):
                for c in range(w):
                    if obj.mask[r, c]:
                        result[obj.row + r, obj.col + c] = derived_color

    elif action == 'remove':
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg

    elif action == 'transform':
        xform = param
        mask = obj.mask
        if xform == 'flip_h':
            mask = mask[::-1, :]
        elif xform == 'flip_v':
            mask = mask[:, ::-1]
        elif xform == 'flip_hv':
            mask = mask[::-1, ::-1]
        elif xform == 'rot90':
            mask = np.rot90(mask)
        elif xform == 'rot180':
            mask = np.rot90(mask, 2)
        # Clear original position first
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg
        # Place transformed
        mh, mw = mask.shape
        for r in range(mh):
            for c in range(mw):
                if mask[r, c]:
                    nr, nc = obj.row + r, obj.col + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = obj.color

    elif action == 'gravity':
        direction = param
        if direction == 'down':
            new_row, new_col = rows - h, obj.col
        elif direction == 'up':
            new_row, new_col = 0, obj.col
        elif direction == 'right':
            new_row, new_col = obj.row, cols - w
        elif direction == 'left':
            new_row, new_col = obj.row, 0
        else:
            return
        # Clear original
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg
        # Place at new position
        for r in range(h):
            for c in range(w):
                if obj.mask[r, c]:
                    nr, nc = new_row + r, new_col + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = obj.color

    elif action == 'fill_interior':
        for r in range(h):
            for c in range(w):
                nr, nc = obj.row + r, obj.col + c
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not obj.mask[r, c] and inp is not None and inp[nr, nc] == bg:
                        result[nr, nc] = param


def _derive_color_from_relation(obj, grid, bg, rel_type, inp):
    """At execution time, find the color of a related object.

    Uses the input grid's perception to find the related object,
    then returns its color.
    """
    if inp is None:
        return None
    facts = perceive(inp)
    # Find our object in the perceived facts (match by position)
    my_obj = None
    for o in facts.objects:
        if o.row == obj.row and o.col == obj.col and o.color == obj.color:
            my_obj = o
            break
    if my_obj is None:
        return None

    for pair in facts.pairs:
        if pair.oid_a == my_obj.oid:
            other = next((o for o in facts.objects if o.oid == pair.oid_b), None)
        elif pair.oid_b == my_obj.oid:
            other = next((o for o in facts.objects if o.oid == pair.oid_a), None)
        else:
            continue
        if other is None or other.color == my_obj.color:
            continue
        if rel_type == 'adjacent' and pair.adjacent:
            return other.color
        if rel_type == 'contained_by':
            if (pair.oid_a == my_obj.oid and pair.b_contains_a) or \
               (pair.oid_b == my_obj.oid and pair.a_contains_b):
                return other.color
        if rel_type == 'contains':
            if (pair.oid_a == my_obj.oid and pair.a_contains_b) or \
               (pair.oid_b == my_obj.oid and pair.b_contains_a):
                return other.color
    return None


def _describe_action(action, param):
    if action == 'keep':
        return 'keep'
    elif action == 'recolor':
        return f'recolor({param})'
    elif action == 'recolor_derived':
        return f'recolor(color_of_{param})'
    elif action == 'remove':
        return 'remove'
    elif action == 'transform':
        return f'transform({param})'
    elif action == 'gravity':
        return f'gravity({param})'
    elif action == 'fill_interior':
        return f'fill_interior({param})'
    return f'{action}({param})'


def _check_output_match(values, contexts):
    """Check if any value matches the output in ALL demos."""
    for (key, ty), vals in values.items():
        if ty not in (Ty.GRID, Ty.REGION):
            continue
        all_match = True
        for i, val in enumerate(vals):
            if val is None:
                all_match = False
                break
            out = contexts[i]['output']
            if not isinstance(val, np.ndarray):
                all_match = False
                break
            if not np.array_equal(val, out):
                all_match = False
                break
        if all_match:
            return _make_program(
                _build_executor(key, vals, contexts),
                f"synth: {key}",
            )
    return None


_MAX_VALUES_PER_LEVEL = 500  # cap to prevent search explosion

def _apply_ops(values, op_lib, contexts):
    """Apply all typed operations to existing values, return new values.

    Pruning:
    - Only keep grid/region values matching output shape (demo 0)
    - Cap total new values per level
    """
    new_values = {}
    n_demos = len(contexts)
    out_shape = contexts[0]['output'].shape

    for op in op_lib:
        if len(new_values) >= _MAX_VALUES_PER_LEVEL:
            break

        if len(op.in_types) == 1:
            in_ty = op.in_types[0]
            for (key, ty), vals in values.items():
                if ty != in_ty:
                    continue
                results = []
                valid = True
                for val in vals:
                    try:
                        r = op.fn(val)
                        results.append(r)
                    except Exception:
                        valid = False
                        break
                if not valid or any(r is None for r in results):
                    continue
                # Shape pruning: only GRID must match output shape
                # REGION values can be sub-regions used in writeback
                if op.out_type == Ty.GRID:
                    if isinstance(results[0], np.ndarray) and results[0].shape != out_shape:
                        continue
                new_key = (f"{op.name}({key})", op.out_type)
                if new_key not in values and new_key not in new_values:
                    new_values[new_key] = results

        elif len(op.in_types) == 2:
            in_ty1, in_ty2 = op.in_types
            for (k1, t1), v1 in values.items():
                if t1 != in_ty1:
                    continue
                for (k2, t2), v2 in values.items():
                    if t2 != in_ty2:
                        continue
                    if k1 == k2:
                        continue
                    if len(new_values) >= _MAX_VALUES_PER_LEVEL:
                        break
                    results = []
                    valid = True
                    for i in range(n_demos):
                        try:
                            r = op.fn(v1[i], v2[i])
                            results.append(r)
                        except Exception:
                            valid = False
                            break
                    if not valid or any(r is None for r in results):
                        continue
                    if op.out_type in (Ty.GRID, Ty.REGION):
                        if isinstance(results[0], np.ndarray) and results[0].shape != out_shape:
                            continue
                    new_key = (f"{op.name}({k1}, {k2})", op.out_type)
                    if new_key not in values and new_key not in new_values:
                        new_values[new_key] = results

    return new_values


def _check_writeback(all_values, contexts):
    """Check if output = input with a sub-region replaced by a computed value.

    For each REGION-typed value, check if placing it at some position in the
    input grid produces the output. The position is derived from an object's
    interior (frame detection).
    """
    from aria.guided.dsl import prim_find_frame

    n_demos = len(contexts)

    for (key, ty), vals in all_values.items():
        if ty != Ty.REGION:
            continue

        # For each demo: find where this region fits in the output
        all_ok = True
        positions = []
        for i, val in enumerate(vals):
            if val is None or not isinstance(val, np.ndarray):
                all_ok = False
                break
            inp = contexts[i]['input']
            out = contexts[i]['output']
            if inp.shape != out.shape:
                all_ok = False
                break
            rh, rw = val.shape
            # Find where inp differs from out
            diff = inp != out
            diff_rows = np.where(np.any(diff, axis=1))[0]
            diff_cols = np.where(np.any(diff, axis=0))[0]
            if len(diff_rows) == 0:
                all_ok = False
                break
            # The changed region bounds
            r0, r1 = diff_rows[0], diff_rows[-1] + 1
            c0, c1 = diff_cols[0], diff_cols[-1] + 1
            if r1 - r0 != rh or c1 - c0 != rw:
                all_ok = False
                break
            # Does placing val at (r0, c0) produce the output?
            test = inp.copy()
            test[r0:r1, c0:c1] = val
            if not np.array_equal(test, out):
                all_ok = False
                break
            positions.append((r0, c0))

        if all_ok and positions:
            # Build executor: compute the region value and place it
            def _make_fn(region_key):
                def _exec(inp):
                    facts = perceive(inp)
                    seeds = {'input': inp}
                    for j, obj in enumerate(facts.objects):
                        seeds[f'obj_{j}'] = obj
                    _add_selected_objects(seeds, facts)
                    for c in range(10):
                        if c != facts.bg:
                            seeds[f'color_{c}'] = c
                    # Compute the region value
                    val = _eval_key(region_key, seeds, inp, facts)
                    if val is None:
                        return inp
                    # Find where to place it (interior of frame objects)
                    for obj in facts.objects:
                        frame = prim_find_frame(obj, inp)
                        if frame:
                            r0, c0, r1, c1 = frame
                            if r1 - r0 - 1 == val.shape[0] and c1 - c0 - 1 == val.shape[1]:
                                result = inp.copy()
                                result[r0+1:r1, c0+1:c1] = val
                                return result
                    return inp
                return _exec

            return _make_program(
                _make_fn(key),
                f"synth: writeback {key}",
            )

    return None


def _search_map(contexts):
    """Search for MAP programs: per-row or per-column operations.

    For each row/col in the grid, derive an operation from its content
    and apply it. The operation and its parameters are induced from
    the input→output relationship of that row/col.

    Per-row operations:
    - periodic_extend: detect period in row, extend to fill
    - fill_constant: fill remaining cells with a constant
    - mirror: reflect the row
    """
    ctx0 = contexts[0]
    inp0 = ctx0['input']
    out0 = ctx0['output']
    bg = ctx0['facts'].bg

    if inp0.shape != out0.shape:
        return None

    rows, cols = inp0.shape

    # --- Strategy: per-row periodic extension ---
    # For each row, check if the output extends the input's pattern
    prog = _try_per_row_periodic(contexts)
    if prog:
        return prog

    # --- Strategy: per-column periodic extension ---
    prog = _try_per_col_periodic(contexts)
    if prog:
        return prog

    return None


def _try_per_row_periodic(contexts):
    """For each row: detect the pattern in input, extend it in the output."""
    ctx0 = contexts[0]
    inp0 = ctx0['input']
    out0 = ctx0['output']
    bg = ctx0['facts'].bg

    if inp0.shape != out0.shape:
        return None

    rows, cols = inp0.shape

    # For demo 0: check if per-row periodic extension explains the output
    def _extend_row(row_in, row_out_len):
        """Given input row, try to find a period and extend to full length."""
        # Find the non-bg content
        non_bg = [(i, int(row_in[i])) for i in range(len(row_in)) if row_in[i] != bg]
        if not non_bg:
            return np.full(row_out_len, bg, dtype=np.uint8)

        # Extract the content portion
        first = non_bg[0][0]
        last = non_bg[-1][0]
        content = row_in[first:last + 1]
        content_len = len(content)

        if content_len == 0:
            return np.full(row_out_len, bg, dtype=np.uint8)

        # Try periods from 1 to content_len
        for period in range(1, content_len + 1):
            tile = content[:period]
            match = True
            for i in range(period, content_len):
                if content[i] != tile[i % period]:
                    match = False
                    break
            if match:
                # Extend this period to fill the full output row
                result = np.full(row_out_len, bg, dtype=np.uint8)
                for i in range(row_out_len):
                    # Map position i to the periodic pattern
                    pos = i - first
                    if pos >= 0:
                        result[i] = tile[pos % period]
                    else:
                        # Extend left too
                        result[i] = tile[pos % period]
                return result

        return None

    # Test on demo 0: apply per-row periodic extension
    result = inp0.copy()
    any_changed = False
    for r in range(rows):
        if np.array_equal(inp0[r], out0[r]):
            continue
        extended = _extend_row(inp0[r], cols)
        if extended is not None:
            result[r] = extended
            any_changed = True

    if not any_changed:
        return None
    if not np.array_equal(result, out0):
        return None

    # Build and verify across all demos
    def _exec(inp):
        f = perceive(inp)
        r = inp.copy()
        h, w = inp.shape
        for row in range(h):
            # Only extend rows that need it
            non_bg_in = np.sum(inp[row] != f.bg)
            if non_bg_in == 0:
                continue
            extended = _extend_row(inp[row], w)
            if extended is not None:
                r[row] = extended
        return r

    prog = _make_program(_exec, "synth: per-row periodic extend")
    all_ok = all(np.array_equal(prog.execute(ctx['input']), ctx['output'])
                 for ctx in contexts)
    if all_ok:
        return prog

    return None


def _try_per_col_periodic(contexts):
    """Same as per-row but transposed."""
    # Transpose all demos, try per-row, transpose back
    transposed = [{'input': ctx['input'].T, 'output': ctx['output'].T,
                    'facts': ctx['facts']}
                   for ctx in contexts]

    ctx0 = transposed[0]
    if ctx0['input'].shape != ctx0['output'].shape:
        return None

    from copy import deepcopy
    result = _try_per_row_periodic(transposed)
    if result is None:
        return None

    # Wrap to transpose input, apply, transpose output
    inner_fn = result._execute_fn

    def _exec(inp):
        return inner_fn(inp.T).T

    return _make_program(_exec, "synth: per-col periodic extend")


def _search_mirror_DEAD(contexts):
    """DEAD — mirror is now in the bottom-up op library."""
    from aria.guided.dsl import prim_mirror_across

    ctx0 = contexts[0]
    facts0 = ctx0['facts']
    if not facts0.separators:
        return None

    for sep in facts0.separators:
        for mirror_mode in ['col', 'col_rtl', 'row', 'row_btt']:
            if sep.axis == 'col' and mirror_mode.startswith('row'):
                continue
            if sep.axis == 'row' and mirror_mode.startswith('col'):
                continue

            # Test on demo 0
            result = prim_mirror_across(ctx0['input'], mirror_mode, sep.index)
            if not np.array_equal(result, ctx0['output']):
                continue

            # Verify all demos
            def _make_fn(mm):
                def _exec(inp):
                    f = perceive(inp)
                    for s in f.separators:
                        r = prim_mirror_across(inp, mm, s.index)
                        return r
                    return inp
                return _exec

            prog = _make_program(_make_fn(mirror_mode), f"synth: mirror {mirror_mode} at separator")
            if all(np.array_equal(prog.execute(inp), out) for inp, out in contexts[1:]):
                # Verify on actual demos (not just contexts)
                all_ok = True
                for ctx in contexts:
                    if not np.array_equal(prog.execute(ctx['input']), ctx['output']):
                        all_ok = False
                        break
                if all_ok:
                    return prog

    return None


def _search_loops(all_values, contexts):
    """Search for FOR_EACH programs over per-object operations.

    Generic: enumerates (selector, operation, params) triples.
    Each registered per-object operation generates candidate rules.
    Rules are composed and verified across all demos.
    """
    # Loop ops mutate the input grid in-place — only works for same-shape tasks
    if any(ctx['input'].shape != ctx['output'].shape for ctx in contexts):
        return None

    n_demos = len(contexts)

    # Search for trace rules with output-guided binding
    candidate_rules = _search_trace_rules(contexts)

    if not candidate_rules:
        return None

    # Try single rules and multi-rule combinations
    from itertools import combinations

    # Single rules
    for rule in candidate_rules:
        if _verify_trace_rules([rule], contexts):
            fn = _make_trace_fn([rule])
            return _make_program(fn, f"synth: foreach {rule.description}")

    # Multi-rule (up to 3, different selectors)
    for n in range(2, min(len(candidate_rules) + 1, 4)):
        for combo in combinations(candidate_rules, n):
            sels = set(r.selector for r in combo)
            if len(sels) < n:
                continue
            if _verify_trace_rules(list(combo), contexts):
                desc = '; '.join(r.description for r in combo)
                fn = _make_trace_fn(list(combo))
                return _make_program(fn, f"synth: foreach {desc}")

    return None


def _verify_trace_rules(rules, contexts):
    """Verify that applying TraceRules produces correct output on all demos."""
    for ctx in contexts:
        result = ctx['input'].copy()
        for rule in rules:
            result = rule.apply(result, ctx['facts'])
        if not np.array_equal(result, ctx['output']):
            return False
    return True


def _make_trace_fn(rules):
    """Create an execute function from a list of TraceRules."""
    rules_copy = list(rules)
    def _exec(inp):
        facts = perceive(inp)
        result = inp.copy()
        for rule in rules_copy:
            result = rule.apply(result, facts)
        return result
    return _exec


# ---------------------------------------------------------------------------
# Typed loop body: all per-object trace operations as data, not closures
# ---------------------------------------------------------------------------

@dataclass
class TraceRule:
    """A typed per-object trace operation. All geometry ops are prim_trace calls."""
    selector: Any           # object color (int), or 'all_ordered', 'markers', (color_a, color_b)
    route: str              # 'ray', 'fill', 'path', 'line', 'ray_derived'
    color: Any              # trace color (int), or 'own' (object's color)
    write: str              # 'bg_only', 'skip', 'overwrite'
    directions: str = None  # for ray/fill: 'cardinal', 'all', 'up', 'toward_sep', etc.
    distance: int = 0       # for ray: max steps
    target_color: int = -1  # for path/line: connect to objects of this color
    order: str = 'hv'       # for path: 'hv' or 'vh'
    stop_colors: frozenset = None  # for fill: colors that stop the trace
    composition: str = 'independent'  # 'independent', 'ordered', 'star'
    description: str = ''

    def apply(self, grid, facts):
        from aria.guided.dsl import (
            prim_trace, _ray_cells, _path_cells, _line_cells, _DIR_MAP,
            prim_cast_rays,
        )
        bg = facts.bg
        result = grid.copy()

        if self.route == 'ray':
            objs = [o for o in facts.objects if o.color == self.selector]
            for obj in objs:
                for dr, dc in _DIR_MAP.get(self.directions, []):
                    cells = _ray_cells(obj.row, obj.col, dr, dc,
                                       self.distance, grid.shape)
                    result = prim_trace(result, cells, self.color, bg,
                                        write=self.write)
            return result

        if self.route == 'fill':
            dir_name = self.directions
            if dir_name == 'toward_sep':
                dirs = _get_separator_direction(facts)
                if not dirs:
                    return grid
            else:
                dirs = _DIR_MAP.get(dir_name, [])
            stop = frozenset(s.color for s in facts.separators) if self.stop_colors is None else self.stop_colors

            if self.composition == 'ordered':
                # All objects, ordered by distance to separator (farthest first)
                if not facts.separators:
                    return grid
                sep = facts.separators[0]
                dr, dc = dirs[0]
                def _dist(obj):
                    if sep.axis == 'col':
                        return abs(obj.center_col - sep.index)
                    return abs(obj.center_row - sep.index)
                objs = sorted([o for o in facts.objects if o.color not in stop],
                              key=_dist, reverse=True)
                fillable = (grid == bg)
                for obj in objs:
                    for rr in range(obj.height):
                        for cc in range(obj.width):
                            if obj.mask[rr, cc]:
                                pr, pc = obj.row + rr + dr, obj.col + cc + dc
                                while 0 <= pr < result.shape[0] and 0 <= pc < result.shape[1]:
                                    if result[pr, pc] in stop:
                                        break
                                    if fillable[pr, pc]:
                                        result[pr, pc] = obj.color
                                    else:
                                        break
                                    pr += dr
                                    pc += dc
                return result
            else:
                objs = [o for o in facts.objects if o.color == self.selector]
                for obj in objs:
                    for dr, dc in dirs:
                        for rr in range(obj.height):
                            for cc in range(obj.width):
                                if obj.mask[rr, cc]:
                                    cells = _ray_cells(obj.row + rr, obj.col + cc,
                                                       dr, dc, max(grid.shape), grid.shape)
                                    result = prim_trace(result, cells, self.selector, bg,
                                                        write='bg_only', stop_colors=stop)
                return result

        if self.route == 'path':
            if self.composition == 'star':
                anchors = [o for o in facts.objects if o.color == self.selector]
                if not anchors:
                    return grid
                a = anchors[0]
                targets = [o for o in facts.objects if o.color != self.selector]
                for t in targets:
                    cells = _path_cells(a.row, a.col, t.row, t.col, self.order)
                    result = prim_trace(result, cells, self.color, bg, write='skip')
                return result
            else:
                # Pairwise: selector is (color_a, color_b)
                ca, cb = self.selector
                oas = [o for o in facts.objects if o.color == ca]
                obs = [o for o in facts.objects if o.color == cb]
                if oas and obs:
                    cells = _path_cells(oas[0].row, oas[0].col,
                                        obs[0].row, obs[0].col, self.order)
                    result = prim_trace(result, cells, self.color, bg, write='skip')
                return result

        if self.route == 'line':
            ca, cb = self.selector
            oas = [o for o in facts.objects if o.color == ca]
            obs = [o for o in facts.objects if o.color == cb]
            if oas and obs:
                cells = _line_cells(int(oas[0].center_row), int(oas[0].center_col),
                                    int(obs[0].center_row), int(obs[0].center_col))
                result = prim_trace(result, cells, self.color, bg, write='skip')
            return result

        if self.route == 'ray_derived':
            # Direction derived from shape geometry
            if len(facts.objects) < 2:
                return grid
            shape = facts.objects[0]
            markers = [o for o in facts.objects if o.color != shape.color and o.size <= 2]
            if not markers:
                return grid
            pd = _derive_shape_direction(shape, markers[0], grid, bg)
            if pd is None:
                return grid
            for m in markers:
                d = max(grid.shape)
                result = prim_cast_rays(result, m.row, m.col, m.color,
                                         pd, d, bg, collision='skip')
            return result

        return result


def _derive_shape_direction(shape, marker, grid, bg):
    """Derive the direction a shape points from its geometry."""
    sub = grid[shape.row:shape.row + shape.height, shape.col:shape.col + shape.width]
    rw = [int(np.sum(sub[r] != bg)) for r in range(shape.height)]
    cw = [int(np.sum(sub[:, c] != bg)) for c in range(shape.width)]
    pd = None
    if len(rw) >= 2:
        if rw[0] < rw[-1]: pd = 'up'
        elif rw[0] > rw[-1]: pd = 'down'
    if pd is None and len(cw) >= 2:
        if cw[0] < cw[-1]: pd = 'left'
        elif cw[0] > cw[-1]: pd = 'right'
    if pd is None:
        dr = marker.center_row - (shape.row + shape.height / 2)
        dc = marker.center_col - (shape.col + shape.width / 2)
        if abs(dr) > abs(dc) * 2:
            pd = 'down' if dr > 0 else 'up'
        elif abs(dc) > abs(dr) * 2:
            pd = 'right' if dc > 0 else 'left'
        elif abs(dr) > 0.5 and abs(dc) > 0.5:
            v = 'down' if dr > 0 else 'up'
            h = 'right' if dc > 0 else 'left'
            pd = f'{v}_{h}'
    return pd


def _get_separator_direction(facts):
    """Determine fill direction from separator position."""
    if not facts.separators:
        return None
    sep = facts.separators[0]
    if sep.axis == 'col':
        if sep.index > facts.cols // 2:
            return [(0, 1)]
        else:
            return [(0, -1)]
    else:
        if sep.index > facts.rows // 2:
            return [(1, 0)]
        else:
            return [(-1, 0)]


# ---------------------------------------------------------------------------
# Unified trace rule search (replaces _gen_* family catalog)
# ---------------------------------------------------------------------------

def _search_trace_rules(contexts):
    """Search for TraceRules with output-guided parameter binding.

    One search function over the full parameter space instead of
    separate _gen_* micro-solvers.
    """
    from aria.guided.dsl import prim_cast_rays, prim_draw_path, prim_draw_line, _DIR_MAP
    rules = []
    ctx0 = contexts[0]
    bg = ctx0['facts'].bg
    inp0 = ctx0['input']
    out0 = ctx0['output']

    all_obj_colors = set()
    all_out_colors = set()
    for ctx in contexts:
        for obj in ctx['facts'].objects:
            all_obj_colors.add(obj.color)
        all_out_colors |= set(int(v) for v in np.unique(ctx['output'])) - {ctx['facts'].bg}

    # --- Ray traces: selector × color × directions × distance × write ---
    for sel_color in all_obj_colors:
        objs = [o for o in ctx0['facts'].objects if o.color == sel_color]
        if not objs:
            continue
        for ray_color in all_out_colors:
            if ray_color == bg:
                continue
            for directions in ['cardinal', 'diagonal', 'all',
                                'up', 'down', 'left', 'right']:
                for write in ['bg_only', 'skip']:
                    for dist in range(1, max(inp0.shape)):
                        coll = 'stop' if write == 'bg_only' else 'skip'
                        result = inp0.copy()
                        for obj in objs:
                            result = prim_cast_rays(
                                result, obj.row, obj.col, ray_color,
                                directions, dist, bg, collision=coll)
                        wrong = (result != inp0) & (result != out0)
                        if np.any(wrong):
                            break
                        if np.any(result != inp0):
                            rules.append(TraceRule(
                                selector=sel_color, route='ray', color=ray_color,
                                write=write, directions=directions, distance=dist,
                                description=f"color_{sel_color}→cast_{directions}(color={ray_color}, dist={dist}, {write})",
                            ))
                            break

    # --- Fill toward: selector × direction ---
    dir_options = [('right', [(0, 1)]), ('left', [(0, -1)]),
                   ('down', [(1, 0)]), ('up', [(-1, 0)]),
                   ('toward_sep', None)]
    for sel_color in all_obj_colors:
        for dir_name, dir_list in dir_options:
            actual_dirs = dir_list
            if dir_name == 'toward_sep':
                actual_dirs = _get_separator_direction(ctx0['facts'])
                if not actual_dirs:
                    continue
            stop = {s.color for s in ctx0['facts'].separators}
            result = ctx0['input'].copy()
            for obj in ctx0['facts'].objects:
                if obj.color == sel_color:
                    for dr, dc in actual_dirs:
                        for r in range(obj.height):
                            for c in range(obj.width):
                                if obj.mask[r, c]:
                                    from aria.guided.dsl import prim_fill_toward
                                    result = prim_fill_toward(
                                        result, obj.row + r, obj.col + c,
                                        dr, dc, sel_color, bg,
                                        stop_colors=stop if stop else None)
            wrong = (result != ctx0['input']) & (result != ctx0['output'])
            if np.any(result != ctx0['input']) and not np.any(wrong):
                rules.append(TraceRule(
                    selector=sel_color, route='fill', color='own',
                    write='bg_only', directions=dir_name,
                    stop_colors=frozenset(stop) if stop else None,
                    description=f"color_{sel_color}→fill_{dir_name}",
                ))

    # --- Fill ordered by distance to separator ---
    if ctx0['facts'].separators:
        sep_dir = _get_separator_direction(ctx0['facts'])
        if sep_dir:
            stop = {s.color for s in ctx0['facts'].separators}
            sep = ctx0['facts'].separators[0]
            dr, dc = sep_dir[0]
            def _dist_to_sep(obj):
                if sep.axis == 'col':
                    return abs(obj.center_col - sep.index)
                return abs(obj.center_row - sep.index)
            non_sep = [o for o in ctx0['facts'].objects if o.color not in stop]
            sorted_objs = sorted(non_sep, key=_dist_to_sep, reverse=True)
            fillable = (ctx0['input'] == bg)
            result = ctx0['input'].copy()
            for obj in sorted_objs:
                for r in range(obj.height):
                    for c in range(obj.width):
                        if obj.mask[r, c]:
                            pr, pc = obj.row + r + dr, obj.col + c + dc
                            while 0 <= pr < result.shape[0] and 0 <= pc < result.shape[1]:
                                if result[pr, pc] in stop:
                                    break
                                if fillable[pr, pc]:
                                    result[pr, pc] = obj.color
                                else:
                                    break
                                pr += dr
                                pc += dc
            wrong = (result != ctx0['input']) & (result != ctx0['output'])
            if not np.any(wrong) and np.any(result != ctx0['input']):
                rules.append(TraceRule(
                    selector='all_ordered', route='fill', color='own',
                    write='bg_only', directions='toward_sep',
                    stop_colors=frozenset(stop),
                    composition='ordered',
                    description='all objects fill toward separator (ordered by distance)',
                ))

    # --- Path/line traces: output-guided endpoint binding ---
    from scipy import ndimage
    new_px = (out0 != bg) & (inp0 == bg)
    if int(np.sum(new_px)) >= 2:
        new_colors = set(int(v) for v in out0[new_px])
        if len(new_colors) == 1:
            path_color = next(iter(new_colors))
            struct = np.ones((3, 3))
            out_nonbg = out0 != bg
            nbr = ndimage.convolve(out_nonbg.astype(int), struct, mode='constant') - out_nonbg.astype(int)
            if np.median(nbr[new_px]) <= 2.5:
                # Find endpoint objects (adjacent to path pixels)
                endpoint_oids = set()
                for obj in ctx0['facts'].objects:
                    for r in range(obj.height):
                        for c in range(obj.width):
                            if obj.mask[r, c]:
                                gr, gc = obj.row + r, obj.col + c
                                for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                    nr, nc = gr+ddr, gc+ddc
                                    if 0 <= nr < inp0.shape[0] and 0 <= nc < inp0.shape[1]:
                                        if new_px[nr, nc]:
                                            endpoint_oids.add(obj.oid)
                ep_objs = [o for o in ctx0['facts'].objects if o.oid in endpoint_oids]

                # Pairwise paths
                for i, a in enumerate(ep_objs):
                    for b in ep_objs[i+1:]:
                        for order in ['hv', 'vh']:
                            result = inp0.copy()
                            result = prim_draw_path(result, a.row, a.col,
                                                     b.row, b.col, path_color, bg, order)
                            wrong = (result != inp0) & (result != out0)
                            if not np.any(wrong) and np.any(result != inp0):
                                rules.append(TraceRule(
                                    selector=(a.color, b.color), route='path',
                                    color=path_color, write='skip', order=order,
                                    description=f"path color_{a.color}↔color_{b.color} ({order}, color={path_color})",
                                ))

                # Star paths (one anchor to all)
                for anchor in ep_objs:
                    targets = [o for o in ep_objs if o.oid != anchor.oid]
                    if not targets:
                        continue
                    for order in ['hv', 'vh']:
                        result = inp0.copy()
                        for t in targets:
                            result = prim_draw_path(result, anchor.row, anchor.col,
                                                     t.row, t.col, path_color, bg, order)
                        wrong = (result != inp0) & (result != out0)
                        if not np.any(wrong) and np.any(result != inp0):
                            rules.append(TraceRule(
                                selector=anchor.color, route='path', color=path_color,
                                write='skip', order=order, composition='star',
                                description=f"path from color_{anchor.color} to all ({order}, color={path_color})",
                            ))

                # Straight lines between pairs
                for i, a in enumerate(ep_objs):
                    for b in ep_objs[i+1:]:
                        result = inp0.copy()
                        result = prim_draw_line(result, int(a.center_row), int(a.center_col),
                                                int(b.center_row), int(b.center_col),
                                                path_color, bg)
                        wrong = (result != inp0) & (result != out0)
                        if not np.any(wrong) and np.any(result != inp0):
                            rules.append(TraceRule(
                                selector=(a.color, b.color), route='line',
                                color=path_color, write='skip',
                                description=f"line color_{a.color}↔color_{b.color} (color={path_color})",
                            ))

    # --- Ray from shape tip: derived direction ---
    if len(ctx0['facts'].objects) >= 2:
        shape = ctx0['facts'].objects[0]
        markers = [o for o in ctx0['facts'].objects if o.color != shape.color and o.size <= 2]
        if markers:
            pd = _derive_shape_direction(shape, markers[0], inp0, bg)
            if pd:
                for marker in markers:
                    dist = max(inp0.shape)
                    result = inp0.copy()
                    result = prim_cast_rays(result, marker.row, marker.col,
                                             marker.color, pd, dist, bg, collision='skip')
                    wrong = (result != inp0) & (result != out0)
                    if not np.any(wrong) and np.any(result != inp0):
                        rules.append(TraceRule(
                            selector='markers', route='ray_derived', color=marker.color,
                            write='skip',
                            description=f"cast ray from marker (color={marker.color}) in shape point direction ({pd}, skip)",
                        ))

    return rules



def _build_executor(key, vals, contexts):
    """Build an execute function from a synthesis result.

    The key describes the composition (e.g., 'crop_bbox(input, sel_largest)').
    We need to reconstruct this at test time.
    """
    # Parse the key to rebuild the computation
    # For now: use a simple approach — store the composition as a lambda
    # that re-derives from the input

    def _exec(inp):
        facts = perceive(inp)
        # Rebuild seeds
        seeds = {'input': inp, 'facts': facts}
        from aria.guided.clause import Predicate, Pred
        predicates = {
            'largest': [Predicate(Pred.IS_LARGEST)],
            'smallest': [Predicate(Pred.IS_SMALLEST)],
            'unique_color': [Predicate(Pred.UNIQUE_COLOR)],
        }
        for name, preds in predicates.items():
            selected = prim_select(facts, preds)
            if selected:
                seeds[f'sel_{name}'] = selected[0]
        for i, obj in enumerate(facts.objects):
            seeds[f'obj_{i}'] = obj
        # Selector constants (only add if key not already used by an ObjFact seed)
        selector_seeds = {
            'sel_largest': [Predicate(Pred.IS_LARGEST)],
            'sel_smallest': [Predicate(Pred.IS_SMALLEST)],
            'sel_unique_color': [Predicate(Pred.UNIQUE_COLOR)],
            'sel_topmost': [Predicate(Pred.IS_TOPMOST)],
            'sel_bottommost': [Predicate(Pred.IS_BOTTOMMOST)],
            'sel_leftmost': [Predicate(Pred.IS_LEFTMOST)],
            'sel_rightmost': [Predicate(Pred.IS_RIGHTMOST)],
        }
        for obj in facts.objects:
            selector_seeds[f'sel_color_{obj.color}'] = [Predicate(Pred.COLOR_EQ, obj.color)]
        for k, v in selector_seeds.items():
            if k not in seeds:
                seeds[k] = v

        # Evaluate the key expression
        return _eval_key(key, seeds, inp, facts)

    return _exec


def _eval_key(key, seeds, inp, facts):
    """Evaluate a synthesis key expression."""
    # Base case: literal seed
    if key in seeds:
        return seeds[key]

    # Color constant: color_N
    if key.startswith('color_') and key[6:].isdigit():
        return int(key[6:])

    # Parse: op_name(arg1, arg2, ...)
    if '(' not in key:
        raise ValueError(f"Cannot evaluate: {key}")
    paren = key.index('(')
    op_name = key[:paren]
    args_str = key[paren + 1:-1]

    # Split args (handle nested parens)
    args = _split_args(args_str)
    arg_vals = [_eval_key(a.strip(), seeds, inp, facts) for a in args]

    # Dispatch
    if op_name == 'crop_bbox':
        return prim_crop_bbox(arg_vals[0], arg_vals[1])
    elif op_name == 'crop_interior':
        return _safe_crop_interior(arg_vals[0], arg_vals[1])
    elif op_name == 'split':
        return prim_split_at_separator(arg_vals[0], facts, 0)
    elif op_name.startswith('combine_'):
        op = op_name[len('combine_'):]
        pair = arg_vals[0]
        if pair is None:
            return None
        a, b = pair
        return prim_combine(a, b, op, facts.bg)
    elif op_name == 'render':
        mask = arg_vals[0]
        color = arg_vals[1]
        if mask is None:
            return None
        return prim_render_mask(mask, color, facts.bg)
    elif op_name.startswith('repair_interior_'):
        axis = op_name[len('repair_interior_'):]
        return _safe_repair_interior(arg_vals[0], arg_vals[1], axis)
    elif op_name.startswith('repair_interior_'):
        axis = op_name[len('repair_interior_'):]
        return _safe_repair_interior(arg_vals[0], arg_vals[1], axis)
    elif op_name == 'repair_all_frames':
        result = _repair_all_frames(arg_vals[0])
        return result if result is not None else arg_vals[0]
    elif op_name.startswith('repair_region'):
        from aria.guided.dsl import prim_repair_region
        result = prim_repair_region(arg_vals[0])
        return result if result is not None else arg_vals[0]
    elif op_name == 'first':
        return arg_vals[0][0] if arg_vals[0] else None
    elif op_name == 'second':
        return arg_vals[0][1] if arg_vals[0] else None
    elif op_name.startswith('grid_'):
        xform = op_name[len('grid_'):]
        g = arg_vals[0]
        if xform == 'flip_h': return g[:, ::-1]
        elif xform == 'flip_v': return g[::-1, :]
        elif xform == 'flip_hv': return g[::-1, ::-1]
        elif xform == 'rot90': return np.rot90(g)
        elif xform == 'rot180': return np.rot90(g, 2)
        return g
    elif op_name.startswith('mirror_'):
        mode = op_name[len('mirror_'):]
        return _safe_mirror(arg_vals[0], mode)
    elif op_name == 'perceive':
        return perceive(arg_vals[0]).objects
    elif op_name == 'get_facts':
        return facts
    elif op_name == 'select':
        return _safe_select(arg_vals[0], arg_vals[1])
    elif op_name == 'get_color':
        return arg_vals[0].color
    elif op_name == 'get_mask':
        return arg_vals[0].mask
    elif op_name == 'bind_contains':
        return _bind_contains(arg_vals[0])
    elif op_name == 'bind_adjacent':
        return _bind_adjacent(arg_vals[0])
    elif op_name == 'bind_same_shape':
        return _bind_same_shape(arg_vals[0])
    elif op_name == 'binding_host':
        return arg_vals[0][0] if arg_vals[0] else None
    elif op_name == 'binding_content':
        return arg_vals[0][1] if arg_vals[0] else None
    elif op_name == 'crop_host_bbox':
        return prim_crop_bbox(arg_vals[0], arg_vals[1][0]) if arg_vals[1] else None
    elif op_name == 'crop_content_bbox':
        return prim_crop_bbox(arg_vals[0], arg_vals[1][1]) if arg_vals[1] else None
    elif op_name.startswith('transform_'):
        xform = op_name[len('transform_'):]
        r = arg_vals[0]
        if xform == 'flip_h': return r[:, ::-1]
        elif xform == 'flip_v': return r[::-1, :]
        elif xform == 'flip_hv': return r[::-1, ::-1]
        elif xform == 'rot90': return np.rot90(r)
        elif xform == 'rot180': return np.rot90(r, 2)
        elif xform == 'transpose': return r.T
        return r
    elif op_name == 'transition_source':
        return arg_vals[0].in_obj if arg_vals[0] else None
    elif op_name == 'transition_target':
        return arg_vals[0].out_obj if arg_vals[0] else None
    elif op_name == 'transition_color_to':
        return arg_vals[0].color_to if arg_vals[0] and arg_vals[0].color_to >= 0 else None
    elif op_name == 'transition_color_from':
        return arg_vals[0].color_from if arg_vals[0] and arg_vals[0].color_from >= 0 else None
    else:
        raise ValueError(f"Unknown op: {op_name}")


def _split_args(s):
    """Split comma-separated args, respecting nested parentheses."""
    args = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(' :
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current))
    return args
