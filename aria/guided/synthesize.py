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
    for xform_name, xfn in [('flip_h', lambda r: r[::-1, :]),
                              ('flip_v', lambda r: r[:, ::-1]),
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
    for xform_name, xfn in [('flip_h', lambda r: r[::-1, :]),
                              ('flip_v', lambda r: r[:, ::-1]),
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
    # Per-demo execution contexts
    contexts = []
    for inp, out in demos:
        facts = perceive(inp)
        ctx = {
            'input': inp,
            'output': out,
            'facts': facts,
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
    }

    # Color-based selectors for each color present
    for obj in facts.objects:
        name = f'sel_color_{obj.color}'
        if (name, Ty.SELECTOR) not in seeds:
            selectors[name] = [Predicate(Pred.COLOR_EQ, obj.color)]

    for name, preds in selectors.items():
        seeds[(name, Ty.SELECTOR)] = preds


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
                     Pred.TOUCHES_BORDER, Pred.NOT_TOUCHES_BORDER]:
            sel = [Predicate(pred)]
            selected = set(o.oid for o in prim_select(facts, sel))
            if selected == target_set:
                candidates.append(sel)

        return candidates

    # For each pair of action groups, try to find selectors
    group_keys = sorted(action_groups.keys(),
                        key=lambda k: len(action_groups[k]), reverse=True)

    for i, key_a in enumerate(group_keys):
        for key_b in group_keys[i+1:]:
            oids_a = action_groups[key_a]
            oids_b = action_groups[key_b]
            sels_a = _find_selector(oids_a, facts0.objects, facts0)
            sels_b = _find_selector(oids_b, facts0.objects, facts0)
            if not sels_a or not sels_b:
                continue

            # Try each combination of selectors
            for sel_a in sels_a[:3]:
                for sel_b in sels_b[:3]:
                    # Verify disjointness
                    selected_a = set(o.oid for o in prim_select(facts0, sel_a))
                    selected_b = set(o.oid for o in prim_select(facts0, sel_b))
                    if selected_a & selected_b:
                        continue

                    # Build the conditional program
                    action_a, param_a = key_a
                    action_b, param_b = key_b

                    def _make_cond_fn(sa, sb, act_a, prm_a, act_b, prm_b):
                        def _exec(inp):
                            facts = perceive(inp)
                            _bg = facts.bg

                            canvas = np.full_like(inp, _bg)
                            matched_a = set(o.oid for o in prim_select(facts, sa))
                            matched_b = set(o.oid for o in prim_select(facts, sb))

                            for obj in facts.objects:
                                if obj.oid in matched_a:
                                    _apply_action(canvas, obj, act_a, prm_a, _bg, inp)
                                elif obj.oid in matched_b:
                                    _apply_action(canvas, obj, act_b, prm_b, _bg, inp)
                                else:
                                    _apply_action(canvas, obj, 'keep', None, _bg, inp)
                            return canvas
                        return _exec

                    fn = _make_cond_fn(sel_a, sel_b, action_a, param_a,
                                        action_b, param_b)

                    # Verify on ALL demos
                    all_ok = True
                    for ctx in contexts:
                        pred = fn(ctx['input'])
                        if not np.array_equal(pred, ctx['output']):
                            all_ok = False
                            break

                    if all_ok:
                        desc_a = _describe_selector(sel_a)
                        desc_b = _describe_selector(sel_b)
                        act_desc_a = _describe_action(action_a, param_a)
                        act_desc_b = _describe_action(action_b, param_b)
                        desc = f"if [{desc_a}] then {act_desc_a}; if [{desc_b}] then {act_desc_b}"
                        return _make_program(fn, f"synth: dispatch {desc}")

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

    elif action == 'fill_interior':
        # Keep the object, fill bg holes with param color
        for r in range(h):
            for c in range(w):
                nr, nc = obj.row + r, obj.col + c
                if 0 <= nr < rows and 0 <= nc < cols:
                    if obj.mask[r, c]:
                        canvas[nr, nc] = obj.color
                    elif inp is not None and inp[nr, nc] == bg:
                        # Check if this bg cell is "inside" the object bbox
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


def _describe_action(action, param):
    if action == 'keep':
        return 'keep'
    elif action == 'recolor':
        return f'recolor({param})'
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
        seeds = {'input': inp}
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
        if xform == 'flip_h': return g[::-1, :]
        elif xform == 'flip_v': return g[:, ::-1]
        elif xform == 'flip_hv': return g[::-1, ::-1]
        elif xform == 'rot90': return np.rot90(g)
        elif xform == 'rot180': return np.rot90(g, 2)
        return g
    elif op_name.startswith('mirror_'):
        mode = op_name[len('mirror_'):]
        return _safe_mirror(arg_vals[0], mode)
    elif op_name == 'perceive':
        return perceive(arg_vals[0]).objects
    elif op_name == 'get_color':
        return arg_vals[0].color
    elif op_name == 'get_mask':
        return arg_vals[0].mask
    elif op_name.startswith('transform_'):
        xform = op_name[len('transform_'):]
        r = arg_vals[0]
        if xform == 'flip_h': return r[::-1, :]
        elif xform == 'flip_v': return r[:, ::-1]
        elif xform == 'flip_hv': return r[::-1, ::-1]
        elif xform == 'rot90': return np.rot90(r)
        elif xform == 'rot180': return np.rot90(r, 2)
        elif xform == 'transpose': return r.T
        return r
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
