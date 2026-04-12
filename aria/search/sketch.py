"""Flat sketch representation for search.

Search operates over SearchProgram (flat sequence of SearchSteps).
Verified sketches lower to canonical AST for execution and storage.

Design from ericagi2's ProgramIR/SeedSchema, adapted for aria's
perception layer and AST execution target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aria.search.ast import ASTNode, Op


# ---------------------------------------------------------------------------
# Param expressions (per-object parameterization)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamExpr:
    """A tiny expression for per-object parameters.

    Evaluated at execution time against a specific object and scene facts.
    Keeps derive output declarative — no code synthesis needed.

    Ops:
      const(v)          → literal value v
      field(name)       → object attribute (color, size, row, col, height, width)
      rank(field)       → 1-based rank among selected objects by field (desc)
      mod(field, k)     → obj.field % k
      count(pred_name)  → count of objects matching a named predicate
      lookup(field, table) → table[obj.field], where table is {int: int}
    """
    op: str
    args: tuple = ()

    def to_dict(self) -> dict:
        d: dict = {'op': self.op}
        if self.args:
            serializable = []
            for a in self.args:
                if isinstance(a, dict):
                    serializable.append({str(k): v for k, v in a.items()})
                else:
                    serializable.append(a)
            d['args'] = serializable
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ParamExpr:
        return cls(op=d['op'], args=tuple(d.get('args', ())))


# ---------------------------------------------------------------------------
# Typed selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepSelect:
    """Typed object/region selection for a search step.

    role: Selection criterion. One of:
        - 'largest', 'smallest', 'unique_color' (structural)
        - 'topmost', 'bottommost', 'leftmost', 'rightmost' (positional)
        - 'by_color' (param: color int)
        - 'by_predicate' (param: predicate list)
        - 'contained_by', 'adjacent_to', 'contains' (relational, param: inner role)
        - 'all' (every non-bg object)
    params: Additional parameters (color, predicate details, etc.)
    from_step: Reuse selection from earlier step (for multi-step compositions).
    """
    role: str
    params: dict[str, Any] = field(default_factory=dict)
    from_step: int | None = None

    def to_dict(self) -> dict:
        d: dict = {'role': self.role}
        if self.params:
            serializable = {}
            for k, v in self.params.items():
                if isinstance(v, list) and v and hasattr(v[0], 'pred'):
                    # Serialize Predicate objects
                    serializable[str(k)] = [p.to_dict() for p in v]
                elif hasattr(v, 'to_dict'):
                    # Serialize nested objects (e.g. StepSelect)
                    serializable[str(k)] = v.to_dict()
                else:
                    serializable[str(k)] = v
            if serializable:
                d['params'] = serializable
        if self.from_step is not None:
            d['from_step'] = self.from_step
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StepSelect:
        params = dict(d.get('params', {}))
        # Deserialize predicates
        if 'predicates' in params and isinstance(params['predicates'], list):
            from aria.guided.clause import Predicate
            deserialized = []
            for p in params['predicates']:
                if isinstance(p, dict) and 'pred' in p:
                    deserialized.append(Predicate.from_dict(p))
                else:
                    deserialized.append(p)
            if deserialized:
                params['predicates'] = deserialized
        # Deserialize nested selector
        if 'selector' in params and isinstance(params['selector'], dict):
            params['selector'] = cls.from_dict(params['selector'])
        return cls(role=d['role'], params=params,
                   from_step=d.get('from_step'))

    def to_predicates(self):
        """Convert to aria.guided.clause Predicate list for execution."""
        from aria.guided.clause import Predicate, Pred

        role = self.role
        if role == 'largest':
            return [Predicate(Pred.IS_LARGEST)]
        if role == 'smallest':
            return [Predicate(Pred.IS_SMALLEST)]
        if role == 'unique_color':
            return [Predicate(Pred.UNIQUE_COLOR)]
        if role == 'topmost':
            return [Predicate(Pred.IS_TOPMOST)]
        if role == 'bottommost':
            return [Predicate(Pred.IS_BOTTOMMOST)]
        if role == 'leftmost':
            return [Predicate(Pred.IS_LEFTMOST)]
        if role == 'rightmost':
            return [Predicate(Pred.IS_RIGHTMOST)]
        if role == 'by_color':
            return [Predicate(Pred.COLOR_EQ, self.params.get('color', 0))]
        if role == 'singleton':
            return [Predicate(Pred.IS_SINGLETON)]
        if role == 'rectangular':
            return [Predicate(Pred.IS_RECTANGULAR)]
        if role == 'line':
            return [Predicate(Pred.IS_LINE)]
        if role == 'touches_border':
            return [Predicate(Pred.TOUCHES_BORDER)]
        if role == 'interior':
            return [Predicate(Pred.NOT_TOUCHES_BORDER)]
        if role == 'contained_by':
            inner = self.params.get('inner', 'largest')
            inner_sel = StepSelect(role=inner)
            return [Predicate(Pred.CONTAINED_BY, inner_sel.to_predicates()[0])]
        if role == 'adjacent_to':
            inner = self.params.get('inner', 'largest')
            inner_sel = StepSelect(role=inner)
            return [Predicate(Pred.ADJACENT_TO, inner_sel.to_predicates()[0])]
        if role == 'all':
            return [Predicate(Pred.SIZE_GT, 0)]
        if role == 'by_predicate':
            return self.params.get('predicates', [])
        # by_rule is NOT routed through to_predicates — it stays at the
        # search level and is resolved by select_objects() directly.
        return []

    def select_objects(self, facts) -> list:
        """Search-level object selection. Handles rule-based selectors.

        Rule-based selectors stay in the search layer (selection_facts.py)
        and never touch guided/clause.py predicates.
        """
        if self.role == 'by_rule':
            from aria.search.selection_facts import select_by_rule
            return select_by_rule(self.params.get('rule', {}), facts)
        from aria.guided.dsl import prim_select
        preds = self.to_predicates()
        return prim_select(facts, preds) if preds else list(facts.objects)


# ---------------------------------------------------------------------------
# Search step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchStep:
    """One step in a search program.

    action: What to do. Maps to AST ops:
        'recolor', 'remove', 'move', 'gravity', 'slide', 'stamp',
        'transform', 'fill_interior', 'fill_enclosed',
        'crop_bbox', 'crop_interior', 'split_combine_render',
        'tile', 'periodic_extend', 'repair_frames',
        'flip_h', 'flip_v', 'flip_hv', 'rot90', 'rot180',
        'mirror', 'trace'
    params: Action-specific parameters (direction, color, offset, etc.)
    select: Which objects this step targets (None = whole grid).
    """
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    select: StepSelect | None = None

    def to_dict(self) -> dict:
        d: dict = {'action': self.action}
        if self.params:
            d['params'] = self.params
        if self.select is not None:
            d['select'] = self.select.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SearchStep:
        sel = StepSelect.from_dict(d['select']) if 'select' in d else None
        return cls(action=d['action'], params=d.get('params', {}), select=sel)

    def to_ast(self) -> ASTNode:
        """Lower this step to an AST node."""
        return _step_to_ast(self)


# ---------------------------------------------------------------------------
# Search program
# ---------------------------------------------------------------------------

@dataclass
class SearchProgram:
    """Flat sequence of search steps. The search representation.

    Search explores these. Verified programs lower to AST for execution.
    """
    steps: list[SearchStep]
    provenance: str = ''    # how this was found (schema name, search path)

    def to_dict(self) -> dict:
        return {
            'steps': [s.to_dict() for s in self.steps],
            'provenance': self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SearchProgram:
        return cls(
            steps=[SearchStep.from_dict(s) for s in d['steps']],
            provenance=d.get('provenance', ''),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SearchProgram:
        return cls.from_dict(json.loads(s))

    @property
    def signature(self) -> str:
        """Structural signature for grouping."""
        return ' -> '.join(s.action for s in self.steps)

    def to_ast(self) -> ASTNode:
        """Lower to canonical AST for execution."""
        if len(self.steps) == 1:
            return self.steps[0].to_ast()
        return ASTNode(Op.COMPOSE, [s.to_ast() for s in self.steps])

    def execute(self, inp: np.ndarray) -> np.ndarray:
        """Execute via AST lowering, with fallback for ops AST can't express."""
        result = inp
        for step in self.steps:
            if step.action == 'gravity_nearest':
                result = _exec_gravity_nearest(result, step)
            elif step.action == 'slide':
                result = _exec_slide(result, step)
            elif step.action == 'panel_boolean':
                result = _exec_panel_boolean(result, step)
            elif step.action == 'marker_stamp':
                result = _exec_marker_stamp(result, step)
            elif step.action == 'registration_transfer':
                result = _exec_registration_transfer(result, step)
            elif step.action == 'grid_fill_between':
                result = _exec_grid_fill_between(result, step)
            elif step.action == 'grid_slot_transfer':
                result = _exec_grid_slot_transfer(result, step)
            elif step.action == 'grid_cell_pack':
                result = _exec_grid_cell_pack(result, step)
            elif step.action == 'grid_conditional_transfer':
                result = _exec_grid_conditional_transfer(result, step)
            elif step.action == 'object_grid_pack':
                result = _exec_object_grid_pack(result, step)
            elif step.action == 'panel_legend_map':
                result = _exec_panel_legend_map(result, step)
            elif step.action == 'correspondence_transfer':
                result = _exec_correspondence_transfer(result, step)
            elif step.action == 'registration_anchor_transfer':
                result = _exec_registration_anchor_transfer(result, step)
            elif step.action == 'quadrant_template_decode':
                result = _exec_quadrant_template_decode(result, step)
            elif step.action == 'frame_bbox_pack':
                result = _exec_frame_bbox_pack(result, step)
            elif step.action == 'cross_stencil_recolor':
                from aria.search.executor import _exec_cross_stencil_recolor
                result = _exec_cross_stencil_recolor(result, step.params or {})
            elif step.action == 'legend_frame_fill':
                result = _exec_legend_frame_fill(result, step)
            elif step.action == 'anomaly_halo':
                from aria.search.executor import _exec_anomaly_halo
                result = _exec_anomaly_halo(result, step.params or {})
            elif step.action == 'object_highlight':
                result = _exec_object_highlight_full(result, step.params or {})
            elif step.action == 'cavity_transfer':
                from aria.search.executor import _exec_cavity_transfer
                result = _exec_cavity_transfer(result, step.params or {})
            elif step.action == 'recolor_map':
                result = _exec_recolor_map(result, step.params or {})
            elif step.action == 'color_stencil':
                from aria.search.derive import _exec_color_stencil
                result = _exec_color_stencil(result, step.params or {})
            elif step.action == 'crop_nonbg':
                result = _exec_crop_nonbg(result)
            elif step.action == 'crop_object':
                result = _exec_crop_object(result, step.params or {}, step.select)
            elif step.action == 'crop_fixed':
                p = step.params or {}
                result = result[p['r0']:p['r0']+p['h'], p['c0']:p['c0']+p['w']]
            elif step.action == 'tile':
                p = step.params or {}
                result = np.tile(result, (p.get('rows', 1), p.get('cols', 1)))
            elif step.action == 'scale':
                f = (step.params or {}).get('factor', 1)
                result = np.repeat(np.repeat(result, f, axis=0), f, axis=1)
            else:
                from aria.search.executor import execute_ast
                ast = step.to_ast()
                r = execute_ast(ast, result)
                result = r if r is not None else result
        return result

    def verify(self, demos: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify against all demo pairs."""
        for inp, out in demos:
            try:
                pred = self.execute(inp)
                if not np.array_equal(pred, out):
                    return False
            except Exception:
                return False
        return True


# ---------------------------------------------------------------------------
# Step → AST lowering
# ---------------------------------------------------------------------------

def _exec_gravity_nearest(inp, step):
    """Per-object gravity: each object moves to nearest border along axis."""
    from aria.guided.perceive import perceive

    facts = perceive(inp)
    bg = facts.bg
    axis = step.params.get('axis', 'vertical')

    targets = step.select.select_objects(facts) if step.select else facts.objects
    rows, cols = inp.shape
    result = inp.copy()

    # Clear all targets
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg

    # Place each at nearest border
    for obj in targets:
        if axis == 'vertical':
            dist_top = obj.center_row
            dist_bottom = rows - 1 - obj.center_row
            new_row = 0 if dist_top <= dist_bottom else rows - obj.height
            new_col = obj.col
        else:
            dist_left = obj.center_col
            dist_right = cols - 1 - obj.center_col
            new_row = obj.row
            new_col = 0 if dist_left <= dist_right else cols - obj.width

        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    nr, nc = new_row + r, new_col + c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = obj.color

    return result


def _exec_slide(inp, step):
    """Per-object slide: each object slides in direction until collision."""
    from aria.guided.perceive import perceive

    facts = perceive(inp)
    bg = facts.bg
    direction = step.params.get('direction', 'down')

    targets = step.select.select_objects(facts) if step.select else facts.objects
    rows, cols = inp.shape
    result = inp.copy()

    dr = {'down': 1, 'up': -1, 'right': 0, 'left': 0}[direction]
    dc = {'down': 0, 'up': 0, 'right': 1, 'left': -1}[direction]

    # Clear all targets
    target_oids = set(o.oid for o in targets)
    for obj in targets:
        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    result[obj.row + r, obj.col + c] = bg

    # Slide each until hitting non-bg (in the cleared grid) or border
    for obj in targets:
        shift = 0
        max_shift = max(rows, cols)
        for s in range(1, max_shift):
            blocked = False
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.mask[r, c]:
                        nr = obj.row + r + s * dr
                        nc = obj.col + c + s * dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            blocked = True
                            break
                        if result[nr, nc] != bg:
                            blocked = True
                            break
                if blocked:
                    break
            if blocked:
                shift = s - 1
                break
            shift = s

        for r in range(obj.height):
            for c in range(obj.width):
                if obj.mask[r, c]:
                    nr = obj.row + r + shift * dr
                    nc = obj.col + c + shift * dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = obj.color

    return result


def _exec_grid_fill_between(inp, step):
    """Fill empty grid cells between repeated content along rows and columns."""
    from aria.guided.perceive import perceive
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
        cell_content_color,
    )

    facts = perceive(inp)
    grid_info = detect_grid(facts)
    if grid_info is None:
        return inp

    params = step.params or {}
    mode = params.get('mode', 'color')
    fill_all = bool(params.get('fill_all', False))
    bg = facts.bg
    result = inp.copy()

    if fill_all:
        keys = []
        contents = []
        for r in range(grid_info.n_rows):
            for c in range(grid_info.n_cols):
                cell = grid_info.cell_at(r, c)
                if cell and cell_has_content(inp, cell, bg):
                    content = cell_content(inp, cell, bg)
                    if mode == 'pattern':
                        key = (content.shape, content.tobytes())
                    else:
                        color = cell_content_color(inp, cell, bg)
                        if color is None:
                            continue
                        key = ('color', int(color))
                    keys.append(key)
                    contents.append(content)

        if not keys:
            return result

        from collections import Counter
        key, _ = Counter(keys).most_common(1)[0]
        template = None
        if mode == 'pattern':
            for content in contents:
                if (content.shape, content.tobytes()) == key:
                    template = content
                    break
        else:
            template = key[1]

        for r in range(grid_info.n_rows):
            for c in range(grid_info.n_cols):
                cell = grid_info.cell_at(r, c)
                if cell and not cell_has_content(inp, cell, bg):
                    if mode == 'pattern' and template is not None:
                        h, w = cell.height, cell.width
                        sh, sw = template.shape
                        if sh > h or sw > w:
                            continue
                        result[cell.r0:cell.r0 + sh,
                               cell.c0:cell.c0 + sw] = template
                    elif mode == 'color':
                        result[cell.r0:cell.r0 + cell.height,
                               cell.c0:cell.c0 + cell.width] = int(template)
        return result

    for axis in ('row', 'col'):
        n_lines = grid_info.n_rows if axis == 'row' else grid_info.n_cols
        n_cross = grid_info.n_cols if axis == 'row' else grid_info.n_rows

        for line in range(n_lines):
            keys_at = {}
            content_at = {}
            for cross in range(n_cross):
                gr = line if axis == 'row' else cross
                gc = cross if axis == 'row' else line
                cell = grid_info.cell_at(gr, gc)
                if cell and cell_has_content(inp, cell, bg):
                    content = cell_content(inp, cell, bg)
                    if mode == 'pattern':
                        key = (content.shape, content.tobytes())
                    else:
                        c = cell_content_color(inp, cell, bg)
                        if c is None:
                            continue
                        key = ('color', int(c))
                    keys_at[cross] = key
                    content_at[cross] = content

            key_positions = {}
            for pos, key in keys_at.items():
                key_positions.setdefault(key, []).append(pos)

            for key, positions in key_positions.items():
                if len(positions) < 2:
                    continue
                positions.sort()
                lo, hi = positions[0], positions[-1]
                src = content_at[lo]
                for cross in range(lo, hi + 1):
                    if cross not in keys_at:
                        gr = line if axis == 'row' else cross
                        gc = cross if axis == 'row' else line
                        cell = grid_info.cell_at(gr, gc)
                        if cell:
                            h, w = cell.height, cell.width
                            sh, sw = src.shape
                            fh, fw = min(h, sh), min(w, sw)
                            result[cell.r0:cell.r0 + fh,
                                   cell.c0:cell.c0 + fw] = src[:fh, :fw]

    return result


def _exec_grid_cell_pack(inp, step):
    """Pack non-empty grid-cell contents into the grid in a fixed order."""
    from aria.guided.perceive import perceive
    from aria.search.grid_detect import (
        detect_grid, cell_content, cell_has_content,
    )

    facts = perceive(inp)
    grid_info = detect_grid(facts)
    if grid_info is None:
        return inp

    params = step.params or {}
    ordering = params.get('ordering', 'row')
    bg = facts.bg

    # Collect non-empty cell contents (full cell patches)
    items = []
    for r in range(grid_info.n_rows):
        for c in range(grid_info.n_cols):
            cell = grid_info.cell_at(r, c)
            if cell and cell_has_content(inp, cell, bg):
                items.append({
                    'row': r,
                    'col': c,
                    'content': cell_content(inp, cell, bg),
                    'cell': cell,
                })

    if ordering == 'col':
        items.sort(key=lambda x: (x['col'], x['row']))
    elif ordering == 'color':
        def _key(it):
            content = it['content']
            flat = content.ravel()
            non_bg = flat[flat != bg]
            if len(non_bg) == 0:
                return (999, it['row'], it['col'])
            # dominant color
            from collections import Counter
            color = Counter(non_bg.tolist()).most_common(1)[0][0]
            return (int(color), it['row'], it['col'])
        items.sort(key=_key)
    else:
        items.sort(key=lambda x: (x['row'], x['col']))

    result = np.full_like(inp, bg)
    # preserve separators if present
    if facts.separators:
        for sep in facts.separators:
            if sep.axis == 'row':
                result[sep.index, :] = sep.color
            else:
                result[:, sep.index] = sep.color

    idx = 0
    for r in range(grid_info.n_rows):
        for c in range(grid_info.n_cols):
            if idx >= len(items):
                break
            cell = grid_info.cell_at(r, c)
            if cell is None:
                continue
            content = items[idx]['content']
            h, w = content.shape
            if h > cell.height or w > cell.width:
                return inp
            result[cell.r0:cell.r0 + h,
                   cell.c0:cell.c0 + w] = content
            idx += 1

    return result


def _exec_grid_slot_transfer(inp, step):
    """Move source cell contents into empty target cells.

    Matching: size compatibility + spatial distance via Hungarian assignment.
    Same rule as derive — no output-content dependency.
    """
    from aria.guided.perceive import perceive
    from aria.search.grid_detect import detect_grid, cell_content, cell_has_content
    from scipy.optimize import linear_sum_assignment

    facts = perceive(inp)
    bg = facts.bg
    g = detect_grid(facts)
    if g is None:
        return inp

    sources = []
    targets = []
    for cell in g.cells:
        if cell_has_content(inp, cell, bg):
            sources.append(cell)
        else:
            targets.append(cell)

    if not sources or not targets or len(sources) > len(targets):
        return inp

    src_contents = [cell_content(inp, c, bg) for c in sources]
    # For execution, target masks are unknown — use empty placeholders
    n_src = len(sources)
    n_tgt = len(targets)
    cost = np.full((n_src, n_tgt), 1e6, dtype=float)
    for si in range(n_src):
        sc = src_contents[si]
        for ti in range(n_tgt):
            tc = targets[ti]
            # Content fits in target?
            if sc.shape[0] <= tc.height and sc.shape[1] <= tc.width:
                dist = abs(sources[si].r0 - tc.r0) + abs(sources[si].c0 - tc.c0)
                cost[si, ti] = dist

    rows, cols = linear_sum_assignment(cost)

    result = inp.copy()
    placed = []
    for si, ti in zip(rows, cols):
        if cost[si, ti] >= 1e6:
            continue
        placed.append((si, ti))

    if not placed:
        return inp

    # Clear sources
    for si, ti in placed:
        cell = sources[si]
        result[cell.r0:cell.r0 + cell.height,
               cell.c0:cell.c0 + cell.width] = bg

    # Place at targets
    for si, ti in placed:
        content = src_contents[si]
        tgt = targets[ti]
        h, w = content.shape
        for r in range(h):
            for c in range(w):
                if content[r, c] != bg:
                    result[tgt.r0 + r, tgt.c0 + c] = content[r, c]

    return result


def _exec_grid_conditional_transfer(inp, step):
    """Fill empty grid cells using a rule derived across demos.

    Rule types stored in step.params:
      - 'nearest_row': copy nearest non-empty cell in the same row
      - 'nearest_col': copy nearest non-empty cell in the same column
      - 'mirror_h': horizontal mirror of the content across grid center
      - 'mirror_v': vertical mirror of the content across grid center
    """
    from aria.guided.perceive import perceive
    from aria.search.grid_detect import detect_grid, cell_content, cell_has_content

    facts = perceive(inp)
    bg = facts.bg
    g = detect_grid(facts)
    if g is None:
        return inp

    params = step.params or {}
    rule = params.get('rule', 'nearest_row')
    result = inp.copy()

    # Build cell content map
    content_map = {}
    for cell in g.cells:
        if cell_has_content(inp, cell, bg):
            content_map[(cell.grid_row, cell.grid_col)] = cell_content(inp, cell, bg)

    for cell in g.cells:
        if cell_has_content(inp, cell, bg):
            continue
        gr, gc = cell.grid_row, cell.grid_col
        source = None

        if rule == 'nearest_row':
            # Find nearest non-empty in same row
            best_dist = float('inf')
            for (r, c), cnt in content_map.items():
                if r == gr and abs(c - gc) < best_dist:
                    best_dist = abs(c - gc)
                    source = cnt
        elif rule == 'nearest_col':
            best_dist = float('inf')
            for (r, c), cnt in content_map.items():
                if c == gc and abs(r - gr) < best_dist:
                    best_dist = abs(r - gr)
                    source = cnt
        elif rule == 'mirror_h':
            mirror_c = g.n_cols - 1 - gc
            source = content_map.get((gr, mirror_c))
        elif rule == 'mirror_v':
            mirror_r = g.n_rows - 1 - gr
            source = content_map.get((mirror_r, gc))
        elif rule == 'induced':
            from aria.search.derive import _CELL_MAPPINGS
            cell_map = params.get('cell_map', 'mirror_h')
            fn = _CELL_MAPPINGS.get(cell_map)
            if fn:
                src_r, src_c = fn(gr, gc, g.n_rows, g.n_cols)
                source = content_map.get((src_r, src_c))
        elif rule == 'parity_conditional':
            from aria.search.derive import _apply_simple_rule
            sub_rule = params.get('even_rule') if gr % 2 == 0 else params.get('odd_rule')
            if sub_rule:
                source = _apply_simple_rule(sub_rule, gr, gc, g, content_map)

        if source is not None:
            h = min(source.shape[0], cell.height)
            w = min(source.shape[1], cell.width)
            result[cell.r0:cell.r0 + h, cell.c0:cell.c0 + w] = source[:h, :w]

    return result


def _exec_correspondence_transfer(inp, step):
    """Place objects via correspondence-derived rules."""
    from aria.guided.perceive import perceive

    facts = perceive(inp)
    bg = facts.bg
    params = step.params or {}
    mode = params.get('mode', 'position_swap')

    if mode == 'color_permutation':
        return _exec_corr_color_permutation(inp, params)
    if mode == 'position_swap':
        return _exec_corr_position_swap(inp, facts, bg)
    return inp


def _exec_corr_color_permutation(inp, params):
    """Apply a global color permutation mapping."""
    mapping = params.get('mapping', {})
    if not mapping:
        return inp
    orig = inp.copy()
    result = inp.copy()
    # Apply mapping against the original grid to avoid overwrite collisions.
    for k, v in mapping.items():
        result[orig == int(k)] = int(v)
    return result


def _exec_corr_position_swap(inp, facts, bg):
    """Permute object positions among same-shape groups."""
    from scipy.optimize import linear_sum_assignment

    objs = [o for o in facts.objects if o.size >= 2]
    if len(objs) < 2:
        return inp

    result = inp.copy()
    # Group by shape signature (height, width, size)
    groups = {}
    for o in objs:
        key = (o.height, o.width, o.size)
        groups.setdefault(key, []).append(o)

    for group in groups.values():
        if len(group) < 2:
            continue
        n = len(group)
        positions = [(o.row, o.col) for o in group]
        cost = np.full((n, n), 1e6, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost[i, j] = abs(positions[i][0] - positions[j][0]) + \
                             abs(positions[i][1] - positions[j][1])

        rows, cols = linear_sum_assignment(cost)
        if any(cost[r, c] >= 1e6 for r, c in zip(rows, cols)):
            return inp

        for o in group:
            for r in range(o.height):
                for c in range(o.width):
                    if o.mask[r, c]:
                        result[o.row + r, o.col + c] = bg

        for i, j in zip(rows, cols):
            src = group[i]
            dst_row, dst_col = positions[j]
            for r in range(src.height):
                for c in range(src.width):
                    if src.mask[r, c]:
                        nr = dst_row + r
                        nc = dst_col + c
                        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                            result[nr, nc] = src.color

    return result


def _exec_object_grid_pack(inp, step):
    """Pack input objects into an output grid by ordering.

    step.params:
      - 'order': 'row_major', 'col_major', 'size_asc', 'size_desc', 'color_asc'
      - 'out_rows', 'out_cols': output grid dimensions in objects
      - 'cell_h', 'cell_w': cell size
      - 'sep': separator width (default 0)
      - 'placement': 'top_left' (default) or 'centered'
    """
    from aria.guided.perceive import perceive
    from aria.search.derive import _sort_objects

    facts = perceive(inp)
    bg = facts.bg
    params = step.params or {}
    order = params.get('order', 'row_major')
    sep = params.get('sep', 0)
    placement = params.get('placement', 'top_left')

    objs = [o for o in facts.objects if o.size > 0]
    if not objs:
        return inp

    _sort_objects(objs, order)

    patches = [inp[o.row:o.row + o.height, o.col:o.col + o.width].copy()
               for o in objs]

    cell_h = params.get('cell_h', max(o.height for o in objs))
    cell_w = params.get('cell_w', max(o.width for o in objs))
    out_cols = params.get('out_cols', len(objs))
    out_rows = params.get('out_rows', (len(objs) + out_cols - 1) // out_cols)

    total_h = out_rows * cell_h + max(0, out_rows - 1) * sep
    total_w = out_cols * cell_w + max(0, out_cols - 1) * sep
    result = np.full((total_h, total_w), bg, dtype=inp.dtype)

    for idx, patch in enumerate(patches):
        gr = idx // out_cols
        gc = idx % out_cols
        if gr >= out_rows:
            break
        r0 = gr * (cell_h + sep)
        c0 = gc * (cell_w + sep)
        ph, pw = patch.shape
        h = min(ph, cell_h)
        w = min(pw, cell_w)
        if placement == 'centered':
            dr = (cell_h - h) // 2
            dc = (cell_w - w) // 2
            result[r0 + dr:r0 + dr + h, c0 + dc:c0 + dc + w] = patch[:h, :w]
        else:
            result[r0:r0 + h, c0:c0 + w] = patch[:h, :w]

    return result


def _exec_panel_legend_map(inp, step):
    """Apply legend-derived color/pattern mapping from legend panel to target panel.

    step.params:
      - 'legend_side': 'left', 'right', 'top', 'bottom'
      - 'sep_idx': separator index splitting legend from target
      - 'axis': 'col' or 'row'
      - 'mapping': dict mapping source_color → target_color
    """
    from aria.guided.perceive import perceive

    facts = perceive(inp)
    bg = facts.bg
    params = step.params or {}
    mapping = params.get('mapping', {})
    sep_idx = params.get('sep_idx')
    axis = params.get('axis', 'col')
    legend_side = params.get('legend_side', 'left')

    if not mapping or sep_idx is None:
        return inp

    # Extract target region
    if axis == 'col':
        if legend_side == 'left':
            target = inp[:, sep_idx + 1:].copy()
        else:
            target = inp[:, :sep_idx].copy()
    else:
        if legend_side == 'top':
            target = inp[sep_idx + 1:, :].copy()
        else:
            target = inp[:sep_idx, :].copy()

    # Apply mapping
    color_map = {int(k): int(v) for k, v in mapping.items()}
    for src_c, tgt_c in color_map.items():
        target[target == src_c] = tgt_c

    # Reconstruct
    result = inp.copy()
    if axis == 'col':
        if legend_side == 'left':
            result[:, sep_idx + 1:] = target
        else:
            result[:, :sep_idx] = target
    else:
        if legend_side == 'top':
            result[sep_idx + 1:, :] = target
        else:
            result[:sep_idx, :] = target

    return result


def _exec_registration_transfer(inp, step):
    """Move modules into frame openings based on shape fit.

    Uses the same _find_frame_openings helper as the derive path
    (interior-only openings, nearest-compatible matching).
    """
    from aria.guided.perceive import perceive
    from aria.search.derive import _find_frame_openings

    facts = perceive(inp)
    bg = facts.bg
    result = inp.copy()

    frames = [o for o in facts.objects if not o.is_rectangular and o.size >= 8]
    if not frames:
        return inp

    frame_openings = _find_frame_openings(inp, frames, bg)
    if not frame_openings:
        return inp

    # Find candidate modules: small objects not part of any frame
    frame_oids = {f.oid for f in frames}
    max_opening_cells = max(nc for _, _, _, _, _, nc, _ in frame_openings)
    modules = [o for o in facts.objects if o.oid not in frame_oids and o.size <= max_opening_cells]

    # Match each module to closest shape-compatible opening
    used_openings = set()
    for m in sorted(modules, key=lambda o: -o.size):  # largest modules first
        best_oi = None
        best_dist = float('inf')
        for oi, (frame, or0, oc0, oh, ow, nc, omask) in enumerate(frame_openings):
            if oi in used_openings:
                continue
            # Check: module fits the opening (same bbox AND same cell count)
            if m.height == oh and m.width == ow and m.size == nc:
                # Also check mask compatibility (module fills exactly the opening cells)
                if np.array_equal(m.mask, omask):
                    dist = abs(m.center_row - (or0 + oh / 2)) + abs(m.center_col - (oc0 + ow / 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_oi = oi

        if best_oi is not None:
            used_openings.add(best_oi)
            _, or0, oc0, oh, ow, _, _ = frame_openings[best_oi]
            # Erase module from old position
            for r in range(m.height):
                for c in range(m.width):
                    if m.mask[r, c]:
                        result[m.row + r, m.col + c] = bg
            # Place at opening
            for r in range(m.height):
                for c in range(m.width):
                    if m.mask[r, c]:
                        nr, nc = or0 + r, oc0 + c
                        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                            result[nr, nc] = m.color

    return result


def _exec_registration_anchor_transfer(inp, step):
    """Move anchored modules to target sites using nearest-anchor assignment."""
    from aria.guided.perceive import perceive
    from aria.search.registration import (
        base_registration_patch,
        cluster_movable_modules,
        extract_anchored_shapes,
        module_anchor_patch,
        module_anchor_origin,
        module_anchor_centroid,
        overlay_registration_candidates,
    )
    from scipy.optimize import linear_sum_assignment

    params = step.params or {}
    shape_color = params.get('shape_color')
    anchor_color = params.get('anchor_color')

    if shape_color is None or anchor_color is None:
        return inp

    facts = perceive(inp)
    if shape_color == facts.bg or anchor_color == facts.bg:
        return inp
    shapes = extract_anchored_shapes(
        inp, shape_color=int(shape_color), anchor_color=int(anchor_color),
    )
    base_idx, modules = cluster_movable_modules(shapes)
    if base_idx is None or not modules:
        return inp

    base_patch, target_sites = base_registration_patch(
        shapes[base_idx], shape_color=int(shape_color),
    )
    if not shapes[base_idx].anchors_global:
        return inp

    target_sites_global = [
        (shapes[base_idx].row + r, shapes[base_idx].col + c)
        for r, c in target_sites
    ]
    if not target_sites_global or len(modules) > len(target_sites_global):
        return inp

    # Assign modules to target sites by centroid distance
    cost = np.zeros((len(modules), len(target_sites_global)), dtype=float)
    for mi, module in enumerate(modules):
        mr, mc = module_anchor_centroid(shapes, module)
        for ti, (tr, tc) in enumerate(target_sites_global):
            cost[mi, ti] = abs(mr - tr) + abs(mc - tc)
    rows, cols = linear_sum_assignment(cost)

    result = inp.copy()
    used_openings = set()
    for mi, ti in zip(rows, cols):
        module = modules[mi]
        target_site = target_sites[ti]
        if ti in used_openings:
            continue
        used_openings.add(ti)

        module_patch, module_mask, source_anchors = module_anchor_patch(
            inp, shapes, module,
            shape_color=int(shape_color), anchor_color=int(anchor_color),
        )
        module_anchors_global = tuple(
            sorted({(ar, ac) for i in module.component_indices
                    for ar, ac in shapes[i].anchors_global})
        )
        if not module_anchors_global:
            continue
        # Pick source anchor closest to target site
        target_global = target_sites_global[ti]
        best_anchor = None
        best_dist = float('inf')
        for ma in module_anchors_global:
            dist = abs(ma[0] - target_global[0]) + abs(ma[1] - target_global[1])
            if dist < best_dist:
                best_dist = dist
                best_anchor = ma
        if best_anchor is None:
            continue

        r0, c0 = module_anchor_origin(shapes, module)
        chosen_source_anchor = (best_anchor[0] - r0, best_anchor[1] - c0)

        dr = target_global[0] - best_anchor[0]
        dc = target_global[1] - best_anchor[1]

        # Erase module from old position
        for obj_idx in module.component_indices:
            obj = shapes[obj_idx]
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.patch[r, c] == obj.color:
                        result[obj.row + r, obj.col + c] = facts.bg
        # Place module at new position
        for obj_idx in module.component_indices:
            obj = shapes[obj_idx]
            for r in range(obj.height):
                for c in range(obj.width):
                    if obj.patch[r, c] == obj.color:
                        nr = obj.row + r + dr
                        nc = obj.col + c + dc
                        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                            result[nr, nc] = obj.color

    return result


def _exec_marker_stamp(inp, step):
    """Execute marker stamp: apply learned templates at marker positions."""
    from aria.guided.perceive import perceive
    facts = perceive(inp)
    bg = step.params.get('bg', facts.bg)
    templates = step.params.get('templates', {})
    from aria.search.derive import _apply_marker_stamp
    return _apply_marker_stamp(inp, facts, templates, bg)


def _exec_panel_boolean(inp, step):
    """Execute panel boolean algebra using panel_boolean_combine."""
    from aria.search.panel_ops import panel_boolean_combine
    p = step.params
    op_name = p.get('op', 'xor')
    color = p.get('color', 0)
    sep_rows = p.get('sep_rows')
    sep_cols = p.get('sep_cols')
    result = panel_boolean_combine(inp, op_name, color, sep_rows=sep_rows, sep_cols=sep_cols)
    return result if result is not None else inp


def _step_to_ast(step: SearchStep) -> ASTNode:
    """Convert a SearchStep to an ASTNode."""
    action = step.action
    p = step.params
    # Rule-based selectors stay as StepSelect objects (resolved at the
    # search execution level); predicate-based selectors lower to Pred lists.
    if step.select and step.select.role == 'by_rule':
        sel_preds = step.select
    else:
        sel_preds = step.select.to_predicates() if step.select else None

    # --- Grid-level transforms ---
    _xform_map = {
        'flip_h': Op.FLIP_H, 'flip_v': Op.FLIP_V, 'flip_hv': Op.FLIP_HV,
        'rot90': Op.ROT90, 'rot180': Op.ROT180, 'transpose': Op.TRANSPOSE,
    }
    if action in _xform_map:
        return ASTNode(_xform_map[action], [ASTNode(Op.INPUT)])

    # --- Extraction ---
    if action == 'crop_bbox':
        obj_node = ASTNode(Op.SELECT, [ASTNode(Op.PERCEIVE, [ASTNode(Op.INPUT)])],
                           param=sel_preds)
        return ASTNode(Op.CROP_BBOX, [ASTNode(Op.INPUT), obj_node])

    if action == 'crop_interior':
        obj_node = ASTNode(Op.SELECT, [ASTNode(Op.PERCEIVE, [ASTNode(Op.INPUT)])],
                           param=sel_preds)
        return ASTNode(Op.CROP_INTERIOR, [ASTNode(Op.INPUT), obj_node])

    if action == 'split_combine_render':
        combine_op = p.get('op', 'xor')
        color = p.get('color', 0)
        return ASTNode(Op.RENDER, [
            ASTNode(Op.COMBINE, [ASTNode(Op.SPLIT, [ASTNode(Op.INPUT)])],
                    param=combine_op),
            ASTNode(Op.CONST_COLOR, param=color),
        ])

    # --- Grid constructors ---
    if action == 'tile':
        return ASTNode(Op.TILE, [ASTNode(Op.INPUT)],
                       param=(p.get('rows', 1), p.get('cols', 1), p.get('transforms', {})))

    if action == 'template_broadcast':
        return ASTNode(Op.TEMPLATE_BROADCAST, [ASTNode(Op.INPUT)], param=p)

    if action == 'periodic_extend':
        return ASTNode(Op.PERIODIC_EXTEND, [ASTNode(Op.INPUT)],
                       param=p.get('color_map'))

    if action == 'repair_frames':
        return ASTNode(Op.REPAIR_FRAMES, [ASTNode(Op.INPUT)])

    if action == 'mirror':
        return ASTNode(Op.MIRROR, [ASTNode(Op.INPUT)], param=p.get('axis', 'col'))

    # --- Object-level actions ---
    if action == 'recolor':
        return ASTNode(Op.RECOLOR, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('color', 0)))

    if action == 'remove':
        bg_override = p.get('bg')
        return ASTNode(Op.REMOVE, [ASTNode(Op.INPUT)],
                       param=(sel_preds, bg_override))

    if action == 'move':
        return ASTNode(Op.MOVE, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('dr', 0), p.get('dc', 0)))

    if action == 'gravity':
        return ASTNode(Op.GRAVITY, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('direction', 'down')))

    if action == 'slide':
        return ASTNode(Op.SLIDE, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('direction', 'down')))

    if action == 'stamp':
        return ASTNode(Op.STAMP, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('dr', 0), p.get('dc', 0)))

    if action == 'transform':
        return ASTNode(Op.TRANSFORM_OBJ, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('xform', 'flip_h')))

    if action == 'fill_interior':
        return ASTNode(Op.FILL_INTERIOR, [ASTNode(Op.INPUT)],
                       param=(sel_preds, p.get('color', 0)))

    if action == 'fill_enclosed':
        return ASTNode(Op.FILL_ENCLOSED, [ASTNode(Op.INPUT)],
                       param=p.get('color', 0))

    if action == 'gravity_nearest':
        return ASTNode(Op.COMPOSE, [], param=f'gravity_nearest:{p.get("axis", "vertical")}')

    # Panel operations — canonical AST ops
    if action == 'panel_odd_select':
        return ASTNode(Op.PANEL_ODD_SELECT, [ASTNode(Op.INPUT)])
    if action == 'panel_majority_select':
        return ASTNode(Op.PANEL_MAJORITY_SELECT, [ASTNode(Op.INPUT)])
    if action == 'panel_repair':
        return ASTNode(Op.PANEL_REPAIR, [ASTNode(Op.INPUT)])
    if action == 'panel_boolean':
        return ASTNode(Op.PANEL_BOOLEAN, [ASTNode(Op.INPUT)],
                       param=(p.get('op', 'xor'), p.get('color', 0),
                              p.get('sep_rows'), p.get('sep_cols')))
    if action == 'symmetry_repair':
        return ASTNode(Op.SYMMETRY_REPAIR, [ASTNode(Op.INPUT)], param=p.get('damage_color', 0))
    if action == 'object_repack':
        return ASTNode(Op.OBJECT_REPACK, [ASTNode(Op.INPUT)], param=p)

    if action == 'quadrant_template_decode':
        return ASTNode(Op.QUADRANT_TEMPLATE_DECODE, [ASTNode(Op.INPUT)], param=p)
    if action == 'frame_bbox_pack':
        return ASTNode(Op.FRAME_BBOX_PACK, [ASTNode(Op.INPUT)], param=p)
    if action == 'cross_stencil_recolor':
        return ASTNode(Op.CROSS_STENCIL_RECOLOR, [ASTNode(Op.INPUT)], param=p)
    if action == 'legend_frame_fill':
        return ASTNode(Op.LEGEND_FRAME_FILL, [ASTNode(Op.INPUT)], param=p)
    if action == 'anomaly_halo':
        return ASTNode(Op.ANOMALY_HALO, [ASTNode(Op.INPUT)], param=p)
    if action == 'object_highlight':
        return ASTNode(Op.OBJECT_HIGHLIGHT, [ASTNode(Op.INPUT)], param=p)
    if action == 'legend_chain_connect':
        return ASTNode(Op.LEGEND_CHAIN_CONNECT, [ASTNode(Op.INPUT)], param=p)
    if action == 'diagonal_collision_trace':
        return ASTNode(Op.DIAGONAL_COLLISION_TRACE, [ASTNode(Op.INPUT)], param=p)
    if action == 'masked_patch_transfer':
        return ASTNode(Op.MASKED_PATCH_TRANSFER, [ASTNode(Op.INPUT)], param=p)
    if action == 'separator_motif_broadcast':
        return ASTNode(Op.SEPARATOR_MOTIF_BROADCAST, [ASTNode(Op.INPUT)], param=p)
    if action == 'line_arith_broadcast':
        return ASTNode(Op.LINE_ARITH_BROADCAST, [ASTNode(Op.INPUT)], param=p)
    if action == 'barrier_port_transfer':
        return ASTNode(Op.BARRIER_PORT_TRANSFER, [ASTNode(Op.INPUT)], param=p)
    if action == 'cavity_transfer':
        return ASTNode(Op.CAVITY_TRANSFER, [ASTNode(Op.INPUT)], param=p)

    if action == 'recolor_map':
        return ASTNode(Op.RECOLOR_MAP, [ASTNode(Op.INPUT)], param=p.get('color_map', {}))

    if action in ('crop_nonbg', 'crop_object', 'crop_fixed', 'color_stencil',
                  'registration_transfer', 'grid_fill_between',
                  'grid_slot_transfer', 'grid_conditional_transfer',
                  'object_grid_pack', 'panel_legend_map',
                  'correspondence_transfer'):
        # These ops don't lower to AST — executed directly in SearchProgram.execute
        return ASTNode(Op.INPUT)

    raise ValueError(f"Unknown search action for AST lowering: {action}")


def _exec_recolor_map(grid, params):
    """Apply a global color substitution map to every cell."""
    color_map = params.get('color_map', {})
    if not color_map:
        return grid
    result = grid.copy()
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            v = int(result[r, c])
            if v in color_map:
                result[r, c] = color_map[v]
    return result


def _exec_crop_nonbg(grid):
    """Crop grid to tightest bounding box around non-bg cells."""
    from aria.guided.perceive import perceive
    facts = perceive(grid)
    nonbg = np.argwhere(grid != facts.bg)
    if len(nonbg) == 0:
        return grid
    r0, c0 = nonbg.min(axis=0)
    r1, c1 = nonbg.max(axis=0)
    return grid[r0:r1+1, c0:c1+1]


def _exec_crop_object(grid, params, select=None):
    """Crop grid to bounding box of an object matching a predicate or selector."""
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    # Rule-based selector path (from step.select)
    if select is not None and params.get('predicate') == 'by_rule':
        facts = perceive(grid)
        selected = select.select_objects(facts)
        if len(selected) == 1:
            obj = selected[0]
            return grid[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
        return grid

    pred_name = params.get('predicate', 'largest')
    _name_to_pred = {
        'largest': Pred.IS_LARGEST, 'smallest': Pred.IS_SMALLEST,
        'unique_color': Pred.UNIQUE_COLOR,
        'topmost': Pred.IS_TOPMOST, 'bottommost': Pred.IS_BOTTOMMOST,
        'leftmost': Pred.IS_LEFTMOST, 'rightmost': Pred.IS_RIGHTMOST,
        'interior': Pred.NOT_TOUCHES_BORDER,
        'touches_border': Pred.TOUCHES_BORDER,
    }
    pred = _name_to_pred.get(pred_name)
    if pred is None:
        # Try color_N format
        if pred_name.startswith('color_'):
            try:
                color = int(pred_name.split('_')[1])
                pred = Pred.COLOR_EQ
                facts = perceive(grid)
                selected = prim_select(facts, [Predicate(pred, color)])
                if len(selected) != 1:
                    return grid
                obj = selected[0]
                return grid[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
            except (ValueError, IndexError):
                pass
        return grid
    facts = perceive(grid)
    selected = prim_select(facts, [Predicate(pred)])
    if len(selected) != 1:
        return grid
    obj = selected[0]
    return grid[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]


def _exec_quadrant_template_decode(inp, step):
    """Execute quadrant template decode at search time."""
    from aria.guided.perceive import perceive
    from aria.search.decode import (
        _extract_quadrant_pattern, _apply_quadrant_template,
    )

    facts = perceive(inp)
    bg = facts.bg
    tmpl_idx = step.params.get('template_quadrant', 0)
    seed_color = step.params.get('seed_color')

    row_seps = sorted([s for s in facts.separators if s.axis == 'row'], key=lambda s: s.index)
    col_seps = sorted([s for s in facts.separators if s.axis == 'col'], key=lambda s: s.index)
    if len(row_seps) != 1 or len(col_seps) != 1:
        return inp

    rs, cs = row_seps[0].index, col_seps[0].index
    h, w = inp.shape

    quads_rc = [(0, rs, 0, cs), (0, rs, cs + 1, w), (rs + 1, h, 0, cs), (rs + 1, h, cs + 1, w)]
    quads = [inp[r0:r1, c0:c1] for r0, r1, c0, c1 in quads_rc]

    tmpl = quads[tmpl_idx]
    tmpl_pattern = _extract_quadrant_pattern(tmpl, bg, seed_color)
    if tmpl_pattern is None:
        return inp

    pat_colors, pat_bbox, pat_relative = tmpl_pattern
    central_color = seed_color if seed_color is not None else pat_colors.get('center')
    if central_color is None:
        return inp

    result = inp.copy()
    tmpl_row = 0 if tmpl_idx < 2 else 1
    tmpl_col = tmpl_idx % 2

    for qi in range(4):
        if qi == tmpl_idx:
            continue
        q = quads[qi]
        seed_cells = [(r, c) for r in range(q.shape[0]) for c in range(q.shape[1])
                       if q[r, c] == central_color]
        if not seed_cells:
            continue

        qi_row = 0 if qi < 2 else 1
        qi_col = qi % 2
        flip_v = (qi_row != tmpl_row)
        flip_h = (qi_col != tmpl_col)

        applied = _apply_quadrant_template(q, pat_relative, seed_cells,
                                            central_color, bg, flip_h, flip_v)
        if applied is not None:
            r0, r1, c0, c1 = quads_rc[qi]
            result[r0:r1, c0:c1] = applied

    return result


def _exec_object_highlight_full(inp, params):
    """Execute object highlight from panel facts plus induced boolean rules."""
    from aria.guided.perceive import perceive
    from aria.search.motif import extract_motifs, extract_panel_facts
    from aria.search.rules import eval_rule

    ground = params.get('ground', 8)
    highlight = params.get('highlight', 3)
    shape_rule = params.get('shape_rule')
    p0_rule = params.get('p0_rule')

    facts = perceive(inp)
    rs = sorted(set(s.index for s in facts.separators if s.axis == 'row'))
    rb = [0] + rs + [inp.shape[0]]
    panels = [(rb[i], rb[i+1]) for i in range(len(rb)-1) if rb[i+1]-rb[i] >= 3]
    if len(panels) < 2:
        return inp

    h, w = inp.shape
    r0_0, r1_0 = panels[0]
    p0_motifs = extract_motifs(inp[r0_0:r1_0, :], bg=0, ground=ground, min_cells=2)
    p0_shapes = frozenset(m.motif for m in p0_motifs)
    p0_colors = frozenset(m.color for m in p0_motifs)
    p0_n = len(p0_motifs)
    p0_order = {m.motif: i for i, m in enumerate(sorted(p0_motifs, key=lambda m: m.col))}

    result = inp.copy()
    highlighted = {}  # pi → (c_lo, c_hi) column range

    # Track matched-shape -> workspace-slot mapping for band-fill propagation.
    shape_slot_map = {i: set() for i in range(len(p0_order))}
    panel_slots = {}
    panel_matches = {}
    any_full = False

    def _slot_id(anchor_col: int) -> int | None:
        c = anchor_col + 1
        if 1 <= c <= 5:
            return 0
        if 6 <= c <= 10:
            return 1
        if 11 <= c <= 15:
            return 2
        if 16 <= c <= 20:
            return 3
        return None

    # --- Pass 1: compute facts and highlight shape-matched panels ---
    ws_facts = []
    for pi, (r0, r1) in enumerate(panels[1:], 1):
        pf = extract_panel_facts(inp[r0:r1, :], pi, p0_shapes, p0_colors, p0_n,
                                  bg=0, ground=ground)
        ws_facts.append((pi, r0, r1, pf))
        motifs = extract_motifs(inp[r0:r1, :], bg=0, ground=ground, min_cells=2)
        panel_slots[pi] = {sid for sid in (_slot_id(m.col) for m in motifs) if sid is not None}
        should_highlight = (
            eval_rule(shape_rule, pf.to_rule_dict())
            if shape_rule is not None else pf.any_match
        )

        if pf.any_match and should_highlight:
            matched = [m for m in motifs if m.motif in p0_shapes]
            panel_matches[pi] = matched
            for m in matched:
                sid = _slot_id(m.col)
                if sid is not None:
                    shape_slot_map[p0_order[m.motif]].add(sid)

            if pf.n_unmatched == 0:
                # All match → full panel interior
                any_full = True
                for r in range(r0, r1):
                    for c in range(1, w - 1):
                        if result[r, c] == ground:
                            result[r, c] = highlight
                highlighted[pi] = (1, w - 1)  # exclusive end (w-1 is last col, range goes to w-1)
            else:
                # Zone-based: each motif owns a column zone (midpoints between neighbors)
                sorted_motifs = sorted(motifs, key=lambda m: m.col)
                zones = {}
                for mi, m in enumerate(sorted_motifs):
                    mc = m.col + m.motif.bbox_w // 2
                    if mi == 0:
                        z_lo = 1
                    else:
                        prev_m = sorted_motifs[mi - 1]
                        prev_mc = prev_m.col + prev_m.motif.bbox_w // 2
                        z_lo = (prev_mc + mc + 1) // 2
                    if mi == len(sorted_motifs) - 1:
                        z_hi = w - 2
                    else:
                        next_m = sorted_motifs[mi + 1]
                        next_mc = next_m.col + next_m.motif.bbox_w // 2
                        z_hi = (mc + next_mc - 1) // 2
                    zones[m.motif] = (max(1, z_lo), min(w - 2, z_hi))

                # Fill zones of matched motifs (keep separate zones)
                panel_zones = []
                for m in matched:
                    z_lo, z_hi = zones.get(m.motif, (1, w - 2))
                    panel_zones.append((z_lo, z_hi + 1))  # exclusive end
                    for r in range(r0, r1):
                        for c in range(z_lo, z_hi + 1):
                            if result[r, c] == ground:
                                result[r, c] = highlight

                if panel_zones:
                    highlighted[pi] = panel_zones  # list of (start, end_exclusive)
        else:
            panel_matches[pi] = []

    # --- Pass 2: band-fill panels inherit a propagated fallback slot ---
    mapped_keys = [k for k, slots in shape_slot_map.items() if slots]
    monotonic_mapping = (
        bool(mapped_keys)
        and all(
            min(shape_slot_map[mapped_keys[i]]) <= min(shape_slot_map[mapped_keys[i + 1]])
            for i in range(len(mapped_keys) - 1)
        )
    )

    if monotonic_mapping and not any_full:
        common_slots = None
        for pi, _, _, pf in ws_facts:
            if not pf.any_match:
                continue
            slots = {sid for sid in (_slot_id(m.col) for m in panel_matches[pi]) if sid is not None}
            common_slots = slots if common_slots is None else (common_slots & slots)

        if common_slots:
            target_slot = min(common_slots)
        else:
            target_slot = min(min(slots) for slots in shape_slot_map.values() if slots)

        # Workspace panels use four canonical 5-column slots inside cols 1..20.
        c_lo = 1 + 5 * target_slot
        c_hi = min(c_lo + 5, w - 1)  # exclusive end
        for pi, r0, r1, pf in ws_facts:
            if pf.any_match or target_slot in panel_slots.get(pi, set()):
                continue
            highlighted[pi] = (c_lo, c_hi)
            for r in range(r0, r1):
                for c in range(c_lo, c_hi):
                    if result[r, c] == ground:
                        result[r, c] = highlight

    # --- Rule 3: P0 highlight (full width) ---
    aggregate_facts = {
        'any_full_match': any_full,
        'all_ws_highlight': bool(ws_facts) and all(pi in highlighted for pi, _, _, _ in ws_facts),
    }
    p0_hl = eval_rule(p0_rule, aggregate_facts) if p0_rule is not None else aggregate_facts['any_full_match']

    if p0_hl:
        highlighted[0] = (0, w)
        r0, r1 = panels[0]
        for r in range(r0, r1):
            for c in range(w):
                if result[r, c] == ground:
                    result[r, c] = highlight

    # --- Rule 4: Separator propagation ---
    content_rows = set(r for r0, r1 in panels for r in range(r0, r1))
    for pi, ranges in sorted(highlighted.items()):
        _, r1_p = panels[pi]
        # ranges is either (start, end_exclusive) or list of (start, end_exclusive)
        if isinstance(ranges, tuple):
            ranges = [ranges]
        for r in range(r1_p, h):
            if r in content_rows:
                break
            if int(inp[r, 0]) == ground:
                for c_lo, c_hi in ranges:
                    for c in range(c_lo, c_hi):
                        if result[r, c] == ground:
                            result[r, c] = highlight
                break

    # --- Rule 5: Bottom band (last 2 rows, full width) ---
    bb_color = highlight if p0_hl else params.get('bottom_alt_color', highlight)
    for r in range(h - 2, h):
        for c in range(w):
            if result[r, c] == ground:
                result[r, c] = bb_color

    return result


def _exec_legend_frame_fill(inp, step):
    """Execute legend-driven frame fill at search time."""
    from aria.search.executor import _exec_legend_frame_fill
    return _exec_legend_frame_fill(inp, step.params or {})


def _exec_frame_bbox_pack(inp, step):
    """Execute frame-bbox pack at search time."""
    return _do_frame_bbox_pack(inp, step.params or {})


def _do_frame_bbox_pack(inp, params):
    """Shared frame-bbox pack logic for both SearchProgram and AST execution."""
    from aria.search.frames import extract_rect_items, render_rect_family_side

    bg = int(np.bincount(inp.ravel()).argmax())
    mode = params.get('mode')
    ordering = params.get('ordering', 'row')
    bh = params.get('block_h')
    bw = params.get('block_w')
    nc = params.get('grid_cols')  # number of columns in output grid

    if mode == 'family_side_lanes':
        items = extract_rect_items(inp, bg=bg, min_span=4)
        if not items:
            return inp
        family_colors = sorted({item.color for item in items})
        if len(family_colors) != 2:
            return inp
        if params.get('family_order', 'desc') == 'desc':
            ordered_colors = sorted(family_colors, reverse=True)
        else:
            ordered_colors = sorted(family_colors)
        result = np.full(inp.shape, bg, dtype=inp.dtype)
        for side, color in zip(('left', 'right'), ordered_colors):
            group = [item for item in items if item.color == color]
            family_canvas = render_rect_family_side(group, shape=inp.shape, bg=bg, side=side)
            mask = family_canvas != bg
            result[mask] = family_canvas[mask]
        return result

    if not bh or not bw or not nc:
        return inp

    frame_infos = [
        {
            'bbox': item.patch,
            'color': item.color,
            'row': item.row,
            'col': item.col,
            'interior_bg': item.interior_bg,
            'kind': item.kind,
        }
        for item in extract_rect_items(inp, bg=bg, min_span=4)
        if item.patch.shape == (bh, bw)
    ]

    if not frame_infos:
        return inp

    if ordering == 'row':
        ordered = sorted(frame_infos, key=lambda f: (f['row'], f['col']))
    elif ordering == 'col':
        ordered = sorted(frame_infos, key=lambda f: (f['col'], f['row']))
    elif ordering == 'color':
        ordered = sorted(frame_infos, key=lambda f: f['color'])
    elif ordering == 'group_cols':
        empty = sorted([f for f in frame_infos if f['interior_bg']], key=lambda f: (f['row'], f['col']))
        filled = sorted([f for f in frame_infos if not f['interior_bg']], key=lambda f: (f['row'], f['col']))
        ordered = []
        for i in range(max(len(empty), len(filled))):
            ordered.append(empty[i] if i < len(empty) else None)
            ordered.append(filled[i] if i < len(filled) else None)
    else:
        ordered = frame_infos

    # Compute grid rows from frame count and column count
    n_slots = len(ordered)
    nr = (n_slots + nc - 1) // nc  # ceil division
    oh, ow = nr * bh, nc * bw
    result = np.full((oh, ow), bg, dtype=inp.dtype)
    for idx in range(nr * nc):
        rb = idx // nc
        cb = idx % nc
        if idx < len(ordered) and ordered[idx] is not None:
            result[rb * bh:(rb + 1) * bh, cb * bw:(cb + 1) * bw] = ordered[idx]['bbox']

    return result
