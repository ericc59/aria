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

    if action in ('crop_nonbg', 'crop_object', 'crop_fixed', 'color_stencil'):
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
