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
            d['params'] = {str(k): v for k, v in self.params.items()}
        if self.from_step is not None:
            d['from_step'] = self.from_step
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StepSelect:
        return cls(role=d['role'], params=d.get('params', {}),
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
        return []


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
    from aria.guided.dsl import prim_select

    facts = perceive(inp)
    bg = facts.bg
    axis = step.params.get('axis', 'vertical')
    sel_preds = step.select.to_predicates() if step.select else None

    targets = prim_select(facts, sel_preds) if sel_preds else facts.objects
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
    from aria.guided.dsl import prim_select

    facts = perceive(inp)
    bg = facts.bg
    direction = step.params.get('direction', 'down')
    sel_preds = step.select.to_predicates() if step.select else None

    targets = prim_select(facts, sel_preds) if sel_preds else facts.objects
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
    """Execute panel boolean algebra."""
    from aria.search.panels import extract_panels
    ps = extract_panels(inp)
    if ps is None:
        return inp
    p = step.params
    i, j = p.get('panel_a', 0), p.get('panel_b', 1)
    if i >= len(ps.panels) or j >= len(ps.panels):
        return inp
    a, b = ps.panels[i].grid, ps.panels[j].grid
    if a.shape != b.shape:
        return inp
    bg = ps.bg
    op_name = p.get('op', 'xor')
    color = p.get('color', 0)
    _OPS = {
        'and': lambda x, y: x & y, 'or': lambda x, y: x | y,
        'xor': lambda x, y: x ^ y, 'nor': lambda x, y: ~x & ~y,
        'nand': lambda x, y: ~(x & y), 'a-b': lambda x, y: x & ~y,
        'b-a': lambda x, y: y & ~x,
    }
    op_fn = _OPS.get(op_name)
    if op_fn is None:
        return inp
    mask = op_fn((a != bg), (b != bg))
    canvas = np.full(a.shape, bg, dtype=np.uint8)
    canvas[mask] = color
    return canvas


def _step_to_ast(step: SearchStep) -> ASTNode:
    """Convert a SearchStep to an ASTNode."""
    action = step.action
    p = step.params
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
        return ASTNode(Op.REMOVE, [ASTNode(Op.INPUT)], param=sel_preds)

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
    if action == 'object_summary_column':
        return ASTNode(Op.OBJECT_SUMMARY_COLUMN, [ASTNode(Op.INPUT)])

    # Fallback
    return ASTNode(Op.HOLE, param=f'unknown:{action}')
