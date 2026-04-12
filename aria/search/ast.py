"""Explicit AST-based program representation for ARC solvers.

The IR for the search engine. Every program is an inspectable, serializable,
learnable tree of typed operations.

This module contains only the data structures — no execution logic.
Execution is in executor.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto

import numpy as np

from aria.types import Grid


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class Op(Enum):
    """All operations in the DSL, as an explicit enum."""
    # Leaves (no children)
    INPUT = auto()          # the input grid
    CONST_COLOR = auto()    # param: int (color value 0-9)
    CONST_INT = auto()      # param: int

    # Perception
    PERCEIVE = auto()       # Grid → Facts
    SELECT = auto()         # Facts × Selector → Object  (param: list[Predicate])
    SELECT_IDX = auto()     # Facts × int → Object       (param: int)

    # Extraction
    CROP_BBOX = auto()      # Grid × Object → Region
    CROP_INTERIOR = auto()  # Grid × Object → Region
    SPLIT = auto()          # Grid → Pair (at separator)
    FIRST = auto()          # Pair → Region
    SECOND = auto()         # Pair → Region

    # Region ops
    COMBINE = auto()        # Pair → Mask  (param: 'and'|'or'|'xor'|'diff'|'rdiff')
    RENDER = auto()         # Mask × Color → Grid

    # Transforms (Grid/Region → Grid/Region)
    FLIP_H = auto()         # horizontal flip (reverse columns)
    FLIP_V = auto()         # vertical flip (reverse rows)
    FLIP_HV = auto()        # both
    ROT90 = auto()
    ROT180 = auto()
    TRANSPOSE = auto()

    # Trace/geometry
    TRACE = auto()          # param: TraceSpec (route, write, stop)

    # Grid constructors
    TILE = auto()           # param: (rows, cols, transforms_dict)
    PERIODIC_EXTEND = auto()  # param: color_map or None
    REPAIR_FRAMES = auto()  # Grid → Grid

    # Panel operations
    PANEL_ODD_SELECT = auto()   # Grid → Grid (select odd-one-out from each panel band)
    PANEL_MAJORITY_SELECT = auto()  # Grid → Grid
    PANEL_REPAIR = auto()       # Grid → Grid (repair periodic patterns in each panel)
    PANEL_BOOLEAN = auto()      # Grid → Grid (boolean combine aligned panels) param: (op, color)

    # Repair
    SYMMETRY_REPAIR = auto()        # Grid → Grid (repair damage color using grid symmetry) param: damage_color

    # Object repacking / structured output
    OBJECT_REPACK = auto()          # Grid → Grid  param: {ordering, layout, payload}

    # Region decode
    QUADRANT_TEMPLATE_DECODE = auto()  # Grid → Grid  param: {template_quadrant, seed_color}
    FRAME_BBOX_PACK = auto()           # Grid → Grid  param: {ordering, block_h, block_w, grid_rows, grid_cols}
    CROSS_STENCIL_RECOLOR = auto()     # Grid → Grid  param: {old_color, new_color}
    LEGEND_FRAME_FILL = auto()         # Grid → Grid  param: {color_map: dict}
    ANOMALY_HALO = auto()              # Grid → Grid  param: {c1, c2, halo_color}
    OBJECT_HIGHLIGHT = auto()          # Grid → Grid  param: {ground, highlight, bottom_alt_color?}
    LEGEND_CHAIN_CONNECT = auto()      # Grid → Grid  param: {control_side?}
    DIAGONAL_COLLISION_TRACE = auto()  # Grid → Grid  param: {point_dir?: 'up_right', include_direct_hit?: bool}
    MASKED_PATCH_TRANSFER = auto()     # Grid → Grid  param: {mask_color, ring?}
    STACKED_GLYPH_TRACE = auto()       # legacy/noncanonical compatibility op
    CORNER_DIAG_FILL = auto()          # legacy/noncanonical compatibility op
    SEPARATOR_MOTIF_BROADCAST = auto()  # Grid → Grid  param: {axis: 'auto'|'row'|'col'}
    LINE_ARITH_BROADCAST = auto()      # Grid → Grid  param: {axis: 'auto'|'row'|'col'}
    BARRIER_PORT_TRANSFER = auto()     # Grid → Grid  param: {mode: 'auto'}
    CAVITY_TRANSFER = auto()           # Grid → Grid  param: {mode: 'auto'}
    RECOLOR_MAP = auto()               # Grid → Grid  param: {old_color: new_color}
    TEMPLATE_BROADCAST = auto()      # Grid → Grid  param: {bg?: int}  mask-driven template placement

    # Object-level actions
    RECOLOR = auto()        # param: (selector_preds, new_color)
    REMOVE = auto()         # param: selector_preds
    MOVE = auto()           # param: (selector_preds, dr, dc)
    GRAVITY = auto()        # param: (selector_preds, direction)
    SLIDE = auto()          # param: (selector_preds, direction)
    STAMP = auto()          # param: (selector_preds, dr, dc)
    TRANSFORM_OBJ = auto()  # param: (selector_preds, xform_name)
    FILL_INTERIOR = auto()  # param: (selector_preds, color)
    FILL_ENCLOSED = auto()  # param: color or 'frame'
    RAY_PROJECT = auto()    # param: {directions, ray_color, stop_on, selector}
    FLOOD_FILL_ADJACENT = auto()  # param: {seed_color, fill_color, connectivity}
    DILATE = auto()         # param: {selector, radius, metric, fill_color}
    ERODE = auto()          # param: {selector, radius, metric}
    DOWNSCALE = auto()      # param: {block_h, block_w, rule}
    ITERATE_FIXED = auto()  # param: {body_action, body_params, body_select, max_steps}

    # Composition
    COMPOSE = auto()        # children executed sequentially on accumulating grid
    FOR_EACH = auto()       # param: (selector, body_spec)
    IF_ELSE = auto()        # param: (selector, then_spec, else_spec)

    # Mirror
    MIRROR = auto()         # param: axis mode ('col', 'col_rtl', 'row', 'row_btt')

    # Hole (for search over partial programs)
    HOLE = auto()           # param: type_tag string


# ---------------------------------------------------------------------------
# AST Node
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """A node in the program AST.

    Leaf nodes have no children. Internal nodes compose their children.
    param holds op-specific constants (colors, directions, selectors, etc.)
    """
    op: Op
    children: list['ASTNode'] = field(default_factory=list)
    param: Any = None

    def __repr__(self):
        if not self.children:
            if self.param is not None:
                return f"{self.op.name}({self.param})"
            return self.op.name
        args = ', '.join(repr(c) for c in self.children)
        if self.param is not None:
            return f"{self.op.name}[{self.param}]({args})"
        return f"{self.op.name}({args})"

    def size(self) -> int:
        """Number of nodes (MDL proxy)."""
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def has_holes(self) -> bool:
        if self.op == Op.HOLE:
            return True
        return any(c.has_holes() for c in self.children)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        d = {'op': self.op.name}
        if self.param is not None:
            d['param'] = _serialize_param(self.param)
        if self.children:
            d['children'] = [c.to_dict() for c in self.children]
        return d

    @staticmethod
    def from_dict(d: dict) -> 'ASTNode':
        """Deserialize from dict."""
        op = Op[d['op']]
        children = [ASTNode.from_dict(c) for c in d.get('children', [])]
        param = d.get('param')
        return ASTNode(op=op, children=children, param=param)


def _serialize_param(param):
    """Best-effort JSON-safe serialization of param values."""
    if isinstance(param, (int, float, str, bool, type(None))):
        return param
    if isinstance(param, np.ndarray):
        return param.tolist()
    if isinstance(param, (list, tuple)):
        return [_serialize_param(x) for x in param]
    if isinstance(param, dict):
        return {str(k): _serialize_param(v) for k, v in param.items()}
    # Predicate objects etc. — fall back to repr
    return repr(param)


# ---------------------------------------------------------------------------
# AST Program wrapper
# ---------------------------------------------------------------------------

class ASTProgram:
    """A program backed by an explicit AST.

    Drop-in compatible with aria.guided.dsl.Program interface.
    """
    def __init__(self, ast: ASTNode, description: str = '',
                 search_program=None):
        self.ast = ast
        self.description = description or repr(ast)
        self.steps = []  # backward compat with guided.dsl.Program
        self.search_program = search_program  # SearchProgram, for trace capture

    def execute(self, inp: Grid) -> Grid:
        # Prefer SearchProgram execution when available (handles ops
        # that don't lower to AST, like registration_transfer)
        if self.search_program is not None:
            return self.search_program.execute(inp)
        from aria.search.executor import execute_ast
        result = execute_ast(self.ast, inp)
        return result if result is not None else inp

    def size(self) -> int:
        return self.ast.size()

    def to_dict(self) -> dict:
        return {'ast': self.ast.to_dict(), 'description': self.description}

    def __repr__(self):
        return f"ASTProgram({self.description})"
