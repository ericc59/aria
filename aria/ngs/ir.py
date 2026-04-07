"""Primitive-graph IR for the next-generation solver.

A PrimitiveGraph is a DAG of typed operations that explains how an output
unit is derived from input support. Nodes are either:
- Leaf: a reference to concrete data (grid, region, mask, color, int)
- PrimCall: application of a root primitive to other nodes

Values flowing between nodes are typed:
- GRID, MASK, REGION (subgrid + position), OBJECTS (set of RawObject)
- SEQ (1D array), COLOR, INT, BOOL, OFFSET (dr, dc)
- COLOR_MAP (dict[int,int]), CORRESPONDENCE (list of object pairs)

Graphs are:
- typed (each node has a declared output type)
- executable (evaluate by topological traversal)
- serializable (to tuple form for hashing/comparison)
- canonicalizable (for cross-demo comparison)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from aria.types import Grid


# ---------------------------------------------------------------------------
# Value types in the graph
# ---------------------------------------------------------------------------

class VType(Enum):
    GRID = auto()
    MASK = auto()          # bool array, same shape as some grid
    REGION = auto()        # (subgrid, row_offset, col_offset)
    OBJECTS = auto()       # list[RawObject]
    OBJECT = auto()        # single RawObject
    SEQ = auto()           # 1D numpy array
    COLOR = auto()         # int 0-9
    INT = auto()
    BOOL = auto()
    OFFSET = auto()        # (dr, dc) tuple
    COLOR_MAP = auto()     # dict[int, int]
    CORRESPONDENCE = auto()  # list[(obj_in, obj_out)]
    BBOX = auto()          # (r0, c0, r1, c1)
    VOID = auto()          # for side-effect-only nodes (shouldn't exist in pure DAG)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    """A concrete value injected into the graph."""
    name: str       # human-readable label (e.g. "input_grid", "bg_color")
    vtype: VType
    value: Any      # the actual data; not used for canonicalization

    def canon_key(self) -> tuple:
        """Canonical form for cross-demo comparison (strips concrete values)."""
        return ("leaf", self.name, self.vtype.name)


@dataclass(frozen=True)
class PrimCall:
    """Application of a named primitive to argument nodes."""
    prim: str                    # primitive name from the registry
    args: tuple[int, ...]        # indices into the graph's node list
    vtype: VType                 # output type
    params: dict[str, Any] = field(default_factory=dict)  # static params (axis, color, etc.)

    def canon_key(self, graph: PrimitiveGraph) -> tuple:
        """Canonical form: (prim, param_canon, arg_canon_keys...)"""
        arg_keys = tuple(graph.nodes[i].canon_key(graph) if isinstance(graph.nodes[i], PrimCall)
                         else graph.nodes[i].canon_key()
                         for i in self.args)
        # Canonicalize params: sort by key, convert numpy to tuple
        param_canon = tuple(sorted(
            (k, _canon_value(v)) for k, v in self.params.items()
        ))
        return ("prim", self.prim, param_canon, arg_keys)


Node = Leaf | PrimCall


def _canon_value(v: Any) -> Any:
    """Make a value hashable for canonicalization."""
    if isinstance(v, np.ndarray):
        return tuple(v.flat)
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_canon_value(x) for x in v)
    return v


# ---------------------------------------------------------------------------
# The graph itself
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveGraph:
    """A DAG of primitive operations explaining one output unit.

    nodes[0..n-1] are the operations. The last node is the output.
    """
    nodes: list[Node] = field(default_factory=list)
    output_idx: int = -1  # index of the output node

    def add_leaf(self, name: str, vtype: VType, value: Any) -> int:
        idx = len(self.nodes)
        self.nodes.append(Leaf(name, vtype, value))
        return idx

    def add_prim(self, prim: str, args: tuple[int, ...], vtype: VType,
                 params: dict[str, Any] | None = None) -> int:
        idx = len(self.nodes)
        self.nodes.append(PrimCall(prim, args, vtype, params or {}))
        return idx

    def set_output(self, idx: int) -> None:
        self.output_idx = idx

    @property
    def output_node(self) -> Node:
        return self.nodes[self.output_idx]

    @property
    def output_type(self) -> VType:
        return self.nodes[self.output_idx].vtype

    def canon_key(self) -> tuple:
        """Canonical representation of the entire graph (structure only, no data)."""
        if self.output_idx < 0:
            return ()
        return self._node_canon(self.output_idx)

    def _node_canon(self, idx: int) -> tuple:
        node = self.nodes[idx]
        if isinstance(node, Leaf):
            return node.canon_key()
        else:
            arg_keys = tuple(self._node_canon(i) for i in node.args)
            param_canon = tuple(sorted(
                (k, _canon_value(v)) for k, v in node.params.items()
            ))
            return ("prim", node.prim, param_canon, arg_keys)

    def depth(self) -> int:
        """Max depth of the graph."""
        if self.output_idx < 0:
            return 0
        return self._depth(self.output_idx, {})

    def _depth(self, idx: int, cache: dict[int, int]) -> int:
        if idx in cache:
            return cache[idx]
        node = self.nodes[idx]
        if isinstance(node, Leaf):
            cache[idx] = 0
            return 0
        d = 1 + max((self._depth(a, cache) for a in node.args), default=0)
        cache[idx] = d
        return d

    def size(self) -> int:
        """Number of PrimCall nodes (excludes leaves)."""
        return sum(1 for n in self.nodes if isinstance(n, PrimCall))

    def serialize(self) -> list[dict]:
        """Serialize to a JSON-friendly list of dicts."""
        result = []
        for i, node in enumerate(self.nodes):
            if isinstance(node, Leaf):
                result.append({
                    "type": "leaf",
                    "name": node.name,
                    "vtype": node.vtype.name,
                })
            else:
                result.append({
                    "type": "prim",
                    "prim": node.prim,
                    "args": list(node.args),
                    "vtype": node.vtype.name,
                    "params": {k: _canon_value(v) for k, v in node.params.items()},
                })
        return result


# ---------------------------------------------------------------------------
# Unified abstract rule
# ---------------------------------------------------------------------------

@dataclass
class AbstractRule:
    """A unified primitive graph with per-demo bindings.

    The graph uses abstract leaf names (e.g. "input", "bg") instead of
    concrete data. Per-demo bindings map those names to actual values.
    """
    graph: PrimitiveGraph
    description: str = ""

    def canon_key(self) -> tuple:
        return self.graph.canon_key()


@dataclass
class DemoBinding:
    """Concrete bindings for one demo pair."""
    demo_idx: int
    leaf_values: dict[str, Any]  # leaf name -> concrete value


@dataclass
class UnifiedRule:
    """An abstract rule + per-demo bindings that verify exactly."""
    rule: AbstractRule
    bindings: list[DemoBinding]
    train_verified: bool = False
    train_diff: int = 0
