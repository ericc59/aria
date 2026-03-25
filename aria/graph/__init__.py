"""State graph extraction module.

Public API:
    extract(grid) -> StateGraph
    extract_with_delta(in_grid, out_grid) -> (StateGraph, StateGraph, Delta)
"""

from aria.graph.extract import extract, extract_with_delta

__all__ = ["extract", "extract_with_delta"]
