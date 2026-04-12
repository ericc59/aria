You are working in /Users/ericc59/Dev/aria. Implement a general dims-change layout strategy for packing objects into a new grid.

Goal
Add a derive strategy that detects when output is a packed layout of input objects and constructs a program that packs them into a new grid size.

Requirements
- Implement in aria/search/derive.py and aria/search/sketch.py.
- New step name: object_grid_pack.
- Must verify across all demos.

Behavior
1. Detect a set of objects in input (non-bg).
2. Determine output grid rows/cols based on output shape and a derived cell size.
3. Pack objects into cells in row/col/size/color order.
4. Place each object at top-left of its target cell or centered if you can do it consistently.

Derive heuristics
1. Cell size inferred from smallest object bbox or common object size.
2. Ordering candidates: row, col, size (largest-first), color.
3. Must reject if any object doesn’t fit in its cell.

Tests
Add tests/test_search_object_grid_pack.py:
1. Synthetic dims-change pack by row order.
2. Synthetic dims-change pack by size order.

Checklist
- Wire derive strategy near other layout/pack strategies.
- Add _exec_object_grid_pack in aria/search/sketch.py.
- Update changelog.md.
- Run pytest -q tests/test_search_object_grid_pack.py.

If you get stuck, use Codex CLI to inspect the repo and keep going.
