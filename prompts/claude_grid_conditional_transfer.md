You are working in /Users/ericc59/Dev/aria. Implement grid-conditional transfer for ARC tasks where each grid cell’s output depends on cell coordinates and cell content features.

Goal
Add a derive strategy + execution path that:
1. Detects a grid (separator or implicit) via aria/search/grid_detect.py detect_grid.
2. Builds per-cell features: row, col, is_empty, dominant color, size, and (optionally) a boolean for “has same pattern as X”.
3. Learns a rule that maps each target cell to a source cell or to a fill pattern.
4. Renders the output grid based on those rules.

Requirements
- Implement in aria/search/derive.py and aria/search/sketch.py.
- New step name: grid_conditional_transfer.
- Must verify across all demos before returning a program.
- Must be domain-general: no task IDs, no hardcoded colors.

Rule scope
Start with simple, interpretable rules:
1. Row/col rules: if row==r or col==c then take content from a fixed source cell.
2. Symmetry rules: mirror across row/col midline, or diagonal.
3. Copy rules: nearest non-empty in row/col, or most frequent pattern in row.
4. Color rules: if cell color==X then map to pattern Y or color Z.

Implementation plan
1. Add a helper to compute per-cell features and serialize them.
2. In derive, attempt a small set of candidate rules and check exact output match.
3. Emit grid_conditional_transfer with a minimal parameter set capturing the rule.
4. In exec, recompute grid and apply the same rule to render output.

Tests
Add tests/test_search_grid_conditional_transfer.py:
1. Synthetic: row-based transfer.
2. Synthetic: mirror rule (left->right).
3. Regression candidate if you find a real ARC task that fits.

Checklist
- Wire derive strategy near other grid strategies.
- Add _exec_grid_conditional_transfer in aria/search/sketch.py.
- Update changelog.md.
- Run pytest -q tests/test_search_grid_conditional_transfer.py.

If you get stuck, use Codex CLI to inspect the repo and keep going.
