You are working in /Users/ericc59/Dev/aria. Align with AGI.md and docs/raw/aria_learning_roadmap.md.
Only low-level, domain-general improvements. No task IDs, no hardcoded colors.

Goal: Grid-conditional rule induction (beyond 4 hardcoded rules).

Implement
1) Add a grid-cell fact extractor (row, col, parity, is_empty, dominant color, pattern signature).
2) In derive.py, add a rule-induction path for grid_conditional_transfer:
   - Build a small DNF over grid-cell facts to predict target cell content source.
   - Start with rules that map from (row, col, parity, symmetry) to a source cell.
3) In exec, recompute the same facts and apply the derived rule.

Constraints
- Rule must generalize across all demos.
- If the rule doesn’t explain every changed cell, reject.
- Do not touch task IDs or add bespoke heuristics.

Tests
Add tests/test_search_grid_rule_induction.py:
- Synthetic: parity-based rule (even rows copy left).
- Synthetic: symmetry-based rule (mirror across grid center).

Checklist
- Update changelog.md.
- Run pytest -q tests/test_search_grid_rule_induction.py.

If stuck, use Codex CLI to inspect the repo and keep going.
