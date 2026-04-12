You are working in /Users/ericc59/Dev/aria. Align with AGI.md and docs/raw/aria_learning_roadmap.md.
Only low-level, domain-general improvements. No task IDs, no hardcoded colors.

Goal: Panel routing v2 (legend → panel) with stricter mapping.

Implement
1) Strengthen legend mapping:
   - Require both source and target colors to appear in legend region.
   - If legend is grid-like, map via row/col pairs instead of free diffs.
2) Support both row and col separators; choose smallest legend region.
3) Apply mapping only to target region.

Tests
Add tests/test_search_panel_routing_v2.py:
- Synthetic: legend left maps 1→2, 3→4.
- Synthetic: legend top maps 1→3.

Checklist
- Update changelog.md.
- Run pytest -q tests/test_search_panel_routing_v2.py.

If stuck, use Codex CLI to inspect the repo and keep going.
