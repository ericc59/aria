You are working in /Users/ericc59/Dev/aria. Align with AGI.md and docs/raw/aria_learning_roadmap.md.
Only low-level, domain-general improvements. No task IDs, no hardcoded colors.

Goal: Dims-change layout builder (object packing into new grid sizes).

Implement
1) Extend object_grid_pack to infer cell size and grid shape robustly:
   - Try cell size from smallest object bbox and from common size mode.
   - Allow separator width 0 or 1.
2) Add ordering candidates: row_major, col_major, size_desc, color_asc.
3) Add a consistent placement rule (top-left or centered).
4) Verify across all demos before returning.

Tests
Add tests/test_search_layout_builder.py:
- Synthetic: pack by col-major.
- Synthetic: pack by size_desc with sep=1.

Checklist
- Update changelog.md.
- Run pytest -q tests/test_search_layout_builder.py.

If stuck, use Codex CLI to inspect the repo and keep going.
