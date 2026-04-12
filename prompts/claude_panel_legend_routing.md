You are working in /Users/ericc59/Dev/aria. Implement legend-to-panel routing for ARC multi-panel tasks.

Goal
Add a derive strategy that:
1. Detects a legend panel (small grid of symbols/colors).
2. Detects a target panel that should be transformed using the legend.
3. Applies a mapping from legend symbols to target replacements.

Requirements
- Implement in aria/search/derive.py and aria/search/sketch.py.
- New step name: panel_legend_map.
- Must verify across all demos.

Legend detection
1. Use separator splits if present.
2. Prefer smallest panel by area as legend.
3. Legend cells define a mapping from input color/pattern to output color/pattern.

Execution
1. Recompute panels and legend at runtime.
2. Apply the same mapping across the target panel.

Tests
Add tests/test_search_panel_legend_map.py:
1. Synthetic legend panel mapping colors in a target panel.

Checklist
- Wire derive strategy near other panel strategies.
- Update changelog.md.
- Run pytest -q tests/test_search_panel_legend_map.py.

If you get stuck, use Codex CLI to inspect the repo and keep going.
