You are working in /Users/ericc59/Dev/aria. Align with AGI.md and docs/raw/aria_learning_roadmap.md.
Only low-level, domain-general improvements. No task IDs, no hardcoded colors.

Goal: Correspondence 2.0 (feature-based matching).

Problem
Current correspondence over-relies on exact shape or nearest distance, failing on per-object movement where shapes grow/shrink or match by feature similarity.

Implement
1) Extend aria/guided/correspond.py:
   - Add a feature-match tier to _match_cost:
     * Same dominant color OR similar color histogram (tolerant).
     * Size ratio >= 0.5.
     * Mask IoU or perimeter similarity >= 0.5.
     * Distance only as tiebreaker.
   - Keep exact tiers higher priority.
   - Use Hungarian assignment for the final matching.
2) Update _classify_match to label these as moved or moved_recolored, not “new”.

Tests
Add tests/test_correspond_feature_match.py:
- Same-color similar-shape moved far matches.
- Different-color should lose to same-color candidate.
- Size ratio guard prevents tiny->huge matches.

Checklist
- Update changelog.md.
- Run pytest -q tests/test_correspond_feature_match.py.

If stuck, use Codex CLI to inspect the repo and keep going.
