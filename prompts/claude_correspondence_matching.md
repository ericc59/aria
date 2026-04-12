You are working in /Users/ericc59/Dev/aria. Upgrade correspondence matching to support feature-based matching beyond nearest/shape-only.

Goal
Add a general correspondence layer that can match objects by feature similarity instead of only nearest position or exact shape.

Requirements
- Implement in aria/guided/correspond.py and any shared helpers.
- Keep the current exact tiers, add a new “feature match” tier.
- Must be domain-general: no task IDs.

Feature match tier
1. Same dominant color OR same color histogram (within a tolerance).
2. Size ratio >= 0.5.
3. Shape similarity via mask IoU or perimeter similarity.
4. Distance used only as tiebreaker, not primary.
5. Use Hungarian assignment for global optimum.

Integration
- Extend _match_cost to incorporate feature similarity cost.
- Extend _classify_match to categorize as moved or moved_recolored when feature match is chosen.
- Ensure higher tiers remain preferred.

Tests
Add tests/test_correspond_feature_match.py:
1. Same-color, similar-shape objects moved far should match.
2. Different-color objects should not match if there is a same-color candidate.
3. Size ratio guard prevents tiny->huge matches.

Checklist
- Update changelog.md.
- Run pytest -q tests/test_correspond_feature_match.py.

If you get stuck, use Codex CLI to inspect the repo and keep going.
