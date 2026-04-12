You are working in /Users/ericc59/Dev/aria. Strengthen macro routing so mined macros actually improve solve rates.

Goal
Improve candidate ranking by matching macro signatures to search candidates with more structure than just action sequence.

Requirements
- Implement in aria/search/macros.py and aria/search/candidate_rank.py (or wherever ranking is done).
- Must remain safe: exact verification still decides correctness.

Improvements
1. Score using provenance + action_signature + selector_signature (full weight).
2. If selector_signature is missing, fallback to provenance + action_signature (0.5 weight).
3. If provenance missing, action_signature only (0.25 weight).
4. Add an optional penalty for macros with low solve_rate.

Tests
Add tests/test_search_macro_routing.py:
1. Full match beats partial match.
2. Low solve_rate macro gets lower score.
3. No macro match leaves score unchanged.

Checklist
- Update changelog.md.
- Run pytest -q tests/test_search_macro_routing.py.

If you get stuck, use Codex CLI to inspect the repo and keep going.
