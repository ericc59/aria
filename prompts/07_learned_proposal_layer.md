You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Add learned proposal hooks for decomposition and sketch ranking.

Why:
Fluidity comes from learning what hypothesis family to try, not from replacing exact verification.

Hard constraints:
- No remote model calls
- Keep learned components optional and pluggable
- Exact verification remains the only correctness gate
- Do not let learned scores silently change semantics

Write scope may include:
- /Users/ericc59/Dev/aria/aria/local_policy.py
- /Users/ericc59/Dev/aria/aria/sketch_rank.py (new)
- /Users/ericc59/Dev/aria/aria/refinement.py
- /Users/ericc59/Dev/aria/training
- /Users/ericc59/Dev/aria/tests

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/runtime

Implement:
1. Add optional ranking hooks for:
   - decomposition candidates
   - sketch candidates
2. Reuse current policy infrastructure where possible.
3. Add training-data exports for:
   - DECOMP_RANK
   - SKETCH_RANK
4. Add tests for deterministic fallback and ranking application.

Success:
- the system can learn which hypotheses to try first
- symbolic execution and verification remain unchanged
