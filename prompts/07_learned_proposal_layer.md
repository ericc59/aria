You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Add learned proposal hooks for decomposition and sketch ranking.

Why:
Fluidity comes from learning what hypothesis family to try, not from replacing exact verification.
This layer should explicitly help with ARC-AGI-2's three capability stresses while respecting efficiency:
- symbolic interpretation: rank decompositions that expose roles instead of literals
- compositional reasoning: rank multi-part sketch families when needed
- contextual rule application: rank relation-dependent sketches over global transforms
- efficiency: learn ordering so fewer candidates need execution

Hard constraints:
- No remote model calls
- Keep learned components optional and pluggable
- Exact verification remains the only correctness gate
- Do not let learned scores silently change semantics
- Treat efficiency as a first-class metric; report whether ranking reduces executed candidates

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
4. Add reporting for ranking cost/benefit:
   - candidates ranked
   - candidates executed after ranking
   - order changed
5. Add tests for deterministic fallback and ranking application.

Success:
- the system can learn which hypotheses to try first
- symbolic execution and verification remain unchanged
- learned proposal explicitly targets ARC-AGI-2 symbolic/compositional/contextual pressures while improving efficiency
