You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Implement sketch fitting for two high-value families:
1. framed periodic repair
2. composite role alignment

Why:
We need to prove the sketch architecture can express task-local hypotheses that the old rule-family engine misses.

Hard constraints:
- No task-specific hacks
- No remote model calls
- Fit sketches from demos; do not hardcode task IDs
- Keep the first pass narrow

Write scope may include:
- /Users/ericc59/Dev/aria/aria/sketch_fit.py (new)
- /Users/ericc59/Dev/aria/aria/decomposition.py
- /Users/ericc59/Dev/aria/aria/observe.py
- /Users/ericc59/Dev/aria/tests
- /Users/ericc59/Dev/aria/DESIGN.md

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/solver.py
- /Users/ericc59/Dev/aria/aria/refinement.py
- /Users/ericc59/Dev/aria/aria/offline_search.py

Implement:
1. Fit a region_periodic_repair sketch from framed-region tasks like 135a2760.
2. Fit a composite_role_alignment sketch from tasks like 581f7754.
3. Emit structured evidence for why the sketch was proposed.
4. Add tests with synthetic tasks and at least one real-task regression harness if already used in repo.

Success:
- the system can propose sketches the old move/recolor/surround families could not express
- proposed sketches are task-local and evidence-backed
