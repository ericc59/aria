You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Split decomposition from rule-family inference.

Why:
observe.py currently mixes extraction, interpretation, and program building.
We need reusable decomposition views that sketches can consume:
- raw connected components
- framed regions
- composite motifs
- row/column partitions
- marker neighborhoods

Hard constraints:
- No task-specific hacks
- No remote model calls
- Keep decompositions explicit and inspectable
- Do not add more bespoke _infer_X_rules as the main abstraction

Write scope may include:
- /Users/ericc59/Dev/aria/aria/observe.py
- /Users/ericc59/Dev/aria/aria/decomposition.py (new)
- /Users/ericc59/Dev/aria/tests/test_observe.py
- /Users/ericc59/Dev/aria/tests/test_correspondence.py
- /Users/ericc59/Dev/aria/tests

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/solver.py
- /Users/ericc59/Dev/aria/aria/refinement.py

Implement:
1. Extract decomposition helpers out of observe.py.
2. Introduce explicit decomposition result types.
3. Add at least these decompositions:
   - framed regions
   - composites
   - marker neighborhoods
   - raw objects
4. Keep existing observation behavior working, but route it through decomposition helpers where practical.
5. Add tests for each decomposition view.

Success:
- decomposition is a first-class layer
- future sketch fitting can consume decompositions directly
- observe.py becomes less monolithic
