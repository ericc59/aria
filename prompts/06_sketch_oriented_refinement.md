You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Refactor refinement from a fixed phase chain into sketch proposal + compile + verify.

Why:
Current refinement is:
- observe
- synthesize
- search
- repair
- structural edit
This locks the system into early hard decisions.
We want:
- propose decompositions
- propose sketches
- compile candidates
- verify
- refine sketch parameters / edits

Hard constraints:
- No task-specific hacks
- No remote model calls
- Preserve exact verification semantics
- Keep existing behavior as fallback until the new path is proven

Write scope may include:
- /Users/ericc59/Dev/aria/aria/refinement.py
- /Users/ericc59/Dev/aria/aria/solver.py
- /Users/ericc59/Dev/aria/aria/sketch.py
- /Users/ericc59/Dev/aria/aria/sketch_fit.py
- /Users/ericc59/Dev/aria/aria/sketch_compile.py
- /Users/ericc59/Dev/aria/tests

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/runtime

Implement:
1. Add a sketch-oriented refinement branch before legacy structural edits.
2. Keep legacy path as fallback.
3. Report:
   - sketchs_proposed
   - sketch_family
   - sketch_compiled
   - sketch_compile_failures
   - sketch_verified
4. Add tests showing the new path can produce executable candidates and does not regress existing solve paths.

Success:
- refinement is no longer entirely phase-locked to old rule families
- sketch-derived candidates become a first-class solve path
