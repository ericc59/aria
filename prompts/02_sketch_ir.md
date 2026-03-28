You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Add a new sketch IR for task-local hypotheses.

Why:
The current system jumps too early from observation to concrete rule families/programs.
We need an intermediate representation that can express task-local hypotheses like:
- inside each framed region, infer periodic field and repair mismatches
- align composite centers to anchor axis
- construct output by packing objects under ordering O
- expand each source cell into motif M
This IR should explicitly support ARC-AGI-2's published demands:
- symbolic interpretation: colors/markers/shapes can be role variables, not literals
- compositional reasoning: multiple interacting subrules in one sketch
- contextual rule application: transforms gated by local role or relation
- efficiency: propose compact sketches before broad search

Hard constraints:
- No task-specific hacks
- No remote model calls
- Keep the IR small and typed
- Do not replace Program or executor
- Do not turn this into an open-ended unconstrained search IR

Write scope may include:
- /Users/ericc59/Dev/aria/aria/types.py
- /Users/ericc59/Dev/aria/aria/sketch.py (new)
- /Users/ericc59/Dev/aria/aria/tests
- /Users/ericc59/Dev/aria/DESIGN.md

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/runtime
- /Users/ericc59/Dev/aria/aria/solver.py

Implement:
1. Define a typed sketch IR with:
   - decomposition reference
   - primitive family
   - parameter slots
   - constraints
   - role variables / symbolic bindings where needed
   - optional confidence/evidence
2. Add a small set of first primitives:
   - region_periodic_repair
   - object_move_by_relation
   - composite_role_alignment
   - canvas_layout_construction
3. Add serialization / pretty-printing.
4. Add tests for construction and serialization.

Success:
- task-local hypotheses can be represented without committing to a full program
- the IR is small, typed, and executable-compiler-friendly
- the IR can express symbolic, compositional, and contextual hypotheses without literal-color overcommitment
