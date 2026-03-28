You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests and then the full suite before claiming success.

Goal:
Compile sketches into executable Programs.

Why:
A sketch is only useful if it becomes an executable candidate that the exact verifier can test.
The compiler should preserve the advantages ARC-AGI-2 demands:
- symbolic role bindings compile to concrete executable logic
- compositional sketches compile without flattening away interactions
- contextual guards survive compilation
- execution stays efficient and bounded

Hard constraints:
- No task-specific hacks
- No remote model calls
- Keep Program/executor as the execution target
- If a sketch cannot yet compile cleanly, fail explicitly rather than emitting fake programs
- Prefer compact compiled programs over brute-force expanded ones

Write scope may include:
- /Users/ericc59/Dev/aria/aria/sketch_compile.py (new)
- /Users/ericc59/Dev/aria/aria/sketch.py
- /Users/ericc59/Dev/aria/aria/types.py
- /Users/ericc59/Dev/aria/tests

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/runtime
- /Users/ericc59/Dev/aria/aria/solver.py

Implement:
1. Add a compiler from the first sketch families into Program.
2. Support:
   - region_periodic_repair
   - composite_role_alignment
   - object_move_by_relation if easy
3. Return structured compile failures when a sketch is underspecified.
4. Add tests that compile sketches and verify the emitted Programs are executable.

Success:
- sketch -> Program is a real path
- compile failures are inspectable
- verifier can now score sketch-derived candidates
- compiled candidates preserve symbolic/compositional/contextual intent instead of collapsing back to literal template hacks
