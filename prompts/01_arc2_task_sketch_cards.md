You are working in /Users/ericc59/Dev/aria.

You are not alone in the codebase. Do not revert anyone else's work. Do not touch anything under /Users/ericc59/Dev/aria/results/. Use apply_patch for edits. Run focused tests before claiming success.

Goal:
Create a per-task ARC-2 sketch-card corpus that describes each task individually in terms of:
- decomposition
- invariant
- construction
- what the current system assumed wrongly
- what sketch would solve it

Why:
We do NOT want broad buckets used as solver design. We need task-specific sketch descriptions over shared primitives.
The output should make ARC-AGI-2's core pressures explicit per task:
- symbolic interpretation
- compositional reasoning
- contextual rule application
- efficiency constraints

Hard constraints:
- No broad “bucket” labels as the main output
- No task-specific solver code
- No remote model calls
- This is analysis infrastructure, not a new solver

Write scope may include:
- /Users/ericc59/Dev/aria/scripts
- /Users/ericc59/Dev/aria/aria
- /Users/ericc59/Dev/aria/logs
- /Users/ericc59/Dev/aria/tests

Do not edit unless absolutely necessary:
- /Users/ericc59/Dev/aria/aria/solver.py
- /Users/ericc59/Dev/aria/aria/refinement.py
- /Users/ericc59/Dev/aria/aria/observe.py

Implement:
1. Build a task-card generator for ARC-2 eval.
2. For each task, emit:
   - task_id
   - same_dims / dims_change
   - ARC-2 pressure(s): symbolic / compositional / contextual / efficiency-sensitive
   - decomposition candidate(s)
   - observed invariant candidate(s)
   - output construction description
   - current-system failure mode
   - proposed sketch shape
   - evidence from signatures / observation / dims reconstruction
3. Keep cards task-specific. Shared tags are allowed only as secondary metadata.
4. Write both JSON and Markdown outputs under /Users/ericc59/Dev/aria/logs.
5. Add tests for schema/serialization if needed.

Success:
- every ARC-2 eval task has an individual sketch card
- cards are concrete enough to drive implementation prompts
- no “one broad bucket = one module” framing
- cards make the ARC-AGI-2 pressure(s) explicit without collapsing tasks into broad solver buckets
