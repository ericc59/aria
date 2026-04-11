# Claude Prompt: Transition Matching Upgrade For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

Use this prompt after the registration/correspondence pass if movement tasks are still under-detected.

## Current diagnosis

The likely bottleneck is not selection anymore.
It is the transition system itself:

- object relocation often shows up as added + removed instead of moved
- derive strategies that rely on `match_type == moved` or preserved geometry never activate
- this blocks registration transfer, anchor-based motion, and per-object destination reasoning

## Goal

Improve transition matching so preserved-geometry object motion is detected explicitly and fed to derive-time reasoning.

This pass is about:

- object alignment
- movement attribution
- exact, bounded transition semantics

This pass is NOT about:

- adding task-specific movement heuristics
- inventing benchmark-family logic
- adding a new runtime op

## What to build

### Part A: Inspect the transition pipeline

Inspect the code that:

- matches input objects to output objects
- assigns `match_type`
- distinguishes:
  - identical
  - recolored
  - moved
  - removed
  - added
  - modified

Identify where movement is being missed.

### Part B: Improve preserved-shape alignment

Strengthen matching for cases where:

- mask shape is identical or near-identical
- color is preserved or predictably changed
- only position changes

Prefer exact geometric matching first.
Remain bounded.

### Part C: Expose richer movement metadata

If useful, expose structured motion facts such as:

- `(dr, dc)` offset
- source-to-target bbox mapping
- shape-preserved movement confidence
- anchor/target-site relation if already available

Do this only if it materially helps derive-time logic.

### Part D: Validate on real tasks

Pick a small set of real movement-like failures from:

- `v1-train`
- `v2-eval`

Validate whether the transition matcher now recognizes moved objects where it previously did not.

I do not require a full task solve in this pass.
I do require that transition attribution gets better in a measurable, testable way.

## Constraints

- no task-id logic
- no benchmark-family ontology
- no hidden relocation solver
- no new runtime op unless a true primitive gap is proven

## Tests

Add focused tests for:

1. preserved-shape movement alignment
2. moved-vs-added/removed attribution
3. any structured motion metadata you expose

## Verification

Run and report:

- py_compile on changed/new modules
- focused transition tests
- the chosen real tasks or a small real-task slice showing improved movement detection

## Changelog

Update:

- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit when done and verified.

## Deliverable format

At the end, report:

1. where movement attribution was failing
2. what matching logic you changed
3. what files changed
4. what tests were added and passed
5. whether real tasks now show improved movement detection
6. whether a deeper registration dependency still remains

## Final reminder

The transition system is upstream of many derive strategies.
If it cannot recognize motion, the rest of the stack cannot reason about it.
