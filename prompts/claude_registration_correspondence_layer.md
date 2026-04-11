# Claude Prompt: Registration / Correspondence Layer For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Before doing anything else, confirm the recent review fixes are already in place:

- `by_predicate` selectors round-trip correctly
- `crop_object_rule` no longer stores raw `StepSelect` inside params
- macro scoring no longer collapses everything to `action_signature` alone

Do not proceed until that is true.

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

## Current diagnosis

From refreshed `v1-train` and `v2-eval` analysis, the next major structural bottleneck is:

**per-object correspondence / registration**

Observed pattern:

- many unsolved tasks require each object to move to a different destination
- the destination is determined by relation to anchors, frames, openings, target sites, or matched objects
- current transition matching often fails to detect these as movements at all
- this blocks both derive-time movement reasoning and any registration-style transfer

So the next layer is not “another action heuristic.”
It is:

- object correspondence
- target-site correspondence
- registration-aware transition matching

## Goal

Build the next clean registration/correspondence substrate for `aria/search`.

This should make the system able to:
1. match source objects to likely target sites/objects
2. derive per-object destination offsets from spatial relations
3. expose those matches to derive-time movement/transfer logic

This is the registration layer from the roadmap.

## What this pass is NOT

Do **not**:
- add task-id logic
- add benchmark-family labels
- add a giant task-shaped solver
- invent a mid-level runtime op unless a true low-level gap is clearly exposed
- bypass the transition system with a one-off relocation hack

## What to build

### Part A: Inspect current registration/transition code

Inspect at least:

- `/Users/ericc59/Dev/aria/aria/search/registration.py`
- `/Users/ericc59/Dev/aria/aria/search/derive.py`
- `/Users/ericc59/Dev/aria/aria/search/geometry.py`
- the current transition matcher logic that decides `match_type`

Identify:

- where correspondence currently fails
- whether the failure is object matching, target-site enumeration, or movement attribution
- what minimal new substrate is needed

Do not guess. Inspect the code first.

### Part B: Add explicit correspondence helpers

Add or extend a registration/correspondence layer that can represent:

- source object/module
- target site / anchor site / candidate destination
- compatibility between source and target
- derived offset preserving local structure

The representation should stay:

- explicit
- typed
- reusable
- search-layer, not benchmark-layer

If a new module is needed, add one under `aria/search/`.

### Part C: Improve transition matching for per-object movement

Strengthen the transition logic so that tasks with object relocation are actually recognized as movements when appropriate.

The current symptom is that many likely movement tasks end up with zero detected movements.
That is the bottleneck.

I want:

- better alignment between input/output objects
- better attribution of moved-vs-added-vs-removed when geometry is preserved
- still exact and bounded

### Part D: Validate on real tasks

Choose 1 to 2 real tasks from the unsolved set that are genuinely registration/correspondence blocked.

The tasks should be ones where:

- movement/placement is the real issue
- not mainly a missing primitive unrelated to registration
- not mainly a selection-only problem

Use them to validate the substrate.
Do not force a solve at any cost.

### Part E: Only add a new low-level primitive if truly necessary

If during this work you discover a true low-level execution gap, report it clearly.

A new primitive is justified only if:

- it is domain-general
- it recurs across tasks
- it cannot be expressed cleanly as derive + existing primitives

Otherwise keep the work in the correspondence/derive layer.

## Constraints

- no task-id logic
- no benchmark-family ontology
- no giant solver bucket
- no hidden relocation hacks
- preserve the primitive / derive / macro boundary

## Tests

Add focused tests for:
1. correspondence helper behavior
2. improved transition matching for moved objects
3. any real task regression you improve

Keep tests explicit and structural.

## Verification

Run and report:

- py_compile on changed/new modules
- focused tests
- chosen real task evals
- if useful, a small refreshed slice showing whether some movement tasks now classify correctly

## Changelog

Update:

- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit when done and verified.

## Deliverable format

At the end, report:

1. where correspondence was failing before
2. what correspondence/registration substrate you added
3. what transition matching you improved
4. what files changed
5. what real task(s) improved, if any
6. whether you found a true low-level primitive gap instead
7. why this remains aligned with:
   - `/Users/ericc59/Dev/aria/docs/AGI.md`
   - `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
   - `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

## Final reminder

The point is to build the missing layer between:

- selection
- action
- per-object destination derivation

That layer is correspondence/registration.
Do not replace it with a task-shaped shortcut.
