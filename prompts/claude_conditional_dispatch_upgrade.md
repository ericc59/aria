# Claude Prompt: Conditional Dispatch Upgrade For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`

Also use the current state of the new selection layer as the starting point.

## Current state

The selection-layer refactor is now in a cleaner place:

- `selection_facts.py` is good substrate
- exact selector rule induction is in `search/`, which is correct
- `Pred.SELECTION_RULE` was removed from `guided/clause.py`
- selection rules now stay at the search level via:
  - `StepSelect.select_objects(...)`
  - executor `_select_targets(...)`

This is good.

The next dependency above that layer is:

**conditional action dispatch**

That is now the main thing to improve.

## Goal

Strengthen the canonical derive path for:

- selecting multiple object groups by explicit exact rules
- deriving different actions for those groups
- composing that into a clean conditional-dispatch solve path

This is the missing layer above the new selection substrate.

## What this pass is NOT

Do **not**:
- add task-id logic
- add benchmark-family labels
- add a giant new AST op
- add a hidden task solver
- add more random selector features unless strictly necessary
- revert the clean boundary that removed `Pred.SELECTION_RULE`

This pass is specifically about:
- per-group action derivation
- conditional dispatch
- staged application over selected subsets

## Problem statement

Right now:
- selection got stronger
- but many “selection” failures are co-blocked on action complexity

Meaning:
- the system can get closer to “which objects”
- but still cannot reliably derive:
  - what action each group should get
  - how those actions differ by group
  - how to compose them cleanly

Existing relevant path:
- `_derive_conditional_dispatch`
- `_derive_group_step_multi`

Those are the places to improve.

## What to build

### Part A: Audit current conditional-dispatch path

Inspect at least:

- `/Users/ericc59/Dev/aria/aria/search/derive.py`
- `/Users/ericc59/Dev/aria/aria/search/sketch.py`
- `/Users/ericc59/Dev/aria/aria/search/executor.py`

I want you to understand:
- how `_derive_conditional_dispatch` currently works
- why it currently produces 0 solves
- where the bottleneck actually is:
  - partitioning?
  - action derivation?
  - verification?
  - composition?

Do not guess. Inspect the code first.

### Part B: Strengthen per-group action derivation

Improve `_derive_group_step_multi` or whatever the relevant group-action helper is so it can derive a richer but still bounded set of actions for each selected subset.

I expect the missing cases to be things like:
- per-group recolor with different colors
- per-group move/translation tied to local geometry
- per-group fill/opening/interior behavior
- per-group action parameters tied to relation with:
  - anchors
  - frames
  - separators
  - nearby objects
  - target sites

This should remain:
- exact
- bounded
- interpretable

Do **not** make it an unbounded search over everything.

### Part C: Improve conditional-dispatch composition

Make the derive path able to:
1. split objects into groups using the new exact selector rules
2. derive an action for each group
3. compose them into one canonical search program

This must remain honest and explicit.

Do **not** hide a task solver inside one “conditional dispatch” blob.

### Part D: Prefer existing execution

Use existing execution semantics if possible.

Do not add a new AST op unless you discover a true low-level action gap that:
- recurs across groups/tasks
- is not just a missing parameterization in derive
- is low-level and domain-general

If you do find such a low-level gap, stop and report it clearly rather than smuggling in a task-shaped fix.

### Part E: Validate on real tasks

Pick 1 to 2 real tasks from the unsolved set that are now plausibly unlocked by:
- stronger selectors
- better per-group action derivation
- conditional dispatch

These should be tasks where:
- selection is a real bottleneck
- output construction is not the only blocker
- a clean conditional-dispatch path is plausible

The point is to validate the new layer, not to force a solve at any cost.

## Constraints

- no task-id logic
- no benchmark-family ontology
- no hidden task-specific decoder
- no giant rewrite
- no random new selector-feature explosion
- preserve the clean layering:
  - guided/ stays low-level
  - search/ owns induced selector rules

## Tests

Add focused tests for:
1. per-group action derivation
2. conditional dispatch composition
3. any real task regression you improve

Keep the tests tight and meaningful.

## Verification

Run and report:
- py_compile on changed modules
- focused tests
- at least the chosen real tasks via `scripts/eval_arc2.py`

## Changelog

Update:
- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit when done and verified.

## Deliverable format

At the end, report:

1. what was wrong with the old conditional-dispatch path
2. what per-group action derivation you improved
3. what files changed
4. whether any real task(s) improved or solved
5. whether you found a true low-level execution gap instead
6. why the result stays aligned with:
   - `/Users/ericc59/Dev/aria/docs/AGI.md`
   - `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`

## Final reminder

The point is not to make “conditional dispatch” into a hidden solver bucket.

The point is to make `aria/search` capable of:

- exact selection by explicit facts/rules
- exact per-group action derivation
- explicit composition of those actions

That is the next real layer above selection.
