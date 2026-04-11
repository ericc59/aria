# Claude Prompt: Macro Reuse In `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

Current state

The first macro miner now exists.

You already have:
- solved trace capture
- macro schema
- macro miner
- a real macro library file

This is good.

But this is still not a closed learning loop yet, because:
- search is not yet actually reusing those macros

This pass is about:
**macro reuse in canonical `aria/search`**

## Goal

Use the mined macro library as a conservative extra proposal source and/or ranking signal inside `aria/search`.

The goal is to make repeated solved structure actually improve future search.

## What this pass is NOT

Do **not**:
- add new AST ops
- add task-id logic
- add benchmark-family labels
- make macros a second runtime language
- add fuzzy matching
- add a giant learned model
- force macro application in ways that break exact verification

Macros must remain:
- reducible to `SearchProgram`
- optional
- conservative
- exact-verifier-gated

## What to build

### Part A: Load macro library in search

Update search so it can load the macro library from:

- `/Users/ericc59/Dev/aria/results/search_macro_library.json`

Likely files:
- `/Users/ericc59/Dev/aria/aria/search/search.py`
- possibly `/Users/ericc59/Dev/aria/aria/search/candidate_rank.py`
- maybe `/Users/ericc59/Dev/aria/aria/search/macros.py`

I want the macro library available as a search-time input.

### Part B: Reuse macros conservatively

The first reuse should be conservative.

Acceptable approaches:
1. use macros as an ordering prior for candidate derivations
2. inject macro-derived `SearchProgram` candidates directly if they lower cleanly
3. use macros as a tiebreaking/ranking signal in candidate ranking

I do NOT want:
- speculative fuzzy macro application
- a separate macro executor
- magical expansion that bypasses exact verification

The cleanest first version is:
- match candidate programs or candidate signatures against macro signatures
- rank or inject accordingly
- exact verification remains final arbiter

### Part C: Keep macros reducible

Every reused macro must still reduce to ordinary `SearchProgram` structure.

Do not create:
- macro-only runtime semantics
- new AST lowering paths just for macros

### Part D: Add tests

Add focused tests for:
1. macro library loading
2. macro-based candidate ordering or injection
3. search behavior remaining stable when macro library is absent
4. search behavior improving or at least using the macro signal when a matching macro exists

Suggested:
- `/Users/ericc59/Dev/aria/tests/test_search_macro_reuse.py`

### Part E: Validate with real tasks

Run a few real tasks where the mined macros should plausibly matter.

For example:
- repeated simple recolor/crop/scale tasks already represented in the library

I do NOT require a huge solve jump in this pass.
I do require that macro reuse is:
- real
- testable
- integrated
- conservative

## Constraints

- no task-id logic
- no benchmark-family ontology
- no new runtime language
- no new AST ops
- no fuzzy macro matching
- exact verification must remain the final arbiter

## Verification

Run and report:
- py_compile on changed/new modules
- focused macro reuse tests
- one or two real-task smoke checks showing macro loading/use
- confirm search still works with no macro library present

## Changelog

Update:
- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit when done and verified.

## Deliverable format

At the end, report:

1. how the macro library is loaded
2. how macro reuse influences search
3. what files changed
4. what tests were added and passed
5. whether macro reuse changed any real task behavior
6. why this remains aligned with:
   - `/Users/ericc59/Dev/aria/docs/AGI.md`
   - `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
   - `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

## Final reminder

This is the first real consolidation loop step.

The point is:
- repeated solved structure gets mined
- mined structure gets reused
- reuse stays exact and reducible
- the primitive layer stays clean
