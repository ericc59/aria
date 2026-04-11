# Claude Prompt: First Macro Miner For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

Current state

The trace layer now exists:

- `aria/search/trace_schema.py`
- `aria/search/trace_capture.py`
- `aria/search/macros.py`
- `scripts/build_search_traces.py`

Solved tasks can emit structured `SolveTrace` records.
A minimal macro schema exists.

This is good.

The next step is:
- build the **first macro miner**

Important:
This pass is NOT about changing runtime execution.
It is about mining repeated solve structure from traces into reusable stored macros.

## Goal

Implement the first exact macro mining path on top of solved search traces.

I want:
- repeated successful compositions identified
- stored as macros
- reducible to `SearchProgram`
- no new runtime ontology
- no new AST ops

## What this pass is NOT

Do **not**:
- add task-id logic
- add benchmark-family labels
- add a macro executor separate from `SearchProgram`
- add new AST ops
- create vague pattern clusters with fuzzy heuristics
- overengineer a giant mining framework

This should be:
- small
- exact
- structural
- useful

## What to build

### Part A: Macro miner

Add:

- `/Users/ericc59/Dev/aria/aria/search/macro_miner.py`

I want a first exact miner that groups solved traces by structural keys such as:
- `step_actions`
- `selector_signature`
- maybe provenance / action signature combinations

This miner should:
1. load solved traces
2. group repeated patterns
3. produce `Macro` objects when a pattern repeats enough to be worth storing
4. keep the macro representation reducible to `SearchProgram`

Start simple.
Exact grouping is fine.
Do not do fuzzy clustering yet.

### Part B: Macro library builder

Add:

- `/Users/ericc59/Dev/aria/scripts/build_macro_library.py`

This should:
- read the exported search traces
- mine exact repeated compositions
- write a macro library JSON file

Suggested output:
- `/Users/ericc59/Dev/aria/results/search_macro_library.json`

### Part C: Macro quality filter

Do not store every repeated thing blindly.

Use a minimal conservative filter such as:
- frequency threshold
- maybe solve-rate metadata if available
- maybe minimum nontriviality, e.g. more than one step or meaningful selector structure

The point is to avoid filling the macro library with junk like trivial single-step identity-like patterns.

### Part D: Tests

Add focused tests for:
1. grouping traces into repeated macro candidates
2. producing `Macro` objects with the expected fields
3. saving/loading the macro library
4. filtering trivial patterns

Suggested:
- `/Users/ericc59/Dev/aria/tests/test_search_macro_miner.py`

## Constraints

- no new runtime ontology
- no new AST ops
- no black-box classifier
- no benchmark-family names as first-class system concepts
- macros must remain:
  - reducible
  - discardable
  - above the primitive layer

## Verification

Run and report:
- py_compile on new/changed modules
- focused tests for macro mining
- the builder script producing a real macro library file

## Changelog

Update:
- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit when done and verified.

## Deliverable format

At the end, report:

1. what macro miner you added
2. how macros are grouped/filtered
3. what files changed
4. what tests were added and passed
5. whether a real macro library file was produced
6. why this is aligned with:
   - `/Users/ericc59/Dev/aria/docs/AGI.md`
   - `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
   - `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

## Final reminder

The point is not to invent new permanent abstractions.

The point is to let repeated successful `SearchProgram` structures become:
- capturable
- groupable
- storable
- later reusable

without polluting the low-level primitive layer.
