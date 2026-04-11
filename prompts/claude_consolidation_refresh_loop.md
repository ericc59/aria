# Claude Prompt: Consolidation Refresh Loop For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Use this prompt after a meaningful architecture change lands and is committed.

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`

## Goal

Run a clean consolidation cycle so the learned artifacts reflect the current search stack.

That means:

1. fresh eval on `v1-train`
2. fresh eval on `v2-eval`
3. rebuild:
   - search prior
   - search corpus
   - search model
   - search traces
   - macro library
4. summarize deltas from the previous cycle

This pass is not about new architecture work unless the refresh exposes a clear crash or correctness bug.

## What to do

### Part A: Run fresh evals

Run the canonical eval path on:

- `v1-train`
- `v2-eval`

Use the current repo state only.
Do not mix stale reports into the interpretation.

### Part B: Rebuild learned artifacts

Rebuild all current search-learning artifacts, including:

- proposal prior
- solved-search corpus
- search family model
- solved traces export
- macro library

### Part C: Summarize the state

Report:

- current solve counts
- changes from the prior run if discoverable
- current dominant failure clusters
- artifact sizes:
  - signature buckets
  - corpus examples
  - model vocabulary
  - trace count
  - macro count

### Part D: Only patch blockers if necessary

If the refresh reveals:

- a crash
- a serialization bug
- a missing low-level executor case
- a clearly unsound accounting issue

then fix it, add a focused test, commit, and resume the refresh.

Do not drift into unrelated architecture work.

## Constraints

- no task-id logic
- no benchmark-family ontology
- no speculative solver work during the refresh
- only fix blockers that the refresh itself exposes

## Verification

Run and report the actual commands used.

## Changelog

Update:

- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.

## Commit

Commit if you had to change code.
If this pass is refresh-only with no code changes, do not make a no-op commit.

## Deliverable format

At the end, report:

1. refreshed `v1-train` solve count
2. refreshed `v2-eval` solve count
3. rebuilt artifact stats
4. top remaining bottlenecks
5. whether any blocker was found and fixed
6. the next recommended architecture move

## Final reminder

The point of the refresh loop is to convert architecture work into updated memory, priors, traces, and macros.
