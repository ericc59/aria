# Claude Prompt: Overnight Autonomous Bottleneck Loop

You are working in:

`/Users/ericc59/Dev/aria`

This prompt is for unattended work. Do not stop after one chunk unless you hit a real blocker.

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`
- `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`
- `/Users/ericc59/Dev/aria/prompts/README.md`

## Goal

Work autonomously on the highest-leverage next bottleneck in canonical `aria/search`, while preserving the architecture constraints.

You should loop:

1. measure
2. diagnose
3. implement one bounded chunk
4. verify
5. commit
6. repeat if there is still a clear next step and no blocker

## Mandatory architecture constraints

- low-level primitives stay small
- no benchmark-family runtime ontology
- no task-id logic
- no hidden solver buckets
- exact verification remains the final arbiter
- macros remain above primitives and reducible to `SearchProgram`

## Required working order

Use this order unless the repo state proves a later item is blocked:

1. fix any outstanding review findings or refresh blockers
2. registration / correspondence layer
3. transition matching upgrade
4. consolidation refresh loop
5. only then: the next newly exposed bottleneck

## How to choose the next chunk

When choosing a chunk, prefer:

- the biggest structural bottleneck across both `v1-train` and `v2-eval`
- layers that unlock many tasks
- clean substrate improvements over task-specific solves

Avoid:

- chasing one benchmark point
- adding mid-level ontology to the runtime
- writing prompts instead of code

## Stop conditions

Stop only if:

- you hit a real ambiguity that cannot be resolved from local context
- the next step would require a risky architectural decision better made by the user
- you have completed a meaningful chunk, verified it, committed it, and the next step is no longer obvious

If you stop, leave a concrete handoff:

1. current solve counts or local result summary
2. the last completed chunk
3. the next exact dependency
4. any real blocker

## If you get stuck

If you get stuck, uncertain, or want a second pass on architecture/code quality:

- use the Codex CLI for code review/help
- use it to review the changed files or pressure-test your plan
- then continue with the work

Do not immediately hand control back just because the next step is nontrivial.

## Verification standard

Do not claim success without evidence.

For each chunk:

- run focused tests
- run the smallest meaningful real-task or eval verification
- update `changelog.md` concisely
- commit after verified completion

## Deliverable format per chunk

Report:

1. what bottleneck you targeted
2. what you changed
3. what files changed
4. what verified
5. what solve behavior changed, if any
6. what the next dependency is

## Final reminder

Your job is not to add more ad hoc solves.
Your job is to make `aria` better at learning and reusing structured reasoning:

- substrate first
- exact search second
- traces/macros/consolidation above that
- no entropy
