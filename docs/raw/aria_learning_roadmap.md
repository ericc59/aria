# `aria` Learning Roadmap

This roadmap turns the AGI/system discussion into a concrete repo plan.

Working thesis:

- keep the canonical base layer small and low-level
- stop adding mid-level benchmark-shaped ops
- make solves leave reusable traces
- learn recurring compositions on top of the base layer
- consolidate those learned patterns into reusable macros and better routing

The target is not "more handwritten family solvers."
The target is:

1. low-level primitives
2. learned composite patterns
3. learned routing over those patterns
4. repeated replay/consolidation over `v1` and `v2`

## Design Rules

Use these as admission rules for all future changes:

1. New AST ops must be low-level, composable, parameterizable, and domain-general.
2. Recurring mid-level patterns should become learned macros, not permanent primitives.
3. Derive should operate over explicit scene facts, not task IDs or named benchmark families.
4. Every solve should produce trace data good enough for replay and consolidation.
5. Eval reports should be truthful enough to train on.

## Current Repo State

Good foundations already exist:

- low-level execution:
  - `/Users/ericc59/Dev/aria/aria/search/ast.py`
  - `/Users/ericc59/Dev/aria/aria/search/executor.py`
  - `/Users/ericc59/Dev/aria/aria/search/sketch.py`
- scene/fact substrates:
  - `/Users/ericc59/Dev/aria/aria/search/binding.py`
  - `/Users/ericc59/Dev/aria/aria/search/motif.py`
  - `/Users/ericc59/Dev/aria/aria/search/geometry.py`
  - `/Users/ericc59/Dev/aria/aria/search/windows.py`
  - `/Users/ericc59/Dev/aria/aria/search/frames.py`
  - `/Users/ericc59/Dev/aria/aria/search/registration.py`
  - `/Users/ericc59/Dev/aria/aria/search/rules.py`
- learned/amortized proposal beginnings:
  - `/Users/ericc59/Dev/aria/aria/search/proposal_memory.py`
  - `/Users/ericc59/Dev/aria/aria/search/proposal_corpus.py`
  - `/Users/ericc59/Dev/aria/aria/search/proposal_model.py`
  - `/Users/ericc59/Dev/aria/aria/search/candidate_rank.py`

What is still missing is the bridge from:

- solved tasks

to:

- learned reusable patterns above primitives

## Phase 0: Harden The Primitive Layer

Goal:

- make the canonical AST/executor surface complete and trustworthy

Why first:

- no learning loop is useful if solved traces depend on missing executor cases or misleading eval accounting

Add:

- `/Users/ericc59/Dev/aria/tests/test_search_ast_surface.py`
  - direct AST-lowering regressions for every search action currently emitted by derive/search
  - focus on low-level ops: `tile`, `scale`, `periodic_extend`, `recolor_map`, transforms, object edits

Refactor:

- `/Users/ericc59/Dev/aria/aria/search/executor.py`
  - implement all low-level AST cases that search can currently emit
  - fail fast only for truly noncanonical/legacy cases
- `/Users/ericc59/Dev/aria/aria/search/sketch.py`
  - ensure every lowered action has one honest AST target
  - remove "AST can't express this" exceptions unless they are genuinely temporary
- `/Users/ericc59/Dev/aria/aria/eval.py`
  - keep `solved` meaning "public test correct when available"
  - keep `program_found` separate

Order:

1. enumerate all search-emittable actions
2. add AST-surface tests
3. fill missing executor cases
4. rerun `v1-train` and `v2-eval` refreshes until they complete without surface crashes

## Phase 1: Shrink And Clarify The Base Ontology

Goal:

- keep only low-level canonical primitives in the permanent execution layer

Add:

- `/Users/ericc59/Dev/aria/docs/raw/primitive_admission.md`
  - explicit bar for new AST ops

Refactor:

- `/Users/ericc59/Dev/aria/aria/search/ast.py`
  - mark noncanonical compatibility ops clearly
  - do not add new mid-level task ops without written admission notes
- `/Users/ericc59/Dev/aria/aria/search/derive.py`
  - push mid-level logic into derive/rules/macros, not executor

Likely low-level primitives still worth adding:

1. `TEMPLATE_BROADCAST`
  - mask-driven template placement / Kronecker-style block stamping
2. `REGISTRATION_TRANSFER`
  - place a source module on anchor sites while preserving local offsets
3. `PROGRESSIVE_FILL`
  - geometric boundary-anchored monotone fill with explicit front semantics

Likely not primitive-level:

- "window transfer"
- "legend decode"
- named task families

Order:

1. define primitive admission criteria
2. audit current AST ops against that bar
3. keep only low-level additions going forward

## Phase 2: Make Solve Traces First-Class

Goal:

- every solve should leave enough structure to support replay and consolidation

Add:

- `/Users/ericc59/Dev/aria/aria/search/trace_schema.py`
  - dataclasses for canonical trace records
- `/Users/ericc59/Dev/aria/aria/search/trace_capture.py`
  - helpers to record:
    - task signatures
    - extracted scene facts
    - chosen derive path
    - candidate list and ranks
    - final program
    - stage boundaries
    - near misses

Refactor:

- `/Users/ericc59/Dev/aria/aria/search/search.py`
  - emit structured trace events, not just final results
- `/Users/ericc59/Dev/aria/aria/eval.py`
  - persist trace summaries and file references in outcomes
- `/Users/ericc59/Dev/aria/aria/trace_store.py`
  - extend from "search result record" to structured solve trace storage

Order:

1. define trace schema
2. instrument search and eval
3. verify a full refresh produces usable trace data

## Phase 3: Learn Macros Above The Primitive Layer

Goal:

- recurring successful compositions become stored macros, not handwritten permanent ops

Macro definition:

- reducible to lower-level AST programs
- named by structural behavior, not benchmark task
- discardable if useless
- rankable by utility/frequency

Add:

- `/Users/ericc59/Dev/aria/aria/search/macros.py`
  - `Macro`
  - `MacroLibrary`
  - `lower_macro()`
- `/Users/ericc59/Dev/aria/aria/search/macro_miner.py`
  - mine repeated subprograms/subgraphs from solved traces
- `/Users/ericc59/Dev/aria/scripts/build_macro_library.py`
  - offline macro builder from trace store / solved corpus
- `/Users/ericc59/Dev/aria/tests/test_search_macros.py`

Refactor:

- `/Users/ericc59/Dev/aria/aria/search/search.py`
  - treat macros as candidate expansions alongside primitive compositions
- `/Users/ericc59/Dev/aria/aria/search/proposal_corpus.py`
  - include macro usage records

Order:

1. define macro schema and lowering
2. mine exact repeated subprograms first
3. add macro candidates to search ordering
4. only later mine looser structural templates

## Phase 4: Learn Fact -> Transform Routing

Goal:

- learn recurring patterns as mappings from scene facts to candidate transforms/macros

Add:

- `/Users/ericc59/Dev/aria/aria/search/fact_features.py`
  - stable feature extraction from bindings/motifs/geometry/windows/frames/registration
- `/Users/ericc59/Dev/aria/aria/search/router_model.py`
  - predict candidate transforms/macros from fact features
- `/Users/ericc59/Dev/aria/scripts/build_router_model.py`
- `/Users/ericc59/Dev/aria/tests/test_search_router_model.py`

Refactor:

- `/Users/ericc59/Dev/aria/aria/search/proposal_model.py`
  - evolve from simple family NB into transform/macro ranking over explicit fact features
- `/Users/ericc59/Dev/aria/aria/search/candidate_rank.py`
  - merge verifier features with router priors and macro priors

Order:

1. define stable fact features
2. train a simple transform classifier/ranker
3. rank candidate transforms/macros with it
4. keep exact verification as final arbiter

## Phase 5: Replay / Sleep / Consolidation

Goal:

- repeatedly run `v1` and `v2`, compress recurring solve structure, and improve future runs

Add:

- `/Users/ericc59/Dev/aria/aria/search/consolidate.py`
  - orchestration for:
    - mining traces
    - refreshing priors
    - rebuilding macro library
    - retraining router/ranker
- `/Users/ericc59/Dev/aria/scripts/consolidate_search.py`
  - one-shot sleep pass
- `/Users/ericc59/Dev/aria/tests/test_search_consolidate.py`

Refactor:

- `/Users/ericc59/Dev/aria/scripts/build_search_prior.py`
- `/Users/ericc59/Dev/aria/scripts/build_search_corpus.py`
- `/Users/ericc59/Dev/aria/scripts/build_search_model.py`
  - fold these into a cleaner consolidation pipeline while preserving simple wrappers

Order:

1. make the current build scripts callable from one orchestrator
2. add macro mining
3. add router retraining
4. compare before/after solve sets on `v1` and `v2`

## Phase 6: Grow The Low-Level Substrate Only When The Loop Demands It

Goal:

- add new primitives only when repeated failures prove a real low-level gap

Use this process:

1. collect clustered failures from refreshed `v1` and `v2`
2. ask whether the gap is:
   - missing fact extraction
   - missing routing
   - missing macro
   - or truly missing primitive
3. add a primitive only if:
   - many failures collapse to the same execution gap
   - it is low-level and domain-general
   - it cannot be cleanly represented as a macro over current primitives

Expected near-term candidates:

- `TEMPLATE_BROADCAST`
- `REGISTRATION_TRANSFER`
- `PROGRESSIVE_FILL`

## Immediate Repo Plan

This is the concrete short-horizon order.

### Step 1

Finish eval-surface hardening.

Files:

- `/Users/ericc59/Dev/aria/aria/search/executor.py`
- `/Users/ericc59/Dev/aria/aria/search/sketch.py`
- `/Users/ericc59/Dev/aria/tests/test_eval_arc2.py`
- `/Users/ericc59/Dev/aria/tests/test_search_ast_surface.py` (new)

Deliverable:

- full `v1-train` and `v2-eval` refreshes complete without AST/executor crashes

### Step 2

Add `TEMPLATE_BROADCAST` as the next canonical low-level primitive.

Files:

- `/Users/ericc59/Dev/aria/aria/search/ast.py`
- `/Users/ericc59/Dev/aria/aria/search/executor.py`
- `/Users/ericc59/Dev/aria/aria/search/derive.py`
- `/Users/ericc59/Dev/aria/aria/search/sketch.py`
- `/Users/ericc59/Dev/aria/tests/test_search_template_broadcast.py` (new)

Deliverable:

- canonical search solve for `007bbfb7`

### Step 3

Make trace capture first-class.

Files:

- `/Users/ericc59/Dev/aria/aria/search/trace_schema.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/trace_capture.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/search.py`
- `/Users/ericc59/Dev/aria/aria/eval.py`
- `/Users/ericc59/Dev/aria/aria/trace_store.py`

Deliverable:

- refreshed reports + trace store with rich solve traces

### Step 4

Add macro mining and a macro library.

Files:

- `/Users/ericc59/Dev/aria/aria/search/macros.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/macro_miner.py` (new)
- `/Users/ericc59/Dev/aria/scripts/build_macro_library.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/search.py`

Deliverable:

- repeated compositions stored and replayed as macros

### Step 5

Replace "family" language in learned routing with explicit fact -> transform/macro priors.

Files:

- `/Users/ericc59/Dev/aria/aria/search/fact_features.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/router_model.py` (new)
- `/Users/ericc59/Dev/aria/aria/search/proposal_model.py`
- `/Users/ericc59/Dev/aria/aria/search/candidate_rank.py`

Deliverable:

- the learned layer predicts transforms/macros from facts, not named benchmark families

### Step 6

Add one orchestrated consolidation loop.

Files:

- `/Users/ericc59/Dev/aria/aria/search/consolidate.py` (new)
- `/Users/ericc59/Dev/aria/scripts/consolidate_search.py` (new)

Deliverable:

- one command that:
  - mines refreshed eval traces
  - rebuilds priors
  - rebuilds corpus
  - rebuilds macro library
  - retrains routing/ranking

## What To Stop Doing

1. Stop adding benchmark-shaped mid-level AST ops.
2. Stop treating named task families as system ontology.
3. Stop using refresh crashes as a discovery mechanism for missing base execution cases; cover them with AST-surface tests.
4. Stop letting solved/unsolved reporting drift from actual public correctness.

## Success Criterion

The roadmap is working when:

- the primitive layer changes slowly
- the macro layer grows automatically from solved traces
- the routing layer improves by retraining, not by handwritten task dispatch
- repeated `v1`/`v2` passes produce:
  - more solves
  - better ordering
  - more reusable macros
  - fewer new primitive additions
