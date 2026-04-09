# Hybrid `aria` + Open Model Architecture

This note sketches a plausible future hybrid system that combines:

- an open language/reasoning model such as Gemma
- the `aria` symbolic/world-model substrate

The point is not to replace `aria` with a larger model.
The point is to combine:

- flexible proposal and coordination from the model
- exact state, operators, and verification from `aria`

## Why a Hybrid Makes Sense

These systems have complementary strengths.

An open model is good at:

- decomposition
- broad reasoning
- candidate generation
- language interaction
- critique and revision
- high-level routing

`aria` is good at:

- explicit entities, roles, and relations
- inspectable operators
- exact execution
- canonical skill structure
- symbolic verification
- controlled architectural growth

If used well, the model becomes an executive layer, not the whole system.

## Intended Division of Labor

### Open Model Responsibilities

The model should primarily:

1. interpret the task
2. propose decompositions
3. suggest hypotheses, bindings, or rule candidates
4. decide which subsystem to call
5. critique failed attempts
6. request more evidence or deeper search
7. summarize results for a user or another agent

It should behave like:

- controller
- critic
- planner
- interface

It should not own final truth.

### `aria` Responsibilities

`aria` should own:

1. explicit state
2. canonical operators
3. relation and binding structure
4. exact transforms
5. execution semantics
6. verifier-backed rejection of bad hypotheses
7. persistent reusable symbolic skills

It should behave like:

- world-model substrate
- exact executor
- structured memory of skills

## Minimal Hybrid Loop

A minimal hybrid loop could look like this:

1. model reads task/context
2. model proposes a decomposition
3. model asks `aria` for structured facts
4. `aria` returns entities, bindings, motifs, relations, candidate operators, and verification results
5. model proposes one or more candidate programs or staged derivations
6. `aria` executes and verifies them
7. model reads failures/diffs and proposes repairs
8. repeat under a bounded budget

This already improves on either side alone:

- the model is grounded by exact checks
- `aria` gains a much stronger controller

## Stronger Future Hybrid

A more mature hybrid would add:

- budgeted routing
- multiple candidate branches
- rule induction over derived facts
- staged program composition
- memory of successful interaction patterns
- offline consolidation of slow traces into cheaper skills

At that point, the model is no longer just chatting around `aria`.
It is acting as an executive control layer over a structured symbolic substrate.

## Why This Is Better Than "Just Use a Bigger Model"

A bigger model alone is unlikely to give:

- exactness
- inspectable skills
- stable symbolic memory
- canonical operator growth
- principled rejection of attractive but false hypotheses

The hybrid is better because:

- the model supplies breadth and flexibility
- `aria` supplies exactness and grounded structure

This is especially important for domains where:

- latent entities matter
- state transitions matter
- exact checks are available
- reusable abstractions are valuable

ARC is just one example of that pattern.

## What `aria` Needs Before This Is Worthwhile

The hybrid is only valuable if `aria` is strong enough to expose meaningful structured state and not just trivial utilities.

Key prerequisites:

1. **better controller hooks**
   - the model must be able to query structured scene/task facts

2. **better staged derivation**
   - `aria` must support multi-step transforms and intermediate facts

3. **rule search over derived facts**
   - many remaining tasks require conditional selection, not just execution

4. **clean execution boundaries**
   - model proposes
   - `aria` executes
   - verifier decides

5. **stable canonical substrate**
   - if `aria` is still drifting into narrow hacks, the hybrid will amplify that mess

## Failure Modes

This hybrid can go wrong in predictable ways.

### 1. Hidden benchmark wrapper

The model starts doing all the real work implicitly and `aria` becomes a decorative executor.

That would increase score while destroying the main architectural value.

### 2. Symbolic substrate as dead weight

If `aria` exposes too little useful structure, the model will route around it.

That means the symbolic layer is not mature enough yet.

### 3. Over-delegation to language

The model invents decompositions that sound plausible but are not grounded in actual structure.

This is why `aria` must remain verifier-backed and stateful.

### 4. No consolidation

If the hybrid only reasons online, it may become capable but inefficient.

A real hybrid should eventually learn:

- which decompositions work
- which routing policies work
- which slow traces should be compiled

## Design Principle

The correct relation is:

- model proposes
- `aria` constrains
- verifier decides
- consolidation compresses

That keeps the architecture both flexible and grounded.

## Summary

A future `aria` + open-model hybrid is plausible and probably desirable.

The model should act as:

- executive controller
- planner
- critic
- interface

`aria` should act as:

- structured world-model substrate
- exact executor
- symbolic skill system

The hybrid becomes worthwhile when `aria` is strong enough to expose meaningful state and the model is used to steer search, not to bypass the substrate.
