# Frontier ARC Systems vs `aria`

This note is a concrete comparison between likely frontier ARC-style systems and the current `aria` architecture.

It is intentionally conservative:

- the exact internal OpenAI reasoning algorithm is not public
- high ARC numbers likely reflect a combined system, not just a raw base model
- the comparison below is based on public patterns in the field plus behavior implied by strong eval performance

The goal is not to guess secret details.
The goal is to identify what `aria` can learn without abandoning its canonical, NDEA-aligned direction.

## Working Thesis

Frontier ARC systems likely win with a stronger **outer loop** than `aria` currently has:

- adaptive test-time compute
- multiple candidate hypotheses
- exact checking
- iterative refinement from failures
- routing across different strategies
- eventual consolidation of successful traces

`aria` currently has the opposite profile:

- weaker outer-loop search and repair
- stronger explicit symbolic substrate
- clearer canonical operator discipline
- better interpretability of what was added and why

This means `aria` is not missing one secret reasoning trick.
It is missing a more capable controller around the substrate it already has.

## Likely Frontier Ingredients

The exact implementation details are unknown, but a high-performing ARC meta-system likely uses some combination of:

1. **Adaptive test-time compute**
   - easy tasks terminate early
   - hard tasks get more retries, more branches, or deeper refinement

2. **Multiple diverse proposals**
   - several candidate decompositions, programs, or outputs
   - diversity may come from different prompts, seeds, or strategy families

3. **Exact verification**
   - candidate outputs or candidate programs are checked against train examples
   - this sharply narrows the search space

4. **Iterative repair**
   - failed attempts are not discarded immediately
   - diffs or execution failures become feedback for a next attempt

5. **Strategy routing**
   - some tasks benefit from direct output prediction
   - others from code/program synthesis
   - others from object-centric decomposition
   - a stronger system routes compute accordingly

6. **Amortization / consolidation**
   - repeated successful reasoning traces become cheaper priors or reusable policies
   - expensive search is gradually compiled into faster competence

None of these require a benchmark-shaped ontology by themselves.
They are controller-level capabilities.

## Current `aria` Strengths

`aria` already has several pieces that frontier systems also need:

1. **Exact verification**
   - candidate programs are checked, not just narrated

2. **Explicit symbolic substrate**
   - operators, roles, relations, and execution semantics are inspectable

3. **Canonical discipline**
   - the project distinguishes between reusable abstractions and task-shaped hacks

4. **Emerging latent structure**
   - binding substrate
   - motif extraction/comparison
   - object- and region-level reasoning

5. **Interpretable skill growth**
   - new capability can be tied to a concrete missing representation, derivation, or execution primitive

These are real assets.
They should not be discarded in pursuit of leaderboard-style glue.

## Current `aria` Gaps Relative to Frontier Systems

The biggest gaps are not in raw perception alone.
They are in the controller and search loop.

### 1. Weak outer-loop refinement

`aria` is still too one-shot in many places:

- derive a hypothesis
- verify it
- stop if it fails

What is missing:

- propose -> verify -> repair loops
- error-driven revision of candidate programs
- search over nearby variants after a near miss

### 2. Limited adaptive compute

`aria` does not yet spend compute selectively based on task uncertainty or promise.

What is missing:

- cheap initial attempts
- escalation rules for hard tasks
- budgeted retries only when evidence suggests a nearby solution

### 3. Weak candidate diversity

`aria` often explores too narrow a hypothesis set.

What is missing:

- parallel strategy variants
- multiple candidate derivations from the same scene facts
- voting or clustering by identical verified outputs

### 4. Weak search over derived relational facts

Recent work exposed a real gap:

- `aria` can derive useful facts
- but it is still weak at inducing small rules over those facts

What is missing:

- compact fact tables
- symbolic rule search over counts, relations, and booleans
- staged conditional reasoning built on previously derived subresults

This is likely required for tasks like `d8e07eb2`, where the remaining rule is not a new execution primitive but a condition over derived structure.

### 5. Weak composition of partial solves

`aria` is improving at decomposing tasks into stages:

- object highlight
- separator propagation
- footer fill

But it is still weak at:

- solving one stage cleanly
- then using that intermediate state as input to search the next stage

What is missing:

- explicit staged derivation
- local postconditions between stages
- composition of verified partial transforms

### 6. No real consolidation loop

The repo improves because humans:

- inspect failures
- propose abstractions
- add canonical support

What is missing:

- automated replay over solved tasks
- clustering repeated reasoning patterns
- compression of expensive search into reusable cheaper policies
- pruning or merging of overlapping ops and heuristics

This is the beginning of the "sleep" problem.

### 7. No learned or policy-guided routing

`aria` has growing substrate diversity, but little explicit routing intelligence.

What is missing:

- when to try object-centric reasoning
- when to try binding-aware derivation
- when to spend budget on rule induction
- when to stop pursuing a blocked task

## What `aria` Should Import Next

The most valuable imports are controller-level, not benchmark-shaped decoders.

### 1. Iterative propose -> verify -> repair

This is the single clearest gap.

Add a thin outer loop that:

- generates a candidate program or staged derivation
- captures exact failure information
- revises the hypothesis a few times under a budget

This should sit above canonical search, not replace it.

### 2. Parallel candidate generation with exact selection

Allow several derivations to run in parallel when the hypothesis space is ambiguous:

- different bindings
- different rule candidates
- different program parameterizations

Then choose by:

- exact verification first
- agreement / identical outputs second

### 3. Rule induction over derived facts

Build a minimal symbolic rule-search layer over:

- counts
- panel/object booleans
- all/any conditions
- small conjunctions

This is a better next step than adding another leaf execution op when the real gap is conditional selection.

### 4. Staged program composition

Let the system search for:

- stage 1 transform
- then stage 2 transform conditioned on stage 1 result

This should remain exact and inspectable.
It is likely necessary for multi-rule ARC tasks.

### 5. Budgeted routing

Introduce lightweight control logic for:

- which derive families to try first
- when to escalate to slower search
- when to abandon a line of attack

This does not need a big learned router at first.
A hand-built policy informed by the ledger is already an improvement.

### 6. Consolidation and replay

After solving tasks:

- replay them
- cluster similar solution traces
- identify reusable subrules
- merge or simplify ops
- record cheap routing priors

This is how `aria` starts becoming more efficient rather than merely broader.

## What `aria` Should Not Import

There are also anti-lessons.

### 1. Do not replace the substrate with benchmark glue

A giant ARC-specific code-generation wrapper may improve raw score, but it loses:

- interpretability
- canonical growth
- reusable world-model structure

`aria` should learn from frontier controllers, not dissolve into them.

### 2. Do not treat every near miss as a new op

The ledger already shows many failures are:

- binding gaps
- rule induction gaps
- staged-composition gaps

not always missing execution primitives.

### 3. Do not confuse score gains with architectural cleanliness

A system can gain leaderboard performance through:

- retries
- brute force
- family-specific prompting
- code generation

That may still be useful, but it should not be mistaken for the same thing as a clean reusable substrate.

## Recommended Near-Term Direction

The best near-term path is two-track:

### Track A: Strengthen the controller

Add:

- iterative repair
- candidate diversity
- rule induction over derived facts
- staged composition
- budgeted routing

This is the fastest path to better eval coverage.

### Track B: Keep improving the substrate

Continue adding only well-justified canonical structure:

- bindings
- motifs
- layout-aware comparisons
- relation/state representations
- clean execution primitives only when truly needed

This is the path that keeps `aria` aligned with NDEA.

## Summary

The likely frontier advantage is not one hidden reasoning algorithm.
It is a stronger **controller around reasoning**:

- more search
- better repair
- more diversity
- better routing
- better use of exact checks

`aria` already has something frontier systems often lack:

- a cleaner and more interpretable symbolic core

So the right move is not to imitate leaderboard systems mechanically.
It is to add the missing controller capabilities around the canonical substrate:

- propose
- verify
- repair
- compose
- consolidate

That is the shortest path toward both higher ARC coverage and a more serious architecture.
