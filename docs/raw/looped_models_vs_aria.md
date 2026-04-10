# Looped Models vs `aria`

This note records what `aria` should and should not learn from looped language model work such as:

- `/Users/ericc59/Desktop/2510.25741v4.pdf`

It is intentionally conservative.
The goal is not to infer vendor architecture from benchmark tables.
The goal is to decide what architectural ideas are worth importing into `aria`.

## Working Thesis

The strongest lesson from looped language models is:

- more intelligence can come from **iterative internal computation**
- not only from:
  - more parameters
  - more visible chain-of-thought tokens
  - or a single fixed forward pass

For `aria`, the useful import is **iterative refinement over explicit structured state**.

The useful import is **not**:

- replacing `aria` with one giant monolithic looped model
- or concluding that any specific frontier model must be using exactly this architecture

## What The Paper Seems To Show

From `/Users/ericc59/Desktop/2510.25741v4.pdf`, the core ideas are:

1. **Iterative latent-space computation**
   - shared parameters reused across recurrent steps
   - deeper internal compute without increasing parameter count

2. **Adaptive depth**
   - easy inputs exit early
   - hard inputs receive more internal computation

3. **Knowledge manipulation vs knowledge storage**
   - the gains are framed as better use/composition of stored knowledge
   - especially on:
     - multi-hop reasoning
     - composition
     - structured internal search

4. **Reasoning before output**
   - internal compute does more of the work
   - not everything is deferred to explicit text CoT

The most relevant conceptual phrase in the paper is the idea that looping helps models search more deeply in a parametric knowledge graph.

That framing maps well to:

- iterative retrieval of relevant internal facts
- repeated refinement of latent state
- repeated composition of partial results

## What Not To Over-Infer

The screenshot / benchmark discussion does **not** prove:

- that a specific closed model is a looped LM
- that graph-search wins necessarily come from looping alone

Strong graph/BFS/search performance could also come from:

- stronger test-time search
- better tool use
- better post-training
- outer-loop repair
- hybrid system design

So the right conclusion is:

- looped models are a plausible and important direction
- but benchmark deltas alone are not architecture proof

## Why This Matters For `aria`

`aria` already has the beginnings of a structured internal substrate:

- explicit facts
- roles
- relations
- low-level primitives
- exact execution
- exact verification

The missing piece is not just "more ops."
The missing piece is a stronger **iterative reasoning loop** over that state.

That means the paper is relevant because it supports:

- more internal iterations
- more adaptive depth
- more reuse of the same core machinery

without requiring:

- a bigger static model
- or longer visible reasoning traces

## The Right Translation Into `aria`

The good import is:

1. **Keep the symbolic substrate**
   - do not replace the world-model layer with a monolith

2. **Add recurrent refinement over structured state**
   - repeatedly update:
     - scene facts
     - bindings
     - selection hypotheses
     - rule candidates
     - macro candidates
     - candidate programs

3. **Add adaptive halting**
   - easy tasks stop after shallow refinement
   - harder tasks get more refinement rounds

4. **Use the same core machinery repeatedly**
   - fact extraction
   - rule induction
   - candidate ranking
   - macro selection

5. **Keep exact verification external**
   - internal iteration proposes/refines
   - exact verifier remains the final arbiter

So the right `aria` analogue of looping is:

- repeated internal refinement over explicit task state

not:

- one opaque recurrent blob replacing the whole solver

## Architectural Mapping

### Looped LM idea

- recurrent hidden-state refinement
- adaptive depth
- reuse of the same parameters

### `aria` analogue

- recurrent refinement over:
  - fact tables
  - scene graphs / bindings
  - induced rules
  - candidate macro/program lists
- adaptive stopping based on:
  - exact verification
  - candidate confidence
  - progress signals

## What `aria` Should Borrow Directly

### 1. Adaptive compute

`aria` should not spend the same search/refinement budget on every task.

It should:

- try cheap passes first
- allocate more internal rounds when evidence suggests payoff
- exit early when the task is already solved or clearly blocked

### 2. Recurrent internal updates

Rather than:

- one derive pass
- one search pass
- stop

`aria` should support:

- derive facts
- induce selectors/rules
- update candidate list
- refine facts/rules based on failures
- try again

### 3. Better knowledge manipulation

The paper’s “knowledge manipulation” framing matches what `aria` needs:

- not more static stored patterns alone
- but better composition of:
  - existing facts
  - existing primitives
  - learned macros

### 4. Iteration before explicit output

For `aria`, this means:

- more internal state refinement before committing to a final program
- not more verbose external chain-of-thought

## What `aria` Should Not Borrow Naively

### 1. Do not collapse the symbolic substrate into one recurrent model

That would lose:

- exact semantics
- inspectability
- canonical primitives
- verifier clarity

### 2. Do not assume latent recurrence alone replaces search

`aria` still needs:

- explicit candidate generation
- exact verification
- replay/consolidation

### 3. Do not treat benchmark advantages as proof of architecture

The paper is useful as architectural inspiration, not as evidence that another model definitely uses the same mechanism.

## Repo Implications

The near-term takeaway for `aria` is:

1. keep the low-level primitive layer small
2. keep building learned macros above it
3. add a recurrent internal refinement loop over:
   - facts
   - selectors
   - rules
   - macro choices
   - candidate programs
4. add adaptive halting/budget allocation
5. keep exact verification outside the loop as the final judge

That means the best next imports are not:

- more task-shaped ops
- more visible chain-of-thought

They are:

- iterative internal search/refinement
- adaptive compute
- replay/consolidation
- learned routing over explicit state

## Bottom Line

The paper supports a direction that is already consistent with `aria`’s best path:

- intelligence should improve through **iterative internal computation**
- not only through larger static models

For `aria`, the correct translation is:

- recurrent refinement over explicit symbolic state
- adaptive compute allocation
- exact verification at the end

That is compatible with:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`

and strengthens the case for:

- low-level primitives
- learned macros above them
- learned routing above that
- repeated replay/consolidation over `v1` and `v2`
