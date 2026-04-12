# AGI Architecture Sketch

This note captures one working hypothesis for how a system like `aria` could fit into a broader AGI architecture.

The core claim is:

- AGI is unlikely to be one monolithic model.
- AGI probably requires multiple interacting systems.
- A key milestone is not just solving more tasks, but learning reusable skills, refining world models, and becoming more efficient over time.

This is an architecture sketch, not a roadmap commitment.

## Thesis

A plausible AGI system needs at least six layers:

1. fast intuition / proposal generation
2. domain world models
3. routing / dispatch
4. deliberate reasoning / executive control
5. memory and skill consolidation
6. offline "sleep" for compression, cleanup, and reorganization

`aria` is best understood as a small instance of the first two layers:

- explicit entities
- latent structure
- operators
- exact verification

Its long-term value is not that it solves ARC directly. Its value is that it explores what a reusable skill/world-model substrate could look like.

## Layer Interfaces (Current Hypothesis)

Each layer must pass structured artifacts, not just free-form text. The working contract in `aria` is:

Layer 1 → Layer 2:
- structured scene facts (objects, roles, relations, grids)
- candidate transforms (small, cheap hypotheses)

Layer 2 → Layer 3:
- verified candidate programs
- compact structural signatures (action + selector patterns)

Layer 3 → Layer 4:
- ranked candidate program families
- traces of which strategies failed and why

Layer 4 → Layer 5:
- verified solutions + reasoning traces
- intermediate states and decision points

Layer 5 → Layer 6:
- consolidation targets (macros, priors, routing weights)
- compression objective (see Layer 6)

This is intentionally concrete: the system should pass **structured state** rather than “insights.”

## Layer 1: Fast Intuition

This layer is responsible for:

- perception
- automatic hypothesis generation
- cheap structural guesses
- pattern completion
- proposing candidate actions quickly

This is the closest analogue to human intuition.

In the current project, this is the role most closely associated with:

- panel/frame/object extraction
- candidate operator proposals
- fast structural inferences

This layer should be fast, noisy, and high-recall.
It does not need to be perfectly reliable by itself.

## Layer 2: Domain World Models

AGI likely needs multiple domain-specific world models rather than one flat knowledge store.

Each world model should represent:

- entities
- state
- relations
- transitions
- constraints
- invariants
- verification or simulation

Examples:

- math world model
  - objects: expressions, proofs, structures
  - transitions: rewrite, compose, derive, prove
  - verification: symbolic checks or proof systems

- physics world model
  - objects: bodies, media, fields, frames
  - transitions: evolve, collide, propagate, conserve
  - verification: simulation and invariant checks

- biology world model
  - objects: cells, tissues, pathways, organisms
  - transitions: regulate, mutate, grow, signal
  - verification: causal consistency and predictive fit

For ARC, the world model is tiny and symbolic:

- panels
- regions
- objects
- roles
- transformations
- layout constraints

The binding layer in `aria/search/binding.py` is a small step toward a world-model layer because it adds:

- typed entities
- roles
- relations

## Layer 3: Router / Dispatcher

A combined AGI system will need a way to decide:

- which subsystem to invoke
- whether one subsystem is enough
- whether multiple systems should be composed
- when to escalate to slower reasoning

This router is probably best understood as hybrid:

- neural for relevance priors and coarse routing
- symbolic/policy logic for constraints, sequencing, and safety

A pure neural router is unlikely to be enough on its own.

Current status in `aria`:
- Routing is hardcoded by phase order in `aria/search/search.py`.
- This is the dominant ceiling on coverage because it allocates compute poorly.
- Learned/adaptive routing is a prerequisite for consolidation to matter.

Hypothesis + success criterion:
- Hypothesis: a simple learned router over scene facts can push the correct strategy family into the top‑K.
- Success criterion: top‑K routing recall > 80% on solved tasks with reduced search cost.

## Layer 4: Deliberate Reasoning / Executive Control

This is the layer most likely to look LLM-like or LLM-hybrid.

Its job is not just to summarize. It should:

- decompose problems
- request information from specialists
- compare conflicting outputs
- generate abstraction candidates
- ask for more evidence
- maintain explicit task state
- decide when to stop

This is the closest analogue to internal thought or reflective reasoning.

An LLM may be a major part of this layer, but probably should not be the whole thing.

Current status in `aria`:
- This layer is effectively **missing**.
- The system has a 2‑step composition ceiling and no explicit planner.
- This is a primary architectural gap behind the current solve ceiling.

Hypothesis + success criterion:
- Hypothesis: a structured decomposition planner (not just longer enumeration) unlocks 3–5 step tasks.
- Success criterion: multi‑step tasks solved without exponential candidate growth.

## Layer 5: Memory and Skill Consolidation

AGI should not only solve new problems. It should turn repeated expensive reasoning into reusable competence.

This layer is responsible for:

- storing successful abstractions
- compiling repeated reasoning traces into skills
- merging overlapping skills
- pruning bad or redundant skills
- improving routing policies
- retaining reusable procedures over time

This is where true skill learning begins.

Without this layer, the system may remain impressive but transient:

- good at solving
- poor at cumulative improvement

Current status in `aria`:
- Trace capture and macro mining exist, but reuse is still shallow.
- Macros must become parameterized, derivation‑bound templates to be useful.

Hypothesis + success criterion:
- Hypothesis: replay + parameterized macro mining compresses repeated solves into reusable templates.
- Success criterion: measurable solve‑rate or search‑budget improvement on held‑out tasks.

## Layer 6: Offline "Sleep"

The system likely also needs an explicit offline phase analogous to sleep.

This phase is not for solving external tasks directly. It is for internal improvement:

- consolidation
- compression
- defragmentation
- replay
- calibration
- world-model refinement

Possible functions:

- replay solved tasks and re-express them with cleaner abstractions
- distill slow reasoning traces into cheaper policies
- merge duplicated skills
- prune narrow benchmark-shaped heuristics
- update confidence and escalation policies
- discover more compact latent representations

This layer is important because AGI should improve not only in capability, but in efficiency.

Compression objective (provisional):
- Minimize description length of the solution corpus while preserving solve coverage.
- A concrete proxy is MDL: total program length + macro library size + routing complexity.
- This forces the system to prefer reusable abstractions over one‑off heuristics.

Hypothesis + success criterion:
- Hypothesis: consolidation that optimizes MDL yields better generalization with stable or lower compute.
- Success criterion: solve coverage increases while average program length or search cost decreases.

## Efficiency as a Core Requirement

A serious AGI system should improve along at least three axes:

1. capability
2. efficiency
3. compression

That means it should:

- solve more problems over time
- solve familiar classes more cheaply over time
- turn repeated experience into shorter reusable procedures

Without this, the system may get broader but remain permanently expensive and fragmented.

## How `aria` Fits

`aria` should not be seen as "the AGI."
It is more useful as a prototype for one class of subsystem:

- explicit structural state
- typed roles and relations
- symbolic operators
- exact verification
- reusable canonical skills

This makes `aria` relevant as:

- a candidate intuition substrate for certain domains
- a candidate world-model substrate for structured reasoning
- a testbed for skill formation and consolidation

The most promising direction is not to scale the current ARC solver into everything.
It is to generalize the architectural pattern:

- entities
- roles
- relations
- transitions
- constraints
- verifiers

Then instantiate that pattern in multiple domains.

## Current Instantiation (Concrete Mapping)

| Layer | Concept | `aria` Implementation | Status |
|---|---|---|---|
| 1 | Fast intuition | `perceive.py`, seed schemas, quick derives | Working |
| 2 | World model | `binding.py`, `selection_facts.py`, `geometry.py` | Partial |
| 3 | Router | Phase ordering in `search.py` | Hardcoded |
| 4 | Deliberate reasoning | (none) | Missing |
| 5 | Consolidation | traces, `macro_miner.py` | Scaffolded |
| 6 | Sleep | build scripts | Manual |

## Multi-System AGI Picture

A plausible overall system could look like:

1. a fast proposal layer
2. multiple ARIA-like domain systems
3. a router that chooses which systems to invoke
4. an executive reasoner that coordinates them
5. a memory/consolidation layer that turns repeated successes into reusable skills
6. an offline sleep loop that compresses and reorganizes the whole system

The result is not one giant model.
It is a coordinated ecology of systems.

## What This Implies for `aria`

- improving explicit latent structure
- improving typed bindings and correspondences
- keeping canonical operators clean and reusable
- separating derive-time reasoning from runtime execution
- making successful abstractions compressible and testable
- eventually enabling offline consolidation of learned skills

In short:

- `aria` matters most as a substrate for reusable structured competence
- not as a bag of benchmark-specific tricks

## Open Questions

- What should the common interface be across multiple ARIA-like domain systems?
- How should routing be trained or updated?
- What should stay symbolic versus be learned?
- How should slow-path reasoning be distilled into fast-path skills?
- What objective should govern sleep-time consolidation?
- How should the system decide when to create, merge, refine, or delete a skill?

These questions matter more than any single benchmark score.
