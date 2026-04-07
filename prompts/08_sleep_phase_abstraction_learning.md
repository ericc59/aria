Run this only if Prompt 07 found reusable graph fragments.

Goal

Turn mined graph fragments into reusable library abstractions.

This is the DreamCoder-style abstraction-learning step:
- take successful primitive graphs
- compress repeated fragments into reusable higher-level units
- extend the language/search space without hand-authoring new families

Constraints

- no handwritten abstractions
- abstractions must be defined in terms of existing primitives or earlier abstractions
- keep abstractions typed, executable, serializable, and inspectable
- no family-zoo naming

What to build

Part A: Abstraction representation
- Define how a learned abstraction is stored:
  - typed inputs/outputs
  - internal graph body
  - canonical key
  - expansion back to primitive graph

Part B: Compression pass
- Replace recurring fragments with learned abstractions in the successful trace corpus.
- Measure compression and reuse.

Part C: Library hygiene
- Keep the library small.
- Add pruning rules for low-value abstractions.

Hard success criterion

Continue only if the learned abstraction library reduces graph complexity meaningfully across the corpus without destroying executability.

Deliverable format

At the end, give me:
1. the abstraction representation
2. the learned abstraction set
3. before/after graph size on the corpus
4. reuse frequency of each abstraction
5. whether the abstraction library is actually helping
