Run this only if Prompt 08 produced a useful abstraction library.

Goal

Implement the DreamCoder-style dreaming phase:
- sample tasks/programs from the learned library
- generate synthetic training data
- retrain the recognizer/proposer on the expanded language

This should let the learned model internalize the new abstractions instead of treating them as rare artifacts.

Constraints

- no benchmark-specific hacks
- no external LLM
- keep symbolic execution exact
- synthetic tasks must come from the learned library and primitive substrate, not arbitrary unrelated generators

What to build

Part A: Dream task generation
- Sample abstract programs from the learned library.
- Execute them to create imagined tasks.
- Serialize them into the same workspace/training format.

Part B: Retrain the recognizer
- Retrain the selector / next-expansion proposer on:
  - original synthetic tasks
  - replayed successful tasks
  - dreamed tasks from the learned library
  - and, if available, successful/near-successful ARC-1 and ARC-2 train traces as replay data

Part C: Compare pre/post dreaming
- On held-out synthetic tasks, measure whether the recognizer is better at:
  - target selection
  - graph expansion
  - train-verified candidate generation

Hard success criterion

Continue only if the dreamed-library retraining improves guided search performance over the pre-library recognizer on held-out synthetic tasks.

Deliverable format

At the end, give me:
1. how dreamed tasks are sampled
2. the retraining setup
3. before/after synthetic metrics
4. whether dreaming actually helps search guidance
