Continue only if Prompt 4 concluded the synthetic engine is strong enough to justify ARC transfer.

Goal

Apply the guided engine to a small ARC slice.

Do NOT try all 120 tasks first.
Use a small honest slice with diverse task types.

Constraints

- no task-id logic
- no benchmark-specific hacks
- no new handwritten families
- exact verification remains final arbiter
- the learned model only proposes/ranks latent decisions

What to do

Part A: ARC workspace serialization
- Adapt ARC tasks into the same workspace representation used for synthetic tasks.

Part B: Guided proposal on ARC
- Use the learned selector and expansion scorer to rank:
  - output units
  - supports
  - target subsets
  - graph expansions

Part C: Compare against unguided ARC search
- Measure on the ARC slice:
  - train_verified candidate rate
  - exact solve rate
  - search cost

Hard success criterion

Continue only if the guided engine materially improves train-verified candidate generation on the ARC slice over the unguided baseline.

Deliverable format

At the end, give me:
1. the ARC slice used
2. how ARC was serialized into workspace state
3. unguided vs guided ARC metrics
4. any train-verified or exact solves
5. whether transfer to ARC shows real signal
