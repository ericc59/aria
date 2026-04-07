Run this only if Prompt 09 showed that library-guided retraining improves synthetic performance.

Goal

Test whether the learned abstraction library + retrained recognizer improves ARC transfer.

Use:
- synthetic pretraining
- ARC-1 training tasks as an intermediate curriculum
- ARC-2 training tasks as domain adaptation / library-mining data
- ARC-2 as the final evaluation target

Constraints

- no task-id logic
- no benchmark-specific hacks
- exact verification remains final arbiter
- no handwritten high-level families
- ARC-2 eval remains the scoreboard

What to do

Part A: ARC-1 curriculum
- Use ARC-1 training tasks as additional experience for:
  - abstraction mining
  - recognizer retraining
- Do not overfit to ARC-1 specifics.

Part A.5: ARC-2 training adaptation
- Use ARC-2 training tasks as the closest real-distribution source for:
  - additional trace collection
  - abstraction mining
  - recognizer adaptation
- Keep ARC-2 eval isolated as the final scoreboard.

Part B: ARC transfer evaluation
- Compare:
  - baseline guided engine
  - guided engine + learned library
- Evaluate on:
  - ARC-1 held-out slice
  - ARC-2 transfer slice

Metrics
- train_verified_candidate_rate
- exact solve rate
- search cost
- abstraction usage frequency on real tasks

Hard success criterion

This branch is only justified if the learned library materially improves ARC transfer over the non-library guided engine.

Deliverable format

At the end, give me:
1. how ARC-1 and ARC-2 train were used in the curriculum
2. baseline vs library-guided ARC metrics
3. which learned abstractions were actually used on ARC
4. whether the library improved transfer
5. final continue/change/stop recommendation
