Continue the guided engine, but only if Prompt 1 produced a working synthetic benchmark and unguided baseline.

Goal

Build the first learned component:
TARGET / SUPPORT SELECTION

This component should predict:
- which output unit matters
- which input support best explains it
- which subset of candidate targets should be acted on

Constraints

- no task-id logic
- no benchmark-specific hacks
- no external LLM
- keep symbolic execution and exact verification external
- do not attempt full program prediction yet
- the model is a proposer/ranker, not a solver

What to build

Part A: Training data
- Use the synthetic generator outputs to build supervised examples for:
  - output-unit selection
  - input-support alignment
  - target subset selection
- Keep the labels latent, not final pixels.

Part B: Model
- Start with a small local model.
- Acceptable forms:
  - transformer over serialized workspace tokens
  - GNN over workspace graph
  - hybrid
- The model should output top-K candidates for target/support selection.

Part C: Evaluation
- Measure:
  - target selection accuracy
  - target selection recall@K
  - support alignment accuracy
  - improvement in train-verified candidate rate when guided selector is used before unguided expansion

Hard success criterion

Continue only if target/support selection materially beats naive or random selection on held-out synthetic tasks.

Deliverable format

At the end, give me:
1. the training example format
2. the model architecture
3. the target/support metrics
4. whether guided selection improves downstream train-verified candidate rate
5. whether the engine is ready for graph-expansion guidance
