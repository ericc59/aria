Run this only if the guided engine prompt pack (01-06) has shown real signal:
- synthetic guided search beats unguided search
- and ARC transfer shows at least some train-verified improvement or credible near-signal

Goal

Start a DreamCoder-style wake phase for ARIA:
- mine reusable primitive-graph fragments from successful or high-quality explanation graphs
- especially from:
  - synthetic tasks
  - ARC-1 training tasks
  - ARC-2 training tasks
  - any ARC-2 train-verified hits

Do NOT hand-author abstractions.
The point is to discover reusable graph fragments automatically.

Constraints

- no task-id logic
- no benchmark-specific hacks
- no external LLM
- exact verification remains final arbiter
- abstractions must be learned from successful traces, not invented by hand
- keep the abstraction objects small and interpretable

What to build

Part A: Collect successful traces
- Gather successful and near-successful primitive graphs from:
  - held-out synthetic tasks
  - ARC-1 training tasks
  - ARC-2 training tasks
  - any ARC-2 train-verified tasks
- Normalize/canonicalize them.

Part B: Mine reusable fragments
- Find repeated connected subgraphs / fragments that:
  - recur across tasks
  - reduce graph size when abstracted
  - preserve interpretability

Part C: Score fragment candidates
- Favor fragments that are:
  - frequent
  - compressive
  - reusable across multiple tasks
  - not too large

Hard success criterion

Continue only if you discover a nontrivial set of reusable fragments that appear across multiple tasks and measurably compress the successful graphs.

Deliverable format

At the end, give me:
1. the trace sources used
2. the fragment mining method
3. the top fragment candidates
4. how often each fragment appears
5. how much graph compression each gives
6. whether library learning is justified
