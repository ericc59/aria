Continue only if Prompt 2 showed real selection/support signal.

Goal

Add learned NEXT-EXPANSION / BINDING guidance over the primitive graph.

The model should rank:
- next graph expansion
- parameter bindings
- stop/continue

Constraints

- no task-id logic
- no benchmark-specific hacks
- no external LLM
- no family labels
- symbolic execution and exact verification remain external
- do not predict final outputs directly

What to build

Part A: Expansion supervision
- From synthetic latent graphs, build training targets for:
  - next expansion action
  - binding choice
  - stop decision

Part B: Guided search
- Integrate the learned scorer into the primitive-graph search.
- Compare:
  - unguided search
  - guided search

Part C: Metrics
- next-step accuracy
- binding accuracy
- train-verified candidate rate on held-out synthetic tasks
- exact solve rate on held-out synthetic tasks
- search cost reduction

Hard success criterion

Continue only if guided search clearly beats unguided search on held-out synthetic tasks in either:
- train-verified candidate rate
or
- exact solve rate
with comparable or lower search cost.

Deliverable format

At the end, give me:
1. how next-step and binding supervision are generated
2. how guidance is integrated into search
3. unguided vs guided synthetic metrics
4. whether the learned proposer is actually useful
