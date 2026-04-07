We are starting a new solver direction.

Do NOT extend the old family-based solver.
Do NOT add more handwritten solver families.
Do NOT treat ARC as the first training ground.

Goal

Build the foundation for a generic local learned search-guidance engine:
- structured workspace
- primitive-graph expansion grammar
- synthetic task generator
- unguided baseline search

This branch is about substrate + benchmark, not yet about solving ARC.

Constraints

- no task-id logic
- no benchmark-specific hacks
- no external LLM
- only root-level primitives may be handwritten
- symbolic execution and exact verification remain external
- keep this under a new namespace, not inside the old family-based solver

What to build

Part A: Workspace representation
- Create a reusable structured workspace representation for one task/demo.
- It should expose:
  - output size / canvas
  - preserved vs residual masks
  - objects / regions / panels / cells when available
  - relations: adjacency, containment, alignment
  - candidate supports for residual units
- It must be serializable for model input.

Part B: Primitive-graph expansion grammar
- Define a small graph-construction language for latent explanations.
- Examples of expansion actions:
  - choose output unit
  - choose input support
  - choose target subset
  - add primitive transform/rewrite node
  - bind color/axis/offset/region parameters
  - stop
- This is not a family inventory.

Part C: Synthetic task generator
- Build a synthetic generator that samples tasks from root primitives and yields:
  - train demos
  - test demos
  - latent rule graph
  - workspace state
  - correct latent decisions/bindings
- Start small. Use concepts like:
  - preserved scaffolds + residual rewrite
  - object/region rewrites
  - periodic patterns
  - simple correspondences
  - fills / recolors / translations

Part D: Unguided baseline search
- Add a baseline unguided search over the graph grammar.
- Measure:
  - train-verified candidate rate
  - exact solve rate
  - search cost / candidate count

Hard success criterion

Continue only if this prompt produces:
- a working synthetic benchmark with held-out tasks
- a working unguided baseline search
- and a clear baseline report we can beat later

Deliverable format

At the end, give me:
1. the workspace representation
2. the graph expansion grammar
3. the synthetic task generator design
4. what modules/files were added
5. the held-out synthetic benchmark split
6. the unguided baseline metrics
7. whether the substrate is ready for learned guidance
