Continue only if Prompt 3 showed guided search beating unguided search on synthetic tasks.

Goal

Stress-test the guided engine on held-out synthetic tasks and confirm it is robust enough to try on ARC.

Constraints

- no ARC transfer yet until the synthetic case is clearly better
- keep the evaluation honest
- report failure modes, not just wins

What to do

Part A: Generalization tests
- Evaluate on held-out synthetic tasks with:
  - larger grids
  - more distractors
  - more residual noise
  - more candidate supports

Part B: Ablations
- selector only
- expansion scorer only
- full guided system
- unguided baseline

Part C: Decide readiness for ARC
- I want a blunt answer:
  - Is the synthetic engine strong enough to justify transfer?
  - Or is it still too weak?

Hard success criterion

Continue to ARC only if the full guided system shows robust gains over unguided search on held-out synthetic tasks across multiple settings.

Deliverable format

At the end, give me:
1. held-out synthetic generalization metrics
2. ablation results
3. the single strongest remaining weakness
4. whether ARC transfer is justified
