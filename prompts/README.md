# Prompt Pack


Suggested order:

1. `01_guided_engine_foundation.md`
2. `02_target_selector.md`
3. `03_graph_expansion_guidance.md`
4. `04_guided_search_eval.md`
5. `05_arc_transfer_slice.md`
6. `06_arc_postmortem_and_decision.md`
7. `07_wake_phase_library_mining.md`
8. `08_sleep_phase_abstraction_learning.md`
9. `09_dream_phase_synthetic_retraining.md`
10. `10_library_guided_arc_transfer.md`

Notes:
- These are for a new local learned search-guidance engine, not the old family-based solver.
- Run them sequentially.
- Each prompt has a hard stop criterion. If the criterion is not met, stop instead of blindly continuing.
- The intended workflow is:
  - prompt 1: build the substrate and synthetic benchmark
  - prompt 2: learn target/support selection
  - prompt 3: learn next-step / graph-expansion guidance
  - prompt 4: show guided search beats unguided search on held-out synthetic tasks
  - prompt 5: transfer the learned guidance to a small ARC slice
  - prompt 6: write the honest postmortem and decide whether to continue
  - prompt 7: mine reusable graph fragments from successful traces (wake)
  - prompt 8: compress those fragments into reusable abstractions/library entries (sleep abstraction)
  - prompt 9: generate dream tasks from the learned library and retrain the recognizer (sleep dreaming)
  - prompt 10: test whether the learned library improves ARC transfer

Training guidance:
- Use synthetic tasks first.
- ARC-1 training tasks are explicitly allowed as an additional curriculum source.
- ARC-2 training tasks should also be used in the later phases for trace collection, abstraction mining, and recognizer adaptation.
- ARC-2 evaluation remains the real scoreboard.
