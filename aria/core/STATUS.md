# aria.core module status

Last reviewed: 2026-03-29

## Active (solve path)

| Module | Role | Status |
|--------|------|--------|
| `graph.py` | ComputationGraph + Specialization IR | **Active** |
| `protocol.py` | Fitter/Specializer/Compiler/Verifier protocols | **Active** |
| `arc.py` | ARC domain bridge + diagnostics + replication | **Active** |
| `seeds.py` | Deterministic seed collection | **Active** |
| `editor_search.py` | Lane-aware multi-step graph-edit search | **Active** |
| `editor_env.py` | Graph editing environment | **Active** |
| `mechanism_evidence.py` | Structural evidence + lane ranking | **Active** |
| `param_priors.py` | Lane-local parameter ordering | **Active** |

## Audit / reporting (not in solve path)

| Module | Role | Status |
|--------|------|--------|
| `benchmark.py` | Change evaluation harness | **Frozen** — stable |
| `guardrails.py` | Regression threshold checks | **Frozen** — stable |
| `slices.py` | Benchmark slice definitions | **Frozen** — stable |
| `mechanism_audit.py` | Per-task audit records | **Frozen** — stable |
| `lane_coverage.py` | Per-lane coverage funnel | **Frozen** — stable |
| `replication_audit.py` | Replication failure labels | **Frozen** — stable |
| `relocation_audit.py` | Relocation failure labels | **Frozen** — stable |
| `composition_audit.py` | Two-stage composition labels | **Frozen** — stable |
| `weak_labels.py` | Weak supervision labels | **Frozen** — stable |
| `budget.py` | Budget allocation policy | **Frozen** — not yet wired into compiler |
| `residual_priors.py` | Residual-to-edit mapping | **Frozen** — integrated but no measurable impact |
| `fragment_mine.py` | Fragment mining from verified programs | **Frozen** — too few programs to mine |
| `trace.py` | Trace data model | **Frozen** — stable |
| `trace_solve.py` | Instrumented solve path | **Frozen** — stable |
| `trace_viewer.py` | HTML trace viewer | **Frozen** — stable |

## Dormant (not in solve path, not in audit)

| Module | Role | Status |
|--------|------|--------|
| `compose.py` | Two-stage pipeline composition | **Dormant** — no new solves; needs richer stage-1 ops |
| `editor_policy.py` | Learned editor MLP policy | **Dormant** — V0, no measurable impact |
| `editor_train.py` | CEM training for learned editor | **Dormant** — V0, connected via arc.py `use_learned_editor` flag |
| `fragment_gen.py` | Dynamic fragment generation | **Dormant** — connected via diagnostics, but fragments don't compile to new solutions |
| `learn.py` | Bootstrap-propose-verify loop | **Dormant** — seed generation infrastructure |
| `library.py` | Graph template library | **Dormant** — used by proposer only |
| `proposer.py` | Compositional graph proposer | **Dormant** — used by learn.py only |
| `pixel_rule.py` | Decision tree pixel rules | **Dormant** — pre-canonical, not connected |
| `stepper.py` | Diff-guided program construction | **Dormant** — pre-canonical, not connected |
| `world.py` | Multi-layer world model | **Dormant** — pre-canonical, not connected |

## Freeze decisions

- **Selector logic**: FROZEN — evidence schema, ranking, anti-evidence all stable
- **Lane parameter surfaces**: FROZEN — all values justified, no expansion warranted
- **Periodic executor**: FROZEN — audit showed near-misses are misclassified non-periodic tasks
- **Replication executor**: FROZEN — gate tightened, 1 solve, remaining gaps are selector FPs
- **Relocation executor**: FROZEN — 0 verifications, needs selector gate tightening first
- **Search heuristics**: FROZEN — lane-aware ordering + residual priors in place, no measurable impact

## Next focus areas

1. **Executable coverage**: The bottleneck is not architecture — it's that only 3 lanes have ops that verify. New compilable mechanisms would unlock more tasks than any search/ranking improvement.
2. **Selector gate tightening for relocation**: Requiring `same_dims` and `n_input_singles > 0` would eliminate 55/92 relocation FPs.
3. **Stage-1 coverage for composition**: `compose.py` needs a real stage-1 op (canvas construction or region extraction) to unlock the 50+ canvas-first tasks.
