---
name: project_guided_gap_analysis
description: Gap analysis of guided engine vs ARC literature — what's missing, what's misunderstood, and actionable next steps
type: project
---

Guided engine gap analysis completed 2026-04-04. Key conclusions:

**Clause language IS the rewrite vocabulary.** The 12 named rewrites in grammar.py are not the real primitive set — the clause language (predicates + actions) is. Don't expand by adding named rewrites; expand by making the clause language richer (predicates, aggregations, actions). The 4 solves came from 4 action types, not 4 hardcoded rewrites.

**Why:** Adding more named rewrites is the wrong abstraction level. The power comes from composable predicates and actions in clauses.

**How to apply:** When evaluating "vocabulary too narrow" complaints, ask whether the clause language can express it, not whether there's a named rewrite for it.

**Multi-step composition exists architecturally but inducer doesn't use it.** The grammar supports multi-clause programs via NEXT. The gap is in the inducer, which only generates single-clause programs currently.

**Why:** Architecture supports it; the search/induction layer is the bottleneck.

**How to apply:** Don't redesign the grammar for multi-step — fix the inducer to emit multi-clause programs.

**Agreed critical gaps (priority order):**
1. Refinement loop — mutate near-miss programs, re-verify. Cheapest, highest leverage.
2. Conditional clauses — per-object IF predicate THEN action_A ELSE action_B. Entity-conditional composition is THE ARC-AGI-2 challenge.
3. Output size inference — unlock 320/1000 diff-shape tasks currently skipped entirely.
4. Richer aggregations — count, rank, sort, pattern matching in the clause language.

**Context:** Top 3 ARC Prize 2025 winners all use neural refinement loops. Guided engine is purely symbolic one-shot search.
