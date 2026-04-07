# Wiki Log

## [2026-04-05] init | Wiki structure created
Established three-layer structure: `raw/` for immutable sources, `wiki/` for synthesized pages, schema in CLAUDE.md. Migrated existing PDFs and synthesized docs.

## [2026-04-05] ingest | S1, S2, S3 — Chollet ARC papers
Ingested three foundational ARC papers into `arc_reference.md`. Single synthesized page covering theory of intelligence, benchmark spec, competition results, refinement loops, knowledge overfitting. Organized by topic, not by paper.

## [2026-04-05] query | Guided engine gap analysis
Compared `aria/guided/` against ARC literature. Created `guided_vs_arc_reference.md`. Key finding: clause language is the real vocabulary (not named rewrites), inducer is the multi-step bottleneck (not grammar), priority stack is output size → multi-clause induction → conditional clauses → refinement loop.

## [2026-04-05] query | Clause vocabulary analysis
Full review of clause.py vocabulary. Found: RECOLOR/PLACE identical, GRAVITY/SLIDE same action with different stop, 6 predicates collapsible into COMPARE/RANK. Key gaps: relational predicates (data exists in PairFact, language can't express it), conditional dispatch, geometric transforms. 16-item priority list created.

## [2026-04-05] review | Clause vocabulary priorities revised
Revised priority ordering after human review. Key changes: output size inference removed (separate upstream work), generic creation vocabulary elevated to #2 (biggest actual capability gap per match-type survey), cleanup refactors demoted to tier 3, new-object creation underweighted in v1.
