"""Guided engine for ARC task solving.

Active pipeline:
  perceive.py    — label-free structural facts (ObjFact, GridFacts)
  correspond.py  — output→input object mapping (ObjMapping, top-K hypotheses)
  output_size.py — output size inference from perception dim_candidates
  clause.py      — relational clause language (Pred, Agg, Act, Clause, ClauseProgram)
  induce.py      — clause induction from demo pairs (cross-demo filtering, top-K corr)
  dsl.py         — typed DSL primitives (crop, split, combine, render, rays, repair, etc.)
  synthesize.py  — typed bottom-up synthesis + loop search over DSL primitives

Entry points:
  dsl.synthesize_program(demos) → Program  (delegates to synthesize.synthesize)
  induce.induce_program(demos)  → ClauseProgram  (clause engine fast path)

Legacy (quarantined — do not extend):
  reason.py      — brittle rule-menu pattern matching
  interpret.py   — threshold-heavy role assignment heuristics
  answer.py      — strategy-menu parallel solver
"""
