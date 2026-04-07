"""Aria core — domain-general few-shot inductive program synthesis.

Canonical architecture (one pipeline, not multiple):

  Hypothesis representation:
    ComputationGraph   — typed DAG of abstract operations (aria.core.graph)
    Specialization     — resolved static bindings extracted from examples

  Protocol (aria.core.protocol):
    Fitter      -> propose ComputationGraph hypotheses from examples
    Specializer -> extract Specialization from graph + examples
    Compiler    -> compile graph + specialization into executable program
    Verifier    -> check program against examples (exact, binary)

    solve() orchestrates: fit -> specialize -> compile -> verify

  ARC domain instantiation (aria.core.arc):
    ARCFitter (direct ComputationGraph for grid_transform/movement,
               SketchGraph adapter for periodic/alignment/canvas)
    ARCSpecializer, ARCCompiler, ARCVerifier

  Seed collection (aria.core.seeds):
    Collect (graph, specialization, provenance) from fitters + library

  Graph editor environment (aria.core.editor_env):
    Typed graph edits over ComputationGraph + Specialization,
    scored by compile/verify + MDL.

  Deterministic graph-edit search (aria.core.editor_search):
    Bounded best-first search over graph edits from seeds.
    Architecture proof before neuralization.

  Learning via graph reuse (aria.core.learn, library, proposer):
    GraphLibrary stores verified templates; proposer constructs new
    hypotheses by adaptation and sequential composition.

Deprecated / experimental (aria.core.experimental):
    hybrid.py, neural.py — stepper-op ranking experiment.
    Not the canonical architecture. Not the basis for new work.

Supporting modules (not part of the core pipeline, but useful tools):
    stepper.py  — diff-guided iterative program construction
    world.py    — multi-layer task understanding
    pixel_rule.py — decision-tree pixel rule induction
"""
