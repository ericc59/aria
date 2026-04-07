"""Experimental / deprecated solvers — NOT part of the canonical architecture.

The canonical architecture is:
    aria.core.graph      — ComputationGraph + Specialization
    aria.core.protocol   — fit -> specialize -> compile -> verify
    aria.core.arc        — ARC domain instantiation

Modules in this package represent abandoned or superseded approaches.
They are preserved for reference and backward compatibility of tests,
but should not be used as the basis for new work.

The future learned path is the per-task recurrent graph editor
(see aria.core.editor_env), not stepper-op ranking.
"""
