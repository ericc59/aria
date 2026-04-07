# Guided Engine vs. ARC Reference: Gap Analysis

How `aria/guided/` maps to the theoretical framework and competitive landscape described in the ARC reference documents.

---

## Architecture Summary

The guided engine is a **multi-step program synthesis system** with:

- **Perception** (`perceive.py`): extracts objects, relationships, separators, regions from grids
- **Workspace** (`workspace.py`): structured representation — objects, relations, residual units (what changed)
- **Interpretation** (`interpret.py`): assigns semantic roles (SCAFFOLD, LEGEND, DATA, TARGET, MARKER) and answer modes (complete, decode, propagate, compare, place, repair)
- **Grammar** (`grammar.py`): action vocabulary — SELECT_TARGET → REWRITE → BIND → NEXT/STOP. Programs are action sequences.
- **Search** (`search.py`): BFS over partial programs. Enumerate extensions, verify on train demos.
- **Informed Search** (`informed_search.py`): residual analysis constrains candidate generation — only generates extensions plausible given the input→output diff
- **Guided Search** (`guided_search.py`): beam search with a learned selector model scoring candidate steps
- **Hypothesize** (`hypothesize.py`): cross-demo hypothesis accumulation — "object X became Y because of Z"
- **Atoms** (`atoms.py`): reusable concept library extracted from training tasks, composed to solve harder ones
- **Generalize** (`generalize.py`): rule generalization across demos
- **Synthetic** (`synthetic.py`): synthetic task generation for training the selector model
- **Expansion** (`expansion.py`): learned next-step and binding prediction

---

## Alignment with Chollet's Vision

Chollet's 2019 paper proposed an ideal ARC solver architecture:

| Chollet's Proposal | Guided Engine | Status |
|---|---|---|
| DSL encoding Core Knowledge priors | Grammar with Target/Rewrite enums | **Partial** — covers fill, recolor, delete, symmetrize, periodic repair, copy/stamp, swap, move. Missing: scaling, rotation, translation, projection, line drawing, pattern continuation |
| Generate candidate programs from I/O | `informed_search` analyzes residuals to generate plausible candidates | **Implemented** |
| Reuse/recombine subprograms | `atoms.py` AtomLibrary for reusable concepts | **Partial** — library exists but composition is limited |
| Select by simplicity or learned likelihood | `guided_search` with StepSelector model; `informed_search` with diff-reduction scoring | **Implemented** |
| Apply top candidates to test | `predict_test()` | **Implemented** |

### Core Knowledge Prior Coverage

| Prior Category | What ARC Needs | What Guided Has |
|---|---|---|
| **Objectness** | Parse grids into objects by color/spatial contiguity | `_extract_objects` — 4-connected components per color |
| Object persistence | Objects persist across transformation | Residual analysis tracks what changed vs preserved |
| Object contact | Contact/adjacency relationships | `_compute_relations` — adjacent, contains, aligned |
| **Goal-directedness** | Model transformations as intentional | `interpret.py` answer modes (complete, decode, repair, etc.) |
| **Numbers/counting** | Count, compare, sort objects | `perceive.py` — `n_same_color`, `n_same_size`, size_counts |
| **Geometry/topology** | Symmetry, rotation, translation, containment, connectivity | Symmetry repair, containment detection, adjacency. **Missing**: rotation, translation, scaling, line drawing, orthogonal projection |

---

## Alignment with Competitive Landscape (2025)

### vs. Top Kaggle Scores (Refinement Loops)

| Approach | Key Mechanism | Guided Engine Equivalent |
|---|---|---|
| **NVARC** (24% v2): test-time training + synthetic data | Neural weights encode task-specific program via gradient descent | `synthetic.py` generates training data, but no neural test-time training loop |
| **ARChitects** (16.5% v2): masked-diffusion LM + self-refinement | 2D-aware neural model recursively refines output | No neural output refinement — search is symbolic only |
| **MindsAI** (12.6% v2): test-time fine-tuning + ensembles | Pretrained model adapted per task | No pretrained neural backbone |

**Gap**: The guided engine is purely symbolic search. The top 3 competition winners all use **neural refinement loops**. The engine has no mechanism for gradient-based adaptation.

### vs. Paper Award Winners

| Approach | Guided Engine Comparison |
|---|---|
| **TRM** (7M params, 8% v2): recursive neural refinement, zero pretraining | Fundamentally different paradigm — encodes programs in weights, not symbols |
| **SOAR** (+52% v1): LLM self-improving evolutionary synthesis | Guided engine has `atoms.py` for concept reuse but no self-improving evolution |
| **CompressARC** (76K params, 4% v2): MDL + VAE | No MDL or compression-based search |

### vs. Evolutionary Program Synthesis

The guided engine is closest to **evolutionary program synthesis** approaches (Berman, Pang):
- Both generate candidate programs and verify against I/O pairs
- Both use a two-phase explore/verify cycle
- **Key difference**: the guided engine uses BFS/beam search over a fixed grammar; evolutionary approaches mutate/recombine programs and can escape local optima

---

## Critical Gaps (Ordered by Impact)

### 1. Rewrite Vocabulary Too Narrow

The grammar has **12 rewrites**: FILL, RECOLOR, DELETE, SYMMETRIZE, PERIODIC_REPAIR, COPY_STAMP, FILL_WITH_FRAME, RECOLOR_TO_ADJ, SWAP_COLORS, MOVE_TO_ENCLOSED, SWAP_ENCLOSED, COLOR_MAP.

**Missing for ARC-AGI-2 compositional generalization**:
- **Geometric transforms**: rotate, flip, translate, scale objects
- **Pattern operations**: tile, repeat, extend, extrapolate
- **Spatial operations**: crop, resize output, grid subdivision
- **Relational operations**: sort objects, align objects, map positions
- **Conditional operations**: if-then-else based on object properties (the "contextual rule application" category that ARC-AGI-2 specifically emphasizes)

### 2. No Entity-Conditional Composition

ARC-AGI-2's hardest category is **contextual rule application**: the same transformation applied differently based on per-object properties. The grammar has no branching — every step applies uniformly. There's no `IF color == red THEN move_left ELSE move_right`.

The `_matches_predicate` function in grammar.py selects objects by predicate, but the action applied is the same for all selected objects. Per-object conditional action is missing.

### 3. No Output Size Prediction

The engine assumes `same_shape` (input shape == output shape). Many ARC tasks produce outputs of different dimensions. There's no mechanism to predict output size or construct a differently-sized output canvas.

### 4. No Multi-Step Dependency

`informed_search` generates multi-step programs by appending NEXT, but each step's extensions are generated independently. There's no mechanism for step N to be conditioned on the result of step N-1 — the "multi-step compositional reasoning" that ARC-AGI-2 emphasizes as a key challenge.

### 5. No Refinement Loop

The ARC Prize 2025 report identifies the **refinement loop** as the central paradigm driving progress. The guided engine does one-shot search: generate candidates, verify, done. There's no mechanism to:
- Take a near-miss program and iteratively improve it
- Use verification failures as feedback to guide next generation
- Apply gradient-based optimization to refine a candidate

### 6. In-Context Symbol Definition

ARC-AGI-2 features tasks where objects define their own meaning within the task (e.g., colored rectangles with N holes mean "use this color for shapes with N holes"). The `interpret.py` LEGEND role detection is a start, but there's no mechanism to learn and apply arbitrary symbol mappings discovered within a single task.

---

## Strengths Relative to the Literature

1. **Residual-driven search** (`informed_search`): analyzing what changed to constrain candidate generation is well-aligned with Chollet's emphasis on controlling search budget. This avoids the >50% waste on useless programs noted in project memory.

2. **Cross-demo hypothesis accumulation** (`hypothesize.py`): building rules from observations across multiple demos is principled — each demo reveals different aspects of the rule.

3. **Semantic interpretation** (`interpret.py`): role assignment (SCAFFOLD, LEGEND, DATA, etc.) and answer modes (complete, decode, repair, etc.) provide structured priors that constrain search space. This is richer than most competing symbolic approaches.

4. **Atom library** (`atoms.py`): concept reuse across tasks parallels SOAR's program abstraction library and Chollet's recommendation to "reuse and recombine subprograms."

5. **Object-relational perception** (`perceive.py`, `workspace.py`): rich structural facts (separators, regions, pairwise relations) encode the Core Knowledge priors more explicitly than neural approaches.

---

## Strategic Recommendations (from the Literature)

1. **Add a refinement loop**: even a simple one — take the best near-miss program, mutate it, re-verify. This is the single biggest paradigm gap vs. the competitive field.

2. **Expand the rewrite vocabulary**: geometric transforms (rotate, flip, translate, scale) and conditional branching would dramatically increase coverage of ARC-AGI-2 compositional tasks.

3. **Consider hybrid neuro-symbolic**: use the existing symbolic grammar for structure, but add a small neural model (TRM-style, ~7M params) for tasks that resist symbolic decomposition. Train per-task at test time.

4. **Evolutionary search over programs**: instead of BFS/beam, evolve programs through mutation and crossover. This handles the multi-step dependency problem naturally — mutations at step N propagate through the execution of subsequent steps.

5. **Self-improving search (SOAR-style)**: fine-tune candidate generation on successful search traces. The synthetic task infrastructure already exists in `synthetic.py`.
