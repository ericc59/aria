# ARC-AGI Reference

Synthesized from three primary sources:
- Chollet, "On the Measure of Intelligence" (2019)
- Chollet et al., "ARC-AGI-2" (2025)
- Chollet et al., "ARC Prize 2025 Technical Report" (2026)

---

## 1. Theory of Intelligence

### Definition

Intelligence is **skill-acquisition efficiency** over a scope of tasks, with respect to priors, experience, and generalization difficulty.

Formally, each task contributes: `E[skill * generalization_difficulty / (priors + experience)]`, averaged over all tasks in scope weighted by task value.

Skill is the output artifact of intelligence. High skill is not high intelligence — unlimited priors or unlimited training data can buy arbitrary skill without generalization.

### Skill vs. Intelligence

| Concept | What it is | Property of |
|---|---|---|
| Skill | Probabilistic average of evaluation scores | A skill program (artifact) |
| Intelligence | Rate of converting priors + experience into skill at novel tasks | The system that generates skill programs |
| Priors | Relevant information embedded in the system at initialization | System + task pair |
| Experience | Cumulative novel, relevant information from training | System + task + curriculum |
| Generalization difficulty | Fraction of solution NOT explained by optimal training-time behavior | Task + curriculum |

### Spectrum of Generalization

| Level | Name | Characterization | Example |
|---|---|---|---|
| 0 | None | No uncertainty, exhaustive enumeration | Tic-tac-toe via minimax |
| 1 | Local (robustness) | Known unknowns, single well-defined task | Image classifier on held-out test set |
| 2 | Broad (flexibility) | Unknown unknowns, related task category | L5 self-driving, domestic robots |
| 3 | Extreme (generality) | Unknown unknowns across unknown domains | Human cognition |

AI history is a slow climb up this spectrum. ARC targets broad generalization.

### Why Measuring Skill Alone Fails

- A locality-sensitive hash table has near-zero generalization but can achieve arbitrary skill given enough data.
- DeepBlue beat Kasparov but taught nothing about intelligence; the knowledge didn't generalize.
- Both priors (hard-coded knowledge) and experience (more training data) can game any skill test.
- "The AI effect": confusing the intelligence of researchers building a system with the intelligence of the system itself.

### Core Knowledge Priors

From developmental psychology (Spelke & Kinzler). Innate assumptions shared by all humans, acquired before age ~4:

**a. Objectness & elementary physics**
- Object cohesion: objects are continuous, connected, bounded wholes
- Object persistence: objects don't spontaneously appear/disappear
- Contact: objects don't act at a distance or interpenetrate

**b. Agentness & goal-directedness**
- Some objects are agents with intentions, pursuing goals efficiently
- Agents act contingently and reciprocally

**c. Natural numbers & elementary arithmetic**
- Innate abstract number representations for small numbers
- Counting, comparison, addition, subtraction
- All quantities in ARC are < ~10

**d. Elementary geometry & topology**
- Distance, orientation, in/out containment
- Lines, rectangular shapes, symmetries
- Rotations, translations, scaling
- Connectivity, containment, orthogonal projections

These are the ONLY priors ARC assumes. No language, no acquired knowledge, no domain expertise.

---

## 2. The ARC Benchmark

### Format (Unchanged Across Versions)

- Tasks defined as I/O pairs of grids
- Grids: 1x1 to 30x30, cells have one of 10 discrete values (colors)
- Each task has 2-5 demonstration pairs (median 3) and 1-3 test pairs
- Goal: infer the transformation rule from demos, apply to test inputs
- Scoring: binary exact match on full output grid
- 2 attempts per test input (v1: 3 attempts)

### Three Defining Characteristics

1. **Resists memorization**: every task is unique, each follows different underlying logic
2. **Minimal prior knowledge**: only Core Knowledge priors required, no specialized world knowledge or language
3. **Feasible for humans**: solvable by regular people without special training

### Dataset Versions

#### ARC-AGI-1 (2019)

| Subset | Size |
|---|---|
| Training | 400 |
| Public Evaluation | 400 |
| Private Evaluation | 200 |

Known limitations: ~49% of private eval susceptible to brute-force program search (aggregate of all 2020 submissions); saturated below human-level (>97% solvable by high-end humans); difficulty inconsistent across subsets; information leakage from 4 years of reusing same 100 private tasks.

#### ARC-AGI-2 (2025)

| Subset | Size | Purpose |
|---|---|---|
| Public Training | 400 | Imported from v1, demonstrates format & priors |
| Semi-Private Eval | 120 | Intermediate leaderboard scoring |
| Private Eval | 120 | Final competition scoring, fully private |

Design changes vs v1:
- **More unique**: every task entirely novel
- **More complex**: larger grids, more objects, more concepts per task
- **Less brute-forcible**: intentionally resists exhaustive search
- **Calibrated difficulty**: mean human accuracy differs <= 1pp across partitions
- **Extensive human testing**: 407 participants, 13,405 attempts, 62% success rate

#### ARC-AGI-3 (Expected Early 2026)

First major format change since 2019. Key differences:
- **Interactive environments** instead of static I/O pairs
- Tests **agentic reasoning**: exploration, planning, memory, goal acquisition, alignment
- **Efficiency metric**: formal comparison of human and AI action efficiency (learning efficiency)
- Hundreds of never-before-seen interactive environments

### What Makes ARC-AGI-2 Harder

#### Compositional Generalization

The core challenge of ARC-AGI-2. Four categories:

**Multi-rule compositional reasoning**: Multiple simultaneous interacting rules. Example: crop to rectangular frame, rescale colored objects, place into matching holes.

**Multi-step compositional reasoning**: Sequential rule application where step N depends on outcome of step N-1. Cannot predict step N+1 without executing prior steps. Example: iteratively placing objects where position/orientation depends on previous placements.

**Contextual rule application**: Core transformation modulated by contextual elements. A form of control flow. Example: isolating shapes and stacking them, but which side (left/right) depends on outline color.

**In-context symbol definition**: Objects whose meaning is defined within the task. Example: colored rectangles with N holes encode "use this color for shapes with N holes." On-the-fly symbolic assignment is a major challenge for frontier AI.

#### Human Testing Data

- 407 participants, diverse professional backgrounds
- Median solve time: 2.2 minutes for successful tasks
- No demographic factor (occupation, programming, math background) predicted performance
- Confirms tasks measure general problem-solving, not domain expertise
- 100% of tasks solved by >= 2 independent non-expert humans

---

## 3. Competition History & State of the Art

### Score Progression

| Year | Competition | Top Score (Private) | Key Method |
|---|---|---|---|
| 2020 | Kaggle ARC-AGI | 20% | Brute-force program search (Icecuber) |
| 2022 | ARCathon 1 | — | — |
| 2023 | ARCathon 2 | — | — |
| 2024 | ARC Prize 2024 | 53.5% (ARChitects) | Test-time adaptation (TTA) |
| 2024 | o3 preview (private) | 76-88% | Massive test-time compute ($200-$20K/task) |
| 2025 | ARC Prize 2025 (v2) | 24.03% (NVARC) | Test-time training + synthetic data |

### ARC-AGI-2 Model Baselines (Semi-Private, May 2025)

| Model | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| o3-mini (High) | 34.5% | 3.0% |
| o3 (Medium) | 53.0% | 3.0% |
| ARChitects (2024) | 56.0% | 2.5% |
| o4-mini (Medium) | 41.8% | 2.4% |
| Icecuber (2020) | 17.0% | 1.6% |
| o1-pro (Low) | 23.3% | 0.9% |
| Claude 3.7 (8K) | 21.2% | 0.9% |

Below 5% on ARC-AGI-2 is generally noise. Signal emerges above 5%.

### ARC Prize 2025 Winners

**Top Scores (ARC-AGI-2 Private)**:
1. NVARC (24.03%, $25k) — test-time training + synthetic data, builds on ARChitects 2024
2. ARChitects (16.53%, $10k) — 2D-aware masked-diffusion LM, recursive self-refinement
3. MindsAI (12.64%, $5k) — test-time fine-tuning + augmentation ensembles + tokenizer dropout

**Paper Awards**:
1. TRM ($50k) — 7M-param recursive model, 45% v1 / 8% v2
2. SOAR ($20k) — self-improving evolutionary program synthesis, +52% on v1
3. CompressARC ($5k) — MDL-based 76K params, 20-34% v1 / 4% v2

**Honorable Mentions**: "ARC-AGI is a Vision Problem", "Product of Experts with LLMs", "Beyond Brute Force (neuro-symbolic)", "Vector Symbolic Algebras", "Evolutionary Test-Time Compute", "Efficient Evolutionary Program Synthesis", "ARC-NCA (developmental)", "Exploring Search and Learn"

---

## 4. The Refinement Loop (Dominant 2025 Paradigm)

At its core: iteratively transform one program or model version into a slightly better one, based on a feedback signal. This is the central theme driving AGI progress in 2025.

### Taxonomy

#### Test-Time Training (Deep Learning)

Refine pretrained model weights on examples of the specific task at test time. Responsible for top Kaggle scores in both 2024 and 2025.
- NVARC: test-time training + synthetic data generation
- ARChitects: 2D-aware masked-diffusion with recursive self-refinement
- MindsAI: test-time fine-tuning + augmentation ensembles

#### Zero-Pretraining Deep Learning

Train from scratch per task. No pretrained weights. Two key examples:

**TRM (Tiny Recursive Model)**:
- 7M parameters, single network with separate answer (y) and latent (z) states
- Process: embed input x and initial answer y, then for N_sup=16 steps: (i) update z given (x, y, z) for n iterations, (ii) update y given (y, z)
- Recursively improves its prediction through the latent
- No pretraining, no external dataset, no branching search — gradient descent only
- 45% ARC-AGI-1, 8% ARC-AGI-2

**CompressARC**:
- 76K parameters, follows MDL (Minimum Description Length) principle
- VAE loss + decoder regularization substitutes for combinatorial search
- No pretraining, one model per task, ~20 min per puzzle on RTX 4070
- 20-34% ARC-AGI-1, 4% ARC-AGI-2
- Architecture: equivariant base + symmetry-breaking layers (short-range spatial, long-range spatial, directional)

#### Evolutionary Program Synthesis

Evolve ARC solution programs through an exploration-verification cycle:
- **Berman (Runner Up)**: evolutionary search harness for natural-language programs
- **Pang (Runner Up)**: similar but in Python, dynamically builds a program abstraction library
- **SOAR (2nd Place Paper)**: self-improving LLM that fine-tunes on its own search traces, +52% on ARC-AGI-1 without human-engineered DSLs

Both use two-phase refinement: (1) exploration generates candidates, (2) verification produces feedback signal. Repeats until a program satisfies all training I/O pairs.

#### Test-Time Chain-of-Thought

Commercial reasoning systems (o3, Gemini, Claude) use extended CoT as a natural-language refinement loop. Longer reasoning = more refinement cycles. Evidence of self-corrective behavior in reasoning traces.

#### Model Refinement Harnesses (Application Layer)

Domain-specific wrappers around commercial models:
- Poetiq harness on Gemini 3 Pro: 31% → 54% on ARC-AGI-2 ($31/task)
- Same on Claude Opus 4.5: comparable gains (~$60/task)
- Techniques like GEPA and DSPy enable general-purpose refinement given a verifier

---

## 5. Knowledge Overfitting & Contamination

A new class of overfitting specific to AI reasoning systems:

- AI reasoning performance is fundamentally coupled to **model knowledge coverage**
- Human reasoning is NOT similarly bound to knowledge — humans display extreme generalization
- This coupling leads to "jagged intelligence" — uneven performance across domains
- Even well-designed benchmarks with private test sets can be overfit if training and test distributions are similar and the model trains on substantial public domain data
- Evidence: Gemini 3 uses correct ARC color mappings in verification despite the harness never mentioning ARC formats — the model has internalized ARC structure from pretraining
- This is happening with both ARC-AGI-1 and ARC-AGI-2

### Implications

- LLM-based approaches get "free" performance from contamination
- Pure program synthesis approaches (like Aria) are immune — genuine generalization, but no free performance
- Benchmark design must continually adapt (motivates ARC-AGI-3's interactive format)
- Separating knowledge from reasoning remains an unsolved fundamental problem

---

## 6. What a Solution Looks Like

### Chollet's Original Vision (2019)

1. Build a **DSL** encoding Core Knowledge priors in sufficiently abstract, combinable form — "a human-like reasoning DSL"
2. **Generate candidate programs** that transform inputs to outputs
3. **Reuse and recombine subprograms** from prior solutions
4. **Select top candidates** by program simplicity or learned likelihood
5. Apply top candidates to test inputs

This is essentially a description of Aria's architecture.

### What Actually Works (2025)

The winning approaches have diverged from pure symbolic program synthesis:

**Test-time training** (top scores): neural networks that learn the transformation from I/O examples through gradient descent at inference time. Not symbolic programs — learned weights that implement the transformation.

**Evolutionary program synthesis** (strong papers): LLM-generated programs refined through evolution. Programs are in natural language or Python, not a hand-crafted DSL.

**Zero-pretraining neural methods** (paradigm-shifting): tiny networks trained from scratch per task prove that the program can be encoded in weights without any prior training.

**Hybrid neuro-symbolic** (honorable mentions): combining neural perception with symbolic reasoning.

### Critical Assessment of Current State

From the ARC Prize 2025 report:
- Grand Prize accuracy gap (to 85%): primarily bottlenecked by **engineering**
- Efficiency gap: bottlenecked by **fundamental science and new ideas**
- "We still need new ideas, such as methods to separate knowledge and reasoning"
- AGI test: "You'll know AGI is here when creating tasks easy for humans but hard for AI becomes simply impossible" — not achieved yet

### Automatable Task Domains (2025)

Tasks with BOTH properties are now reliably automatable:
1. Sufficient task knowledge coverage in pretraining corpus
2. The task provides a verifiable feedback signal

ARC tasks satisfy (2) but deliberately violate (1) — each task is novel by design.

---

## 7. Benchmark Design Principles

### Requirements for an Ideal Intelligence Benchmark (Chollet 2019)

1. Describe scope and establish **validity** (predictive of real-world performance)
2. Be **reliable** (reproducible results)
3. Measure **broad abilities** and **developer-aware generalization** (not just skill)
4. Feature tasks **unknown** to both system and developers
5. **Quantify generalization difficulty** (or at least characterize it)
6. **Control for experience** (no unlimited training data)
7. **Explicitly describe assumed priors**
8. **Work fairly** for both humans and machines (Core Knowledge priors only)

### ARC-AGI-2 Task Selection Protocol

1. Source: newly authored tasks + unused reserves from prior iterations
2. Human testing: advance only if >= 2 independent participants solve >= 1 sub-pair in first 2 attempts
3. Difficulty calibration: group tasks so mean human accuracy differs <= 1pp across partitions
4. Redundancy detection: flag tasks where one programmatic solution would solve both
5. Final validation: two-layer (external testers + internal review)

### The Adaptation Principle

> The critical concept is *adaptation*. Adaptation represents the core mode of intelligence. This process extends beyond creating effective benchmarks — it constitutes the ultimate measure of general intelligence itself.

Benchmarks must continually adapt in response to AI progress. ARC-AGI-1 → ARC-AGI-2 → ARC-AGI-3 is this principle in action.

---

## 8. Key Numbers

| Metric | Value |
|---|---|
| ARC-AGI-1 training tasks | 400 |
| ARC-AGI-1 eval tasks | 600 (400 public + 200 private) |
| ARC-AGI-2 eval tasks | 240 (120 semi-private + 120 private) |
| Grid size range | 1x1 to 30x30 |
| Color values | 10 (0-9) |
| Demo pairs per task | 2-5 (median 3) |
| Test pairs per task | 1-3 (68% have 1) |
| Attempts per test input | 2 |
| Human median solve time (v2) | 2.2 minutes |
| Human success rate (v2) | 75% per task, 66% per test pair |
| Top AI score on v2 private | 24.03% (NVARC, Nov 2025) |
| ARC Prize Grand Prize threshold | >= 85% on v2 private eval |
| Grand Prize purse | $700,000 |
| TRM parameters | 7M |
| CompressARC parameters | 76K |
